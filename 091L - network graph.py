from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# ============================================================
# SETTINGS
# ============================================================

BASE = Path(__file__).resolve().parent

TC_THRESHOLD = 0.70
MAX_SCHEMBL_PER_COMPOUND = 5     # None = all SCHEMBL structures with Tc >= threshold
MAX_PATENTS_PER_SCHEMBL = 10   # None = all patents

LABEL_SCHEMBL = None
LABEL_PATENTS = 1500

INPUT = BASE / "005 - surechembl search.csv.gz"
OUTPUT = BASE / (Path(__file__).stem + ".png")
OUTPUT_SVG = BASE / (Path(__file__).stem + ".svg")

def compound_sort_key(name):
    m = re.search(r"\d+", name)
    return (int(m.group()) if m else 10**9, name)

def take_top(df_sorted, group_col, n):
    """Keep the first n rows per group; n=None keeps everything.
    Always returns a DataFrame."""
    if n:
        return df_sorted.groupby(group_col, group_keys=False).head(n)
    return df_sorted

# ============================================================
# 1. LOAD AND CLEAN DATA
# ============================================================

df = pd.read_csv(INPUT, compression="gzip")
df.columns = df.columns.str.strip()

required = ["Compound ID", "SCHEMBL ID", "Patent No"]

for col in required:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

if "Tanimoto Similarity" in df.columns:
    score_col = "Tanimoto Similarity"
    df["Tc"] = pd.to_numeric(df[score_col], errors="coerce")
elif "Tanimoto distance" in df.columns:
    score_col = "Tanimoto distance"
    # similarity = 1 - distance
    df["Tc"] = 1 - pd.to_numeric(df[score_col], errors="coerce")
else:
    raise ValueError("Tanimoto score column not found.")

# ============================================================
# 2. SELECT ALL COMPOUNDS WITH Tc >= 0.70
# ============================================================

pairs = (
    df.dropna(subset=["Compound ID", "SCHEMBL ID", "Tc"])
    .groupby(["Compound ID", "SCHEMBL ID"], as_index=False)["Tc"]
    .median()
)

total_compounds = pairs["Compound ID"].nunique()

# Keep only pairs that pass the threshold.
pairs = pairs[pairs["Tc"] >= TC_THRESHOLD].copy()

compound_scores = (
    pairs.groupby("Compound ID")["Tc"]
    .max()
    .sort_values(ascending=False)
)

top_compounds = sorted(
    compound_scores.index.tolist(),
    key=compound_sort_key
)

print("\nCOMPOUND SELECTION")
print("-" * 45)
print("Total scored compounds:", total_compounds)
print(f"Compounds with Tc >= {TC_THRESHOLD}:", len(top_compounds))
print("Selected compounds:")
print(top_compounds)

if not top_compounds:
    raise ValueError(f"No compounds with Tc >= {TC_THRESHOLD}.")

# ============================================================
# 3. SELECT SCHEMBL PER COMPOUND (Tc >= 0.70)
# ============================================================

pairs_sorted = pairs.sort_values(
    ["Compound ID", "Tc", "SCHEMBL ID"],
    ascending=[True, False, True]
)

top_pairs = take_top(
    pairs_sorted,
    "Compound ID",
    MAX_SCHEMBL_PER_COMPOUND
).copy()

# ============================================================
# 4. SELECT PATENTS PER SCHEMBL
# ============================================================

patent_data = (
    df[["Compound ID", "SCHEMBL ID", "Patent No"]]
    .dropna()
    .drop_duplicates()
    .merge(
        top_pairs[["Compound ID", "SCHEMBL ID"]],
        on=["Compound ID", "SCHEMBL ID"],
        how="inner"
    )
)

patent_rank = (
    patent_data.groupby(["SCHEMBL ID", "Patent No"])["Compound ID"]
    .nunique()
    .reset_index(name="Frequency")
    .sort_values(
        ["SCHEMBL ID", "Frequency", "Patent No"],
        ascending=[True, False, True]
    )
)

selected_patents = take_top(
    patent_rank,
    "SCHEMBL ID",
    MAX_PATENTS_PER_SCHEMBL
).copy()

# ============================================================
# 5. BUILD NETWORK
# ============================================================

G = nx.Graph()

for compound in top_compounds:
    G.add_node("C:" + compound)

for compound, schembl in top_pairs[
    ["Compound ID", "SCHEMBL ID"]
].itertuples(index=False, name=None):
    G.add_edge("C:" + compound, "S:" + schembl)

for schembl, patent in selected_patents[
    ["SCHEMBL ID", "Patent No"]
].itertuples(index=False, name=None):
    G.add_edge("S:" + schembl, "P:" + patent)

# ============================================================
# 6. RADIAL LAYOUT (compounds inner ring, SCHEMBL middle, patents outer)
# ============================================================

compound_nodes = ["C:" + c for c in top_compounds]
schembl_nodes = sorted(n for n in G if n.startswith("S:"))
patent_nodes = sorted(n for n in G if n.startswith("P:"))

RADII = {"C": 1.5, "S": 2.5, "P": 3.5}

angle = {}

# Compounds: evenly spaced, in numeric order.
for i, node in enumerate(compound_nodes):
    angle[node] = 2 * np.pi * i / len(compound_nodes)

def circular_mean(values):
    if not values:
        return 0.0
    return float(np.arctan2(np.mean(np.sin(values)),
                            np.mean(np.cos(values))) % (2 * np.pi))

def place_ring(nodes):
    """Order nodes by the angle of their already-placed neighbors,
    then spread them evenly around the ring."""
    keyed = sorted(
        nodes,
        key=lambda n: (
            circular_mean([angle[v] for v in G.neighbors(n) if v in angle]),
            n
        )
    )
    for i, node in enumerate(keyed):
        angle[node] = 2 * np.pi * i / len(keyed)
    return keyed

schembl_nodes = place_ring(schembl_nodes)
patent_nodes = place_ring(patent_nodes)

groups = [compound_nodes, schembl_nodes, patent_nodes]

positions = {}
for node, a in angle.items():
    r = RADII[node[0]]
    positions[node] = (r * np.cos(a), r * np.sin(a))

# ============================================================
# 7. IMPORTANT NODE LABELS
# ============================================================

schembl_importance = sorted(
    schembl_nodes,
    key=lambda n: (-sum(v.startswith("C:") for v in G.neighbors(n)), n)
)

patent_importance = sorted(
    patent_nodes,
    key=lambda n: (-G.degree(n), n)
)

labeled_schembl = set(schembl_importance[:LABEL_SCHEMBL])
labeled_patents = set(patent_importance[:LABEL_PATENTS])

# ============================================================
# 8. VISUALIZATION
# ============================================================

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sns.set_theme(style="white", context="paper")

fig, ax = plt.subplots(figsize=(18, 18))
ax.set_aspect("equal")

colors = ["#2c7a91", "#e2a355", "#8e6dac"]

# Light guide rings.
theta = np.linspace(0, 2 * np.pi, 400)
for r in RADII.values():
    ax.plot(r * np.cos(theta), r * np.sin(theta),
            color="#dddddd", lw=0.6, zorder=0)

# Edges (fast: one LineCollection per edge type).
def segments(edge_list):
    return [[positions[u], positions[v]] for u, v in edge_list]

compound_edges = [
    (u, v) for u, v in G.edges()
    if u.startswith("C:") or v.startswith("C:")
]
patent_edges = [
    (u, v) for u, v in G.edges()
    if u.startswith("P:") or v.startswith("P:")
]

ax.add_collection(LineCollection(
    segments(compound_edges), colors="#2c7a91",
    linewidths=0.7, alpha=0.35, zorder=1))
ax.add_collection(LineCollection(
    segments(patent_edges), colors="#8e6dac",
    linewidths=0.4, alpha=0.20, zorder=1))

# Nodes.
node_size = {"C": 38, "S": 14, "P": 6}

for nodes, color in zip(groups, colors):
    xy = np.array([positions[n] for n in nodes])
    ax.scatter(
        xy[:, 0], xy[:, 1],
        s=[node_size[n[0]] + min(G.degree(n), 15) * 1.5 for n in nodes],
        c=color, linewidths=0, zorder=3
    )

# Radial label helper: text follows the ring direction, never upside down.
def radial_label(node, offset, fontsize, color, inward=False):
    a = angle[node]
    r = RADII[node[0]] + (-offset if inward else offset)
    x, y = r * np.cos(a), r * np.sin(a)

    on_right = np.cos(a) >= 0
    rot = np.degrees(a) if on_right else np.degrees(a) + 180

    if inward:
        ha = "right" if on_right else "left"
    else:
        ha = "left" if on_right else "right"

    ax.text(x, y, node[2:], rotation=rot, rotation_mode="anchor",
            ha=ha, va="center", fontsize=fontsize, color=color, zorder=4)

# All compound labels (inside the inner ring).
for node in compound_nodes:
    radial_label(node, 0.06, 7.0, "#234350", inward=True)

# Selected SCHEMBL labels (outside the middle ring).
for node in schembl_nodes:
    if node in labeled_schembl:
        radial_label(node, 0.06, 6.0, "#8a5b20")

# Selected patent labels (outside the outer ring).
for node in patent_nodes:
    if node in labeled_patents:
        radial_label(node, 0.08, 6.0, "#664b80")

# Legend.
ax.legend(
    handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[0],
               markersize=9, label=f"Compounds (Tc >= {TC_THRESHOLD:.2f}, n = {len(compound_nodes)}) - inner ring"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[1],
               markersize=7, label=f"SCHEMBL structures (n = {len(schembl_nodes)}) - middle ring"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[2],
               markersize=5, label=f"Patents (n = {len(patent_nodes)}) - outer ring"),
    ],
    loc="upper left", frameon=False, fontsize=11
)

ax.set_title(
    "Compound-SCHEMBL-Patent Network\n"
    f"All Compounds with Tanimoto Similarity >= {TC_THRESHOLD:.2f}",
    fontsize=14, fontweight="bold"
)

lim = RADII["P"] + 1.0
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.axis("off")

fig.tight_layout()

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor="white")

plt.close(fig)

# ============================================================
# 9. SAVE DATA
# ============================================================

top_pairs.to_csv(
    BASE / "tc07_compound_schembl_pairs.csv", index=False
)

selected_patents.to_csv(
    BASE / "tc07_schembl_patents.csv", index=False
)

pd.DataFrame({
    "Compound ID": top_compounds,
    "Best Tc": compound_scores.loc[top_compounds].values
}).to_csv(
    BASE / "tc07_selected_compounds.csv", index=False
)

# ============================================================
# 10. FINAL VERIFICATION
# ============================================================

graph_compounds = {
    n[2:] for n in G.nodes() if n.startswith("C:")
}

print("\nNETWORK STATISTICS")
print("-" * 45)

print(f"Compounds with Tc >= {TC_THRESHOLD}:", len(top_compounds))
print("Compounds in network:", len(graph_compounds))
print("SCHEMBL structures:", len(schembl_nodes))
print("Patents:", len(patent_nodes))
print("Total edges:", G.number_of_edges())

assert graph_compounds == set(top_compounds)
assert all(G.degree("C:" + c) > 0 for c in top_compounds)  # no isolated compounds

print("\nSUCCESS: all compounds with Tc >= 0.70 are in the network.")

print("\nSaved:")
print(OUTPUT)
print(OUTPUT_SVG)