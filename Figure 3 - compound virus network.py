"""
201 - Compound Virus Network
ALL compounds, Tanimoto Similarity (Tc >= 0.70)

Node size (virus groups) = number of patents
Edge width and opacity = maximum Tanimoto similarity
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# Keep text as editable text in the SVG (Illustrator / Inkscape / Affinity)
plt.rcParams["svg.fonttype"] = "none"

# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

MIN_TC = 0.70
SHOW_ISOLATED = False   # False = show only compounds with at least one Tc >= MIN_TC link

COMPOUND_NODE_SIZE = 1150   # size of compound circles
COMPOUND_LABEL_SIZE = 14    # font size of the number inside the circle (was 8)

SURECHEMBL_FILE = ROOT / "005 - surechembl search.csv.gz"
EDGES_FILE = ROOT / "010 - edges.csv.gz"
GROUPS_FILE = ROOT / "020 - virus group nodes.csv.gz"

OUTPUT = ROOT / (Path(__file__).stem + ".png")
OUTPUT_SVG = ROOT / (Path(__file__).stem + ".svg")
OUTPUT_EDGES = ROOT / "201 - all_compound_virus_edges.csv"

# ============================================================
# 1. VIRUS GROUP COLORS AND POSITIONS
# ============================================================

colors = {
    "HIV / retrovirus": "#f46527",
    "Generic antiviral (unspecified virus)": "#bebebe",
    "Coronavirus": "#8078b5",
    "HBV / other hepatitis": "#82bce0",
    "Herpesviruses": "#d76476",
    "Viral vector / gene therapy": "#888888",
    "HPV / pox / other DNA": "#29c4d6",
    "HCV": "#388cc6",
    "Influenza": "#39a867",
    "RSV / respiratory": "#b59b65",
    "Flavi / alphaviruses": "#f4c35c",
    "Filovirus / other hemorrhagic": "#a4605a",
}

anchors = {
    "HIV / retrovirus": (1.15, -0.07),
    "Generic antiviral (unspecified virus)": (0.97, -0.58),
    "Coronavirus": (-0.82, 0.70),
    "HBV / other hepatitis": (0.57, 0.69),
    "Herpesviruses": (-1.27, 0.30),
    "Viral vector / gene therapy": (0.56, -0.91),
    "HPV / pox / other DNA": (-0.16, -0.99),
    "HCV": (1.02, 0.30),
    "Influenza": (-0.16, 0.78),
    "RSV / respiratory": (-1.43, -0.07),
    "Flavi / alphaviruses": (-1.26, -0.58),
    "Filovirus / other hemorrhagic": (-0.79, -0.90),
}

# ============================================================
# 2. NORMALIZE COMPOUND IDENTIFIERS
# ============================================================

def normalize_compound(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    match = re.fullmatch(
        r"(?:Compound[\s_-]*|C)(\d+)",
        value,
        flags=re.IGNORECASE
    )

    if match:
        return f"Compound {int(match.group(1))}"

    return value


def compound_sort_key(name):
    m = re.search(r"\d+", name)
    return (int(m.group()) if m else 10**9, name)

# ============================================================
# 3. LIST ALL SCORED COMPOUNDS
# ============================================================

df = pd.read_csv(
    SURECHEMBL_FILE,
    compression="gzip"
)

df.columns = df.columns.str.strip()

for col in ("Compound ID", "SCHEMBL ID"):
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

if "Tanimoto Similarity" in df.columns:
    score_col = "Tanimoto Similarity"
elif "Tanimoto distance" in df.columns:
    score_col = "Tanimoto distance"
else:
    raise ValueError("Tanimoto score column not found.")

df["Ranking Score"] = pd.to_numeric(
    df[score_col],
    errors="coerce"
)

scored = df.dropna(
    subset=["Compound ID", "SCHEMBL ID", "Ranking Score"]
)

all_compounds = sorted(
    {normalize_compound(c) for c in scored["Compound ID"].unique()},
    key=compound_sort_key
)

print("\nCOMPOUND SELECTION")
print("-" * 45)
print("Total compounds (ALL):", len(all_compounds))

# ============================================================
# 4. LOAD COMPOUND-VIRUS GROUP ASSOCIATIONS
# ============================================================

edges = pd.read_csv(
    EDGES_FILE,
    compression="gzip"
)

groups = pd.read_csv(
    GROUPS_FILE,
    compression="gzip"
)

edges.columns = edges.columns.str.strip()
groups.columns = groups.columns.str.strip()

needed = {
    "Compound",
    "Virus group",
    "Tc max"
}

if not needed.issubset(edges.columns):
    raise ValueError(
        f"Missing edge columns: {needed - set(edges.columns)}"
    )

edges["Compound"] = (
    edges["Compound"]
    .map(normalize_compound)
    .astype("string")
)

edges["Virus group"] = (
    edges["Virus group"]
    .astype("string")
    .str.strip()
)

edges["Tc max"] = pd.to_numeric(
    edges["Tc max"],
    errors="coerce"
)

groups["Virus group"] = (
    groups["Virus group"]
    .astype("string")
    .str.strip()
)

# ============================================================
# 5. FILTER Tc >= 0.70 (NO TOP 25% FILTER)
# ============================================================

edges = edges[
    edges["Tc max"] >= MIN_TC
].dropna(
    subset=["Compound", "Virus group", "Tc max"]
)

# Maximum Tanimoto similarity for each association.
edges = (
    edges.groupby(
        ["Compound", "Virus group"],
        as_index=False
    )["Tc max"]
    .max()
)

# Include compounds that appear only in the edges file.
all_compounds = sorted(
    set(all_compounds) | set(edges["Compound"]),
    key=compound_sort_key
)

represented = set(edges["Compound"])

missing_links = sorted(
    set(all_compounds) - represented,
    key=compound_sort_key
)

if SHOW_ISOLATED:
    compound_nodes = all_compounds
else:
    compound_nodes = [c for c in all_compounds if c in represented]

print("\nNETWORK COVERAGE")
print("-" * 45)
print("All compounds:", len(all_compounds))
print("Compounds with qualifying links:", len(represented))
print("Compounds without qualifying links:", len(missing_links))
print("Compounds displayed:", len(compound_nodes))

# ============================================================
# 6. PATENT COUNTS FOR VIRUS GROUPS
# ============================================================

patent_col = next(
    (
        c for c in groups.columns
        if c.startswith("Patents")
    ),
    None
)

if patent_col is None:
    raise ValueError(
        "Virus group nodes table has no Patents column."
    )

groups[patent_col] = pd.to_numeric(
    groups[patent_col],
    errors="coerce"
).fillna(0)

group_patents = (
    groups.groupby("Virus group")[patent_col]
    .max()
    .to_dict()
)

# ============================================================
# 7. BUILD NETWORK
# ============================================================

G = nx.Graph()

for compound in compound_nodes:
    G.add_node(
        compound,
        node_type="compound"
    )

for compound, group, tc in edges[
    ["Compound", "Virus group", "Tc max"]
].itertuples(index=False, name=None):

    G.add_node(
        group,
        node_type="virus_group"
    )

    G.add_edge(
        compound,
        group,
        tc=float(tc)
    )

group_nodes = sorted(
    edges["Virus group"].unique().tolist()
)

unknown_groups = set(group_nodes) - set(anchors)

if unknown_groups:
    raise ValueError(
        f"Unexpected virus groups: {sorted(unknown_groups)}"
    )

# ============================================================
# 8. NETWORK LAYOUT
# ============================================================

rng = np.random.default_rng(23)

starting = {
    g: np.array(anchors[g], dtype=float)
    for g in group_nodes
}

# Position compounds near their associated virus groups.
for compound in compound_nodes:

    adjacent = list(G.neighbors(compound))

    if adjacent:
        center = np.mean(
            [starting[g] for g in adjacent],
            axis=0
        )

        starting[compound] = (
            center * (0.50 if len(adjacent) > 2 else 1.15)
            + rng.normal(0, 0.09, 2)
        )

    else:
        starting[compound] = np.array(
            [-1.65, 1.15]
        ) + rng.normal(0, 0.05, 2)

# Spring layout with fixed virus group anchors.
pos = nx.spring_layout(
    G,
    pos=starting,
    fixed=group_nodes if group_nodes else None,
    seed=23,
    iterations=350,
    k=0.36,
    weight=None
)

# Move highly connected compounds toward the center.
for compound in compound_nodes:

    degree = G.degree(compound)

    if degree >= 4:
        pos[compound] = (
            0.75 * pos[compound]
            + 0.25 * np.array([0.0, -0.05])
        )

# Isolated compounds (only present if SHOW_ISOLATED = True):
# arranged in a grid in the top-left corner.
isolated = [
    c for c in compound_nodes
    if G.degree(c) == 0
]

N_COLS = 8

for i, compound in enumerate(isolated):
    pos[compound] = np.array([
        -1.65 + 0.14 * (i % N_COLS),
        1.20 + 0.14 * (i // N_COLS)
    ])

# ============================================================
# 9. VISUALIZATION
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False
})

fig, ax = plt.subplots(
    figsize=(20, 15)
)

fig.patch.set_facecolor("white")

# Draw edges according to Tanimoto similarity.
for compound, group, tc in edges.sort_values("Tc max")[
    ["Compound", "Virus group", "Tc max"]
].itertuples(index=False, name=None):

    if compound not in G or group not in G:
        continue

    strength = np.clip(
        (float(tc) - MIN_TC) / (1 - MIN_TC),
        0,
        1
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(compound, group)],
        ax=ax,
        width=1.3 + 4.5 * strength,
        edge_color=[colors[group]],
        alpha=0.50 + 0.45 * strength
    )

# Compound nodes.
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=compound_nodes,
    ax=ax,
    node_shape="o",
    node_color="#24314e",
    node_size=COMPOUND_NODE_SIZE,
    edgecolors="white",
    linewidths=0.8
)

# Compact labels inside compound nodes.
for compound in compound_nodes:
    x, y = pos[compound]

    m = re.search(r"\d+", compound)
    label = f"C{m.group()}" if m else compound

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=COMPOUND_LABEL_SIZE if len(label) <= 3 else COMPOUND_LABEL_SIZE - 2,
        fontweight="bold",
        color="white",
        zorder=10
    )

# Virus group node sizes represent patent counts.
patent_counts = np.array(
    [
        float(group_patents.get(g, 0))
        for g in group_nodes
    ],
    dtype=float
)

if len(patent_counts):
    max_patents = max(float(patent_counts.max()), 1.0)

    sizes = (
        650
        + 1900 * np.sqrt(patent_counts / max_patents)
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=group_nodes,
        ax=ax,
        node_shape="D",
        node_color=[
            colors[g]
            for g in group_nodes
        ],
        node_size=sizes,
        edgecolors="#202020",
        linewidths=1.2,
        alpha=0.97
    )

# Virus group labels.
for group in group_nodes:
    x, y = pos[group]

    direction = -1 if y < -0.48 else 1

    ax.annotate(
        f"{group}\n({int(group_patents.get(group, 0)):,} patents)",
        (x, y),
        xytext=(0, 23 * direction),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        zorder=11
    )

# ax.set_title(
#    "Compound-Virus Group Network\n"
#    f"All Compounds by Tanimoto Similarity (Tc >= {MIN_TC:.2f})",
#    loc="center",
#    fontsize=14,
#    fontweight="bold",
#    pad=25
#)

# Automatic limits avoid clipping nodes.
xy = np.array(list(pos.values()), dtype=float)

ax.set_xlim(
    xy[:, 0].min() - 0.35,
    xy[:, 0].max() + 0.35
)

ax.set_ylim(
    xy[:, 1].min() - 0.35,
    xy[:, 1].max() + 0.35
)

ax.axis("off")

fig.tight_layout(pad=1.5)

# ============================================================
# 10. SAVE FIGURES (PNG + SVG)
# ============================================================

fig.savefig(
    OUTPUT,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUTPUT_SVG,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

# ============================================================
# 11. SAVE NETWORK DATA
# ============================================================

edges.to_csv(OUTPUT_EDGES, index=False)

# ============================================================
# 12. FINAL STATISTICS
# ============================================================

graph_compounds = {
    n for n in G.nodes()
    if G.nodes[n].get("node_type") == "compound"
}

assert graph_compounds == set(compound_nodes)

print("\nFINAL NETWORK STATISTICS")
print("-" * 45)

print("All compounds:", len(all_compounds))
print("Compounds displayed:", len(graph_compounds))
print("Compounds not displayed (no link Tc >= MIN_TC):",
      len(all_compounds) - len(graph_compounds))
print("Virus groups:", len(group_nodes))
print("Total qualifying edges:", G.number_of_edges())

print("\nSaved:")
print(OUTPUT)
print(OUTPUT_SVG)
print(OUTPUT_EDGES)