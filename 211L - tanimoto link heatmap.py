"""Tanimoto heatmap: ONLY compounds with at least one Tc >= 0.70 link."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

TC_THRESHOLD = 0.7
ORDER_BY = "number"      # "number" = Compound 1, 2, 3...
                         # "tc"     = sorted by number of links, then best Tc

EDGES_FILE = ROOT / "010 - edges.csv.gz"
GROUPS_FILE = ROOT / "020 - virus group nodes.csv.gz"

OUTPUT = ROOT / (Path(__file__).stem + ".png")
OUTPUT_SVG = ROOT / (Path(__file__).stem + ".svg")
OUTPUT_MATRIX = ROOT / "211 - tc07_tanimoto_heatmap_matrix.csv"

# ============================================================
# HELPERS
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
# 1. LOAD COMPOUND-VIRUS ASSOCIATIONS
# ============================================================

edges = pd.read_csv(EDGES_FILE, compression="gzip")
groups = pd.read_csv(GROUPS_FILE, compression="gzip")

edges.columns = edges.columns.str.strip()
groups.columns = groups.columns.str.strip()

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

groups["Virus group"] = (
    groups["Virus group"]
    .astype("string")
    .str.strip()
)

edges["Tc max"] = pd.to_numeric(
    edges["Tc max"],
    errors="coerce"
)

# ============================================================
# 2. KEEP ONLY Tc >= 0.7
# ============================================================

edges = edges[
    edges["Tc max"] >= TC_THRESHOLD
].dropna(
    subset=["Compound", "Virus group", "Tc max"]
)

edges = (
    edges.groupby(
        ["Compound", "Virus group"],
        as_index=False
    )
    .agg({"Tc max": "max"})
)

# Compounds = ONLY those with at least one qualifying link.
compounds = sorted(
    edges["Compound"].unique(),
    key=compound_sort_key
)

print("\nHEATMAP COVERAGE")
print("-" * 45)
print(f"Compounds with Tc >= {TC_THRESHOLD}:", len(compounds))
print("Qualifying compound-virus links:", len(edges))

# ============================================================
# 3. BUILD HEATMAP MATRIX
# ============================================================

available_groups = set(edges["Virus group"])

virus_groups = [
    group
    for group in groups["Virus group"].dropna().unique()
    if group in available_groups
]

virus_groups += sorted(
    available_groups - set(virus_groups)
)

if not virus_groups:
    raise ValueError(
        f"No virus-group links with Tc >= {TC_THRESHOLD}."
    )

matrix = (
    edges.pivot(
        index="Compound",
        columns="Virus group",
        values="Tc max"
    )
    .reindex(
        index=compounds,
        columns=virus_groups
    )
)

if ORDER_BY == "tc":
    order = (
        matrix.assign(
            _n=matrix.notna().sum(axis=1),
            _best=matrix.max(axis=1),
        )
        .sort_values(["_n", "_best"], ascending=False)
        .index
    )
    matrix = matrix.loc[order]
    compounds = list(order)

matrix.to_csv(OUTPUT_MATRIX)

# ============================================================
# 4. HEATMAP VISUALIZATION
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False
})

fig_width = max(12, len(virus_groups) * 0.95)
fig_height = max(9, len(compounds) * 0.30)

fig, ax = plt.subplots(
    figsize=(fig_width, fig_height)
)

masked = np.ma.masked_invalid(
    matrix.to_numpy(dtype=float)
)

cmap = plt.get_cmap("YlOrRd").copy()
cmap.set_bad("#eeeeee")   # gray = this compound has no link to THIS virus group

im = ax.imshow(
    masked,
    cmap=cmap,
    vmin=TC_THRESHOLD,
    vmax=1.0,
    aspect="auto",
    interpolation="nearest"
)

ax.set_xticks(np.arange(len(virus_groups)))
ax.set_xticklabels(
    virus_groups,
    rotation=45,
    ha="right",
    fontsize=9
)

ax.set_yticks(np.arange(len(compounds)))
ax.set_yticklabels(
    compounds,
    fontsize=8
)

# Subtle separators.
ax.set_xticks(
    np.arange(-0.5, len(virus_groups), 1),
    minor=True
)
ax.set_yticks(
    np.arange(-0.5, len(compounds), 1),
    minor=True
)

ax.grid(
    which="minor",
    color="white",
    linewidth=0.6
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)

ax.set_xlabel("Virus group", fontsize=11)
ax.set_ylabel(f"Compounds (n = {len(compounds)})", fontsize=11)

ax.set_title(
    "Tanimoto Similarity Heatmap\n"
    f"Compounds with Tc >= {TC_THRESHOLD} | Compound-Virus Group Associations",
    fontsize=12,
    fontweight="bold",
    pad=15
)

fig.colorbar(
    im,
    ax=ax,
    label="Maximum Tanimoto similarity",
    shrink=0.5
)

fig.tight_layout()

# ============================================================
# 5. SAVE FIGURES
# ============================================================

fig.savefig(
    OUTPUT,
    dpi=300,
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
# 6. FINAL SUMMARY
# ============================================================

print("\nFINAL HEATMAP STATISTICS")
print("-" * 45)

print("Heatmap rows (compounds):", len(matrix))
print("Virus groups:", len(virus_groups))
print("Available links:", int(matrix.notna().sum().sum()))

assert matrix.notna().any(axis=1).all()   # no empty rows

print("\nSaved:")
print(OUTPUT)
print(OUTPUT_SVG)
print(OUTPUT_MATRIX)