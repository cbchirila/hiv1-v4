
"""
Compound Overlap Heatmap
TOP 25% compounds by Tanimoto Similarity
Jaccard similarity of associated virus groups
"""

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

TOP_PERCENT = 0.25
TC_THRESHOLD = 0.7

SURECHEMBL_FILE = ROOT / "005 - surechembl search.csv.gz"
EDGES_FILE = ROOT / "010 - edges.csv.gz"

OUTPUT = ROOT / (Path(__file__).stem + ".png")
OUTPUT_SVG = ROOT / (Path(__file__).stem + ".svg")
OUTPUT_CSV = ROOT / "261 - top25_jaccard_matrix.csv"

EXPECTED_COMPOUNDS = {
    "Compound 118", "Compound 122",
    "Compound 24", "Compound 25",
    "Compound 27", "Compound 28",
    "Compound 29", "Compound 30",
    "Compound 31", "Compound 32",
    "Compound 33", "Compound 34",
    "Compound 35", "Compound 42",
    "Compound 43", "Compound 54",
    "Compound 59", "Compound 62",
    "Compound 63", "Compound 69",
    "Compound 73", "Compound 74",
    "Compound 75", "Compound 98"
}

# ============================================================
# 1. SELECT TOP 25% COMPOUNDS
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
    higher_is_better = True

elif "Tanimoto distance" in df.columns:
    score_col = "Tanimoto distance"
    higher_is_better = False

else:
    raise ValueError("Tanimoto score column not found.")

df["Ranking Score"] = pd.to_numeric(
    df[score_col],
    errors="coerce"
)

if not higher_is_better:
    df["Ranking Score"] = -df["Ranking Score"]

pairs = (
    df.dropna(subset=[
        "Compound ID",
        "SCHEMBL ID",
        "Ranking Score"
    ])
    .groupby(
        ["Compound ID", "SCHEMBL ID"],
        as_index=False
    )["Ranking Score"]
    .median()
)

compound_scores = (
    pairs.groupby("Compound ID")["Ranking Score"]
    .max()
    .sort_values(ascending=False)
)

n_top = max(
    1,
    int(np.ceil(len(compound_scores) * TOP_PERCENT))
)

top_compounds = compound_scores.head(n_top).index.tolist()

# ============================================================
# 2. VERIFY COMPOUND SELECTION
# ============================================================

actual = set(top_compounds)

missing = sorted(EXPECTED_COMPOUNDS - actual)
unexpected = sorted(actual - EXPECTED_COMPOUNDS)

print("\nTOP 25% COMPOUND SELECTION")
print("-" * 45)

print("Total scored compounds:", len(compound_scores))
print("Selected TOP 25%:", len(top_compounds))
print("Missing compounds:", missing)
print("Unexpected compounds:", unexpected)

if missing or unexpected:
    raise ValueError(
        "Compound selection differs from "
        "the other TOP 25% visualizations."
    )

# ============================================================
# 3. LOAD COMPOUND-VIRUS ASSOCIATIONS
# ============================================================

edges = pd.read_csv(
    EDGES_FILE,
    compression="gzip"
)

edges.columns = edges.columns.str.strip()

edges["Compound"] = (
    edges["Compound"]
    .astype("string")
    .str.strip()
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

# Normalize compound identifiers.
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

edges["Compound"] = (
    edges["Compound"]
    .map(normalize_compound)
    .astype("string")
)

# ============================================================
# 4. FILTER ASSOCIATIONS
# ============================================================

edges = edges[
    edges["Compound"].isin(top_compounds)
].copy()

edges = edges[
    edges["Tc max"] >= TC_THRESHOLD
].dropna(
    subset=["Compound", "Virus group", "Tc max"]
)

# Keep one record for every compound-virus group pair.
edges = (
    edges.groupby(
        ["Compound", "Virus group"],
        as_index=False
    )["Tc max"]
    .max()
)

# ============================================================
# 5. BUILD MEMBERSHIP MATRIX
# ============================================================

# Preserve all 24 compounds, including those without
# qualifying virus-group associations.
compounds = top_compounds

virus_groups = sorted(
    edges["Virus group"].unique().tolist()
)

if not virus_groups:
    raise ValueError(
        "No qualifying virus-group associations "
        "at the selected Tanimoto threshold."
    )

membership_df = (
    pd.crosstab(
        edges["Compound"],
        edges["Virus group"]
    )
    .reindex(
        index=compounds,
        columns=virus_groups,
        fill_value=0
    )
    .gt(0)
)

membership = membership_df.to_numpy(dtype=np.int64)

# ============================================================
# 6. CALCULATE JACCARD SIMILARITY
# ============================================================

# Number of shared virus groups.
intersection = membership @ membership.T

# Number of virus groups per compound.
sizes = membership.sum(axis=1)

# Number of virus groups in either compound.
union = (
    sizes[:, None]
    + sizes[None, :]
    - intersection
)

# Jaccard = intersection / union.
# Undefined comparisons (union = 0) remain NaN.
jaccard = np.divide(
    intersection.astype(float),
    union,
    out=np.full(
        intersection.shape,
        np.nan,
        dtype=float
    ),
    where=union > 0
)

jaccard_df = pd.DataFrame(
    jaccard,
    index=compounds,
    columns=compounds
)

# Save numerical values.
jaccard_df.to_csv(OUTPUT_CSV)

# ============================================================
# 7. VERIFICATION
# ============================================================

represented = set(edges["Compound"])

missing_links = sorted(
    set(top_compounds) - represented
)

print("\nJACCARD MATRIX STATISTICS")
print("-" * 45)

print("Selected compounds:", len(compounds))
print("Compounds with qualifying associations:",
      len(represented))
print("Compounds without qualifying associations:",
      missing_links)
print("Virus groups:", len(virus_groups))
print("Matrix dimensions:", jaccard_df.shape)

assert set(jaccard_df.index) == EXPECTED_COMPOUNDS
assert set(jaccard_df.columns) == EXPECTED_COMPOUNDS

# ============================================================
# 8. VISUALIZATION
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False
})

fig, ax = plt.subplots(figsize=(13, 11))

cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#eeeeee")

masked = np.ma.masked_invalid(jaccard)

im = ax.imshow(
    masked,
    cmap=cmap,
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="equal"
)

# All 24 compounds on both axes.
ax.set_xticks(
    np.arange(len(compounds))
)

ax.set_xticklabels(
    compounds,
    rotation=90,
    fontsize=8
)

ax.set_yticks(
    np.arange(len(compounds))
)

ax.set_yticklabels(
    compounds,
    fontsize=8
)

# Subtle grid lines.
ax.set_xticks(
    np.arange(-0.5, len(compounds), 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, len(compounds), 1),
    minor=True
)

ax.grid(
    which="minor",
    color="white",
    linewidth=0.35
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)

ax.set_xlabel("Compound", fontsize=11)
ax.set_ylabel("Compound", fontsize=11)

ax.set_title(
    "Compound Overlap Heatmap\n"
    "Top 25% Compounds | Shared Virus Groups (Jaccard)",
    fontsize=12,
    fontweight="bold",
    pad=15
)

fig.colorbar(
    im,
    ax=ax,
    label="Jaccard similarity of connected virus groups",
    shrink=0.80
)

fig.tight_layout()

# ============================================================
# 9. SAVE FIGURES
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
# 10. FINAL SUMMARY
# ============================================================

print("\nFINAL VERIFICATION")
print("-" * 45)

print("TOP 25% compounds:", len(top_compounds))
print("Heatmap compounds:", len(jaccard_df))
print("Matrix shape:", jaccard_df.shape)

print("\nSelected compounds:")
print(sorted(top_compounds))

print("\nSUCCESS: Same 24 compounds as the other plots.")

print("\nSaved:")
print(OUTPUT)
print(OUTPUT_SVG)
print(OUTPUT_CSV)
