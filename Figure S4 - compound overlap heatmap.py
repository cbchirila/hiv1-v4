
"""
Compound Overlap Heatmap
All compounds with Tanimoto Similarity >= 0.7
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

TC_THRESHOLD = 0.7

SURECHEMBL_FILE = ROOT / "005 - surechembl search.csv.gz"
EDGES_FILE = ROOT / "010 - edges.csv.gz"

OUTPUT = ROOT / (Path(__file__).stem + ".png")
OUTPUT_SVG = ROOT / (Path(__file__).stem + ".svg")
OUTPUT_CSV = ROOT / "261 - all_tc07_jaccard_matrix.csv"


# ============================================================
# 1. LOAD SURECHEMBL DATA
# ============================================================

df = pd.read_csv(
    SURECHEMBL_FILE,
    compression="gzip"
)

df.columns = df.columns.str.strip()

required_columns = {"Compound ID", "SCHEMBL ID"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in SureChEMBL file: {missing_columns}"
    )

for col in ("Compound ID", "SCHEMBL ID"):
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


# ============================================================
# 2. IDENTIFY TANIMOTO SCORE COLUMN
# ============================================================

if "Tanimoto Similarity" in df.columns:
    score_col = "Tanimoto Similarity"
    higher_is_better = True

elif "Tanimoto distance" in df.columns:
    score_col = "Tanimoto distance"
    higher_is_better = False

else:
    raise ValueError(
        "Neither 'Tanimoto Similarity' nor "
        "'Tanimoto distance' was found."
    )

df["Ranking Score"] = pd.to_numeric(
    df[score_col],
    errors="coerce"
)

# Convert distance to similarity if the column
# represents the complement of Tanimoto similarity.
if not higher_is_better:
    df["Ranking Score"] = 1 - df["Ranking Score"]


# ============================================================
# 3. SELECT ALL COMPOUNDS WITH Tc >= 0.7
# ============================================================

pairs = (
    df.dropna(
        subset=[
            "Compound ID",
            "SCHEMBL ID",
            "Ranking Score"
        ]
    )
    .groupby(
        ["Compound ID", "SCHEMBL ID"],
        as_index=False
    )["Ranking Score"]
    .max()
)

# Keep compound-SCHEMBL pairs meeting the threshold.
qualifying_pairs = pairs[
    pairs["Ranking Score"] >= TC_THRESHOLD
].copy()

# Select every compound with at least one qualifying pair.
compounds = sorted(
    qualifying_pairs["Compound ID"]
    .unique()
    .tolist()
)

if not compounds:
    raise ValueError(
        f"No compounds have Tanimoto similarity >= "
        f"{TC_THRESHOLD}."
    )

print("\nCOMPOUND SELECTION")
print("-" * 55)
print("Tanimoto threshold:", TC_THRESHOLD)
print("Total compounds in input:",
      df["Compound ID"].nunique())
print("Qualifying compound-SCHEMBL pairs:",
      len(qualifying_pairs))
print("Compounds selected:", len(compounds))


# ============================================================
# 4. LOAD COMPOUND-VIRUS ASSOCIATIONS
# ============================================================

edges = pd.read_csv(
    EDGES_FILE,
    compression="gzip"
)

edges.columns = edges.columns.str.strip()

required_edge_columns = {
    "Compound",
    "Virus group",
    "Tc max"
}

missing_edge_columns = (
    required_edge_columns - set(edges.columns)
)

if missing_edge_columns:
    raise ValueError(
        f"Missing columns in edges file: "
        f"{missing_edge_columns}"
    )

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


# ============================================================
# 5. NORMALIZE COMPOUND IDENTIFIERS
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


edges["Compound"] = (
    edges["Compound"]
    .map(normalize_compound)
    .astype("string")
)


# ============================================================
# 6. FILTER COMPOUND-VIRUS ASSOCIATIONS
# ============================================================

# Retain associations for selected compounds only.
edges = edges[
    edges["Compound"].isin(compounds)
].copy()

# Retain only associations with Tc max >= 0.7.
edges = edges[
    edges["Tc max"] >= TC_THRESHOLD
].dropna(
    subset=["Compound", "Virus group", "Tc max"]
)

# Keep one record per compound-virus group pair.
edges = (
    edges.groupby(
        ["Compound", "Virus group"],
        as_index=False
    )["Tc max"]
    .max()
)


# ============================================================
# 7. BUILD MEMBERSHIP MATRIX
# ============================================================

# Preserve all selected compounds, including those without
# qualifying virus-group associations.
virus_groups = sorted(
    edges["Virus group"].unique().tolist()
)

if not virus_groups:
    raise ValueError(
        "No qualifying virus-group associations were found "
        f"at Tc max >= {TC_THRESHOLD} for the selected compounds."
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
# 8. CALCULATE JACCARD SIMILARITY
# ============================================================

# Number of shared virus groups for each compound pair.
intersection = membership @ membership.T

# Number of associated virus groups per compound.
sizes = membership.sum(axis=1)

# Number of virus groups in the union of each pair.
union = (
    sizes[:, None]
    + sizes[None, :]
    - intersection
)

# Jaccard similarity = intersection / union.
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

# Save the numerical Jaccard matrix.
jaccard_df.to_csv(OUTPUT_CSV)


# ============================================================
# 9. VERIFICATION
# ============================================================

represented = set(edges["Compound"].dropna())

missing_links = sorted(
    set(compounds) - represented
)

print("\nJACCARD MATRIX STATISTICS")
print("-" * 55)

print("Selected compounds:", len(compounds))
print("Compounds with qualifying virus associations:",
      len(represented))
print("Compounds without qualifying virus associations:",
      len(missing_links))

if missing_links:
    print("Compounds without qualifying associations:")
    print(missing_links)

print("Virus groups:", len(virus_groups))
print("Matrix dimensions:", jaccard_df.shape)

assert jaccard_df.shape == (
    len(compounds),
    len(compounds)
)

assert jaccard_df.index.tolist() == compounds
assert jaccard_df.columns.tolist() == compounds


# ============================================================
# 10. VISUALIZATION
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Adapt figure size to the number of compounds.
n_compounds = len(compounds)

fig_size = max(11, min(30, n_compounds * 0.38))

fig, ax = plt.subplots(
    figsize=(fig_size, fig_size)
)

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

# Compound labels on both axes.
ax.set_xticks(np.arange(n_compounds))
ax.set_xticklabels(
    compounds,
    rotation=90,
    fontsize=7 if n_compounds > 40 else 8
)

ax.set_yticks(np.arange(n_compounds))
ax.set_yticklabels(
    compounds,
    fontsize=7 if n_compounds > 40 else 8
)

# Grid lines between cells.
ax.set_xticks(
    np.arange(-0.5, n_compounds, 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, n_compounds, 1),
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
    "All Compounds with Tanimoto Similarity >= 0.7\n"
    "Shared Virus Groups (Jaccard Similarity)",
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
# 11. SAVE FIGURES
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
# 12. FINAL SUMMARY
# ============================================================

print("\nFINAL VERIFICATION")
print("-" * 55)

print("Tanimoto threshold:", TC_THRESHOLD)
print("Selected compounds:", len(compounds))
print("Heatmap compounds:", len(jaccard_df))
print("Matrix shape:", jaccard_df.shape)
print("Virus groups:", len(virus_groups))

print("\nSaved files:")
print(OUTPUT)
print(OUTPUT_SVG)
print(OUTPUT_CSV)

print("\nSUCCESS: All compounds with Tc >= 0.7 "
      "are included in the analysis.")
