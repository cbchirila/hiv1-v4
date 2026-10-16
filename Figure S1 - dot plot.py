from pathlib import Path
import re
import tempfile
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
INPUT = BASE / "005 - surechembl search.csv.gz"
OUTPUT = BASE / (Path(__file__).stem + ".png")
OUTPUT_SVG = BASE / (Path(__file__).stem + ".svg")

TC_THRESHOLD = 0.70
ONLY_COMPOUNDS_ABOVE_THRESHOLD = True   # True = only compounds with at least one pair Tc >= threshold
X_MIN = 0.5
DOT_SIZE = 45                           # bigger dots
CMAP = "plasma"                         # try: "viridis", "turbo", "magma", "coolwarm"
ORDER_BY = "number"                     # "number" = Compound 24, 25, ... | "median" = by median Tc


def compound_sort_key(name):
    m = re.search(r"\d+", str(name))
    return (int(m.group()) if m else 10**9, str(name))


def safe_save(fig, target, **kw):
    try:
        fig.savefig(target, **kw)
        print("Saved:", target)
    except OSError as e:
        print(f"Could not write {target}\n  -> {e}")
        fallback = Path(tempfile.gettempdir()) / target.name
        fig.savefig(fallback, **kw)
        print("Saved instead to:", fallback)
        try:
            shutil.copy(fallback, target)
        except OSError as e2:
            print("Copy to target failed too:", e2)


# 1. Load data and compute Tc.
df = pd.read_csv(INPUT, compression="gzip")
df.columns = df.columns.str.strip()
for col in ("Compound ID", "SCHEMBL ID", "Patent No"):
    df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

if "Tanimoto Similarity" in df:
    df["Tc"] = pd.to_numeric(df["Tanimoto Similarity"], errors="coerce")
elif "Tanimoto distance" in df:
    df["Tc"] = 1 - pd.to_numeric(df["Tanimoto distance"], errors="coerce")
else:
    raise ValueError("Tanimoto score column not found.")

pairs = (df.dropna(subset=["Compound ID", "SCHEMBL ID", "Tc"])
           .groupby(["Compound ID", "SCHEMBL ID"], as_index=False)["Tc"].median())

best_tc = pairs.groupby("Compound ID")["Tc"].max()
print("Total scored compounds:", len(best_tc))

if ONLY_COMPOUNDS_ABOVE_THRESHOLD:
    keep = best_tc[best_tc >= TC_THRESHOLD].index
    pairs = pairs[pairs["Compound ID"].isin(keep)].copy()
    best_tc = best_tc.loc[keep]

if pairs.empty:
    raise ValueError("No data to plot.")

# 2. Order (top to bottom).
if ORDER_BY == "number":
    order = sorted(best_tc.index, key=compound_sort_key)
else:
    order = pairs.groupby("Compound ID")["Tc"].median().sort_values().index.tolist()

row = {name: i for i, name in enumerate(order)}
pairs["y"] = pairs["Compound ID"].map(row)
print("Compounds plotted:", len(order))
print("Pairs plotted:", len(pairs))

# 3. Plot.
sns.set_theme(style="white", context="paper")
fig, ax = plt.subplots(figsize=(12, max(8, min(40, len(order) * 0.28))))

# Light row guides.
for i in range(len(order)):
    ax.axhline(i, color="#eeeeee", lw=0.6, zorder=0)

# Threshold line.
ax.axvline(TC_THRESHOLD, color="#444444", ls="--", lw=0.9, zorder=1,
           label=f"Tc = {TC_THRESHOLD:.2f}")

sc = ax.scatter(
    pairs["Tc"], pairs["y"],
    c=pairs["Tc"], cmap=CMAP, vmin=X_MIN, vmax=1.0,
    s=DOT_SIZE, alpha=0.85,
    edgecolors="white", linewidths=0.5, zorder=3
)

labels = [f"{n} ({best_tc[n]:.3f})" for n in order]
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()                      # first compound on top
ax.set_ylim(len(order) - 0.5, -0.5)

ax.set_xlim(X_MIN, 1.005)
ax.set_xticks(np.arange(X_MIN, 1.001, 0.05))
ax.tick_params(axis="x", labelsize=9)

ax.set_xlabel("Tanimoto Similarity", fontsize=11)
ax.set_ylabel("Compound ID (best Tc)", fontsize=11)
ax.set_title("Individual Compound-SCHEMBL similarity scores"
             + (f" | compounds with Tc >= {TC_THRESHOLD:.2f} (n = {len(order)})"
                if ONLY_COMPOUNDS_ABOVE_THRESHOLD else ""),
             fontsize=12, fontweight="bold")

for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
cb.set_label("Tanimoto Similarity", fontsize=10)

ax.legend(loc="lower left", frameon=False, fontsize=9)
fig.tight_layout()

safe_save(fig, OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
safe_save(fig, OUTPUT_SVG, bbox_inches="tight", facecolor="white")
plt.close(fig)