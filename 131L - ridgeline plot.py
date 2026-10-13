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
from scipy.stats import gaussian_kde, norm

BASE = Path(__file__).resolve().parent
INPUT = BASE / "005 - surechembl search.csv.gz"
OUTPUT = BASE / (Path(__file__).stem + ".png")
OUTPUT_SVG = BASE / (Path(__file__).stem + ".svg")

TC_THRESHOLD = 0.70
ONLY_ABOVE_THRESHOLD = True   # False = distribution of ALL pairs of the selected compounds (0-1)
                               # True  = distribution only of pairs with Tc >= threshold


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

# 2. Median Tc per (compound, SCHEMBL) pair.
pairs = (df.dropna(subset=["Compound ID", "SCHEMBL ID", "Tc"])
           .groupby(["Compound ID", "SCHEMBL ID"], as_index=False)["Tc"].median())
total_compounds = pairs["Compound ID"].nunique()

# 3. Select all compounds with at least one pair Tc >= threshold.
best_tc = pairs.groupby("Compound ID")["Tc"].max()
selected = best_tc[best_tc >= TC_THRESHOLD]
order = sorted(selected.index, key=compound_sort_key)   # ascending, top to bottom

if not order:
    raise ValueError(f"No compounds with Tc >= {TC_THRESHOLD}")

print("Total scored compounds:", total_compounds)
print(f"Compounds with Tc >= {TC_THRESHOLD}:", len(order))

# 4. Ridgeline.
# x-axis starts where the data starts (rounded down to 0.05), not at 0.
if ONLY_ABOVE_THRESHOLD:
    x_min = TC_THRESHOLD
else:
    all_scores = pairs.loc[pairs["Compound ID"].isin(order), "Tc"]
    x_min = max(0.0, np.floor((all_scores.min() - 0.02) * 20) / 20)

xx = np.linspace(x_min, 0.999, 500)

fig, ax = plt.subplots(figsize=(11, max(6, len(order) * 0.45)))

for y, name in enumerate(order[::-1]):
    scores = pairs.loc[pairs["Compound ID"] == name, "Tc"].to_numpy()
    if ONLY_ABOVE_THRESHOLD:
        scores = scores[scores >= TC_THRESHOLD]

    if len(scores) >= 3 and np.std(scores) > 1e-5:
        density = gaussian_kde(scores, bw_method=0.4)(xx)
    else:
        density = norm.pdf(xx, loc=scores.mean(), scale=0.025)

    density = density / density.max() if density.max() else density
    ax.fill_between(xx, y, y + density * 0.85, alpha=0.6)
    ax.plot(xx, y + density * 0.85, lw=0.8)

if not ONLY_ABOVE_THRESHOLD:
    ax.axvline(TC_THRESHOLD, color="red", ls="--", lw=0.9,
               label=f"Tc = {TC_THRESHOLD:.2f}")
    ax.legend(loc="lower left", frameon=False)

labels = [f"{n} ({best_tc[n]:.3f})" for n in order[::-1]]
ax.set(yticks=np.arange(len(order)) + 0.4, yticklabels=labels,
       xlim=(x_min, 1), xlabel="Tanimoto Similarity",
       title=f"Ridgeline distributions by compound | All compounds with Tc >= {TC_THRESHOLD:.2f} "
             f"(n = {len(order)})")
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", visible=False)
fig.tight_layout()

safe_save(fig, OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
safe_save(fig, OUTPUT_SVG, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("Selected compounds:")
print(order)