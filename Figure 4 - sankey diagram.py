from pathlib import Path
from collections import defaultdict
import re
import tempfile
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

BASE = Path(__file__).resolve().parent
INPUT = BASE / '005 - surechembl search.csv.gz'
OUTPUT = BASE / (Path(__file__).stem + '.png')
OUTPUT_SVG = BASE / (Path(__file__).stem + '.svg')

TC_THRESHOLD = 0.70
TC_DECIMALS = 3
MAX_SCHEMBL_PER_COMPOUND = 3     # None = all SCHEMBL structures with Tc >= threshold
MAX_PATENTS_PER_PAIR = None      # e.g. 50 = cap patents per (compound, SCHEMBL); None = all
LABEL_SCHEMBL = 100               # label only the largest N structures (None = all)
MIN_NODE_FRAC = 0.012            # minimum node height (fraction of the column) so small ones stay visible
SQRT_SCALE = True                # True = height ~ sqrt(patents): reduces domination by big nodes


def compound_sort_key(name):
    m = re.search(r'\d+', str(name))
    return (int(m.group()) if m else 10**9, str(name))


def safe_save(fig, target, **kw):
    try:
        fig.savefig(target, **kw)
        print('Saved:', target)
    except OSError as e:
        print(f'Could not write {target}\n  -> {e}')
        fallback = Path(tempfile.gettempdir()) / target.name
        fig.savefig(fallback, **kw)
        print('Saved instead to:', fallback)
        try:
            shutil.copy(fallback, target)
        except OSError as e2:
            print('Copy to target failed too:', e2)


# 1. Read data and compute Tanimoto similarity (Tc).
df = pd.read_csv(INPUT, compression='gzip')
df.columns = df.columns.str.strip()
for col in ('Compound ID', 'SCHEMBL ID', 'Patent No'):
    df[col] = df[col].astype('string').str.strip().replace('', pd.NA)

if 'Tanimoto Similarity' in df:
    df['Tc'] = pd.to_numeric(df['Tanimoto Similarity'], errors='coerce')
elif 'Tanimoto distance' in df:
    df['Tc'] = 1 - pd.to_numeric(df['Tanimoto distance'], errors='coerce')
else:
    raise ValueError('Missing Tanimoto Similarity / Tanimoto distance column')

# 2. Median Tc per (compound, SCHEMBL) pair, keep only Tc >= threshold.
pairs = (df.dropna(subset=['Compound ID', 'SCHEMBL ID', 'Tc'])
         .groupby(['Compound ID', 'SCHEMBL ID'], as_index=False)['Tc'].median())
total_compounds = pairs['Compound ID'].nunique()
pairs = pairs[pairs['Tc'] >= TC_THRESHOLD].copy()
if pairs.empty:
    raise ValueError(f'No compound-SCHEMBL pairs with Tc >= {TC_THRESHOLD}')

# 3. Best SCHEMBL structures per compound.
pairs_sorted = pairs.sort_values(['Compound ID', 'Tc', 'SCHEMBL ID'],
                                 ascending=[True, False, True])
top_pairs = (pairs_sorted.groupby('Compound ID', group_keys=False)
             .head(MAX_SCHEMBL_PER_COMPOUND)
             if MAX_SCHEMBL_PER_COMPOUND else pairs_sorted)

trips = (df[['Compound ID', 'SCHEMBL ID', 'Patent No']]
         .dropna().drop_duplicates()
         .merge(top_pairs[['Compound ID', 'SCHEMBL ID']],
                on=['Compound ID', 'SCHEMBL ID'], how='inner'))

if MAX_PATENTS_PER_PAIR:
    trips = (trips.sort_values(['Compound ID', 'SCHEMBL ID', 'Patent No'])
             .groupby(['Compound ID', 'SCHEMBL ID'], group_keys=False)
             .head(MAX_PATENTS_PER_PAIR))

selected = sorted(top_pairs['Compound ID'].unique(), key=compound_sort_key)
missing = sorted(set(selected) - set(trips['Compound ID']), key=compound_sort_key)
if missing:
    print('WARNING: compounds with Tc >= threshold but no patent association '
          '(not shown):', missing)

trips['Patent jurisdiction'] = trips['Patent No'].str.split('-').str[0]

best_tc = top_pairs.groupby('Compound ID')['Tc'].max().to_dict()
patents_per_jur = trips.groupby('Patent jurisdiction')['Patent No'].nunique().to_dict()
patents_per_comp = trips.groupby('Compound ID')['Patent No'].nunique().to_dict()
patents_per_struct = trips.groupby('SCHEMBL ID')['Patent No'].nunique().to_dict()


def node_label(stage, name):
    if stage == 0:
        return f"{name} ({patents_per_comp[name]}) | Tc {best_tc[name]:.{TC_DECIMALS}f}"
    if stage == 1:
        return f"{name} ({patents_per_struct[name]})"
    return f"{name} ({patents_per_jur[name]})"


# 4. Flows (each row = one unique compound-SCHEMBL-patent association).
columns = ['Compound ID', 'SCHEMBL ID', 'Patent jurisdiction']
flows = [trips.groupby(columns[i:i + 2]).size().to_dict() for i in (0, 1)]
counts = [trips[col].value_counts() for col in columns]

# 5. Orders (barycenter).
compound_order = sorted(counts[0].index, key=compound_sort_key)
compound_index = {c: i for i, c in enumerate(compound_order)}
structure_order = sorted(counts[1].index, key=lambda s: (
    np.average([compound_index[c] for (c, ss), n in flows[0].items() if ss == s],
               weights=[n for (c, ss), n in flows[0].items() if ss == s]), str(s)))
structure_index = {s: i for i, s in enumerate(structure_order)}
jurisdiction_order = sorted(counts[2].index, key=lambda p: (
    np.average([structure_index[s] for (s, pp), n in flows[1].items() if pp == p],
               weights=[n for (s, pp), n in flows[1].items() if pp == p]), str(p)))

# Smallest compound number on top.
orders = [list(reversed(compound_order)),
          list(reversed(structure_order)),
          list(reversed(jurisdiction_order))]

# 6. Node heights: a transformed size with a minimum height.
def transform(n):
    return np.sqrt(n) if SQRT_SCALE else float(n)


gap = 0.006
positions = []
node_scale = []   # per column: height per unit of raw flow, per node
for col_counts, order in zip(counts, orders):
    raw = {name: transform(int(col_counts[name])) for name in order}
    usable = 1.0 - gap * (len(order) - 1)
    total = sum(raw.values())
    heights = {name: max(raw[name] / total * usable, MIN_NODE_FRAC) for name in order}
    # Renormalize so the column keeps height <= 1.
    s = sum(heights.values())
    if s > usable:
        heights = {k: v * usable / s for k, v in heights.items()}
    cursor = (1 - (sum(heights.values()) + gap * (len(order) - 1))) / 2
    pos = {}
    for name in order:
        pos[name] = (cursor, heights[name])
        cursor += heights[name] + gap
    positions.append(pos)

palette = plt.get_cmap('tab20')
compound_colors = {c: palette(i % 20) for i, c in enumerate(compound_order)}
colors = [compound_colors,
          {s: '#e3a14c' for s in structure_order},
          {p: '#8065a5' for p in jurisdiction_order}]

# 7. Figure.
n_comp, n_struct = len(compound_order), len(structure_order)
fig_height = max(12, n_comp * 0.32, n_struct * 0.16)
comp_fs = float(np.clip(fig_height / n_comp * 22, 6.0, 9.5))

fig, ax = plt.subplots(figsize=(17, fig_height))

# Ribbons: each node distributes its height among its flows proportionally.
out_cursor = [defaultdict(float) for _ in range(3)]
in_cursor = [defaultdict(float) for _ in range(3)]

node_out_total = [defaultdict(float) for _ in range(2)]
node_in_total = [defaultdict(float) for _ in range(3)]
for stage, mapping in enumerate(flows):
    for (s, t), n in mapping.items():
        node_out_total[stage][s] += n
        node_in_total[stage + 1][t] += n

for stage, mapping in enumerate(flows):
    for (source, target), n in sorted(
            mapping.items(),
            key=lambda kv: (orders[stage].index(kv[0][0]),
                            orders[stage + 1].index(kv[0][1]))):
        sh = positions[stage][source][1] * n / node_out_total[stage][source]
        th = positions[stage + 1][target][1] * n / node_in_total[stage + 1][target]
        sy = positions[stage][source][0] + out_cursor[stage][source]
        ty = positions[stage + 1][target][0] + in_cursor[stage + 1][target]
        out_cursor[stage][source] += sh
        in_cursor[stage + 1][target] += th
        x0, x1 = stage + 0.055, stage + 0.945
        dx = x1 - x0
        verts = [(x0, sy), (x0 + dx * .46, sy), (x1 - dx * .46, ty), (x1, ty),
                 (x1, ty + th), (x1 - dx * .46, ty + th), (x0 + dx * .46, sy + sh),
                 (x0, sy + sh), (x0, sy)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts, codes),
                               facecolor=colors[0][source] if stage == 0 else colors[1][source],
                               edgecolor='none', alpha=0.30 if stage == 0 else 0.22))

# Which SCHEMBL get labels (largest by patents).
if LABEL_SCHEMBL:
    labeled_struct = set(counts[1].sort_values(ascending=False).head(LABEL_SCHEMBL).index)
else:
    labeled_struct = set(structure_order)

for stage, order in enumerate(orders):
    for name in order:
        y, height = positions[stage][name]
        ax.add_patch(Rectangle((stage - 0.055, y), 0.11, height,
                               facecolor=colors[stage][name],
                               edgecolor='white', linewidth=0.3))
        if stage == 1 and name not in labeled_struct:
            continue
        fs = comp_fs if stage == 0 else (6.5 if stage == 1 else 9)
        ax.text(stage + 0.08, y + height / 2, node_label(stage, name),
                ha='left', va='center', fontsize=fs, color='#333333')

for x, title in enumerate((f'COMPOUNDS (Tc >= {TC_THRESHOLD:.2f}, n = {n_comp})',
                           f'SCHEMBL STRUCTURES (n = {n_struct})',
                           'PATENT JURISDICTION')):
    ax.text(x, 1.045, title, ha='center', va='center', fontsize=11, fontweight='bold')

note = ('Node height ~ sqrt(unique associations)' if SQRT_SCALE
        else 'Node height = number of unique associations')
ax.text(1, -0.045, note + '; labels show best Tc and unique patents',
        ha='center', fontsize=9)
ax.set(xlim=(-0.14, 2.58), ylim=(-0.07, 1.09),
       title=f'Compound -> SCHEMBL -> Patent jurisdiction | All compounds with Tc >= {TC_THRESHOLD:.2f}')
ax.axis('off')
fig.tight_layout()

safe_save(fig, OUTPUT, dpi=200, bbox_inches='tight', facecolor='white')
safe_save(fig, OUTPUT_SVG, bbox_inches='tight', facecolor='white')
plt.close(fig)

# 8. Save the underlying data (every patent remains findable).
trips.to_csv(BASE / 'Figure 4 - sankey_all_associations.csv', index=False)

print('Total scored compounds:', total_compounds)
print(f'Compounds with Tc >= {TC_THRESHOLD}:', len(selected))
print('SCHEMBL structures:', n_struct)
print('Patents per jurisdiction:', patents_per_jur)