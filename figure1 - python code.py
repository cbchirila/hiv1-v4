"""
Bibliometric charts for a literature review
============================================
Generates 4 publication-ready figures:
  1. Publications per year (area + line chart, with annotations)
  2. Top document types (horizontal bar, gradient-colored)
  3. Top 15 Web of Science categories (horizontal bar with value labels)
  4. Combined dashboard (2x2) - "wow" version for a slide/poster

Run with: python bibliometric_charts_en.py
Output: PNG files in the current folder (300 dpi, publication-ready)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ----------------------------------------------------------------------
# 1. YOUR DATA (extracted from the tables)
# ----------------------------------------------------------------------

document_types = {
    "ARTICLE": 361133,
    "MEETING ABSTRACT": 64622,
    "REVIEW": 40813,
    "EDITORIAL MATERIAL": 20012,
    "PROCEEDINGS PAPER": 18426,
    "LETTER": 18011,
    "BOOK CHAPTER": 4437,
    "NEWS ITEM": 3398,
    "NOTE": 3230,
    "CORRECTION": 3077,
}

publication_years = {
    2027: 4, 2026: 10523, 2025: 19599, 2024: 17712, 2023: 18461, 2022: 19676,
    2021: 21588, 2020: 21190, 2019: 21021, 2018: 20798, 2017: 19659, 2016: 20579,
    2015: 19184, 2014: 20062, 2013: 20155, 2012: 19239, 2011: 18182, 2010: 17528,
    2009: 15889, 2008: 14512, 2007: 13855, 2006: 12401, 2005: 11162, 2004: 10831,
    2003: 9887, 2002: 9451, 2001: 8835, 2000: 9571, 1999: 9007, 1998: 9273,
    1997: 8759, 1996: 8771, 1995: 8623, 1994: 7893, 1993: 7519, 1992: 6446,
    1991: 5517, 1990: 4257, 1989: 2750, 1988: 2413, 1987: 1471, 1986: 160,
}

wos_categories = {
    "INFECTIOUS DISEASES": 130324,
    "IMMUNOLOGY": 95507,
    "PUBLIC ENVIRONMENTAL OCCUPATIONAL HEALTH": 69247,
    "VIROLOGY": 64306,
    "MEDICINE GENERAL INTERNAL": 39326,
    "PHARMACOLOGY PHARMACY": 31618,
    "MICROBIOLOGY": 31245,
    "BIOCHEMISTRY MOLECULAR BIOLOGY": 27655,
    "MULTIDISCIPLINARY SCIENCES": 21937,
    "SOCIAL SCIENCES BIOMEDICAL": 17389,
    "MEDICINE RESEARCH EXPERIMENTAL": 16932,
    "RESPIRATORY SYSTEM": 13197,
    "HEALTH POLICY SERVICES": 12923,
    "CELL BIOLOGY": 12012,
    "HEALTH CARE SCIENCES SERVICES": 10231,
}

# ----------------------------------------------------------------------
# 2. GLOBAL STYLE ("wow" but clean, suitable for a review/publication)
# ----------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

BLUE = "#1f77b4"
DARK = "#0b3d5c"
ACCENT = "#e67e22"
CMAP = plt.get_cmap("viridis")


def style_ax(ax, title):
    ax.set_title(title, fontsize=14, fontweight="bold", color="#111111", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
# 3. CHART 1 -- Publications per year (area + line)
# ----------------------------------------------------------------------

def plot_years(save_path="chart1_publications_per_year.png"):
    years = sorted([y for y in publication_years if y >= 1987])  # trim tiny tail
    counts = [publication_years[y] for y in years]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.fill_between(years, counts, color=BLUE, alpha=0.18, zorder=2)
    ax.plot(years, counts, color=DARK, linewidth=2.4, zorder=3)
    ax.scatter(years, counts, color=DARK, s=14, zorder=4)

    # highlight the peak year
    peak_year = years[int(np.argmax(counts))]
    peak_val = max(counts)
    ax.scatter([peak_year], [peak_val], color=ACCENT, s=90, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate(f"Peak: {peak_year}\n({peak_val:,} publications)",
                xy=(peak_year, peak_val), xytext=(peak_year - 9, peak_val + 1500),
                fontsize=10, fontweight="bold", color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5))

    # context annotation (e.g. pandemic period) - optional, edit for your field
    ax.axvspan(2020, 2022, color=ACCENT, alpha=0.08, zorder=1)
    ax.text(2021, max(counts) * 1.05, "2020-2022", ha="center", fontsize=9,
            color=ACCENT, fontweight="bold")

    ax.set_xlim(min(years), max(years))
    ax.set_ylim(0, max(counts) * 1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of publications")
    style_ax(ax, "Publication trend over time (1987-2026)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ----------------------------------------------------------------------
# 4. CHART 2 -- Top document types (horizontal bar, gradient)
# ----------------------------------------------------------------------

def plot_doc_types(save_path="chart2_document_types.png"):
    items = sorted(document_types.items(), key=lambda x: x[1])
    labels = [k.title() for k, _ in items]
    values = [v for _, v in items]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = CMAP(np.linspace(0.15, 0.85, len(values)))

    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.65, zorder=3)

    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)",
                va="center", fontsize=9.5, fontweight="bold", color="#222222")

    ax.set_xlim(0, max(values) * 1.28)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Number of publications")
    style_ax(ax, "Top 10 document types")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ----------------------------------------------------------------------
# 5. CHART 3 -- Top 15 Web of Science categories
# ----------------------------------------------------------------------

def plot_categories(save_path="chart3_wos_categories.png"):
    items = sorted(wos_categories.items(), key=lambda x: x[1])
    labels = [k.title() for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = CMAP(np.linspace(0.1, 0.9, len(values)))

    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.68, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9.5,
                fontweight="bold", color="#222222")

    ax.set_xlim(0, max(values) * 1.18)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Number of publications")
    style_ax(ax, "Top 15 Web of Science categories")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ----------------------------------------------------------------------
# 6. CHART 4 -- Combined dashboard (2x2), "wow" version for slide/poster
# ----------------------------------------------------------------------

def plot_dashboard(save_path="chart4_dashboard.png"):
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.35, wspace=0.28)

   # fig.suptitle("Bibliometric overview of the analyzed corpus",
                 #fontsize=19, fontweight="bold", color="#0b3d5c", y=0.98)

    # --- (a) publications per year, top row, full width ---
    ax1 = fig.add_subplot(gs[0, :])
    years = sorted([y for y in publication_years if y >= 1987])
    counts = [publication_years[y] for y in years]
    ax1.fill_between(years, counts, color=BLUE, alpha=0.18)
    ax1.plot(years, counts, color=DARK, linewidth=2.2)
    peak_year = years[int(np.argmax(counts))]
    ax1.scatter([peak_year], [max(counts)], color=ACCENT, s=80, zorder=5,
                edgecolor="white", linewidth=1.5)
    ax1.set_ylim(0, max(counts) * 1.15)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000):,}k"))
    style_ax(ax1, "A. Publications per year")

    # --- (b) document types, bottom-left ---
    ax2 = fig.add_subplot(gs[1, 0])
    items = sorted(document_types.items(), key=lambda x: x[1])[-8:]
    labels = [k.title() for k, _ in items]
    values = [v for _, v in items]
    colors = CMAP(np.linspace(0.15, 0.85, len(values)))
    ax2.barh(labels, values, color=colors, edgecolor="white", height=0.65)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    style_ax(ax2, "B. Top document types")

    # --- (c) WoS categories, bottom-right ---
    ax3 = fig.add_subplot(gs[1, 1])
    items = sorted(wos_categories.items(), key=lambda x: x[1])[-8:]
    labels = [k.title() for k, _ in items]
    values = [v for _, v in items]
    colors = CMAP(np.linspace(0.15, 0.85, len(values)))
    ax3.barh(labels, values, color=colors, edgecolor="white", height=0.65)
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    style_ax(ax3, "C. Top Web of Science categories")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")



# ----------------------------------------------------------------------
# 7. CHART 5 -- Cele 3 grafice una langa alta (1 x 3)
# ----------------------------------------------------------------------

def plot_side_by_side(save_path="chart5_side_by_side.png"):
    import textwrap
    fig, axes = plt.subplots(1, 3, figsize=(26, 8),
                             gridspec_kw={"width_ratios": [1.3, 1, 1.1], "wspace": 0.55})
    ax1, ax2, ax3 = axes

    # --- (a) publicatii pe an ---
    years = sorted([y for y in publication_years if y >= 1987])
    counts = [publication_years[y] for y in years]
    ax1.fill_between(years, counts, color=BLUE, alpha=0.18, zorder=2)
    ax1.plot(years, counts, color=DARK, linewidth=2.2, zorder=3)
    ax1.scatter(years, counts, color=DARK, s=12, zorder=4)
    peak_year = years[int(np.argmax(counts))]
    peak_val = max(counts)
    ax1.scatter([peak_year], [peak_val], color=ACCENT, s=80, zorder=5,
                edgecolor="white", linewidth=1.5)
    ax1.annotate(f"Peak: {peak_year}\n({peak_val:,})", xy=(peak_year, peak_val),
                 xytext=(peak_year - 12, peak_val + 1500), fontsize=10,
                 fontweight="bold", color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5))
    ax1.set_xlim(min(years), max(years))
    ax1.set_ylim(0, peak_val * 1.18)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.set_xlabel("Publication year")
    ax1.set_ylabel("Number of publications")
    style_ax(ax1, "A. Publications per year")
    ax1.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax1.grid(axis="x", visible=False)

    # --- (b) tipuri de documente ---
    items = sorted(document_types.items(), key=lambda x: x[1])
    labels = [k.title() for k, _ in items]
    values = [v for _, v in items]
    total = sum(values)
    colors = CMAP(np.linspace(0.15, 0.85, len(values)))
    bars = ax2.barh(labels, values, color=colors, edgecolor="white", height=0.65, zorder=3)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,} ({val / total * 100:.1f}%)", va="center",
                 fontsize=9, fontweight="bold")
    ax2.set_xlim(0, max(values) * 1.4)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax2.set_xlabel("Number of publications")
    style_ax(ax2, "B. Top 10 document types")

    # --- (c) categorii WoS ---
    items = sorted(wos_categories.items(), key=lambda x: x[1])
    labels = ["\n".join(textwrap.wrap(k.title(), 28)) for k, _ in items]
    values = [v for _, v in items]
    colors = CMAP(np.linspace(0.1, 0.9, len(values)))
    bars = ax3.barh(labels, values, color=colors, edgecolor="white", height=0.68, zorder=3)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=9, fontweight="bold")
    ax3.set_xlim(0, max(values) * 1.2)
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax3.set_xlabel("Number of publications")
    ax3.tick_params(axis="y", labelsize=9)
    style_ax(ax3, "C. Top 15 Web of Science categories")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    plot_years()
    plot_doc_types()
    plot_categories()
    plot_dashboard()
    plot_side_by_side()
    print("\nAll charts generated successfully.")
