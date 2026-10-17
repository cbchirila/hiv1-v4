
# HIV-1 Compound–Patent Landscape: SureChEMBL Similarity Analysis and Visualizations

## Overview

This repository contains Python scripts, input tables, and publication-oriented figures supporting a research review of HIV-related compounds and their relationships to antiviral patent literature. The workflow combines **chemical-structure similarity searches** against SureChEMBL with **patent-title filtering**, compound–patent association analysis, and network-based visualizations spanning HIV and other virus groups.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `003 - compounds/123cpds+32appHIV.smi` | Query chemical structures in SMILES format, with compound identifiers. |
| `002 - surechembl prefetch.py` | Downloads a dated SureChEMBL bulk-data snapshot of ~18 GB. |
| `002 - surechembl prefetch/` | Destination for downloaded datasets; the archive contains a source link but **not** the bulk datasets. |
| `005 - surechembl search.py` | Performs FPSim2 fingerprint similarity searches and joins matches to patent records using DuckDB. |
| `005 - surechembl search.csv.gz` | Bundled, compressed compound–SureChEMBL–patent association table used by the figure scripts. |
| `010 - edges.csv.gz` | Precomputed compound–virus-group edge data. |
| `020 - virus group nodes.csv.gz` | Precomputed virus-group summary data. |
| `030 - compounds degree.csv.gz` | Precomputed compound connectivity statistics. |
| `Figure 1 - bibliometric chart.py` | Bibliometric publication-year, document-type, and Web of Science category charts. |
| `Figure 3 - compound virus network.py` | Compound-to-virus-group network figure and corresponding edge table. |
| `Figure 4 - sankey diagram.py` | Compound–SureChEMBL–patent-jurisdiction Sankey-like visualization. |
| `Figure 5 - network graph.py` | Compound–SureChEMBL–patent network with selection tables. |
| `Figure S1 - dot plot.py` | Compound similarity dot plot. |
| `Figure S2 - tanimoto link heatmap.py` | Tanimoto-similarity heatmap based on network inputs. |
| `Figure S3 - ridgeline plot.py` | Distributional view of Tanimoto similarity. |
| `Figure S4 - compound overlap heatmap.py` | Compound-overlap/Jaccard heatmap. |
| `Figure *.png`, `Figure *.svg` | Previously generated figures; PNG and editable-vector SVG are supplied where available. |
| `Figure *.csv` | Supporting association, selection, and matrix tables for individual figures. |
| `requirements.txt` | Third-party Python dependencies (versions not pinned). |
| `run-l1.bat`, `run-l2.bat`, `run-l3.bat` | Windows batch scripts for individual workflow stages. |

**Figure numbering follows the associated manuscript/review materials**; Figure 2 is not included in this repository.

## Data sources

1. **Query structures:** `003 - compounds/123cpds+32appHIV.smi`, supplied with the project. The filename describes 123 compounds plus 32 additional HIV-related compounds; the active search configuration currently reads **only the first 123 records** (`MAX_LINES = 123`).

2. **SureChEMBL:** the prefetch script specifies the EBI-hosted bulk-data snapshot dated **2026-10-06**: <https://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/bulk_data/2026-10-06/>. The linked data include `compounds.parquet`, `fpsim2_fingerprints.h5`, `patent_compound_map.parquet`, and `patents.parquet` (plus additional Parquet tables). The accompanying `_link.txt` estimates approximately **18 GB** for the complete set; check disk availability and the source's current accessibility before downloading.

TODO 3. **Precomputed derived tables:** the compressed CSV files distributed in this repository supply the existing figure-production inputs. The process used to assign records to virus groups is not fully documented by a standalone script here; treat those mappings as supplied analytical inputs, not automatically regenerated outputs of the similarity-search script.

4. **Bibliometrics:** publication counts, document types, and Web of Science categories are **embedded directly in** `Figure 1 - bibliometric chart.py`. 

## Installation

Python 3 is required. A Windows environment is convenient for the included `.bat` launchers, but individual Python scripts can also be run on other operating systems.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```
Dependencies include `duckdb`, `FPSim2`, `matplotlib`, `networkx`, `numpy`, `pandas`, `scipy`, `seaborn`, and `tqdm`.

## Usage

Run the commands **from the repository root**; the similarity-search script uses paths relative to the current working directory, and some figure outputs are saved to the current directory.

### Option A: Regenerate figures from the included derived data

The repository includes the compressed association and network tables required by the visualization scripts, so downloading the full SureChEMBL database is **not necessary** to work with the supplied figure inputs.

On Windows:

```bat
"run-l3.bat"
```

The batch file run the bibliometric and Figure 3–5/S1–S4 generation scripts.

Run a single figure, for example:

```bash
python "Figure 3 - compound virus network.py"
```

**Note:** These scripts generate and overwrite figures and some companion CSVs. Preserve a copy of the published outputs if you need byte-for-byte comparison.

### Option B: Recompute the similarity-to-patent search

This stage requires additional large source files.

1. Download the SureChEMBL snapshot:

   ```bash
   python "002 - surechembl prefetch.py"
   ```

   On Windows, `run-l1.bat` provides the same stage. The downloader supports resuming partial files.

2. Confirm that these files exist under `002 - surechembl prefetch/`:

   ```text
   compounds.parquet
   fpsim2_fingerprints.h5
   patent_compound_map.parquet
   patents.parquet
   ```

3. Inspect the configuration constants at the top of `005 - surechembl search.py` (query selection, Tanimoto thresholds, and patent-title keywords), then run:

   ```bash
   python "005 - surechembl search.py"
   ```

   On Windows, `run-l2.bat` provides the same stage.

The script searches FPSim2 fingerprints for structures similar to each selected SMILES, retaining matches with **0.50 < Tanimoto similarity < 1.00** by default (exact matches are excluded). It joins candidate SureChEMBL identifiers to patent metadata using DuckDB and restricts the resulting patents using antiviral/virus-related terms in patent titles. Title-keyword selection is a retrieval heuristic and can include irrelevant patents or miss relevant ones.
The search stage also writes `005 - surechembl hits.csv.gz` as an intermediate file, which is not present in the supplied archive.
