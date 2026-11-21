# Evaluating Cellular-Composition Effects and Incremental Predictive Value in Periodontitis Transcriptomic Gene Signatures

This repository archives the reference resources, fixed-identity random-panel gene lists, array-level microarray overlap and scan-date annotations, end-to-end benchmarking and plotting scripts, numerical outputs, and supplementary files accompanying the manuscript:

> **Evaluating cellular-composition effects and incremental predictive value in periodontitis transcriptomic gene signatures**
>
> Qiao Kang, Long Chen, Zihan Ma, Mingyu Liu

## Repository Structure

- `data/`
  - `gse152042_150_composition_markers.csv`: 150 mutually exclusive GPL570-mapped single-cell markers across five gingival compartments (30 per compartment) derived from GSE152042.
  - `gse164241_360_composition_markers.csv`: 360 mutually exclusive markers across 12 compartments (including a dedicated neutrophil cluster) derived from GSE164241.
  - `GSE10334_GSE16134_identical_arrays.tsv`: Array-level mapping of the 244 identical Affymetrix GPL570 CEL files shared between GSE10334 and GSE16134 (verified by DAT scanner headers and raw probe-intensity MD5 checksums).
  - `GSE16134_scan_dates_and_patients.csv`: Patient IDs, clinical site labels, and CEL hardware scan timestamps (2004–2008) for the 307 analyzed GSE16134 biopsies.
  - `published_42_panels_metadata.tsv`: Curated metadata for all 42 published periodontitis gene signatures (Panels 1–27 exploration; Panels 28–42 held-out confirmation).
  - `fixed_random_panels_exploration_B1000.csv.gz` & `fixed_random_panels_confirmation_B20000.tsv.gz`: Fixed-identity, marginal-|Welch t|-matched random gene sets used across all 20 × 5 patient-clustered cross-validation splits.
- `scripts/` *(archived stage-by-stage Python/R analysis scripts sanitized of local machine paths; end-to-end re-execution from raw NCBI GEO downloads requires supplying downloaded GEO archives via each script's `--data-dir` / path arguments, while all derived marker catalogs, fixed random-panel gene lists, and numerical outputs are provided directly in `data/` and `results/`)*
  - `01_exploration/`: Data preparation, GSE152042/GSE164241 single-cell marker derivation, fixed-identity random panel generation, 27-panel incremental benchmark, and selection-aware patient-clustered bootstrap.
  - `02_confirmation/`: Held-out 15-panel benchmark (*B* = 20,000), Clopper–Pearson boundary analysis, and pre-specified confirmation rule checks.
  - `03_sensitivities_and_calibration/`: De novo CEL-level RMA renormalization (`cel_rma_renormalize.R`), scan-year 2008 exclusion, within-patient paired-biopsy restriction, multivariate gene-level L2 / penalty / label-free probe checks, Panel 38 gene ablation and overlap-to-unique transfer, fold-adaptive and per-fold-redraw alternative null distributions, orthogonal spike-in detectability calibration, and external healthy-donor RNA-seq evaluations (GSE173078 and GSE223924).
  - `04_figures/`: Publication figure generator (`make_figures.py`) producing `Fig1.tif`–`Fig6.tif` and `S1_Fig.tif`.
- `results/`: Complete per-panel and per-test numerical CSV tables supporting all main tables, figures, and sensitivity analyses.
- `figures/`: Publication-ready 300-dpi RGB LZW-compressed TIFF figures (`Fig1.tif`–`Fig6.tif`, `S1_Fig.tif`).
- `supplementary/`: Complete supplementary documents (`S1_Text.docx` and multi-sheet `S1_Table.xlsx`–`S6_Table.xlsx`).

## Public GEO Datasets

All expression data are publicly available from NCBI GEO without restriction: `GSE16134`, `GSE10334`, `GSE152042`, `GSE164241`, `GSE173078`, and `GSE223924`.

## Software Environment

- **Python 3.9.6** (see `requirements.txt`: `numpy==2.0.2`, `scipy==1.13.1`, `pandas==2.3.3`, `scikit-learn==1.6.1`, `scanpy==1.10.3`, `anndata==0.10.9`, `harmonypy==2.1.0`).
- **R 4.5.3** with Bioconductor 3.22 (`affy==1.88.0`) for de novo CEL-level RMA renormalization.

## License

Code is released under the MIT License (see `LICENSE`).
