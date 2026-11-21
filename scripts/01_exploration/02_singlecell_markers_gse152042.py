#!/usr/bin/env python3
"""GSE152042 single-cell marker derivation, v3.

Correction vs v2: marker selection is BLIND to every bulk panel.  No bulk
panel gene is removed before or during marker selection.  When a published
panel is later evaluated, only that panel's own genes are removed from the
composition baseline (dynamic leave-target-panel-genes-out, see
04_incremental_v3.py).  This makes exploration and confirmation panels
symmetric.

No GSE16134 expression or label is read here.  Marker direction in the bulk
score is fixed +1; no bulk label is used to orient markers.
"""
import argparse, json, os, re, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import harmonypy
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

CANON = {
    "plasma": ["MZB1", "JCHAIN", "IGHG1", "IGKC", "TNFRSF17", "DERL3", "XBP1", "FKBP11", "SSR4", "HERPUD1"],
    "other_immune": ["PTPRC", "CD3D", "CD3E", "CD2", "IL7R", "LTB", "CD79A", "MS4A1", "LYZ", "CD68",
                     "CCL5", "NKG7", "FCGR3B", "TPSAB1", "TPSB2", "CPA3", "CLEC9A", "LILRA4"],
    "keratinocyte": ["KRT5", "KRT14", "KRT1", "KRT10", "KRT13", "KRT4", "KRT6A", "KRT16", "KRT17",
                     "DSG3", "DSC3", "LGALS7"],
    "fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA", "FAP", "THY1", "COL5A1", "COL6A3"],
    "endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "EGFL7", "PLVAP", "KDR", "ESAM", "PROX1", "CCL21"],
}

TYPES = ["plasma", "other_immune", "keratinocyte", "fibroblast", "endothelial"]


def load_10x(data_dir):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
    if not files:
        raise FileNotFoundError(f"no .h5 files in {data_dir}")
    ads = []
    for f in files:
        a = sc.read_10x_h5(os.path.join(data_dir, f))
        a.var_names_make_unique()
        sample = re.sub(r"^GSM\d+_filtered_feature_bc_matrix_", "", f).replace(".h5", "")
        a.obs["sample"] = sample
        ads.append(a)
    a = ad.concat(ads, join="inner", merge="same")
    a.var_names_make_unique()
    a.obs_names_make_unique()
    return a, files


def annotate_cluster(mean_expr, sig_scores):
    g = lambda x: float(mean_expr.get(x, 0.0))
    s = lambda x: float(sig_scores.get(x, 0.0))
    if (g("MZB1") > 1.0 and (g("IGHG1") > 1.0 or g("IGKC") > 2.0 or g("IGLL5") > 1.0)
            and g("LILRA4") < 1.0 and g("GZMB") < 1.5 and s("plasma") > 0.8):
        return "plasma"
    if (g("PECAM1") > 0.5 or g("VWF") > 0.3 or g("CDH5") > 0.5) and s("endothelial") > 0.4:
        return "endothelial"
    if ((g("KRT5") + g("KRT14")) / 2 > 1.0) and s("keratinocyte") > 0.4:
        return "keratinocyte"
    if g("COL1A1") > 1.0 and g("DCN") > 0.5 and s("fibroblast") > 0.4:
        return "fibroblast"
    if (g("PTPRC") > 0.5 or g("TPSB2") > 1.0 or g("TPSAB1") > 1.0 or g("CLEC9A") > 1.0
            or (g("LYZ") > 2.0 and g("CD68") > 0.5) or g("MS4A1") > 1.0
            or g("CD3D") > 0.8 or g("LILRA4") > 1.0):
        return "other_immune"
    return "other"


def clean_symbol(g, valid_genes):
    g = str(g)
    if g in valid_genes:
        return g
    m = re.sub(r"-\d+$", "", g)
    return m if m in valid_genes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-top", type=int, default=30)
    args = ap.parse_args()

    sc_dir = os.path.join(args.data_dir, "sc")
    os.makedirs(os.path.join(args.out_dir, "singlecell"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    g2p = json.load(open(os.path.join(args.data_dir, "gpl570_gene2probe.json")))
    valid_genes = set(g2p)

    a, files = load_10x(sc_dir)
    raw_cells = a.n_obs
    a.var["mt"] = a.var_names.str.startswith("MT-")
    a.var["ribo"] = a.var_names.str.match(r"^RP[SL]")
    a.var["hb"] = a.var_names.str.match(r"^HB[AB]")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt", "ribo", "hb"], percent_top=None,
                               log1p=False, inplace=True)
    qc_raw = a.obs.groupby("sample").agg(
        cells_raw=("n_genes_by_counts", "size"),
        median_genes=("n_genes_by_counts", "median"),
        median_counts=("total_counts", "median"),
        median_pct_mt=("pct_counts_mt", "median"),
    )

    sc.pp.filter_cells(a, min_genes=300)
    sc.pp.filter_cells(a, min_counts=500)
    a = a[(a.obs["pct_counts_mt"] < 20) &
          (a.obs["n_genes_by_counts"] < 6000) &
          (a.obs["total_counts"] < 50000)].copy()
    sc.pp.filter_genes(a, min_cells=10)
    qc_after = a.obs.groupby("sample").agg(
        cells_after=("n_genes_by_counts", "size"),
        median_genes_after=("n_genes_by_counts", "median"),
        median_counts_after=("total_counts", "median"),
        median_pct_mt_after=("pct_counts_mt", "median"),
    )
    qc = qc_raw.join(qc_after).reset_index()
    qc.to_csv(os.path.join(args.out_dir, "singlecell", "singlecell_qc.csv"), index=False)

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a
    sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key="sample", flavor="seurat")
    b = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=30, svd_solver="arpack", random_state=0)
    ho = harmonypy.run_harmony(b.obsm["X_pca"], b.obs, "sample",
                               max_iter_harmony=20, random_state=0)
    b.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr)[:, :30]

    X = b.obsm["X_pca_harmony"]
    nn = NearestNeighbors(n_neighbors=16, metric="euclidean", n_jobs=-1).fit(X)
    dist, idx = nn.kneighbors(X)
    n = X.shape[0]
    rows, cols, vals = [], [], []
    for k in range(1, 16):
        rows.extend(range(n)); cols.extend(idx[:, k]); vals.extend(np.ones(n))
    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)
    sc.tl.leiden(b, resolution=0.8, key_added="leiden", random_state=0,
                 flavor="igraph", n_iterations=2, directed=False, adjacency=W)
    a.obs["leiden"] = b.obs["leiden"].values

    for k, genes in CANON.items():
        genes = [g for g in genes if g in a.var_names]
        sc.tl.score_genes(a, genes, score_name=f"score_{k}", use_raw=False)

    genes_for_means = sorted({g for v in CANON.values() for g in v if g in a.var_names})
    expr = pd.DataFrame(a[:, genes_for_means].X.toarray(), columns=genes_for_means)
    expr["leiden"] = a.obs["leiden"].values
    mean_expr = expr.groupby("leiden").mean()
    sig = a.obs[[f"score_{k}" for k in CANON] + ["leiden"]].groupby("leiden").mean()

    cluster_rows = []
    cluster_map = {}
    for cl in mean_expr.index:
        me = mean_expr.loc[cl].to_dict()
        ss = {k.replace("score_", ""): v for k, v in sig.loc[cl].to_dict().items()}
        celltype = annotate_cluster(me, ss)
        cluster_map[str(cl)] = celltype
        row = {"cluster": str(cl), "n_cells": int((a.obs["leiden"] == cl).sum()),
               "celltype": celltype}
        for k in CANON:
            row[f"score_{k}"] = float(ss.get(k, np.nan))
        for g in genes_for_means:
            row[f"mean_{g}"] = float(me.get(g, np.nan))
        cluster_rows.append(row)
    clusters = pd.DataFrame(cluster_rows).sort_values("cluster", key=lambda s: s.astype(int))
    comp = pd.crosstab(a.obs["leiden"], a.obs["sample"])
    comp.columns = [f"n_{c}" for c in comp.columns]
    clusters = clusters.merge(comp.reset_index().rename(columns={"leiden": "cluster"}),
                              on="cluster", how="left")
    clusters.to_csv(os.path.join(args.out_dir, "singlecell", "singlecell_clusters.csv"), index=False)

    a.obs["celltype"] = a.obs["leiden"].map(cluster_map).astype(str)
    print(a.obs["celltype"].value_counts().to_string())

    groups = [g for g in TYPES if (a.obs["celltype"] == g).sum() >= 30]
    sc.tl.rank_genes_groups(a, "celltype", groups=groups, reference="rest",
                            method="wilcoxon", use_raw=True, n_genes=2000, pts=True)

    rows = []
    for grp in groups:
        df = sc.get.rank_genes_groups_df(a, group=grp).copy()
        df["celltype"] = grp
        df["delta_pct"] = df["pct_nz_group"] - df["pct_nz_reference"]
        df["spec"] = df["logfoldchanges"] * df["delta_pct"]
        df["gene_symbol"] = [clean_symbol(g, valid_genes) for g in df["names"]]
        df = df[df["gene_symbol"].notna()].copy()
        df = df[(df["logfoldchanges"] > 1.0) & (df["pct_nz_group"] > 0.25)
                & (df["delta_pct"] > 0.10) & (df["pvals_adj"] < 0.01)]
        df = df[~df["gene_symbol"].str.match(r"^(MT-|RPL|RPS|MRPL|MRPS|HB[AB])", na=False)]
        # v3 correction: NO panel-gene exclusion here.  Marker selection is blind
        # to every bulk panel; dynamic per-panel removal happens at evaluation.
        rows.append(df)
    markers = pd.concat(rows, ignore_index=True)

    markers = markers.sort_values(["gene_symbol", "spec", "celltype"],
                                  ascending=[True, False, True])
    markers = markers.drop_duplicates("gene_symbol", keep="first").copy()
    markers["rank"] = markers.groupby("celltype")["spec"].rank(method="first", ascending=False).astype(int)
    markers["selected_top"] = markers["rank"] <= args.n_top
    markers = markers.sort_values(["celltype", "rank"])
    markers.to_csv(os.path.join(args.out_dir, "results", "markers_celltype_v3.csv"), index=False)
    top = markers[markers["selected_top"]].copy()
    top[["celltype", "rank", "gene_symbol", "logfoldchanges", "pct_nz_group",
         "pct_nz_reference", "delta_pct", "spec", "scores", "pvals_adj"]] \
        .to_csv(os.path.join(args.out_dir, "results", "markers_celltype_top_v3.csv"), index=False)

    pcs = b.obsm["X_pca_harmony"][:, :2]
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"plasma": "#d62728", "other_immune": "#1f77b4", "keratinocyte": "#2ca02c",
              "fibroblast": "#ff7f0e", "endothelial": "#9467bd", "other": "#7f7f7f"}
    for ct in ["other", "fibroblast", "endothelial", "keratinocyte", "other_immune", "plasma"]:
        m = (a.obs["celltype"] == ct).values
        if m.sum():
            ax.scatter(pcs[m, 0], pcs[m, 1], s=2, c=colors.get(ct, "#333333"),
                       label=f"{ct} (n={m.sum()})", alpha=0.6, linewidths=0)
    ax.set_xlabel("Harmony PC1"); ax.set_ylabel("Harmony PC2")
    ax.set_title("GSE152042: annotated clusters (v3, markers blind to bulk panels)")
    ax.legend(markerscale=4, fontsize=8, frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "figures", "singlecell_pca_v3.png"), dpi=200)
    plt.close(fig)

    with open(os.path.join(args.out_dir, "singlecell", "singlecell_session_info_v3.txt"), "w") as f:
        f.write(f"scanpy {sc.__version__}\n")
        import anndata
        f.write(f"anndata {anndata.__version__}\n")
        import sklearn
        f.write(f"scikit-learn {sklearn.__version__}\n")
        f.write(f"harmonypy {getattr(harmonypy, '__version__', 'unknown')}\n")
        f.write(f"numpy {np.__version__}\n")
        f.write(f"pandas {pd.__version__}\n")
        f.write(f"files: {files}\n")
        f.write(f"raw cells: {raw_cells}; after QC: {a.n_obs}; genes: {a.n_vars}\n")
        f.write(f"clusters: {a.obs['leiden'].nunique()}; resolution 0.8\n")
        f.write(f"cell type counts:\n{a.obs['celltype'].value_counts().to_string()}\n")
        f.write("panel gene pre-filter: NONE (v3 correction)\n")

    print(f"markers pass filters: {len(markers)}; selected top{args.n_top}: {len(top)}")
    print(top.groupby("celltype").size().to_string())


if __name__ == "__main__":
    main()
