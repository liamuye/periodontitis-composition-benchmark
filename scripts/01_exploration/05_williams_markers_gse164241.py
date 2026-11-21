#!/usr/bin/env python3
"""Williams 2021 (GSE164241) marker sensitivity.

Two marker resources are produced:
  1. markers_williams_published.csv  -- the paper's Table S5 (supp-12.xlsx),
     three broad groups (T/NK, B/Plasma, Granulocyte/Myeloid).  Table S4
     (supp-11.xlsx) contains only healthy-tissue clusters (epithelial,
     fibroblast, lymphatic/vascular endothelium, melanocyte, smooth muscle) and
     no immune clusters.  Neither table isolates neutrophils or separates
     macrophage/mast/DC/T/NK, so the published table alone cannot satisfy the
     "neutrophil + major immune subsets" requirement.
  2. markers_williams_derived*.csv -- regenerated here from GSE164241_RAW.tar
     with a neutrophil-aware annotation, so neutrophils and the major immune
     subsets are explicit.  Marker selection is blind to all bulk panels.
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

SIG = {
    "Neutrophil": ["FCGR3B", "CSF3R", "CXCR2", "S100A8", "S100A9", "ELANE", "MPO",
                   "DEFA3", "LCN2", "IFITM2", "NAMPT", "CMTM2"],
    "T": ["CD3D", "CD3E", "CD3G", "TRAC", "TRBC2", "IL7R", "CD2", "CD7", "LTB"],
    "NK": ["NKG7", "GNLY", "KLRD1", "GZMB", "PRF1", "KLRF1", "NCAM1", "GZMH"],
    "B": ["MS4A1", "CD79A", "CD19", "BANK1", "CD37", "CD74", "HLA-DRA"],
    "Plasma": ["MZB1", "JCHAIN", "IGHG1", "IGKC", "DERL3", "TNFRSF17", "XBP1", "IGHG3"],
    "Macrophage": ["CD68", "C1QA", "C1QB", "CD14", "LYZ", "AIF1", "FCGR1A", "MRC1", "CD163"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "MS4A2", "KIT", "HDC"],
    "DC": ["CLEC9A", "CD1C", "LILRA4", "IRF7", "IRF8", "CLEC10A", "FCER1A"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "COL3A1", "FAP"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "EGFL7", "PLVAP", "KDR"],
    "Epithelial": ["KRT5", "KRT14", "KRT13", "KRT6A", "SFN", "KRT17", "DSG3", "KRT1"],
    "SmoothMuscle": ["ACTA2", "MYH11", "TAGLN", "DES", "MYL9"],
    "Melanocyte": ["MLANA", "PMEL", "TYR", "DCT", "S100B"],
    "Lymphatic": ["LYVE1", "PROX1", "CCL21", "PDPN", "MMRN1"],
}
ORDER = ["Neutrophil", "Mast", "Plasma", "B", "DC", "Macrophage", "T", "NK",
         "Lymphatic", "Endothelial", "SmoothMuscle", "Fibroblast", "Epithelial",
         "Melanocyte"]


def load_all(root):
    prefixes = sorted({f[:-len("_matrix.mtx.gz")] for f in os.listdir(root)
                       if f.endswith("_matrix.mtx.gz")})
    # Gingiva only: the disease-relevant tissue.  Buccal (BM*) samples are
    # excluded; they add epithelial cells but not the periodontal neutrophil
    # compartment this sensitivity is about.
    prefixes = [p for p in prefixes if not p.split("_", 1)[1].startswith("BM")]
    ads = []
    for pre in prefixes:
        # sample id = text after GSM accession and first underscore
        sample = pre.split("_", 1)[1]
        a = sc.read_10x_mtx(root, prefix=pre + "_", var_names="gene_symbols",
                            make_unique=True, cache=False)
        a.obs["sample"] = sample
        a.obs["gsm"] = pre.split("_")[0]
        ads.append(a)
    a = ad.concat(ads, join="inner", merge="same")
    a.var_names_make_unique()
    a.obs_names_make_unique()
    return a, prefixes


def annotate(mean_expr, ss):
    g = lambda x: float(mean_expr.get(x, 0.0))
    s = lambda x: float(ss.get(x, 0.0))
    if g("FCGR3B") > 0.5 or (g("CSF3R") > 0.5 and g("S100A8") > 1.0 and g("CD68") < 0.5):
        return "Neutrophil"
    if g("TPSB2") > 1.0 or g("TPSAB1") > 1.0 or (g("CPA3") > 0.5 and g("MS4A2") > 0.3):
        return "Mast"
    if g("MZB1") > 1.0 and g("JCHAIN") > 0.8:
        return "Plasma"
    if g("MS4A1") > 0.5 and g("CD79A") > 0.5 and g("MZB1") < 1.0:
        return "B"
    if g("CLEC9A") > 0.5 or g("CD1C") > 0.5 or g("LILRA4") > 0.5:
        return "DC"
    if g("CD68") > 0.5 and (g("C1QA") > 0.5 or g("C1QB") > 0.5):
        return "Macrophage"
    if g("CD3D") > 0.8 and g("CD3E") > 0.5:
        return "T"
    if g("NKG7") > 1.0 and g("CD3D") < 0.3:
        return "NK"
    if g("LYVE1") > 0.5 and g("PROX1") > 0.5:
        return "Lymphatic"
    if g("PECAM1") > 0.5 or g("VWF") > 0.3 or g("CDH5") > 0.5:
        return "Endothelial"
    if g("ACTA2") > 1.0 and g("MYH11") > 0.5:
        return "SmoothMuscle"
    if g("COL1A1") > 1.0 and g("DCN") > 0.5:
        return "Fibroblast"
    if (g("KRT5") + g("KRT14")) / 2 > 1.0:
        return "Epithelial"
    if g("MLANA") > 0.5 or g("PMEL") > 0.5:
        return "Melanocyte"
    return "other"


def clean_symbol(g, valid_genes):
    g = str(g)
    if g in valid_genes:
        return g
    m = re.sub(r"-\d+$", "", g)
    return m if m in valid_genes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="./data/GSE164241")
    ap.add_argument("--supp-dir", default="./data/geo_supp")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-top", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "singlecell"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    g2p = json.load(open("./data/gpl570_gene2probe.json"))
    valid_genes = set(g2p)

    # ---- published tables
    s5 = pd.read_excel(os.path.join(args.supp_dir, "supp-12.xlsx"), header=0)
    s5["celltype"] = s5["cluster"].map({"T/NK": "T_NK", "B/Plasma": "B_Plasma",
                                        "Granulocyte/Myeloid": "Granulocyte_Myeloid"})
    s5[["celltype", "gene", "avg_logFC", "pct.1", "pct.2", "p_val_adj"]].to_csv(
        os.path.join(args.out_dir, "results", "markers_williams_published.csv"), index=False)
    s4 = pd.read_excel(os.path.join(args.supp_dir, "supp-11.xlsx"), header=0)
    s4.groupby("cluster").size().rename("n_markers").to_csv(
        os.path.join(args.out_dir, "results", "williams_tableS4_clusters.csv"))
    print("published S5 groups:", s5.groupby("celltype").size().to_dict())
    print("published S4 clusters:", s4.cluster.nunique(), "(healthy tissue only)")

    # ---- raw atlas
    a, prefixes = load_all(args.raw_dir)
    raw_cells = a.n_obs
    a.var["mt"] = a.var_names.str.startswith("MT-")
    a.var["ribo"] = a.var_names.str.match(r"^RP[SL]")
    a.var["hb"] = a.var_names.str.match(r"^HB[AB]")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt", "ribo", "hb"], percent_top=None,
                               log1p=False, inplace=True)
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_cells(a, min_counts=500)
    a = a[(a.obs["pct_counts_mt"] < 20) &
          (a.obs["n_genes_by_counts"] < 6000) &
          (a.obs["total_counts"] < 50000)].copy()
    sc.pp.filter_genes(a, min_cells=10)
    print(f"cells raw {raw_cells} -> {a.n_obs}; genes {a.n_vars}")

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
    sc.tl.leiden(b, resolution=1.5, key_added="leiden", random_state=0,
                 flavor="igraph", n_iterations=2, directed=False, adjacency=W)
    a.obs["leiden"] = b.obs["leiden"].values

    for k, genes in SIG.items():
        genes = [g for g in genes if g in a.var_names]
        sc.tl.score_genes(a, genes, score_name=f"score_{k}", use_raw=False)

    genes_for_means = sorted({g for v in SIG.values() for g in v if g in a.var_names})
    expr = pd.DataFrame(a[:, genes_for_means].X.toarray(), columns=genes_for_means)
    expr["leiden"] = a.obs["leiden"].values
    mean_expr = expr.groupby("leiden").mean()
    sig = a.obs[[f"score_{k}" for k in SIG] + ["leiden"]].groupby("leiden").mean()

    rows, cluster_map = [], {}
    for cl in mean_expr.index:
        me = mean_expr.loc[cl].to_dict()
        ss = {k.replace("score_", ""): v for k, v in sig.loc[cl].to_dict().items()}
        ct = annotate(me, ss)
        cluster_map[str(cl)] = ct
        row = {"cluster": str(cl), "n_cells": int((a.obs["leiden"] == cl).sum()), "celltype": ct}
        for k in SIG:
            row[f"score_{k}"] = float(ss.get(k, np.nan))
        for g in genes_for_means:
            row[f"mean_{g}"] = float(me.get(g, np.nan))
        rows.append(row)
    clusters = pd.DataFrame(rows).sort_values("cluster", key=lambda s: s.astype(int))
    comp = pd.crosstab(a.obs["leiden"], a.obs["sample"])
    comp.columns = [f"n_{c}" for c in comp.columns]
    clusters = clusters.merge(comp.reset_index().rename(columns={"leiden": "cluster"}),
                              on="cluster", how="left")
    clusters.to_csv(os.path.join(args.out_dir, "singlecell", "williams_clusters.csv"), index=False)
    a.obs["celltype"] = a.obs["leiden"].map(cluster_map).astype(str)
    print(a.obs["celltype"].value_counts().to_string())
    a.write_h5ad(os.path.join(args.out_dir, "singlecell", "williams_processed.h5ad"))
    print(clusters[["cluster", "n_cells", "celltype"]].to_string(index=False))

    groups = [g for g in ORDER if (a.obs["celltype"] == g).sum() >= 30]
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
        rows.append(df)
    markers = pd.concat(rows, ignore_index=True)
    markers = markers.sort_values(["gene_symbol", "spec", "celltype"],
                                  ascending=[True, False, True])
    markers = markers.drop_duplicates("gene_symbol", keep="first").copy()
    markers["rank"] = markers.groupby("celltype")["spec"].rank(method="first", ascending=False).astype(int)
    markers["selected_top"] = markers["rank"] <= args.n_top
    markers = markers.sort_values(["celltype", "rank"])
    markers.to_csv(os.path.join(args.out_dir, "results", "markers_williams_derived.csv"), index=False)
    top = markers[markers["selected_top"]].copy()
    top[["celltype", "rank", "gene_symbol", "logfoldchanges", "pct_nz_group",
         "pct_nz_reference", "delta_pct", "spec", "scores", "pvals_adj"]] \
        .to_csv(os.path.join(args.out_dir, "results", "markers_williams_derived_top.csv"), index=False)

    # per-sample composition
    ct = pd.crosstab(a.obs["sample"], a.obs["celltype"], normalize="index")
    ct.to_csv(os.path.join(args.out_dir, "singlecell", "williams_composition.csv"))

    # dot plot of canonical markers
    dp = [g for g in ["FCGR3B", "CSF3R", "CXCR2", "S100A8", "S100A9", "CD3D", "NKG7",
                      "MS4A1", "MZB1", "JCHAIN", "CD68", "C1QA", "TPSAB1", "CPA3",
                      "CLEC9A", "COL1A1", "DCN", "PECAM1", "VWF", "KRT5", "KRT14",
                      "ACTA2", "MLANA"] if g in a.var_names]
    sc.pl.dotplot(a, dp, groupby="leiden", standard_scale="var", show=False)
    plt.savefig(os.path.join(args.out_dir, "figures", "williams_marker_dotplot.png"),
                dpi=200, bbox_inches="tight")
    plt.close("all")

    with open(os.path.join(args.out_dir, "singlecell", "williams_session_info.txt"), "w") as f:
        f.write(f"scanpy {sc.__version__}\n")
        import anndata
        f.write(f"anndata {anndata.__version__}\n")
        f.write(f"numpy {np.__version__}; pandas {pd.__version__}\n")
        f.write(f"samples: {len(prefixes)}\n")
        f.write(f"raw cells: {raw_cells}; after QC: {a.n_obs}; genes: {a.n_vars}\n")
        f.write(f"clusters: {a.obs['leiden'].nunique()}; resolution 0.8\n")
        f.write(f"cell type counts:\n{a.obs['celltype'].value_counts().to_string()}\n")
        f.write("panel gene pre-filter: NONE\n")

    print(f"derived markers pass filters: {len(markers)}; top{args.n_top}: {len(top)}")
    print(top.groupby("celltype").size().to_string())


if __name__ == "__main__":
    main()
