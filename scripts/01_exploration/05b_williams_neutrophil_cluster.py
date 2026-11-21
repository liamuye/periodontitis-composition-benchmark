#!/usr/bin/env python3
"""Rescue the rare neutrophil population from the macrophage cluster.

Global Leiden clustering merged the rare FCGR3B/CSF3R/CXCR2-high neutrophil
population into the macrophage cluster.  We subcluster the macrophage cells at
resolution 1.5 and assign the neutrophil-like subcluster(s) to Neutrophil, then
re-derive markers for all types (including Neutrophil) with the same
Wilcoxon/filter/top-30 pipeline.  Marker selection remains blind to all bulk
panels.
"""
import argparse, json, os, re, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

ORDER = ["Neutrophil", "Mast", "Plasma", "B", "DC", "Macrophage", "T", "NK",
         "Lymphatic", "Endothelial", "SmoothMuscle", "Fibroblast", "Epithelial",
         "Melanocyte"]


def clean_symbol(g, valid_genes):
    g = str(g)
    if g in valid_genes:
        return g
    m = re.sub(r"-\d+$", "", g)
    return m if m in valid_genes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-top", type=int, default=30)
    args = ap.parse_args()

    h5 = os.path.join(args.out_dir, "singlecell", "williams_processed.h5ad")
    a = sc.read_h5ad(h5)
    a.obs["celltype"] = a.obs["celltype"].astype(str)
    g2p = json.load(open("./data/gpl570_gene2probe.json"))
    valid_genes = set(g2p)
    print("loaded", a.shape, a.obs.celltype.value_counts().to_dict())

    if "Neutrophil" in set(a.obs.celltype):
        print("Neutrophil already present; no rescue needed")
    else:
        mac = a[a.obs.celltype == "Macrophage"].copy()
        print("macrophage cells", mac.shape)
        sc.pp.normalize_total(mac, target_sum=1e4)
        sc.pp.log1p(mac)
        mac.raw = mac
        sc.pp.highly_variable_genes(mac, n_top_genes=2000, flavor="seurat")
        b = mac[:, mac.var.highly_variable].copy()
        sc.pp.scale(b, max_value=10)
        sc.tl.pca(b, n_comps=30, svd_solver="arpack", random_state=0)
        X = b.obsm["X_pca"]
        nn = NearestNeighbors(n_neighbors=16, metric="euclidean", n_jobs=-1).fit(X)
        dist, idx = nn.kneighbors(X)
        n = X.shape[0]
        rows, cols, vals = [], [], []
        for k in range(1, 16):
            rows.extend(range(n)); cols.extend(idx[:, k]); vals.extend(np.ones(n))
        W = csr_matrix((vals, (rows, cols)), shape=(n, n)); W = W.maximum(W.T)
        sc.tl.leiden(b, resolution=1.5, key_added="subcl", random_state=0,
                     flavor="igraph", n_iterations=2, directed=False, adjacency=W)
        mac.obs["subcl"] = b.obs["subcl"].values

        genes = ["FCGR3B", "CSF3R", "CXCR2", "ELANE", "MPO", "DEFA3", "S100A8",
                 "S100A9", "CD68", "C1QA", "CD14", "LYZ", "CD3D", "COL1A1", "TPSAB1"]
        genes = [g for g in genes if g in mac.var_names]
        expr = pd.DataFrame(mac[:, genes].X.toarray(), columns=genes)
        expr["subcl"] = mac.obs["subcl"].values
        m = expr.groupby("subcl").mean()
        m["n"] = mac.obs["subcl"].value_counts()
        m["neutrophil_like"] = ((m["FCGR3B"] > 1.0) & (m["CD68"] < 0.3) & (m["C1QA"] < 0.3)) | \
                               ((m["CSF3R"] > 2.0) & (m["CD68"] < 0.3) & (m["C1QA"] < 0.3))
        m.to_csv(os.path.join(args.out_dir, "singlecell", "williams_neutrophil_rescue.csv"))
        print(m.sort_values("FCGR3B", ascending=False).round(2).to_string())

        neut_sub = m.index[m["neutrophil_like"]].tolist()
        names = mac.obs_names[mac.obs["subcl"].isin(neut_sub)]
        print(f"assigning {len(names)} cells from subclusters {neut_sub} to Neutrophil")
        a.obs.loc[names, "celltype"] = "Neutrophil"
        # keep the original global cluster label for audit
        a.obs["macrophage_subcluster"] = "NA"
        a.obs.loc[mac.obs_names, "macrophage_subcluster"] = mac.obs["subcl"].values

    print(a.obs.celltype.value_counts().to_string())
    a.write_h5ad(h5)

    # re-derive markers on the updated celltype
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

    ct = pd.crosstab(a.obs["sample"], a.obs["celltype"], normalize="index")
    ct.to_csv(os.path.join(args.out_dir, "singlecell", "williams_composition.csv"))
    print(f"derived markers pass filters: {len(markers)}; top{args.n_top}: {len(top)}")
    print(top.groupby("celltype").size().to_string())


if __name__ == "__main__":
    main()
