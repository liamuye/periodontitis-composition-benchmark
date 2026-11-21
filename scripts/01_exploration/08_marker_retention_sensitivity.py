#!/usr/bin/env python3
"""Marker-overlap sensitivity: observed increment without dynamic removal.

Primary (required) analysis removes each tested panel's own genes from the
single-cell marker set.  That can weaken the composition baseline when a
published panel contains a marker gene, while random null panels (which contain
no markers by construction) keep the full baseline.  To show how much of each
increment is marker-driven, this script recomputes the observed increment with
the FULL marker set (no removal).  The primary null distribution is the correct
reference for this no-removal observed value because null panels never remove
a marker.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_base, FoldData, comp_features, scale_train,
                    fit_logistic, predict_logistic, logloss, fast_auc, CELLTYPES)

warnings.filterwarnings("ignore")


def load_folds(path, n_repeats, n_sites):
    d = pd.read_csv(path)
    folds = []
    for r in range(n_repeats):
        sub = d[d.repeat == r]
        fr = []
        for f in range(5):
            te = np.sort(sub[sub.fold == f].site.values)
            mask = np.ones(n_sites, dtype=bool); mask[te] = False
            fr.append((np.where(mask)[0], te))
        folds.append(fr)
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    X, probe_ids, patient, label, gsm, title, panels, g2p = load_base(args.data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    markers = pd.read_csv(os.path.join(args.out_dir, "results", "markers_celltype_top_v3.csv"))
    marker_genes = {ct: markers[markers.celltype == ct].sort_values("rank").gene_symbol.tolist()
                    for ct in CELLTYPES}
    folds = load_folds(os.path.join(args.out_dir, "results", f"folds_seed{args.seed}.csv"),
                       args.n_repeats, len(label))
    genes_all = list(g2p.keys())
    nP = len(panels); nR = args.n_repeats
    delta = np.full((nP, 2, 2, nR), np.nan)
    base_metric = np.full((nP, 2, 2, nR), np.nan)
    pred_base = np.full((nP, 2, nR, len(label)), np.nan, dtype=np.float32)
    pred_full = np.full((nP, 2, nR, len(label)), np.nan, dtype=np.float32)
    t0 = time.time()
    for r in range(nR):
        pb = np.full((nP, 2, len(label)), np.nan)
        pf = np.full((nP, 2, len(label)), np.nan)
        for (tr, te) in folds[r]:
            fd = FoldData(X, probe_ids, probe_idx, g2p, genes_all, label, tr, te)
            y_tr = label[tr]
            for pi, panel in enumerate(panels):
                comp = comp_features(fd, marker_genes, set())  # FULL marker set
                s, k, _, _ = fd.score([(m["symbol"], m.get("direction_published", 0))
                                       for m in panel["mapped"]])
                s_tr, s_te, _, _ = scale_train(s[tr], s[te])
                for bi, (ftr, fte) in enumerate([(comp["pc1_tr"], comp["pc1_te"]),
                                                 (comp["full_tr"], comp["full_te"])]):
                    if bi == 0:
                        b_tr, b_te, _, _ = scale_train(ftr, fte)
                        b_tr = b_tr[:, None]; b_te = b_te[:, None]
                    else:
                        b_tr, b_te = ftr, fte
                    pb[pi, bi, te] = predict_logistic(fit_logistic(b_tr, y_tr), b_te)
                    pf[pi, bi, te] = predict_logistic(
                        fit_logistic(np.column_stack([b_tr, s_tr]), y_tr),
                        np.column_stack([b_te, s_te]))
        y = label
        for pi in range(nP):
            for bi in range(2):
                if np.isnan(pb[pi, bi]).any() or np.isnan(pf[pi, bi]).any():
                    continue
                pred_base[pi, bi, r] = pb[pi, bi]; pred_full[pi, bi, r] = pf[pi, bi]
                base_metric[pi, bi, 0, r] = logloss(y, pb[pi, bi])
                base_metric[pi, bi, 1, r] = fast_auc(y, pb[pi, bi])
                delta[pi, bi, 0, r] = logloss(y, pb[pi, bi]) - logloss(y, pf[pi, bi])
                delta[pi, bi, 1, r] = fast_auc(y, pf[pi, bi]) - fast_auc(y, pb[pi, bi])
        print(f"repeat {r+1}/{nR} elapsed {time.time()-t0:.0f}s", flush=True)

    rows = []
    for pi, panel in enumerate(panels):
        row = {"panel_id": panel["panel_id"], "pathway": panel["pathway"]}
        for bi, bn in enumerate(["comp1d", "compfull"]):
            row[f"base_auc_{bn}"] = float(np.nanmean(base_metric[pi, bi, 1]))
            for mi, mn in enumerate(["logloss", "auc"]):
                row[f"noremoval_d{mn}_{bn}_mean"] = float(np.nanmean(delta[pi, bi, mi]))
                row[f"noremoval_d{mn}_{bn}_sd"] = float(np.nanstd(delta[pi, bi, mi], ddof=1))
        rows.append(row)
    res = pd.DataFrame(rows)

    # merge the primary null distribution (null panels keep the full baseline)
    z = np.load(os.path.join(args.out_dir, "results", "null_panel_means_primary.npz"),
                allow_pickle=True)
    nm = z["mean"]  # panel x baseline x metric x null
    for bi, bn in enumerate(["comp1d", "compfull"]):
        for mi, mn in enumerate(["logloss", "auc"]):
            vals = nm[:, bi, mi, :]
            obs = res[f"noremoval_d{mn}_{bn}_mean"].values
            res[f"null95_{mn}_{bn}"] = np.nanpercentile(vals, 95, axis=1)
            res[f"p_noremoval_{mn}_{bn}"] = [
                float((1 + np.sum(vals[i][np.isfinite(vals[i])] >= obs[i])) /
                      (np.isfinite(vals[i]).sum() + 1)) for i in range(len(res))]
            res[f"null_sd_{mn}_{bn}"] = np.nanstd(vals, axis=1, ddof=1)

    # conditional CI for the no-removal observed values
    rng = np.random.default_rng(args.seed + 888)
    patients = np.unique(patient)
    pat_sites = {int(p): np.where(patient == p)[0] for p in patients}
    boot_idx = [np.concatenate([pat_sites[int(p)] for p in rng.choice(patients, len(patients), replace=True)])
                for _ in range(args.boot)]
    for pi in range(nP):
        for bi, bn in enumerate(["comp1d", "compfull"]):
            for mi, mn in enumerate(["logloss", "auc"]):
                vals = []
                for idx in boot_idx:
                    per = []
                    for r in range(nR):
                        yb = label[idx]
                        pb = pred_base[pi, bi, r][idx]; pf = pred_full[pi, bi, r][idx]
                        per.append(logloss(yb, pb) - logloss(yb, pf) if mn == "logloss"
                                   else fast_auc(yb, pf) - fast_auc(yb, pb))
                    vals.append(np.nanmean(per))
                res.loc[pi, f"noremoval_d{mn}_{bn}_ci_lo"] = float(np.nanpercentile(vals, 2.5))
                res.loc[pi, f"noremoval_d{mn}_{bn}_ci_hi"] = float(np.nanpercentile(vals, 97.5))
    res.to_csv(os.path.join(args.out_dir, "results", "results_marker_noremoval.csv"), index=False)
    print(res[["panel_id", "pathway", "noremoval_dauc_comp1d_mean", "noremoval_dauc_compfull_mean"]]
          .to_string(index=False))
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
