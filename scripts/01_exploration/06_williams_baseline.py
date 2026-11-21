#!/usr/bin/env python3
"""Williams 2021 marker sensitivity for the 27 panels.

Two marker resources:
  * derived  : GSE164241_RAW.tar annotation with explicit Neutrophil and major
               immune subsets (T, NK, B, Plasma, Macrophage, Mast, DC) plus
               stromal/epithelial/endothelial types;
  * published: Table S5's three broad groups (T/NK, B/Plasma,
               Granulocyte/Myeloid).  It has no separate neutrophil group and
               is therefore reported as a coarser sensitivity only.

Both are applied with the same dynamic leave-target-panel-genes-out rule as
the primary GSE152042 baseline.  Null distributions use the same fixed random
panels as the primary analysis.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_base, FoldData, comp_features, panel_gene_sets,
                    scale_train, scale_train_batch, fit_logistic, predict_logistic,
                    fit_logistic_batch, predict_logistic_batch, logloss, logloss_batch,
                    fast_auc, fast_auc_batch, bh_q)

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


def null_score_matrix(fd, genes_per_null):
    rows, cols, vals = [], [], []
    for b, genes in enumerate(genes_per_null):
        pos = [fd.gene_pos[g] for g in genes if g in fd.gene_pos]
        k = len(pos)
        if k == 0:
            continue
        for p in pos:
            s = 1.0 if fd.gene_t[fd.genes[p]] >= 0 else -1.0
            rows.append(p); cols.append(b); vals.append(s / k)
    W = csr_matrix((np.asarray(vals, float), (rows, cols)),
                   shape=(len(fd.genes), len(genes_per_null)))
    return np.asarray(fd.Z @ W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    X, probe_ids, patient, label, gsm, title, panels, g2p = load_base(args.data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    nP, nS, nR, nB = len(panels), len(label), args.n_repeats, args.n_null
    sets = panel_gene_sets(panels)
    alias = {p["panel_id"]: {m["symbol"] for m in p["mapped"]} | {m["gene"] for m in p["mapped"]}
             for p in panels}

    derived = pd.read_csv(os.path.join(args.out_dir, "results", "markers_williams_derived_top.csv"))
    m_derived = {ct: g.sort_values("rank").gene_symbol.tolist()
                 for ct, g in derived.groupby("celltype")}
    pub = pd.read_csv(os.path.join(args.out_dir, "results", "markers_williams_published.csv"))
    m_pub = {ct: g.sort_values("avg_logFC", ascending=False).gene.tolist()
             for ct, g in pub.groupby("celltype")}
    print("derived types:", {k: len(v) for k, v in m_derived.items()})
    print("published types:", {k: len(v) for k, v in m_pub.items()})

    ident = pd.read_csv(os.path.join(args.out_dir, "results", "null_panels_identity.csv.gz"))
    null_genes = {}
    for p in panels:
        sub = ident[ident.panel_id == p["panel_id"]].sort_values(["null_id", "position"])
        by = sub.groupby("null_id")["null_gene"].apply(list)
        null_genes[p["panel_id"]] = [by[k] for k in sorted(by.index)][:nB]
        assert len(null_genes[p["panel_id"]]) == nB

    folds = load_folds(os.path.join(args.out_dir, "results", f"folds_seed{args.seed}.csv"), nR, nS)
    genes_all = list(g2p.keys())
    names = ["williams_derived_1d", "williams_derived_full", "williams_published_1d", "williams_published_full"]
    obs_delta = np.full((nP, 4, 2, nR), np.nan)
    obs_base = np.full((nP, 4, 2, nR), np.nan)
    obs_pred_base = np.full((nP, 4, nR, nS), np.nan, dtype=np.float32)
    obs_pred_full = np.full((nP, 4, nR, nS), np.nan, dtype=np.float32)
    null_delta = np.full((nP, 4, 2, nB, nR), np.nan, dtype=np.float32)
    n_types = np.full((nP, 4, nR), np.nan)

    t0 = time.time()
    for r in range(nR):
        pred_null_rep = np.full((nP, 4, nS, nB), np.nan, dtype=np.float32)
        pred_base_rep = np.full((nP, 4, nS), np.nan)
        pred_full_rep = np.full((nP, 4, nS), np.nan)
        for f, (tr, te) in enumerate(folds[r]):
            fd = FoldData(X, probe_ids, probe_idx, g2p, genes_all, label, tr, te)
            y_tr = label[tr]
            for pi, panel in enumerate(panels):
                obs_s, k, _, _ = fd.score([(m["symbol"], m.get("direction_published", 0))
                                           for m in panel["mapped"]])
                s_tr, s_te, _, _ = scale_train(obs_s[tr], obs_s[te])
                S_null = null_score_matrix(fd, null_genes[panel["panel_id"]])
                S_tr, S_te, _, _ = scale_train_batch(S_null[tr], S_null[te])
                for mg, bases in [(m_derived, (0, 1)), (m_pub, (2, 3))]:
                    comp = comp_features(fd, mg, alias[panel["panel_id"]])
                    for outbase in bases:
                        if outbase in (0, 2):
                            b_tr, b_te, _, _ = scale_train(comp["pc1_tr"], comp["pc1_te"])
                            b_tr = b_tr[:, None]; b_te = b_te[:, None]
                        else:
                            b_tr = comp["full_tr"]; b_te = comp["full_te"]
                        pred_base_rep[pi, outbase, te] = predict_logistic(fit_logistic(b_tr, y_tr), b_te)
                        pred_full_rep[pi, outbase, te] = predict_logistic(
                            fit_logistic(np.column_stack([b_tr, s_tr]), y_tr),
                            np.column_stack([b_te, s_te]))
                        n_types[pi, outbase, r] = len(comp["types"])
                        coef = fit_logistic_batch(b_tr, S_tr, y_tr)
                        pred_null_rep[pi, outbase, te, :] = predict_logistic_batch(coef, b_te, S_te)
        y = label
        for pi in range(nP):
            for bi in range(4):
                base = pred_base_rep[pi, bi]; full = pred_full_rep[pi, bi]
                if np.isnan(base).any() or np.isnan(full).any():
                    continue
                obs_pred_base[pi, bi, r] = base
                obs_pred_full[pi, bi, r] = full
                obs_base[pi, bi, 0, r] = logloss(y, base)
                obs_base[pi, bi, 1, r] = fast_auc(y, base)
                obs_delta[pi, bi, 0, r] = logloss(y, base) - logloss(y, full)
                obs_delta[pi, bi, 1, r] = fast_auc(y, full) - fast_auc(y, base)
                P = pred_null_rep[pi, bi]
                ok = ~np.isnan(P[0])
                if ok.any():
                    null_delta[pi, bi, 0, ok, r] = logloss(y, base) - logloss_batch(y, P[:, ok])
                    null_delta[pi, bi, 1, ok, r] = fast_auc_batch(y, P[:, ok]) - fast_auc(y, base)
        del pred_null_rep
        print(f"repeat {r+1}/{nR} elapsed {time.time()-t0:.0f}s", flush=True)

    # summaries
    rows = []
    for pi, panel in enumerate(panels):
        row = {"panel_id": panel["panel_id"], "pathway": panel["pathway"],
               "n_genes": panel["n_genes"]}
        for bi, nm in enumerate(names):
            row[f"n_types_{nm}"] = float(np.nanmean(n_types[pi, bi]))
            row[f"base_logloss_{nm}"] = float(np.nanmean(obs_base[pi, bi, 0]))
            row[f"base_auc_{nm}"] = float(np.nanmean(obs_base[pi, bi, 1]))
            for mi, mn in enumerate(["logloss", "auc"]):
                vals = obs_delta[pi, bi, mi]
                row[f"d{mn}_{nm}_mean"] = float(np.nanmean(vals))
                row[f"d{mn}_{nm}_sd"] = float(np.nanstd(vals, ddof=1))
                nmv = np.nanmean(null_delta[pi, bi, mi], axis=1)
                nmv = nmv[np.isfinite(nmv)]
                row[f"p_{mn}_{nm}"] = float((1 + np.sum(nmv >= np.nanmean(vals))) / (len(nmv) + 1))
                row[f"null_median_{mn}_{nm}"] = float(np.nanmedian(nmv))
                row[f"null95_{mn}_{nm}"] = float(np.nanpercentile(nmv, 95))
                row[f"null_sd_{mn}_{nm}"] = float(np.nanstd(nmv, ddof=1))
                row[f"n_null_{mn}_{nm}"] = int(len(nmv))
        rows.append(row)
    res = pd.DataFrame(rows)
    for nm in names:
        for mn in ["logloss", "auc"]:
            res[f"q_{mn}_{nm}"] = bh_q(res[f"p_{mn}_{nm}"].values)

    # conditional CI
    rng = np.random.default_rng(args.seed + 777)
    patients = np.unique(patient)
    pat_samples = {int(p): np.where(patient == p)[0] for p in patients}
    boot_idx = [np.concatenate([pat_samples[int(p)] for p in rng.choice(patients, len(patients), replace=True)])
                for _ in range(args.boot)]
    for pi in range(nP):
        for bi, nm in enumerate(names):
            for mi, mn in enumerate(["logloss", "auc"]):
                vals = []
                for idx in boot_idx:
                    per = []
                    for r in range(nR):
                        yb = label[idx]
                        pb = obs_pred_base[pi, bi, r][idx]; pf = obs_pred_full[pi, bi, r][idx]
                        per.append(logloss(yb, pb) - logloss(yb, pf) if mn == "logloss"
                                   else fast_auc(yb, pf) - fast_auc(yb, pb))
                    vals.append(np.nanmean(per))
                res.loc[pi, f"d{mn}_{nm}_ci_lo"] = float(np.nanpercentile(vals, 2.5))
                res.loc[pi, f"d{mn}_{nm}_ci_hi"] = float(np.nanpercentile(vals, 97.5))
                res.loc[pi, f"d{mn}_{nm}_ci_type"] = "conditional_patient_bootstrap_fixed_predictions"

    res.to_csv(os.path.join(args.out_dir, "results", "results_williams_sensitivity.csv"), index=False)
    print(res[["panel_id", "pathway", "dauc_williams_derived_full_mean",
               "q_auc_williams_derived_full", "dauc_williams_derived_full_ci_lo",
               "dauc_williams_derived_full_ci_hi"]].to_string(index=False))
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
