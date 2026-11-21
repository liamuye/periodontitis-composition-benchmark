#!/usr/bin/env python3
"""v3 incremental-value analysis with FIXED random gene panels.

Baselines (each reported separately):
  comp1d   : PC1 of 5 GSE152042 cell-type scores, tested panel's genes removed
             dynamically from the marker set before scoring/PCA;
  compfull : all 5 cell-type scores in the logistic model (same dynamic removal);
  shared   : PC1 of the other 26 panels' gene-score matrix, tested panel's genes
             removed (panel-redundancy baseline, NOT tissue composition).

Null: for every published panel, B fixed random gene panels (identity constant
over all folds and all repeats), matched on the full-cohort gene-level |t| bin.
Each random panel goes through exactly the same outer splits, representative-
probe rule, direction rule, baselines, models and metrics as the published
panel.  The observed statistic is the 20-repeat mean of one fixed panel; the
null statistic is the 20-repeat mean of each fixed random panel.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_base, make_folds, FoldData, comp_features, panel_gene_sets,
                    shared_gene_dirs, scale_train, scale_train_batch, pca1,
                    fit_logistic, predict_logistic, fit_logistic_batch,
                    predict_logistic_batch, logloss, logloss_batch,
                    fast_auc, fast_auc_batch, bh_q, CELLTYPES)

warnings.filterwarnings("ignore")
BASELINES = ["comp1d", "compfull", "shared"]


def load_folds(path, n_repeats, n_sites):
    d = pd.read_csv(path)
    folds = []
    for r in range(n_repeats):
        sub = d[d.repeat == r]
        fr = []
        for f in range(5):
            te = np.sort(sub[sub.fold == f].site.values)
            mask = np.ones(n_sites, dtype=bool)
            mask[te] = False
            tr = np.where(mask)[0]
            fr.append((tr, te))
        folds.append(fr)
    return folds


def shared_features(fd, panel, panels):
    pairs = shared_gene_dirs(panel, panels)
    cols, dirs = [], []
    for g, d in pairs:
        pos = fd.gene_pos.get(g)
        if pos is None:
            continue
        if d == 0:
            d = 1 if fd.gene_t[g] >= 0 else -1
        cols.append(pos)
        dirs.append(d)
    if len(cols) < 3:
        return None
    M = fd.Z[:, cols].astype(np.float64) * np.asarray(dirs, dtype=float)[None, :]
    pc_tr, pc_te, var, load = pca1(M[fd.tr], M[fd.te])
    if np.mean(load) < 0:
        pc_tr, pc_te, load = -pc_tr, -pc_te, -load
    return {"pc_tr": pc_tr, "pc_te": pc_te, "var": var, "load": load, "k": len(cols)}


def null_score_matrix(fd, genes_per_null):
    rows, cols, vals = [], [], []
    missing = 0
    for b, genes in enumerate(genes_per_null):
        pos = []
        for g in genes:
            p = fd.gene_pos.get(g)
            if p is None:
                missing += 1
            else:
                pos.append(p)
        k = len(pos)
        if k == 0:
            continue
        sgn = np.array([1.0 if fd.gene_t[fd.genes[p]] >= 0 else -1.0 for p in pos])
        for p, s in zip(pos, sgn):
            rows.append(p); cols.append(b); vals.append(s / k)
    B = len(genes_per_null)
    W = csr_matrix((np.asarray(vals, dtype=float), (rows, cols)),
                   shape=(len(fd.genes), B))
    S = np.asarray(fd.Z @ W)  # n x B
    return S, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--max-panels", type=int, default=27)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--save-per-repeat", action="store_true")
    ap.add_argument("--no-removal", action="store_true",
                    help="marker-overlap sensitivity: keep the FULL marker set")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    X, probe_ids, patient, label, gsm, title, panels, g2p = load_base(args.data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    panels = panels[: args.max_panels]
    nP = len(panels)
    nS = len(label)
    nR = args.n_repeats
    nB = args.n_null

    markers = pd.read_csv(os.path.join(args.out_dir, "results", "markers_celltype_top_v3.csv"))
    marker_genes = {ct: markers[markers.celltype == ct].sort_values("rank").gene_symbol.tolist()
                    for ct in CELLTYPES}
    sets = panel_gene_sets(panels)
    alias = {p["panel_id"]: {m["symbol"] for m in p["mapped"]} | {m["gene"] for m in p["mapped"]}
             for p in panels}

    ident = pd.read_csv(os.path.join(args.out_dir, "results", "null_panels_identity.csv.gz"))
    ident = ident[ident.panel_id.isin([p["panel_id"] for p in panels])]
    null_genes = {}
    for p in panels:
        sub = ident[ident.panel_id == p["panel_id"]]
        by = sub.sort_values(["null_id", "position"]).groupby("null_id")["null_gene"].apply(list)
        null_genes[p["panel_id"]] = [by[k] for k in sorted(by.index)][:nB]
        assert len(null_genes[p["panel_id"]]) == nB, (p["panel_id"], len(null_genes[p["panel_id"]]))

    folds = load_folds(os.path.join(args.out_dir, "results", f"folds_seed{args.seed}.csv"),
                       nR, nS)
    genes_all = list(g2p.keys())

    obs_delta = np.full((nP, 3, 2, nR), np.nan, dtype=np.float64)
    obs_base = np.full((nP, 3, 2, nR), np.nan, dtype=np.float64)
    obs_pred_base = np.full((nP, 3, nR, nS), np.nan, dtype=np.float32)
    obs_pred_full = np.full((nP, 3, nR, nS), np.nan, dtype=np.float32)
    null_delta = np.full((nP, 3, 2, nB, nR), np.nan, dtype=np.float32)
    k_used = np.full((nP, nR, 5), np.nan)
    base_var = np.full((nP, 3, nR), np.nan)
    corr_panel = np.full((nP, 3, nR), np.nan)
    null_missing = np.zeros(nP, dtype=int)
    n_null_ok = np.zeros((nP, 3, 2), dtype=int)

    t0 = time.time()
    for r in range(nR):
        pred_null_rep = np.full((nP, 3, nS, nB), np.nan, dtype=np.float32)
        pred_base_rep = np.full((nP, 3, nS), np.nan, dtype=np.float64)
        pred_full_rep = np.full((nP, 3, nS), np.nan, dtype=np.float64)
        for f, (tr, te) in enumerate(folds[r]):
            fd = FoldData(X, probe_ids, probe_idx, g2p, genes_all, label, tr, te)
            y_tr, y_te = label[tr], label[te]
            for pi, panel in enumerate(panels):
                comp = comp_features(fd, marker_genes,
                                     set() if args.no_removal else alias[panel["panel_id"]])
                shared = shared_features(fd, panel, panels)
                pairs = [(m["symbol"], m.get("direction_published", 0)) for m in panel["mapped"]]
                obs_s, k, _, _ = fd.score(pairs)
                if obs_s is None:
                    continue
                k_used[pi, r, f] = k

                designs = {}
                b1_tr, b1_te, _, _ = scale_train(comp["pc1_tr"], comp["pc1_te"])
                designs["comp1d"] = (b1_tr[:, None], b1_te[:, None])
                designs["compfull"] = (comp["full_tr"], comp["full_te"])
                if shared is not None:
                    bs_tr, bs_te, _, _ = scale_train(shared["pc_tr"], shared["pc_te"])
                    designs["shared"] = (bs_tr[:, None], bs_te[:, None])

                s_tr, s_te, _, _ = scale_train(obs_s[tr], obs_s[te])
                for bi, bname in enumerate(BASELINES):
                    if bname not in designs:
                        continue
                    Btr, Bte = designs[bname]
                    pred_base_rep[pi, bi, te] = predict_logistic(fit_logistic(Btr, y_tr), Bte)
                    pred_full_rep[pi, bi, te] = predict_logistic(
                        fit_logistic(np.column_stack([Btr, s_tr]), y_tr),
                        np.column_stack([Bte, s_te]))
                    base_var[pi, bi, r] = comp["var"] if bname != "shared" else shared["var"]
                    corr_panel[pi, bi, r] = np.corrcoef(
                        obs_s[tr], comp["pc1_tr"] if bname != "shared" else shared["pc_tr"])[0, 1]

                # null panels: fixed gene identities, fold-specific probes/directions
                S, miss = null_score_matrix(fd, null_genes[panel["panel_id"]])
                null_missing[pi] += miss
                S_tr, S_te, _, _ = scale_train_batch(S[tr], S[te])
                for bi, bname in enumerate(BASELINES):
                    if bname not in designs:
                        continue
                    Btr, Bte = designs[bname]
                    coef = fit_logistic_batch(Btr, S_tr, y_tr)
                    pred_null_rep[pi, bi, te, :] = predict_logistic_batch(coef, Bte, S_te)

        y = label
        for pi, panel in enumerate(panels):
            for bi, bname in enumerate(BASELINES):
                base = pred_base_rep[pi, bi]
                full = pred_full_rep[pi, bi]
                if np.isnan(base).any() or np.isnan(full).any():
                    continue
                obs_pred_base[pi, bi, r] = base
                obs_pred_full[pi, bi, r] = full
                obs_base[pi, bi, 0, r] = logloss(y, base)
                obs_base[pi, bi, 1, r] = fast_auc(y, base)
                obs_delta[pi, bi, 0, r] = logloss(y, base) - logloss(y, full)
                obs_delta[pi, bi, 1, r] = fast_auc(y, full) - fast_auc(y, base)
                P = pred_null_rep[pi, bi]
                llb = logloss(y, base)
                aub = fast_auc(y, base)
                ok = ~np.isnan(P[0])
                if ok.any():
                    null_delta[pi, bi, 0, ok, r] = llb - logloss_batch(y, P[:, ok])
                    null_delta[pi, bi, 1, ok, r] = fast_auc_batch(y, P[:, ok]) - aub
                    n_null_ok[pi, bi, 0] = max(n_null_ok[pi, bi, 0], int(ok.sum()))
                    n_null_ok[pi, bi, 1] = max(n_null_ok[pi, bi, 1], int(ok.sum()))
        del pred_null_rep
        print(f"[{args.tag}] repeat {r+1}/{nR} elapsed {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------ summaries
    rows = []
    null_summary = []
    null_panel_mean = np.full((nP, 3, 2, nB), np.nan, dtype=np.float32)
    for bi in range(3):
        for mi in range(2):
            null_panel_mean[:, bi, mi, :] = np.nanmean(null_delta[:, bi, mi, :, :], axis=2)

    for pi, panel in enumerate(panels):
        row = {"panel_id": panel["panel_id"], "pmid": panel["pmid"], "doi": panel.get("doi", ""),
               "year": panel.get("year", ""), "journal": panel.get("journal", ""),
               "pathway": panel["pathway"], "n_genes": panel["n_genes"],
               "n_mapped": panel["n_mapped"], "genes": ";".join(panel["genes"])}
        row["k_mean"] = float(np.nanmean(k_used[pi]))
        row["k_min"] = float(np.nanmin(k_used[pi]))
        row["k_max"] = float(np.nanmax(k_used[pi]))
        for bi, bname in enumerate(BASELINES):
            row[f"base_{bname}_logloss_mean"] = float(np.nanmean(obs_base[pi, bi, 0]))
            row[f"base_{bname}_auc_mean"] = float(np.nanmean(obs_base[pi, bi, 1]))
            row[f"base_{bname}_pc1_var"] = float(np.nanmean(base_var[pi, bi]))
            row[f"corr_panel_{bname}"] = float(np.nanmean(corr_panel[pi, bi]))
            for mi, mname in enumerate(["logloss", "auc"]):
                vals = obs_delta[pi, bi, mi]
                nm = null_panel_mean[pi, bi, mi]
                nm = nm[np.isfinite(nm)]
                row[f"d{mname}_{bname}_mean"] = float(np.nanmean(vals))
                row[f"d{mname}_{bname}_sd"] = float(np.nanstd(vals, ddof=1))
                row[f"d{mname}_{bname}_min"] = float(np.nanmin(vals))
                row[f"d{mname}_{bname}_max"] = float(np.nanmax(vals))
                row[f"p_{mname}_{bname}"] = float((1 + np.sum(nm >= np.nanmean(vals))) / (len(nm) + 1))
                row[f"null95_{mname}_{bname}"] = float(np.nanpercentile(nm, 95))
                row[f"null_median_{mname}_{bname}"] = float(np.nanmedian(nm))
                row[f"null_sd_{mname}_{bname}"] = float(np.nanstd(nm, ddof=1))
                row[f"n_null_{mname}_{bname}"] = int(len(nm))
                # within-panel across-repeat SD (diagnostic of repeat noise)
                row[f"null_within_sd_{mname}_{bname}"] = float(
                    np.nanmean(np.nanstd(null_delta[pi, bi, mi], axis=1, ddof=1)))
                null_summary.append({"panel_id": panel["panel_id"], "pathway": panel["pathway"],
                                     "baseline": bname, "metric": mname,
                                     "n_null": int(len(nm)), "null_mean": float(np.nanmean(nm)),
                                     "null_sd": float(np.nanstd(nm, ddof=1)),
                                     "null_median": float(np.nanmedian(nm)),
                                     "null95": float(np.nanpercentile(nm, 95)),
                                     "null99": float(np.nanpercentile(nm, 99)),
                                     "null_min": float(np.nanmin(nm)),
                                     "null_max": float(np.nanmax(nm)),
                                     "null_within_sd": float(
                                         np.nanmean(np.nanstd(null_delta[pi, bi, mi], axis=1, ddof=1)))})
        row["n_null"] = int(n_null_ok[pi].max())
        row["n_null_missing_genes"] = int(null_missing[pi])
        row["seed"] = args.seed
        row["n_repeats"] = nR
        row["n_boot"] = args.boot
        row["lambda"] = 1.0
        rows.append(row)

    res = pd.DataFrame(rows)
    for bname in BASELINES:
        for mname in ["logloss", "auc"]:
            res[f"q_{mname}_{bname}"] = bh_q(res[f"p_{mname}_{bname}"].values)
    pcols = [f"p_{m}_{b}" for b in BASELINES for m in ["logloss", "auc"]]
    qall = bh_q(res[pcols].values.ravel()).reshape(len(res), len(pcols))
    for k, col in enumerate(pcols):
        res[col.replace("p_", "q_all_")] = qall[:, k]

    # ------------------------------------------------- conditional bootstrap
    rng = np.random.default_rng(args.seed + 555)
    patients = np.unique(patient)
    pat_samples = {int(p): np.where(patient == p)[0] for p in patients}
    boot_idx = []
    for b in range(args.boot):
        bp = rng.choice(patients, size=len(patients), replace=True)
        boot_idx.append(np.concatenate([pat_samples[int(p)] for p in bp]))

    for pi in range(nP):
        for bi, bname in enumerate(BASELINES):
            for mi, mname in enumerate(["logloss", "auc"]):
                vals = []
                base_all = obs_pred_base[pi, bi]
                full_all = obs_pred_full[pi, bi]
                for idx in boot_idx:
                    per = []
                    for r in range(nR):
                        yb = label[idx]
                        pb = base_all[r][idx]; pf = full_all[r][idx]
                        if mname == "logloss":
                            per.append(logloss(yb, pb) - logloss(yb, pf))
                        else:
                            per.append(fast_auc(yb, pf) - fast_auc(yb, pb))
                    vals.append(float(np.nanmean(per)))
                res.loc[pi, f"d{mname}_{bname}_ci_lo"] = float(np.nanpercentile(vals, 2.5))
                res.loc[pi, f"d{mname}_{bname}_ci_hi"] = float(np.nanpercentile(vals, 97.5))
                res.loc[pi, f"d{mname}_{bname}_ci_type"] = "conditional_patient_bootstrap_fixed_predictions"

    res.to_csv(os.path.join(args.out_dir, "results", f"results_per_panel_{args.tag}.csv"), index=False)
    pd.DataFrame(null_summary).to_csv(
        os.path.join(args.out_dir, "results", f"null_summary_{args.tag}.csv"), index=False)

    # null panel means (long)
    np.savez_compressed(os.path.join(args.out_dir, "results", f"null_panel_means_{args.tag}.npz"),
                        panel_id=np.array([p["panel_id"] for p in panels]),
                        baseline=np.array(BASELINES),
                        metric=np.array(["logloss", "auc"]),
                        null_id=np.arange(1, nB + 1),
                        mean=null_panel_mean)
    if args.save_per_repeat:
        # long table: panel, baseline, metric, null_id, repeat, value
        recs = []
        for pi, panel in enumerate(panels):
            for bi, bname in enumerate(BASELINES):
                for mi, mname in enumerate(["logloss", "auc"]):
                    arr = null_delta[pi, bi, mi]  # B x R
                    for b in range(nB):
                        for r in range(nR):
                            recs.append((panel["panel_id"], bname, mname, b + 1, r + 1,
                                         float(arr[b, r])))
        pd.DataFrame(recs, columns=["panel_id", "baseline", "metric", "null_id", "repeat", "value"]) \
            .to_csv(os.path.join(args.out_dir, "results", f"null_per_repeat_{args.tag}.csv.gz"),
                    index=False, compression="gzip")

    np.savez_compressed(os.path.join(args.out_dir, "results", f"obs_predictions_{args.tag}.npz"),
                        y=label, patient=patient,
                        base=obs_pred_base, full=obs_pred_full,
                        obs_delta=obs_delta, obs_base=obs_base,
                        baseline=np.array(BASELINES))
    np.savez_compressed(os.path.join(args.out_dir, "results", f"null_delta_{args.tag}.npz"),
                        delta=null_delta, panel_id=np.array([p["panel_id"] for p in panels]),
                        baseline=np.array(BASELINES), metric=np.array(["logloss", "auc"]))

    print(res[["panel_id", "pathway", "k_mean",
               "dlogloss_comp1d_mean", "q_logloss_comp1d",
               "dlogloss_compfull_mean", "q_logloss_compfull",
               "dauc_shared_mean", "q_auc_shared"]].to_string(index=False))
    print(f"saved {args.tag}; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
