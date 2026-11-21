#!/usr/bin/env python3
"""Batch & renormalization sensitivity, v2 (post-review fix).

Two versions, 42 published panels each, CV seed 42, first 1,000 fixed random
identities per panel:

  (a) RMA : raw CEL files re-normalised with affy::rma(), aligned to the same
            307 sites / 119 patients as the main analysis; main-analysis folds
            reused (site-for-site identical).
  (b) NO2008 : series matrix with all 33 sites whose scan year is 2008 removed.

Fix vs v1 (audit review, 2026-11-01):
  * `shared` now uses the main-analysis implementation: shared_gene_dirs over
    the FIXED 27 discovery panels (which also excludes the tested panel's
    genes), direction-aligned gene-score matrix, PC1, sign fixed so mean
    loading > 0.  v1 used a plain mean score, which is not comparable.
  * BH correction is applied within the main-analysis families:
    discovery (panels 1-27) and confirmation (panels 101-115) separately,
    each baseline x metric.  v1 pooled all 42 panels, which changed q.
  * Both versions keep B = 1,000 identities.  Main analysis: discovery
    B = 1,000, confirmation B = 20,000.

No re-adjudication of criteria 1-4; positivity here is descriptive q < 0.05.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = "."
V3S = f"{ROOT}/exploration_v3/scripts"
sys.path.insert(0, V3S)
from common import (load_base, make_folds, FoldData, comp_features, shared_gene_dirs,
                    scale_train, scale_train_batch, pca1, fit_logistic, predict_logistic,
                    fit_logistic_batch, predict_logistic_batch, logloss, logloss_batch,
                    fast_auc, fast_auc_batch, bh_q, CELLTYPES)

warnings.filterwarnings("ignore")
BASELINES = ["comp1d", "compfull", "shared"]
METRICS = ["logloss", "auc"]


def load_panels(data_dir):
    """Normalized panel dicts, identical to the ones the main analysis used.

    27 discovery panels come from the data dir's panels.json (written by the
    v3 preparation step), 15 confirmation panels from the frozen confirmation
    JSON.  Both carry `mapped` with symbol / direction_published, so the
    shared-axis reference and the tested-panel signs are exactly the main
    analysis's.
    """
    out = [dict(p) for p in json.load(open(os.path.join(data_dir, "panels.json")))]
    for p in json.load(open(f"{ROOT}/confirmation_v1/01_panels_confirmation.json")):
        q = dict(p)
        q["panel_id"] = int(p["panel_id"])
        out.append(q)
    return out


def load_identity(path, panel_ids, nB, sep=','):
    df = pd.read_csv(path, sep=sep)
    df = df[df.panel_id.isin(panel_ids)]
    out = {}
    for pid, sub in df.groupby("panel_id"):
        by = sub.sort_values(["null_id", "position"]).groupby("null_id")["null_gene"].apply(list)
        arr = [by[k] for k in sorted(by.index)][:nB]
        out[int(pid)] = arr
    return out


def null_score_matrix(fd, genes_per_null):
    rows, cols, vals = [], [], []
    missing = 0
    inv = {v: k for k, v in fd.gene_pos.items()}
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
        sgn = np.array([1.0 if fd.gene_t[inv[p]] >= 0 else -1.0 for p in pos])
        for p, s in zip(pos, sgn):
            rows.append(p); cols.append(b); vals.append(s / k)
    W = csr_matrix((np.asarray(vals, float), (rows, cols)), shape=(fd.Z.shape[1], len(genes_per_null)))
    return np.asarray(fd.Z @ W), missing


def shared_features_main(fd, panel, reference_panels):
    """Identical to the main analysis (v3 discovery / confirmation 03_observed)."""
    pairs = shared_gene_dirs(panel, reference_panels)
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
    return {"pc_tr": pc_tr, "pc_te": pc_te, "var": var, "k": len(cols)}


def load_folds(path, nR, nS):
    d = pd.read_csv(path)
    reps = []
    for r in range(nR):
        sub = d[d.repeat == r]
        fr = []
        for f in range(5):
            te = np.sort(sub[sub.fold == f].site.values)
            mask = np.ones(nS, bool); mask[te] = False
            fr.append((np.where(mask)[0], te))
        reps.append(fr)
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--folds", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    X, probe_ids, patient, label, gsm, title, _p, g2p = load_base(args.data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    panels = load_panels(args.data_dir)
    # fixed shared-axis reference: the 27 discovery panels, in main-analysis order
    reference_panels = [p for p in panels if p["panel_id"] <= 27]
    nP, nS, nR, nB = len(panels), len(label), args.n_repeats, args.n_null
    print(f"[{args.tag}] X={X.shape} panels={nP} B={nB} ref_panels={len(reference_panels)}",
          flush=True)

    mk = pd.read_csv(f"{ROOT}/exploration_v3/results/markers_celltype_top_v3.csv")
    marker_genes = {ct: mk[mk.celltype == ct].sort_values("rank").gene_symbol.tolist() for ct in CELLTYPES}

    ident = {}
    ident.update(load_identity(f"{ROOT}/exploration_v3/results/null_panels_identity.csv.gz",
                               set(range(1, 28)), nB))
    ident.update(load_identity(f"{ROOT}/confirmation_v1/results/null_panels_identity.csv.gz",
                               set(range(101, 116)), nB, sep="\t"))
    short = 0
    for p in panels:
        arr = ident.get(p["panel_id"], [])
        if len(arr) < nB:
            short += 1
    if short: print(f"[warn] {short} panels have fewer than {nB} identities")

    if args.folds:
        folds = load_folds(args.folds, nR, nS)
    else:
        folds = make_folds(patient, label, args.seed, nR, 5)
        rows = []
        for r, fl in enumerate(folds):
            for f, (tr, te) in enumerate(fl):
                for i in te:
                    rows.append({"site": int(i), "patient": int(patient[i]), "repeat": r, "fold": f})
        pd.DataFrame(rows).to_csv(f"{args.out_dir}/results/folds_seed{args.seed}.csv", index=False)

    alias = {p["panel_id"]: {m["symbol"] for m in p["mapped"]} for p in panels}
    genes_all = list(g2p.keys())

    obs_delta = np.full((nP, 3, 2, nR), np.nan)
    null_delta = np.full((nP, 3, 2, nB, nR), np.nan)
    shared_k = np.full((nP, nR), np.nan)
    shared_var = np.full((nP, nR), np.nan)
    t0 = time.time()
    for r in range(nR):
        pred_null = np.full((nP, 3, nS, nB), np.nan, dtype=np.float32)
        pred_base = np.full((nP, 3, nS), np.nan)
        pred_full = np.full((nP, 3, nS), np.nan)
        for f, (tr, te) in enumerate(folds[r]):
            fd = FoldData(X, probe_ids, probe_idx, g2p, genes_all, label, tr, te)
            y_tr, y_te = label[tr], label[te]
            for pi, panel in enumerate(panels):
                comp = comp_features(fd, marker_genes, alias[panel["panel_id"]])
                pairs = [(m["symbol"], m["direction_published"]) for m in panel["mapped"]]
                obs_s, k, _, _ = fd.score(pairs)
                if obs_s is None:
                    continue
                designs = {}
                b1_tr, b1_te, _, _ = scale_train(comp["pc1_tr"], comp["pc1_te"])
                designs["comp1d"] = (b1_tr[:, None], b1_te[:, None])
                designs["compfull"] = (comp["full_tr"], comp["full_te"])
                sh = shared_features_main(fd, panel, reference_panels)
                if sh is not None:
                    bs_tr, bs_te, _, _ = scale_train(sh["pc_tr"], sh["pc_te"])
                    designs["shared"] = (bs_tr[:, None], bs_te[:, None])
                    shared_k[pi, r] = sh["k"]
                    shared_var[pi, r] = sh["var"]
                s_tr, s_te, _, _ = scale_train(obs_s[tr], obs_s[te])
                for bi, bn in enumerate(BASELINES):
                    if bn not in designs: continue
                    Btr, Bte = designs[bn]
                    pred_base[pi, bi, te] = predict_logistic(fit_logistic(Btr, y_tr), Bte)
                    pred_full[pi, bi, te] = predict_logistic(
                        fit_logistic(np.column_stack([Btr, s_tr]), y_tr),
                        np.column_stack([Bte, s_te]))
                S, _miss = null_score_matrix(fd, ident[panel["panel_id"]])
                S_tr, S_te, _, _ = scale_train_batch(S[tr], S[te])
                for bi, bn in enumerate(BASELINES):
                    if bn not in designs: continue
                    Btr, Bte = designs[bn]
                    coef = fit_logistic_batch(Btr, S_tr, y_tr)
                    pred_null[pi, bi, te, :] = predict_logistic_batch(coef, Bte, S_te)
        y = label
        for pi in range(nP):
            for bi in range(3):
                base, full = pred_base[pi, bi], pred_full[pi, bi]
                if np.isnan(base).any() or np.isnan(full).any(): continue
                llb, aub = logloss(y, base), fast_auc(y, base)
                obs_delta[pi, bi, 0, r] = llb - logloss(y, full)
                obs_delta[pi, bi, 1, r] = fast_auc(y, full) - aub
                P = pred_null[pi, bi]
                ok = ~np.isnan(P[0])
                if ok.any():
                    null_delta[pi, bi, 0, ok, r] = llb - logloss_batch(y, P[:, ok])
                    null_delta[pi, bi, 1, ok, r] = fast_auc_batch(y, P[:, ok]) - aub
        del pred_null
        print(f"[{args.tag}] repeat {r+1}/{nR} elapsed {time.time()-t0:.0f}s", flush=True)

    null_mean = np.nanmean(null_delta, axis=4)  # panel, base, metric, B
    rows = []
    for pi, panel in enumerate(panels):
        for bi, bn in enumerate(BASELINES):
            for mi, mn in enumerate(METRICS):
                obs = float(np.nanmean(obs_delta[pi, bi, mi]))
                nb = null_mean[pi, bi, mi]
                nb = nb[np.isfinite(nb)]
                if len(nb) == 0: continue
                p = (1 + np.sum(nb >= obs)) / (1 + len(nb))
                rows.append(dict(panel_id=panel["panel_id"], baseline=bn, metric=mn,
                                 observed_mean=obs, null_mean=float(nb.mean()),
                                 null95=float(np.percentile(nb, 95)), p=p, n_null=len(nb),
                                 shared_k=float(np.nanmean(shared_k[pi])) if bn == "shared" else np.nan,
                                 shared_pc1_var=float(np.nanmean(shared_var[pi])) if bn == "shared" else np.nan))
    res = pd.DataFrame(rows)
    res["cohort"] = np.where(res.panel_id <= 27, "discovery", "confirmation")
    res["q"] = np.nan
    for (co, bn, mn), g in res.groupby(["cohort", "baseline", "metric"]):
        res.loc[g.index, "q"] = bh_q(g.p.values)
    res = res[["panel_id", "cohort", "baseline", "metric", "observed_mean", "null_mean",
               "null95", "p", "q", "n_null", "shared_k", "shared_pc1_var"]]
    res.to_csv(f"{args.out_dir}/results/results_per_test_{args.tag}.csv", index=False)
    print(res[res.q < 0.05][["panel_id", "cohort", "baseline", "metric", "observed_mean", "p", "q"]]
          .to_string(index=False))
    print(f"[{args.tag}] done {time.time()-t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
