#!/usr/bin/env python3
"""Spike-in sensitivity v2 (audit 2026-11-02): add signal to random panels.

New score = random-panel score + w * orthogonalised synthetic signal, re-standardised
inside the training fold.  Panel 113 (k = 5, confirmation): identities 1-1000 are the
null, identities 1001-1200 are the 200 spike bases (shared across all w).  The synthetic
signal keeps patient structure with the empirical median ICC of the 200 bases.

p = (1 + #{null >= delta}) / 1001.  Detection at p < 0.05, plus the approximate
family thresholds 0.05/27 and 0.05/15 (reported separately).
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd

ROOT = "."
sys.path.insert(0, f"{ROOT}/exploration_v3/scripts")
sys.path.insert(0, f"{ROOT}/scripts/03_sensitivities_and_calibration")
from common import (load_base, FoldData, comp_features, scale_train, scale_train_batch,
                    fit_logistic, predict_logistic, fit_logistic_batch, predict_logistic_batch,
                    logloss, logloss_batch, fast_auc, fast_auc_batch, welch_t, CELLTYPES)
from run_batch_rma_v2 import null_score_matrix

warnings.filterwarnings("ignore")
DATA = "./data"
OUT = f"{ROOT}/results/spikein"
os.makedirs(OUT, exist_ok=True)
BASELINES = ["comp1d", "compFull"]
METRICS = ["logloss", "auc"]
WGRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
EXTEND_IF = 2.0
NREP, NFOLD = 20, 5
EPS = 1e-12


def load_folds(path, nR, nS):
    d = pd.read_csv(path)
    reps = []
    for r in range(nR):
        sub = d[d.repeat == r]
        fr = []
        for f in range(NFOLD):
            te = np.sort(sub[sub.fold == f].site.values)
            mask = np.ones(nS, bool); mask[te] = False
            fr.append((np.where(mask)[0], te))
        reps.append(fr)
    return reps


def icc_oneway(scores, patient):
    df = pd.DataFrame({"s": scores, "p": patient})
    m = df.groupby("p").s.agg(["mean", "count", "var"])
    k, a = len(df), len(m)
    msb = ((m["mean"] - df.s.mean()) ** 2 * m["count"]).sum() / (a - 1)
    msw = (m["var"] * (m["count"] - 1)).sum() / (k - a)
    m0 = (k - (m["count"] ** 2).sum() / k) / (a - 1)
    return float((msb - msw) / (msb + (m0 - 1) * msw))


def main():
    X, probe_ids, patient, label, gsm, title, _p, g2p = load_base(DATA)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    panels = json.load(open(f"{DATA}/panels.json"))
    panels += json.load(open(f"{ROOT}/confirmation_v1/01_panels_confirmation.json"))
    panels = [dict(p, panel_id=int(p["panel_id"])) for p in panels]
    p113 = next(p for p in panels if p["panel_id"] == 113)
    alias = {m["symbol"] for m in p113["mapped"]}
    marker_genes = {ct: pd.read_csv(f"{ROOT}/exploration_v3/results/markers_celltype_top_v3.csv")
                    .query("celltype == @ct").sort_values("rank").gene_symbol.tolist() for ct in CELLTYPES}
    nS = len(label)
    folds = load_folds(f"{ROOT}/exploration_v3/results/folds_seed42.csv", NREP, nS)

    ident = pd.read_csv(f"{ROOT}/confirmation_v1/results/null_panels_identity.csv.gz", sep="\t")
    ident = ident[ident.panel_id == 113].sort_values(["null_id", "position"])
    by = ident.groupby("null_id")["null_gene"].apply(list)
    null_ids = [by[k] for k in range(1, 1001)]
    base_ids = [by[k] for k in range(1001, 1201)]
    nB = len(base_ids)
    print(f"[spike] sites={nS} nulls={len(null_ids)} bases={nB} w={WGRID}", flush=True)

    # ---- empirical ICC of the 200 bases (fold-free whole-cohort scores) ----
    t = welch_t(X, label)
    rep = {}
    for g, ps in g2p.items():
        best, key = None, None
        for p in ps:
            j = probe_idx.get(p)
            if j is None:
                continue
            kk = (-abs(t[j]), p)
            if key is None or kk < key:
                key, best = kk, j
        if best is not None:
            rep[g] = best
    Zcache = {}
    def zgene(g):
        if g not in Zcache:
            j = rep.get(g)
            if j is None:
                Zcache[g] = None
            else:
                v = X[:, j].astype(float)
                Zcache[g] = (v - v.mean()) / v.std(ddof=1)
        return Zcache[g]
    iccs = []
    for genes in base_ids:
        cols, signs = [], []
        for g in genes:
            z = zgene(g)
            if z is None:
                continue
            cols.append(z); signs.append(1.0 if t[rep[g]] >= 0 else -1.0)
        if cols:
            iccs.append(icc_oneway(np.mean(np.column_stack(cols) * np.array(signs), axis=1), patient))
    ICC = float(np.median(iccs))
    print(f"[spike] ICC median={ICC:.4f} n={len(iccs)}", flush=True)

    # ---- synthetic signal per base: d*y + patient effect + site noise, fixed over folds ----
    from scipy.stats import norm
    yc = (label - label.mean()) / label.std(ddof=1)
    gap = yc[label == 1].mean() - yc[label == 0].mean()
    d_sig = np.sqrt(2) * norm.ppf(0.80) / gap          # target standalone AUC 0.80
    rng = np.random.default_rng(20261102)
    pts = np.unique(patient)
    pidx = {p: i for i, p in enumerate(pts)}
    pe = rng.normal(0, np.sqrt(max(ICC, 0)), size=(len(pts), nB))
    noise = rng.normal(0, np.sqrt(max(1 - ICC, 0)), size=(nS, nB))
    Z = d_sig * yc[:, None] + pe[[pidx[p] for p in patient]] + noise
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
    # achieved standalone AUC of the signal (before orthogonalisation)
    from common import fast_auc as _fa
    auc_z = float(np.mean([_fa(label, Z[:, j]) for j in range(nB)]))
    print(f"[spike] d_sig={d_sig:.4f} standalone AUC={auc_z:.3f}", flush=True)

    nW = len(WGRID)
    pred_base = np.full((NREP, nS, 2), np.nan)
    pred_comb = np.full((NREP, nS, nB, 2, nW), np.nan, dtype=np.float32)
    pred_null = np.full((NREP, nS, 1000, 2), np.nan, dtype=np.float32)

    t0 = time.time()
    for r, fl in enumerate(folds):
        for f, (tr, te) in enumerate(fl):
            fd = FoldData(X, probe_ids, probe_idx, g2p, list(g2p.keys()), label, tr, te)
            comp = comp_features(fd, marker_genes, alias)
            y_tr = label[tr]
            b1_tr, b1_te, _, _ = scale_train(comp["pc1_tr"], comp["pc1_te"])
            designs = {"comp1d": (b1_tr[:, None], b1_te[:, None]),
                       "compFull": (comp["full_tr"], comp["full_te"])}
            S = null_score_matrix(fd, base_ids)[0]
            S_tr, S_te, _, _ = scale_train_batch(S[tr], S[te])
            for bi, bn in enumerate(BASELINES):
                Btr, Bte = designs[bn]
                pred_base[r, te, bi] = predict_logistic(fit_logistic(Btr, y_tr), Bte)
                A_tr = np.column_stack([np.ones(len(tr)), Btr])
                A_te = np.column_stack([np.ones(len(te)), Bte])
                beta = np.linalg.lstsq(A_tr, Z[tr], rcond=None)[0]
                Zr_tr = Z[tr] - A_tr @ beta
                Zr_te = Z[te] - A_te @ beta
                Zr_tr, Zr_te, _, _ = scale_train_batch(Zr_tr, Zr_te)
                for wi, w in enumerate(WGRID):
                    C_tr = S_tr + w * Zr_tr
                    C_te = S_te + w * Zr_te
                    C_tr, C_te, _, _ = scale_train_batch(C_tr, C_te)
                    coef = fit_logistic_batch(Btr, C_tr, y_tr)
                    pred_comb[r, te, :, bi, wi] = predict_logistic_batch(coef, Bte, C_te)
            Ns = null_score_matrix(fd, null_ids)[0]
            Ns_tr, Ns_te, _, _ = scale_train_batch(Ns[tr], Ns[te])
            for bi, bn in enumerate(BASELINES):
                Btr, Bte = designs[bn]
                coef = fit_logistic_batch(Btr, Ns_tr, y_tr)
                pred_null[r, te, :, bi] = predict_logistic_batch(coef, Bte, Ns_te)
        print(f"[spike] repeat {r+1}/{NREP} {time.time()-t0:.0f}s", flush=True)

    y = label
    d_comb = np.full((NREP, nB, 2, nW, 2), np.nan)  # rep, base, baseline, w, metric
    d_null = np.full((NREP, 1000, 2, 2), np.nan)
    for r in range(NREP):
        for bi in range(2):
            yb = pred_base[r, :, bi]
            llb, aub = logloss(y, yb), fast_auc(y, yb)
            P = pred_comb[r, :, :, bi, :]
            for wi in range(nW):
                d_comb[r, :, bi, wi, 0] = llb - logloss_batch(y, P[:, :, wi])
                d_comb[r, :, bi, wi, 1] = fast_auc_batch(y, P[:, :, wi]) - aub
            Nn = pred_null[r, :, :, bi]
            d_null[r, :, bi, 0] = llb - logloss_batch(y, Nn)
            d_null[r, :, bi, 1] = fast_auc_batch(y, Nn) - aub
    d_comb_m = np.nanmean(d_comb, axis=0)   # base, baseline, w, metric
    d_null_m = np.nanmean(d_null, axis=0)   # null, baseline, metric

    rows = []
    for bi, bn in enumerate(BASELINES):
        for mi, mn in enumerate(METRICS):
            null = d_null_m[:, bi, mi]
            for wi, w in enumerate(WGRID):
                obs = d_comb_m[:, bi, wi, mi]
                p = np.array([(1 + np.sum(null >= o)) / (1 + len(null)) for o in obs])
                base0 = d_comb_m[:, bi, 0, mi]
                inc = obs - base0
                rows.append(dict(baseline=bn, metric=mn, w=w,
                                 delta_median=float(np.median(obs)),
                                 delta_q1=float(np.percentile(obs, 25)),
                                 delta_q3=float(np.percentile(obs, 75)),
                                 paired_inc_median=float(np.median(inc)),
                                 det_p05=float(np.mean(p < 0.05)),
                                 det_fam27=float(np.mean(p <= 0.05 / 27)),
                                 det_fam15=float(np.mean(p <= 0.05 / 15)),
                                 null95=float(np.percentile(null, 95))))
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/results_spikein.csv", index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    json.dump(dict(icc=ICC, n_bases=nB, n_null=1000, wgrid=WGRID), open(f"{OUT}/spikein_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
