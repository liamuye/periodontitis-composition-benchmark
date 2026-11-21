#!/usr/bin/env python3
"""Spike-in, family-corrected column (audit 2026-11-02 follow-up).

Null: panel 113's archived 20,000 random-panel mean deltas
(`confirmation_v1/results/chunks_primary_fixed/panel_113_chunk_*.npz`), no rerun.
Observed arm: 200 spike bases (identities 1001-1200) x w grid x three synthetic
signal strengths (standalone AUC 0.80 / 0.90 / 0.95), same folds and ICC.
p = (1 + #{null >= delta}) / 20001; thresholds 0.05, 0.05/15, 0.05/27.
"""
import glob, json, os, sys, time, warnings
import numpy as np
import pandas as pd

ROOT = "."
sys.path.insert(0, f"{ROOT}/exploration_v3/scripts")
sys.path.insert(0, f"{ROOT}/scripts/03_sensitivities_and_calibration")
from common import (load_base, FoldData, comp_features, scale_train, scale_train_batch,
                    fit_logistic, predict_logistic, fit_logistic_batch, predict_logistic_batch,
                    logloss, logloss_batch, fast_auc, fast_auc_batch, welch_t, CELLTYPES)
from run_batch_rma_v2 import null_score_matrix
from scipy.stats import norm

warnings.filterwarnings("ignore")
DATA = "./data"
OUT = f"{ROOT}/results/spikein"
WGRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
SIG_AUC = [0.80, 0.90, 0.95]
NREP, NFOLD = 20, 5
BASELINES = ["comp1d", "compFull"]


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
    panels = json.load(open(f"{DATA}/panels.json")) + \
        json.load(open(f"{ROOT}/confirmation_v1/01_panels_confirmation.json"))
    panels = [dict(p, panel_id=int(p["panel_id"])) for p in panels]
    p113 = next(p for p in panels if p["panel_id"] == 113)
    alias = {m["symbol"] for m in p113["mapped"]}
    mk = pd.read_csv(f"{ROOT}/exploration_v3/results/markers_celltype_top_v3.csv")
    marker_genes = {ct: mk[mk.celltype == ct].sort_values("rank").gene_symbol.tolist() for ct in CELLTYPES}
    nS = len(label)
    folds = load_folds(f"{ROOT}/exploration_v3/results/folds_seed42.csv", NREP, nS)

    ident = pd.read_csv(f"{ROOT}/confirmation_v1/results/null_panels_identity.csv.gz", sep="\t")
    ident = ident[ident.panel_id == 113].sort_values(["null_id", "position"])
    by = ident.groupby("null_id")["null_gene"].apply(list)
    base_ids = [by[k] for k in range(1001, 1201)]
    nB = len(base_ids)

    # archived 20,000 null means: (3 baselines, 2 metrics, 20000)
    fs = sorted(glob.glob(f"{ROOT}/confirmation_v1/results/chunks_primary_fixed/panel_113_chunk_*.npz"))
    M = np.concatenate([np.load(f, allow_pickle=True)["means"] for f in fs], axis=2)
    null = {"comp1d": {"logloss": M[0, 0], "auc": M[0, 1]},
            "compFull": {"logloss": M[1, 0], "auc": M[1, 1]}}
    null_sorted = {b: {m: np.sort(v) for m, v in d.items()} for b, d in null.items()}
    print(f"[family] bases={nB} sig={SIG_AUC} null=20000", flush=True)

    # ICC of the 200 bases
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
    Zc = {}
    def zgene(g):
        if g not in Zc:
            j = rep.get(g)
            Zc[g] = None if j is None else (lambda v: (v - v.mean()) / v.std(ddof=1))(X[:, j].astype(float))
        return Zc[g]
    iccs = []
    for genes in base_ids:
        cols, sgn = [], []
        for g in genes:
            z = zgene(g)
            if z is None:
                continue
            cols.append(z); sgn.append(1.0 if t[rep[g]] >= 0 else -1.0)
        if cols:
            iccs.append(icc_oneway(np.mean(np.column_stack(cols) * np.array(sgn), axis=1), patient))
    ICC = float(np.median(iccs))
    print(f"[family] ICC={ICC:.4f}", flush=True)

    yc = (label - label.mean()) / label.std(ddof=1)
    gap = yc[label == 1].mean() - yc[label == 0].mean()
    rng = np.random.default_rng(20261102)
    pts = np.unique(patient); pidx = {p: i for i, p in enumerate(pts)}
    pe = rng.normal(0, np.sqrt(max(ICC, 0)), size=(len(pts), nB))
    noise = rng.normal(0, np.sqrt(max(1 - ICC, 0)), size=(nS, nB))
    Zs = {}
    for a in SIG_AUC:
        d = np.sqrt(2) * norm.ppf(a) / gap
        Z = d * yc[:, None] + pe[[pidx[p] for p in patient]] + noise
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
        Zs[a] = Z

    nW, nA = len(WGRID), len(SIG_AUC)
    d_comb = np.full((NREP, nB, 2, nA, nW, 2), np.nan)
    t0 = time.time()
    for r, fl in enumerate(folds):
        pred_base = np.full((nS, 2), np.nan)
        pred_comb = np.full((nS, nB, 2, nA, nW), np.nan, dtype=np.float32)
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
                pred_base[te, bi] = predict_logistic(fit_logistic(Btr, y_tr), Bte)
                A_tr = np.column_stack([np.ones(len(tr)), Btr])
                A_te = np.column_stack([np.ones(len(te)), Bte])
                for ai, a in enumerate(SIG_AUC):
                    Zr = Zs[a]
                    beta = np.linalg.lstsq(A_tr, Zr[tr], rcond=None)[0]
                    Zr_tr = Zr[tr] - A_tr @ beta
                    Zr_te = Zr[te] - A_te @ beta
                    Zr_tr, Zr_te, _, _ = scale_train_batch(Zr_tr, Zr_te)
                    for wi, w in enumerate(WGRID):
                        C_tr = S_tr + w * Zr_tr
                        C_te = S_te + w * Zr_te
                        C_tr, C_te, _, _ = scale_train_batch(C_tr, C_te)
                        coef = fit_logistic_batch(Btr, C_tr, y_tr)
                        pred_comb[te, :, bi, ai, wi] = predict_logistic_batch(coef, Bte, C_te)
        for bi in range(2):
            yb = pred_base[:, bi]
            llb, aub = logloss(label, yb), fast_auc(label, yb)
            for ai in range(nA):
                P = pred_comb[:, :, bi, ai, :]
                for wi in range(nW):
                    d_comb[r, :, bi, ai, wi, 0] = llb - logloss_batch(label, P[:, :, wi])
                    d_comb[r, :, bi, ai, wi, 1] = fast_auc_batch(label, P[:, :, wi]) - aub
        print(f"[family] repeat {r+1}/{NREP} {time.time()-t0:.0f}s", flush=True)

    d_m = np.nanmean(d_comb, axis=0)  # base, baseline, sig, w, metric
    rows = []
    for bi, bn in enumerate(BASELINES):
        for mi, mn in enumerate(["logloss", "auc"]):
            ns = null_sorted[bn][mn]
            for ai, a in enumerate(SIG_AUC):
                base0 = d_m[:, bi, ai, 0, mi]
                for wi, w in enumerate(WGRID):
                    obs = d_m[:, bi, ai, wi, mi]
                    cnt = len(ns) - np.searchsorted(ns, obs, side="left")  # #{null >= obs}
                    p = (1 + cnt) / (1 + len(ns))
                    rows.append(dict(baseline=bn, metric=mn, sig_auc=a, w=w,
                                     delta_median=float(np.median(obs)),
                                     paired_inc_median=float(np.median(obs - base0)),
                                     det_p05=float(np.mean(p < 0.05)),
                                     det_fam15=float(np.mean(p <= 0.05 / 15)),
                                     det_fam27=float(np.mean(p <= 0.05 / 27)),
                                     null95=float(np.percentile(ns, 95))))
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/results_spikein_family.csv", index=False)
    print(res[res.w <= 2].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    json.dump(dict(icc=ICC, sig_auc=SIG_AUC, n_bases=nB, n_null=len(M[0, 0]), wgrid=WGRID),
              open(f"{OUT}/spikein_family_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
