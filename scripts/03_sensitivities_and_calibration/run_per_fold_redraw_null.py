#!/usr/bin/env python3
"""Decomposition cell: FULL-cohort |t| bins + per-fold redrawn identities.

Main analysis = full-cohort bins + fixed identity.
v2            = training-fold bins + per-fold adaptive identity.
This cell     = full-cohort bins + per-fold redraw (random draw inside the fixed
                full-cohort bin of each published gene).

Purpose: separate "matching source" from "identity strategy".  This cell still
uses full-cohort labels by construction; it is a diagnostic, not a label-clean
sensitivity.
"""
import argparse, os, sys, time, warnings
import numpy as np, pandas as pd
ROOT = "."
sys.path.insert(0, f"{ROOT}/exploration_v3/scripts")
sys.path.insert(0, f"{ROOT}/scripts/03_sensitivities_and_calibration")
from common import (load_base, FoldData, comp_features, scale_train, scale_train_batch,
                    fit_logistic, predict_logistic, fit_logistic_batch, predict_logistic_batch,
                    logloss, logloss_batch, fast_auc, fast_auc_batch, CELLTYPES)
from run_batch_rma_v2 import load_panels, load_folds, shared_features_main
warnings.filterwarnings("ignore")
DATA = "./data"; OUT = f"{ROOT}/results/foldwise_null"
BAS = ["comp1d", "compfull", "shared"]; MET = ["logloss", "auc"]; SEED = 20261106

MK = pd.read_csv(f"{ROOT}/exploration_v3/results/markers_celltype_top_v3.csv")
MARKERS = {ct: MK[MK.celltype == ct].sort_values("rank").gene_symbol.tolist() for ct in CELLTYPES}


def panel_genes(panels):
    s = set()
    for p in panels:
        for m in p["mapped"]:
            s.add(m["symbol"]); s.add(m.get("gene", m["symbol"]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--tag", default="fullredraw")
    a = ap.parse_args()
    B, NREP = a.n_null, a.n_repeats
    X, probe_ids, patient, label, gsm, title, _p, g2p = load_base(DATA)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    panels_all = load_panels(DATA); panels = panels_all
    nP, nS = len(panels), len(label); y = label
    folds = load_folds(f"{ROOT}/exploration_v3/results/folds_seed42.csv", NREP, nS)
    gt = pd.read_csv(f"{ROOT}/exploration_v3/results/gene_t_full.csv")
    genes = gt.gene.astype(str).tolist(); gi = {g: i for i, g in enumerate(genes)}
    nG = len(genes); full_bin = gt.bin.values.astype(np.int16)
    fs = ["markers_celltype_top_v3.csv", "markers_williams_derived_top.csv", "markers_williams_published.csv"]
    mset = set()
    for f in fs:
        d = pd.read_csv(f"{ROOT}/exploration_v3/results/{f}")
        col = "gene_symbol" if "gene_symbol" in d.columns else ("gene" if "gene" in d.columns else d.columns[0])
        mset |= set(d[col].dropna().astype(str))
    def vec(gs):
        v = np.zeros(nG, bool)
        for g in gs:
            if g in gi: v[gi[g]] = True
        return v
    forb = {True: vec(mset | panel_genes([p for p in panels if p["panel_id"] <= 27])),
            False: vec(mset | panel_genes(panels))}
    print(f"[fullredraw] markers={len(mset)} pools={int(forb[True].sum())}/{int(forb[False].sum())}", flush=True)
    alias = {p["panel_id"]: {m["symbol"] for m in p["mapped"]} for p in panels}
    reference = [p for p in panels_all if p["panel_id"] <= 27]
    kk = {p["panel_id"]: [m["symbol"] for m in p["mapped"]] for p in panels}
    pred_base = np.full((NREP, nS, nP, 3), np.nan, dtype=np.float32)
    pred_null = np.full((NREP, nS, nP, 3, B), np.nan, dtype=np.float32)
    dn = np.full((NREP, nP, 3, 2, B), np.nan)
    nfold = 0; t0 = time.time()
    for r, fl in enumerate(folds):
        for f, (tr, te) in enumerate(fl):
            fd = FoldData(X, probe_ids, probe_idx, g2p, list(g2p.keys()), label, tr, te)
            colarr = np.full(nG, -1, dtype=np.int32)
            tval = np.full(nG, np.nan)
            for g, col in fd.gene_pos.items():
                colarr[gi[g]] = col; tval[gi[g]] = fd.gene_t[g]
            for pi, p in enumerate(panels):
                klist = kk[p["panel_id"]]
                bad = forb[p["panel_id"] <= 27]
                rng = np.random.default_rng(SEED + 7919 * nfold + p["panel_id"])
                sel = np.full((B, len(klist)), -1, dtype=np.int32)
                for i, g in enumerate(klist):
                    j = int(full_bin[gi[g]])
                    pool = np.where((full_bin == j) & (~bad) & (colarr >= 0))[0]
                    if len(pool) < len(klist):
                        raise RuntimeError(f"panel {p['panel_id']} bin {j}: pool {len(pool)}")
                    pick = pool[rng.integers(0, len(pool), B)]
                    while True:
                        dup = np.zeros(B, bool)
                        for prev in range(i):
                            dup |= pick == sel[:, prev]
                        if not dup.any(): break
                        pick[dup] = pool[rng.integers(0, len(pool), int(dup.sum()))]
                    sel[:, i] = pick
                S = np.zeros((nS, B))
                for i, g in enumerate(klist):
                    S += fd.Z[:, colarr[sel[:, i]]] * np.sign(tval[sel[:, i]]) / len(klist)
                comp = comp_features(fd, MARKERS, alias[p["panel_id"]])
                b1_tr, b1_te, _, _ = scale_train(comp["pc1_tr"], comp["pc1_te"])
                designs = {"comp1d": (b1_tr[:, None], b1_te[:, None]),
                           "compfull": (comp["full_tr"], comp["full_te"])}
                sh = shared_features_main(fd, p, reference)
                if sh is not None:
                    bs_tr, bs_te, _, _ = scale_train(sh["pc_tr"], sh["pc_te"])
                    designs["shared"] = (bs_tr[:, None], bs_te[:, None])
                S_tr, S_te, _, _ = scale_train_batch(S[tr], S[te])
                for bi, bn in enumerate(BAS):
                    if bn not in designs: continue
                    Btr, Bte = designs[bn]
                    pred_base[r, te, pi, bi] = predict_logistic(fit_logistic(Btr, label[tr]), Bte)
                    coef = fit_logistic_batch(Btr, S_tr, label[tr])
                    pred_null[r, te, pi, bi, :] = predict_logistic_batch(coef, Bte, S_te)
            nfold += 1
        if np.isnan(pred_null[r]).any() or np.isnan(pred_base[r]).any():
            raise RuntimeError(f"repeat {r} incomplete")
        for bi in range(3):
            P = pred_null[r, :, :, bi, :]
            for pi in range(nP):
                pb = pred_base[r, :, pi, bi]
                dn[r, pi, bi, 0] = logloss(y, pb) - logloss_batch(y, P[:, pi, :])
                dn[r, pi, bi, 1] = fast_auc_batch(y, P[:, pi, :]) - fast_auc(y, pb)
        print(f"[fullredraw] repeat {r+1}/{NREP} {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(f"{OUT}/foldwise_null_{a.tag}.npz", delta=np.nanmean(dn, axis=0),
                        completed=np.ones(NREP, bool))


if __name__ == "__main__":
    main()
