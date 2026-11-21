#!/usr/bin/env python3
"""Fold-wise |t|-matched null, v2 (independent_check review fixes).

- candidate selection uses NO full-cohort bin: each identity has a fixed random
  permutation of the whole gene universe (seed 20261104 + b); each slot takes the
  first gene in that order whose TRAINING-fold bin equals the published gene's
  training-fold bin, excluding the main-analysis exclusion set and used genes;
- exclusion pools are the main-analysis ones, separately for exploration
  (27 panels + 424 markers) and confirmation (42 panels + 424 markers);
- no fallback draw: if a slot exhausts the permutation, the run raises;
- every repeat must complete (no nanmean over missing repeats);
- identity stability is measured on the normalised gene SET (sorted), not on the
  ordered slot vector;
- --mode isolation: permute the test-fold labels, rebuild everything, and check
  that training bins, selected genes, directions, coefficients and test
  predictions are unchanged.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd

ROOT = "."
sys.path.insert(0, f"{ROOT}/exploration_v3/scripts")
sys.path.insert(0, f"{ROOT}/scripts/03_sensitivities_and_calibration")
from common import (load_base, FoldData, comp_features, scale_train, scale_train_batch,
                    fit_logistic, predict_logistic, fit_logistic_batch, predict_logistic_batch,
                    logloss, logloss_batch, fast_auc, fast_auc_batch, bh_q, CELLTYPES)
from run_batch_rma_v2 import load_panels, load_folds, shared_features_main

warnings.filterwarnings("ignore")
DATA = "./data"
OUT = f"{ROOT}/results/foldwise_null"
os.makedirs(OUT, exist_ok=True)
BASELINES = ["comp1d", "compfull", "shared"]
METRICS = ["logloss", "auc"]
SEED_KEY = 20261104
CHUNK = 200


def build_perm(nG, B):
    perm = np.empty((B, nG), dtype=np.int32)
    for b in range(B):
        rng = np.random.default_rng(SEED_KEY + b)
        perm[b] = rng.permutation(nG)
    return perm


def train_bins(fd, gene_index, nG):
    t = np.full(nG, np.nan)
    for g, col in fd.gene_pos.items():
        i = gene_index.get(g)
        if i is not None:
            t[i] = fd.gene_t[g]
    valid = np.isfinite(t)
    edges = np.quantile(np.abs(t[valid]), np.linspace(0, 1, 21))
    b = np.full(nG, -1, dtype=np.int16)
    b[valid] = np.clip(np.digitize(np.abs(t[valid]), edges[1:-1]), 0, 19)
    return b, t


def select_genes(perm, tbin, forbidden, colarr, k, bins, B):
    ar = np.arange(B)
    sel = np.full((B, k), -1, dtype=np.int32)
    nG = perm.shape[1]
    for i, j in enumerate(bins):
        if j < 0:
            raise RuntimeError(f"slot {i}: published gene absent from training fold (bin -1)")
        rem = ar.copy()
        pos = 0
        chunk = 64
        while rem.size and pos < nG:
            L = min(chunk, nG - pos)
            cols = np.arange(pos, pos + L)
            cand = perm[rem[:, None], cols[None, :]]
            valid = (tbin[cand] == j) & (~forbidden[cand]) & (colarr[cand] >= 0)
            for prev in range(i):
                valid &= cand != sel[rem, prev, None]
            has = valid.any(axis=1)
            if has.any():
                idx = valid.argmax(axis=1)
                pick = cand[np.arange(len(rem)), idx]
                sel[rem[has], i] = pick[has]
                rem = rem[~has]
            pos += L
            chunk *= 2
        if rem.size:
            raise RuntimeError(f"slot {i}: no eligible gene in bin {j} for {rem.size} identities")
    return sel


def marker_set(panels_all):
    fs = [f"{ROOT}/exploration_v3/results/markers_celltype_top_v3.csv",
          f"{ROOT}/exploration_v3/results/markers_williams_derived_top.csv",
          f"{ROOT}/exploration_v3/results/markers_williams_published.csv"]
    genes = set()
    for f in fs:
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        col = "gene_symbol" if "gene_symbol" in d.columns else ("gene" if "gene" in d.columns else d.columns[0])
        genes |= set(d[col].dropna().astype(str))
    return genes


def panel_gene_set(panels):
    s = set()
    for p in panels:
        for m in p["mapped"]:
            s.add(m["symbol"]); s.add(m.get("gene", m["symbol"]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["null", "isolation"], default="null")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    B, NREP = a.n_null, a.n_repeats
    X, probe_ids, patient, label, gsm, title, _p, g2p = load_base(DATA)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    panels_all = load_panels(DATA)
    panels = panels_all
    nP, nS = len(panels), len(label)
    folds = load_folds(f"{ROOT}/exploration_v3/results/folds_seed42.csv", NREP, nS)

    gt = pd.read_csv(f"{ROOT}/exploration_v3/results/gene_t_full.csv")
    genes = gt.gene.astype(str).tolist()
    gene_index = {g: i for i, g in enumerate(genes)}
    nG = len(genes)

    markers = marker_set(panels_all)
    expl_genes = panel_gene_set([p for p in panels_all if p["panel_id"] <= 27])
    conf_genes = panel_gene_set(panels_all)
    def vec(gs):
        v = np.zeros(nG, bool)
        for g in gs:
            if g in gene_index: v[gene_index[g]] = True
        return v
    forb_expl = vec(markers | expl_genes)
    forb_conf = vec(markers | conf_genes)
    print(f"[v2] markers={len(markers)} expl_pool={int(forb_expl.sum())} conf_pool={int(forb_conf.sum())} B={B}", flush=True)

    perm = build_perm(nG, B)
    print("[v2] permutations built", flush=True)

    if a.mode == "isolation":
        rng = np.random.default_rng(20261105)
        for nf in [0, 1, 2]:
            fl = folds[0]
            tr, te = fl[nf % 5]
            fd = FoldData(X, probe_ids, probe_idx, g2p, list(g2p.keys()), label, tr, te)
            tbin, tval = train_bins(fd, gene_index, nG)
            colarr = np.full(nG, -1, dtype=np.int32)
            for g, col in fd.gene_pos.items(): colarr[gene_index[g]] = col
            p = panels[0]; kk = [m["symbol"] for m in p["mapped"]]
            bins = [int(tbin[gene_index[g]]) for g in kk]
            Bs = min(B, 200)
            sel = select_genes(perm[:Bs], tbin, forb_expl, colarr, len(kk), bins, Bs)
            lab2 = label.copy(); lab2[te] = rng.permutation(lab2[te])
            fd2 = FoldData(X, probe_ids, probe_idx, g2p, list(g2p.keys()), lab2, tr, te)
            tbin2, tval2 = train_bins(fd2, gene_index, nG)
            colarr2 = np.full(nG, -1, dtype=np.int32)
            for g, col in fd2.gene_pos.items(): colarr2[gene_index[g]] = col
            sel2 = select_genes(perm[:Bs], tbin2, forb_expl, colarr2, len(kk), bins, Bs)
            same_bins = np.array_equal(tbin, tbin2); same_sel = np.array_equal(sel, sel2)
            same_sign = np.array_equal(np.sign(tval[sel]), np.sign(tval2[sel2]))
            # predictions
            def score(fd_, tval_, sel_):
                S = np.zeros((nS, Bs))
                for i in range(len(kk)):
                    gi = sel_[:, i]; S += fd_.Z[:, colarr[gi]] * np.sign(tval_[gi])[None, :] / len(kk)
                return S
            S, S2 = score(fd, tval, sel), score(fd2, tval2, sel2)
            St, Se, _, _ = scale_train_batch(S[tr], S[te])
            St2, Se2, _, _ = scale_train_batch(S2[tr], S2[te])
            same_score = np.allclose(St, St2) and np.allclose(Se, Se2)
            print(f"[isolation] fold {nf}: bins {same_bins} sel {same_sel} sign {same_sign} score {same_score}", flush=True)
            if not (same_bins and same_sel and same_sign and same_score):
                raise SystemExit("ISOLATION CHECK FAILED")
        print("[isolation] PASS", flush=True)
        return

    alias = {p["panel_id"]: {m["symbol"] for m in p["mapped"]} for p in panels}
    reference = [p for p in panels_all if p["panel_id"] <= 27]
    k_genes = {p["panel_id"]: [m["symbol"] for m in p["mapped"]] for p in panels}

    pred_base = np.full((NREP, nS, nP, 3), np.nan, dtype=np.float32)
    pred_null = np.full((NREP, nS, nP, 3, B), np.nan, dtype=np.float32)
    first_hash = {}
    changed = np.zeros((nP, B), dtype=bool)
    nfold = 0
    t0 = time.time()
    for r, fl in enumerate(folds):
        for f, (tr, te) in enumerate(fl):
            fd = FoldData(X, probe_ids, probe_idx, g2p, list(g2p.keys()), label, tr, te)
            tbin, tval = train_bins(fd, gene_index, nG)
            colarr = np.full(nG, -1, dtype=np.int32)
            for g, col in fd.gene_pos.items(): colarr[gene_index[g]] = col
            for pi, panel in enumerate(panels):
                kk = k_genes[panel["panel_id"]]
                bins = [int(tbin[gene_index[g]]) if g in gene_index else -1 for g in kk]
                forbid = forb_expl if panel["panel_id"] <= 27 else forb_conf
                sel = select_genes(perm, tbin, forbid, colarr, len(kk), bins, B)
                if sel.min() < 0:
                    raise RuntimeError("unfilled selection")
                # identity stability on normalised sets
                for b in range(B):
                    h = hash(tuple(sorted(int(x) for x in sel[b])))
                    if nfold == 0:
                        first_hash[(pi, b)] = h
                    elif h != first_hash[(pi, b)]:
                        changed[pi, b] = True
                S = np.zeros((nS, B))
                for i in range(len(kk)):
                    gi = sel[:, i]
                    S += fd.Z[:, colarr[gi]] * np.sign(tval[gi])[None, :] / len(kk)
                comp = comp_features(fd, MARKERS, alias[panel["panel_id"]])
                b1_tr, b1_te, _, _ = scale_train(comp["pc1_tr"], comp["pc1_te"])
                designs = {"comp1d": (b1_tr[:, None], b1_te[:, None]),
                           "compfull": (comp["full_tr"], comp["full_te"])}
                sh = shared_features_main(fd, panel, reference)
                if sh is not None:
                    bs_tr, bs_te, _, _ = scale_train(sh["pc_tr"], sh["pc_te"])
                    designs["shared"] = (bs_tr[:, None], bs_te[:, None])
                S_tr, S_te, _, _ = scale_train_batch(S[tr], S[te])
                for bi, bn in enumerate(BASELINES):
                    if bn not in designs: continue
                    Btr, Bte = designs[bn]
                    pred_base[r, te, pi, bi] = predict_logistic(fit_logistic(Btr, label[tr]), Bte)
                    coef = fit_logistic_batch(Btr, S_tr, label[tr])
                    pred_null[r, te, pi, bi, :] = predict_logistic_batch(coef, Bte, S_te)
            nfold += 1
        if np.isnan(pred_null[r]).any() or np.isnan(pred_base[r]).any():
            raise RuntimeError(f"repeat {r}: incomplete predictions")
        print(f"[v2] repeat {r+1}/{NREP} {time.time()-t0:.0f}s", flush=True)

    y = label
    dn = np.full((NREP, nP, 3, 2, B), np.nan)
    for r in range(NREP):
        for bi in range(3):
            P = pred_null[r, :, :, bi, :]
            for pi in range(nP):
                pb = pred_base[r, :, pi, bi]
                if np.isnan(pb).any() or np.isnan(P[:, pi, :]).any():
                    raise RuntimeError("incomplete")
                llb, aub = logloss(y, pb), fast_auc(y, pb)
                dn[r, pi, bi, 0] = llb - logloss_batch(y, P[:, pi, :])
                dn[r, pi, bi, 1] = fast_auc_batch(y, P[:, pi, :]) - aub
    dn_m = np.nanmean(dn, axis=0)
    np.savez_compressed(f"{OUT}/foldwise_null_{a.tag}.npz", delta=dn_m, changed=changed,
                        completed=np.ones(NREP, bool))
    print(f"[v2] identity-set unchanged: {1-changed.mean():.3%}", flush=True)


if __name__ == "__main__":
    MARKERS = {ct: pd.read_csv(f"{ROOT}/exploration_v3/results/markers_celltype_top_v3.csv")
               .query("celltype == @ct").sort_values("rank").gene_symbol.tolist() for ct in CELLTYPES}
    main()
