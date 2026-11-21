#!/usr/bin/env python3
"""Within-patient sensitivity (audit _.md §4, 2026-11-02).

Restrict GSE16134 to the 68 patients who contribute both diseased and unaffected
sites (204 sites: 136 cases / 68 controls).  Same engine, seed 42, 20 x 5
patient-clustered folds recomputed on the subset, first 1,000 fixed random
identities, 27/15 BH families, shared axis with the fixed 27-panel reference.

Plus a fold-free descriptive metric: for each of the 68 patients, is the mean
score of the diseased sites higher than the single unaffected site?  Reported
per panel and for the composition axes (whole-cohort z-scoring).
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd

ROOT = "."
sys.path.insert(0, f"{ROOT}/exploration_v3/scripts")
sys.path.insert(0, f"{ROOT}/scripts/03_sensitivities_and_calibration")
from common import (load_base, make_folds, FoldData, comp_features, scale_train,
                    scale_train_batch, pca1, fit_logistic, predict_logistic,
                    fit_logistic_batch, predict_logistic_batch, logloss, logloss_batch,
                    fast_auc, fast_auc_batch, bh_q, CELLTYPES)
from run_batch_rma_v2 import (load_panels, load_identity, null_score_matrix, shared_features_main)

warnings.filterwarnings("ignore")
BASELINES = ["comp1d", "compfull", "shared"]
METRICS = ["logloss", "auc"]
OUT = f"{ROOT}/results/within_patient"
os.makedirs(OUT, exist_ok=True)
B = 1000


def main():
    data_dir = "./data"
    X, probe_ids, patient, label, gsm, title, _p, g2p = load_base(data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    panels = load_panels(data_dir)
    reference = [p for p in panels if p["panel_id"] <= 27]
    both = np.array([p for p in np.unique(patient)
                     if len(set(label[patient == p])) == 2])
    keep = np.isin(patient, both)
    X, patient, label, gsm = X[keep], patient[keep], label[keep], np.asarray(gsm)[keep]
    nS = len(label)
    print(f"[within] sites={nS} cases={int(label.sum())} patients={len(np.unique(patient))}", flush=True)

    mk = pd.read_csv(f"{ROOT}/exploration_v3/results/markers_celltype_top_v3.csv")
    marker_genes = {ct: mk[mk.celltype == ct].sort_values("rank").gene_symbol.tolist() for ct in CELLTYPES}

    ident = {}
    ident.update(load_identity(f"{ROOT}/exploration_v3/results/null_panels_identity.csv.gz",
                               set(range(1, 28)), B))
    ident.update(load_identity(f"{ROOT}/confirmation_v1/results/null_panels_identity.csv.gz",
                               set(range(101, 116)), B, sep="\t"))

    nR = 20
    folds = make_folds(patient, label, 42, nR, 5)
    rows = []
    for r, fl in enumerate(folds):
        for f, (tr, te) in enumerate(fl):
            for i in te:
                rows.append({"site": int(i), "patient": int(patient[i]), "repeat": r, "fold": f})
    pd.DataFrame(rows).to_csv(f"{OUT}/folds_within_seed42.csv", index=False)

    alias = {p["panel_id"]: {m["symbol"] for m in p["mapped"]} for p in panels}
    genes_all = list(g2p.keys())
    nP = len(panels)
    obs_delta = np.full((nP, 3, 2, nR), np.nan)
    null_delta = np.full((nP, 3, 2, B, nR), np.nan)

    t0 = time.time()
    for r in range(nR):
        pred_null = np.full((nP, 3, nS, B), np.nan, dtype=np.float32)
        pred_base = np.full((nP, 3, nS), np.nan)
        pred_full = np.full((nP, 3, nS), np.nan)
        for f, (tr, te) in enumerate(folds[r]):
            fd = FoldData(X, probe_ids, probe_idx, g2p, genes_all, label, tr, te)
            y_tr = label[tr]
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
                sh = shared_features_main(fd, panel, reference)
                if sh is not None:
                    bs_tr, bs_te, _, _ = scale_train(sh["pc_tr"], sh["pc_te"])
                    designs["shared"] = (bs_tr[:, None], bs_te[:, None])
                s_tr, s_te, _, _ = scale_train(obs_s[tr], obs_s[te])
                for bi, bn in enumerate(BASELINES):
                    if bn not in designs:
                        continue
                    Btr, Bte = designs[bn]
                    pred_base[pi, bi, te] = predict_logistic(fit_logistic(Btr, y_tr), Bte)
                    pred_full[pi, bi, te] = predict_logistic(
                        fit_logistic(np.column_stack([Btr, s_tr]), y_tr),
                        np.column_stack([Bte, s_te]))
                S, _ = null_score_matrix(fd, ident[panel["panel_id"]])
                S_tr, S_te, _, _ = scale_train_batch(S[tr], S[te])
                for bi, bn in enumerate(BASELINES):
                    if bn not in designs:
                        continue
                    Btr, Bte = designs[bn]
                    coef = fit_logistic_batch(Btr, S_tr, y_tr)
                    pred_null[pi, bi, te, :] = predict_logistic_batch(coef, Bte, S_te)
        y = label
        for pi in range(nP):
            for bi in range(3):
                base, full = pred_base[pi, bi], pred_full[pi, bi]
                if np.isnan(base).any() or np.isnan(full).any():
                    continue
                llb, aub = logloss(y, base), fast_auc(y, base)
                obs_delta[pi, bi, 0, r] = llb - logloss(y, full)
                obs_delta[pi, bi, 1, r] = fast_auc(y, full) - aub
                P = pred_null[pi, bi]
                ok = ~np.isnan(P[0])
                if ok.any():
                    null_delta[pi, bi, 0, ok, r] = llb - logloss_batch(y, P[:, ok])
                    null_delta[pi, bi, 1, ok, r] = fast_auc_batch(y, P[:, ok]) - aub
        del pred_null
        print(f"[within] repeat {r+1}/{nR} {time.time()-t0:.0f}s", flush=True)

    null_mean = np.nanmean(null_delta, axis=4)
    rows = []
    for pi, panel in enumerate(panels):
        for bi, bn in enumerate(BASELINES):
            for mi, mn in enumerate(METRICS):
                obs = float(np.nanmean(obs_delta[pi, bi, mi]))
                nb = null_mean[pi, bi, mi]
                nb = nb[np.isfinite(nb)]
                if len(nb) == 0:
                    continue
                p = (1 + np.sum(nb >= obs)) / (1 + len(nb))
                rows.append(dict(panel_id=panel["panel_id"], baseline=bn, metric=mn,
                                 observed_mean=obs, null_mean=float(nb.mean()),
                                 null95=float(np.percentile(nb, 95)), p=p, n_null=len(nb)))
    res = pd.DataFrame(rows)
    res["cohort"] = np.where(res.panel_id <= 27, "discovery", "confirmation")
    res["q"] = np.nan
    for (_co, _b, _m), g in res.groupby(["cohort", "baseline", "metric"]):
        res.loc[g.index, "q"] = bh_q(g.p.values)
    res.to_csv(f"{OUT}/results_per_test_within.csv", index=False)
    print(res[res.q < 0.05][["panel_id", "cohort", "baseline", "metric", "observed_mean", "p", "q"]]
          .to_string(index=False))
    print(f"[within] done {time.time()-t0:.0f}s")

    # ---- descriptive within-patient concordance (fold-free z-scoring) ----
    # collapse probes to genes (representative probe = largest |t|)
    from common import welch_t
    t = welch_t(X, label)
    g2p_json = g2p
    rows = []
    for g, ps in g2p_json.items():
        best, key0 = None, None
        for p in ps:
            j = probe_idx.get(p)
            if j is None:
                continue
            k = (-abs(t[j]), p)
            if key0 is None or k < key0:
                key0, best = k, j
        if best is not None:
            rows.append((g, best))
    rep = dict(rows)
    G = pd.DataFrame({g: X[:, j] for g, j in rows}, index=[str(x) for x in gsm])
    G = G.loc[:, ~G.columns.duplicated()]
    Gz = G.sub(G.mean(), axis=1).div(G.std(ddof=1).replace(0, np.nan), axis=1)

    def score(pairs):
        cols = [g for g, _ in pairs if g in Gz.columns]
        s = np.array([d for g, d in pairs if g in Gz.columns], float)
        return Gz[cols].values @ s / len(cols)

    pts = np.unique(patient)
    disc = []
    for p in panels:
        pairs = [(m["symbol"], m.get("direction_published", 0)) for m in p["mapped"]]
        # missing direction -> whole-cohort t sign
        pairs = [(g, d if d != 0 else (1.0 if t[rep[g]] >= 0 else -1.0))
                 if g in Gz.columns else (g, d) for g, d in pairs]
        s = score([(g, d) for g, d in pairs if g in Gz.columns])
        hit = []
        for pt in pts:
            case = s[(patient == pt) & (label == 1)]
            ctrl = s[(patient == pt) & (label == 0)]
            if len(ctrl) and len(case):
                hit.append(case.mean() > ctrl.mean())
        disc.append(dict(panel_id=p["panel_id"], scope="exploration" if p["panel_id"] <= 27 else "confirmation",
                         n_patients=len(hit), frac_case_gt_ctrl=float(np.mean(hit))))
    # composition axes
    comp = {}
    for ct, gs in marker_genes.items():
        gg = [g for g in gs if g in Gz.columns]
        comp[ct] = Gz[gg].mean(axis=1).values
    C = pd.DataFrame(comp)
    Cz = C.sub(C.mean()).div(C.std(ddof=1))
    Cc = Cz.values - Cz.values.mean(axis=0)
    u, sv, vt = np.linalg.svd(Cc, full_matrices=False)
    pc1 = u[:, 0] * sv[0]
    if np.corrcoef(pc1, Cz["other_immune"].values)[0, 1] < 0:
        pc1 = -pc1
    for name, s in [("comp1d_PC1", pc1)] + [(f"comp_{ct}", v) for ct, v in comp.items()]:
        hit = []
        for pt in pts:
            case = s[(patient == pt) & (label == 1)]
            ctrl = s[(patient == pt) & (label == 0)]
            if len(ctrl) and len(case):
                hit.append(case.mean() > ctrl.mean())
        disc.append(dict(panel_id="axis:" + name, scope="composition", n_patients=len(hit),
                         frac_case_gt_ctrl=float(np.mean(hit))))
    d = pd.DataFrame(disc)
    d.to_csv(f"{OUT}/within_patient_concordance.csv", index=False)
    print(d.groupby("scope").frac_case_gt_ctrl.describe().round(3).to_string())
    print("axis rows:")
    print(d[d.scope == "composition"].to_string(index=False))
    json.dump(dict(n_sites=int(nS), n_patients=int(len(pts)),
                   n_case=int(label.sum()), n_ctrl=int((1 - label).sum()),
                   median_frac=d[d.scope != "composition"].frac_case_gt_ctrl.median(),
                   range=[float(d[d.scope != "composition"].frac_case_gt_ctrl.min()),
                          float(d[d.scope != "composition"].frac_case_gt_ctrl.max())]),
              open(f"{OUT}/within_summary.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
