#!/usr/bin/env python3
"""Selection-aware patient bootstrap.

The conditional interval in 04_incremental_v3.py re-uses fixed out-of-fold
predictions; it does NOT include probe selection, direction learning, marker
re-scoring, PCA loadings or model refitting.  This script re-runs the whole
training pipeline inside every patient bootstrap replicate:

  for each bootstrap b:
    draw 119 patients with replacement
    for each of the first R repeats and each of the 5 folds:
      resample the training patients with the bootstrap multiplicities
      recompute Welch t, representative probes, directions, z-scores,
      cell-type scores (dynamic leave-target-panel-genes-out), PC1 / 5-D
      baseline, panel score and logistic models
      predict the original held-out sites
    evaluate on the bootstrap patient sample (with multiplicity), average
    over the R repeats

The resulting percentile interval includes feature-selection uncertainty
conditional on the published gene list.  It still does NOT include uncertainty
in how the original publication selected its genes.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_base, make_folds, welch_t, panel_gene_sets, CELLTYPES,
                    comp_features, scale_train, fit_logistic, predict_logistic,
                    logloss, fast_auc)

warnings.filterwarnings("ignore")


class SmallFold:
    def __init__(self, X, genes, g2p, probe_idx, probe_ids, label, tr, te):
        self.tr = np.asarray(tr, int); self.te = np.asarray(te, int)
        self.y_tr = label[self.tr]; self.y_te = label[self.te]
        self.n = X.shape[0]
        cols = sorted({probe_idx[p] for g in genes for p in g2p.get(g, []) if p in probe_idx})
        self.cols = np.asarray(cols, int)
        Xtr = X[self.tr][:, self.cols].astype(np.float64)
        mu = Xtr.mean(0); sd = Xtr.std(0, ddof=1)
        valid = np.isfinite(sd) & (sd > 1e-6)
        t = welch_t(Xtr, self.y_tr)
        local = {c: i for i, c in enumerate(self.cols)}
        self.gene_pos = {}; self.gene_t = {}; self.gene_probe = {}
        chosen = []
        for g in genes:
            best = None; best_key = None
            for p in g2p.get(g, []):
                j = probe_idx.get(p)
                if j is None or j not in local:
                    continue
                li = local[j]
                if not valid[li]:
                    continue
                key = (-abs(t[li]), p)
                if best_key is None or key < best_key:
                    best_key = key; best = li
            if best is None:
                continue
            self.gene_pos[g] = len(chosen)
            self.gene_t[g] = float(t[best])
            self.gene_probe[g] = probe_ids[self.cols[best]]
            chosen.append(self.cols[best])
        self.Z = X[:, np.asarray(chosen, int)].astype(np.float64)
        self.Z = (self.Z - mu[[local[c] for c in chosen]]) / sd[[local[c] for c in chosen]]

    def score(self, pairs):
        cols, dirs = [], []
        for g, d in pairs:
            p = self.gene_pos.get(g)
            if p is None:
                continue
            if d == 0:
                d = 1 if self.gene_t[g] >= 0 else -1
            cols.append(p); dirs.append(d)
        if not cols:
            return None, 0, [], []
        S = self.Z[:, cols] @ (np.asarray(dirs, float) / len(cols))
        return S, len(cols), cols, np.asarray(dirs, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--max-panels", type=int, default=27)
    ap.add_argument("--tag", default="primary")
    args = ap.parse_args()

    X, probe_ids, patient, label, gsm, title, panels, g2p = load_base(args.data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    panels = panels[: args.max_panels]
    markers = pd.read_csv(os.path.join(args.out_dir, "results", "markers_celltype_top_v3.csv"))
    marker_genes = {ct: markers[markers.celltype == ct].sort_values("rank").gene_symbol.tolist()
                    for ct in CELLTYPES}
    alias = {p["panel_id"]: {m["symbol"] for m in p["mapped"]} | {m["gene"] for m in p["mapped"]}
             for p in panels}
    folds_all = make_folds(patient, label, args.seed, args.n_repeats)
    folds = folds_all[: args.n_repeats]

    patients = np.unique(patient)
    pat_sites = {int(p): np.where(patient == p)[0] for p in patients}
    rng = np.random.default_rng(args.seed + 999)
    boot_patients = [rng.choice(patients, size=len(patients), replace=True) for _ in range(args.n_boot)]

    nP = len(panels)
    delta = np.full((nP, 2, 2, args.n_boot), np.nan)  # baseline 1d/full, metric, boot
    base_metric = np.full((nP, 2, 2, args.n_boot), np.nan)
    t0 = time.time()

    for b, bp in enumerate(boot_patients):
        boot_sites = np.concatenate([pat_sites[int(p)] for p in bp])
        # multiplicity of each patient in this bootstrap
        mult = pd.Series(bp).value_counts().to_dict()
        for pi, panel in enumerate(panels):
            needed = list(dict.fromkeys(
                [m["symbol"] for m in panel["mapped"]] +
                [g for ct in CELLTYPES for g in marker_genes[ct]]))
            # predictions for this bootstrap, per repeat
            preds = {0: np.full((args.n_repeats, len(label)), np.nan),
                     1: np.full((args.n_repeats, len(label)), np.nan),
                     2: np.full((args.n_repeats, len(label)), np.nan),
                     3: np.full((args.n_repeats, len(label)), np.nan)}
            for r in range(args.n_repeats):
                for (tr, te) in folds[r]:
                    # bootstrap training set: training patients with multiplicity
                    tr_pat = set(patient[tr].tolist())
                    tr_idx = []
                    for p, m in mult.items():
                        if int(p) in tr_pat:
                            tr_idx.extend(list(pat_sites[int(p)]) * int(m))
                    if len(tr_idx) < 20:
                        continue
                    fd = SmallFold(X, needed, g2p, probe_idx, probe_ids, label,
                                   np.asarray(tr_idx, int), te)
                    comp = comp_features(fd, marker_genes, alias[panel["panel_id"]])
                    s, k, _, _ = fd.score([(m["symbol"], m.get("direction_published", 0))
                                           for m in panel["mapped"]])
                    s_tr, s_te, _, _ = scale_train(s[fd.tr], s[fd.te])
                    for bi, feat_tr, feat_te in [
                        (0, comp["pc1_tr"], comp["pc1_te"]),
                        (1, comp["full_tr"], comp["full_te"])]:
                        if bi == 0:
                            b_tr, b_te, _, _ = scale_train(feat_tr, feat_te)
                            b_tr = b_tr[:, None]; b_te = b_te[:, None]
                        else:
                            b_tr, b_te = feat_tr, feat_te
                        preds[bi][r, te] = predict_logistic(fit_logistic(b_tr, fd.y_tr), b_te)
                        preds[bi + 2][r, te] = predict_logistic(
                            fit_logistic(np.column_stack([b_tr, s_tr]), fd.y_tr),
                            np.column_stack([b_te, s_te]))
            for bi in range(2):
                for mi, mn in enumerate(["logloss", "auc"]):
                    per = []
                    for r in range(args.n_repeats):
                        pb = preds[bi][r][boot_sites]
                        pf = preds[bi + 2][r][boot_sites]
                        yb = label[boot_sites]
                        if np.isnan(pb).any() or np.isnan(pf).any():
                            continue
                        per.append(logloss(yb, pb) - logloss(yb, pf) if mn == "logloss"
                                   else fast_auc(yb, pf) - fast_auc(yb, pb))
                    if per:
                        delta[pi, bi, mi, b] = float(np.mean(per))
                        base_metric[pi, bi, mi, b] = float(np.mean(
                            [logloss(label[boot_sites], preds[bi][r][boot_sites]) if mn == "logloss"
                             else fast_auc(label[boot_sites], preds[bi][r][boot_sites])
                             for r in range(args.n_repeats)]))
        print(f"boot {b+1}/{args.n_boot} elapsed {time.time()-t0:.0f}s", flush=True)

    rows = []
    for pi, panel in enumerate(panels):
        row = {"panel_id": panel["panel_id"], "pathway": panel["pathway"],
               "n_boot": args.n_boot, "n_repeats_used": args.n_repeats}
        for bi, bn in enumerate(["comp1d", "compfull"]):
            for mi, mn in enumerate(["logloss", "auc"]):
                v = delta[pi, bi, mi]
                v = v[np.isfinite(v)]
                row[f"sel_{mn}_{bn}_mean"] = float(np.nanmean(v))
                row[f"sel_{mn}_{bn}_sd"] = float(np.nanstd(v, ddof=1))
                row[f"sel_{mn}_{bn}_ci_lo"] = float(np.nanpercentile(v, 2.5))
                row[f"sel_{mn}_{bn}_ci_hi"] = float(np.nanpercentile(v, 97.5))
                row[f"sel_{mn}_{bn}_n"] = int(len(v))
                row[f"sel_{mn}_{bn}_ci_type"] = "selection_aware_patient_bootstrap_full_refit"
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "results", f"bootstrap_selection_{args.tag}.csv"), index=False)
    np.savez_compressed(os.path.join(args.out_dir, "results", f"bootstrap_selection_{args.tag}.npz"),
                        delta=delta, base_metric=base_metric,
                        panel_id=np.array([p["panel_id"] for p in panels]),
                        baseline=np.array(["comp1d", "compfull"]),
                        metric=np.array(["logloss", "auc"]))
    print(res[["panel_id", "pathway", "sel_auc_comp1d_ci_lo", "sel_auc_comp1d_ci_hi",
               "sel_auc_compfull_ci_lo", "sel_auc_compfull_ci_hi"]].to_string(index=False))
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
