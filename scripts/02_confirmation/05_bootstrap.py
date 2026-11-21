#!/usr/bin/env python3
"""Step 4a: selection-aware patient bootstrap (200 x 10) for comp1d/compfull.

Every bootstrap replicate draws 119 patients with replacement ONCE and uses
the same draw for all 15 panels.  For each panel it refits the whole training
pipeline (Welch t, representative probes, directions, z-scores, cell-type
scores with dynamic removal, PC1/5-D baseline, panel score, logistic model)
and predicts the original held-out sites.  The interval therefore includes
feature-selection uncertainty conditional on the published gene list, but not
the original publication's gene-selection process.

The joint median dAUC across the 15 panels is computed per bootstrap replicate
from the same patient draw, and its 2.5/97.5 percentiles are saved.

Outputs:
  results/bootstrap_selection_primary.csv
  results/bootstrap_selection_primary.npz
  results/joint_median_primary.json
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_base, load_folds, welch_t, comp_features, scale_train,
                    fit_logistic, predict_logistic, logloss, fast_auc, CELLTYPES)

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
    ap.add_argument("--v3-dir", required=True)
    ap.add_argument("--panels-json", default="01_panels_confirmation.json")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-repeats", type=int, default=10)
    args = ap.parse_args()

    res = os.path.join(args.out_dir, "results")
    X, probe_ids, patient, label, gsm, title, _, g2p = load_base(args.data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    panels = json.load(open(os.path.join(args.out_dir, args.panels_json)))
    alias = {p["panel_id"]: {m["symbol"] for m in p["mapped"]} | {m["gene"] for m in p["mapped"]}
             for p in panels}
    markers = pd.read_csv(os.path.join(args.v3_dir, "results", "markers_celltype_top_v3.csv"))
    marker_genes = {ct: markers[markers.celltype == ct].sort_values("rank").gene_symbol.tolist()
                    for ct in CELLTYPES}

    folds = load_folds(os.path.join(res, f"folds_seed{args.seed}.csv"),
                       args.n_repeats, len(label))
    patients = np.unique(patient)
    pat_sites = {int(p): np.where(patient == p)[0] for p in patients}
    rng = np.random.default_rng(args.seed + 999)
    boot_patients = np.stack([rng.choice(patients, size=len(patients), replace=True)
                              for _ in range(args.n_boot)])

    nP = len(panels)
    delta = np.full((nP, 2, 2, args.n_boot), np.nan)
    base_metric = np.full((nP, 2, 2, args.n_boot), np.nan)
    t0 = time.time()

    for b in range(args.n_boot):
        bp = boot_patients[b]
        boot_sites = np.concatenate([pat_sites[int(p)] for p in bp])
        mult = pd.Series(bp).value_counts().to_dict()
        for pi, panel in enumerate(panels):
            needed = list(dict.fromkeys(
                [m["symbol"] for m in panel["mapped"]] +
                [g for ct in CELLTYPES for g in marker_genes[ct]]))
            preds = {k: np.full((args.n_repeats, len(label)), np.nan)
                     for k in range(4)}
            for r in range(args.n_repeats):
                for (tr, te) in folds[r]:
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
                    if s is None:
                        continue
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
    resdf = pd.DataFrame(rows)
    resdf.to_csv(os.path.join(res, "bootstrap_selection_primary.csv"), index=False)

    joint = {}
    for bi, bn in enumerate(["comp1d", "compfull"]):
        for mi, mn in enumerate(["logloss", "auc"]):
            med = np.nanmedian(delta[:, bi, mi, :], axis=0)
            joint[f"{mn}_{bn}"] = {
                "median_per_boot": med.tolist(),
                "observed_panel_median": float(np.nanmedian(np.nanmean(
                    np.load(os.path.join(res, "obs_predictions_primary.npz"),
                            allow_pickle=True)["obs_delta"][:, bi, mi, :], axis=1))),
                "ci_lo": float(np.nanpercentile(med, 2.5)),
                "ci_hi": float(np.nanpercentile(med, 97.5)),
                "n_boot": int(np.isfinite(med).sum()),
            }
    np.savez_compressed(os.path.join(res, "bootstrap_selection_primary.npz"),
                        delta=delta, base_metric=base_metric,
                        boot_patients=boot_patients,
                        panel_id=np.array([p["panel_id"] for p in panels]),
                        baseline=np.array(["comp1d", "compfull"]),
                        metric=np.array(["logloss", "auc"]))
    json.dump(joint, open(os.path.join(res, "joint_median_primary.json"), "w"),
              ensure_ascii=False, indent=2)
    print(resdf[["panel_id", "pathway", "sel_auc_comp1d_ci_lo", "sel_auc_comp1d_ci_hi"]]
          .to_string(index=False))
    print("joint medians:")
    for k, v in joint.items():
        print(f"  {k}: obs median {v['observed_panel_median']:.6f}, "
              f"bootstrap CI [{v['ci_lo']:.6f}, {v['ci_hi']:.6f}]")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
