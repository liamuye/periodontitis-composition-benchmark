#!/usr/bin/env python3
"""Step 3a: observed out-of-fold predictions and deltas (float64).

Tags:
  primary          : comp1d, compfull, shared (GSE152042 5-class markers)
  seed20261031     : comp1d, compfull, shared on the second fold seed
  williams         : williams_derived_1d, williams_derived_full (12 classes)

The observed panel is always one fixed published panel; the 20x5 outer folds
come from results/folds_seed<seed>.csv.  No random panels are used here.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_base, load_folds, fold_stats, FoldData, comp_features,
                    panel_symbol_alias, panel_gene_sets, scale_train, pca1,
                    fit_logistic, predict_logistic, logloss, fast_auc, CELLTYPES)

warnings.filterwarnings("ignore")


def shared_features(fd, panel, reference_panels):
    from common import shared_gene_dirs
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
    M = fd.Z[:, cols] * np.asarray(dirs, dtype=float)[None, :]
    pc_tr, pc_te, var, load = pca1(M[fd.tr], M[fd.te])
    if np.mean(load) < 0:
        pc_tr, pc_te, load = -pc_tr, -pc_te, -load
    return {"pc_tr": pc_tr, "pc_te": pc_te, "var": var, "load": load, "k": len(cols)}


def load_markers(path, marker_set):
    d = pd.read_csv(path)
    if marker_set == "primary":
        out = {ct: d[d.celltype == ct].sort_values("rank").gene_symbol.tolist()
               for ct in CELLTYPES}
    else:
        out = {ct: g.sort_values("rank").gene_symbol.tolist()
               for ct, g in d.groupby("celltype")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--v3-dir", required=True)
    ap.add_argument("--panels-json", default="01_panels_confirmation.json")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--marker-set", choices=["primary", "williams"], default="primary")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    res = os.path.join(args.out_dir, "results")
    os.makedirs(res, exist_ok=True)
    out_path = os.path.join(res, f"obs_predictions_{args.tag}.npz")
    if os.path.exists(out_path) and not args.force:
        print(f"exists, skip: {out_path}")
        return

    X, probe_ids, patient, label, gsm, title, _, g2p = load_base(args.data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    panels = json.load(open(os.path.join(args.out_dir, args.panels_json)))
    exploration = json.load(open(os.path.join(args.data_dir, "panels.json")))
    alias = panel_symbol_alias(panels)
    nP, nS, nR = len(panels), len(label), args.n_repeats

    if args.marker_set == "primary":
        marker_path = os.path.join(args.v3_dir, "results", "markers_celltype_top_v3.csv")
        baselines = ["comp1d", "compfull", "shared"]
    else:
        marker_path = os.path.join(args.v3_dir, "results", "markers_williams_derived_top.csv")
        baselines = ["williams_derived_1d", "williams_derived_full"]
    marker_genes = load_markers(marker_path, args.marker_set)
    print(f"markers: {len(marker_genes)} types, "
          f"{sum(len(v) for v in marker_genes.values())} genes")

    folds = load_folds(os.path.join(res, f"folds_seed{args.seed}.csv"), nR, nS)
    nB = len(baselines)
    obs_delta = np.full((nP, nB, 2, nR), np.nan, dtype=np.float64)
    obs_base = np.full((nP, nB, 2, nR), np.nan, dtype=np.float64)
    pred_base = np.full((nP, nB, nR, nS), np.nan, dtype=np.float64)
    pred_full = np.full((nP, nB, nR, nS), np.nan, dtype=np.float64)
    k_used = np.full((nP, nR, 5), np.nan)
    base_var = np.full((nP, nB, nR), np.nan)
    corr_panel = np.full((nP, nB, nR), np.nan)

    # genes needed by any observed fit
    genes_obs = set()
    for p in panels:
        genes_obs |= {m["symbol"] for m in p["mapped"]}
    for v in marker_genes.values():
        genes_obs |= set(v)
    if args.marker_set == "primary":
        for p in exploration:
            genes_obs |= {m["symbol"] for m in p["mapped"]}
    genes_obs = sorted(genes_obs)
    print(f"observed gene universe: {len(genes_obs)}")

    t0 = time.time()
    stats_cache = {}
    for r in range(nR):
        for f, (tr, te) in enumerate(folds[r]):
            if (r, f) not in stats_cache:
                stats_cache[(r, f)] = fold_stats(X, label, tr)
            fd = FoldData(X, probe_ids, probe_idx, g2p, genes_obs, label, tr, te,
                          stats=stats_cache[(r, f)])
            y_tr = label[tr]
            for pi, panel in enumerate(panels):
                obs_s, k, _, _ = fd.score([(m["symbol"], m.get("direction_published", 0))
                                           for m in panel["mapped"]])
                if obs_s is None:
                    continue
                k_used[pi, r, f] = k
                comp = comp_features(fd, marker_genes, alias[panel["panel_id"]])
                shared = shared_features(fd, panel, exploration) if args.marker_set == "primary" else None

                b1_tr, b1_te, _, _ = scale_train(comp["pc1_tr"], comp["pc1_te"])
                designs = {}
                if args.marker_set == "primary":
                    designs["comp1d"] = (b1_tr[:, None], b1_te[:, None], comp["var"])
                    designs["compfull"] = (comp["full_tr"], comp["full_te"], comp["var"])
                    if shared is not None:
                        bs_tr, bs_te, _, _ = scale_train(shared["pc_tr"], shared["pc_te"])
                        designs["shared"] = (bs_tr[:, None], bs_te[:, None], shared["var"])
                else:
                    designs["williams_derived_1d"] = (b1_tr[:, None], b1_te[:, None], comp["var"])
                    designs["williams_derived_full"] = (comp["full_tr"], comp["full_te"], comp["var"])

                s_tr, s_te, _, _ = scale_train(obs_s[tr], obs_s[te])
                for bi, bname in enumerate(baselines):
                    if bname not in designs:
                        continue
                    Btr, Bte, var = designs[bname]
                    pb = predict_logistic(fit_logistic(Btr, y_tr), Bte)
                    pf = predict_logistic(
                        fit_logistic(np.column_stack([Btr, s_tr]), y_tr),
                        np.column_stack([Bte, s_te]))
                    pred_base[pi, bi, r, te] = pb
                    pred_full[pi, bi, r, te] = pf
                    base_var[pi, bi, r] = var
                    pc = comp["pc1_tr"] if bname.endswith("1d") or bname == "comp1d" else None
                    if pc is None and bname == "compfull":
                        pc = comp["pc1_tr"]
                    if pc is None and bname == "williams_derived_full":
                        pc = comp["pc1_tr"]
                    if pc is not None:
                        corr_panel[pi, bi, r] = np.corrcoef(obs_s[tr], pc)[0, 1]

        for pi in range(nP):
            for bi, bname in enumerate(baselines):
                base = pred_base[pi, bi, r]
                full = pred_full[pi, bi, r]
                if np.isnan(base).any() or np.isnan(full).any():
                    continue
                obs_base[pi, bi, 0, r] = logloss(label, base)
                obs_base[pi, bi, 1, r] = fast_auc(label, base)
                obs_delta[pi, bi, 0, r] = logloss(label, base) - logloss(label, full)
                obs_delta[pi, bi, 1, r] = fast_auc(label, full) - fast_auc(label, base)
        print(f"[{args.tag}] repeat {r+1}/{nR} elapsed {time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(
        out_path,
        obs_delta=obs_delta, obs_base=obs_base,
        pred_base=pred_base, pred_full=pred_full,
        k_used=k_used, base_var=base_var, corr_panel=corr_panel,
        panel_id=np.array([p["panel_id"] for p in panels]),
        baseline=np.array(baselines), metric=np.array(["logloss", "auc"]),
        y=label, patient=patient,
        seed=args.seed, n_repeats=nR, tag=args.tag, marker_set=args.marker_set,
    )
    print(f"saved {out_path}; elapsed {time.time()-t0:.0f}s")
    for bi, bname in enumerate(baselines):
        print(f"  {bname}: dAUC mean {np.nanmean(obs_delta[:, bi, 1, :]):.6f}, "
              f"dLL mean {np.nanmean(obs_delta[:, bi, 0, :]):.6f}")


if __name__ == "__main__":
    main()
