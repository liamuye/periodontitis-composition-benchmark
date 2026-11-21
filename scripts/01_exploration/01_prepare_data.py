#!/usr/bin/env python3
"""Prepare v3 inputs: cohort checks, full-cohort |t| matching scale, folds.

The fixed random-panel design needs one gene-level |t| per gene that does not
change across folds or repeats.  Scope used here (stated in the protocol):
all 119 patients / 307 sites of GSE16134, both labels, representative probe =
largest |Welch t| among valid probes (ties: lexicographically smallest probe).
This is the matching scale only; it never enters the observed panel fitting.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_base, welch_t, make_folds, folds_to_frame, panel_gene_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42,20261031")
    ap.add_argument("--n-repeats", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    X, probe_ids, patient, label, gsm, title, panels, g2p = load_base(args.data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}

    assert X.shape == (307, 54675), X.shape
    assert len(np.unique(patient)) == 119
    assert int(label.sum()) == 239 and int((1 - label).sum()) == 68
    assert not np.isin(72, patient), "patient 72 must be excluded"

    meta = pd.DataFrame({"site": np.arange(len(patient)), "patient": patient,
                         "label": label, "gsm": gsm, "title": title})
    meta.to_csv(os.path.join(args.out_dir, "results", "site_meta.csv"), index=False)

    # Full-cohort representative probe and gene-level |t|.
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, ddof=1, dtype=np.float64)
    valid = np.isfinite(sd) & (sd > 1e-6)
    t = welch_t(X, label)
    rows = []
    for g, ps in g2p.items():
        best = None
        best_key = None
        for p in ps:
            j = probe_idx.get(p)
            if j is None or not valid[j]:
                continue
            key = (-abs(t[j]), p)
            if best_key is None or key < best_key:
                best_key = key
                best = j
        if best is None:
            continue
        rows.append({"gene": g, "probe": probe_ids[best], "t_full": float(t[best]),
                     "abs_t_full": float(abs(t[best]))})
    gt = pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)
    abs_t = gt["abs_t_full"].values
    edges = np.quantile(abs_t, np.linspace(0, 1, 21))
    gt["bin"] = np.clip(np.digitize(abs_t, edges[1:-1]), 0, 19)
    gt.to_csv(os.path.join(args.out_dir, "results", "gene_t_full.csv"), index=False)
    json.dump({"scope": "GSE16134 all 119 patients / 307 sites, both labels",
               "n_genes": int(len(gt)), "edges": edges.tolist()},
              open(os.path.join(args.out_dir, "results", "matching_bins.json"), "w"), indent=2)

    # Folds for the primary and comparison seeds.
    for seed in [int(s) for s in args.seeds.split(",")]:
        reps = make_folds(patient, label, seed, args.n_repeats)
        fr = folds_to_frame(patient, seed, reps)
        fr.to_csv(os.path.join(args.out_dir, "results", f"folds_seed{seed}.csv"), index=False)
        # sanity: every site predicted once per repeat, no patient overlap
        for r in range(args.n_repeats):
            sub = fr[fr.repeat == r]
            assert len(sub) == len(patient)
            for f in range(5):
                te_p = set(sub[sub.fold == f].patient)
                tr_p = set(patient) - te_p
                assert not (te_p & tr_p)
        print(f"seed {seed}: folds ok")


    sets = panel_gene_sets(panels)
    print("panels", len(panels), "unique mapped genes", len({g for v in sets.values() for g in v}))
    print(f"gene t table: {len(gt)} genes; bins 0-19; median |t| {np.median(abs_t):.3f}")


if __name__ == "__main__":
    main()
