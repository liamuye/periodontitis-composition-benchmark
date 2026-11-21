#!/usr/bin/env python3
"""Step 2a: cohort checks, full-cohort gene-level |t| scale, matching bins and
the two patient-fold sets.  No panel model is fitted.

Scope of the matching scale is exactly v3: all 119 patients / 307 sites, both
labels, representative probe = largest |Welch t| (ties lexicographically
smallest probe id).  This scale is used only to match random panels; it never
enters the observed panel fitting.
"""
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_base, welch_t, make_folds, folds_to_frame


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42,20261031")
    ap.add_argument("--n-repeats", type=int, default=20)
    args = ap.parse_args()

    res = os.path.join(args.out_dir, "results")
    os.makedirs(res, exist_ok=True)
    X, probe_ids, patient, label, gsm, title, panels, g2p = load_base(args.data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}

    assert X.shape == (307, 54675), X.shape
    assert len(np.unique(patient)) == 119
    assert int(label.sum()) == 239 and int((1 - label).sum()) == 68
    assert not np.isin(72, patient), "patient 72 must be excluded"

    meta = pd.DataFrame({"site": np.arange(len(patient)), "patient": patient,
                         "label": label, "gsm": gsm, "title": title})
    meta.to_csv(os.path.join(res, "site_meta.csv"), index=False)

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
    gt.to_csv(os.path.join(res, "gene_t_full.csv"), index=False)
    json.dump({"scope": "GSE16134 all 119 patients / 307 sites, both labels",
               "n_genes": int(len(gt)), "edges": edges.tolist()},
              open(os.path.join(res, "matching_bins.json"), "w"), indent=2)
    print(f"gene t table: {len(gt)} genes; median |t| {np.median(abs_t):.3f}")

    for seed in [int(s) for s in args.seeds.split(",")]:
        reps = make_folds(patient, label, seed, args.n_repeats)
        fr = folds_to_frame(patient, seed, reps)
        fr.to_csv(os.path.join(res, f"folds_seed{seed}.csv"), index=False)
        for r in range(args.n_repeats):
            sub = fr[fr.repeat == r]
            assert len(sub) == len(patient)
            for f in range(5):
                te_p = set(sub[sub.fold == f].patient)
                tr_p = set(patient) - te_p
                assert not (te_p & tr_p)
        print(f"seed {seed}: folds ok")

    # Input hashes before any model fit.
    inputs = [
        os.path.join(args.out_dir, "00_protocol_confirmation.md"),
        os.path.join(args.out_dir, "01_panel_table.tsv"),
        os.path.join(args.out_dir, "01_panels_confirmation.json"),
        os.path.join(args.out_dir, "02_pathway_families.tsv"),
        os.path.join(args.out_dir, "03_shared_axis_reference.tsv"),
        os.path.join(args.out_dir, "SHA256SUMS_premodel.txt"),
        os.path.join(args.data_dir, "gse16134.npz"),
        os.path.join(args.data_dir, "panels.json"),
        os.path.join(args.data_dir, "gpl570_gene2probe.json"),
        os.path.join(args.out_dir, "..", "_v0.md"),
        os.path.join(args.out_dir, "..", "exploration_v3", "scripts", "common.py"),
        os.path.join(args.out_dir, "..", "exploration_v3", "scripts", "04_incremental_v3.py"),
        os.path.join(args.out_dir, "..", "exploration_v3", "results", "markers_celltype_top_v3.csv"),
        os.path.join(args.out_dir, "..", "exploration_v3", "results", "markers_williams_derived_top.csv"),
        os.path.join(args.out_dir, "..", "exploration_v3", "results", "markers_williams_published.csv"),
    ]
    pre = {}
    for p in inputs:
        pre[os.path.relpath(os.path.abspath(p), os.path.abspath(args.out_dir))] = (
            sha256(p) if os.path.exists(p) else "MISSING")
    json.dump(pre, open(os.path.join(res, "input_sha256_pre.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"hashed {len(pre)} inputs before fitting")


if __name__ == "__main__":
    main()
