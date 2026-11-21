#!/usr/bin/env python3
"""Build FIXED random gene panels matched to each published panel.

Key properties (v3 correction):
  * one gene-level draw per published gene; identities stay fixed across all
    folds and all 20 repeats;
  * matching scale = full-cohort gene-level |t| (119 patients / 307 sites,
    representative probe = max |t|, ties lexicographic);
  * 20 quantile bins of |t|; a published gene's bin is matched exactly;
  * exclusions: all 27 published panel genes (original and mapped symbols),
    all single-cell marker genes used by any baseline, and genes already drawn
    into the same random panel;
  * the same representative-probe rule is applied later inside each training
    fold, exactly as for published panel genes.
"""
import argparse, gzip, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_base, panel_gene_sets


def load_markers(paths):
    genes = set()
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        col = ("gene_symbol" if "gene_symbol" in d.columns
               else "gene" if "gene" in d.columns else d.columns[0])
        genes |= set(d[col].dropna().astype(str))
    return genes


def build_panel(panel, gt, bins, forbidden, n_null, rng_seed, k_genes):
    """Return rows for n_null fixed random panels for one published panel."""
    abs_t = gt["abs_t_full"].values
    gene_names = gt["gene"].values
    bin_members = [np.where(gt["bin"].values == b)[0] for b in range(20)]
    rows = []
    n_fallback = 0
    for b in range(n_null):
        rng = np.random.default_rng(rng_seed + b)
        used = set()
        for pos, (src, sbin) in enumerate(zip(k_genes["gene"], k_genes["bin"])):
            chosen = None
            fallback = 0
            for step in range(0, 20):
                cand_bins = [sbin] if step == 0 else [sbin - step, sbin + step]
                for bb in cand_bins:
                    if bb < 0 or bb > 19:
                        continue
                    cand = bin_members[bb]
                    if len(cand) == 0:
                        continue
                    names = gene_names[cand]
                    mask = np.array([(g not in forbidden) and (g not in used) for g in names])
                    if mask.any():
                        pick = cand[mask][rng.integers(int(mask.sum()))]
                        chosen = int(pick)
                        fallback = step
                        break
                if chosen is not None:
                    break
            if chosen is None:
                raise RuntimeError(f"panel {panel['panel_id']} null {b}: no gene for bin {sbin}")
            g = gene_names[chosen]
            used.add(g)
            n_fallback += int(fallback > 0)
            rows.append({"panel_id": panel["panel_id"], "null_id": b + 1, "position": pos + 1,
                         "source_gene": src, "source_bin": int(sbin),
                         "null_gene": g, "null_abs_t_full": float(abs_t[chosen]),
                         "null_bin": int(gt["bin"].values[chosen]),
                         "fallback_steps": fallback})
    return rows, n_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--marker-files", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=20261031)
    args = ap.parse_args()

    X, probe_ids, patient, label, gsm, title, panels, g2p = load_base(args.data_dir)
    gt = pd.read_csv(os.path.join(args.out_dir, "results", "gene_t_full.csv"))
    bin_edges = json.load(open(os.path.join(args.out_dir, "results", "matching_bins.json")))["edges"]
    gt["bin"] = np.clip(np.digitize(gt["abs_t_full"].values, np.asarray(bin_edges)[1:-1]), 0, 19)

    marker_files = args.marker_files or [
        os.path.join(args.out_dir, "results", "markers_celltype_top_v3.csv"),
        os.path.join(args.out_dir, "results", "markers_williams_derived_top.csv"),
        os.path.join(args.out_dir, "results", "markers_williams_published.csv"),
    ]
    markers = load_markers(marker_files)
    sets = panel_gene_sets(panels)
    panel_forbidden = set()
    for p in panels:
        for m in p["mapped"]:
            panel_forbidden.add(m["symbol"])
            panel_forbidden.add(m["gene"])
    forbidden = panel_forbidden | markers
    print(f"forbidden genes: panel {len(panel_forbidden)}, markers {len(markers)}, union {len(forbidden)}")

    all_rows, summary = [], []
    for panel in panels:
        k = sets[panel["panel_id"]]
        gt_map = gt.set_index("gene")
        k_genes = {"gene": [], "bin": []}
        for g in k:
            if g not in gt_map.index:
                raise RuntimeError(f"published gene {g} missing from full-cohort t table")
            k_genes["gene"].append(g)
            k_genes["bin"].append(int(gt_map.loc[g, "bin"]))
        rows, nfb = build_panel(panel, gt, None, forbidden, args.n_null,
                                args.seed + 1000 * (panel["panel_id"] + 1), k_genes)
        all_rows.extend(rows)
        summary.append({"panel_id": panel["panel_id"], "pathway": panel["pathway"],
                        "k": len(k), "n_null": args.n_null, "n_fallback_draws": nfb})
        print(f"panel {panel['panel_id']:2d} k={len(k):2d} nulls={args.n_null} fallback={nfb}")

    ident = pd.DataFrame(all_rows)
    # every null panel must have exactly k unique genes
    for (pid, nid), g in ident.groupby(["panel_id", "null_id"]):
        k = len(sets[pid])
        if len(g) != k or g["null_gene"].nunique() != k:
            raise RuntimeError(f"panel {pid} null {nid}: expected {k} unique genes, got {len(g)}")
    ident.to_csv(os.path.join(args.out_dir, "results", "null_panels_identity.csv.gz"),
                 index=False, compression="gzip")
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "results", "null_panels_summary.csv"),
                                 index=False)

    # Seed-variant identity for panel 13, used only for a width-stability check.
    panel13 = [p for p in panels if p["panel_id"] == 13][0]
    k = sets[13]
    gt_map = gt.set_index("gene")
    k_genes = {"gene": k, "bin": [int(gt_map.loc[g, "bin"]) for g in k]}
    rows13, nfb13 = build_panel(panel13, gt, None, forbidden, 200, 777000, k_genes)
    pd.DataFrame(rows13).to_csv(os.path.join(args.out_dir, "results", "null_panels_seed777_panel13.csv.gz"),
                                index=False, compression="gzip")
    print(f"panel 13 seed-variant 200 nulls; fallback draws {nfb13}")

    # audit: number of distinct genes used across all null panels per published panel
    audit = ident.groupby("panel_id").agg(n_null=("null_id", "nunique"),
                                          n_rows=("null_gene", "size"),
                                          n_unique_genes=("null_gene", "nunique")).reset_index()
    audit.to_csv(os.path.join(args.out_dir, "results", "null_panels_audit.csv"), index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
