#!/usr/bin/env python3
"""Step 2b: generate the fixed random panels for the 15 confirmation panels.

B = 20,000 per panel, identity seed 20261101, identical B for all panels.
One gene-level draw per published gene, matched on the full-cohort gene-level
|t| bin (20 quantile bins), excluding all 27 exploration-panel and 15
confirmation-panel symbols/aliases plus all single-cell marker genes.

Generator (recorded in the metadata):
    numpy.random.default_rng(
        numpy.random.SeedSequence([20261101, panel_id, null_id]))
with null_id = 1..20000.  The identity of every random panel is fixed across
all folds and all 20 repeats.

Outputs:
  results/null_panels_identity.csv.gz   readable one row per (panel,null,gene)
  results/null_panels_genes.json.gz     compact {panel_id: [[genes], ...]}
  results/null_panels_summary.csv
  results/null_panels_audit.csv
  results/null_panels_meta.json
"""
import argparse, gzip, hashlib, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--v3-dir", required=True)
    ap.add_argument("--panels-json", default="01_panels_confirmation.json")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--n-null", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()

    res = os.path.join(args.out_dir, "results")
    os.makedirs(res, exist_ok=True)
    panels = json.load(open(os.path.join(args.out_dir, args.panels_json)))
    exploration = json.load(open(os.path.join(args.data_dir, "panels.json")))
    gt = pd.read_csv(os.path.join(res, "gene_t_full.csv"))
    bins = json.load(open(os.path.join(res, "matching_bins.json")))["edges"]
    gt["bin"] = np.clip(np.digitize(gt["abs_t_full"].values, np.asarray(bins)[1:-1]), 0, 19)
    gene_names = gt["gene"].values
    abs_t = gt["abs_t_full"].values

    marker_files = [
        os.path.join(args.v3_dir, "results", "markers_celltype_top_v3.csv"),
        os.path.join(args.v3_dir, "results", "markers_williams_derived_top.csv"),
        os.path.join(args.v3_dir, "results", "markers_williams_published.csv"),
    ]
    markers = load_markers(marker_files)

    forbidden = set(markers)
    n_expl = n_conf = 0
    for p in exploration:
        for m in p["mapped"]:
            forbidden.add(m["symbol"]); forbidden.add(m["gene"]); n_expl += 1
    for p in panels:
        for m in p["mapped"]:
            forbidden.add(m["symbol"]); forbidden.add(m["gene"]); n_conf += 1
    print(f"forbidden: markers {len(markers)}, exploration entries {n_expl}, "
          f"confirmation entries {n_conf}, union {len(forbidden)}")

    eligible = []
    for b in range(20):
        idx = np.where((gt["bin"].values == b) & (~gt["gene"].isin(forbidden)))[0]
        eligible.append(idx)
        if len(idx) == 0:
            raise SystemExit(f"bin {b} has no eligible genes")

    gt_map = gt.set_index("gene")
    id_path = os.path.join(res, "null_panels_identity.csv.gz")
    json_path = os.path.join(res, "null_panels_genes.json.gz")
    summary, audit = [], []
    compact = {}
    t0 = time.time()
    with gzip.open(id_path, "wt", newline="") as fh:
        fh.write("panel_id\tnull_id\tposition\tsource_gene\tsource_bin\t"
                 "null_gene\tnull_abs_t_full\tnull_bin\tfallback_steps\n")
        for pi, panel in enumerate(panels):
            pid = int(panel["panel_id"])
            src = [m["symbol"] for m in panel["mapped"]]
            src_bins = [int(gt_map.loc[g, "bin"]) for g in src]
            panel_lists = []
            n_fb = 0
            n_rows = 0
            for b in range(1, args.n_null + 1):
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, pid, b]))
                used = set()
                genes = []
                for pos, (sg, sb) in enumerate(zip(src, src_bins)):
                    chosen = None
                    fb = 0
                    for step in range(0, 20):
                        cand_bins = [sb] if step == 0 else [sb - step, sb + step]
                        for bb in cand_bins:
                            if bb < 0 or bb > 19:
                                continue
                            cand = eligible[bb]
                            if len(cand) == 0:
                                continue
                            # rejection sampling for genes already used in this panel
                            for _ in range(1000):
                                j = int(cand[rng.integers(len(cand))])
                                g = gene_names[j]
                                if g not in used:
                                    chosen = j
                                    fb = step
                                    break
                            if chosen is not None:
                                break
                        if chosen is not None:
                            break
                    if chosen is None:
                        raise RuntimeError(f"panel {pid} null {b}: no gene for bin {sb}")
                    g = gene_names[chosen]
                    used.add(g); genes.append(g)
                    n_rows += 1
                    n_fb += int(fb > 0)
                    fh.write(f"{pid}\t{b}\t{pos+1}\t{sg}\t{sb}\t{g}\t"
                             f"{abs_t[chosen]:.10g}\t{int(gt['bin'].values[chosen])}\t{fb}\n")
                if len(genes) != len(src) or len(set(genes)) != len(src):
                    raise RuntimeError(f"panel {pid} null {b}: bad gene count")
                if any(g in forbidden for g in genes):
                    raise RuntimeError(f"panel {pid} null {b}: forbidden gene")
                panel_lists.append(genes)
            compact[str(pid)] = panel_lists
            summary.append({"panel_id": pid, "pathway": panel["pathway"],
                            "k": len(src), "n_null": args.n_null,
                            "n_fallback_draws": n_fb})
            audit.append({"panel_id": pid, "n_null": args.n_null, "n_rows": n_rows,
                          "n_unique_genes": len({g for gl in panel_lists for g in gl})})
            print(f"panel {pid:3d} k={len(src):2d} nulls={args.n_null} "
                  f"fallback={n_fb} elapsed={time.time()-t0:.0f}s", flush=True)

    with gzip.open(json_path, "wt") as fh:
        json.dump(compact, fh)

    pd.DataFrame(summary).to_csv(os.path.join(res, "null_panels_summary.csv"), index=False)
    pd.DataFrame(audit).to_csv(os.path.join(res, "null_panels_audit.csv"), index=False)

    meta = {
        "n_panels": len(panels),
        "n_null_per_panel": args.n_null,
        "identity_seed": args.seed,
        "generator": "numpy.random.default_rng(numpy.random.SeedSequence([seed, panel_id, null_id]))",
        "numpy_version": np.__version__,
        "numpy_rng_bit_generator": "PCG64",
        "exclusion": {"marker_genes": len(markers),
                      "exploration_panel_entries": n_expl,
                      "confirmation_panel_entries": n_conf,
                      "union": len(forbidden)},
        "k_by_panel": {str(p["panel_id"]): len(p["mapped"]) for p in panels},
        "sha256_identity_csv_gz": sha256(id_path),
        "sha256_genes_json_gz": sha256(json_path),
        "source_gene_t_sha256": sha256(os.path.join(res, "gene_t_full.csv")),
        "source_matching_bins_sha256": sha256(os.path.join(res, "matching_bins.json")),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    json.dump(meta, open(os.path.join(res, "null_panels_meta.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nsummary:")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nidentity: {id_path}  sha256={meta['sha256_identity_csv_gz']}")
    print(f"compact : {json_path}  sha256={meta['sha256_genes_json_gz']}")


if __name__ == "__main__":
    main()
