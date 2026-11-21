#!/usr/bin/env python3
"""Step 3b: fixed-random-panel null distributions, chunked and resumable.

For every confirmation panel and every fixed random panel, the random panel
goes through the same 20x5 outer folds, training-fold representative probe,
direction rule, baselines, models and metrics as the observed panel.  The
statistic is the 20-repeat mean; it is accumulated in float64.

Chunking: null panels 1..B are processed in blocks of --chunk-size (default
5000).  A chunk file is written only after all 20 repeats are complete; on
restart, completed chunks are skipped.  No per-repeat CSV is written.

Outputs:
  results/chunks_<tag>/panel_<pid>_chunk_<start>_<end>.npz
  results/null_means_panel_<tag>_<pid>.npz
  results/null_means_<tag>.npz
  results/null_meta_<tag>.json
"""
import argparse, gzip, hashlib, json, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_base, load_folds, fold_stats, FoldData, comp_features,
                    panel_symbol_alias, scale_train, scale_train_batch, pca1,
                    fit_logistic_batch, predict_logistic_batch, logloss,
                    logloss_batch, fast_auc, fast_auc_batch, CELLTYPES)

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
        cols.append(pos); dirs.append(d)
    if len(cols) < 3:
        return None
    M = fd.Z[:, cols] * np.asarray(dirs, dtype=float)[None, :]
    pc_tr, pc_te, var, load = pca1(M[fd.tr], M[fd.te])
    if np.mean(load) < 0:
        pc_tr, pc_te, load = -pc_tr, -pc_te, -load
    return {"pc_tr": pc_tr, "pc_te": pc_te, "var": var, "load": load, "k": len(cols)}


def null_score_matrix(fd, genes_per_null):
    rows, cols, vals = [], [], []
    for b, genes in enumerate(genes_per_null):
        pos = [fd.gene_pos[g] for g in genes if g in fd.gene_pos]
        k = len(pos)
        if k == 0:
            continue
        sgn = np.array([1.0 if fd.gene_t[fd.genes[p]] >= 0 else -1.0 for p in pos])
        for p, s in zip(pos, sgn):
            rows.append(p); cols.append(b); vals.append(s / k)
    W = csr_matrix((np.asarray(vals, dtype=float), (rows, cols)),
                   shape=(len(fd.genes), len(genes_per_null)))
    return np.asarray(fd.Z @ W)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_markers(path, marker_set):
    d = pd.read_csv(path)
    if marker_set == "primary":
        return {ct: d[d.celltype == ct].sort_values("rank").gene_symbol.tolist()
                for ct in CELLTYPES}
    return {ct: g.sort_values("rank").gene_symbol.tolist()
            for ct, g in d.groupby("celltype")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--v3-dir", required=True)
    ap.add_argument("--panels-json", default="01_panels_confirmation.json")
    ap.add_argument("--identity-json", default="")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--marker-set", choices=["primary", "williams"], default="primary")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--n-null", type=int, default=20000)
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--panels", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    res = os.path.join(args.out_dir, "results")
    chunk_dir = os.path.join(res, f"chunks_{args.tag}")
    os.makedirs(chunk_dir, exist_ok=True)
    panels = json.load(open(os.path.join(args.out_dir, args.panels_json)))
    exploration = json.load(open(os.path.join(args.data_dir, "panels.json")))
    if args.panels:
        keep = {int(x) for x in args.panels.split(",") if x.strip()}
        panels = [p for p in panels if int(p["panel_id"]) in keep]
    nP = len(panels)
    if nP == 0:
        raise SystemExit("no panels selected")

    X, probe_ids, patient, label, gsm, title, _, g2p = load_base(args.data_dir)
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    nS = len(label)
    nR = args.n_repeats
    alias = panel_symbol_alias(panels)

    if args.marker_set == "primary":
        marker_path = os.path.join(args.v3_dir, "results", "markers_celltype_top_v3.csv")
        baselines = ["comp1d", "compfull", "shared"]
    else:
        marker_path = os.path.join(args.v3_dir, "results", "markers_williams_derived_top.csv")
        baselines = ["williams_derived_1d", "williams_derived_full"]
    marker_genes = load_markers(marker_path, args.marker_set)
    nB = len(baselines)

    obs = np.load(os.path.join(res, f"obs_predictions_{args.tag}.npz"), allow_pickle=True)
    obs_base = obs["obs_delta"]
    pred_base = obs["pred_base"]
    if list(obs["baseline"]) != baselines:
        raise SystemExit(f"baseline mismatch: {list(obs['baseline'])} vs {baselines}")
    if pred_base.shape[0] != len(json.load(open(os.path.join(args.out_dir, args.panels_json)))):
        pass

    with gzip.open(os.path.join(res, "null_panels_genes.json.gz"), "rt") as fh:
        ident = json.load(fh)
    if args.identity_json:
        if args.identity_json.endswith(".gz"):
            fh = gzip.open(args.identity_json, "rt")
        else:
            fh = open(args.identity_json, "rt")
        with fh:
            ident = json.load(fh)
    folds = load_folds(os.path.join(res, f"folds_seed{args.seed}.csv"), nR, nS)

    base_genes = set()
    for p in panels:
        base_genes |= {m["symbol"] for m in p["mapped"]}
    for v in marker_genes.values():
        base_genes |= set(v)
    if args.marker_set == "primary":
        for p in exploration:
            base_genes |= {m["symbol"] for m in p["mapped"]}
    print(f"[{args.tag}] panels={nP} baselines={baselines} B={args.n_null} "
          f"chunk={args.chunk_size} base_genes={len(base_genes)}")

    t0 = time.time()
    stats_cache = {}
    combined = {}
    for pi, panel in enumerate(panels):
        pid = int(panel["panel_id"])
        all_genes = ident[str(pid)][: args.n_null]
        if len(all_genes) != args.n_null:
            raise SystemExit(f"panel {pid}: identity has {len(all_genes)} nulls, need {args.n_null}")
        pi_global = list(obs["panel_id"]).index(pid)
        pieces = []
        for start in range(0, args.n_null, args.chunk_size):
            end = min(start + args.chunk_size, args.n_null)
            chunk_path = os.path.join(chunk_dir, f"panel_{pid}_chunk_{start}_{end}.npz")
            if os.path.exists(chunk_path) and not args.force:
                d = np.load(chunk_path, allow_pickle=True)
                if bool(d["done"]) and int(d["n_null"]) == end - start:
                    pieces.append(d["means"])
                    print(f"  panel {pid} chunk {start}-{end} exists, skip", flush=True)
                    continue
            chunk_genes = all_genes[start:end]
            nchunk = len(chunk_genes)
            sums = np.zeros((nB, 2, nchunk), dtype=np.float64)
            for r in range(nR):
                # complete 307-site holdout prediction for every null panel, filled across the 5 folds
                rep_pred = {b: np.full((nS, nchunk), np.nan, dtype=np.float64)
                            for b in baselines}
                for f, (tr, te) in enumerate(folds[r]):
                    key = (r, f)
                    if key not in stats_cache:
                        stats_cache[key] = fold_stats(X, label, tr)
                    genes = base_genes | {g for gl in chunk_genes for g in gl}
                    fd = FoldData(X, probe_ids, probe_idx, g2p, sorted(genes), label,
                                  tr, te, stats=stats_cache[key])
                    y_tr = label[tr]
                    comp = comp_features(fd, marker_genes, alias[pid])
                    shared = shared_features(fd, panel, exploration) if args.marker_set == "primary" else None

                    b1_tr, b1_te, _, _ = scale_train(comp["pc1_tr"], comp["pc1_te"])
                    designs = {}
                    if args.marker_set == "primary":
                        designs["comp1d"] = (b1_tr[:, None], b1_te[:, None])
                        designs["compfull"] = (comp["full_tr"], comp["full_te"])
                        if shared is not None:
                            bs_tr, bs_te, _, _ = scale_train(shared["pc_tr"], shared["pc_te"])
                            designs["shared"] = (bs_tr[:, None], bs_te[:, None])
                    else:
                        designs["williams_derived_1d"] = (b1_tr[:, None], b1_te[:, None])
                        designs["williams_derived_full"] = (comp["full_tr"], comp["full_te"])

                    S = null_score_matrix(fd, chunk_genes)
                    S_tr, S_te, _, _ = scale_train_batch(S[tr], S[te])
                    for bi, bname in enumerate(baselines):
                        if bname not in designs:
                            raise RuntimeError(f"panel {pid} baseline {bname} repeat {r}: "
                                               f"design unavailable (shared axis too small)")
                        Btr, Bte = designs[bname]
                        coef = fit_logistic_batch(Btr, S_tr, y_tr)
                        rep_pred[bname][te, :] = predict_logistic_batch(coef, Bte, S_te)

                # one delta per repeat, on the full 307-site holdout (same definition as 03_observed)
                for bi, bname in enumerate(baselines):
                    P_rep = rep_pred[bname]
                    if np.isnan(P_rep).any():
                        n_bad = int(np.isnan(P_rep).any(axis=0).sum())
                        raise RuntimeError(f"panel {pid} baseline {bname} repeat {r}: "
                                           f"{n_bad}/{nchunk} null panels have incomplete "
                                           f"holdout predictions")
                    base = pred_base[pi_global, bi, r]
                    if np.isnan(base).any():
                        raise RuntimeError(f"panel {pid} baseline {bname} repeat {r}: "
                                           f"observed base prediction has NaN")
                    sums[bi, 0] += logloss(label, base) - logloss_batch(label, P_rep)
                    sums[bi, 1] += fast_auc_batch(label, P_rep) - fast_auc(label, base)
                print(f"  panel {pid} chunk {start}-{end} repeat {r+1}/{nR} "
                      f"elapsed {time.time()-t0:.0f}s", flush=True)
            means = sums / float(nR)
            np.savez_compressed(chunk_path,
                                means=means, sums=sums, n_repeats=nR, n_null=nchunk,
                                start=start, end=end, panel_id=pid,
                                baseline=np.array(baselines), done=True)
            pieces.append(means)
        panel_means = np.concatenate(pieces, axis=2)
        if panel_means.shape != (nB, 2, args.n_null):
            raise RuntimeError(f"panel {pid}: assembled shape {panel_means.shape}")
        np.savez_compressed(os.path.join(res, f"null_means_panel_{args.tag}_{pid}.npz"),
                            mean=panel_means, panel_id=pid,
                            baseline=np.array(baselines), metric=np.array(["logloss", "auc"]))
        combined[pid] = panel_means
        print(f"panel {pid} done; means {panel_means.shape}; elapsed {time.time()-t0:.0f}s",
              flush=True)

    pids = [int(p["panel_id"]) for p in panels]
    arr = np.stack([combined[p] for p in pids], axis=0)
    out = os.path.join(res, f"null_means_{args.tag}.npz")
    np.savez_compressed(out, mean=arr, panel_id=np.array(pids),
                        baseline=np.array(baselines), metric=np.array(["logloss", "auc"]),
                        n_null=args.n_null, tag=args.tag)
    meta = {
        "tag": args.tag, "n_panels": len(pids), "panel_ids": pids,
        "baselines": baselines, "n_null": args.n_null, "chunk_size": args.chunk_size,
        "n_repeats": nR, "seed": args.seed, "marker_set": args.marker_set,
        "shape": list(arr.shape), "elapsed_sec": round(time.time() - t0, 1),
        "sha256_null_means": sha256(out),
        "identity_sha256": sha256(os.path.join(res, "null_panels_genes.json.gz")),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    json.dump(meta, open(os.path.join(res, f"null_meta_{args.tag}.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"saved {out} shape={arr.shape}; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
