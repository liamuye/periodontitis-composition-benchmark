#!/usr/bin/env python3
"""Additional sensitivity analyses on GSE16134:
1. Multivariate gene-level L2 logistic regression (p genes entered directly vs equal-weighted score),
   benchmarked against B=1,000 fixed-identity |t|-matched random p-gene panels entered directly.
2. L2 penalty sensitivity (lambda = 0.1 and lambda = 10.0 vs primary lambda = 1.0).
3. Label-free representative probe selection (highest training-fold mean expression vs max |Welch t|).
4. Panel 38 dissection: train-on-overlap (89 patients, 241 biopsies) -> test-on-unique (30 patients, 66 biopsies),
   leave-one-gene-out ablation on 5D baseline, and per-gene correlation with single-cell compartments.
"""
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

ROOT = "."
CACHE = os.path.join(ROOT, "_cache_gse16134")
V3S = os.path.join(ROOT, "exploration_v3", "scripts")
OUT = os.path.join(ROOT, "results", "multivariate_sens")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, V3S)

from common import (
    make_folds, FoldData, comp_features, scale_train, scale_train_batch,
    fit_logistic, predict_logistic, fit_logistic_batch, predict_logistic_batch,
    logloss, logloss_batch, fast_auc, fast_auc_batch, bh_q, CELLTYPES, EPS
)


def build_panels_and_load():
    D = np.load(os.path.join(CACHE, "gse16134.npz"), allow_pickle=True)
    X = D["X"].astype(np.float32)
    probe_ids = np.asarray(D["probe_ids"]).astype(str)
    patient = np.asarray(D["patient"]).astype(int)
    label = np.asarray(D["label"]).astype(int)
    gsm = np.asarray(D["gsm"]).astype(str)
    g2p = json.load(open(os.path.join(CACHE, "gpl570_gene2probe.json")))

    # Reconstruct 27 exploration panels from exact v2 panels_resolved.csv
    res2 = pd.read_csv(os.path.join(ROOT, "exploration_v2", "results", "panels_resolved.csv"))
    exp_panels = []
    for pid, sub in res2.groupby("panel_id", sort=True):
        mapped = []
        for _, r in sub.iterrows():
            mapped.append({
                "gene": str(r["gene"]),
                "symbol": str(r["mapped_symbol"]),
                "direction_published": int(r["direction_published"]),
            })
        exp_panels.append({
            "panel_id": int(pid),
            "pmid": str(sub["pmid"].iloc[0]),
            "cohort": "exploration",
            "mapped": mapped,
        })

    conf_raw = json.load(open(os.path.join(ROOT, "confirmation_v1", "01_panels_confirmation.json")))
    conf_panels = []
    for p in conf_raw:
        conf_panels.append({
            "panel_id": int(p["panel_id"]),
            "pmid": str(p["pmid"]),
            "cohort": "confirmation",
            "mapped": p["mapped"],
        })

    panels = exp_panels + conf_panels
    mk = pd.read_csv(os.path.join(ROOT, "exploration_v3", "results", "markers_celltype_top_v3.csv"))
    marker_genes = {ct: mk[mk.celltype == ct].sort_values("rank").gene_symbol.tolist() for ct in CELLTYPES}
    return X, probe_ids, patient, label, gsm, g2p, panels, marker_genes


def load_identities(nB=1000):
    ident = {}
    for path, sep, pids in [
        (os.path.join(ROOT, "exploration_v3", "results", "null_panels_identity.csv.gz"), ",", set(range(1, 28))),
        (os.path.join(ROOT, "confirmation_v1", "results", "null_panels_identity.csv.gz"), "\t", set(range(101, 116))),
    ]:
        df = pd.read_csv(path, sep=sep)
        df = df[df.panel_id.isin(pids)]
        for pid, sub in df.groupby("panel_id"):
            by = sub.sort_values(["null_id", "position"]).groupby("null_id")["null_gene"].apply(list)
            ident[int(pid)] = [by[k] for k in sorted(by.index)][:nB]
    return ident


def fit_logistic_multi_batch(B, G_batch, y, lam=1.0, max_iter=18, tol=1e-6):
    """Vectorized BLAS-accelerated Newton-Raphson L2 logistic regression for nb models simultaneously.
    B: (n, q) scaled baseline predictors
    G_batch: (nb, n, p) scaled panel gene predictors
    y: (n,) binary labels
    Returns coef: (nb, 1 + q + p)
    """
    n, q = B.shape
    nb, _, p = G_batch.shape
    k = 1 + q + p
    X_base = np.column_stack([np.ones(n), B])  # (n, 1+q)
    X_all = np.empty((nb, n, k), dtype=np.float64)
    X_all[:, :, :1 + q] = X_base[None, :, :]
    X_all[:, :, 1 + q:] = G_batch
    XT = np.ascontiguousarray(X_all.transpose(0, 2, 1))  # (nb, k, n)

    coef = np.zeros((nb, k), dtype=np.float64)
    p0 = np.clip(np.mean(y), EPS, 1 - EPS)
    coef[:, 0] = np.log(p0 / (1.0 - p0))
    eye_pen = np.eye(k, dtype=np.float64)
    eye_pen[0, 0] = 0.0
    reg = (2.0 * lam * eye_pen)[None, :, :]

    for _ in range(max_iter):
        eta = np.matmul(X_all, coef[:, :, None])[:, :, 0]
        np.clip(eta, -30.0, 30.0, out=eta)
        pr = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(pr * (1.0 - pr), 1e-6, None)
        r = y[None, :] - pr
        g = np.matmul(XT, r[:, :, None])[:, :, 0]
        g[:, 1:] -= 2.0 * lam * coef[:, 1:]
        Xw = X_all * w[:, :, None]
        H = np.matmul(XT, Xw) + reg
        try:
            step = np.linalg.solve(H, g[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = np.empty_like(g)
            for b in range(nb):
                step[b] = np.linalg.lstsq(H[b] + 1e-6 * np.eye(k), g[b], rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        coef += step
        if np.max(np.abs(step)) < tol:
            break
    return coef


def predict_logistic_multi_batch(coef, B_te, G_te_batch):
    n_te, q = B_te.shape
    X_base = np.column_stack([np.ones(n_te), B_te])
    eta = (
        X_base @ coef[:, :1 + q].T
        + np.matmul(G_te_batch, coef[:, 1 + q:, None])[:, :, 0].T
    )
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


class FoldDataMeanProbe(FoldData):
    """Same as FoldData but selects the representative probe with highest training-fold mean expression (label-free)."""
    def __init__(self, X, probe_ids, probe_idx, g2p, genes, label, tr, te):
        self.tr = np.asarray(tr, dtype=int)
        self.te = np.asarray(te, dtype=int)
        self.y_tr = label[self.tr]
        self.y_te = label[self.te]
        mu = X[self.tr].mean(axis=0, dtype=np.float64)
        sd = X[self.tr].std(axis=0, ddof=1, dtype=np.float64)
        self.mu = mu
        self.sd = sd
        self.valid = np.isfinite(sd) & (sd > 1e-6)
        from common import welch_t
        self.t = welch_t(X[self.tr], self.y_tr)
        self.abs_t = np.abs(self.t)

        genes = list(dict.fromkeys(genes))
        self.genes = genes
        self.gene_pos = {}
        self.gene_probe = {}
        self.gene_t = {}
        cols = []
        for g in genes:
            ps = g2p.get(g, [])
            best = None
            best_key = None
            for p in ps:
                j = probe_idx.get(p)
                if j is None or not self.valid[j]:
                    continue
                key = (-self.mu[j], p)  # highest training-fold mean expression (label-free)
                if best_key is None or key < best_key:
                    best_key = key
                    best = j
            if best is None:
                continue
            self.gene_pos[g] = len(cols)
            self.gene_probe[g] = probe_ids[best]
            self.gene_t[g] = float(self.t[best])
            cols.append(best)
        self.cols = np.asarray(cols, dtype=int)
        Z = X[:, self.cols].astype(np.float64)
        Z = (Z - mu[self.cols]) / sd[self.cols]
        self.Z = Z.astype(np.float32)
        self.n = X.shape[0]


def null_score_matrix(fd, genes_per_null):
    rows, cols, vals = [], [], []
    inv = {v: k for k, v in fd.gene_pos.items()}
    for b, genes in enumerate(genes_per_null):
        pos = [fd.gene_pos[g] for g in genes if g in fd.gene_pos]
        k = len(pos)
        if k == 0:
            continue
        sgn = np.array([1.0 if fd.gene_t[inv[p]] >= 0 else -1.0 for p in pos])
        for p, s in zip(pos, sgn):
            rows.append(p); cols.append(b); vals.append(s / k)
    W = csr_matrix((np.asarray(vals, float), (rows, cols)), shape=(fd.Z.shape[1], len(genes_per_null)))
    return np.asarray(fd.Z @ W)


def summarize_results(obs_delta, null_delta, panels, tag):
    """obs_delta: (nP, 2, 2, nR), null_delta: (nP, 2, 2, nB, nR)"""
    bnames = ["comp1d", "compfull"]
    mnames = ["logloss", "auc"]
    obs_mean = obs_delta.mean(axis=3)      # (nP, 2, 2)
    null_mean = null_delta.mean(axis=4)    # (nP, 2, 2, nB)
    nB = null_mean.shape[3]
    rows = []
    for pi, p in enumerate(panels):
        for bi, b in enumerate(bnames):
            for mi, m in enumerate(mnames):
                om = float(obs_mean[pi, bi, mi])
                nm = null_mean[pi, bi, mi]
                k = int(np.sum(nm >= om))
                pval = (1.0 + k) / (1.0 + nB)
                rows.append({
                    "tag": tag,
                    "panel_id": p["panel_id"],
                    "cohort": p["cohort"],
                    "baseline": b,
                    "metric": m,
                    "observed_mean": om,
                    "null_median": float(np.median(nm)),
                    "null95": float(np.percentile(nm, 95)),
                    "k": k,
                    "B": nB,
                    "p": pval,
                })
    df = pd.DataFrame(rows)
    df["q"] = np.nan
    for (coh, b, m), idx in df.groupby(["cohort", "baseline", "metric"]).groups.items():
        df.loc[idx, "q"] = bh_q(df.loc[idx, "p"].values)
    df["pass_q"] = (df["q"] < 0.05) & (df["observed_mean"] > df["null95"])
    df.to_csv(os.path.join(OUT, f"results_{tag}.csv"), index=False)
    return df


def print_summary(df, label):
    print(f"\n==================== {label} ====================")
    for coh in ["exploration", "confirmation"]:
        sub = df[df.cohort == coh]
        n_tot = sub.panel_id.nunique()
        parts = []
        for b in ["comp1d", "compfull"]:
            sb = sub[sub.baseline == b]
            pos_ll = sorted(sb[(sb.metric == "logloss") & sb.pass_q].panel_id.tolist())
            pos_au = sorted(sb[(sb.metric == "auc") & sb.pass_q].panel_id.tolist())
            pos_any = sorted(set(pos_ll) | set(pos_au))
            med_auc = sb[sb.metric == "auc"]["observed_mean"].median()
            med_ll = sb[sb.metric == "logloss"]["observed_mean"].median()
            disp = [x - 73 if x >= 101 else x for x in pos_any]
            parts.append(f"{b}: {len(pos_any)}/{n_tot} {disp} (med dLL={med_ll:.4f}, dAUC={med_auc:.4f})")
        print(f"  {coh:12s} | " + " || ".join(parts))


def run_panel38_dissection(X, probe_ids, probe_idx, patient, label, gsm, g2p, panels, marker_genes):
    print("\n==================== Panel 38 Dissection ====================")
    ov = pd.read_csv(os.path.join(ROOT, "overlap_audit", "GSE10334_GSE16134_identical_arrays.tsv"), sep="\t")
    ov_gsms = set(ov["GSE16134_GSM"].astype(str))
    is_ov = np.array([g in ov_gsms for g in gsm], dtype=bool)
    tr = np.where(is_ov)[0]
    te = np.where(~is_ov)[0]
    # Verify patient disjointness between overlap and unique
    ov_pats = set(patient[tr])
    unq_pats = set(patient[te])
    print(f"Overlap patients={len(ov_pats)} (biopsies={len(tr)}), Unique patients={len(unq_pats)} (biopsies={len(te)}), shared patients={len(ov_pats & unq_pats)}")

    # Gather all needed genes
    genes_needed = set()
    for ct in CELLTYPES:
        genes_needed.update(marker_genes[ct])
    for p in panels:
        for m in p["mapped"]:
            genes_needed.add(m["symbol"])
    genes_needed = sorted(genes_needed)

    # 1. Train on 89 overlapping patients -> test on 30 unique patients (all 15 confirmation panels)
    fd = FoldData(X, probe_ids, probe_idx, g2p, genes_needed, label, tr, te)
    y_tr, y_te = label[tr], label[te]
    dissect_rows = []
    for p in [x for x in panels if x["cohort"] == "confirmation"]:
        alias = {m["symbol"] for m in p["mapped"]} | {m["gene"] for m in p["mapped"]}
        comp = comp_features(fd, marker_genes, alias)
        pairs = [(m["symbol"], m["direction_published"]) for m in p["mapped"]]
        s, _, _, _ = fd.score(pairs)
        s_tr, s_te, _, _ = scale_train(s[tr], s[te])
        b1_tr, b1_te, _, _ = scale_train(comp["pc1_tr"], comp["pc1_te"])
        for bname, (B_tr, B_te) in [("comp1d", (b1_tr[:, None], b1_te[:, None])), ("compfull", (comp["full_tr"], comp["full_te"]))]:
            wb = fit_logistic(B_tr, y_tr, lam=1.0)
            pb = predict_logistic(wb, B_te)
            wf = fit_logistic(np.column_stack([B_tr, s_tr]), y_tr, lam=1.0)
            pf = predict_logistic(wf, np.column_stack([B_te, s_te]))
            dissect_rows.append({
                "panel_display": p["panel_id"] - 73,
                "baseline": bname,
                "base_ll": logloss(y_te, pb),
                "full_ll": logloss(y_te, pf),
                "dLL_unique": logloss(y_te, pb) - logloss(y_te, pf),
                "base_auc": fast_auc(y_te, pb),
                "full_auc": fast_auc(y_te, pf),
                "dAUC_unique": fast_auc(y_te, pf) - fast_auc(y_te, pb),
            })
    ddf = pd.DataFrame(dissect_rows)
    ddf.to_csv(os.path.join(OUT, "train_overlap_test_unique_conf15.csv"), index=False)
    print("Train-on-overlap (89 pts, 241 biopsies) -> Test-on-unique (30 pts, 66 biopsies):")
    for bname in ["comp1d", "compfull"]:
        sub = ddf[ddf.baseline == bname]
        p38 = sub[sub.panel_display == 38].iloc[0]
        others = sub[sub.panel_display != 38]
        print(f"  {bname}: Panel 38 dLL={p38.dLL_unique:+.4f}, dAUC={p38.dAUC_unique:+.4f} (base AUC={p38.base_auc:.4f}->full {p38.full_auc:.4f}) | other 14 median dLL={others.dLL_unique.median():+.4f}, dAUC={others.dAUC_unique.median():+.4f}")

    # 2. Panel 38 leave-one-gene-out and single-gene ablation on full 20x5 CV (5D baseline)
    p38 = [x for x in panels if x["panel_id"] == 111][0]
    folds = make_folds(patient, label, 42, 20, 5)
    p38_alias = {m["symbol"] for m in p38["mapped"]} | {m["gene"] for m in p38["mapped"]}
    p38_pairs = [(m["symbol"], m["direction_published"]) for m in p38["mapped"]]
    p38_syms = [m["symbol"] for m in p38["mapped"]]
    # Precompute 20x5 FoldData on needed genes only (super fast!)
    nS = len(label)
    logo_ll = {g: [] for g in ["FULL"] + [f"drop_{s}" for s in p38_syms] + [f"only_{s}" for s in p38_syms]}
    logo_auc = {g: [] for g in logo_ll}
    # Also full-cohort correlation of each gene with 1D & 5 compartments
    fd_all = FoldData(X, probe_ids, probe_idx, g2p, genes_needed, label, np.arange(nS), np.arange(nS))
    comp_all = comp_features(fd_all, marker_genes, p38_alias)
    corr_rows = []
    for sym, d in p38_pairs:
        sgn = d if d != 0 else 1
        g_vec = fd_all.Z[:, fd_all.gene_pos[sym]].astype(np.float64) * sgn
        row = {"gene": sym, "dir": d, "r_1D": float(np.corrcoef(g_vec, comp_all["pc1_tr"])[0, 1])}
        for ci, ct in enumerate(CELLTYPES):
            row[f"r_{ct}"] = float(np.corrcoef(g_vec, comp_all["full_tr"][:, ci])[0, 1])
        corr_rows.append(row)

    for r in range(20):
        pb_all = np.zeros(nS)
        pf_dict = {k: np.zeros(nS) for k in logo_ll}
        for f, (ftr, fte) in enumerate(folds[r]):
            fdf = FoldData(X, probe_ids, probe_idx, g2p, genes_needed, label, ftr, fte)
            ytr = label[ftr]
            comp = comp_features(fdf, marker_genes, p38_alias)
            Btr, Bte = comp["full_tr"], comp["full_te"]
            wb = fit_logistic(Btr, ytr, lam=1.0)
            pb_all[fte] = predict_logistic(wb, Bte)
            variants = {"FULL": p38_pairs}
            for s in p38_syms:
                variants[f"drop_{s}"] = [(g, d) for g, d in p38_pairs if g != s]
                variants[f"only_{s}"] = [(g, d) for g, d in p38_pairs if g == s]
            for name, vpairs in variants.items():
                sc, _, _, _ = fdf.score(vpairs)
                str_, ste_, _, _ = scale_train(sc[ftr], sc[fte])
                wf = fit_logistic(np.column_stack([Btr, str_]), ytr, lam=1.0)
                pf_dict[name][fte] = predict_logistic(wf, np.column_stack([Bte, ste_]))
        base_ll_r = logloss(label, pb_all)
        base_auc_r = fast_auc(label, pb_all)
        for name in logo_ll:
            logo_ll[name].append(base_ll_r - logloss(label, pf_dict[name]))
            logo_auc[name].append(fast_auc(label, pf_dict[name]) - base_auc_r)

    ab_rows = []
    full_ll = float(np.mean(logo_ll["FULL"]))
    full_auc = float(np.mean(logo_auc["FULL"]))
    print(f"Panel 38 5D FULL: dLL={full_ll:.4f}, dAUC={full_auc:.4f}")
    for c_row in corr_rows:
        s = c_row["gene"]
        d_ll = float(np.mean(logo_ll[f"drop_{s}"]))
        d_au = float(np.mean(logo_auc[f"drop_{s}"]))
        o_ll = float(np.mean(logo_ll[f"only_{s}"]))
        o_au = float(np.mean(logo_auc[f"only_{s}"]))
        c_row.update({
            "drop_dLL": d_ll, "loss_on_drop_dLL": full_ll - d_ll,
            "drop_dAUC": d_au, "loss_on_drop_dAUC": full_auc - d_au,
            "only_dLL": o_ll, "only_dAUC": o_au,
        })
        ab_rows.append(c_row)
    ab_df = pd.DataFrame(ab_rows).sort_values("loss_on_drop_dLL", ascending=False)
    ab_df.to_csv(os.path.join(OUT, "panel38_gene_ablation.csv"), index=False)
    print(ab_df[["gene", "dir", "r_1D", "r_plasma", "r_keratinocyte", "only_dLL", "only_dAUC", "loss_on_drop_dLL", "loss_on_drop_dAUC"]].to_string(index=False))


_G_SHARED = {}


def _run_repeat(r):
    g = _G_SHARED
    X_, probe_ids_, probe_idx_, g2p_ = g["X"], g["probe_ids"], g["probe_idx"], g["g2p"]
    genes_needed_, label_, folds_ = g["genes_needed"], g["label"], g["folds"]
    panels_, marker_genes_, alias_, ident_ = g["panels"], g["marker_genes"], g["alias"], g["ident"]
    nP_, nS_, nB_ = g["nP"], g["nS"], g["nB"]

    pb_l1 = np.zeros((nP_, 2, nS_)); pf_l1 = np.zeros((nP_, 2, nS_))
    pb_l01 = np.zeros((nP_, 2, nS_)); pf_l01 = np.zeros((nP_, 2, nS_)); pn_l01 = np.zeros((nP_, 2, nS_, nB_), dtype=np.float32)
    pb_l10 = np.zeros((nP_, 2, nS_)); pf_l10 = np.zeros((nP_, 2, nS_)); pn_l10 = np.zeros((nP_, 2, nS_, nB_), dtype=np.float32)
    pf_mv = np.zeros((nP_, 2, nS_)); pn_mv = np.zeros((nP_, 2, nS_, nB_), dtype=np.float32)
    pb_mp = np.zeros((nP_, 2, nS_)); pf_mp = np.zeros((nP_, 2, nS_)); pn_mp = np.zeros((nP_, 2, nS_, nB_), dtype=np.float32)

    for f, (tr, te) in enumerate(folds_[r]):
        fd = FoldData(X_, probe_ids_, probe_idx_, g2p_, genes_needed_, label_, tr, te)
        fd_mp = FoldDataMeanProbe(X_, probe_ids_, probe_idx_, g2p_, genes_needed_, label_, tr, te)
        y_tr = label_[tr]
        Z64 = fd.Z.astype(np.float64)

        for pi, panel in enumerate(panels_):
            pid = panel["panel_id"]
            comp = comp_features(fd, marker_genes_, alias_[pid])
            b1_tr, b1_te, _, _ = scale_train(comp["pc1_tr"], comp["pc1_te"])
            designs = [(0, b1_tr[:, None], b1_te[:, None]), (1, comp["full_tr"], comp["full_te"])]

            pairs = [(m["symbol"], m["direction_published"]) for m in panel["mapped"]]
            obs_s, _, _, _ = fd.score(pairs)
            s_tr, s_te, _, _ = scale_train(obs_s[tr], obs_s[te])
            N_mat = null_score_matrix(fd, ident_[pid])
            N_tr, N_te, _, _ = scale_train_batch(N_mat[tr], N_mat[te])

            obs_cols = [fd.gene_pos[m["symbol"]] for m in panel["mapped"]]
            G_obs_tr = Z64[tr][:, obs_cols]
            G_obs_te = Z64[te][:, obs_cols]
            null_idx = np.array([[fd.gene_pos[gn] for gn in g_list] for g_list in ident_[pid]], dtype=int)
            G_null_tr = Z64[tr][:, null_idx].transpose(1, 0, 2)
            G_null_te = Z64[te][:, null_idx].transpose(1, 0, 2)

            for bi, B_tr, B_te in designs:
                wb1 = fit_logistic(B_tr, y_tr, lam=1.0)
                pb_l1[pi, bi, te] = predict_logistic(wb1, B_te)
                wf1 = fit_logistic(np.column_stack([B_tr, s_tr]), y_tr, lam=1.0)
                pf_l1[pi, bi, te] = predict_logistic(wf1, np.column_stack([B_te, s_te]))

                w_mv = fit_logistic(np.column_stack([B_tr, G_obs_tr]), y_tr, lam=1.0)
                pf_mv[pi, bi, te] = predict_logistic(w_mv, np.column_stack([B_te, G_obs_te]))
                for start in range(0, nB_, 250):
                    end = min(start + 250, nB_)
                    c_nb = fit_logistic_multi_batch(B_tr, G_null_tr[start:end], y_tr, lam=1.0)
                    pn_mv[pi, bi, te, start:end] = predict_logistic_multi_batch(c_nb, B_te, G_null_te[start:end]).astype(np.float32)

                for lam_val, pb_arr, pf_arr, pn_arr in [
                    (0.1, pb_l01, pf_l01, pn_l01),
                    (10.0, pb_l10, pf_l10, pn_l10),
                ]:
                    wb = fit_logistic(B_tr, y_tr, lam=lam_val)
                    pb_arr[pi, bi, te] = predict_logistic(wb, B_te)
                    wf = fit_logistic(np.column_stack([B_tr, s_tr]), y_tr, lam=lam_val)
                    pf_arr[pi, bi, te] = predict_logistic(wf, np.column_stack([B_te, s_te]))
                    cn = fit_logistic_batch(B_tr, N_tr, y_tr, lam=lam_val)
                    pn_arr[pi, bi, te, :] = predict_logistic_batch(cn, B_te, N_te).astype(np.float32)

            comp_mp = comp_features(fd_mp, marker_genes_, alias_[pid])
            b1m_tr, b1m_te, _, _ = scale_train(comp_mp["pc1_tr"], comp_mp["pc1_te"])
            obs_sm, _, _, _ = fd_mp.score(pairs)
            sm_tr, sm_te, _, _ = scale_train(obs_sm[tr], obs_sm[te])
            Nm_mat = null_score_matrix(fd_mp, ident_[pid])
            Nm_tr, Nm_te, _, _ = scale_train_batch(Nm_mat[tr], Nm_mat[te])
            for bi, B_tr, B_te in [(0, b1m_tr[:, None], b1m_te[:, None]), (1, comp_mp["full_tr"], comp_mp["full_te"])]:
                wbm = fit_logistic(B_tr, y_tr, lam=1.0)
                pb_mp[pi, bi, te] = predict_logistic(wbm, B_te)
                wfm = fit_logistic(np.column_stack([B_tr, sm_tr]), y_tr, lam=1.0)
                pf_mp[pi, bi, te] = predict_logistic(wfm, np.column_stack([B_te, sm_te]))
                cnm = fit_logistic_batch(B_tr, Nm_tr, y_tr, lam=1.0)
                pn_mp[pi, bi, te, :] = predict_logistic_batch(cnm, B_te, Nm_te).astype(np.float32)

    o_l1 = np.zeros((nP_, 2, 2))
    o_l01 = np.zeros((nP_, 2, 2)); n_l01 = np.zeros((nP_, 2, 2, nB_))
    o_l10 = np.zeros((nP_, 2, 2)); n_l10 = np.zeros((nP_, 2, 2, nB_))
    o_mv = np.zeros((nP_, 2, 2)); n_mv = np.zeros((nP_, 2, 2, nB_))
    o_mp = np.zeros((nP_, 2, 2)); n_mp = np.zeros((nP_, 2, 2, nB_))

    for pi in range(nP_):
        for bi in range(2):
            bll1 = logloss(label_, pb_l1[pi, bi]); bauc1 = fast_auc(label_, pb_l1[pi, bi])
            o_l1[pi, bi, 0] = bll1 - logloss(label_, pf_l1[pi, bi])
            o_l1[pi, bi, 1] = fast_auc(label_, pf_l1[pi, bi]) - bauc1

            o_mv[pi, bi, 0] = bll1 - logloss(label_, pf_mv[pi, bi])
            o_mv[pi, bi, 1] = fast_auc(label_, pf_mv[pi, bi]) - bauc1
            n_mv[pi, bi, 0] = bll1 - logloss_batch(label_, pn_mv[pi, bi])
            n_mv[pi, bi, 1] = fast_auc_batch(label_, pn_mv[pi, bi]) - bauc1

            for oa, na, pba, pfa, pna in [
                (o_l01, n_l01, pb_l01, pf_l01, pn_l01),
                (o_l10, n_l10, pb_l10, pf_l10, pn_l10),
                (o_mp, n_mp, pb_mp, pf_mp, pn_mp),
            ]:
                bll = logloss(label_, pba[pi, bi]); bauc = fast_auc(label_, pba[pi, bi])
                oa[pi, bi, 0] = bll - logloss(label_, pfa[pi, bi])
                oa[pi, bi, 1] = fast_auc(label_, pfa[pi, bi]) - bauc
                na[pi, bi, 0] = bll - logloss_batch(label_, pna[pi, bi])
                na[pi, bi, 1] = fast_auc_batch(label_, pna[pi, bi]) - bauc
    return r, o_l1, o_l01, n_l01, o_l10, n_l10, o_mv, n_mv, o_mp, n_mp


def main():
    t0 = time.time()
    X, probe_ids, patient, label, gsm, g2p, panels, marker_genes = build_panels_and_load()
    probe_idx = {p: i for i, p in enumerate(probe_ids)}
    nB = 1000
    ident = load_identities(nB=nB)
    folds = make_folds(patient, label, 42, 20, 5)
    nP, nS, nR = len(panels), len(label), 20

    # Collect minimal gene universe (all panel genes + all 150 markers + all 1000 random genes for 42 panels)
    genes_needed = set()
    for ct in CELLTYPES:
        genes_needed.update(marker_genes[ct])
    for p in panels:
        for m in p["mapped"]:
            genes_needed.add(m["symbol"])
        for g_list in ident[p["panel_id"]]:
            genes_needed.update(g_list)
    genes_needed = sorted(genes_needed)
    print(f"Loaded GSE16134 {X.shape}, {nP} panels, {len(genes_needed)} unique genes in universe (B={nB})")

    alias = {p["panel_id"]: ({m["symbol"] for m in p["mapped"]} | {m["gene"] for m in p["mapped"]}) for p in panels}

    # --- Parallel execution across 20 repeats using fork ---
    import multiprocessing as mp
    _G_SHARED.clear()
    _G_SHARED.update({
        "X": X, "probe_ids": probe_ids, "probe_idx": probe_idx, "g2p": g2p,
        "genes_needed": genes_needed, "label": label, "folds": folds,
        "panels": panels, "marker_genes": marker_genes, "alias": alias,
        "ident": ident, "nP": nP, "nS": nS, "nB": nB,
    })

    obs_eq_l1 = np.zeros((nP, 2, 2, nR))
    obs_eq_l01 = np.zeros((nP, 2, 2, nR)); null_eq_l01 = np.zeros((nP, 2, 2, nB, nR))
    obs_eq_l10 = np.zeros((nP, 2, 2, nR)); null_eq_l10 = np.zeros((nP, 2, 2, nB, nR))
    obs_mv = np.zeros((nP, 2, 2, nR)); null_mv = np.zeros((nP, 2, 2, nB, nR))
    obs_mp = np.zeros((nP, 2, 2, nR)); null_mp = np.zeros((nP, 2, 2, nB, nR))

    ctx = mp.get_context("fork")
    n_workers = min(8, os.cpu_count() or 4)
    print(f"Launching {n_workers} worker processes across {nR} repeats...", flush=True)
    with ctx.Pool(n_workers) as pool:
        for res in pool.imap_unordered(_run_repeat, range(nR)):
            r, o_l1, o_l01, n_l01, o_l10, n_l10, o_mv, n_mv, o_mp, n_mp = res
            obs_eq_l1[:, :, :, r] = o_l1
            obs_eq_l01[:, :, :, r] = o_l01; null_eq_l01[:, :, :, :, r] = n_l01
            obs_eq_l10[:, :, :, r] = o_l10; null_eq_l10[:, :, :, :, r] = n_l10
            obs_mv[:, :, :, r] = o_mv; null_mv[:, :, :, :, r] = n_mv
            obs_mp[:, :, :, r] = o_mp; null_mp[:, :, :, :, r] = n_mp
            print(f"  Completed repeat {r+1:2d}/20 (elapsed {time.time()-t0:.1f}s)", flush=True)

    # Sanity check against saved primary results
    prim_e = pd.read_csv(os.path.join(ROOT, "exploration_v3", "results", "results_per_panel_primary_with_status.csv"))
    prim_c = pd.read_csv(os.path.join(ROOT, "confirmation_v1", "results", "results_per_test_primary.csv"))
    max_diff = 0.0
    for pi in range(27):
        pid = panels[pi]["panel_id"]
        row = prim_e[prim_e.panel_id == pid].iloc[0]
        max_diff = max(max_diff, abs(obs_eq_l1[pi, 0, 0].mean() - row.dlogloss_comp1d_mean),
                       abs(obs_eq_l1[pi, 1, 1].mean() - row.dauc_compfull_mean))
    for pi in range(27, 42):
        pid = panels[pi]["panel_id"]
        r1 = prim_c[(prim_c.panel_id == pid) & (prim_c.baseline == "comp1d") & (prim_c.metric == "logloss")].iloc[0]
        r2 = prim_c[(prim_c.panel_id == pid) & (prim_c.baseline == "compfull") & (prim_c.metric == "auc")].iloc[0]
        max_diff = max(max_diff, abs(obs_eq_l1[pi, 0, 0].mean() - r1.observed_mean),
                       abs(obs_eq_l1[pi, 1, 1].mean() - r2.observed_mean))
    print(f"\n[SANITY CHECK] Max absolute difference vs primary v3/confirmation results = {max_diff:.2e}")

    df_mv = summarize_results(obs_mv, null_mv, panels, "multivariate_l2")
    df_l01 = summarize_results(obs_eq_l01, null_eq_l01, panels, "lambda_0p1")
    df_l10 = summarize_results(obs_eq_l10, null_eq_l10, panels, "lambda_10")
    df_mp = summarize_results(obs_mp, null_mp, panels, "probe_maxmean")

    # Compute correlation with primary increments across all 42 panels
    prim_map = {}
    for pi in range(27):
        pid = panels[pi]["panel_id"]
        row = prim_e[prim_e.panel_id == pid].iloc[0]
        prim_map[(pid, "comp1d", "logloss")] = row.dlogloss_comp1d_mean
        prim_map[(pid, "comp1d", "auc")] = row.dauc_comp1d_mean
        prim_map[(pid, "compfull", "logloss")] = row.dlogloss_compfull_mean
        prim_map[(pid, "compfull", "auc")] = row.dauc_compfull_mean
    for _, row in prim_c.iterrows():
        if row.baseline in ("comp1d", "compfull"):
            prim_map[(int(row.panel_id), row.baseline, row.metric)] = float(row.observed_mean)

    for name, df in [
        ("Multivariate gene-level L2 (lam=1.0)", df_mv),
        ("Equal-weighted L2 (lam=0.1)", df_l01),
        ("Equal-weighted L2 (lam=10.0)", df_l10),
        ("Highest-mean-expression probe (lam=1.0)", df_mp),
    ]:
        print_summary(df, name)
        xv = [prim_map[(int(r.panel_id), r.baseline, r.metric)] for _, r in df.iterrows()]
        yv = df["observed_mean"].values
        r_all = np.corrcoef(xv, yv)[0, 1]
        print(f"  Correlation vs primary increments across 168 tests: r = {r_all:.4f}")

    # Run Panel 38 dissection
    run_panel38_dissection(X, probe_ids, probe_idx, patient, label, gsm, g2p, panels, marker_genes)
    print(f"\nALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
