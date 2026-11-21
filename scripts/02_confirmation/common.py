#!/usr/bin/env python3
"""Shared utilities for the v0.5 confirmation-set analysis.

Derived from exploration_v3/scripts/common.py.  Numerical rules are
unchanged; the only differences are:
  * confirmation panels are read from a separate JSON;
  * per-fold Welch t / mean / SD can be precomputed once and reused across
    null-panel chunks (same formulas as v3);
  * FoldData keeps its gene-level z-scores in float64 (v3 used float32);
  * shared-axis reference panels are passed explicitly (fixed 27 exploration
    panels), never the other confirmation panels;
  * a Clopper-Pearson helper for the Monte Carlo bounds.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, beta

EPS = 1e-9
CELLTYPES = ["plasma", "other_immune", "keratinocyte", "fibroblast", "endothelial"]


# ---------------------------------------------------------------- metrics
def bh_q(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        val = p[idx] * m / (m - rank + 1)
        prev = min(prev, val)
        q[idx] = prev
    return np.clip(q, 0, 1)


def fast_auc(y, s):
    y = np.asarray(y, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s, method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fast_auc_batch(y, S):
    """AUC for each column of S (n x B)."""
    y = np.asarray(y, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(S.shape[1], np.nan)
    r = rankdata(S, method="average", axis=0)
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def logloss(y, p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logloss_batch(y, P):
    P = np.clip(np.asarray(P, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)[:, None]
    return -np.mean(y * np.log(P) + (1 - y) * np.log(1 - P), axis=0)


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided Clopper-Pearson interval for k successes in n trials."""
    k = int(k)
    n = int(n)
    if n <= 0:
        return 0.0, 1.0
    if k <= 0:
        lo = 0.0
    else:
        lo = float(beta.ppf(alpha / 2.0, k, n - k + 1))
    if k >= n:
        hi = 1.0
    else:
        hi = float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------- models
def fit_logistic(F, y, lam=1.0, max_iter=80):
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    n, p = F.shape
    Xd = np.column_stack([np.ones(n), F])
    b = np.zeros(p + 1)
    b[0] = np.log(max(np.mean(y), EPS) / max(1 - np.mean(y), EPS))
    for _ in range(max_iter):
        z = np.clip(Xd @ b, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(pr * (1 - pr), 1e-6, None)
        A = Xd.T @ (Xd * w[:, None]) + 2 * lam * np.eye(p + 1)
        A[0, 0] -= 2 * lam
        g = Xd.T @ (y - pr) - 2 * lam * np.r_[0.0, b[1:]]
        try:
            step = np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(A + 1e-6 * np.eye(p + 1), g, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def predict_logistic(b, F):
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    z = np.clip(np.column_stack([np.ones(len(F)), F]) @ b, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_batch(B, S, y, lam=1.0, max_iter=30, tol=1e-8):
    """Fit y ~ B @ w + S[:,b] * c_b for every column b of S.

    B: n x q baseline design (already scaled); S: n x nb panel scores (raw).
    Returns coef (nb, q+2): [intercept, w..., c_b].
    """
    B = np.asarray(B, dtype=float)
    S = np.asarray(S, dtype=float)
    n, q = B.shape
    nb = S.shape[1]
    Xb = np.column_stack([np.ones(n), B])  # n x k
    k = q + 1
    Xb2 = (Xb[:, :, None] * Xb[:, None, :]).reshape(n, k * k)
    coef = np.zeros((nb, k + 1))
    coef[:, 0] = np.log(max(np.mean(y), EPS) / max(1 - np.mean(y), EPS))
    XbT = Xb.T
    for _ in range(max_iter):
        eta = coef[:, 0][None, :] + B @ coef[:, 1:k].T + S * coef[:, k][None, :]
        eta = np.clip(eta, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(pr * (1 - pr), 1e-6, None)
        r = y[:, None] - pr
        wS = w * S
        A_bb = (Xb2.T @ w).reshape(k, k, nb).transpose(2, 0, 1)
        A_bs = np.empty((nb, k))
        for j in range(k):
            A_bs[:, j] = (wS * Xb[:, j][:, None]).sum(axis=0)
        A_ss = (wS * S).sum(axis=0)
        g_b = XbT @ r  # k x nb
        g_s = (S * r).sum(axis=0)
        A = np.empty((nb, k + 1, k + 1))
        A[:, :k, :k] = A_bb
        A[:, :k, k] = A_bs
        A[:, k, :k] = A_bs
        A[:, k, k] = A_ss
        idx = np.arange(1, k + 1)
        A[:, idx, idx] += 2 * lam
        g = np.column_stack([g_b.T, g_s])
        g[:, 1:] -= 2 * lam * coef[:, 1:]
        try:
            step = np.linalg.solve(A, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = np.empty_like(g)
            for b in range(nb):
                step[b] = np.linalg.lstsq(A[b] + 1e-6 * np.eye(k + 1), g[b], rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        coef += step
        if np.max(np.abs(step)) < tol:
            break
    return coef


def predict_logistic_batch(coef, B, S):
    k = B.shape[1] + 1
    eta = coef[:, 0][None, :] + B @ coef[:, 1:k].T + S * coef[:, k][None, :]
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def scale_train(x_tr, x_te):
    mu = float(np.mean(x_tr))
    sd = float(np.std(x_tr, ddof=1))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return (x_tr - mu) / sd, (x_te - mu) / sd, mu, sd


def scale_train_batch(M_tr, M_te):
    mu = M_tr.mean(axis=0, keepdims=True)
    sd = M_tr.std(axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    return (M_tr - mu) / sd, (M_te - mu) / sd, mu, sd


def pca1(M_tr, M_te):
    center = M_tr.mean(axis=0, keepdims=True)
    A = M_tr - center
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    pc_tr = U[:, 0] * S[0]
    pc_te = (M_te - center) @ Vt[0]
    tot = float(np.sum(S ** 2))
    var = float(S[0] ** 2 / tot) if tot > 0 else np.nan
    return pc_tr, pc_te, var, Vt[0]


# ---------------------------------------------------------------- folds
def make_folds(patient, label, seed, n_repeats, n_folds=5):
    """Patient-cluster folds; stratified by has-unaffected-site and site count."""
    patients = np.unique(patient)
    site_count = {int(p): int((patient == p).sum()) for p in patients}
    has_unaff = {int(p): int((label[patient == p] == 0).sum() > 0) for p in patients}
    repeats = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + 1000 * r)
        strata = {}
        for p in patients:
            strata.setdefault((has_unaff[int(p)], site_count[int(p)]), []).append(int(p))
        fold_patients = [[] for _ in range(n_folds)]
        for key in sorted(strata):
            arr = np.array(strata[key])
            rng.shuffle(arr)
            for p in arr:
                sizes = [len(x) for x in fold_patients]
                f = int(np.argmin(sizes))
                fold_patients[f].append(int(p))
        folds = []
        for f in range(n_folds):
            te_pat = set(fold_patients[f])
            te = np.where(np.isin(patient, list(te_pat)))[0]
            tr = np.where(~np.isin(patient, list(te_pat)))[0]
            folds.append((tr, te))
        repeats.append(folds)
    return repeats


def folds_to_frame(patient, seed, repeats):
    rows = []
    for r, folds in enumerate(repeats):
        for f, (tr, te) in enumerate(folds):
            for i in te:
                rows.append({"site": i, "patient": int(patient[i]), "repeat": r, "fold": f})
    return pd.DataFrame(rows)


def load_folds(path, n_repeats, n_sites):
    d = pd.read_csv(path)
    folds = []
    for r in range(n_repeats):
        sub = d[d.repeat == r]
        fr = []
        for f in range(5):
            te = np.sort(sub[sub.fold == f].site.values)
            mask = np.ones(n_sites, dtype=bool)
            mask[te] = False
            tr = np.where(mask)[0]
            fr.append((tr, te))
        folds.append(fr)
    return folds


# ---------------------------------------------------------------- data
def load_base(data_dir, panels_path=None):
    D = np.load(os.path.join(data_dir, "gse16134.npz"), allow_pickle=True)
    X = D["X"].astype(np.float32)
    probe_ids = np.asarray(D["probe_ids"]).astype(str)
    patient = np.asarray(D["patient"]).astype(int)
    label = np.asarray(D["label"]).astype(int)
    gsm = np.asarray(D["gsm"]).astype(str)
    title = np.asarray(D["title"]).astype(str)
    panels = json.load(open(panels_path or os.path.join(data_dir, "panels.json")))
    g2p = json.load(open(os.path.join(data_dir, "gpl570_gene2probe.json")))
    return X, probe_ids, patient, label, gsm, title, panels, g2p


def welch_t(Xtr, ytr):
    ya = ytr == 1
    yu = ~ya
    ma = Xtr[ya].mean(axis=0, dtype=np.float64)
    mu = Xtr[yu].mean(axis=0, dtype=np.float64)
    va = Xtr[ya].var(axis=0, ddof=1, dtype=np.float64)
    vu = Xtr[yu].var(axis=0, ddof=1, dtype=np.float64)
    se = np.sqrt(va / max(int(ya.sum()), 1) + vu / max(int(yu.sum()), 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ma - mu) / se
    t[~np.isfinite(t)] = 0.0
    return t


def fold_stats(X, label, tr):
    """Per-probe training-fold mean, SD, Welch t and validity (float64)."""
    mu = X[tr].mean(axis=0, dtype=np.float64)
    sd = X[tr].std(axis=0, ddof=1, dtype=np.float64)
    t = welch_t(X[tr], label[tr])
    valid = np.isfinite(sd) & (sd > 1e-6)
    return mu, sd, t, valid


class FoldData:
    """Training-fold parameters plus gene-level z-scored expression.

    Only genes in `genes` are materialised; each gene uses its training-fold
    representative probe (largest |Welch t|, ties by lexicographically smallest
    probe id), exactly the rule applied to published panel genes.
    """

    def __init__(self, X, probe_ids, probe_idx, g2p, genes, label, tr, te, stats=None):
        self.tr = np.asarray(tr, dtype=int)
        self.te = np.asarray(te, dtype=int)
        self.y_tr = label[self.tr]
        self.y_te = label[self.te]
        if stats is None:
            mu, sd, t, valid = fold_stats(X, label, self.tr)
        else:
            mu, sd, t, valid = stats
        self.mu = mu
        self.sd = sd
        self.valid = valid
        self.t = t
        self.abs_t = np.abs(t)

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
                key = (-self.abs_t[j], p)
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
        if len(cols):
            Z = X[:, self.cols].astype(np.float64)
            Z = (Z - mu[self.cols]) / sd[self.cols]
            self.Z = Z
        else:
            self.Z = np.zeros((X.shape[0], 0), dtype=np.float64)
        self.n = X.shape[0]

    def score(self, gene_dirs):
        """Mean z-score over genes; direction 0 -> training-fold t sign."""
        cols, dirs = [], []
        for g, d in gene_dirs:
            pos = self.gene_pos.get(g)
            if pos is None:
                continue
            if d == 0:
                d = 1 if self.gene_t[g] >= 0 else -1
            cols.append(pos)
            dirs.append(d)
        if not cols:
            return None, 0, [], []
        d = np.asarray(dirs, dtype=float)
        S = self.Z[:, cols] @ (d / len(cols))
        return S, len(cols), cols, d


# ---------------------------------------------------------------- panels
def panel_gene_sets(panels):
    out = {}
    for p in panels:
        syms = []
        for m in p["mapped"]:
            if m["symbol"] not in syms:
                syms.append(m["symbol"])
        out[p["panel_id"]] = syms
    return out


def panel_symbol_alias(panels):
    out = {}
    for p in panels:
        s = set()
        for m in p["mapped"]:
            s.add(m["symbol"])
            s.add(m["gene"])
        out[p["panel_id"]] = s
    return out


def shared_gene_dirs(panel, reference_panels):
    """Genes/directions of the fixed reference panels, excluding the tested panel.

    Direction aggregation is v3's rule: sum of published nonzero directions;
    positive -> +1, negative -> -1, exact tie or no direction -> 0.
    """
    tested = set()
    for m in panel["mapped"]:
        tested.add(m["symbol"])
        tested.add(m["gene"])
    rec = {}
    for other in reference_panels:
        if other["panel_id"] == panel["panel_id"]:
            continue
        for m in other["mapped"]:
            if m["symbol"] in tested or m["gene"] in tested:
                continue
            r = rec.setdefault(m["symbol"], [])
            d = m.get("direction_published", 0)
            if d != 0:
                r.append(d)
    out = []
    for sym, dirs in rec.items():
        if not dirs:
            d = 0
        elif sum(dirs) > 0:
            d = 1
        elif sum(dirs) < 0:
            d = -1
        else:
            d = 0
        out.append((sym, d))
    return out


def comp_features(fd, marker_genes, panel_alias, n_types=None):
    """Cell-type scores with the tested panel's genes dynamically removed."""
    types = list(marker_genes.keys()) if n_types is None else list(marker_genes.keys())[:n_types]
    scores = []
    kept = {}
    for ct in types:
        gs = [g for g in marker_genes[ct] if g not in panel_alias]
        kept[ct] = len(gs)
        S, k, _, _ = fd.score([(g, 1) for g in gs])
        if S is None:
            S = np.zeros(fd.n, dtype=float)
        scores.append(S)
    M = np.column_stack(scores)
    mu = M[fd.tr].mean(axis=0)
    sd = M[fd.tr].std(axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    Mz = (M - mu) / sd
    pc_tr, pc_te, var, load = pca1(Mz[fd.tr], Mz[fd.te])
    idx = {ct: i for i, ct in enumerate(types)}
    orient = load[idx.get("plasma", 0)] + load[idx.get("other_immune", 0)]
    if orient < 0:
        pc_tr, pc_te, load = -pc_tr, -pc_te, -load
    return {"pc1_tr": pc_tr, "pc1_te": pc_te, "full_tr": Mz[fd.tr],
            "full_te": Mz[fd.te], "var": var, "load": load,
            "types": types, "kept": kept}
