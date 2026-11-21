#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Healthy-donor cohorts v2: normalized published directions from the main analysis.

Fixes (independent_check P1-1):
  * directions are read from the main analysis's normalized `mapped` table
    (data_dir/panels.json + 01_panels_confirmation.json), not re-parsed from
    the Chinese strings (which lost 48 direction entries in 8 panels);
  * gene lookup tries the GPL570 symbol first, then the current alias
    (`gene`), e.g. GPR116 -> ADGRF5;
  * "complete published direction" subset = all mapped genes have a published
    direction (22 panels = 17 exploration + 5 confirmation), not 16;
  * per-cohort marker availability is reported (GSE173078 is not 150/150).
Only AUC and axis correlation; no delta tests, no significance claims.
"""
import json, os, re
import numpy as np
import pandas as pd

BASE = "."
DATA = "./data/external_rnaseq"
OUT = os.path.join(DATA, "out_v2")
os.makedirs(OUT, exist_ok=True)


def auc_score(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_panels():
    out = []
    for p in json.load(open("./data/panels.json")):
        out.append(dict(panel_id=int(p["panel_id"]), scope="exploration",
                        pathway=p.get("pathway", ""), pmid=str(p.get("pmid", "")),
                        mapped=[dict(symbol=m["symbol"], gene=m.get("gene", m["symbol"]),
                                     d=int(m.get("direction_published", 0))) for m in p["mapped"]]))
    for p in json.load(open(os.path.join(BASE, "confirmation_v1/01_panels_confirmation.json"))):
        out.append(dict(panel_id=int(p["panel_id"]), scope="confirmation",
                        pathway=p.get("pathway", ""), pmid=str(p.get("pmid", "")),
                        mapped=[dict(symbol=m["symbol"], gene=m.get("gene", m["symbol"]),
                                     d=int(m.get("direction_published", 0))) for m in p["mapped"]]))
    return out


PANELS = load_panels()
MK = pd.read_csv(os.path.join(BASE, "exploration_v3/results/markers_celltype_top_v3.csv"))
COMPS = {ct: list(sub.sort_values("rank")["gene_symbol"]) for ct, sub in MK.groupby("celltype")}
CELLTYPES = list(COMPS)


def zscore(X):
    X = X[~X.index.duplicated(keep="first")]
    return X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1, ddof=1).replace(0, np.nan), axis=0).dropna(how="any")


def resolve(Z, m):
    for g in (m["symbol"], m["gene"]):
        if g in Z.index:
            return g
    return None


def evaluate(X, y, cohort):
    Z = zscore(X)
    rows = []
    for p in PANELS:
        used, dirs, missing, src = [], [], [], set()
        for m in p["mapped"]:
            g = resolve(Z, m)
            if g is None:
                missing.append(m["symbol"]); continue
            d = m["d"]
            if d == 0:
                a = Z.loc[g].values[y == 1]; b = Z.loc[g].values[y == 0]
                d = 1.0 if a.mean() > b.mean() else -1.0
                src.add("t")
            else:
                src.add("published")
            used.append(g); dirs.append(float(d))
        if not used:
            rows.append(dict(cohort=cohort, panel_id=p["panel_id"], scope=p["scope"], pathway=p["pathway"],
                             pmid=p["pmid"], n_genes=len(p["mapped"]), n_used=0,
                             missing=";".join(missing), auc=np.nan, r_axis=np.nan, dir_source="none",
                             complete_published=False, _score=None))
            continue
        S = (Z.loc[used].values * np.array(dirs)[:, None]).mean(axis=0)
        rows.append(dict(cohort=cohort, panel_id=p["panel_id"], scope=p["scope"], pathway=p["pathway"],
                         pmid=p["pmid"], n_genes=len(p["mapped"]), n_used=len(used),
                         missing=";".join(missing), auc=auc_score(y, S), r_axis=np.nan,
                         dir_source="+".join(sorted(src)),
                         complete_published=all(m["d"] != 0 for m in p["mapped"]), _score=S))
    # composition axis
    comp, avail = {}, {}
    for ct, gs in COMPS.items():
        gg = [g for g in gs if g in Z.index]
        avail[ct] = (len(gg), len(gs))
        comp[ct] = Z.loc[gg].mean(axis=0).values
    C = pd.DataFrame(comp)
    Cz = C.sub(C.mean(), axis=1).div(C.std(ddof=1), axis=1)
    Cc = Cz.values - Cz.values.mean(axis=0)
    u, s, vt = np.linalg.svd(Cc, full_matrices=False)
    pc1 = u[:, 0] * s[0]
    if np.corrcoef(pc1, Cz["other_immune"].values)[0, 1] < 0:
        pc1 = -pc1
    var = float(s[0] ** 2 / (s ** 2).sum())
    for i in range(len(rows)):
        sc = rows[i].pop("_score")
        if sc is not None:
            rows[i]["r_axis"] = float(np.corrcoef(sc, pc1)[0, 1])
    res = pd.DataFrame(rows)
    compdf = pd.DataFrame({"sample": list(X.columns), "group": np.where(y == 1, "periodontitis", "healthy"),
                           "axis_pc1": pc1, **{f"comp_{k}": v for k, v in comp.items()}})
    return res, compdf, var, avail


def summarize(res, cohort, var, avail, extra):
    ok = res.dropna(subset=["auc"])
    sub = ok[ok.complete_published]
    d = dict(cohort=cohort, n_panels=int(len(ok)),
             auc_median=float(ok.auc.median()), auc_q1=float(ok.auc.quantile(.25)),
             auc_q3=float(ok.auc.quantile(.75)), auc_min=float(ok.auc.min()), auc_max=float(ok.auc.max()),
             auc_median_expl=float(ok[ok.scope == "exploration"].auc.median()),
             auc_median_conf=float(ok[ok.scope == "confirmation"].auc.median()),
             r_median=float(ok.r_axis.median()), r_min=float(ok.r_axis.min()), r_max=float(ok.r_axis.max()),
             r_q1=float(ok.r_axis.quantile(.25)), r_q3=float(ok.r_axis.quantile(.75)),
             r_median_expl=float(ok[ok.scope == "exploration"].r_axis.median()),
             r_median_conf=float(ok[ok.scope == "confirmation"].r_axis.median()),
             n_r_ge_080=int((ok.r_axis >= 0.80).sum()),
             n_auc_ge_080=int((ok.auc >= 0.80).sum()),
             n_auc_ge_084=int((ok.auc >= 0.84).sum()),
             pc1_var=var, markers={k: f"{a}/{b}" for k, (a, b) in avail.items()},
             markers_total=f"{sum(a for a, _ in avail.values())}/{sum(b for _, b in avail.values())}",
             complete_published_n=int(len(sub)),
             complete_auc_median=float(sub.auc.median()) if len(sub) else np.nan,
             complete_auc_min=float(sub.auc.min()) if len(sub) else np.nan,
             complete_auc_max=float(sub.auc.max()) if len(sub) else np.nan,
             complete_r_median=float(sub.r_axis.median()) if len(sub) else np.nan,
             n_dir_t=int(res.dir_source.str.contains("t").sum()))
    d.update(extra)
    return d


# GSE223924
mat = pd.read_csv(os.path.join(DATA, "GSE223924_lcpm_exp.csv.gz"), index_col=0)
mat.index = mat.index.astype(str).str.strip()
grp = np.array(["healthy" if c.startswith("H") else ("periodontitis" if c.startswith("PT") else "periimplantitis")
                for c in mat.columns])
keep = grp != "periimplantitis"
X1 = mat.loc[:, keep]
y1 = (grp[keep] == "periodontitis").astype(int)
r1, c1, v1, a1 = evaluate(X1, y1, "GSE223924")
s1 = summarize(r1, "GSE223924", v1, a1, dict(
    design="10 periodontitis vs 10 healthy (paired peri-implantitis samples excluded)",
    data="GEO supplementary lcpm matrix (submitter-normalized log CPM)"))

# GSE173078
cnt = pd.read_csv(os.path.join(DATA, "GSE173078_rnaseq_raw_counts.txt.gz"), sep="\t", index_col=0, quotechar='"')
cnt.columns = [c.strip('"').strip() for c in cnt.columns]
cnt.index = cnt.index.astype(str).str.strip('"').str.strip()
gmap = pd.read_csv(os.path.join(DATA, "gsm_map.tsv"), sep="\t", header=None,
                   names=["gsm", "title", "phenotype", "batch"])
gmap["col"] = gmap["title"].str.extract(r"^([A-Z]{2}\d+)")[0]
ph = dict(zip(gmap["col"], gmap["phenotype"]))
cols = [c for c in cnt.columns if c in ph]
cnt = cnt[cols]
y2all = np.array([ph[c] for c in cols])
keep2 = y2all != "Gingivitis"
cnt = cnt.loc[:, keep2]; y2all = y2all[keep2]
lib = cnt.sum(axis=0)
cpm = cnt.div(lib, axis=1) * 1e6
mask = (cnt >= 10).sum(axis=1) >= 5
X2 = np.log2(cpm[mask] + 1)
y2 = (y2all == "Periodontitis").astype(int)
r2, c2, v2, a2 = evaluate(X2, y2, "GSE173078")
s2 = summarize(r2, "GSE173078", v2, a2, dict(
    design="12 periodontitis vs 12 healthy (12 gingivitis samples excluded; batch A/B balanced)",
    data="GEO supplementary raw counts, log2(CPM+1), genes with counts>=10 in >=5 samples"))

res = pd.concat([r1, r2], ignore_index=True)
res.to_csv(os.path.join(OUT, "ph_sens_panel_level_v2.csv"), index=False)
c1.to_csv(os.path.join(OUT, "GSE223924_composition_v2.csv"), index=False)
c2.to_csv(os.path.join(OUT, "GSE173078_composition_v2.csv"), index=False)
json.dump({"GSE223924": s1, "GSE173078": s2}, open(os.path.join(OUT, "ph_sens_summary_v2.json"), "w"),
          indent=2, ensure_ascii=False)
print(json.dumps({"GSE223924": s1, "GSE173078": s2}, indent=2, ensure_ascii=False))

# wide table like the old TSV
w = res.pivot_table(index=["panel_id", "scope", "pathway", "n_genes"], columns="cohort",
                    values=["auc", "r_axis", "n_used"], aggfunc="first")
w.columns = [f"{a}_{b}" for a, b in w.columns]
w = w.reset_index()
w.to_csv(os.path.join(OUT, "external_healthy_rnaseq_42_panels.tsv"), sep="\t", index=False)
print("\ncomplete-published panels:")
print(res[res.complete_published].sort_values(["cohort", "panel_id"])[
    ["cohort", "panel_id", "scope", "n_used", "auc", "r_axis"]].to_string(index=False))
print("\nlowest AUC per cohort:")
print(res.sort_values("auc").groupby("cohort").head(3)[
    ["cohort", "panel_id", "n_used", "missing", "auc", "r_axis"]].to_string(index=False))
