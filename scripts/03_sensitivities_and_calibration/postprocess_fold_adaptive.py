#!/usr/bin/env python3
"""Rebuild the fold-wise null test table from the saved delta array."""
import json, sys
import numpy as np, pandas as pd
ROOT = "."
sys.path.insert(0, f"{ROOT}/scripts/03_sensitivities_and_calibration")
from run_batch_rma_v2 import load_panels
sys.path.insert(0, f"{ROOT}/exploration_v3/scripts")
from common import bh_q

OUT = f"{ROOT}/results/foldwise_null"
z = np.load(f"{OUT}/foldwise_null_primary.npz", allow_pickle=True)
dn_m = z["delta"]; changed = z["changed"]
panels = load_panels("./data")
disc = pd.read_csv(f"{ROOT}/exploration_v3/results/results_per_panel_primary_with_status.csv").set_index("panel_id")
conf = pd.read_csv(f"{ROOT}/confirmation_v1/results/results_per_test_primary.csv")
BAS = ["comp1d", "compfull", "shared"]; MET = ["logloss", "auc"]
rows = []
for pi, p in enumerate(panels):
    pid = p["panel_id"]
    for bi, bn in enumerate(BAS):
        for mi, mn in enumerate(MET):
            nb = dn_m[pi, bi, mi]; nb = nb[np.isfinite(nb)]
            if len(nb) == 0: continue
            if pid <= 27:
                m = "dlogloss" if mn == "logloss" else "dauc"
                obs = float(disc.loc[pid, f"{m}_{bn}_mean"])
                sel = f"sel_{mn}_{bn}_ci_lo"
                ci = float(disc.loc[pid, sel if sel in disc.columns else f"{m}_{bn}_ci_lo"])
                old_med = float(disc.loc[pid, f"null_median_{mn}_{bn}"])
                old_95 = float(disc.loc[pid, f"null95_{mn}_{bn}"])
            else:
                sub = conf[(conf.panel_id == pid) & (conf.baseline == bn) & (conf.metric == mn)]
                obs = float(sub.observed_mean.iloc[0]); ci = float(sub.ci_lo.iloc[0])
                old_med = float(sub.null_median.iloc[0]); old_95 = float(sub.null95.iloc[0])
            pval = (1 + np.sum(nb >= obs)) / (1 + len(nb))
            rows.append(dict(panel_id=pid, baseline=bn, metric=mn, observed=obs, ci_lo=ci,
                             p=pval, null_median=float(np.median(nb)), null95=float(np.percentile(nb, 95)),
                             old_median=old_med, old95=old_95, null_sd=float(np.std(nb, ddof=1))))
res = pd.DataFrame(rows)
res["cohort"] = np.where(res.panel_id <= 27, "discovery", "confirmation")
res["q"] = np.nan
for (_c, _b, _m), g in res.groupby(["cohort", "baseline", "metric"]):
    res.loc[g.index, "q"] = bh_q(g.p.values)
res["positive"] = (res.q < 0.05) & (res.ci_lo > 0) & (res.observed > res.null95)
res.to_csv(f"{OUT}/foldwise_null_tests_primary.csv", index=False)
print(res[res.q < 0.05][["panel_id","cohort","baseline","metric","observed","p","q","ci_lo","null95"]].to_string(index=False))
print("\npositives:", res[res.positive][["panel_id","cohort","baseline","metric","q"]].to_string(index=False))
print("\nnull median r", round(float(np.corrcoef(res.null_median, res.old_median)[0,1]),4),
      "max diff", float(np.abs(res.null_median-res.old_median).max()))
print("null95 r", round(float(np.corrcoef(res.null95, res.old95)[0,1]),4),
      "max diff", float(np.abs(res.null95-res.old95).max()))
p13 = res[(res.panel_id==13)&(res.baseline=="comp1d")&(res.metric=="auc")]
print("panel13 comp1d auc null SD", float(p13.null_sd.iloc[0]) if len(p13) else None)
print("identity unchanged", float(1-changed.mean()))
