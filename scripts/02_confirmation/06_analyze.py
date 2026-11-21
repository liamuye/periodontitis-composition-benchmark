#!/usr/bin/env python3
"""Step 6: assemble tests, Monte Carlo bounds and the v0.5 criteria.

Reads the observed OOF predictions, the fixed-random-panel means, and the
selection-aware bootstrap.  All p values are reconstructed from integer k and
B: p = (1+k)/(1+B).  The 6 BH families are baseline x metric, each across the
15 panels; the 90-test global BH is reported as a sensitivity only.

Writes:
  results/results_per_test_primary.csv
  results/results_per_panel_primary.csv
  results/monte_carlo_primary.csv
  results/results_per_test_seed20261031.csv
  results/results_per_test_williams.csv
  results/criteria_v05.csv
  results/summary_v05.json
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import bh_q, clopper_pearson, logloss, fast_auc

METRICS = ["logloss", "auc"]


def conditional_ci(pred_base, pred_full, label, patient, seed=42, nboot=1000, nR=20):
    """Patient-cluster bootstrap with fixed OOF predictions (conditional)."""
    rng = np.random.default_rng(seed + 555)
    patients = np.unique(patient)
    pat_sites = {int(p): np.where(patient == p)[0] for p in patients}
    boot_idx = [np.concatenate([pat_sites[int(p)] for p in rng.choice(patients, len(patients), replace=True)])
                for _ in range(nboot)]
    out = {}
    for mi, mn in enumerate(METRICS):
        vals = []
        for idx in boot_idx:
            yb = label[idx]
            per = []
            for r in range(nR):
                pb = pred_base[r][idx]; pf = pred_full[r][idx]
                per.append(logloss(yb, pb) - logloss(yb, pf) if mn == "logloss"
                           else fast_auc(yb, pf) - fast_auc(yb, pb))
            vals.append(float(np.nanmean(per)))
        out[mn] = (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))
    return out


def analyze_tag(tag, res, panels, selection=None, joint=None, do_conditional=True):
    obs = np.load(os.path.join(res, f"obs_predictions_{tag}.npz"), allow_pickle=True)
    null = np.load(os.path.join(res, f"null_means_{tag}.npz"), allow_pickle=True)
    obs_delta = obs["obs_delta"]
    pred_base = obs["pred_base"]
    pred_full = obs["pred_full"]
    baselines = [str(x) for x in obs["baseline"]]
    pids = [int(x) for x in obs["panel_id"]]
    null_mean = null["mean"]
    B = int(null_mean.shape[-1])
    nR = int(obs_delta.shape[-1])

    meta = {int(p["panel_id"]): p for p in panels}
    y_all = np.asarray(obs["y"])
    patient_all = np.asarray(obs["patient"])
    rows = []
    for pi, pid in enumerate(pids):
        p = meta[pid]
        for bi, bname in enumerate(baselines):
            use_selection = (selection is not None and bname in ("comp1d", "compfull"))
            ci = None
            if not use_selection and do_conditional:
                ci = conditional_ci(pred_base[pi, bi], pred_full[pi, bi],
                                    y_all, patient_all,
                                    seed=int(obs["seed"]), nboot=1000, nR=nR)
            for mi, mname in enumerate(METRICS):
                vals = obs_delta[pi, bi, mi]
                obs_mean = float(np.nanmean(vals))
                obs_sd = float(np.nanstd(vals, ddof=1))
                nm = null_mean[pi, bi, mi]
                nm = nm[np.isfinite(nm)]
                k = int(np.sum(nm >= obs_mean))
                B_eff = int(len(nm))
                pval = (1 + k) / (1 + B_eff)
                row = {
                    "panel_id": pid, "pmid": p["pmid"], "pathway": p["pathway"],
                    "family": p.get("family", ""), "baseline": bname, "metric": mname,
                    "observed_mean": obs_mean, "observed_sd": obs_sd,
                    "null_mean": float(np.nanmean(nm)), "null_sd": float(np.nanstd(nm, ddof=1)),
                    "null_median": float(np.nanmedian(nm)), "null95": float(np.nanpercentile(nm, 95)),
                    "k": k, "B": B_eff, "p": pval,
                    "n_repeats": nR,
                }
                # Clopper-Pearson MC bounds on k/B, then BH in the family later
                lo, hi = clopper_pearson(k, B_eff)
                row["mc_p_lo"] = lo
                row["mc_p_hi"] = hi
                # CI: selection-aware for comp1d/compfull when available, else conditional
                if selection is not None and bname in ("comp1d", "compfull"):
                    row["ci_lo"] = float(selection.loc[(pid, bname, mname), "ci_lo"])
                    row["ci_hi"] = float(selection.loc[(pid, bname, mname), "ci_hi"])
                    row["ci_type"] = "selection_aware_patient_bootstrap_full_refit"
                elif ci is not None:
                    row["ci_lo"], row["ci_hi"] = ci[mname]
                    row["ci_type"] = "conditional_patient_bootstrap_fixed_predictions"
                else:
                    row["ci_lo"], row["ci_hi"] = np.nan, np.nan
                    row["ci_type"] = "none"
                rows.append(row)
    df = pd.DataFrame(rows)

    # BH within each baseline x metric family (15 panels)
    for bname in baselines:
        for mname in METRICS:
            mask = (df.baseline == bname) & (df.metric == mname)
            df.loc[mask, "q"] = bh_q(df.loc[mask, "p"].values)
            df.loc[mask, "q_L"] = bh_q(df.loc[mask, "mc_p_lo"].values)
            df.loc[mask, "q_U"] = bh_q(df.loc[mask, "mc_p_hi"].values)
    # global BH (90 tests) as sensitivity
    df["q_all"] = bh_q(df["p"].values)

    def mc_status(row):
        if not bool(row["positive"]):
            return "n/a"
        if row["q_U"] < 0.05:
            return "stable"
        if row["q_L"] >= 0.05:
            return "not_borderline"
        return "uncertain"

    df["positive"] = (df["q"] < 0.05) & (df["ci_lo"] > 0) & (df["observed_mean"] > df["null95"])
    df["mc_status"] = df.apply(mc_status, axis=1)
    df["positive_class"] = np.where(~df["positive"], "",
                                    np.where(df["mc_status"] == "stable", "",
                                             np.where(df["mc_status"] == "uncertain", "",
                                                      "")))
    return df, obs, null


def panel_summary(df, panels):
    rows = []
    for p in panels:
        pid = int(p["panel_id"])
        sub = df[df.panel_id == pid]
        rec = {"panel_id": pid, "pmid": p["pmid"], "pathway": p["pathway"],
               "family": p.get("family", ""), "n_genes": p["n_genes"],
               "n_mapped": p["n_mapped"]}
        for bname in sub.baseline.unique():
            s = sub[sub.baseline == bname]
            for mname in METRICS:
                r = s[s.metric == mname].iloc[0]
                rec[f"d{mname}_{bname}_mean"] = r["observed_mean"]
                rec[f"q_{mname}_{bname}"] = r["q"]
                rec[f"ci_lo_{mname}_{bname}"] = r["ci_lo"]
                rec[f"null95_{mname}_{bname}"] = r["null95"]
                rec[f"positive_{mname}_{bname}"] = bool(r["positive"])
                rec[f"mc_{mname}_{bname}"] = r["mc_status"]
            pos = s[s.positive]
            if len(pos) == 0:
                cls = "" if np.isfinite(s.ci_lo).any() else ""
            elif (pos.positive_class == "").any():
                cls = ""
            else:
                cls = ""
            rec[f"panel_positive_{bname}"] = cls
            rec[f"panel_n_positive_metrics_{bname}"] = int(len(pos))
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    res = os.path.join(args.out_dir, "results")
    panels = json.load(open(os.path.join(args.out_dir, "01_panels_confirmation.json")))

    # selection-aware CIs
    sel = pd.read_csv(os.path.join(res, "bootstrap_selection_primary.csv"))
    selection = {}
    for _, r in sel.iterrows():
        for bname in ["comp1d", "compfull"]:
            for mname in METRICS:
                selection[(int(r.panel_id), bname, mname)] = {
                    "ci_lo": float(r[f"sel_{mname}_{bname}_ci_lo"]),
                    "ci_hi": float(r[f"sel_{mname}_{bname}_ci_hi"]),
                }
    selection = pd.DataFrame.from_dict(selection, orient="index")
    selection.index.names = ["panel_id", "baseline", "metric"]

    df, obs, null = analyze_tag("primary", res, panels, selection=selection)
    df.to_csv(os.path.join(res, "results_per_test_primary.csv"), index=False)
    ps = panel_summary(df, panels)
    ps.to_csv(os.path.join(res, "results_per_panel_primary.csv"), index=False)
    df[["panel_id", "baseline", "metric", "k", "B", "p", "q", "q_L", "q_U",
        "mc_p_lo", "mc_p_hi", "mc_status", "positive", "positive_class"]] \
        .to_csv(os.path.join(res, "monte_carlo_primary.csv"), index=False)

    # sensitivity: seed 20261031 (conditional CI only)
    df_seed, _, _ = analyze_tag("seed20261031", res, panels, selection=None)
    df_seed.to_csv(os.path.join(res, "results_per_test_seed20261031.csv"), index=False)
    panel_summary(df_seed, panels).to_csv(
        os.path.join(res, "results_per_panel_seed20261031.csv"), index=False)

    # sensitivity: Williams 12-class (conditional CI only)
    df_w, _, _ = analyze_tag("williams", res, panels, selection=None)
    df_w.to_csv(os.path.join(res, "results_per_test_williams.csv"), index=False)
    panel_summary(df_w, panels).to_csv(
        os.path.join(res, "results_per_panel_williams.csv"), index=False)

    # ---- criteria
    joint = json.load(open(os.path.join(res, "joint_median_primary.json")))
    crit = []
    for bname, label in [("compfull", "15"),
                         ("comp1d", "2"),
                         ("shared", "3")]:
        sub = ps[ps[f"panel_positive_{bname}"].isin(["", "", ""])]
        n_explicit = int((ps[f"panel_positive_{bname}"] == "").sum())
        n_potential = int((ps[f"panel_positive_{bname}"] == "").sum())
        n_uneval = int((ps[f"panel_positive_{bname}"] == "").sum())
        total = n_explicit + n_potential + n_uneval
        result = "" if total <= 3 else ""
        crit.append({
            "criterion": label, "baseline": bname,
            "explicit_positive": n_explicit, "potential_positive": n_potential,
            "unevaluable": n_uneval, "count_total": total, "denominator": 15,
            "threshold": "<=3/15", "result": result,
            "detail": (f" {n_explicit} +  {n_potential} +  {n_uneval} = {total}; "
                       f"{'<=3' if total <= 3 else '>3 v0.5 '}"),
        })
    med = float(np.nanmedian(ps["dauc_comp1d_mean"].values))
    jhi = float(joint["auc_comp1d"]["ci_hi"])
    jlo = float(joint["auc_comp1d"]["ci_lo"])
    c4 = "" if (med <= 0.02 and jhi <= 0.02) else ""
    crit.append({
        "criterion": "4 ΔAUC ", "baseline": "comp1d",
        "observed_median_dauc": med, "joint_ci_lo": jlo, "joint_ci_hi": jhi,
        "threshold": "median<=0.02  joint_ci_hi<=0.02", "result": c4,
        "detail": f" {med:.6f};  95%  [{jlo:.6f}, {jhi:.6f}]",
    })
    cdf = pd.DataFrame(crit)
    cdf.to_csv(os.path.join(res, "criteria_v05.csv"), index=False)

    # family-clustered counts (reported, not used to relax the criteria)
    fam_counts = {}
    for bname in ["compfull", "comp1d", "shared"]:
        fams = ps[ps[f"panel_positive_{bname}"].isin(["", ""])]["family"]
        fam_counts[bname] = {str(k): int(v) for k, v in fams.value_counts().items()}

    overall = ("" if all(cdf.result == "")
               else "" if any(cdf.result == "")
               else "")
    summary = {
        "criteria": cdf.to_dict(orient="records"),
        "family_clustered_counts": fam_counts,
        "overall": overall,
        "primary_B": int(df["B"].max()),
        "n_panels": 15,
        "note": ("15 "
                 ""),
    }
    json.dump(summary, open(os.path.join(res, "summary_v05.json"), "w"),
              ensure_ascii=False, indent=2)

    print(cdf[["criterion", "result", "detail"]].to_string(index=False))
    print(f"\noverall: {overall}")
    print("\nper-panel primary:")
    cols = ["panel_id", "family", "panel_positive_comp1d", "panel_positive_compfull",
            "panel_positive_shared", "dauc_comp1d_mean", "dlogloss_comp1d_mean"]
    print(ps[cols].to_string(index=False))


if __name__ == "__main__":
    main()
