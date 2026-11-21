#!/usr/bin/env python3
"""Publication-grade Figures 1–6 and S1 Fig (Final v15.9).

Visual system:
- Fig 1: Classic stark black-and-white rectangular PRISMA flow (A, B) + real-axis horizontal bar chart (C).
- Figs 2–6 & S1 Fig: Nature-figure skill restrained palette (#0F4D92 deep blue, #3775BA secondary blue,
  #B64342 muted brick red, #42949E teal, #272727/#4D4D4D/#767676 neutrals), zero decorative infographic arrows,
  uppercase bold PLOS panel labels (A, B, C...), editable TrueType PDF/SVG + 300 dpi RGB LZW TIFF.
"""
import os
import shutil
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcfg"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import gaussian_kde
from PIL import Image

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = os.path.abspath(os.path.join(REPO, ".."))
OUT = os.path.join(ROOT, "figures_png")
SUB = os.path.join(ROOT, "figures_submission")
V3 = os.path.join(ROOT, "exploration_v3", "results")
CONF = os.path.join(ROOT, "confirmation_v1", "results")
SP = os.path.join(REPO, "analysis_spikein")
MV = os.path.join(REPO, "analysis_multivariate_sens")
FW = os.path.join(REPO, "analysis_foldwise_null_v2")
PH = os.path.join(REPO, "analysis_posthoc_healthy")
BR = os.path.join(REPO, "analysis_batch_rma")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.6,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.edgecolor": "#272727",
    "text.color": "#1E1E1E",
    "axes.labelcolor": "#1E1E1E",
    "xtick.color": "#272727",
    "ytick.color": "#272727",
    "axes.unicode_minus": True,
    "savefig.dpi": 300,
})

# Nature-figure skill restrained palette
NAT_BLUE = "#0F4D92"       # primary deep blue
NAT_BLUE2 = "#3775BA"      # secondary blue
NAT_BLUE_LT = "#9DB8DB"    # pale blue
NAT_RED = "#B64342"        # muted brick red (confirmation / Panel 38 / positive)
NAT_RED_LT = "#E9A6A1"     # pale brick red
NAT_TEAL = "#42949E"       # teal (fold-adaptive null)
NAT_AMBER = "#C87D20"      # restrained ochre/amber (15-test threshold in Fig 6)
NAT_DARK = "#272727"       # primary dark ink
NAT_MID = "#767676"        # muted grey
NAT_LIGHT = "#D8D8D8"      # histogram / soft fill grey
NAT_BAND = "#F4F5F7"       # subtle zebra stripe


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=10.0, fontweight="bold", va="top", ha="left", color="#111111")


PLOS_NAME_MAP = {
    "Fig1A_screening_flow": "Fig1",
    "Fig2_panel_correlation_and_null_calibration": "Fig2",
    "Fig3_exploration_27panels": "Fig3",
    "Fig4_confirmation_15panels": "Fig4",
    "Fig5_panel38_nulls_external": "Fig5",
    "Fig6_spikein_detectability": "Fig6",
    "S1_Fig_sensitivity_concordance": "S1_Fig",
}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB, exist_ok=True)
    repo_fig_dir = os.path.join(REPO, "figures")
    os.makedirs(repo_fig_dir, exist_ok=True)
    png = os.path.join(OUT, f"{name}.png")
    pdf = os.path.join(OUT, f"{name}.pdf")
    svg = os.path.join(OUT, f"{name}.svg")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    shutil.copy2(pdf, os.path.join(SUB, f"{name}.pdf"))
    shutil.copy2(pdf, os.path.join(repo_fig_dir, f"{name}.pdf"))
    im = Image.open(png).convert("RGB")
    im.save(os.path.join(SUB, f"{name}.tif"), compression="tiff_lzw", dpi=(300, 300))
    plos_short = PLOS_NAME_MAP.get(name)
    if plos_short:
        im.save(os.path.join(SUB, f"{plos_short}.tif"), compression="tiff_lzw", dpi=(300, 300))
    plt.close(fig)
    w, h = im.size
    print(f"saved {name} (+ {plos_short}.tif): {w}x{h} px = {w/300:.2f}x{h/300:.2f} in, mode={im.mode}")


# ----------------------------------------------------------------------------- Fig 1 (V1 classic minimal PRISMA + Panel C)
def fig1():
    W, H = 7.5, 6.2
    fig = plt.figure(figsize=(W, H))

    ax = fig.add_axes([0, 0.30, 1, 0.70])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H * 0.70)
    ax.axis("off")
    Ht = H * 0.70

    def rect_box(cx, cy, w, h, text, final=False, excl=False):
        fc = "#F4F4F4" if final else "#FFFFFF"
        lw = 1.15 if final else 0.70
        ec = "#111111" if not excl else "#555555"
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=6.5, color="#111111",
                fontweight="bold" if final else "normal", linespacing=1.28, zorder=3)

    def arrow_down(cx, y1, y2):
        ax.annotate("", xy=(cx, y2), xytext=(cx, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#222222",
                                    mutation_scale=7.0, shrinkA=0, shrinkB=0))

    def arrow_right(x1, x2, y):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.70, color="#444444",
                                    mutation_scale=6.5, shrinkA=0, shrinkB=0))

    bh, top, step = 0.50, Ht - 0.58, 0.77
    ys = [top - i * step for i in range(5)]

    # Panel A: Exploration set
    mx, mw, sx, sw = 1.12, 1.98, 2.94, 1.28
    ax.text(0.10, Ht - 0.08, "A", fontsize=10, fontweight="bold", va="top", color="#111111")
    ax.text(0.35, Ht - 0.10, "Exploration set (19-pathway PubMed search)",
            fontsize=7.5, fontweight="bold", va="top", color="#111111")
    rows_a = [
        "PubMed search: 19 pathway queries\ncombined with base biomarker query",
        "211 unique records\nafter de-duplication",
        "202 primary research articles\nscreened by title and abstract",
        "145 full-text articles evaluated\n(3–20 mapped genes, diagnostic AUC)",
        "27 exploration panels (Panels 1–27)",
    ]
    for i, (y, txt) in enumerate(zip(ys, rows_a)):
        rect_box(mx, y, mw, bh, txt, final=(i == 4))
        if i:
            arrow_down(mx, ys[i - 1] - bh / 2, y + bh / 2)
    for i, txt in ((1, "9 review articles\nexcluded"),
                   (2, "57 excluded\n(out of scope / non-panel)"),
                   (3, "118 not entered in\nexploration (see Methods)")):
        rect_box(sx, ys[i], sw, bh, txt, excl=True)
        arrow_right(mx + mw / 2, sx - sw / 2, ys[i])

    # Subtle vertical divider
    ax.plot([3.76, 3.76], [0.16, Ht - 0.10], color="#CCCCCC", lw=0.6)

    # Panel B: Cohort-reuse survey & confirmation set
    mx2, mw2, sx2, sw2 = 4.90, 1.98, 6.72, 1.28
    ax.text(3.88, Ht - 0.08, "B", fontsize=10, fontweight="bold", va="top", color="#111111")
    ax.text(4.13, Ht - 0.10, "Cohort-reuse survey and confirmation set",
            fontsize=7.5, fontweight="bold", va="top", color="#111111")
    rows_b = [
        "Search without GEO series filters:\n96 records returned (2020–2026)",
        "59 primary full-text articles\nassessed for GEO cohort reuse (Panel C)",
        "58 unique confirmation candidates\n(43 survey + 8 overflow + 7 recovered)",
        "Full-text eligibility screening\nunder identical 3-criterion rules",
        "15 confirmation panels (Panels 28–42)\nwithheld until rules were fixed",
    ]
    for i, (y, txt) in enumerate(zip(ys, rows_b)):
        rect_box(mx2, y, mw2, bh, txt, final=(i == 4))
        if i:
            arrow_down(mx2, ys[i - 1] - bh / 2, y + bh / 2)
    for i, txt in ((0, "37 reviews or\nout-of-scope records"),
                   (1, "16 exploration studies\nremoved (43 left)"),
                   (3, "43 excluded on\nfull-text eligibility")):
        rect_box(sx2, ys[i], sw2, bh, txt, excl=True)
        arrow_right(mx2 + mw2 / 2, sx2 - sw2 / 2, ys[i])

    # Panel C: Real-axis horizontal bar chart for 59 surveyed articles
    axC = fig.add_axes([0.28, 0.08, 0.65, 0.19])
    cats = ["Other datasets only", "GSE10334 only", "GSE16134 only", "Both GSE16134 & GSE10334"]
    cnts = [7, 6, 8, 38]
    pcts = [100 * c / 59 for c in cnts]
    cols = ["#C8C8C8", "#969696", "#636363", "#252525"]
    yC = np.arange(len(cats))
    axC.barh(yC, cnts, height=0.58, color=cols, edgecolor="#111111", lw=0.45)
    for yi, c, p in zip(yC, cnts, pcts):
        axC.text(c + 0.8, yi, f"{c} ({p:.1f}%)", va="center", ha="left", fontsize=6.5,
                 fontweight="bold" if c == 38 else "normal", color="#111111")
    axC.set_yticks(yC)
    axC.set_yticklabels(cats, fontsize=6.6, color="#111111")
    axC.set_xlim(0, 46)
    axC.set_xlabel("Number of primary biomarker articles (n = 59 surveyed; 52/59 [88.1%] reuse GSE16134 or GSE10334)",
                   fontsize=6.8, color="#111111")
    axC.tick_params(axis="y", length=0, pad=4)
    clean(axC)
    letter(fig, 0.013, 0.295, "C")
    fig.text(0.048, 0.291, "GEO cohort reuse across 59 primary studies (2020–2026)",
             fontsize=7.5, fontweight="bold", va="top", color="#111111")

    save(fig, "Fig1A_screening_flow")


# ----------------------------------------------------------------------------- Fig 2 (Nature style, PLOS uppercase A–C)
def fig2():
    e = pd.read_csv(f"{V3}/results_per_panel_primary_with_status.csv")
    d = np.load(f"{CONF}/obs_predictions_primary.npz", allow_pickle=True)
    conf_r = np.abs(np.nanmean(d["corr_panel"][:, 0, :], axis=1))
    conf_r = conf_r[np.isfinite(conf_r)]
    exp_r = np.abs(e.corr_panel_comp1d.values)
    exp_r = exp_r[np.isfinite(exp_r)]
    all_r = np.concatenate([exp_r, conf_r])
    n_ge70 = int((all_r >= 0.70).sum())

    fig = plt.figure(figsize=(7.5, 5.65))
    axA = fig.add_axes([0.075, 0.59, 0.415, 0.365])
    axB = fig.add_axes([0.566, 0.59, 0.415, 0.365])

    bins = np.linspace(0.4, 1.0, 13)
    axA.hist(exp_r, bins=bins, color=NAT_BLUE2, alpha=0.80, edgecolor="white", lw=0.6,
             label=f"Exploration (n = {len(exp_r)}; median {np.median(exp_r):.3f})")
    axA.hist(conf_r, bins=bins, histtype="step", color=NAT_RED, lw=1.55,
             label=f"Confirmation (n = {len(conf_r)}; median {np.median(conf_r):.3f})")
    axA.axvline(0.7, ymax=0.72, color=NAT_MID, ls=(0, (2.5, 2)), lw=0.75)
    axA.text(0.692, 6.8, f"|r| = 0.70\n{n_ge70}/42 ({100*n_ge70/42:.0f}%)",
             ha="right", va="top", fontsize=6.2, color=NAT_DARK, linespacing=1.25)
    axA.set_xlim(0.4, 1.0)
    axA.set_ylim(0, 9.6)
    axA.set_xlabel("|r| between panel score and 1D composition axis")
    axA.set_ylabel("Number of panels")
    axA.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axA.legend(frameon=False, loc="upper left", handlelength=1.4)
    clean(axA)

    labels = ["Fixed-identity\nbenchmark\n(1,000 panels)", "Unaligned draw,\nsingle repeat",
              "Unaligned draw,\n20-repeat average"]
    sd = [0.006336, 0.007255, 0.001753]
    p95 = [0.018102, 0.015082, 0.006565]
    x = np.arange(3)
    axB.bar(x, sd, width=0.46, color=["#4D4D4D", "#B8B8B8", NAT_RED], edgecolor="none", zorder=2)
    for i, (s, p) in enumerate(zip(sd, p95)):
        axB.text(i, s + 0.00045, f"{s:.4f}", ha="center", va="bottom", fontsize=6.2, color=NAT_DARK)
        axB.plot(i, p, marker="D", ms=4.3, mfc="white", mec=NAT_DARK, mew=0.9, ls="none", zorder=4)
        axB.text(i + 0.13, p, f"{p:.4f}", ha="left", va="center", fontsize=6.2, color=NAT_DARK)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=6.3)
    axB.set_xlim(-0.5, 2.65)
    axB.set_ylim(0, 0.021)
    axB.set_ylabel("Null ΔAUC dispersion (Panel 13, 1D)")
    axB.legend(handles=[Patch(color="#B8B8B8", label="SD"),
                        Line2D([], [], marker="D", ls="none", mfc="white", mec=NAT_DARK, ms=4.2,
                               label="95th percentile")],
               frameon=False, loc="upper right", handlelength=1.0)
    clean(axB)

    nul_e = np.load(f"{V3}/null_panel_means_primary.npz")
    nul_c = np.load(f"{CONF}/null_means_primary_fixed.npz")
    pt = pd.read_csv(f"{CONF}/results_per_test_primary.csv")
    examples = []
    for pid, title in ((25, "Necroptosis (25)\nexploration"),
                       (5, "Immune markers (5)\nexploration"),
                       (111, "Fatty acid metabolism (38)\nconfirmation")):
        if pid < 100:
            r = e[e.panel_id == pid].iloc[0]
            null = nul_e["mean"][list(nul_e["panel_id"]).index(pid), 0, 0]
            obs, lo, q = r.dlogloss_comp1d_mean, r.sel_logloss_comp1d_ci_lo, r.q_logloss_comp1d
        else:
            r = pt[(pt.panel_id == pid) & (pt.baseline == "comp1d") & (pt.metric == "logloss")].iloc[0]
            null = nul_c["mean"][list(nul_c["panel_id"]).index(pid), 0, 0]
            obs, lo, q = r.observed_mean, r.ci_lo, r.q
        examples.append((pid, title, null, obs, lo, q))
    xmax = max(max(np.max(n), o) for _, _, n, o, _, _ in examples) * 1.15
    xmin = min(np.min(n) for _, _, n, _, _, _ in examples) - 0.002
    w, gap, left = 0.278, 0.036, 0.075
    for i, (pid, title, null, obs, lo, q) in enumerate(examples):
        ax = fig.add_axes([left + i * (w + gap), 0.085, w, 0.295])
        B = len(null)
        k = int((null >= obs).sum())
        p95 = np.percentile(null, 95)
        ax.hist(null, bins=np.linspace(xmin, xmax, 44), color=NAT_LIGHT, edgecolor="white",
                lw=0.25, weights=np.full(B, 1 / B), zorder=1)
        ymax = ax.get_ylim()[1] * 1.28
        ax.axvline(p95, color="#4D4D4D", lw=0.85, ls=(0, (3, 2)), zorder=2)
        col = NAT_RED if pid in (5, 111) else NAT_DARK
        ax.axvline(obs, color=col, lw=1.6, zorder=3)
        ax.axvline(0, color="#999999", lw=0.5, zorder=1)
        ax.set_ylim(0, ymax)
        ax.set_xlim(xmin, xmax)
        pstr = f"p = {(1 + k) / (1 + B):.3f}" if k else f"k = 0/{B:,}"
        if pid == 25:
            tx, ha_s = p95 + 0.005, "left"
        elif pid == 5:
            tx, ha_s = obs + 0.0055, "left"
        else:
            tx, ha_s = obs - 0.0055, "right"
        ax.text(tx, ymax * 0.96, f"obs {obs:.4f}\n95% CI lo {lo:.4f}\n{pstr}\nq = {q:.3g}",
                ha=ha_s, va="top", fontsize=6.1, color=col, linespacing=1.30, zorder=5)
        ax.set_title(title, fontsize=6.9, pad=4, fontweight="bold", color=NAT_DARK)
        ax.set_xlabel("1D composition ΔlogLoss", labelpad=2.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        if i == 0:
            ax.set_ylabel("Fraction of matched\nrandom panels")
        else:
            ax.set_yticklabels([])
        clean(ax)

    fig.legend(handles=[Patch(color=NAT_LIGHT, label="Matched random panels (fixed identities)"),
                        Line2D([], [], color="#4D4D4D", lw=0.85, ls=(0, (3, 2)), label="95th percentile"),
                        Line2D([], [], color=NAT_DARK, lw=1.6, label="Published panel"),
                        Line2D([], [], color=NAT_RED, lw=1.6, label="Published panel (q < 0.05)")],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.525, 0.448), columnspacing=1.15)
    letter(fig, 0.010, 0.988, "A")
    letter(fig, 0.505, 0.988, "B")
    letter(fig, 0.010, 0.492, "C")
    save(fig, "Fig2_panel_correlation_and_null_calibration")


# ----------------------------------------------------------------------------- Figs 3–4 (Nature palette forest plots)
BL = [("comp1d", "1D composition axis"), ("compfull", "5D composition"), ("shared", "Shared axis")]
MET = [("logloss", "ΔlogLoss"), ("auc", "ΔAUC")]
TBL = pd.read_csv(os.path.join(REPO, "tables", "panel_labels_en.tsv"), sep="\t")
NAME = dict(zip(TBL.panel_id.astype(int), TBL.pathway_en_short.astype(str)))


def label(pid):
    disp = int(pid) - 73 if int(pid) >= 101 else int(pid)
    return f"{NAME[int(pid)]} ({disp})"


def forest(ax, s, order, positive):
    s = s.set_index("panel_id").reindex(order).reset_index()
    y = np.arange(len(s))
    for k in y[::2]:
        ax.axhspan(k - 0.5, k + 0.5, color=NAT_BAND, lw=0, zorder=0)
    ax.axvline(0, color="#8F8F8F", lw=0.65, ls=(0, (3, 2)), zorder=1)
    pos = s.panel_id.astype(int).isin(positive).values
    ax.hlines(y, s["lo"], s["hi"], color=np.where(pos, NAT_RED, "#6E6E6E"), lw=0.9, zorder=2)
    ax.scatter(s["null95"], y, marker="|", s=38, color=NAT_MID, linewidths=1.3, zorder=2.5)
    if (~pos).any():
        ax.scatter(s["mean"][~pos], y[~pos], s=14, color=NAT_DARK, zorder=3, linewidths=0)
    if pos.any():
        ax.scatter(s["mean"][pos], y[pos], s=22, color=NAT_RED, edgecolors="white", linewidths=0.45, zorder=4)
    ax.set_ylim(len(s) - 0.5, -0.5)
    ax.set_yticks(y)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    clean(ax)
    ax.spines["left"].set_visible(False)
    return s


def forest_grid(fig, gs, per_bm, positives, order):
    axes = {}
    for r, (m, mlab) in enumerate(MET):
        for c, (code, clab) in enumerate(BL):
            ax = fig.add_subplot(gs[r, c])
            s = forest(ax, per_bm[(code, m)], order, positives.get((code, m), []))
            if c == 0:
                ax.set_yticklabels([label(p) for p in s.panel_id], fontsize=6.4)
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(clab, fontweight="bold", pad=5, fontsize=7.6, color=NAT_DARK)
            ax.set_xlabel(f"{mlab} over baseline", labelpad=2.5)
            axes[(r, c)] = ax
    return axes


def key_handles(pos_label):
    return [Line2D([], [], marker="o", color=NAT_DARK, ms=3.8, lw=0.9, label="Mean gain with 95% CI"),
            Line2D([], [], marker="|", color=NAT_MID, ms=7.5, mew=1.3, ls="none",
                   label="95th percentile of matched random panels"),
            Line2D([], [], marker="o", color=NAT_RED, ms=4.3, lw=0.9, label=pos_label)]


def fig3():
    e = pd.read_csv(f"{V3}/results_per_panel_primary_with_status.csv")
    per = {}
    for code, _ in BL:
        for m, _ in MET:
            lo = f"sel_{m}_{code}_ci_lo" if f"sel_{m}_{code}_ci_lo" in e else f"d{m}_{code}_ci_lo"
            hi = lo[:-3] + "_hi"
            per[(code, m)] = pd.DataFrame({"panel_id": e.panel_id.astype(int),
                                           "mean": e[f"d{m}_{code}_mean"], "lo": e[lo], "hi": e[hi],
                                           "null95": e[f"null95_{m}_{code}"]})
    order = per[("comp1d", "logloss")].sort_values("mean", ascending=False).panel_id.tolist()
    fig = plt.figure(figsize=(7.5, 8.5))
    gs = fig.add_gridspec(2, 3, left=0.235, right=0.985, top=0.922, bottom=0.052,
                          wspace=0.12, hspace=0.145)
    axes = forest_grid(fig, gs, per, {("comp1d", "logloss"): [4, 5, 12]}, order)
    letter(fig, 0.010, axes[(0, 0)].get_position().y1 + 0.025, "A")
    letter(fig, 0.010, axes[(1, 0)].get_position().y1 + 0.025, "B")
    fig.legend(handles=key_handles("Exceeds matched random benchmark (q < 0.05)"),
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.56, 0.992),
               handletextpad=0.45, columnspacing=1.35)
    save(fig, "Fig3_exploration_27panels")


def fig4():
    pt = pd.read_csv(f"{CONF}/results_per_test_primary.csv")
    per = {}
    for code, _ in BL:
        for m, _ in MET:
            s = pt[(pt.baseline == code) & (pt.metric == m)]
            per[(code, m)] = pd.DataFrame({"panel_id": s.panel_id.astype(int), "mean": s.observed_mean,
                                           "lo": s.ci_lo, "hi": s.ci_hi, "null95": s.null95})
    order = per[("comp1d", "logloss")].sort_values("mean", ascending=False).panel_id.tolist()
    fig = plt.figure(figsize=(7.5, 5.7))
    gs = fig.add_gridspec(2, 3, left=0.245, right=0.985, top=0.902, bottom=0.075,
                          wspace=0.12, hspace=0.29)
    axes = forest_grid(fig, gs, per, {(c, m): [111] for c, _ in BL for m, _ in MET}, order)
    letter(fig, 0.010, axes[(0, 0)].get_position().y1 + 0.035, "A")
    letter(fig, 0.010, axes[(1, 0)].get_position().y1 + 0.035, "B")
    fig.legend(handles=key_handles("Panel 38 (exceeds matched random benchmark)"),
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.57, 0.992),
               handletextpad=0.45, columnspacing=1.35)
    save(fig, "Fig4_confirmation_15panels")


# ----------------------------------------------------------------------------- Fig 5 (Nature style, zero infographic arrows)
def fig5():
    ab = pd.read_csv(os.path.join(MV, "panel38_gene_ablation.csv")).sort_values("only_dLL", ascending=True).reset_index(drop=True)
    null_fixed = np.load(f"{V3}/null_panel_means_primary.npz")["mean"][12, 0, 1]
    null_adapt = np.load(os.path.join(FW, "foldwise_null_primary.npz"))["delta"][12, 0, 1]
    null_redraw = np.load(os.path.join(FW, "foldwise_null_fullredraw.npz"))["delta"][12, 0, 1]
    e = pd.read_csv(f"{V3}/results_per_panel_primary_with_status.csv")
    obs_p13 = float(e[e.panel_id == 13].iloc[0].dauc_comp1d_mean)
    ph = pd.read_csv(os.path.join(PH, "ph_sens_panel_level_v2.csv"))
    piv = ph.pivot(index="panel_id", columns="cohort", values="auc")
    dir_map = ph.groupby("panel_id")["complete_published"].first()

    fig = plt.figure(figsize=(7.5, 4.35))
    axA = fig.add_axes([0.088, 0.145, 0.240, 0.72])
    axB = fig.add_axes([0.398, 0.145, 0.300, 0.72])
    axC = fig.add_axes([0.768, 0.145, 0.210, 0.72])

    # A: Panel 38 gene ablation
    y = np.arange(len(ab))
    h_bar = 0.34
    is_lg = (ab["gene"] == "LGALSL").values
    axA.axvline(0, color=NAT_MID, lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    axA.barh(y + h_bar / 2, ab["only_dLL"], height=h_bar,
             color=[NAT_RED if g else NAT_BLUE for g in is_lg], edgecolor="none", label="Single-gene gain", zorder=3)
    axA.barh(y - h_bar / 2, ab["loss_on_drop_dLL"], height=h_bar,
             color=[NAT_RED_LT if g else NAT_BLUE_LT for g in is_lg], edgecolor="none", label="Leave-one-out drop", zorder=3)
    lg_idx = int(np.where(is_lg)[0][0])
    axA.text(ab.loc[lg_idx, "only_dLL"] + 0.0014, lg_idx + h_bar / 2, f"+{ab.loc[lg_idx, 'only_dLL']:.4f}",
             va="center", ha="left", fontsize=6.0, fontweight="bold", color=NAT_RED)
    axA.text(ab.loc[lg_idx, "loss_on_drop_dLL"] + 0.0014, lg_idx - h_bar / 2, f"{ab.loc[lg_idx, 'loss_on_drop_dLL']:.4f}",
             va="center", ha="left", fontsize=6.0, color=NAT_DARK)
    axA.set_yticks(y)
    axA.set_yticklabels([f"${g}$" for g in ab["gene"]], fontsize=6.6, color=NAT_DARK)
    axA.tick_params(axis="y", length=0, pad=3)
    axA.set_xlim(-0.004, 0.053)
    axA.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    axA.set_xlabel("5D composition ΔlogLoss")
    axA.set_title("Panel 38 gene ablation", fontweight="bold", pad=6, color=NAT_DARK)
    axA.legend(frameon=False, loc="lower right", fontsize=6.0, handlelength=1.1, borderaxespad=0.2)
    clean(axA)

    # B: Alternative null distributions (clean statistical densities, med/SD in legend)
    xs = np.linspace(-0.014, 0.026, 400)
    curves = [
        (null_fixed, "#4D4D4D", "-", 0.11),
        (null_adapt, NAT_TEAL, "--", 0.12),
        (null_redraw, NAT_RED, "-", 0.12),
    ]
    ymax_kde = 0.0
    for arr, col, ls, alp in curves:
        kde = gaussian_kde(arr)
        yv = kde(xs)
        ymax_kde = max(ymax_kde, float(yv.max()))
        axB.fill_between(xs, 0, yv, color=col, alpha=alp, zorder=1)
        axB.plot(xs, yv, color=col, ls=ls, lw=1.35, zorder=3)
        p95 = float(np.percentile(arr, 95))
        axB.plot([p95, p95], [0, kde(p95)[0] + 15], color=col, ls=":", lw=0.9, zorder=2.5)
        axB.scatter([p95], [0], marker="^", s=16, color=col, zorder=4, clip_on=False)

    axB.axvline(obs_p13, ymax=0.72, color=NAT_DARK, lw=1.4, zorder=4)
    axB.text(obs_p13 + 0.0009, ymax_kde * 0.56, f"Panel 13\nobs {obs_p13:.4f}",
             ha="left", va="center", fontsize=6.0, color=NAT_DARK, linespacing=1.28)
    axB.set_xlim(-0.014, 0.026)
    axB.set_ylim(0, ymax_kde * 1.42)
    axB.set_yticks([])
    axB.set_ylabel("Null density (Panel 13, B = 1,000)")
    axB.set_xlabel("1D composition ΔAUC")
    axB.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    axB.set_title("Alternative null distributions (29/42 always null)", fontweight="bold", pad=6, color=NAT_DARK)
    leg_h = [
        Line2D([], [], color="#4D4D4D", lw=1.35, label="Fixed (med 0.0064, SD 0.0063; 4/42*)"),
        Line2D([], [], color=NAT_TEAL, lw=1.35, ls="--", label="Fold-adaptive (med 0.0013, SD 0.0056; 11/42*)"),
        Line2D([], [], color=NAT_RED, lw=1.35, label="Per-fold redraw (med 0.0072, SD 0.0014; 13/42*)"),
        Line2D([], [], marker="^", ms=4.0, color="#4D4D4D", ls="none", label="95th percentile (*any baseline/metric)"),
    ]
    axB.legend(handles=leg_h, frameon=False, loc="upper left", fontsize=5.6, handlelength=1.3,
               bbox_to_anchor=(0.0, 0.995), borderaxespad=0.1, labelspacing=0.30)
    clean(axB)

    # C: External healthy-donor RNA-seq slope plot
    for pid, row in piv.iterrows():
        if int(pid) == 111:
            continue
        is_dir = bool(dir_map.loc[pid])
        col = NAT_BLUE2 if is_dir else "#B0B0B0"
        alp = 0.34 if is_dir else 0.28
        axC.plot([0, 1], [row["GSE173078"], row["GSE223924"]], color=col, lw=0.65, alpha=alp, zorder=1)
        axC.scatter([0, 1], [row["GSE173078"], row["GSE223924"]], s=7, color=col, alpha=alp + 0.15, linewidths=0, zorder=2)
    axC.plot([0, 1], [0.653, 0.970], color=NAT_DARK, lw=1.65, ls="--", zorder=4)
    axC.scatter([0, 1], [0.653, 0.970], marker="s", s=20, color=NAT_DARK, zorder=5)
    p38_1, p38_2 = float(piv.loc[111, "GSE173078"]), float(piv.loc[111, "GSE223924"])
    axC.plot([0, 1], [p38_1, p38_2], color=NAT_RED, lw=1.75, zorder=5)
    axC.scatter([0, 1], [p38_1, p38_2], s=24, color=NAT_RED, edgecolors="white", linewidths=0.4, zorder=6)
    axC.text(-0.08, 0.625, "1D 0.653", ha="right", va="top", fontsize=5.8, fontweight="bold", color=NAT_DARK)
    axC.text(-0.08, 0.715, "P38 0.688", ha="right", va="bottom", fontsize=5.8, fontweight="bold", color=NAT_RED)
    axC.text(1.07, 0.970, "0.970\n(both)", ha="left", va="center", fontsize=5.8, fontweight="bold", color=NAT_DARK, linespacing=1.40)
    axC.set_xlim(-0.74, 1.56)
    axC.set_ylim(0.32, 1.04)
    axC.set_xticks([0, 1])
    axC.set_xticklabels(["GSE173078\n(12 vs 12)", "GSE223924\n(10 vs 10)"], fontsize=6.0)
    axC.set_ylabel("Standalone ROC AUC")
    axC.set_title("External RNA-seq (n = 42)", fontweight="bold", pad=6, color=NAT_DARK)
    axC.legend(handles=[
        Line2D([], [], color=NAT_BLUE2, lw=1.0, alpha=0.7, label="Directed (22)"),
        Line2D([], [], color="#B0B0B0", lw=1.0, alpha=0.7, label="Other (20)"),
        Line2D([], [], color=NAT_DARK, lw=1.5, ls="--", label="1D axis"),
        Line2D([], [], color=NAT_RED, lw=1.6, label="Panel 38"),
    ], frameon=False, loc="lower right", fontsize=5.7, handlelength=1.2, borderaxespad=0.15, labelspacing=0.32)
    clean(axC)

    letter(fig, 0.008, 0.965, "A")
    letter(fig, 0.342, 0.965, "B")
    letter(fig, 0.712, 0.965, "C")
    save(fig, "Fig5_panel38_nulls_external")


# ----------------------------------------------------------------------------- Fig 6 (Nature palette spike-in curves)
def fig6():
    single = pd.read_csv(os.path.join(SP, "results_spikein.csv"))
    fam = pd.read_csv(os.path.join(SP, "results_spikein_family.csv"))
    fam = fam[np.isclose(fam.sig_auc, 0.90)]
    cross = pd.read_csv(os.path.join(SP, "results_spikein_family_crossings.csv"))
    summ = pd.read_csv(os.path.join(SP, "results_spikein_summary.csv"))
    expl = pd.read_csv(f"{V3}/results_per_panel_primary_with_status.csv")
    conf = pd.read_csv(f"{CONF}/results_per_test_primary.csv")

    rows = [("comp1d", "comp1d", "1D composition"), ("compfull", "compFull", "5D composition")]
    cols = [("logloss", "ΔlogLoss"), ("auc", "ΔAUC")]
    letters = [["A", "B"], ["C", "D"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.75), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.982, top=0.885, bottom=0.085, wspace=0.10, hspace=0.35)
    for r, (b, bb, blab) in enumerate(rows):
        for c, (m, mlab) in enumerate(cols):
            ax = axes[r, c]
            s1 = single[(single.baseline == bb) & (single.metric == m)].sort_values("w").reset_index(drop=True)
            s1 = s1.iloc[: int(s1["paired_inc_median"].idxmax()) + 1]
            s2 = fam[(fam.baseline == bb) & (fam.metric == m)].sort_values("w").reset_index(drop=True)
            s2 = s2.iloc[: int(s2["paired_inc_median"].idxmax()) + 1]

            ax.axvline(0.0, color="#CCCCCC", lw=0.7, ls=(0, (2, 2)), zorder=0)
            ax.axhline(0.8, color="#B8B8B8", lw=0.7, ls=(0, (3, 2)), zorder=0)
            ax.plot(s1.paired_inc_median, s1.det_p05, color=NAT_DARK, lw=1.35, zorder=2)
            ax.plot(s2.paired_inc_median, s2.det_fam15, color=NAT_AMBER, lw=1.35, ls=(0, (4.5, 1.8)), zorder=2)
            ax.plot(s2.paired_inc_median, s2.det_fam27, color=NAT_BLUE, lw=1.35, ls=(0, (1.4, 1.4)), zorder=2)

            ss = summ[(summ.baseline == bb) & (summ.metric == m)]
            if len(ss) and pd.notna(ss.inc80.iloc[0]):
                ax.plot(ss.inc80.iloc[0], 0.8, "o", ms=4.2, mfc="white", mec=NAT_DARK, mew=1.0, zorder=4)
            for thr, cc in (("fam15", NAT_AMBER), ("fam27", NAT_BLUE)):
                q = cross[(cross.sig == 0.90) & (cross.baseline == bb) & (cross.metric == m) & (cross.thr == thr)]
                if len(q) and pd.notna(q.inc80.iloc[0]):
                    ax.plot(q.inc80.iloc[0], 0.8, "o", ms=4.2, mfc="white", mec=cc, mew=1.0, zorder=4)

            e = (expl[f"d{m}_{b}_mean"] - expl[f"null_median_{m}_{b}"]).values
            cf = conf[(conf.baseline == b) & (conf.metric == m)]
            v = (cf.observed_mean - cf.null_median).values
            ax.vlines(e, -0.050, -0.016, color=NAT_BLUE2, lw=0.8, zorder=3)
            ax.vlines(v, -0.092, -0.058, color=NAT_RED, lw=0.85, zorder=3)
            x111 = float(v[list(cf.panel_id).index(111)])
            ax.annotate("Panel 38", xy=(x111, -0.055), xytext=(x111, 0.085), fontsize=6.1,
                        fontweight="bold", color=NAT_RED, ha="center",
                        arrowprops=dict(arrowstyle="-", lw=0.65, color=NAT_RED))

            ax.set_ylim(-0.11, 1.04)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
            clean(ax)
            ax.set_title(f"{blab}  ·  {mlab}", fontweight="bold", pad=5, fontsize=7.6, color=NAT_DARK)
            if c == 0:
                ax.set_ylabel("Empirical detection rate")
            ax.set_xlabel("Paired gain over matched random median", labelpad=2.5)

            pos = ax.get_position()
            letter(fig, pos.x0 - (0.055 if c == 0 else 0.025), pos.y1 + 0.038, letters[r][c])

    h = [Line2D([], [], color=NAT_DARK, lw=1.35, label="Single test (p < 0.05)"),
         Line2D([], [], color=NAT_AMBER, lw=1.35, ls=(0, (4.5, 1.8)), label="15-test Bonf."),
         Line2D([], [], color=NAT_BLUE, lw=1.35, ls=(0, (1.4, 1.4)), label="27-test Bonf."),
         Line2D([], [], marker="o", ls="none", mfc="white", mec=NAT_DARK, ms=4.2, label="80% crossing"),
         Line2D([], [], color=NAT_BLUE2, lw=1.3, label="Exploration (n=27)"),
         Line2D([], [], color=NAT_RED, lw=1.3, label="Confirmation (n=15)")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.992),
               handlelength=1.8, columnspacing=1.5, labelspacing=0.35)
    save(fig, "Fig6_spikein_detectability")


# ----------------------------------------------------------------------------- S1 Fig (Nature palette 2x3 concordance matrix, all n = 42)
def s1_fig():
    e = pd.read_csv(f"{V3}/results_per_panel_primary_with_status.csv")
    c = pd.read_csv(f"{CONF}/results_per_test_primary.csv")
    alt_e = pd.read_csv(f"{V3}/results_per_panel_seed20261031.csv")
    alt_c = pd.read_csv(f"{CONF}/results_per_test_seed20261031.csv")
    wil_e = pd.read_csv(f"{V3}/results_williams_sensitivity_B1000.csv")
    wil_c = pd.read_csv(f"{CONF}/results_per_test_williams.csv")

    prim_rows = []
    for _, r in e.iterrows():
        for bl in ("comp1d", "compfull"):
            prim_rows.append((int(r.panel_id), "exp", bl, float(r[f"dlogloss_{bl}_mean"])))
    for _, r in c.iterrows():
        if r.baseline in ("comp1d", "compfull") and r.metric == "logloss":
            prim_rows.append((int(r.panel_id), "conf", str(r.baseline), float(r.observed_mean)))
    prim42 = pd.DataFrame(prim_rows, columns=["panel_id", "stage", "baseline", "prim"])

    # Build 42-panel alt CV (1D) and GSE164241 (12D vs primary 5D) tables
    alt_rows = [(int(r.panel_id), float(r["dlogloss_comp1d_mean"])) for _, r in alt_e.iterrows()] + [
        (int(r.panel_id), float(r.observed_mean))
        for _, r in alt_c[(alt_c.baseline == "comp1d") & (alt_c.metric == "logloss")].iterrows()
    ]
    alt42 = pd.DataFrame(alt_rows, columns=["panel_id", "observed_mean"])
    alt42_m = prim42[prim42.baseline == "comp1d"].merge(alt42, on="panel_id")

    wil_rows = [(int(r.panel_id), float(r["dlogloss_williams_derived_full_mean"])) for _, r in wil_e.iterrows()] + [
        (int(r.panel_id), float(r.observed_mean))
        for _, r in wil_c[(wil_c.baseline == "williams_derived_full") & (wil_c.metric == "logloss")].iterrows()
    ]
    wil42 = pd.DataFrame(wil_rows, columns=["panel_id", "observed_mean"])
    wil42_m = prim42[prim42.baseline == "compfull"].merge(wil42, on="panel_id")

    rma = pd.read_csv(os.path.join(BR, "results_per_test_rma_v2.csv"))
    no08 = pd.read_csv(os.path.join(BR, "results_per_test_no2008_v2.csv"))
    mvl2 = pd.read_csv(os.path.join(MV, "results_multivariate_l2.csv"))
    maxm = pd.read_csv(os.path.join(MV, "results_probe_maxmean.csv"))

    def prep42(df, bl):
        sub = df[(df.baseline == bl) & (df.metric == "logloss")][["panel_id", "observed_mean"]]
        m = prim42[prim42.baseline == bl].merge(sub, on="panel_id")
        return m["prim"].values, m["observed_mean"].values, m["stage"].values, m["panel_id"].values

    panels = [
        ("A", "Alternative CV split (1D)", "Primary 1D ΔlogLoss", "Alt. CV 1D ΔlogLoss",
         alt42_m["prim"].values, alt42_m["observed_mean"].values, alt42_m["stage"].values, alt42_m["panel_id"].values),
        ("B", "12-comp GSE164241 (5D)", "Primary 5D ΔlogLoss", "GSE164241 12-comp ΔlogLoss",
         wil42_m["prim"].values, wil42_m["observed_mean"].values, wil42_m["stage"].values, wil42_m["panel_id"].values),
        ("C", "De novo CEL-level RMA (5D)", "Primary 5D ΔlogLoss", "RMA 5D ΔlogLoss",
         *prep42(rma, "compfull")),
        ("D", "Exclude 2008 scan batch (5D)", "Primary 5D ΔlogLoss", "2004–2007 5D ΔlogLoss",
         *prep42(no08, "compfull")),
        ("E", "Multivariate gene L2 (5D)", "Primary 5D ΔlogLoss", "Multivariate L2 5D ΔlogLoss",
         *prep42(mvl2, "compfull")),
        ("F", "Label-free max-mean probe (5D)", "Primary 5D ΔlogLoss", "Max-mean probe 5D ΔlogLoss",
         *prep42(maxm, "compfull")),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.95))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.095, wspace=0.34, hspace=0.40)

    for idx, (let, title, xlab, ylab, xv, yv, stg, pids) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        r = float(np.corrcoef(xv, yv)[0, 1])
        lo, hi = min(float(xv.min()), float(yv.min())), max(float(xv.max()), float(yv.max()))
        pad = 0.08 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#8F8F8F", lw=0.7, ls=(0, (3, 2)), zorder=1)
        is_exp = (stg == "exp")
        is_conf = (stg == "conf")
        ax.scatter(xv[is_exp], yv[is_exp], s=14, color=NAT_BLUE2, alpha=0.82, linewidths=0, zorder=2)
        if is_conf.any():
            ax.scatter(xv[is_conf], yv[is_conf], s=17, color=NAT_RED, alpha=0.92,
                       edgecolors="white", linewidths=0.35, zorder=3)
            i38 = int(np.where(pids == 111)[0][0])
            ax.annotate("P38", xy=(xv[i38], yv[i38]), xytext=(xv[i38] - 0.17 * (hi - lo), yv[i38] - 0.02 * (hi - lo)),
                        fontsize=5.9, fontweight="bold", color=NAT_RED, ha="right", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.55, color=NAT_RED))

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.text(0.05, 0.94, f"r = {r:.4f}\nn = {len(xv)}", transform=ax.transAxes,
                va="top", ha="left", fontsize=6.2, linespacing=1.2, color=NAT_DARK)
        ax.set_title(title, fontsize=7.2, fontweight="bold", pad=4, color=NAT_DARK)
        ax.set_xlabel(xlab, fontsize=6.4, labelpad=2)
        ax.set_ylabel(ylab, fontsize=6.4, labelpad=2)
        ax.tick_params(labelsize=6.0)
        clean(ax)
        pos = ax.get_position()
        letter(fig, pos.x0 - 0.062, pos.y1 + 0.036, let)
        print(f"  S1 {let} {title}: r = {r:.4f} (n={len(xv)})")

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=NAT_BLUE2, ms=4.2, label="Exploration panels (n = 27)"),
        Line2D([], [], marker="o", ls="none", color=NAT_RED, ms=4.5, label="Confirmation panels (n = 15; P38 labeled)"),
        Line2D([], [], color="#8F8F8F", lw=0.8, ls=(0, (3, 2)), label="Identity (y = x)"),
    ], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.992), columnspacing=1.5)
    save(fig, "S1_Fig_sensitivity_concordance")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    s1_fig()
