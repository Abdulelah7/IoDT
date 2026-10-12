"""Regenerates every data figure of the revised manuscript from ../results.
Figures are drawn at their printed size (3.5 in single column, 7.16 in double
column) so that all text is at least 8 pt in print."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
F = Path(__file__).resolve().parent.parent / "figs"
F.mkdir(exist_ok=True)

# fixed categorical order (validated palette, light surface); every series
# also has its own marker so the figures survive grey-scale printing
STY = {
    "ema_p":    dict(color="#2a78d6", marker="o", label="Extension I"),
    "and":      dict(color="#eb6834", marker="s", label="Binary AND [4]"),
    "ema":      dict(color="#1baf7a", marker="^", label="Weighted + EMA"),
    "weighted": dict(color="#eda100", marker="D", label="Weighted, no EMA"),
    "and_db":   dict(color="#e87ba4", marker="v", label="Debounced AND"),
    "kof4_p":   dict(color="#008300", marker="X", label="2-of-4 rule"),
    "ema_p_supp": dict(color="#6250d6", marker="P", label="Ext. I + domain suppression"),
}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4, "legend.frameon": False,
    "savefig.dpi": 400, "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
})
SC, DC = 3.5, 7.16


def load(name):
    return json.loads((R / f"{name}.json").read_text())


def pctfmt(ax):
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))


# --- q sweep ---------------------------------------------------------------
d = load("e1_qsweep")
h = load("e0_headline")
qa, qb = h["q_read_a"], h["q_read_b"]
fig, ax = plt.subplots(2, 1, figsize=(SC, 4.9), sharex=True)
for a in ["and", "and_db", "kof4_p", "ema_p"]:
    rows = sorted([r for r in d["rows"] if r["agg"] == a], key=lambda r: r["q"])
    nz = [(r["q"], 100 * r["fpr"]) for r in rows if r["fpr"] > 0]
    ax[0].plot([x for x, _ in nz], [y for _, y in nz], **STY[a])
    ax[1].plot([r["q"] for r in rows], [100 * r["false_iso"] for r in rows], **STY[a])
qp = [p["q"] for p in d["prediction"]]
ax[0].plot(qp, [100 * p["and_"] for p in d["prediction"]], ls="--", lw=0.8,
           color="#52514e", label="AND, closed form")
ax[0].set_yscale("log"); ax[0].set_xscale("log")
ax[0].set_ylim(5e-5, 100)
ax[0].set_ylabel("per-round FPR (%)")
ax[1].set_ylabel("false isolation in 16 rounds (%)")
ax[1].set_xlabel("per-factor congestion rate $q$")
for a_ in ax:
    for qv in (qa, 0.04):
        a_.axvline(qv, color="#c3c2b7", lw=0.7, ls=":")
ax[0].text(qa * 1.08, 30, "reading A", fontsize=7, color="#52514e")
ax[0].text(0.04 * 1.08, 30, "stress", fontsize=7, color="#52514e")
ax[0].yaxis.set_major_locator(mt.LogLocator(numticks=8))
ax[0].yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
hd, lb = ax[0].get_legend_handles_labels()
fig.legend(hd, lb, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.15, 1, 1))
fig.savefig(F / "fig_qsweep.png"); plt.close(fig)

# --- horizon ---------------------------------------------------------------
d = load("e1_long")["rows"]
fig, ax = plt.subplots(2, 1, figsize=(SC, 4.4), sharex=True)
for k, q in enumerate([0.005, 0.04]):
    for a in ["and", "and_db", "kof4_p", "ema_p"]:
        r = [x for x in d if x["q"] == q and x["agg"] == a][0]
        ax[k].plot(r["horizons"], [100 * v for v in r["false_iso"]], **STY[a])
    ax[k].set_xscale("log", base=2)
    ax[k].set_ylabel("false isolation (%)")
    ax[k].set_title(f"$q={q}$", loc="left")
ax[1].set_xlabel("horizon (rounds)")
ax[1].xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
hd, lb = ax[0].get_legend_handles_labels()
fig.legend(hd, lb, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(F / "fig_horizon.png"); plt.close(fig)

# --- robustness 2x2 ----------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(DC, 4.9))
b = load("e6_bursty")["rows"]
for a in ["and", "and_db", "kof4_p", "ema_p"]:
    rr = sorted([r for r in b if r["agg"] == a and r["scope"] == "node"], key=lambda r: r["dwell"])
    ax[0, 0].plot([r["dwell"] for r in rr], [100 * r["false_iso"] for r in rr], **STY[a])
ax[0, 0].set_xscale("log", base=2); ax[0, 0].set_yscale("log")
ax[0, 0].xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
pctfmt(ax[0, 0])
ax[0, 0].set_xlabel("mean burst length (rounds); mean $q$ fixed at 0.04")
ax[0, 0].set_ylabel("false isolation (%)")
ax[0, 0].set_title("(a) bursty congestion, per node", loc="left")

c = load("e6_corr")["rows"]
for a in ["and", "and_db", "kof4_p", "ema_p", "ema_p_supp"]:
    rr = sorted([r for r in c if r["agg"] == a and r["q_corr"] > 0], key=lambda r: r["q_corr"])
    ax[0, 1].plot([r["q_corr"] for r in rr], [100 * r["false_iso"] for r in rr], **STY[a])
ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log"); pctfmt(ax[0, 1])
ax[0, 1].set_xlabel("common-mode event rate $q_{\\mathrm{corr}}$ per fog domain")
ax[0, 1].set_ylabel("false isolation (%)")
ax[0, 1].set_title("(b) fog-level common-mode events, $q=0.04$", loc="left")

x = load("e6_xfactor")["rows"]
for a in ["and", "and_db", "ema_p"]:
    rr = sorted([r for r in x if r["agg"] == a and r["fpr"] > 0], key=lambda r: r["q_node"])
    ax[1, 0].plot([r["q_node"] for r in rr], [100 * r["fpr"] for r in rr], **STY[a])
qq = np.array([0.0025, 0.04])
ax[1, 0].plot(qq, 0.22 * (qq / 0.04) ** 2, ls="--", lw=0.8, color="#52514e")
ax[1, 0].text(0.016, 0.0016, "slope 2", fontsize=7, color="#52514e")
ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); pctfmt(ax[1, 0])
ax[1, 0].set_xlabel("rate of events failing $P$, $T$, $W$ together")
ax[1, 0].set_ylabel("per-round FPR (%)")
ax[1, 0].set_title("(c) cross-factor events, no other noise", loc="left")

u = load("e6_uneven")["rows"]
profs = ["uniform", "P-heavy", "E-heavy", "W-heavy"]
aggs = ["and", "and_db", "kof4_p", "ema_p"]
wbar = 0.18
for j, a in enumerate(aggs):
    vals = [100 * [r for r in u if r["profile"] == pr and r["agg"] == a][0]["false_iso"] for pr in profs]
    ax[1, 1].bar(np.arange(4) + (j - 1.5) * wbar, vals, width=wbar * 0.88,
                 color=STY[a]["color"], label=STY[a]["label"])
ax[1, 1].set_yscale("log"); pctfmt(ax[1, 1])
ax[1, 1].set_xticks(range(4))
ax[1, 1].set_xticklabels(["uniform", "PDR 0.13", "energy 0.13", "notif. 0.13"])
ax[1, 1].set_ylabel("false isolation (%)")
ax[1, 1].set_title("(d) uneven congestion, mean $q=0.04$", loc="left")
ax[1, 1].grid(axis="x", visible=False)

for a_ in ax.ravel():
    a_.xaxis.set_minor_formatter(mt.NullFormatter())
    a_.yaxis.set_minor_formatter(mt.NullFormatter())
ax[1, 0].set_xticks([0.0025, 0.005, 0.01, 0.02, 0.04])
ax[1, 0].xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
ax[0, 1].xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
hd, lb = ax[0, 1].get_legend_handles_labels()
fig.legend(hd, lb, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(F / "fig_robust.png"); plt.close(fig)

print("figures written to", F)
