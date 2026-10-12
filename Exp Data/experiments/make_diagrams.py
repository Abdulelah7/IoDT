"""Draws the system-architecture and per-round sequence diagrams (Figs. 1-2)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

F = Path(__file__).resolve().parent.parent / "figs"
INK, MUTED = "#1f1f1f", "#52514e"
BLUE, BLUE_L = "#2a78d6", "#e6f0fb"
ORANGE, ORANGE_L = "#d95926", "#fdeee6"
GREEN, GREEN_L = "#138a5e", "#e4f5ee"
GREY_L = "#f2f1ee"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, fc, ec, fs=7.5, bold=False, ls="-", r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=1.0, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="normal", linespacing=1.15)


def arrow(ax, p, q, color=INK, ls="-", lw=1.0, rad=0.0, text=None, toff=(0, 0), fs=6.6,
          tcol=None, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, fontsize=fs,
                color=tcol or color, ha=ha, va="center")


# =========================================================================
# Fig. 1 - architecture
# =========================================================================
fig, ax = plt.subplots(figsize=(7.16, 3.55))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

# layer bands
for y0, h, lab in [(0.05, 1.05, "Sensors"), (1.25, 0.85, "Fog\nlayer"),
                   (2.25, 2.7, "Cloud\nlayer")]:
    ax.add_patch(FancyBboxPatch((0.05, y0), 9.9, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=GREY_L, ec="none"))
    ax.text(0.2, y0 + h / 2, lab.replace("\n", " "), fontsize=7.0, color=MUTED, va="center",
            ha="center", fontweight="normal", rotation=90)

# sensors
xs = [1.25 + i * 0.62 for i in range(8)]
names = ["N1", "N2", "N3", "N4", "N5", "N6", "M1", "M2"]
for x, n in zip(xs, names):
    att = n.startswith("C")
    box(ax, x, 0.35, 0.5, 0.42, n, ORANGE_L if att else "white", ORANGE if att else MUTED,
        fs=7, r=0.04)
ax.text(3.45, 0.18, "IoT sensors report $E$, PDR, $T$, $W$ evidence every round "
        "(M1, M2 = compromised)", fontsize=6.6, color=MUTED, ha="center")

# fog
box(ax, 1.25, 1.42, 2.2, 0.5, "Fog node 1\n(relay, isolation actuator)", "white", MUTED, fs=6.8)
box(ax, 3.75, 1.42, 2.2, 0.5, "Fog node 2\n(relay, isolation actuator)", "white", MUTED, fs=6.8)
for x in xs[:4]:
    arrow(ax, (x + 0.25, 0.79), (2.35, 1.40), color=MUTED, lw=0.7)
for x in xs[4:]:
    arrow(ax, (x + 0.25, 0.79), (4.85, 1.40), color=MUTED, lw=0.7)

# replicas
box(ax, 1.0, 2.45, 5.2, 1.3, "", BLUE_L, BLUE, r=0.04)
ax.text(3.6, 3.57, "Ext. II: $n$ replicas, each running Ext. I on the same evidence",
        fontsize=7.2, ha="center", color=BLUE, fontweight="normal")
for i in range(5):
    x = 1.18 + i * 0.99
    corrupt = i == 0
    box(ax, x, 2.6, 0.86, 0.75,
        f"DT$_{i+1}$\nEMA +\npersistence" + ("\n(corrupted)" if corrupt else ""),
        "#fbe3dc" if corrupt else "white", ORANGE if corrupt else BLUE, fs=5.9, r=0.03)
arrow(ax, (2.35, 1.94), (2.35, 2.43), color=INK, text="evidence $\\mathbf{x}_t(i)$",
      toff=(0.48, 0), fs=6.4)
arrow(ax, (4.85, 1.94), (4.85, 2.43), color=INK)

# trust engine
box(ax, 1.0, 4.05, 5.2, 0.7,
    "Trust engine (no ledger access): verify signed vote vectors; quorum rule\nTRUST if $\\geq k$ TRUST votes, FLAG if $\\geq n-k+1$ FLAG votes, otherwise hold",
    "white", INK, fs=6.4)
arrow(ax, (3.6, 3.77), (3.6, 4.03), color=BLUE, lw=1.2,
      text="signed vote vectors $(v_r,\\sigma_r)$", toff=(0.75, 0), fs=6.4)

# decision back to fog (curved, left side)
ax.add_patch(FancyArrowPatch((1.0, 4.40), (1.25, 1.70), arrowstyle="-|>", mutation_scale=8,
                             color=GREEN, lw=1.3, connectionstyle="arc3,rad=0.45"))
ax.text(0.6, 2.2, "verdict to the\nnode's fog node", fontsize=5.6, color=GREEN, ha="left",
        fontweight="normal")

# off-path audit
box(ax, 6.75, 2.45, 3.05, 2.3, "", GREEN_L, GREEN, r=0.04)
ax.text(8.27, 4.55, "Ext. III: signed Merkle audit log", fontsize=7.2, ha="center",
        color=GREEN, fontweight="normal")
box(ax, 6.95, 3.6, 2.65, 0.7, "Off-chain store: events (factors,\n$\\tau_t$, verdict), signed votes, state",
    "white", GREEN, fs=6.4)
box(ax, 6.95, 2.6, 2.65, 0.75,
    "Every $K$ rounds: header with Merkle roots\nof events, votes and state, endorsed by\n$k$ replicas; append-only log or ledger", "white", GREEN, fs=6.2)
arrow(ax, (8.27, 3.58), (8.27, 3.37), color=GREEN)
arrow(ax, (6.22, 4.40), (6.93, 3.95), color=GREEN, ls=(0, (3, 2)), lw=1.0)
box(ax, 6.95, 1.42, 2.65, 0.5, "External checkpoint\n(timestamping / KSI / public chain)",
    "white", MUTED, fs=6.4, ls=(0, (3, 2)))
arrow(ax, (8.27, 2.58), (8.27, 1.94), color=MUTED, ls=(0, (3, 2)), text="header hash",
      toff=(0.45, 0), fs=6.2, tcol=MUTED)

# legend
ax.plot([6.95, 7.3], [0.62, 0.62], color=INK, lw=1.2)
ax.text(7.38, 0.62, "real-time decision path", fontsize=6.4, va="center")
ax.plot([6.95, 7.3], [0.32, 0.32], color=GREEN, lw=1.2, ls=(0, (3, 2)))
ax.text(7.38, 0.32, "off-path (audit only, every $K$ rounds)", fontsize=6.4, va="center")
fig.tight_layout(pad=0.1)
fig.savefig(F / "fig_architecture.png", dpi=300)
plt.close(fig)

# =========================================================================
# Fig. 2 - per-round sequence
# =========================================================================
fig, ax = plt.subplots(figsize=(7.16, 3.35))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
lanes = [("Sensor $i$", 0.75), ("Fog node", 2.3), ("Twin replicas\nDT$_1..$DT$_n$", 3.95),
         ("Trust\nengine", 5.6), ("Off-chain\nstore", 7.15), ("Audit log +\ncheckpoint", 8.85)]
for lab, x in lanes:
    box(ax, x - 0.62, 5.55, 1.24, 0.55, lab, "white", INK, fs=6.6)
    ax.plot([x, x], [0.15, 5.53], color="#bdbcb6", lw=0.8, ls=(0, (2, 2)))
X = dict(zip(["s", "f", "r", "t", "o", "l"], [x for _, x in lanes]))

# decision-path shading
ax.add_patch(FancyBboxPatch((0.05, 2.5), 6.4, 2.95, boxstyle="round,pad=0,rounding_size=0.06",
                            fc=BLUE_L, ec="none", zorder=0))
ax.text(6.38, 5.32, "real-time decision path (every round)", fontsize=6.6, color=BLUE,
        fontweight="normal", ha="right")
ax.add_patch(FancyBboxPatch((6.5, 0.15), 3.45, 2.25, boxstyle="round,pad=0,rounding_size=0.06",
                            fc=GREEN_L, ec="none", zorder=0))
ax.text(9.85, 2.25, "off-path (no effect on latency)", fontsize=6.6, color=GREEN,
        fontweight="normal", ha="right")


def msg(y, a, b, text, color=INK, ls="-", above=True):
    arrow(ax, (X[a], y), (X[b], y), color=color, ls=ls, lw=1.0)
    ax.text((X[a] + X[b]) / 2, y + (0.11 if above else -0.13), text, fontsize=6.3,
            ha="center", va="center", color=color)


def act(y, a, text, color=INK, left=False):
    ax.text(X[a] + (-0.1 if left else 0.1), y, text, fontsize=6.2,
            ha="right" if left else "left", va="center", color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.7))


msg(5.2, "s", "f", "1  factor evidence")
msg(4.85, "f", "r", "2  $\\mathbf{x}_t(i)$ to each replica")
act(4.45, "r", "3  Alg. 1: $s_t$, $\\tau_t$, persistence")
msg(3.95, "r", "t", "4  signed vote vectors $(v_r,\\sigma_r)$", color=BLUE)
act(3.55, "t", "5  Alg. 2: verify sigs,\nquorum rule", left=True)
msg(3.1, "t", "f", "6  verdict for node $i$", color=GREEN)
msg(2.75, "f", "s", "7  fog isolates $i$ if flagged", color=GREEN)
msg(2.05, "t", "o", "8  signed votes (fog node logs the verdict)", color=GREEN, ls=(0, (3, 2)))
act(1.62, "o", "9  every $K$ rounds: header formed,\nchecked and signed by $k$ replicas", color=GREEN)
msg(0.98, "o", "l", "10  header + $k$ endorsements", color=GREEN, ls=(0, (3, 2)))
act(0.45, "l", "11  append; periodically\npublish hash (C1)", color=GREEN, left=True)
fig.tight_layout(pad=0.1)
fig.savefig(F / "fig_sequence.png", dpi=300)
plt.close(fig)
print("diagrams written")
