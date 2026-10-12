"""Emits the LaTeX table bodies of the revised manuscript from ../results/*.json
into ../tables/*.tex.  build.py pastes them into paper.tex, so no number in a
table is typed by hand.  Percentages carry two significant figures (whole
numbers from 10 % upward)."""
import json
import math
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
O = Path(__file__).resolve().parent.parent / "tables"
O.mkdir(exist_ok=True)
T_ROUND_MS = 117.0

NAME = {"and": "Binary AND~\\cite{soula2024iodt}",
        "and_db": "Debounced AND",
        "kof4_p": "2-of-4 rule",
        "beta": "Beta reputation",
        "weighted": "Weighted, no EMA",
        "ema": "Weighted + EMA",
        "ema_p": "\\textbf{Extension~I}",
        "ema_p_long": "Ext.~I + 8-of-32 window"}


SHORT = {"and": "AND~\\cite{soula2024iodt}", "and_db": "Debounced AND",
         "kof4_p": "2-of-4 rule", "ema": "EMA only", "ema_p": "\\textbf{Extension~I}",
         "ema_p_long": "Ext.~I + 8 of 32"}


def load(n):
    return json.loads((R / f"{n}.json").read_text())


def sig(v):
    """v already in percent; two significant figures, integers from 10."""
    if v == 0:
        return "0"
    if 99.5 <= v < 100:
        return f"{v:.1f}" if v < 99.95 else "$>$99.9"
    if v >= 10:
        return f"{v:.0f}"
    e = math.floor(math.log10(abs(v)))
    d = max(0, 1 - e)
    s = f"{v:.{d}f}"
    if float(s) >= 10:
        return f"{float(s):.0f}"
    return s


def p(x):
    return sig(100 * x)


def pz(x, bound):
    """percentage, or a zero with its one-sided 95 % bound."""
    return f"0 ($<${sig(100*bound)})" if x == 0 else p(x)


def ci(c):
    return f"[{sig(100*c[0])},\\,{sig(100*c[1])}]"


def put(name, s):
    (O / f"{name}.tex").write_text(s.rstrip("\n") + "\n")
    print("wrote", name)


def sci(x):
    if x is None:
        return "--"
    e = int(math.floor(math.log10(x)))
    m = x / 10 ** e
    if e < 4:
        v = round(x, -(e - 1)) if e >= 1 else round(x, 1 - e)
        if e < 1:
            return f"{v:.{1 - e}f}"
        return f"{v:,.0f}".replace(",", "{,}")
    return f"${m:.1f}\\times10^{{{e}}}$"


# ------------------------------------------------------------------ headline
d = load("e0_headline")
qa, qb = d["q_read_a"], d["q_read_b"]
lines = []
labels = {qb: f"$q={qb*1e5:.1f}\\times10^{{-5}}$ (reading B of~\\cite{{soula2024iodt}})",
          qa: "$q=0.001$ (reading A of~\\cite{soula2024iodt})",
          0.005: "$q=0.005$", 0.02: "$q=0.02$", 0.04: "$q=0.04$ (stress level)"}
for i, q in enumerate([qb, qa, 0.005, 0.02, 0.04]):
    if i:
        lines.append("\\midrule")
    lines.append(f"\\multicolumn{{6}}{{l}}{{{labels[q]}}} \\\\")
    for v in ["and", "and_db", "kof4_p", "ema_p"]:
        r = [x for x in d["rows"] if x["q"] == q and x["variant"] == v][0]
        zb = r["zero_upper"]
        fpr = pz(r["fpr"], zb) + ("" if r["fpr"] == 0 else f" {ci(r['fpr_ci'])}")
        fi = pz(r["false_iso"], zb) + ("" if r["false_iso"] == 0 else f" {ci(r['false_iso_ci'])}")
        lines.append(f"{NAME[v]} & {fpr} & {fi} & {r['ttd1_rounds']:.2f} & "
                     f"{p(r['detW1'])} & {r['ttdW1']:.1f} \\\\")
put("tab_headline", "\n".join(lines))

# ------------------------------------------------------------------ long horizon
d = load("e1_long")["rows"]
lines = []
for i, q in enumerate([0.001, 0.005, 0.04]):
    if i:
        lines.append("\\midrule")
    lines.append(f"\\multicolumn{{5}}{{l}}{{$q={q}$}} \\\\")
    for a in ["and", "and_db", "kof4_p", "ema_p"]:
        r = [x for x in d if x["q"] == q and x["agg"] == a][0]
        fi = r["false_iso"][r["horizons"].index(1024)]
        mt = r["mttfi_rounds"]
        lo = r["mttfi_rounds_lower"]
        mtt = sci(mt) if r["onsets"] >= 10 else f"$\\geq${sci(lo)}"
        lines.append(f"{SHORT[a]} & {p(fi)} & {mtt} & {p(r['unavailability'])} & "
                     f"{sig(r['false_quarantines_per_node_hour'])} \\\\")
put("tab_long", "\n".join(lines))

# ------------------------------------------------------------------ stealth
d = load("e2_stealth")["rows"]
lines = []
for i, q in enumerate([0.001, 0.04]):
    if i:
        lines.append("\\midrule")
    fi = {a: [x for x in d if x["q"] == q and x["rounds"] == 16 and x["agg"] == a][0]["false_iso"]
          for a in ["and", "and_db", "kof4_p", "ema_p"]}
    short = {"and": "AND", "and_db": "debounced", "kof4_p": "2-of-4", "ema_p": "Ext.~I"}
    lines.append(f"\\multicolumn{{9}}{{l}}{{$q={q}$}} \\\\")
    for pl in [0.5, 0.25, 0.10]:
        cells = []
        for prof in ["TW", "W"]:
            for a in ["and", "and_db", "kof4_p", "ema_p"]:
                r = [x for x in d if x["q"] == q and x["profile"] == prof and x["rounds"] == 16
                     and x["p"] == pl and x["agg"] == a][0]
                cells.append(p(r["det1"]))
        lines.append(f"{pl:g} & " + " & ".join(cells) + " \\\\")
put("tab_stealth", "\n".join(lines))
# 64-round numbers used in the text
txt = []
for q in [0.001, 0.04]:
    for prof in ["TW", "W"]:
        for pl in [0.25, 0.10]:
            r = [x for x in d if x["q"] == q and x["profile"] == prof and x["rounds"] == 64
                 and x["p"] == pl and x["agg"] == "ema_p"][0]
            txt.append(f"% q={q} {prof} p={pl} Ext.I 64 rounds: det {p(r['det1'])} fi {p(r['false_iso'])}")
put("txt_stealth64", "\n".join(txt))

# ------------------------------------------------------------------ rule-aware
aw = load("e2_aware")
ra = load("e1_readmit")["rows"]
lines = []
for dn in ["ema_p", "ema_p_long", "and"]:
    cells = []
    for prof in ["TW", "W"]:
        r = [x for x in aw["rows"] if x["profile"] == prof and x["q"] == 0.0 and x["rounds"] == 256
             and x["defence"] == dn][0]
        cells.append(f"{100*r['lie_rate']:.1f}" if r["det1"] == 0 else "0")
    for prof in ["TW", "W"]:
        r = [x for x in aw["rows"] if x["profile"] == prof and x["q"] == 0.001 and x["rounds"] == 256
             and x["defence"] == dn][0]
        cells.append(p(r["det1"]))
    for q in [0.001, 0.04]:
        r = [x for x in ra if x["q"] == q and x["attacker"] == "aware_TW" and x["agg"] == dn][0]
        cells.append(p(r["lies_delivered_per_round"]))
    r = [x for x in ra if x["q"] == 0.04 and x["attacker"] == "aware_TW" and x["agg"] == dn][0]
    cells.append(p(r["unavailability"]))
    lines.append(f"{SHORT[dn]} & " + " & ".join(cells) + " \\\\")
put("tab_aware", "\n".join(lines))

# ------------------------------------------------------------------ corrupted twins
d = load("e3_collusion")
rows = {r["setting"]: r for r in d["rows"]}
lines = []


def crow(label, r):
    nq = r.get("no_quorum_rounds")
    return (f"{label} & {p(r['det1'])} & {p(r['tpr1_steady'])} & {p(r['fpr'])} & "
            f"{p(r['false_iso'])} & {('--' if nq is None else p(nq))} \\\\")


for key, label in [("single twin, honest", "Single twin, honest"),
                   ("single twin, corrupted", "Single twin, corrupted"),
                   ("5 twins, c=2, framing", "3-of-5, $c=2$ shield + frame"),
                   ("5 twins, c=3, framing", "3-of-5, $c=3$ shield + frame"),
                   ("7 twins, c=3, framing", "4-of-7, $c=3$ shield + frame")]:
    lines.append(crow(label, rows[key]))
lines.append("\\midrule")
mm = d["missing"]
for (rule, pm, c, label) in [("quorum", 0.05, 2, "$c=2$ withhold, 5\\% timeouts"),
                             ("quorum", 0.20, 0, "20\\% timeouts"),
                             ("quorum", 0.20, 2, "$c=2$ withhold, 20\\% timeouts")]:
    r = [x for x in mm if x["rule"] == rule and x["p_miss"] == pm and x["c"] == c][0]
    lines.append(crow(label, r))
put("tab_collusion", "\n".join(lines))

# ------------------------------------------------------------------ audit log
d = load("e4_audit")
lines = []
for r in d["rows"]:
    lines.append(f"{r['K']} & {r['blocks']} & {r['onchain']:,} & {r['offchain']:,} & "
                 f"{r['max_block']} \\\\".replace(",", "{,}"))
put("tab_chain", "\n".join(lines))
tam = [t for t in d["tamper"] if t["K"] == 10]
KIND = [("verdict", "Verdict (with its votes)"), ("tau", "Trust value $\\tau_t$"),
        ("state", "State record"), ("vote", "Signed vote"),
        ("interior", "Interior-node substitution"), ("duplicate", "Duplicated event"),
        ("truncate", "Truncated window"), ("replay", "Replayed vote")]
lines = []
for k, label in KIND:
    def g(adv, col):
        x = [t for t in tam if t["kind"] == k and t["adversary"] == adv]
        return "--" if not x or x[0][col] is None else p(x[0][col])
    lines.append(f"{label} & {g('edit_only','det_none')} & {g('recompute','det_none')} & "
                 f"{g('k-1_keys','det_none')} & {g('k_keys','det_none')} & "
                 f"{g('k_keys','det_older')} & {g('k_keys','det_latest')} \\\\")
put("tab_tamper", "\n".join(lines))

# ------------------------------------------------------------------ latency
L = load("e4_latency")
lines = [
    f"EMA update, one node & \\multicolumn{{2}}{{c}}{{{L['ema_update_ms']:.4f}}} \\\\",
    f"Replica: sign one vote vector & {L['replica_batched_sign_8_ms']:.2f} & {L['replica_batched_sign_100_ms']:.2f} \\\\",
    f"Engine: verify $5$ vote vectors & {L['engine_batched_8_ms']:.1f} & {L['engine_batched_100_ms']:.1f} \\\\",
]
put("tab_latency", "\n".join(lines))

# ------------------------------------------------------------------ public traces
d = load("e7_trace")
lines = []
for i, (ds, lab) in enumerate([("intel", "Intel Lab"), ("flocklab", "FlockLab")]):
    if i:
        lines.append("\\midrule")
    dd = d[ds]
    rows = {(x["r"], x["agg"]): x for x in dd["rows"]}
    iid = {(x["r"], x["agg"]): x for x in dd["iid"]}
    for r in sorted({x["r"] for x in dd["rows"]}):
        e = rows[(r, "ema_p")]
        cells = [p(rows[(r, a)]["false_iso"]) for a in ["and", "and_db", "kof4_p", "ema_p"]]
        cells[-1] = "\\textbf{" + cells[-1] + "}"
        lines.append(f"{lab if r == min(x['r'] for x in dd['rows']) else ''} & {r} & {p(e['q'])} & "
                     + " & ".join(cells)
                     + f" & {p(e['false_iso_typical'])} & {p(iid[(r, 'ema_p')]['false_iso'])} \\\\")
put("tab_trace", "\n".join(lines))

print("tables done")
