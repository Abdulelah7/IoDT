"""
Runs every experiment reported in the manuscript
and writes one JSON file per experiment to ../results/.

    python3 run_all.py              # everything
    python3 run_all.py e0_headline  # one experiment

Seeds: every reported cell uses seed 0 (REPORT_SEED).  Every parameter that
was chosen by a rule (the defaults of Extension I, the persistence window and
the beta baseline) is selected on seed 1 (TUNE_SEED); see e5_tuning.
"""
from __future__ import annotations

import copy
import json
import platform
import sys
import time
from dataclasses import asdict
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim"))
from iodt_sim import (Config, simulate, quarantine, wilson, boot_ci,  # noqa: E402
                      WEIGHTS)
import audit_chain as ac                                              # noqa: E402

OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)
T_ROUND_MS = 117.0      # assumed evaluation period: 1875 ms / 16 (Soula et al.)
RUNS = 2000
REPORT_SEED, TUNE_SEED = 0, 1

# Aggregators compared throughout (single twin, identical evidence)
BASE = {
    "and":    dict(agg="and"),
    "and_db": dict(agg="and_db"),
    "kof4_p": dict(agg="kof4_p"),
    "beta":   dict(agg="beta"),
    "weighted": dict(agg="weighted"),
    "ema":    dict(agg="ema"),
    "ema_p":  dict(agg="ema_p"),
}
# Full system: Ext. I in each of 5 replicas, 3-of-5 quorum; the replicas
# receive identical evidence (eta = 0) unless stated otherwise.
FULL = dict(agg="ema_p", n_rep=5)


def save(name, obj):
    obj = dict(experiment=name, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               **obj)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    print(f"[saved] {name}", flush=True)


def metrics(r, steady_from=3):
    cfg = r.cfg
    R = cfg.runs
    out = {}
    for j in range(cfg.n_attack):
        k = int(r.detect_any_runs(j).sum())
        mean_ttd, med_ttd, nd = r.ttd_rounds(j)
        out[f"det{j+1}"] = k / R
        out[f"det{j+1}_ci"] = list(wilson(k, R))
        out[f"ttd{j+1}_rounds"] = mean_ttd
        out[f"tpr{j+1}_steady"] = float(r.attack[:, steady_from - 1:, j].mean())
    per_run = r.fpr_round_per_run()
    out["fpr"] = float(per_run.mean())
    out["fpr_ci"] = list(boot_ci(per_run))
    iso = r.legit.any(axis=1)
    k = int(iso.sum())
    out["false_iso"] = k / iso.size
    out["false_iso_ci"] = list(wilson(k, iso.size))
    out["false_iso_count"] = k
    out["node_runs"] = int(iso.size)
    # run-level zero bound (rule of three over honest node-runs); since the
    # per-round FPR cannot exceed the false-isolation rate, it bounds both
    out["zero_upper"] = 3 / iso.size
    out["tau_min_legit_mean"] = float(np.nanmean(r.tau_min_legit))
    if r.cfg.n_rep > 1 or r.cfg.p_miss > 0:
        tc = ((~r.replica_flag) & r.present).sum(-1)
        fc = (r.replica_flag & r.present).sum(-1)
        n, kk = r.cfg.n_rep, r.cfg.kk()
        out["no_quorum_rounds"] = float(((tc < kk) & (fc < n - kk + 1)).mean())
    return out


def cell(seed=REPORT_SEED, **kw):
    cfg = Config(**{"runs": RUNS, "seed": seed, **kw})
    return dict(config=asdict(cfg), **metrics(simulate(cfg)))


def fpr_and(q):
    return 1 - (1 - q) ** 4


def fpr_weighted_mc(q, theta=0.70, n=2_000_000, seed=7, pdr_ok=(92, 100),
                    pdr_bad=(60, 89)):
    rng = np.random.default_rng(seed)
    f = rng.random((n, 4)) < q
    pdr = np.where(f[:, 1], rng.uniform(*pdr_bad, n), rng.uniform(*pdr_ok, n))
    parts = (~f).astype(float)
    parts[:, 1] = pdr / 100
    return float(((parts * WEIGHTS).sum(1) < theta).mean())


def vote_pred(n, c, k, eta_v):
    """Theorem 1: Pr[an attacker is flagged in a round] when c <= k-1
    replicas shield it and the n-c honest replicas err independently with
    probability eta_v."""
    h = n - c
    need = n - k + 1
    return sum(comb(h, j) * (1 - eta_v) ** j * eta_v ** (h - j)
               for j in range(need, h + 1))


# Calibration to Soula et al.: "N3 is incorrectly identified as compromised
# 2%" (one of six honest nodes).  Reading A: a per-round, per-node rate of
# 2%/6 -> q = 0.001 (AND FPR 0.40 %).  Reading B: a per-run (per-case)
# false-isolation rate of 2%/6 over 16 rounds -> 1-(1-q)^64 = 0.0033.
Q_READ_A = 0.001
Q_READ_B = float(1 - (1 - 0.02 / 6) ** (1 / 64))


# ---------------------------------------------------------------------------
# E0  headline: congestion tolerance against simple baselines
# ---------------------------------------------------------------------------
def e0_headline():
    rows = []
    variants = [("and", BASE["and"]), ("and_db", BASE["and_db"]),
                ("kof4_p", BASE["kof4_p"]), ("beta", BASE["beta"]),
                ("weighted", BASE["weighted"]), ("ema", BASE["ema"]),
                ("ema_p", BASE["ema_p"]),
                ("ema_p_corrupt", dict(agg="ema_p", c_corrupt=1)),
                ("full_c2", dict(FULL, c_corrupt=2, frame_legit=True))]
    for q in [Q_READ_B, Q_READ_A, 0.005, 0.02, 0.04]:
        for name, kw in variants:
            m = cell(q=q, **kw)
            w = cell(q=q, profile="W", **kw)
            rows.append(dict(q=q, variant=name, **m, detW1=w["det1"],
                             detW1_ci=w["det1_ci"], ttdW1=w["ttd1_rounds"]))
        print("  e0 q", q, flush=True)
    save("e0_headline", dict(rows=rows, q_read_a=Q_READ_A, q_read_b=Q_READ_B))


def e1_qsweep():
    qs = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20]
    rows = []
    for q in qs:
        for a in ["and", "and_db", "kof4_p", "weighted", "ema", "ema_p"]:
            rows.append(dict(q=q, agg=a, **cell(q=q, **BASE[a])))
    pred = [dict(q=q, and_=fpr_and(q), weighted=fpr_weighted_mc(q, n=400_000))
            for q in qs]
    # log-log slopes over the small-q range (6 points, 0.0005 .. 0.02 for the
    # AND / weighted; the EMA uses q >= 0.01 where it is non-zero)
    save("e1_qsweep", dict(rows=rows, prediction=pred))


# ---------------------------------------------------------------------------
# E1b long horizons, mean time to first false isolation, re-admission
# ---------------------------------------------------------------------------
def _long(cfg_kw, runs, rounds, chunk=100, seed=REPORT_SEED):
    """Run in chunks; returns per-horizon false isolation, onset counts and
    quarantine metrics."""
    horizons = [16, 64, 256, 1024, rounds]
    fi = np.zeros(len(horizons))
    onsets, node_rounds = 0, 0
    unavail, entries = 0.0, 0
    lies_deliv, lies_all, att_quar = 0, 0, 0.0
    det_any = 0
    first_fi = []
    nc = runs // chunk
    for c in range(nc):
        cfg = Config(**{"runs": chunk, "rounds": rounds, "seed": 100 * (seed + 1) + c,   # 100.. for seed 0; never the tuning seed
                        **cfg_kw})
        r = simulate(cfg)
        L = r.legit
        for h_i, hz in enumerate(horizons):
            fi[h_i] += L[:, :hz].any(1).sum()
        on = L[:, 1:] & ~L[:, :-1]
        onsets += int(on.sum() + L[:, 0].sum())
        node_rounds += L.size
        ff = np.where(L.any(1), L.argmax(1) + 1, rounds + 1)
        first_fi.append(ff.ravel())
        qq, ent = quarantine(r)
        nl = cfg.n_legit
        unavail += qq[:, :, :nl].mean() / nc
        entries += int(ent[:, :nl].sum())
        if r.lies is not None and cfg.profile != "none":
            la = r.lies[:, :, nl:]
            prev_q = np.concatenate([np.zeros_like(qq[:, :1, nl:]),
                                     qq[:, :-1, nl:]], axis=1)
            lies_deliv += int((la & ~prev_q).sum())
            lies_all += int(la.sum())
            att_quar += qq[:, :, nl:].mean() / nc
            det_any += int(r.attack.any(1).sum())
    n_nodes = runs * Config(**cfg_kw).n_legit
    hours = node_rounds * T_ROUND_MS / 3.6e6
    rate = onsets / node_rounds
    lo, hi = (wilson(onsets, node_rounds) if onsets else (0.0, 3 / node_rounds))
    return dict(horizons=horizons, false_iso=(fi / n_nodes).tolist(),
                onset_rate=rate, onset_rate_ci=[lo, hi],
                onsets=onsets, node_rounds=node_rounds,
                mttfi_rounds=(1 / rate if rate > 0 else None),
                mttfi_rounds_lower=(1 / hi if hi > 0 else None),
                unavailability=unavail,
                false_quarantines_per_node_hour=entries / hours,
                lies_delivered_per_round=(lies_deliv / (runs * rounds * Config(**cfg_kw).n_attack)),
                lies_per_round=(lies_all / (runs * rounds * Config(**cfg_kw).n_attack)),
                attacker_quarantined=att_quar,
                attack_detected_runs=det_any / (runs * Config(**cfg_kw).n_attack))


def e1_long():
    rows = []
    for q in [Q_READ_A, 0.005, 0.04]:
        for a in ["and", "and_db", "kof4_p", "ema", "ema_p"]:
            rows.append(dict(q=q, agg=a, **_long(dict(q=q, profile="none", **BASE[a]),
                                                 runs=600, rounds=4096)))
        print("  e1_long q", q, flush=True)
    save("e1_long", dict(rows=rows, t_round_ms=T_ROUND_MS))


def e1_readmit():
    """Quarantine with re-admission over 2,048 rounds (about 4 minutes at the
    assumed period): honest unavailability and attacker throughput."""
    rows = []
    attackers = [("naive", dict(profile="TW")),
                 ("bernoulli_0.25", dict(profile="TW", p_lie=0.25)),
                 ("aware_TW", dict(profile="TW", attacker="aware")),
                 ("aware_W", dict(profile="W", attacker="aware"))]
    aggs = [("and", BASE["and"]), ("and_db", BASE["and_db"]),
            ("ema", dict(agg="ema", persist_m=0)),
            ("ema_p", BASE["ema_p"]),
            ("ema_p_long", dict(agg="ema_p", persist2_m=8, persist2_w=32))]
    for q in [Q_READ_A, 0.04]:
        for an, akw in attackers:
            for gn, gkw in aggs:
                rows.append(dict(q=q, attacker=an, agg=gn,
                                 **_long(dict(q=q, **akw, **gkw), runs=400,
                                         rounds=2048)))
        print("  e1_readmit q", q, flush=True)
    save("e1_readmit", dict(rows=rows, readmit_theta=0.90, readmit_r=8))


# ---------------------------------------------------------------------------
# E2  stealthy, minimal and rule-aware attackers
# ---------------------------------------------------------------------------
def e2_stealth():
    rows = []
    for q in [Q_READ_A, 0.04]:
        for profile in ["TW", "W"]:
            for rounds in [16, 64]:
                for p in [1.0, 0.5, 0.25, 0.10]:
                    for a in ["and", "and_db", "kof4_p", "ema", "ema_p"]:
                        rows.append(dict(q=q, profile=profile, rounds=rounds,
                                         p=p, agg=a,
                                         **cell(p_lie=p, profile=profile,
                                                rounds=rounds, q=q, **BASE[a])))
    save("e2_stealth", dict(rows=rows))


def e2_aware():
    """Rule-aware on-off liar: it knows (alpha, theta, m, w[, m2, w2]) and
    lies only when no trigger would fire assuming no congestion.  Analytic
    lie budget = min over windows of (m-1)/w (persistence) or the EMA
    steady state; measured detection and achieved lie rate."""
    rows = []
    defences = [("ema", dict(agg="ema", persist_m=0)),
                ("ema_p", dict(agg="ema_p")),
                ("ema_p_long", dict(agg="ema_p", persist2_m=8, persist2_w=32)),
                ("and", dict(agg="and")),
                ("and_db", dict(agg="and_db")),
                ("kof4_p", dict(agg="kof4_p"))]
    for profile in ["TW", "W"]:
        for q in [0.0, Q_READ_A, 0.04]:
            for rounds in [16, 256]:
                for dn, dkw in defences:
                    cfg = Config(runs=RUNS if rounds == 16 else 500, seed=REPORT_SEED,
                                 q=q, rounds=rounds, profile=profile,
                                 attacker="aware", **dkw)
                    r = simulate(cfg)
                    rows.append(dict(profile=profile, q=q, rounds=rounds,
                                     defence=dn, lie_rate=float(r.lies[:, :, 6:].mean()),
                                     det1=float(r.detect_any(0)),
                                     det1_ci=list(wilson(int(r.detect_any_runs(0).sum()), cfg.runs)),
                                     fpr=float(r.fpr_round()),
                                     false_iso=float(r.false_isolation())))
        print("  e2_aware", profile, flush=True)
    # periodic schedules at q = 0 (deterministic budget check)
    per = []
    for profile in ["TW", "W"]:
        for period in [2, 3, 4, 8]:
            for dn, dkw in defences[:3]:
                r = simulate(Config(runs=500, q=0.0, rounds=64, profile=profile,
                                    attacker="periodic", period=period, **dkw))
                per.append(dict(profile=profile, period=period, defence=dn,
                                det1=float(r.detect_any(0))))
    budget = dict(ema_TW=None, ema_p=3 / 8, ema_p_long=min(3 / 8, 7 / 32))
    save("e2_aware", dict(rows=rows, periodic=per, analytic_budget=budget))


def e2_probe():
    rows = []
    for a in ["ema", "ema_p"]:
        for sig in [0.0, 0.05, 0.10]:
            for p in [1.0, 0.5, 0.25]:
                rows.append(dict(agg=a, sigma_theta=sig, p=p,
                                 **cell(agg=a, profile="W", p_lie=p,
                                        sigma_theta=sig)))
    for (m, w) in [(3, 4), (3, 8), (4, 8), (5, 8), (6, 16)]:
        for p in [1.0, 0.5, 0.25]:
            rows.append(dict(agg="ema_p", persist=[m, w], p=p,
                             **cell(agg="ema_p", profile="W", p_lie=p,
                                    persist_m=m, persist_w=w)))
    save("e2_probe", dict(rows=rows))


# ---------------------------------------------------------------------------
# E3  corrupted twins: shielding, framing, withholding, missing votes
# ---------------------------------------------------------------------------
def _vote_cell(seed=REPORT_SEED, **kw):
    base = {"runs": RUNS, "agg": "ema_p", "q": 0.04}
    cfg = Config(**{**base, "seed": seed, **kw})
    r = simulate(cfg)
    m = metrics(r)
    c = cfg.c_corrupt
    rf = r.replica_flag[:, 2:, cfg.n_legit:, c:] if c < cfg.n_rep else np.zeros(0)
    eta_v = float(1 - rf.mean()) if rf.size else float("nan")
    return cfg, r, dict(config=asdict(cfg), eta_v=eta_v, **m)


def e3_collusion():
    rows = []

    def add(setting, **kw):
        _, _, d = _vote_cell(**kw)
        rows.append(dict(setting=setting, **d))

    add("single twin, honest", n_rep=1)
    add("single twin, corrupted", n_rep=1, c_corrupt=1)
    for n in [3, 5, 7]:
        add(f"{n} twins, 1 corrupted", n_rep=n, c_corrupt=1)
    for c in [0, 1, 2]:
        add(f"5 twins, c={c}", n_rep=5, c_corrupt=c)
        add(f"5 twins, c={c}, framing", n_rep=5, c_corrupt=c, frame_legit=True)
    add("5 twins, c=3, framing", n_rep=5, c_corrupt=3, frame_legit=True)
    for c in [2, 3]:
        add(f"7 twins, c={c}, framing", n_rep=7, c_corrupt=c, frame_legit=True)
    # missing and withheld votes
    miss = []
    for rule in ["naive", "quorum"]:
        for pm in [0.0, 0.05, 0.20]:
            for (c, wh) in [(0, False), (2, True)]:
                _, _, d = _vote_cell(n_rep=5, c_corrupt=c, withhold=wh,
                                     p_miss=pm, miss_rule=rule)
                miss.append(dict(rule=rule, p_miss=pm, c=c, withhold=wh, **d))
    save("e3_collusion", dict(rows=rows, missing=miss))


def e3_independent():
    """Deployment with independent evidence paths: each replica misreads
    each factor independently with probability eta (shared share rho).
    Theorem 1 is validated with eta_v estimated on the tuning seed."""
    rows = []
    for eta in [0.01, 0.02, 0.05, 0.10, 0.20]:
        for (n, c) in [(1, 0), (5, 0), (5, 1), (5, 2)]:
            _, _, tune = _vote_cell(seed=TUNE_SEED, n_rep=n, c_corrupt=c, eta=eta)
            _, r, d = _vote_cell(n_rep=n, c_corrupt=c, eta=eta)
            pred = vote_pred(n, c, Config(n_rep=n).kk(), tune["eta_v"]) if n > 1 else 1 - tune["eta_v"]
            meas = d["tpr1_steady"]
            rows.append(dict(eta=eta, n=n, c=c, eta_v_tune=tune["eta_v"],
                             thm1_pred=pred, miss_meas=1 - meas, miss_pred=1 - pred,
                             **d))
    frame = []
    for eta in [0.05, 0.10]:
        for c in [1, 2]:
            _, _, d = _vote_cell(n_rep=5, c_corrupt=c, eta=eta, frame_legit=True)
            _, _, eps = _vote_cell(n_rep=1, eta=eta)
            frame.append(dict(eta=eta, c=c, approx=(5 - c) * eps["fpr"], **d))
    rho = []
    for rh in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _, _, d = _vote_cell(n_rep=5, c_corrupt=1, eta=0.05, rho=rh)
        rho.append(dict(rho=rh, **d))
    save("e3_independent", dict(rows=rows, rho=rho, framing=frame))


# ---------------------------------------------------------------------------
# E4  audit log: storage, tamper evidence, forgeries, endorsement
# ---------------------------------------------------------------------------
def _build_chain(K, res, rounds, p_bad_engine=None):
    cfg = res.cfg
    N = cfg.n_legit + cfg.n_attack
    chain = ac.AuditChain(K=K, n_rep=cfg.n_rep, k=cfg.kk(), n_nodes=N)
    honest = cfg.c_corrupt          # first honest replica
    for t in range(rounds):
        rv = [[ac.FLAG if res.replica_flag[0, t, i, r] else ac.TRUST
               for i in range(N)] for r in range(cfg.n_rep)]
        fv = res.fail_v[0, t, :, honest, :]
        factors = [[0.0 if fv[i, f] else 1.0 for f in range(4)] for i in range(N)]
        for i in range(N):
            factors[i][1] = float(res.pdr[0, t, i, honest]) / 100
        taus = [float(res.tau[0, t, i, honest]) for i in range(N)]
        present = [bool(res.present[0, t, 0, r]) for r in range(cfg.n_rep)]
        chain.add_round(t + 1, rv, factors, taus, present)
    chain.flush(rounds)
    return chain


def _clone(chain):
    c2 = copy.copy(chain)
    c2.blocks = [copy.copy(b) for b in chain.blocks]
    for b in c2.blocks:
        b.sigs = list(b.sigs)
    c2.windows = [ac.Window(list(w.events), list(w.votes), list(w.state))
                  for w in chain.windows]
    return c2


def _edit(c2, kind, b, rng, stolen):
    w = c2.windows[b]
    if kind == "verdict":
        e = int(rng.integers(len(w.events)))
        node, rnd, f, tau, v = ac.decode_event(w.events[e])
        w.events[e] = ac.encode_event(node, rnd, f, tau, 1 - v)
        for si, srec in enumerate(w.state):
            sn, sr, st_, sv = __import__("struct").unpack(ac.STATE_FMT, srec)
            if sn == node and sr == rnd:
                w.state[si] = ac.encode_state(sn, sr, st_, 1 - v)
        if stolen:   # forge the stolen replicas' votes to support the edit
            out = []
            for rec, sig in w.votes:
                rid, vr, ep, vv = ac.decode_vote(rec)
                if rid in stolen and vr == rnd:
                    vv = list(vv)
                    vv[node] = 1 - v
                    rec = ac.vote_record(rid, vr, ep, vv)
                out.append((rec, sig))
            w.votes = out
            ac.resign_votes(c2, b, stolen)
    elif kind == "tau":
        e = int(rng.integers(len(w.events)))
        node, rnd, f, tau, v = ac.decode_event(w.events[e])
        w.events[e] = ac.encode_event(node, rnd, f, min(1.0, tau + 0.2), v)
        for si, srec in enumerate(w.state):
            sn, sr, st_, sv = __import__("struct").unpack(ac.STATE_FMT, srec)
            if sn == node and sr == rnd:
                w.state[si] = ac.encode_state(sn, sr, min(1.0, tau + 0.2), sv)
    elif kind == "state":      # change a node's latest tau consistently in
        s = int(rng.integers(len(w.state)))   # its state record and last event
        node, rnd, tau, v = __import__("struct").unpack(ac.STATE_FMT, w.state[s])
        w.state[s] = ac.encode_state(node, rnd, min(1.0, tau + 0.2), v)
        for e in range(len(w.events) - 1, -1, -1):
            n2, r2, f, t2, v2 = ac.decode_event(w.events[e])
            if n2 == node:
                w.events[e] = ac.encode_event(n2, r2, f, min(1.0, tau + 0.2), v2)
                break
    elif kind == "vote":
        cand = [j for j, (rec, _) in enumerate(w.votes)
                if not stolen or rec[0] in stolen]
        j = int(cand[int(rng.integers(len(cand)))])
        rec, sig = w.votes[j]
        raw = bytearray(rec)
        raw[-1 - int(rng.integers(8))] ^= 1
        w.votes[j] = (bytes(raw), sig)
        if stolen:
            ac.resign_votes(c2, b, stolen)
    elif kind == "interior":   # replace events by interior-node blobs
        hs = [ac.h(e) for e in w.events]
        w.events = [hs[i] + hs[i + 1] for i in range(0, len(hs) - 1, 2)]
    elif kind == "duplicate":
        w.events = w.events + [w.events[-1]]
    elif kind == "truncate":
        w.events = w.events[:-1]
    elif kind == "replay":     # replay a vote record from an earlier round
        recs = w.votes
        j = int(rng.integers(1, len(recs)))
        w.votes = recs[:j] + [recs[0]] + recs[j + 1:]
    else:
        raise ValueError(kind)


def e4_audit():
    N_SENS, ROUNDS = 100, 30
    cfg = Config(runs=1, seed=REPORT_SEED, rounds=ROUNDS, n_legit=98, n_attack=2,
                 n_fog=5, q=0.04, agg="ema_p", n_rep=5, c_corrupt=1, p_miss=0.05,
                 keep=True)
    res = simulate(cfg)
    rng = np.random.default_rng(3)
    rows, tamper = [], []
    for K in [1, 5, 10, 20]:
        chain = _build_chain(K, res, ROUNDS)
        st = chain.storage()
        hdr = chain.blocks[0]
        rows.append(dict(K=K, **st, header_bytes=ac.HEADER_BYTES,
                         block_bytes_first=hdr.onchain_bytes(),
                         t_conf_rounds=K, verify_clean=ac.verify(chain),
                         n_blocks=len(chain.blocks)))
        if K not in (1, 10):
            continue
        n_trials = 30
        kinds = ["verdict", "tau", "state", "vote", "interior", "duplicate",
                 "truncate", "replay"]
        advs = [("edit_only", None, ()), ("recompute", True, ()),
                ("k-1_keys", True, (1, 2)), ("k_keys", True, (1, 2, 3))]
        for kind in kinds:
            for an, rec, stolen in advs:
                if kind in ("interior", "duplicate", "truncate", "replay") and an != "edit_only" and an != "k_keys":
                    continue
                det = {"none": 0, "latest": 0, "older": 0}
                n_old = 0
                for _ in range(n_trials):
                    b = int(rng.integers(len(chain.blocks)))
                    c2 = _clone(chain)
                    _edit(c2, kind, b, rng, stolen)
                    if rec:
                        ac.recompute_forward(c2, b, stolen=stolen)
                    for ck in ["none", "latest", "older"]:
                        if ck == "none":
                            cp = None
                        elif ck == "latest":
                            cp = (len(chain.blocks) - 1, chain.blocks[-1].hash)
                        else:
                            if b == 0:
                                continue
                            cp = (b - 1, chain.blocks[b - 1].hash)
                        det[ck] += not ac.verify(c2, cp)
                    n_old += b > 0
                tamper.append(dict(K=K, kind=kind, adversary=an, trials=n_trials,
                                   det_none=det["none"] / n_trials,
                                   det_latest=det["latest"] / n_trials,
                                   det_older=(det["older"] / n_old) if n_old else None))
            print("  e4", K, kind, flush=True)
    # legacy tree: forgeries that keep the root unchanged
    chain = _build_chain(10, res, ROUNDS)
    ev = chain.windows[0].events
    hs = [ac.h(e) for e in ev]
    blobs = [hs[i] + hs[i + 1] for i in range(0, len(hs) - 1, 2)]
    legacy = dict(
        interior_same_root_legacy=ac.merkle_root_legacy(blobs) == ac.merkle_root_legacy(ev) if len(ev) % 2 == 0 else None,
        duplicate_same_root_legacy=ac.merkle_root_legacy(ev[:3]) == ac.merkle_root_legacy(ev[:3] + [ev[2]]),
        interior_same_root_rfc6962=ac.merkle_root(blobs) == ac.merkle_root(ev),
        duplicate_same_root_rfc6962=ac.merkle_root(ev[:3]) == ac.merkle_root(ev[:3] + [ev[2]]))
    # a trust engine that writes a verdict the votes do not support: the
    # replicas refuse to endorse, so the block cannot obtain k signatures
    committed = 0
    for trial in range(30):
        c = ac.AuditChain(K=100, n_rep=5, k=3, n_nodes=8)
        for t in range(1, 10):
            c.add_round(t, [[ac.TRUST] * 8] * 5, [[1, 0.95, 1, 1]] * 8, [1.0] * 8)
        i = trial % 8
        c._ev[-8 + i] = ac.encode_event(i, 9, [1, 0.95, 1, 1], 1.0, ac.FLAG)
        committed += bool(c.commit(9))
    save("e4_audit", dict(rows=rows, tamper=tamper, legacy=legacy,
                          engine_equivocation_committed=committed,
                          engine_equivocation_trials=30,
                          sizes=dict(event=ac.EVENT_BYTES, state=ac.STATE_BYTES,
                                     header=ac.HEADER_BYTES,
                                     vote_header=8 + 3)))


def e4_latency():
    import hashlib
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes

    def bench(fn, n):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e3

    key = ec.generate_private_key(ec.SECP256K1())
    pub = key.public_key()
    out = {}
    for N in [8, 100]:
        rec = ac.vote_record(0, 1, 0, [1] * N)
        sig = key.sign(rec, ec.ECDSA(hashes.SHA256()))
        out[f"sign_vector_{N}_ms"] = bench(lambda: key.sign(rec, ec.ECDSA(hashes.SHA256())), 1000)
        out[f"verify_vector_{N}_ms"] = bench(lambda: pub.verify(sig, rec, ec.ECDSA(hashes.SHA256())), 1000)
    msg = b"x" * 16
    sig = key.sign(msg, ec.ECDSA(hashes.SHA256()))
    out["sign_single_ms"] = bench(lambda: key.sign(msg, ec.ECDSA(hashes.SHA256())), 2000)
    out["verify_single_ms"] = bench(lambda: pub.verify(sig, msg, ec.ECDSA(hashes.SHA256())), 2000)
    tau = [1.0]

    def ema():
        s = 0.3 * 1 + 0.3 * 0.95 + 0.2 + 0.2
        tau[0] = 0.5 * s + 0.5 * tau[0]
        return tau[0] < 0.7
    out["ema_update_ms"] = bench(ema, 200000)
    out["sha256_29B_ms"] = bench(lambda: hashlib.sha256(bytes(29)).digest(), 200000)
    leaves = [bytes(29)] * 1000
    out["merkle_1000_ms"] = bench(lambda: ac.merkle_root(leaves), 50)
    # per-round load at the trust engine, n = 5 replicas
    for N in [8, 100]:
        out[f"engine_per_node_votes_{N}_ms"] = 5 * N * out["verify_single_ms"]
        out[f"engine_batched_{N}_ms"] = 5 * out[f"verify_vector_{N}_ms"]
        out[f"replica_per_node_sign_{N}_ms"] = N * out["sign_single_ms"]
        out[f"replica_batched_sign_{N}_ms"] = out[f"sign_vector_{N}_ms"]
    try:
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
               if l.startswith("model name")]
    except OSError:
        cpu = [platform.processor()]
    import cryptography
    out["platform"] = dict(cpu=cpu[0] if cpu else "", vcpus=len(cpu),
                           python=platform.python_version(),
                           cryptography=cryptography.__version__,
                           numpy=np.__version__, single_thread=True)
    save("e4_latency", out)


# ---------------------------------------------------------------------------
# E5  held-out tuning: parameters selected on seed 1, reported on seed 0
# ---------------------------------------------------------------------------
def e5_tuning():
    grid = []
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        for theta in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            row = dict(alpha=alpha, theta=theta)
            for seed, tag in [(TUNE_SEED, "tune"), (REPORT_SEED, "report")]:
                for q in [0.02, 0.04]:
                    m = cell(seed=seed, agg="ema_p", alpha=alpha, theta=theta, q=q)
                    row[f"{tag}_q{q}"] = dict(fpr=m["fpr"], false_iso=m["false_iso"],
                                              ttd=m["ttd1_rounds"], det=m["det1"])
                mw = cell(seed=seed, agg="ema", alpha=alpha, theta=theta, q=0.04,
                          profile="W")
                row[f"{tag}_emaW"] = mw["det1"]
            grid.append(row)
    # selection rule (on the tuning seed): mean TTD of a naive liar <= 2
    # rounds at q = 0.04, then the lowest per-round FPR at q = 0.04
    ok = [g for g in grid if g["tune_q0.04"]["ttd"] <= 2.0 + 1e-9]
    sel = min(ok, key=lambda g: g["tune_q0.04"]["fpr"])
    # persistence window: lowest FPR at q = 0.04 among windows that detect a
    # minimal liar with p = 0.5 in >= 90 % of runs (tuning seed)
    win = []
    for (m, w) in [(3, 4), (3, 8), (4, 8), (5, 8), (6, 16)]:
        d = cell(seed=TUNE_SEED, agg="ema_p", persist_m=m, persist_w=w,
                 profile="W", p_lie=0.5)
        f = cell(seed=TUNE_SEED, agg="ema_p", persist_m=m, persist_w=w)
        win.append(dict(m=m, w=w, det_p05=d["det1"], fpr=f["fpr"]))
    # rule: flag a persistent minimal liar within four rounds (m <= 4),
    # detect a minimal liar with p = 0.5 in >= 90 % of runs, lowest FPR
    okw = [x for x in win if x["det_p05"] >= 0.90 and x["m"] <= 4]
    selw = min(okw, key=lambda x: x["fpr"])
    # without the m <= 4 constraint (reported as the alternative)
    alt = min([x for x in win if x["det_p05"] >= 0.90], key=lambda x: x["fpr"])
    # beta baseline: same rule as for (alpha, theta)
    beta = []
    for lam in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for th in np.round(np.arange(0.30, 0.96, 0.01), 2):
            r0 = simulate(Config(runs=20, q=0.0, agg="beta", beta_lambda=lam,
                                 beta_theta=float(th), seed=TUNE_SEED))
            if r0.ttd_rounds(0)[0] > 2 or r0.detect_any(0) < 1:
                continue
            m = cell(seed=TUNE_SEED, agg="beta", beta_lambda=lam, beta_theta=float(th))
            beta.append(dict(lam=lam, theta=float(th), fpr=m["fpr"]))
    bmin = min(b["fpr"] for b in beta)
    ties = [b for b in beta if b["fpr"] <= bmin + 2e-5]
    save("e5_tuning", dict(beta_grid=beta, beta_min_fpr=bmin, beta_near_ties=ties,
                           grid=grid, selected=dict(alpha=sel["alpha"], theta=sel["theta"]),
                           windows=win, selected_window=[selw["m"], selw["w"]],
                           alternative_window=[alt["m"], alt["w"]]))


# ---------------------------------------------------------------------------
# E6  robustness: bursts, common-mode events, cross-factor events,
#     uneven congestion, scale
# ---------------------------------------------------------------------------
ROB_AGGS = ["and", "and_db", "kof4_p", "ema", "ema_p"]


def e6_bursty():
    """Mean congestion fixed at 0.04 (q_low 0.02, q_high 0.20, 1/9 of the
    time in the high state); only the mean burst length changes."""
    rows = []
    pi_h = 1 / 9
    for scope in ["node", "network"]:
        for dwell in [1, 2, 4, 8, 16, 32]:
            p_hl = 1 / dwell
            p_lh = p_hl * pi_h / (1 - pi_h)
            for a in ROB_AGGS:
                rows.append(dict(scope=scope, dwell=dwell, agg=a,
                                 **cell(markov=(0.02, 0.20, p_lh, p_hl),
                                        markov_scope=scope, **BASE[a])))
        print("  e6_bursty", scope, flush=True)
    save("e6_bursty", dict(rows=rows, mean_q=0.02 * (1 - pi_h) + 0.20 * pi_h))


def e6_corr():
    rows = []
    for qc in [0.0, 0.001, 0.003, 0.005, 0.01, 0.02]:
        for a in ROB_AGGS:
            rows.append(dict(q_corr=qc, agg=a, **cell(q_corr=qc, **BASE[a])))
        rows.append(dict(q_corr=qc, agg="ema_p_supp",
                         **cell(q_corr=qc, agg="ema_p", suppress=0.5)))
    save("e6_corr", dict(rows=rows))


def e6_xfactor():
    """Node-level events that fail P, T and W together (no i.i.d. noise)."""
    rows = []
    for qn in [0.0025, 0.005, 0.01, 0.02, 0.04]:
        for a in ROB_AGGS:
            rows.append(dict(q_node=qn, agg=a,
                             **cell(q=0.0, q_node=qn, runs=4000, **BASE[a])))
    save("e6_xfactor", dict(rows=rows))


def e6_uneven():
    rows = []
    vecs = {"uniform": (0.04, 0.04, 0.04, 0.04),
            "P-heavy": (0.01, 0.13, 0.01, 0.01),
            "E-heavy": (0.13, 0.01, 0.01, 0.01),
            "W-heavy": (0.01, 0.01, 0.01, 0.13)}
    for name, v in vecs.items():
        for a in ROB_AGGS:
            m = cell(q_vec=v, **BASE[a])
            w = cell(q_vec=v, profile="W", **BASE[a])
            rows.append(dict(profile=name, q_vec=v, agg=a, **m, detW1=w["det1"]))
    save("e6_uneven", dict(rows=rows))


def e6_large():
    rows = []
    for qc in [0.0, 0.005]:
        for name, kw in [("and", BASE["and"]), ("and_db", BASE["and_db"]),
                         ("ema_p", BASE["ema_p"]),
                         ("ema_p_supp", dict(agg="ema_p", suppress=0.5)),
                         ("full", FULL)]:
            rows.append(dict(q_corr=qc, agg=name,
                             **cell(n_legit=98, n_attack=2, n_fog=5, q_corr=qc,
                                    runs=500, **kw)))
    save("e6_large", dict(rows=rows))


EXPS = {f.__name__: f for f in [e0_headline, e1_qsweep, e1_long, e1_readmit,
                                e2_stealth, e2_aware, e2_probe, e3_collusion,
                                e3_independent, e4_audit, e4_latency,
                                e5_tuning, e6_bursty, e6_corr, e6_xfactor,
                                e6_uneven, e6_large]}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(EXPS)
    for name in todo:
        t0 = time.time()
        EXPS[name]()
        print(f"  {name}: {time.time()-t0:.1f}s", flush=True)
