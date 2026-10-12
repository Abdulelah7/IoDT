"""Trace-driven evaluation on the Intel Berkeley Research Lab deployment
(experiment e7_trace).  Honest-node evidence failures come from the real
packet-delivery record of 54 motes (sim/traces.py); attackers are real motes
on which lies are injected exactly as in the synthetic experiments.

    python3 prep_intel.py      # once: builds datasets/intel_lab/intel_slots.npz
    python3 run_trace.py       # writes results/e7_trace.json

Seeds: window/mote sampling and the simulator use seed 0 (report seed); no
parameter is tuned on the trace -- every rule keeps the defaults selected on
synthetic data (seed 1).
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim"))
sys.path.insert(0, str(HERE))
import traces                                                           # noqa: E402
from iodt_sim import Config, simulate, quarantine, wilson              # noqa: E402
from run_all import metrics, save, REPORT_SEED                         # noqa: E402

N_ATTACK = 2
RUNS16, RUNS_LONG = 1000, 300
DATASETS = {
    # name: (honest nodes per window, retry counts, (r, rounds) for the long runs)
    "intel": dict(n_legit=30, retries=[4, 8, 16], long=(8, 128),
                  label="Intel Berkeley Research Lab, 54 motes, 28 Feb - 5 Apr 2004"),
    # two of the 24 FlockLab sensors ran only during the first 18 of 262 tests,
    # so windows use 22 nodes (20 honest + 2 attackers) to span the month
    "flocklab": dict(n_legit=20, retries=[1, 2, 4], long=(1, 256),
                     label="FlockLab TelosB link tests, 24 sensors + 2 gateways, Aug 2019"),
}
AGGS = {"and": dict(agg="and"), "and_db": dict(agg="and_db"),
        "kof4_p": dict(agg="kof4_p"), "ema_p": dict(agg="ema_p")}


def mote_quality(r, ds):
    """Long-run per-factor failure rate of each node at retry count r."""
    fail, usable = traces.round_fail(r, ds)
    return np.array([fail[m][usable[m]].mean() if usable[m].sum() > 0 else np.nan
                     for m in range(fail.shape[0])])


def cluster_ci(per_run, groups, B=2000, seed=12345):
    """95 % bootstrap interval resampling trace windows (groups), because
    runs that share a window are not independent."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    sums = np.array([per_run[idx[g]].sum() for g in ug])
    cnts = np.array([len(idx[g]) for g in ug])
    pick = rng.integers(0, len(ug), (B, len(ug)))
    est = sums[pick].sum(1) / cnts[pick].sum(1)
    return [float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


def window_stats(fb, n_legit):
    """Loss statistics of the honest nodes in the windows actually evaluated."""
    f = fb[:, :, :n_legit]
    q = float(f.mean())
    k = f.sum(-1)
    p2 = float((k >= 2).mean())
    p2i = 1 - (1 - q) ** 4 - 4 * q * (1 - q) ** 3
    a, b = f[:, :-1], f[:, 1:]
    return dict(q=q, p_ge2=p2, p_ge2_indep=p2i, ratio_ge2=p2 / p2i,
                p_all4=float((k == 4).mean()),
                p_fail_given_prev=float((a & b).sum() / a.sum()),
                node_q_p50=float(np.median(f.mean((1, 3)))),
                node_q_p90=float(np.percentile(f.mean((1, 3)), 90)))


def paired_ci(a, b, groups, B=2000, seed=12345):
    """Window-level bootstrap of mean(a) - mean(b) (per-run values)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    sa = np.array([a[groups == g].sum() for g in ug])
    sb = np.array([b[groups == g].sum() for g in ug])
    n = np.array([(groups == g).sum() for g in ug])
    pick = rng.integers(0, len(ug), (B, len(ug)))
    est = (sa[pick].sum(1) - sb[pick].sum(1)) / n[pick].sum(1)
    return [float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


def run_cell(fb, meta, r, rounds, ds, n_legit, silence=True, **kw):
    cfg = Config(runs=fb.shape[0], rounds=rounds, seed=REPORT_SEED,
                 n_legit=n_legit, n_attack=N_ATTACK, q=0.0,
                 fail_base=fb, silence_hold=silence, **kw)
    res = simulate(cfg)
    out = metrics(res)
    iso = res.legit.any(1)                                   # (runs, n_legit)
    groups = meta[:, 0] // rounds                            # window block
    out["false_iso_ci_cluster"] = cluster_ci(iso.mean(1), groups)
    out["per_run_iso"] = iso.mean(1)
    out["clusters"] = int(len(np.unique(groups)))
    q_m = mote_quality(r, ds)
    med, p75 = np.nanmedian(q_m), np.nanpercentile(q_m, 75)
    qn = q_m[meta[:, 1:1 + n_legit]]                         # (runs, n_legit)
    typ, lossy = qn <= med, qn > p75
    out["false_iso_typical"] = float(iso[typ].mean())
    out["false_iso_lossy"] = float(iso[lossy].mean())
    out["share_typical"] = float(typ.mean())
    cfgd = asdict(cfg)
    cfgd["fail_base"] = f"{ds} trace, r={r}"
    out["config"] = cfgd
    return out, res


def run_dataset(ds):
    p = DATASETS[ds]
    nl = p["n_legit"]
    out = dict(dataset=p["label"], n_legit=nl, n_attack=N_ATTACK,
               stats=[traces.stats(r, ds) for r in p["retries"]],
               window_stats={}, paired={}, paired_silence_off={},
               rows=[], iid=[], silence_off=[], long=[], windows16={})
    for r in p["retries"]:
        fb, meta = traces.sample(r, RUNS16, 16, nl + N_ATTACK, seed=REPORT_SEED, dataset=ds)
        st = window_stats(fb, nl)
        out["window_stats"][str(r)] = st
        out["windows16"][str(r)] = int(len(np.unique(meta[:, 0] // 16)))
        groups = meta[:, 0] // 16
        per, per_off = {}, {}
        for a, akw in AGGS.items():
            tw, _ = run_cell(fb, meta, r, 16, ds, nl, profile="TW", **akw)
            per[a] = tw.pop("per_run_iso")
            w, _ = run_cell(fb, meta, r, 16, ds, nl, profile="W", **akw)
            out["rows"].append(dict(r=r, agg=a, q=st["q"], **tw,
                                    detW1=w["det1"], ttdW1=w["ttd1_rounds"],
                                    detW2=w["det2"]))
            iid = Config(runs=2000, rounds=16, seed=REPORT_SEED, n_legit=nl,
                         n_attack=N_ATTACK, q=st["q"], **akw)
            m = metrics(simulate(iid))
            out["iid"].append(dict(r=r, agg=a, q=st["q"], fpr=m["fpr"],
                                   false_iso=m["false_iso"], ttd1_rounds=m["ttd1_rounds"]))
            so, _ = run_cell(fb, meta, r, 16, ds, nl, silence=False, profile="TW", **akw)
            per_off[a] = so.pop("per_run_iso")
            w.pop("per_run_iso", None)
            out["silence_off"].append(dict(r=r, agg=a, fpr=so["fpr"],
                                           false_iso=so["false_iso"], det1=so["det1"]))
        out["paired"][str(r)] = {a: paired_ci(per[a], per["ema_p"], groups)
                                 for a in ["and", "and_db", "kof4_p"]}
        out["paired_silence_off"][str(r)] = {a: paired_ci(per_off[a], per_off["ema_p"], groups)
                                             for a in ["and", "and_db", "kof4_p"]}
        print(f"  e7 {ds} r={r}", flush=True)

    # long operation, re-admission and the rule-aware attacker
    r, rounds = p["long"]
    fb, meta = traces.sample(r, RUNS_LONG, rounds, nl + N_ATTACK, seed=REPORT_SEED, dataset=ds)
    for a, akw in {**AGGS, "ema_p_long": dict(agg="ema_p", persist2_m=8,
                                              persist2_w=32)}.items():
        o, res = run_cell(fb, meta, r, rounds, ds, nl, profile="TW", attacker="aware", **akw)
        o.pop("per_run_iso")
        qq, ent = quarantine(res)
        la = res.lies[:, :, nl:]
        prev_q = np.concatenate([np.zeros_like(qq[:, :1, nl:]), qq[:, :-1, nl:]], axis=1)
        out["long"].append(dict(
            r=r, rounds=rounds, agg=a,
            false_iso_long=float(res.legit.any(1).mean()),
            false_iso_16=float(res.legit[:, :16].any(1).mean()),
            unavailability=float(qq[:, :, :nl].mean()),
            lies_per_round=float(la.mean()),
            lies_delivered_per_round=float((la & ~prev_q).mean()),
            detected_runs=float(res.attack.any(1).mean())))
    out["windows_long"] = int(len(np.unique(meta[:, 0] // rounds)))
    return out


def main():
    save("e7_trace", dict(intel=run_dataset("intel"), flocklab=run_dataset("flocklab")))


if __name__ == "__main__":
    main()
