"""Trace-driven evidence failures for honest behaviour.

Two public traces are supported:
* "intel": Intel Berkeley Research Lab, 54 motes; for every mote and 31-s
  slot, whether its reading reached the base station (multi-hop, lossy);
* "flocklab": FlockLab TelosB link-quality tests, August 2019; for each
  sensor, 100 back-to-back packets to its fog gateway every ~2 h (single hop).
Both are mapped onto the factor evidence of the IoDT model as follows (see
the paper, Section V, "Trace-driven evaluation"):

* one evaluation round of a node uses 4*r consecutive transmission
  opportunities of one node (Intel: 31-s slots; FlockLab: packets of one
  test; rounds never straddle two FlockLab tests);
* the evidence message of factor f (E, P, T, W, in that order) is sent in r
  attempts, one per slot, and is lost only if all r attempts are lost
  (r = 1 is no retransmission);
* a factor fails in a round when its evidence message is lost.

Runs draw random windows of `rounds` rounds and random distinct motes that are
alive over the whole window, and windows that overlap a logging gap are
rejected, so every failure comes from the radio and multi-hop path of the
real deployment.  Losses are therefore bursty in time, correlated across the
four factors of a node, uneven across motes and correlated across motes, as
they were in the deployment.
"""
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent / "datasets"
_cache = {}


def _load():
    if "z" not in _cache:
        z = np.load(_ROOT / "intel_lab" / "intel_slots.npz")
        _cache["z"] = (z["delivered"], z["alive"], z["gap"])
    return _cache["z"]


def _round_fail_flocklab(r: int):
    z = np.load(_ROOT / "flocklab" / "flocklab_sky_2019-08.npz")
    st, valid = z["streams"], z["valid"]                    # (S, J, 100)
    B = 4 * r
    per = 100 // B
    d = st[:, :, :per * B].reshape(st.shape[0], st.shape[1], per, 4, r)
    fail = ~d.any(-1)                                        # (S, J, per, 4)
    fail = fail.reshape(st.shape[0], st.shape[1] * per, 4)
    usable = np.repeat(valid, per, axis=1)
    return fail, usable


def round_fail(r: int, dataset: str = "intel"):
    """Per-node, per-round factor failures for retry count r.
    Returns fail[m, t, 4], usable[m, t] (alive, no logging gap)."""
    key = ("rf", dataset, r)
    if key not in _cache and dataset == "flocklab":
        _cache[key] = _round_fail_flocklab(r)
    if key not in _cache:
        dlv, alive, gap = _load()
        M, S = dlv.shape
        B = 4 * r
        T = S // B
        d = dlv[:, :T * B].reshape(M, T, 4, r)
        fail = ~d.any(-1)
        usable = alive[:, :T * B].reshape(M, T, B).all(-1) & \
            ~gap[:T * B].reshape(T, B).any(-1)[None]
        _cache[key] = (fail, usable)
    return _cache[key]


def sample(r: int, runs: int, rounds: int, n_nodes: int, seed: int,
           max_tries: int = 2000, dataset: str = "intel"):
    """fail[runs, rounds, n_nodes, 4] from random usable windows.  Windows
    start on a grid of `rounds` rounds, so two runs either share a window
    exactly or do not overlap (needed for the window-level bootstrap)."""
    fail, usable = round_fail(r, dataset)
    M, T = usable.shape
    rng = np.random.default_rng(seed)
    # cumulative usable counts for O(1) window checks
    cu = np.concatenate([np.zeros((M, 1), np.int32), np.cumsum(usable, 1, dtype=np.int32)], 1)
    out = np.zeros((runs, rounds, n_nodes, 4), dtype=bool)
    meta = np.zeros((runs, 1 + n_nodes), dtype=np.int64)
    for i in range(runs):
        for _ in range(max_tries):
            t0 = rounds * int(rng.integers(0, T // rounds))
            ok = (cu[:, t0 + rounds] - cu[:, t0]) == rounds
            cand = np.flatnonzero(ok)
            if len(cand) >= n_nodes:
                motes = rng.choice(cand, n_nodes, replace=False)
                out[i] = fail[motes, t0:t0 + rounds].transpose(1, 0, 2)
                meta[i] = [t0, *motes]
                break
        else:
            raise RuntimeError(f"no usable window of {rounds} rounds (r={r})")
    return out, meta


def stats(r: int, dataset: str = "intel"):
    """Descriptive statistics of the mapped failure process."""
    fail, usable = round_fail(r, dataset)
    f = fail[usable]
    k = f.sum(1)
    q = f.mean()
    nxt = usable[:, :-1] & usable[:, 1:]
    a, b = fail[:, :-1][nxt], fail[:, 1:][nxt]
    per_mote = np.array([fail[m][usable[m]].mean() for m in range(fail.shape[0])
                         if usable[m].sum() > 100])
    return dict(dataset=dataset, r=r, q=float(q),
                p_ge2=float(np.mean(k >= 2)),
                p_ge2_indep=float(1 - (1 - q) ** 4 - 4 * q * (1 - q) ** 3),
                p_all4=float(np.mean(k == 4)),
                p_fail_given_prev=float((a & b).sum() / a.sum()),
                q_mote_p10=float(np.percentile(per_mote, 10)),
                q_mote_p50=float(np.percentile(per_mote, 50)),
                q_mote_p90=float(np.percentile(per_mote, 90)),
                usable_node_rounds=int(usable.sum()))
