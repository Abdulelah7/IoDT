"""
IoDT trust-management simulator.

Reimplements, from the manuscript's specification, the factor-level model of
Soula et al. (IEEE Access 2024) and the extensions of the paper:

  Ext. I   weighted continuous score + EMA smoothing + per-factor persistence
           check (optionally a second, longer persistence window)
  Ext. II  n digital-twin replicas, quorum vote, c corrupted replicas,
           missing votes (timeouts) and withheld votes
  Ext. III signed Merkle audit log (see audit_chain.py)

Baselines: binary AND (Soula et al.), AND debounced over r
consecutive rounds, per-round k-of-4 factor rule (optionally with the
persistence check), and a beta reputation score with forgetting computed on
the binary AND observation (Josang & Ismail 2002).

Evidence models: i.i.d. per-factor congestion q (or a per-factor vector),
two-state Markov bursts (network-wide or per node), fog-level common-mode
events, and node-level cross-factor events that fail P, T and W together.

Attackers: Bernoulli liars, periodic on-off liars, and a rule-aware on-off
liar that knows (alpha, theta, m, w[, m2, w2]) and lies only when its own
predicted state stays below every trigger.

Everything is vectorised with NumPy over (runs, rounds, nodes, replicas,
factors); all randomness comes from numpy.random.default_rng(seed).

Factor order everywhere: 0 = E (energy), 1 = P (PDR), 2 = T (traceability),
3 = W (twin notification).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

E, P, T, W = 0, 1, 2, 3
WEIGHTS = np.array([0.30, 0.30, 0.20, 0.20])
PDR_THRESHOLD = 90.0            # Soula et al. use a 90 % PDR threshold


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    runs: int = 1000
    seed: int = 0
    rounds: int = 16            # evaluation rounds per run
    n_legit: int = 6            # N1..N6 (8-sensor topology of Soula et al.)
    n_attack: int = 2           # COMP1, COMP2
    n_fog: int = 2              # fog domains (nodes assigned round-robin)

    # --- congestion / noise on the physical evidence ----------------------
    q: float = 0.04             # per-factor, per-round i.i.d. failure prob.
    q_vec: tuple | None = None  # per-factor rates (E, P, T, W); overrides q
    q_corr: float = 0.0         # per-round, per-fog common-mode event prob.
    corr_factors: int = 4       # factors knocked out by a common-mode event
    q_node: float = 0.0         # per node-round event failing P, T, W together
    markov: tuple | None = None # (q_low, q_high, p_lh, p_hl) bursty congestion
    markov_scope: str = "network"   # "network" (one chain) | "node"

    pdr_ok: tuple = (92.0, 100.0)
    pdr_bad: tuple = (60.0, 89.0)

    # --- attacker behaviour -------------------------------------------------
    profile: str = "TW"         # "TW": a lie fails T and W; "W": W only;
                                # "none": attackers behave honestly
    attacker: str = "bernoulli" # "bernoulli" | "periodic" | "aware"
    p_lie: float = 1.0          # Bernoulli lying probability
    period: int = 3             # periodic: one lie every `period` rounds
    aware_margin: float = 0.02  # aware: keeps predicted tau >= theta + margin

    # --- trust aggregation inside each twin ---------------------------------
    agg: str = "ema_p"          # "and" | "and_db" | "kof4" | "kof4_p" | "beta"
                                # | "weighted" | "ema" | "ema_p"
    alpha: float = 0.5
    theta: float = 0.70
    sigma_theta: float = 0.0
    persist_m: int = 4
    persist_w: int = 8
    persist2_m: int = 0         # second (long) persistence window; 0 = off
    persist2_w: int = 32
    debounce_r: int = 2         # and_db: AND fails in r consecutive rounds
    kof: int = 2                # kof4: >= kof factors fail in the round
    beta_lambda: float = 0.5    # beta: forgetting factor (tuned on seed 1)
    beta_theta: float = 0.38    # beta: flag if expected trust < beta_theta

    # --- replication / voting -----------------------------------------------
    n_rep: int = 1
    k: int | None = None        # default floor(n/2)+1
    c_corrupt: int = 0          # corrupted replicas (vote TRUST for attackers)
    frame_legit: bool = False   # corrupted replicas also vote FLAG on honest
    withhold: bool = False      # corrupted replicas send no votes at all
    eta: float = 0.0            # per-factor misreading prob. per replica
                                #   (0 = replicas see identical evidence)
    rho: float = 0.0            # share of misreading common to all replicas
    p_miss: float = 0.0         # per replica-round prob. a vote times out
    miss_rule: str = "quorum"   # "quorum": TRUST iff >=k TRUST votes, FLAG iff
                                #   >= n-k+1 FLAG votes, else keep previous
                                # "naive": FLAG iff fewer than k TRUST votes

    # --- fog-domain suppression ----------------------------------------------
    suppress: float = 0.0       # >0: if more than this share of a domain is
                                #   flagged in a round, hold those verdicts

    # --- quarantine / re-admission (long-horizon metrics) --------------------
    readmit_theta: float = 0.90
    readmit_r: int = 8

    keep: bool = False          # keep tau / scores (for the audit-log run)

    # --- trace-driven honest evidence (traces.py) ------------------------------
    fail_base: object = None    # bool (runs, rounds, N, 4): replaces the
                                #   synthetic congestion when given
    silence_hold: bool = False  # a round in which no evidence of a node
                                #   arrives (all four factors fail) is treated
                                #   as unavailability: no update, verdict held

    def kk(self) -> int:
        return self.k if self.k is not None else self.n_rep // 2 + 1


# ----------------------------------------------------------------------------
# Evidence generation
# ----------------------------------------------------------------------------
def _qmat(cfg: Config, rng):
    R, Tn, N = cfg.runs, cfg.rounds, cfg.n_legit + cfg.n_attack
    base = np.array(cfg.q_vec if cfg.q_vec is not None else [cfg.q] * 4)
    if cfg.markov is None:
        return np.broadcast_to(base, (R, Tn, N, 4)), np.zeros((R, Tn), bool)
    q_lo, q_hi, p_lh, p_hl = cfg.markov
    shape = (R,) if cfg.markov_scope == "network" else (R, N)
    pi_h = p_lh / (p_lh + p_hl)
    state = rng.random(shape) < pi_h
    high = np.zeros((R, Tn) + shape[1:], dtype=bool)
    for t in range(Tn):
        high[:, t] = state
        u = rng.random(shape)
        state = np.where(state, u >= p_hl, u < p_lh)
    q = np.where(high, q_hi, q_lo)
    q = q[:, :, None, None] if cfg.markov_scope == "network" else q[..., None]
    hf = high if cfg.markov_scope == "network" else high.mean(-1) > 0.5
    return np.broadcast_to(q, (R, Tn, N, 4)), hf


def _aware_lies(fail, cfg: Config):
    """Rule-aware on-off liar.  Before each round it predicts its own EMA
    value and window counts assuming no congestion this round, and lies only
    if no trigger of Extension I (EMA threshold, short and long persistence
    windows) would fire.  It observes its own past evidence exactly and knows
    the re-admission rule: once flagged (e.g. because congestion added a
    failure) it behaves honestly until it would be re-admitted."""
    R, Tn, N, _ = fail.shape
    A = slice(cfg.n_legit, N)
    nA = cfg.n_attack
    lied = [T, W] if cfg.profile == "TW" else [W]
    s_ok = 0.3 + 0.3 * 0.96 + 0.4            # attacker's estimate, PDR ~96 %
    lies = np.zeros((R, Tn, nA), dtype=bool)
    tau = np.ones((R, nA))
    hist = np.zeros((R, Tn, nA, 4), dtype=bool)
    quar = np.zeros((R, nA), dtype=bool)
    streak = np.zeros((R, nA), dtype=np.int32)
    windows = [(m, w) for (m, w) in [(cfg.persist_m, cfg.persist_w),
                                     (cfg.persist2_m, cfg.persist2_w)] if m > 0]
    for t in range(Tn):
        s_lie = s_ok - sum(WEIGHTS[f] for f in lied)
        tau_lie = cfg.alpha * s_lie + (1 - cfg.alpha) * tau
        ok = (tau_lie >= cfg.theta + cfg.aware_margin) & ~quar
        for (m, w) in windows:
            lo = max(0, t - w + 1)
            cnt = hist[:, lo:t, :, :].sum(1)              # (R,nA,4)
            for f in lied:
                ok &= (cnt[..., f] + 1) <= m - 1
        lies[:, t] = ok
        f_t = fail[:, t, A, :].copy()
        for f in lied:
            f_t[..., f] |= ok
        hist[:, t] = f_t
        parts = (~f_t).astype(float)
        parts[..., P] = np.where(f_t[..., P], 0.745, 0.96)
        s = (parts * WEIGHTS).sum(-1)
        tau = cfg.alpha * s + (1 - cfg.alpha) * tau
        # would the twin flag it this round?  (its own view of the rule)
        flagged = tau < cfg.theta
        for (m, w) in windows:
            lo = max(0, t - w + 1)
            flagged |= (hist[:, lo:t + 1].sum(1) >= m).any(-1)
        quar |= flagged
        streak = np.where(~flagged & (tau >= cfg.readmit_theta), streak + 1, 0)
        leave = quar & (streak >= cfg.readmit_r)
        quar &= ~leave
        streak = np.where(leave, 0, streak)
    return lies


def _ground_truth_fail(cfg: Config, rng: np.random.Generator):
    """Boolean array fail[r, t, i, f]: factor f of node i fails in round t."""
    R, Tn = cfg.runs, cfg.rounds
    N = cfg.n_legit + cfg.n_attack
    qmat, high = _qmat(cfg, rng)
    if cfg.fail_base is not None:
        fail = np.array(cfg.fail_base, dtype=bool, copy=True)
        assert fail.shape == (R, Tn, N, 4), fail.shape
    else:
        fail = rng.random((R, Tn, N, 4)) < qmat

    if cfg.q_corr > 0:
        fog_of = np.arange(N) % cfg.n_fog
        ev = rng.random((R, Tn, cfg.n_fog)) < cfg.q_corr
        node_ev = ev[:, :, fog_of]
        if cfg.corr_factors >= 4:
            hit = np.ones((R, Tn, N, 4), dtype=bool)
        else:
            order = rng.random((R, Tn, N, 4)).argsort(-1)
            hit = order < cfg.corr_factors
        fail |= node_ev[..., None] & hit

    if cfg.q_node > 0:
        ev = rng.random((R, Tn, N)) < cfg.q_node
        for f in (P, T, W):
            fail[..., f] |= ev

    lies = np.zeros((R, Tn, N), dtype=bool)
    if cfg.profile != "none" and cfg.n_attack > 0:
        if cfg.attacker == "bernoulli":
            la = rng.random((R, Tn, cfg.n_attack)) < cfg.p_lie
        elif cfg.attacker == "periodic":
            phase = rng.integers(0, cfg.period, (R, 1, cfg.n_attack))
            la = ((np.arange(Tn)[None, :, None] + phase) % cfg.period) == 0
        elif cfg.attacker == "aware":
            la = _aware_lies(fail, cfg)
        else:
            raise ValueError(cfg.attacker)
        lies[:, :, cfg.n_legit:] = la
        if cfg.profile == "TW":
            fail[..., T] |= lies
            fail[..., W] |= lies
        elif cfg.profile == "W":
            fail[..., W] |= lies
        else:
            raise ValueError(cfg.profile)
    return fail, lies, high


def _pdr_values(fail_p: np.ndarray, cfg: Config, rng):
    ok = rng.uniform(*cfg.pdr_ok, fail_p.shape)
    bad = rng.uniform(*cfg.pdr_bad, fail_p.shape)
    return np.where(fail_p, bad, ok)


# ----------------------------------------------------------------------------
# Per-twin aggregation
# ----------------------------------------------------------------------------
def _window_count(x, w):
    """x: bool (R,T,...) -> number of True in the last w rounds incl. t."""
    c = np.cumsum(x, axis=1, dtype=np.int32)
    sh = np.zeros_like(c)
    sh[:, w:] = c[:, :-w]
    return c - sh


def aggregate(fail_view: np.ndarray, pdr_view: np.ndarray, cfg: Config, rng):
    """fail_view (R,T,N,X,4) -> flag (R,T,N,X) (True = untrusted), tau.
    tau is the EMA value for "ema"/"ema_p", the expected beta trust for
    "beta", and the instantaneous weighted score otherwise."""
    if cfg.silence_hold:
        return _aggregate_hold(fail_view, pdr_view, cfg, rng)
    return _aggregate(fail_view, pdr_view, cfg, rng)


def _aggregate_hold(fail_view, pdr_view, cfg, rng):
    """Silence rule, applied identically to every aggregator: in a round in
    which all four factors fail (no evidence arrived), the twin makes no
    update and keeps its previous verdict.  Silent rounds add no failures to
    persistence windows, do not advance the EMA or beta counts, and neither
    break nor extend a debounce run."""
    obs = ~fail_view.all(-1)                                # (R,T,N,X)
    fv = fail_view & obs[..., None]
    Tn = fv.shape[1]
    if cfg.agg == "and_db":
        cnt = np.zeros(fv.shape[:1] + fv.shape[2:-1], dtype=np.int32)
        raw = np.zeros(obs.shape, dtype=bool)
        anyf = fv.any(-1)
        for t in range(Tn):
            cnt = np.where(obs[:, t], np.where(anyf[:, t], cnt + 1, 0), cnt)
            raw[:, t] = cnt >= cfg.debounce_r
        s = None
    elif cfg.agg in ("ema", "ema_p", "beta"):
        binary = (~fv).astype(float)
        binary[..., P] = pdr_view / 100.0
        s = (binary * WEIGHTS).sum(-1)
        tau = np.empty_like(s)
        if cfg.agg == "beta":
            lam = cfg.beta_lambda
            x = (~fv.any(-1)).astype(float)
            r = np.full(s.shape[:1] + s.shape[2:], 1.0 / (1 - lam))
            f = np.zeros_like(r)
            for t in range(Tn):
                o = obs[:, t]
                r = np.where(o, lam * r + x[:, t], r)
                f = np.where(o, lam * f + (1 - x[:, t]), f)
                tau[:, t] = (r + 1) / (r + f + 2)
            raw = tau < cfg.beta_theta
        else:
            prev = np.ones(s.shape[:1] + s.shape[2:])
            for t in range(Tn):
                prev = np.where(obs[:, t], cfg.alpha * s[:, t] + (1 - cfg.alpha) * prev, prev)
                tau[:, t] = prev
            raw = tau < cfg.theta
            if cfg.agg == "ema_p":
                raw = raw | (_window_count(fv, cfg.persist_w) >= cfg.persist_m).any(-1)
                if cfg.persist2_m > 0:
                    raw = raw | (_window_count(fv, cfg.persist2_w) >= cfg.persist2_m).any(-1)
        s = tau
    else:
        raw, s = _aggregate(fv, pdr_view, cfg, rng)
    flag = np.empty_like(raw)
    prevf = np.zeros(raw.shape[:1] + raw.shape[2:], dtype=bool)
    for t in range(Tn):
        prevf = np.where(obs[:, t], raw[:, t], prevf)
        flag[:, t] = prevf
    return flag, s


def _aggregate(fail_view: np.ndarray, pdr_view: np.ndarray, cfg: Config, rng):
    Tn = fail_view.shape[1]
    binary = (~fail_view).astype(float)
    binary[..., P] = pdr_view / 100.0
    s = (binary * WEIGHTS).sum(-1)                         # (R,T,N,X)
    anyf = fail_view.any(-1)

    if cfg.sigma_theta > 0:
        theta_t = rng.uniform(cfg.theta - cfg.sigma_theta,
                              cfg.theta + cfg.sigma_theta, s.shape[:2])
        theta_t = theta_t[:, :, None, None]
    else:
        theta_t = cfg.theta

    def persist(flag):
        m, w = cfg.persist_m, cfg.persist_w
        flag = flag | (_window_count(fail_view, w) >= m).any(-1)
        if cfg.persist2_m > 0:
            flag = flag | (_window_count(fail_view, cfg.persist2_w)
                           >= cfg.persist2_m).any(-1)
        return flag

    if cfg.agg == "and":
        return anyf, s
    if cfg.agg == "and_db":
        return _window_count(anyf, cfg.debounce_r) >= cfg.debounce_r, s
    if cfg.agg in ("kof4", "kof4_p"):
        flag = fail_view.sum(-1) >= cfg.kof
        return (persist(flag) if cfg.agg == "kof4_p" else flag), s
    if cfg.agg == "weighted":
        return s < theta_t, s
    if cfg.agg == "beta":
        lam = cfg.beta_lambda
        x = (~anyf).astype(float)
        r = np.full(s.shape[:1] + s.shape[2:], 1.0 / (1 - lam))   # trusted start
        f = np.zeros_like(r)
        ev = np.empty_like(s)
        for t in range(Tn):
            r = lam * r + x[:, t]
            f = lam * f + (1 - x[:, t])
            ev[:, t] = (r + 1) / (r + f + 2)
        return ev < cfg.beta_theta, ev

    tau = np.empty_like(s)
    prev = np.ones(s.shape[:1] + s.shape[2:])
    for t in range(Tn):
        prev = cfg.alpha * s[:, t] + (1 - cfg.alpha) * prev
        tau[:, t] = prev
    flag = tau < theta_t
    if cfg.agg == "ema_p":
        flag = persist(flag)
    elif cfg.agg != "ema":
        raise ValueError(cfg.agg)
    return flag, tau


# ----------------------------------------------------------------------------
# Full simulation
# ----------------------------------------------------------------------------
@dataclass
class Result:
    cfg: Config
    flag: np.ndarray            # (R,T,N) final per-round verdict, True=flagged
    clean: np.ndarray           # (R,T,N) re-admission quorum ("clean") votes
    tau_min_legit: np.ndarray
    high_frac: np.ndarray
    replica_flag: np.ndarray | None = None
    lies: np.ndarray | None = None
    tau: np.ndarray | None = None
    fail_v: np.ndarray | None = None
    pdr: np.ndarray | None = None
    present: np.ndarray | None = None
    suppressed: np.ndarray | None = None

    @property
    def legit(self):
        return self.flag[:, :, : self.cfg.n_legit]

    @property
    def attack(self):
        return self.flag[:, :, self.cfg.n_legit:]

    def fpr_round(self):
        return self.legit.mean()

    def fpr_round_per_run(self):
        return self.legit.mean(axis=(1, 2))

    def false_isolation(self):
        return self.legit.any(axis=1).mean()

    def detect_any(self, j=None):
        a = self.attack.any(axis=1)
        return a.mean(0) if j is None else a[:, j].mean()

    def detect_any_runs(self, j):
        return self.attack.any(axis=1)[:, j]

    def tpr_round(self, j=None):
        a = self.attack.mean(axis=1)
        return a.mean(0) if j is None else a[:, j].mean()

    def ttd_rounds(self, j=0):
        a = self.attack[:, :, j]
        det = a.any(1)
        first = a.argmax(1)[det] + 1
        if det.sum() == 0:
            return np.nan, np.nan, 0
        return float(first.mean()), float(np.median(first)), int(det.sum())


def _quorum(trust_c, flag_c, n, k, rule):
    """trust_c / flag_c: (R,T,N) counts of received TRUST / FLAG votes."""
    if rule == "naive":
        return trust_c < k
    R, Tn, N = trust_c.shape
    out = np.zeros((R, Tn, N), dtype=bool)
    prev = np.zeros((R, N), dtype=bool)                    # start TRUSTED
    for t in range(Tn):
        f = np.where(trust_c[:, t] >= k, False,
                     np.where(flag_c[:, t] >= n - k + 1, True, prev))
        out[:, t] = f
        prev = f
    return out


def simulate(cfg: Config) -> Result:
    rng = np.random.default_rng(cfg.seed)
    fail, lies, high = _ground_truth_fail(cfg, rng)
    R, Tn, N, _ = fail.shape
    n, k = cfg.n_rep, cfg.kk()

    fail_v = np.repeat(fail[:, :, :, None, :], n, axis=3)  # (R,T,N,n,4)
    if cfg.eta > 0:
        flip = rng.random(fail_v.shape) < cfg.eta * (1 - cfg.rho)
        if cfg.rho > 0:
            flip |= rng.random((R, Tn, N, 1, 4)) < cfg.eta * cfg.rho
        fail_v = fail_v ^ flip
    if cfg.eta > 0:
        pdr = _pdr_values(fail_v[..., P], cfg, rng)
    else:   # identical evidence: every replica reads the same PDR value
        pdr = np.repeat(_pdr_values(fail[..., P], cfg, rng)[..., None], n, -1)

    flag_r, tau = aggregate(fail_v, pdr, cfg, rng)          # (R,T,N,n)
    if cfg.agg in ("ema", "ema_p", "beta"):
        clean_r = (~flag_r) & (tau >= cfg.readmit_theta)
    else:
        clean_r = ~flag_r

    c = cfg.c_corrupt
    if c > 0:
        flag_r = flag_r.copy()
        flag_r[:, :, cfg.n_legit:, :c] = False
        clean_r[:, :, cfg.n_legit:, :c] = True
        if cfg.frame_legit:
            flag_r[:, :, : cfg.n_legit, :c] = True
            clean_r[:, :, : cfg.n_legit, :c] = False

    present = np.ones((R, Tn, 1, n), dtype=bool)
    if cfg.p_miss > 0:
        present = rng.random((R, Tn, 1, n)) >= cfg.p_miss
    if cfg.withhold and c > 0:
        present = present.copy()
        present[..., :c] = False
    trust_c = ((~flag_r) & present).sum(-1)
    flag_c = (flag_r & present).sum(-1)
    final = _quorum(trust_c, flag_c, n, k, cfg.miss_rule)
    clean = (clean_r & present).sum(-1) >= k

    suppressed = None
    if cfg.suppress > 0:
        fog_of = np.arange(N) % cfg.n_fog
        suppressed = np.zeros((R, Tn, N), dtype=bool)
        for d in range(cfg.n_fog):
            idx = np.where(fog_of == d)[0]
            frac = final[:, :, idx].mean(-1)                 # (R,T)
            hold = frac > cfg.suppress
            suppressed[:, :, idx] = hold[..., None] & final[:, :, idx]
        final = final & ~suppressed

    if cfg.agg in ("ema", "ema_p"):
        tau_min = tau[:, :, : cfg.n_legit].min(axis=(1, 2, 3))
    else:
        tau_min = np.full(R, np.nan)
    res = Result(cfg, final, clean, tau_min, high.mean(1), flag_r, lies,
                 present=present, suppressed=suppressed)
    if cfg.keep:
        res.tau, res.fail_v, res.pdr = tau, fail_v, pdr
    return res


# ----------------------------------------------------------------------------
# Quarantine with re-admission (long-horizon operation)
# ----------------------------------------------------------------------------
def quarantine(res: Result):
    """A node enters quarantine at its first FLAG verdict and is re-admitted
    after readmit_r consecutive rounds whose re-admission quorum is "clean"
    (k replicas report TRUST with tau >= readmit_theta).  Returns
    q[R,T,N] (True = quarantined) and the number of quarantine entries."""
    cfg = res.cfg
    R, Tn, N = res.flag.shape
    q = np.zeros((R, Tn, N), dtype=bool)
    state = np.zeros((R, N), dtype=bool)
    streak = np.zeros((R, N), dtype=np.int32)
    entries = np.zeros((R, N), dtype=np.int32)
    for t in range(Tn):
        new = res.flag[:, t] & ~state
        entries += new
        state = state | res.flag[:, t]
        streak = np.where(res.clean[:, t] & ~res.flag[:, t], streak + 1, 0)
        leave = state & (streak >= cfg.readmit_r)
        state = state & ~leave
        streak = np.where(leave, 0, streak)
        q[:, t] = state
    return q, entries


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def boot_ci(per_run: np.ndarray, B: int = 2000, seed: int = 12345):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_run), (B, len(per_run)))
    means = per_run[idx].mean(1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
