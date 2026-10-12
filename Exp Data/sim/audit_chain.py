"""
Extension III: signed, hash-chained Merkle audit log of trust verdicts,
optionally anchored in a permissioned ledger and an external checkpoint.

Implementation notes:
  * RFC 6962-style Merkle tree: leaf hash H(0x00 || d), interior hash
    H(0x01 || l || r), split at the largest power of two below n, no
    duplication of the last node; the leaf count is committed in the header.
  * Every record has a fixed length that the verifier checks.
  * The signed replica votes are logged and anchored (votes root), each vote
    is bound to (replica, round, epoch), and the verifier checks every
    verdict against the quorum of its votes.
  * The per-node state is committed by a Merkle root; only the header goes
    on chain, so the on-chain size per block no longer grows with |N|.
  * Replicas co-sign a block only after re-checking it (endorsement), so a
    trust engine that writes a verdict its votes do not support cannot obtain
    k signatures.
  * Replica keys are grouped by epoch; the epoch is in every vote and header.

The ordering service of a ledger is NOT simulated; its latency is a
parameter (T_prop).  Nothing in the tamper-evidence argument depends on it.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

# ---------------------------------------------------------------------------
# Record formats (fixed length; the verifier rejects any other length)
# ---------------------------------------------------------------------------
EVENT_FMT = "<II4ffB"          # node, round, 4 factor scores, tau, verdict
EVENT_BYTES = struct.calcsize(EVENT_FMT)                 # 29
STATE_FMT = "<IIfB"            # node, last round, tau, verdict
STATE_BYTES = struct.calcsize(STATE_FMT)                 # 13
VOTE_HDR_FMT = "<BIIH"         # replica id, round, epoch, number of nodes
HEADER_FMT = "<IIIIII32s32s32s32s"
#            height, first round, last round, #events, #votes, epoch,
#            prev hash, events root, votes root, state root
HEADER_BYTES = struct.calcsize(HEADER_FMT)               # 152
SIG_SLOT_BYTES = 1 + 72        # signer id + DER ECDSA (max 72 B)

TRUST, FLAG = 1, 0


def h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    """RFC 6962 Merkle Tree Hash."""
    n = len(leaves)
    if n == 0:
        return h(b"")
    if n == 1:
        return h(b"\x00" + leaves[0])
    k = 1
    while k * 2 < n:
        k *= 2
    return h(b"\x01" + merkle_root(leaves[:k]) + merkle_root(leaves[k:]))


def merkle_root_legacy(leaves: list[bytes]) -> bytes:
    """The tree used before this revision (kept only to demonstrate the
    forgeries it admitted)."""
    if not leaves:
        return h(b"")
    level = [h(x) for x in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [h(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def encode_event(node, rnd, factors, tau, verdict) -> bytes:
    return struct.pack(EVENT_FMT, node, rnd, *map(float, factors), tau, verdict)


def decode_event(b: bytes):
    node, rnd, f0, f1, f2, f3, tau, verdict = struct.unpack(EVENT_FMT, b)
    return node, rnd, (f0, f1, f2, f3), tau, verdict


def encode_state(node, rnd, tau, verdict) -> bytes:
    return struct.pack(STATE_FMT, node, rnd, tau, verdict)


def vote_record(replica: int, rnd: int, epoch: int, verdicts: list[int]) -> bytes:
    """One replica's verdict vector for one round (batch vote): header +
    one byte per node (TRUST=1 / FLAG=0)."""
    return struct.pack(VOTE_HDR_FMT, replica, rnd, epoch, len(verdicts)) + \
        bytes(verdicts)


def decode_vote(rec: bytes):
    hl = struct.calcsize(VOTE_HDR_FMT)
    rid, rnd, epoch, n = struct.unpack(VOTE_HDR_FMT, rec[:hl])
    v = list(rec[hl:])
    if len(v) != n:
        raise ValueError("vote length")
    return rid, rnd, epoch, v


def quorum_verdict(votes_for_node: list[int], n: int, k: int, prev: int) -> int:
    """Quorum rule of Algorithm 2: TRUST iff >= k TRUST votes, FLAG iff
    >= n-k+1 FLAG votes, otherwise (votes missing) keep the previous verdict."""
    t = sum(1 for v in votes_for_node if v == TRUST)
    f = sum(1 for v in votes_for_node if v == FLAG)
    if t >= k:
        return TRUST
    if f >= n - k + 1:
        return FLAG
    return prev


def _sig(key, msg: bytes) -> bytes:
    return key.sign(msg, ec.ECDSA(hashes.SHA256()))


def _ok(pub, sig: bytes, msg: bytes) -> bool:
    try:
        pub.verify(sig, msg, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


@dataclass
class Block:
    height: int
    first_round: int
    last_round: int
    n_events: int
    n_votes: int
    epoch: int
    prev_hash: bytes
    events_root: bytes
    votes_root: bytes
    state_root: bytes
    sigs: list = field(default_factory=list)      # (replica id, DER sig)
    hash: bytes = b""

    def header(self) -> bytes:
        return struct.pack(HEADER_FMT, self.height, self.first_round,
                           self.last_round, self.n_events, self.n_votes,
                           self.epoch, self.prev_hash, self.events_root,
                           self.votes_root, self.state_root)

    def compute_hash(self) -> bytes:
        return h(self.header() + b"".join(bytes([i]) + s for i, s in self.sigs))

    def onchain_bytes(self) -> int:
        return HEADER_BYTES + sum(1 + len(s) for _, s in self.sigs)


@dataclass
class Window:
    """Off-chain data of one block: events, signed votes, state records."""
    events: list[bytes]
    votes: list[tuple[bytes, bytes]]              # (vote record, signature)
    state: list[bytes]


class AuditChain:
    def __init__(self, K: int, n_rep: int = 5, k: int = 3, n_nodes: int = 8):
        self.K, self.n, self.k, self.n_nodes = K, n_rep, k, n_nodes
        self.epoch = 0
        self.keys = {0: [ec.generate_private_key(ec.SECP256K1())
                         for _ in range(n_rep)]}
        self.blocks: list[Block] = []
        self.windows: list[Window] = []
        self._ev: list[bytes] = []
        self._votes: list[tuple[bytes, bytes]] = []
        self._first = None
        self._state: dict[int, bytes] = {}
        self._verdict = {i: TRUST for i in range(n_nodes)}

    # ---- key management -------------------------------------------------
    def pub(self, epoch, i):
        return self.keys[epoch][i].public_key()

    def rotate(self):
        self.epoch += 1
        self.keys[self.epoch] = [ec.generate_private_key(ec.SECP256K1())
                                 for _ in range(self.n)]

    # ---- one round ------------------------------------------------------
    def add_round(self, rnd: int, replica_verdicts, factors, taus,
                  present=None):
        """replica_verdicts[r][i] in {TRUST, FLAG}; present[r] False means the
        replica's vote did not arrive (timeout).  The trust engine computes
        the verdict by the quorum rule and appends event + votes."""
        if self._first is None:
            self._first = rnd
        present = present or [True] * self.n
        received = []
        for r in range(self.n):
            if not present[r]:
                continue
            rec = vote_record(r, rnd, self.epoch, list(replica_verdicts[r]))
            self._votes.append((rec, _sig(self.keys[self.epoch][r], rec)))
            received.append(list(replica_verdicts[r]))
        for i in range(self.n_nodes):
            v = quorum_verdict([vv[i] for vv in received], self.n, self.k,
                               self._verdict[i])
            self._verdict[i] = v
            ev = encode_event(i, rnd, factors[i], taus[i], v)
            self._ev.append(ev)
            self._state[i] = encode_state(i, rnd, taus[i], v)
        if rnd % self.K == 0:
            self.commit(rnd)

    def _propose(self, rnd) -> tuple[Block, Window]:
        prev = self.blocks[-1].hash if self.blocks else b"\0" * 32
        state = [self._state[i] for i in sorted(self._state)]
        w = Window(list(self._ev), list(self._votes), state)
        b = Block(len(self.blocks), self._first, rnd, len(w.events),
                  len(w.votes), self.epoch, prev, merkle_root(w.events),
                  merkle_root([r + s for r, s in w.votes]), merkle_root(state))
        return b, w

    def commit(self, rnd, block_window=None):
        if not self._ev:
            return False
        b, w = block_window or self._propose(rnd)
        prev_verdicts = self._prev_verdicts()
        signers = [r for r in range(self.n)
                   if endorse(b, w, self, prev_verdicts)]   # each replica checks
        b.sigs = [(r, _sig(self.keys[b.epoch][r], b.header()))
                  for r in signers[: self.k]]
        if len(b.sigs) < self.k:
            return False                                    # cannot be committed
        b.hash = b.compute_hash()
        self.blocks.append(b)
        self.windows.append(w)
        self._ev, self._votes, self._first = [], [], None
        return True

    def flush(self, rnd):
        self.commit(rnd)

    def _prev_verdicts(self, upto=None):
        """Verdict of every node at the end of block `upto`-1 (TRUST before
        the first block)."""
        v = {i: TRUST for i in range(self.n_nodes)}
        for w in self.windows[: upto if upto is not None else len(self.windows)]:
            for ev in w.events:
                node, _, _, _, verdict = decode_event(ev)
                v[node] = verdict
        return v

    def storage(self) -> dict:
        on = sum(b.onchain_bytes() for b in self.blocks)
        off = sum(len(w.events) * EVENT_BYTES + sum(len(r) + len(s) for r, s in w.votes)
                  + len(w.state) * STATE_BYTES for w in self.windows)
        return dict(blocks=len(self.blocks), onchain=on, offchain=off,
                    max_block=max((b.onchain_bytes() for b in self.blocks), default=0))


# ---------------------------------------------------------------------------
# Checks shared by replicas (before signing) and by the auditor
# ---------------------------------------------------------------------------
def check_window(b: Block, w: Window, chain: AuditChain, prev_verdicts) -> bool:
    n, k = chain.n, chain.k
    if len(w.events) != b.n_events or len(w.votes) != b.n_votes:
        return False
    if any(len(e) != EVENT_BYTES for e in w.events):
        return False
    if any(len(s) != STATE_BYTES for s in w.state):
        return False
    if merkle_root(w.events) != b.events_root:
        return False
    if merkle_root([r + s for r, s in w.votes]) != b.votes_root:
        return False
    if merkle_root(w.state) != b.state_root:
        return False
    if b.epoch not in chain.keys:
        return False
    # votes: signature, epoch, round window, one vote per replica per round
    by_round: dict[int, dict[int, list[int]]] = {}
    for rec, sig in w.votes:
        try:
            rid, rnd, epoch, v = decode_vote(rec)
        except Exception:
            return False
        if epoch != b.epoch or not (b.first_round <= rnd <= b.last_round):
            return False
        if rid >= n or len(v) != chain.n_nodes:
            return False
        if not _ok(chain.pub(epoch, rid), sig, rec):
            return False
        if rid in by_round.setdefault(rnd, {}):
            return False                                   # duplicate vote
        by_round[rnd][rid] = v
    # every verdict must follow from its round's votes (quorum rule)
    verdict = dict(prev_verdicts)
    last = {}
    seen = set()
    for ev in w.events:
        node, rnd, _, tau, v = decode_event(ev)
        if (node, rnd) in seen or not (b.first_round <= rnd <= b.last_round):
            return False
        seen.add((node, rnd))
        votes = [vv[node] for vv in by_round.get(rnd, {}).values()]
        if v != quorum_verdict(votes, n, k, verdict[node]):
            return False
        verdict[node] = v
        last[node] = (rnd, tau, v)
    # state records = latest event of each node
    for s in w.state:
        node, rnd, tau, v = struct.unpack(STATE_FMT, s)
        if node in last and (rnd, v) != (last[node][0], last[node][2]):
            return False
        if node in last and abs(tau - last[node][1]) > 0:
            return False
    return True


def endorse(b: Block, w: Window, chain: AuditChain, prev_verdicts) -> bool:
    """An honest replica signs the header only if the off-chain data it
    commits to is well formed and every verdict follows from the votes."""
    return check_window(b, w, chain, prev_verdicts)


def verify(chain: AuditChain, checkpoint: tuple[int, bytes] | None = None) -> bool:
    prev = b"\0" * 32
    verdicts = {i: TRUST for i in range(chain.n_nodes)}
    for b, w in zip(chain.blocks, chain.windows):
        if b.prev_hash != prev or b.compute_hash() != b.hash:
            return False
        good = {i for i, s in b.sigs
                if i < chain.n and _ok(chain.pub(b.epoch, i), s, b.header())}
        if len(good) < chain.k:
            return False
        if not check_window(b, w, chain, verdicts):
            return False
        for ev in w.events:
            node, _, _, _, v = decode_event(ev)
            verdicts[node] = v
        prev = b.hash
    if checkpoint is not None:
        j, hj = checkpoint
        if j >= len(chain.blocks) or chain.blocks[j].hash != hj:
            return False
    return True


def recompute_forward(chain: AuditChain, start: int, stolen=()):
    """Adversary re-derives roots and hashes from block `start` onward after
    an edit; it re-signs with the replica keys it holds (stolen) and leaves
    other signatures untouched (they no longer verify)."""
    for i in range(start, len(chain.blocks)):
        b, w = chain.blocks[i], chain.windows[i]
        b.n_events, b.n_votes = len(w.events), len(w.votes)
        b.events_root = merkle_root(w.events)
        b.votes_root = merkle_root([r + s for r, s in w.votes])
        b.state_root = merkle_root(w.state)
        b.prev_hash = chain.blocks[i - 1].hash if i else b"\0" * 32
        if stolen:
            others = [(r, s) for r, s in b.sigs if r not in stolen]
            b.sigs = [(r, _sig(chain.keys[b.epoch][r], b.header()))
                      for r in stolen] + others
        b.hash = b.compute_hash()


def resign_votes(chain: AuditChain, i: int, stolen):
    """With replica keys, the adversary can also forge those replicas' votes."""
    w = chain.windows[i]
    out = []
    for rec, sig in w.votes:
        rid = rec[0]
        if rid in stolen:
            sig = _sig(chain.keys[chain.blocks[i].epoch][rid], rec)
        out.append((rec, sig))
    w.votes = out
