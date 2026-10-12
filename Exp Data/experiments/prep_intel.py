"""Pre-processes the Intel Berkeley Research Lab trace (54 Mica2Dot motes,
28 Feb - 5 Apr 2004; https://db.csail.mit.edu/labdata/labdata.html) into a
delivery matrix on a 31-s slot grid.

Input : ../datasets/intel_lab/data.txt.gz  (date time epoch moteid temp hum light voltage)
Output: ../datasets/intel_lab/intel_slots.npz
    delivered[m, s]  a reading of mote m+1 arrived in 31-s slot s
    alive[m, s]      slot s lies between the mote's first and last reading
    gap[s]           logging gap: fewer than 5 % of the live motes delivered in
                     slot s (base-station or logger outage), or fewer than 10
                     motes alive; such slots carry no evidence about any node
Epoch numbers wrap and drift against the clock in this trace, so slots are
computed from the base-station timestamps.  Because of timestamp jitter two
consecutive readings of a mote can fall into the same slot; the later one
is then moved to the next slot (it cannot be more than one slot late), so
jitter is not counted as a loss.  Exact duplicates (same mote, same epoch)
count once.
"""
import datetime as dt
import gzip
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / "datasets" / "intel_lab"
N_MOTES, SLOT_S = 54, 31.0
T0 = dt.datetime(2004, 2, 28).timestamp()


def main():
    ts, ms, skipped = [], [], 0
    with gzip.open(D / "data.txt.gz", "rt") as f:
        for line in f:
            p = line.split()
            try:
                m = int(p[3])
                t = dt.datetime.strptime((p[0] + " " + p[1])[:19], "%Y-%m-%d %H:%M:%S")
            except (IndexError, ValueError):
                skipped += 1
                continue
            if not 1 <= m <= N_MOTES:
                skipped += 1
                continue
            ts.append(t.timestamp() - T0)
            ms.append(m - 1)
    ts, ms = np.array(ts), np.array(ms)
    order = np.lexsort((ts, ms))
    ts, ms = ts[order], ms[order]
    raw = np.floor(ts / SLOT_S).astype(np.int64)
    slot = raw.copy()
    moved = 0
    for i in range(1, len(slot)):
        if ms[i] == ms[i - 1] and slot[i] <= slot[i - 1]:
            if slot[i - 1] + 1 - ts[i] / SLOT_S < 1.0 and ts[i] - ts[i - 1] > 5.0:
                slot[i] = slot[i - 1] + 1
                moved += 1
    S = int(slot.max()) + 1
    delivered = np.zeros((N_MOTES, S), dtype=bool)
    delivered[ms, slot] = True
    first = delivered.argmax(1)
    last = S - 1 - delivered[:, ::-1].argmax(1)
    alive = np.zeros_like(delivered)
    for m in range(N_MOTES):
        alive[m, first[m]:last[m] + 1] = True
    n_alive = alive.sum(0)
    frac = (delivered & alive).sum(0) / np.maximum(n_alive, 1)
    gap = (frac < 0.05) | (n_alive < 10)
    np.savez_compressed(D / "intel_slots.npz", delivered=delivered, alive=alive, gap=gap)
    print(f"readings {len(ts):,}  moved {moved:,}  skipped lines {skipped:,}  slots {S:,}  "
          f"gap slots {gap.mean():.3%}  delivery in live slots "
          f"{delivered[alive][:].mean():.3f}")


if __name__ == "__main__":
    main()
