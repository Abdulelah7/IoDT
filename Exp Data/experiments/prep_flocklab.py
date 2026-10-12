"""Pre-processes the FlockLab link-quality dataset (Jacob et al., DATA'19;
https://zenodo.org/records/3731498, file 2019-08_preprocessed_all.zip,
TelosB ("sky") platform, August 2019, channel 26, 0 dBm, 8-byte payload).

In every test (about every two hours) each node sends 100 back-to-back
packets ("strobes") and every other node logs which of them it received.

Topology used here (fixed before looking at any trust result): two nodes act
as fog gateways; every other node sends its evidence directly to the gateway
it hears best on average (single hop, as in the IoDT topology of Soula et
al.).  The gateway pair is the one that puts the most sensors in range (best
long-run packet reception ratio, PRR, of at least 50 %), ties broken by
the mean PRR of the chosen links; sensors out of range of both gateways
are excluded.  A test in which a sensor was heard by no node at all is
treated as a gap (the sensor did not run), not as packet loss.

Output: ../datasets/flocklab/flocklab_sky_2019-08.npz
    streams[s, j, 100]  strobe k of test j from sensor s reached its gateway
    valid[s, j]         the test has a record for that link
    sensors, gateways, test_numbers, test_times
"""
import itertools
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent.parent / "datasets" / "flocklab"
CSV = "2019-08/2019-08_FlockLab_sky.csv"


def main():
    with zipfile.ZipFile(D / "2019-08_preprocessed_all.zip") as z:
        df = pd.read_csv(z.open(CSV), dtype={"rcv_stream": str})
    df = df[df.rcv_stream.str.len() == 100]
    prr = df.groupby(["snd_id", "rcv_id"]).rcv_total.mean().unstack() / 100
    nodes = sorted(set(df.snd_id) | set(df.rcv_id))
    prr = prr.reindex(index=nodes, columns=nodes).fillna(0.0)
    best_pair, best_val = None, (-1, -1.0)
    for g in itertools.combinations(nodes, 2):
        s = [n for n in nodes if n not in g]
        v = prr.loc[s, list(g)].max(axis=1)
        val = (int((v >= 0.5).sum()), float(v[v >= 0.5].mean()))
        if val > best_val:
            best_pair, best_val = g, val
    gws = list(best_pair)
    sens = [n for n in nodes if n not in gws and prr.loc[n, gws].max() >= 0.5]
    parent = {n: gws[int(np.argmax(prr.loc[n, gws].values))] for n in sens}
    tests = df[["test_number", "date_time"]].drop_duplicates().sort_values("date_time")
    tnum = tests.test_number.values
    tidx = {t: i for i, t in enumerate(tnum)}
    streams = np.zeros((len(sens), len(tnum), 100), dtype=bool)
    valid = np.zeros((len(sens), len(tnum)), dtype=bool)
    heard = df.groupby(["snd_id", "test_number"]).rcv_total.sum()
    for si, s in enumerate(sens):
        sub = df[(df.snd_id == s) & (df.rcv_id == parent[s])]
        for t, st in zip(sub.test_number.values, sub.rcv_stream.values):
            if heard.get((s, t), 0) == 0:
                continue
            j = tidx[t]
            streams[si, j] = np.frombuffer(st.encode(), dtype=np.uint8) == ord("1")
            valid[si, j] = True
    np.savez_compressed(D / "flocklab_sky_2019-08.npz", streams=streams, valid=valid,
                        sensors=np.array(sens), gateways=np.array(gws),
                        parents=np.array([parent[s] for s in sens]),
                        test_numbers=tnum, test_times=tests.date_time.values.astype(str))
    lr = streams[valid].mean()
    print(f"gateways {gws}  sensors {len(sens)}  tests {len(tnum)}  "
          f"valid link-tests {valid.mean():.3f}  strobe PRR {lr:.4f}")


if __name__ == "__main__":
    main()
