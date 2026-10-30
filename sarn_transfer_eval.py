"""Evaluate frozen SARN embeddings for the Rome emission task (LOCO transfer).

Maps the SARN segment embeddings (186,664 directed) back onto the project's
undirected edges, joins the emission labels, and runs the leave-one-class-out
(LOCO) transfer test with XGBoost on the FROZEN embeddings.  This is the
"unsupervised SARN" arm, to compare against the supervised GCN result
(gnn_transfer.json).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = Path("/Users/gurban.aliyevuni-a.de/workspace/unipi")
SARN = BASE / "gps2exposure" / "SARN"
MAP_FEATURES = BASE / "PhD_Thesis_Gurban_Aliyev" / "data" / "rome" / "imputed" / "v3" / "map_complete" / "map_features.parquet"
EMBS = SARN / "exp" / "snapshots" / "OSM_Rome_raw_SARN_best_embs.pickle"
SEG = SARN / "data" / "OSM_Rome_raw_segment"
ADJ = SARN / "data" / "OSM_Rome_raw_adjsegment"
OUT = SARN / "exp" / "sarn_transfer_eval.json"
SEED = 42


def mass_metrics(true, pred, length):
    t = np.asarray(true, float); p = np.asarray(pred, float); l = np.asarray(length, float)
    m = np.isfinite(t) & np.isfinite(p) & np.isfinite(l)
    t, p, l = t[m], p[m], l[m]
    if t.sum() <= 0:
        return {"n": int(m.sum()), "wape_pct": np.nan, "bias_pct": np.nan, "mape_pct": np.nan}
    return {"n": int(m.sum()),
            "wape_pct": float(100 * np.sum(l * np.abs(p - t)) / np.sum(l * t)),
            "bias_pct": float(100 * np.sum(l * (p - t)) / np.sum(l * t)),
            "mape_pct": float(100 * np.mean(np.abs(p - t) / np.maximum(t, 1e-6)))}


def main():
    print("loading embeddings ...", flush=True)
    embs = pickle.load(open(EMBS, "rb"))
    embs = embs.cpu().numpy().astype(np.float32)
    n_emb = embs.shape[0]
    print(f"  embeddings {embs.shape}", flush=True)

    # segment inc_id -> (s_id, e_id)
    seg = pd.read_csv(SEG, usecols=["s_id", "e_id", "inc_id"], dtype={"s_id": "int64", "e_id": "int64", "inc_id": "int64"})
    inc_to_se = dict(zip(seg.inc_id, zip(seg.s_id, seg.e_id)))

    # dual graph from adjsegment (s_id, e_id are inc_ids), keep largest WCC
    adj = pd.read_csv(ADJ, usecols=["s_id", "e_id"])
    G = nx.from_pandas_edgelist(adj, "s_id", "e_id", create_using=nx.DiGraph())
    wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    surviving = sorted(wccs[0])
    print(f"  graph: {len(G)} nodes, largest WCC {len(surviving)} (embeddings {n_emb})", flush=True)
    assert len(surviving) == n_emb, f"mismatch {len(surviving)} vs {n_emb}"

    # directed (s_id, e_id) -> embedding index
    se_to_idx = {}
    for i, inc in enumerate(surviving):
        s, e = inc_to_se[inc]
        se_to_idx[(int(s), int(e))] = i

    # ---- map to undirected edges -----------------------------------------
    f = pd.read_parquet(MAP_FEATURES)
    obs = f[f.observed == 1].copy()
    obs["mean_weekly_cpm"] = obs.annual_CO2 / (obs.length_m * obs.n_obs)
    obs = obs[np.isfinite(obs.mean_weekly_cpm) & (obs.mean_weekly_cpm > 0)].copy()

    def edge_emb(a, b):
        idx = []
        if (a, b) in se_to_idx:
            idx.append(se_to_idx[(a, b)])
        if (b, a) in se_to_idx:
            idx.append(se_to_idx[(b, a)])
        return embs[idx].mean(axis=0) if idx else None

    rows = []
    matched = 0
    for a, b in zip(obs.a.values, obs.b.values):
        a, b = int(a), int(b)
        e = edge_emb(a, b)
        if e is not None:
            rows.append(e)
            matched += 1
        else:
            rows.append(np.full(embs.shape[1], np.nan, dtype=np.float32))
    X = np.vstack(rows)
    print(f"  observed edges {len(obs)}, matched embeddings {matched} ({100*matched/len(obs):.1f}%)", flush=True)

    # drop unmatched (no embedding) for the evaluation
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]; obs = obs[ok].reset_index(drop=True)
    y = np.log1p(obs.mean_weekly_cpm.values)
    length = obs.length_m.values
    hw = obs.hw_group.values
    print(f"  evaluating on {len(obs)} observed edges with embeddings", flush=True)

    # ---- random holdout 80/20 --------------------------------------------
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(obs))
    tr, te = idx[:int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
    m = HistGradientBoostingRegressor(max_iter=400, max_depth=6, learning_rate=0.05,
                                      max_leaf_nodes=63, random_state=SEED)
    m.fit(X[tr], y[tr])
    pred = np.expm1(m.predict(X[te]))
    holdout = mass_metrics(np.expm1(y[te]), pred, length[te])
    print(f"[holdout] n={holdout['n']} WAPE={holdout['wape_pct']:.1f}% bias={holdout['bias_pct']:+.1f}%", flush=True)

    # ---- leave-one-class-out ---------------------------------------------
    classes = [c for c in np.unique(hw) if (hw == c).sum() >= 200]
    print(f"\n{'class':14s} {'n':>6s} {'bias%':>8s}  (frozen SARN embedding)", flush=True)
    loco = {}
    for c in classes:
        trm = hw != c
        tem = hw == c
        if trm.sum() < 100 or tem.sum() < 10:
            continue
        mdl = HistGradientBoostingRegressor(max_iter=400, max_depth=6, learning_rate=0.05,
                                            max_leaf_nodes=63, random_state=SEED)
        mdl.fit(X[trm], y[trm])
        pr = np.expm1(mdl.predict(X[tem]))
        met = mass_metrics(np.expm1(y[tem]), pr, length[tem])
        loco[str(c)] = met
        print(f"{c:14s} {met['n']:6d} {met['bias_pct']:+8.1f}%", flush=True)

    out = {"n_emb": int(n_emb), "matched_edges": int(ok.sum()),
           "holdout": holdout, "loco": loco}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
