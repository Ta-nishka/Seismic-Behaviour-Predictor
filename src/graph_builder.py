"""
SeismoNet - graph_builder.py
Builds a heterogeneous spatio-temporal graph from events.csv + stations.csv.

Nodes : event, station
Edges : event  -> station   (5 nearest stations, distance-weighted)
        station -> event    (reverse of the above, so info flows both ways)
        station -> station  (5 nearest neighbours, symmetric)
        event  -> event     (earlier event -> later event, within 50 km / 7 days)

Design choices (write these in your report):
  * magnitude and depth are NOT input features, because they are the targets.
  * event-station edges use nearest-k stations, NOT a magnitude-based radius,
    otherwise the graph structure would leak magnitude.
  * event-event edges only point forward in time (past -> future).
  * the split is chronological: first 70% train, next 15% val, last 15% test.
  * feature scaling uses training-set statistics only.

Run from the gml_mp root:   python src/graph_builder.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import HeteroData

PROC = Path("data/processed")
GRAPHS = Path("data/graphs")

K_EVENT_STATION = 5      # stations linked to each event
K_STATION_STATION = 5    # neighbours per station
EE_KM = 50.0             # event-event spatial radius
EE_HOURS = 7 * 24.0      # event-event time window
EE_MAX = 10              # max past neighbours per event
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
DEPTH_MAP = {"shallow": 0, "mid": 1, "deep": 2}

EVENT_FEATURES = ["latitude", "longitude", "nst", "gap", "dmin", "rms",
                  "log_dt_prev", "log_density_past"]


def to_xy_km(lat, lon, lat0=34.5, lon0=-117.5):
    """Simple flat projection to kilometres (fine for a ~500 km region)."""
    x = (np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * 111.0
    return x, y


def build_event_event_edges(t, x, y):
    """Directed edges earlier -> later. Also returns past-density feature."""
    n = len(t)
    src, dst, attr = [], [], []
    density = np.zeros(n)
    for i in range(n):
        lo = np.searchsorted(t, t[i] - EE_HOURS, side="left")
        if lo >= i:
            continue
        cand = np.arange(lo, i)
        d = np.hypot(x[cand] - x[i], y[cand] - y[i])
        keep = d <= EE_KM
        cand, d = cand[keep], d[keep]
        density[i] = len(cand)
        if len(cand) > EE_MAX:
            nearest = np.argsort(d)[:EE_MAX]
            cand, d = cand[nearest], d[nearest]
        for j, dj in zip(cand, d):
            src.append(j)
            dst.append(i)
            attr.append([dj / EE_KM, (t[i] - t[j]) / EE_HOURS])
    edge_index = np.array([src, dst], dtype=np.int64).reshape(2, -1)
    return edge_index, np.array(attr, dtype=np.float32).reshape(-1, 2), density


def build_graph(events: pd.DataFrame, stations: pd.DataFrame):
    events = events.copy()
    events["time"] = pd.to_datetime(events["time"], utc=True, format="ISO8601")
    events = events.sort_values("time").reset_index(drop=True)
    n_ev, n_st = len(events), len(stations)

    # time in hours since first event (avoids datetime resolution issues)
    t = ((events["time"] - events["time"].min()).dt.total_seconds() / 3600).to_numpy()
    ex, ey = to_xy_km(events["latitude"], events["longitude"])
    sx, sy = to_xy_km(stations["latitude"], stations["longitude"])

    # ---- chronological split
    n_train, n_val = int(TRAIN_FRAC * n_ev), int(VAL_FRAC * n_ev)
    split = np.array(["train"] * n_train + ["val"] * n_val +
                     ["test"] * (n_ev - n_train - n_val))

    # ---- edges
    st_tree = cKDTree(np.c_[sx, sy])
    dist, idx = st_tree.query(np.c_[ex, ey], k=K_EVENT_STATION)
    es_src = np.repeat(np.arange(n_ev), K_EVENT_STATION)
    es_dst = idx.ravel()
    es_attr = np.c_[dist.ravel() / 100.0, 1.0 / (1.0 + dist.ravel() / 10.0)].astype(np.float32)

    _, nn = st_tree.query(np.c_[sx, sy], k=K_STATION_STATION + 1)   # first hit is itself
    ss_src = np.repeat(np.arange(n_st), K_STATION_STATION)
    ss_dst = nn[:, 1:].ravel()
    ss_dist = np.hypot(sx[ss_src] - sx[ss_dst], sy[ss_src] - sy[ss_dst])
    ss_attr = np.c_[ss_dist / 100.0, 1.0 / (1.0 + ss_dist / 10.0)].astype(np.float32)
    # make station-station symmetric
    ss_index = np.concatenate([np.c_[ss_src, ss_dst].T, np.c_[ss_dst, ss_src].T], axis=1)
    ss_attr = np.concatenate([ss_attr, ss_attr])

    ee_index, ee_attr, density = build_event_event_edges(t, ex, ey)

    # ---- event features (scaled with train stats only)
    events["log_dt_prev"] = np.log1p(events["dt_prev_hours"].clip(lower=0))
    events["log_density_past"] = np.log1p(density)
    feats = events[EVENT_FEATURES].astype(float)
    train_mask = split == "train"
    mu, sd = feats[train_mask].mean(), feats[train_mask].std().replace(0, 1)
    ev_x = ((feats - mu) / sd).to_numpy(dtype=np.float32)

    st_feats = stations[["latitude", "longitude", "elevation"]].astype(float)
    st_x = ((st_feats - st_feats.mean()) / st_feats.std().replace(0, 1)).to_numpy(dtype=np.float32)

    # ---- assemble HeteroData
    data = HeteroData()
    data["event"].x = torch.tensor(ev_x)
    data["event"].y_mag = torch.tensor(events["mag"].to_numpy(dtype=np.float32))
    data["event"].y_depth = torch.tensor(events["depth_class"].map(DEPTH_MAP).to_numpy(dtype=np.int64))
    data["event"].time = torch.tensor(t, dtype=torch.float32)
    data["event"].train_mask = torch.tensor(split == "train")
    data["event"].val_mask = torch.tensor(split == "val")
    data["event"].test_mask = torch.tensor(split == "test")
    data["station"].x = torch.tensor(st_x)

    data["event", "recorded_by", "station"].edge_index = torch.tensor(np.c_[es_src, es_dst].T, dtype=torch.long)
    data["event", "recorded_by", "station"].edge_attr = torch.tensor(es_attr)
    data["station", "rev_recorded_by", "event"].edge_index = torch.tensor(np.c_[es_dst, es_src].T, dtype=torch.long)
    data["station", "rev_recorded_by", "event"].edge_attr = torch.tensor(es_attr)
    data["station", "near", "station"].edge_index = torch.tensor(ss_index, dtype=torch.long)
    data["station", "near", "station"].edge_attr = torch.tensor(ss_attr)
    data["event", "precedes", "event"].edge_index = torch.tensor(ee_index, dtype=torch.long)
    data["event", "precedes", "event"].edge_attr = torch.tensor(ee_attr)

    # ---- tabular copy for the ML baselines (same features, same split)
    table = events[["id", "time"] + EVENT_FEATURES + ["mag", "depth_class"]].copy()
    table["split"] = split
    return data, table


if __name__ == "__main__":
    GRAPHS.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(PROC / "events.csv")
    stations = pd.read_csv(PROC / "stations.csv")
    data, table = build_graph(events, stations)

    torch.save(data, GRAPHS / "seismo_graph.pt")
    table.to_csv(PROC / "features.csv", index=False)

    print(data)
    print("\nSplit sizes:", table["split"].value_counts().to_dict())
    for split in ["train", "val", "test"]:
        s = table[table["split"] == split]
        print(f"  {split:5s}: {str(s['time'].min())[:10]} -> {str(s['time'].max())[:10]} | "              f"mean mag {s['mag'].mean():.2f}")
    print("\nSaved graph -> data/graphs/seismo_graph.pt")
    print("Saved baseline table -> data/processed/features.csv")