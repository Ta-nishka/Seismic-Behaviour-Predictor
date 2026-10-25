"""
SeismoNet - train.py
Trains ML baselines and GNN variants (full model + two ablations) over several
seeds, evaluates on the chronological TEST split, and writes:
    results/metrics_all.csv        one row per (model, seed)
    results/metrics_summary.csv    mean ± std per model
    results/model_comparison.png   bar charts
    results/event_embeddings.npy   GNN event embeddings   (for clustering / dashboard)
    results/station_embeddings.npy GNN station embeddings (for station relationship maps)
    results/gnn_predictions.csv    GNN predictions for every event

Run from the gml_mp root:   python src/train.py
Quick test:                  python src/train.py --seeds 1 --epochs 50
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)

from graph_builder import DEPTH_MAP, EVENT_FEATURES
from models import ALL_EDGE_TYPES, ET_RECORDED, ET_REVERSE, ET_STATION, ET_TEMPORAL, SeismoGNN, baseline_models

PROC, GRAPHS, RESULTS = Path("data/processed"), Path("data/graphs"), Path("results")
INV_DEPTH = {v: k for k, v in DEPTH_MAP.items()}


# ------------------------------------------------------------ metrics
def reg_metrics(y, p):
    return {"mag_MAE": mean_absolute_error(y, p),
            "mag_RMSE": mean_squared_error(y, p) ** 0.5,
            "mag_R2": r2_score(y, p)}


def cls_metrics(y, p):
    return {"depth_Acc": accuracy_score(y, p),
            "depth_MacroF1": f1_score(y, p, average="macro")}


# ------------------------------------------------------------ baselines
def run_baselines(table: pd.DataFrame, seeds: list[int]) -> list[dict]:
    tr, te = table[table["split"] == "train"], table[table["split"] == "test"]
    Xtr, Xte = tr[EVENT_FEATURES].to_numpy(), te[EVENT_FEATURES].to_numpy()
    ymag_tr, ymag_te = tr["mag"].to_numpy(), te["mag"].to_numpy()
    ydep_tr = tr["depth_class"].map(DEPTH_MAP).to_numpy()
    ydep_te = te["depth_class"].map(DEPTH_MAP).to_numpy()

    rows = []
    for seed in seeds:
        for name, (reg, clf) in baseline_models(seed).items():
            reg.fit(Xtr, ymag_tr)
            clf.fit(Xtr, ydep_tr)
            row = {"model": name, "seed": seed,
                   **reg_metrics(ymag_te, reg.predict(Xte)),
                   **cls_metrics(ydep_te, clf.predict(Xte))}
            rows.append(row)
            print(f"[baseline] {name:24s} seed {seed}  MAE {row['mag_MAE']:.3f}  F1 {row['depth_MacroF1']:.3f}")
    return rows


# ------------------------------------------------------------ GNN
def run_gnn(data, name, edge_types, seed, epochs, device, patience=40, lr=5e-3, save_extras=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ev = data["event"]
    tr_mask, va_mask, te_mask = ev.train_mask, ev.val_mask, ev.test_mask
    mu, sd = ev.y_mag[tr_mask].mean(), ev.y_mag[tr_mask].std()

    # Temporal protection: while training, the model only sees train events;
    # for validation it sees train+val events; for testing the full graph.
    g_train = data.subgraph({"event": tr_mask})
    g_val = data.subgraph({"event": tr_mask | va_mask})

    model = SeismoGNN(ev.x.size(1), data["station"].x.size(1), edge_types=edge_types).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def loss_fn(mag, logit, y_mag, y_dep):
        return F.smooth_l1_loss(mag, (y_mag - mu) / sd) + F.cross_entropy(logit, y_dep)

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        mag, logit, _ = model(g_train)
        loss = loss_fn(mag, logit, g_train["event"].y_mag, g_train["event"].y_depth)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            mag, logit, _ = model(g_val)
            m = g_val["event"].val_mask
            vloss = loss_fn(mag[m], logit[m], g_val["event"].y_mag[m], g_val["event"].y_depth[m]).item()
        if vloss < best - 1e-4:
            best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mag, logit, h = model(data)
    pred_mag = (mag * sd + mu).cpu().numpy()
    pred_dep = logit.argmax(1).cpu().numpy()
    m = te_mask.cpu().numpy()
    y_mag, y_dep = ev.y_mag.cpu().numpy(), ev.y_depth.cpu().numpy()

    row = {"model": name, "seed": seed,
           **reg_metrics(y_mag[m], pred_mag[m]),
           **cls_metrics(y_dep[m], pred_dep[m])}
    print(f"[gnn] {name:26s} seed {seed}  epochs {epoch + 1:3d}  "
          f"MAE {row['mag_MAE']:.3f}  F1 {row['depth_MacroF1']:.3f}")

    extras = None
    if save_extras:
        extras = dict(event_emb=h["event"].cpu().numpy(), station_emb=h["station"].cpu().numpy(),
                      pred_mag=pred_mag, pred_dep=pred_dep)
    return row, extras


# ------------------------------------------------------------ main
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ("model", "seed")]
    g = df.groupby("model", sort=False)[cols]
    mean, std = g.mean(), g.std().fillna(0)
    return mean.round(3).astype(str) + " ± " + std.round(3).astype(str), mean, std


def plot(mean, std, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, col, title in [(ax[0], "mag_MAE", "Magnitude MAE (lower is better)"),
                          (ax[1], "depth_MacroF1", "Depth-class macro-F1 (higher is better)")]:
        a.barh(mean.index, mean[col], xerr=std[col], color="#4C72B0")
        a.set_title(title)
        a.invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    RESULTS.mkdir(exist_ok=True)

    table = pd.read_csv(PROC / "features.csv")
    rows = run_baselines(table, seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(GRAPHS / "seismo_graph.pt", weights_only=False).to(device)

    configs = {
        "GNN (full)": ALL_EDGE_TYPES,
        "GNN w/o temporal edges": [ET_RECORDED, ET_REVERSE, ET_STATION],
        "GNN w/o station nodes": [ET_TEMPORAL],
    }
    extras = None
    for name, edge_types in configs.items():
        for seed in seeds:
            keep = name == "GNN (full)" and seed == seeds[0]
            row, ex = run_gnn(data, name, edge_types, seed, args.epochs, device, save_extras=keep)
            rows.append(row)
            if keep:
                extras = ex

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "metrics_all.csv", index=False)
    pretty, mean, std = summarize(df)
    pretty.to_csv(RESULTS / "metrics_summary.csv")
    plot(mean, std, RESULTS / "model_comparison.png")

    np.save(RESULTS / "event_embeddings.npy", extras["event_emb"])
    np.save(RESULTS / "station_embeddings.npy", extras["station_emb"])
    preds = table[["id", "split", "mag", "depth_class"]].copy()
    preds["mag_pred"] = extras["pred_mag"]
    preds["depth_pred"] = [INV_DEPTH[i] for i in extras["pred_dep"]]
    preds.to_csv(RESULTS / "gnn_predictions.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== TEST-SET RESULTS (mean ± std over seeds) ===")
    print(pretty.to_string())
    print("\nSaved to results/ : metrics_summary.csv, model_comparison.png, embeddings, predictions")


if __name__ == "__main__":
    main()