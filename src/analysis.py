"""
SeismoNet - analysis.py
Event clusters + station relationship maps from the trained GNN embeddings.

Needs (created earlier): results/event_embeddings.npy, results/station_embeddings.npy,
                         data/processed/features.csv, events.csv, stations.csv
Creates: results/figures/*.png and results/*.csv (the dashboard will read these)

Run from the gml_mp root:   python src/analysis.py          (k chosen automatically)
                            python src/analysis.py --k 6    (force k)
Optional nicer 2D plot:     pip install umap-learn          (falls back to t-SNE)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

PROC, RESULTS = Path("data/processed"), Path("results")
FIG = RESULTS / "figures"


def to_xy_km(lat, lon, lat0=34.5, lon0=-117.5):
    return (np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * 111.0


def best_k(X, ks=range(4, 11)):
    scores = {}
    for k in ks:
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
        scores[k] = silhouette_score(X, lab, sample_size=3000, random_state=0)
    return max(scores, key=scores.get), scores


def eta_squared(y, labels):
    """Share of magnitude variance explained by cluster membership."""
    gm = y.mean()
    ss_b = sum((labels == c).sum() * (y[labels == c].mean() - gm) ** 2 for c in np.unique(labels))
    return ss_b / ((y - gm) ** 2).sum()


def evaluate(name, X, labels, df, test):
    return {"space": name,
            "silhouette (own space)": silhouette_score(X, labels, sample_size=3000, random_state=0),
            "davies_bouldin (own space)": davies_bouldin_score(X, labels),
            "NMI vs depth class (test)": normalized_mutual_info_score(df.loc[test, "depth_class"], labels[test]),
            "magnitude eta^2 (test)": eta_squared(df.loc[test, "mag"].to_numpy(), labels[test])}


def project_2d(emb):
    try:
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(emb), "UMAP"
    except ImportError:
        return TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb), "t-SNE"


# ----------------------------------------------------------------------------
def event_clusters(df, emb, stations, k_arg):
    feat_cols = [c for c in df.columns if c not in ("id", "time", "mag", "depth_class", "split", "depth", "place", "year")]
    k, scores = (k_arg, None) if k_arg else best_k(emb)
    print(f"Event clusters: k = {k}" + ("" if k_arg else f"  (silhouette by k: { {a: round(b, 3) for a, b in scores.items()} })"))

    labels = KMeans(k, n_init=10, random_state=0).fit_predict(emb)
    df["cluster"] = labels
    test = (df["split"] == "test").to_numpy()

    spaces = {"GNN embeddings": (emb, labels)}
    ll = StandardScaler().fit_transform(df[["latitude", "longitude"]])
    spaces["Raw lat/lon"] = (ll, KMeans(k, n_init=10, random_state=0).fit_predict(ll))
    ff = StandardScaler().fit_transform(df[feat_cols])
    spaces["Raw tabular features"] = (ff, KMeans(k, n_init=10, random_state=0).fit_predict(ff))
    comp = pd.DataFrame([evaluate(n, X, lab, df, test) for n, (X, lab) in spaces.items()]).round(3)
    comp.to_csv(RESULTS / "cluster_comparison.csv", index=False)

    prof = df.groupby("cluster").agg(n=("id", "size"), mean_mag=("mag", "mean"), max_mag=("mag", "max"),
                                     mean_depth_km=("depth", "mean"), center_lat=("latitude", "mean"),
                                     center_lon=("longitude", "mean"), median_year=("year", "median")).round(2)
    prof["dominant_depth_class"] = df.groupby("cluster")["depth_class"].agg(lambda s: s.value_counts().idxmax())
    prof["top_place"] = df.groupby("cluster")["place"].agg(
        lambda s: s.dropna().str.split(" of ").str[-1].value_counts().idxmax())
    prof.to_csv(RESULTS / "cluster_profiles.csv")

    xy, method = project_2d(emb)
    df["proj_x"], df["proj_y"] = xy[:, 0], xy[:, 1]
    df[["id", "cluster", "proj_x", "proj_y"]].to_csv(RESULTS / "event_clusters.csv", index=False)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=4)
    ax[0].set_title(f"Event embeddings ({method}), colored by cluster")
    ax[1].scatter(stations["longitude"], stations["latitude"], marker="^", s=6, c="lightgray", label="stations")
    ax[1].scatter(df["longitude"], df["latitude"], c=labels, cmap="tab10", s=4)
    ax[1].set_title("Same clusters on the map"); ax[1].set_xlabel("longitude"); ax[1].set_ylabel("latitude")
    plt.tight_layout(); plt.savefig(FIG / "event_clusters.png", dpi=150); plt.close()

    print("\nCluster comparison (higher NMI / eta^2 = clusters explain more; silhouette is NOT comparable across spaces):")
    print(comp.to_string(index=False))
    print("\nCluster profiles:"); print(prof.to_string())


def station_relationships(stations, semb):
    n = len(stations)
    Z = StandardScaler().fit_transform(semb)               # centre so similarities are informative
    sim = cosine_similarity(Z)
    sx, sy = to_xy_km(stations["latitude"], stations["longitude"])
    dist = cdist(np.c_[sx, sy], np.c_[sx, sy])

    iu = np.triu_indices(n, 1)
    pick = np.random.default_rng(0).choice(len(iu[0]), size=min(100_000, len(iu[0])), replace=False)
    rho = spearmanr(sim[iu][pick], -dist[iu][pick])[0]
    print(f"\nStation embeddings: Spearman(similarity, closeness) = {rho:.2f}  "
          f"(1 = similarity is only geography; lower = captures more than proximity)")

    st_cluster = KMeans(5, n_init=10, random_state=0).fit_predict(Z)
    out = stations[["station_id", "latitude", "longitude"]].copy()
    out["cluster"] = st_cluster
    out.to_csv(RESULTS / "station_clusters.csv", index=False)

    # strongest relationships between stations that are far apart (> 50 km)
    far = dist[iu] > 50
    order = np.argsort(-np.where(far, sim[iu], -np.inf))[:150]
    pairs = pd.DataFrame({"station_a": stations["station_id"].to_numpy()[iu[0][order]],
                          "station_b": stations["station_id"].to_numpy()[iu[1][order]],
                          "cosine_sim": sim[iu][order].round(3), "dist_km": dist[iu][order].round(1)})
    pairs.to_csv(RESULTS / "station_similar_pairs.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    for a, b in zip(iu[0][order], iu[1][order]):
        ax.plot([stations["longitude"][a], stations["longitude"][b]],
                [stations["latitude"][a], stations["latitude"][b]], c="gray", lw=0.4, alpha=0.5)
    sc = ax.scatter(stations["longitude"], stations["latitude"], c=st_cluster, cmap="tab10", s=18)
    ax.set_title("Station relationship map\n(color = station group, lines = most similar pairs > 50 km apart)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    plt.tight_layout(); plt.savefig(FIG / "station_relationship_map.png", dpi=150); plt.close()

    order_idx = np.argsort(st_cluster, kind="stable")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim[np.ix_(order_idx, order_idx)], cmap="viridis", vmin=np.percentile(sim, 5), vmax=1)
    ax.set_title("Station similarity (sorted by group)"); plt.colorbar(im)
    plt.tight_layout(); plt.savefig(FIG / "station_similarity_heatmap.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0, help="number of event clusters (0 = choose automatically)")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(PROC / "features.csv")
    events = pd.read_csv(PROC / "events.csv")
    stations = pd.read_csv(PROC / "stations.csv")
    df = table.merge(events[["id", "depth", "place"]], on="id", how="left")
    df["year"] = pd.to_datetime(df["time"], utc=True, format="ISO8601").dt.year
    emb = np.load(RESULTS / "event_embeddings.npy")
    semb = np.load(RESULTS / "station_embeddings.npy")
    assert len(df) == len(emb) and len(stations) == len(semb), "embeddings do not match data - rerun train.py"

    event_clusters(df, emb, stations, args.k)
    station_relationships(stations, semb)
    print("\nSaved figures to results/figures/ and tables to results/")


if __name__ == "__main__":
    main()