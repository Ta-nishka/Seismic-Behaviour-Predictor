# Seismic Behaviour Predictor

Graph neural network analysis of Southern California seismicity, with an interactive dashboard.

The project models the region as a **heterogeneous graph** of earthquakes and seismic stations, trains a GraphSAGE model on it, and compares it with classical machine learning. The dashboard lets you explore every event, every graph edge and every model result in the browser.

**Live demo:** [Open the dashboard](https://seismic-behaviour.streamlit.app/)
![Dashboard overview](docs/screenshots/catalog.png)

> **Scope.** This is retrospective analysis and statistical estimation. It does not forecast when or where a specific earthquake will occur, and it should not be used for safety decisions.

## What it does

| Section | What you can do |
|---|---|
| **Catalog** | Map of all 7,639 events coloured by magnitude type, depth, period or cluster; play activity by year; magnitude and depth over time; Gutenberg-Richter fit; click years to cross-filter the whole app |
| **Graph** | Draw the full graph (stations, events, and all three edge types) on a map; hub stations; edge-length and space-time clustering plots; spotlight any station's neighbourhood |
| **Embeddings** | 2D projection of the GNN's event embeddings, clusters on the map, cluster profiles, GNN vs raw-feature clustering |
| **Stations** | Station groups learned from embeddings and the most similar distant station pairs |
| **Models** | GNN, two ablations and four classical baselines on a held-out future period, with predicted-vs-actual, confusion matrix and error over time |
| **Predict** | (1) Estimate magnitude and depth class for a described event with an 80% range. (2) Regional activity outlook with a back-test on the held-out period |

Every chart has a plain-language insight underneath that updates with the filters. Filtered events can be downloaded as CSV.

## The graph

```
Event  --recorded_by-->  Station        5 nearest stations per event
Station --rev_recorded_by--> Event      reverse edge so information flows both ways
Station --near-->  Station              5 nearest neighbours, symmetric
Event  --precedes--> Event              earlier to later, within 50 km and 7 days
```

Design choices that keep the evaluation honest:

- Magnitude and depth are **never** model inputs; they are the prediction targets.
- Event-station edges use nearest stations, not a magnitude-based radius, so graph structure does not leak magnitude.
- Event-event edges only point forward in time.
- Chronological split (70% train, 15% validation, 15% test) and feature scaling from the training set only.
- While training, the model only sees training events; validation sees train + validation; testing sees the full graph.
- Results are averaged over several seeds, with ablations that remove temporal edges or station nodes.

## Quick start (dashboard only)

```bash
git clone https://github.com/YOUR-USERNAME/seismic-behaviour-predictor.git
cd seismic-behaviour-predictor
pip install -r requirements.txt
streamlit run app.py
```

The Catalog, Graph and Predict sections work from the committed `data/processed/` files alone. Embeddings, Stations and Models appear once `results/` is populated (next section).

## Re-create everything from scratch

```bash
pip install -r requirements-pipeline.txt     # includes PyTorch and PyTorch Geometric
python src/data_loader.py                    # USGS events + IRIS stations -> data/processed
python src/graph_builder.py                  # build the heterogeneous graph + feature table
python src/train.py                          # baselines + GNN + ablations -> results/
python src/analysis.py                       # clusters and station relationships -> results/
streamlit run app.py
```

Quick test of the training step: `python src/train.py --seeds 1 --epochs 50`.

## Results

_Paste the table from `results/metrics_summary.csv` here after running `train.py`._

| Model | Magnitude MAE | Depth macro-F1 |
|---|---|---|
| Mean / majority baseline | | |
| Random forest | | |
| MLP | | |
| GNN (full) | | |
| GNN without temporal edges | | |
| GNN without station nodes | | |

## Limitations

- The public catalog does not say which stations detected which event, so event-station links are a modelling assumption.
- Magnitude is only weakly predictable from the available inputs; the dashboard reports this directly.
- The 2019 Ridgecrest sequence falls in the training period, so test results reflect distribution shift.
- The Predict section uses a random forest and a Poisson base rate. The base rate ignores aftershock clustering, which the back-test shows.

## Repository layout

```
app.py                  Streamlit dashboard
src/                    data download, graph construction, models, training, analysis
data/processed/         cleaned events, stations and the feature table
results/                model outputs read by the dashboard
tests/                  smoke tests (pytest)
.github/workflows/      CI that runs the smoke tests
```

## Deploy

1. Push the repo to GitHub with `data/processed/` and `results/` committed.
2. On [Streamlit Community Cloud](https://streamlit.io/cloud), create an app from the repo with `app.py` as the main file.

## Data sources

- [USGS ComCat earthquake catalog](https://earthquake.usgs.gov/fdsnws/event/1/) (public domain)
- [IRIS station service](https://service.iris.edu/fdsnws/station/1/), network CI (Southern California Seismic Network)

## License

MIT, see [LICENSE](LICENSE).
