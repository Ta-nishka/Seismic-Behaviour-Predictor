# Seismic Behaviour Predictor

Graph neural network analysis of Southern California seismicity, with an interactive dashboard.

The project models the region as a **heterogeneous graph** of earthquakes and seismic stations, trains a GraphSAGE model on it, and compares it with classical machine learning. The dashboard lets you explore every event, every graph edge and every model result in the browser.

**Live demo:** [Open the dashboard](https://seismic-behaviour.streamlit.app/)

> **Scope.** This is retrospective analysis and statistical estimation. It does not forecast when or where a specific earthquake will occur, and it should not be used for safety decisions.

## Key findings

- The GNN gives the clearest gain on **depth class**, reaching about 0.57 macro-F1 against 0.54 for a random forest and 0.12 for the majority baseline.
- **Magnitude is only weakly predictable** from the available inputs. The best R² is about 0.19, and the GNN's MAE edge over the random forest is small (0.283 vs 0.291).
- **Station nodes help.** Removing them worsens both magnitude and depth results.
- **Temporal edges did not help** on this data. See the results below and the limitations.

## What it does

| Section        | What you can do                                                                                                                                                                                       |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Catalog**    | Map of all 7,639 events coloured by magnitude type, depth, period or cluster; animate activity by year; magnitude and depth over time; Gutenberg-Richter fit; click years to cross-filter the whole app |
| **Graph**      | Draw the full graph (stations, events, and all three edge types) on a map; hub stations; edge-length and space-time clustering plots; spotlight any station's neighbourhood                           |
| **Embeddings** | 2D projection of the GNN's event embeddings, clusters on the map, cluster profiles, GNN vs raw-feature clustering                                                                                     |
| **Stations**   | Station groups learned from embeddings and the most similar distant station pairs                                                                                                                     |
| **Models**     | GNN, two ablations and four classical baselines on a held-out future period, with predicted-vs-actual, confusion matrix and error over time                                                           |
| **Predict**    | (1) Estimate magnitude and depth class for a described event with an 80% range. (2) Regional activity outlook with a back-test on the held-out period                                                 |

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

```
git clone https://github.com/Ta-nishka/Seismic-Behaviour-Predictor.git
cd Seismic-Behaviour-Predictor
pip install -r requirements.txt
streamlit run app.py
```

All sections work from the committed `data/processed/` and `results/` files, so no training is needed to explore the dashboard.

## Re-create everything from scratch

```
pip install -r requirements-pipeline.txt     # includes PyTorch and PyTorch Geometric
python src/data_loader.py                    # USGS events + IRIS stations -> data/processed
python src/graph_builder.py                  # build the heterogeneous graph + feature table
python src/train.py                          # baselines + GNN + ablations -> results/
python src/analysis.py                       # clusters and station relationships -> results/
streamlit run app.py
```

Quick test of the training step: `python src/train.py --seeds 1 --epochs 50`.

## Results

Evaluated on the held-out final 15% of events (chronological split), mean ± standard deviation over several seeds. For magnitude, lower MAE/RMSE and higher R² are better. For depth class, higher accuracy and macro-F1 are better. Full output is in `results/metrics_summary.csv`.

| Model                      | Mag MAE   | Mag RMSE  | Mag R²    | Depth Acc | Depth macro-F1 |
|----------------------------|-----------|-----------|-----------|-----------|----------------|
| Mean / majority baseline   | 0.348     | 0.450     | -0.003    | 0.223     | 0.122          |
| Linear / logistic          | 0.312     | 0.428     | 0.092     | 0.415     | 0.385          |
| Random forest              | 0.291     | 0.407     | 0.180     | 0.561     | 0.538          |
| MLP                        | 0.307     | 0.433     | 0.073     | 0.516     | 0.487          |
| **GNN (full)**             | **0.283** | 0.411     | 0.161     | 0.608     | 0.572          |
| GNN without temporal edges | **0.283** | **0.405** | **0.188** | **0.612** | **0.575**      |
| GNN without station nodes  | 0.294     | 0.422     | 0.117     | 0.590     | 0.541          |

### Takeaways

- **Depth class is where the graph helps most.** The GNN reaches about 0.57 macro-F1 against 0.54 for the random forest and 0.12 for the majority baseline.
- **Magnitude is only weakly predictable.** The best R² is about 0.19, so even the best models explain under a fifth of the variance. The GNN's MAE edge over the random forest is small (0.283 vs 0.291).
- **Station nodes help.** Removing them raises MAE from 0.283 to 0.294 and lowers depth macro-F1 from 0.572 to 0.541.
- **Temporal edges did not help here.** The model without them scores the same or slightly better, though the gap is within seed-to-seed noise. This may reflect the 2019 Ridgecrest sequence sitting in the training period (see Limitations).

## Limitations

- The public catalog does not say which stations detected which event, so event-station links are a modelling assumption.
- Magnitude is only weakly predictable from the available inputs; the dashboard reports this directly.
- The 2019 Ridgecrest sequence falls in the training period, so test results reflect distribution shift.
- The Predict section uses a random forest and a Poisson base rate. The base rate ignores aftershock clustering, which the back-test shows.
- Standard deviations are reported over a small number of seeds, so small differences between models should not be over-interpreted.

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
