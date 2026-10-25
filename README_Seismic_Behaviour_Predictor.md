# Seismic Behaviour Predictor

**Graph Neural Network Analysis of Southern California Seismicity**

An interactive, graph-based machine learning project for exploring earthquake patterns, analysing relationships between seismic events and stations, and evaluating earthquake-related prediction tasks using GraphSAGE and classical machine learning models.

[**Live Dashboard**](https://seismic-behaviour.streamlit.app/) · [**GitHub Repository**](https://github.com/Ta-nishka/Seismic-Behaviour-Predictor)

> **Important:** This project focuses on retrospective seismic analysis and statistical estimation. It does not predict the exact time, location, or occurrence of future earthquakes and must not be used as an earthquake early-warning system or for safety-critical decisions.

---

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Key Features](#key-features)
4. [System Architecture](#system-architecture)
5. [Heterogeneous Graph Design](#heterogeneous-graph-design)
6. [Machine Learning Models](#machine-learning-models)
7. [Data and Preprocessing](#data-and-preprocessing)
8. [Experimental Setup and Evaluation](#experimental-setup-and-evaluation)
9. [Results](#results)
10. [Installation and Quick Start](#installation-and-quick-start)
11. [Reproducing the Experiments](#reproducing-the-experiments)
12. [Repository Structure](#repository-structure)
13. [Limitations](#limitations)
14. [Future Improvements](#future-improvements)
15. [Data Sources](#data-sources)
16. [License](#license)

---

## Overview

Earthquake events are not independent observations. Their spatial and temporal relationships, along with the surrounding seismic monitoring infrastructure, can provide useful context for studying regional seismic behaviour.

**Seismic Behaviour Predictor** represents Southern California seismicity as a heterogeneous graph containing earthquake events and seismic stations. A GraphSAGE-based graph neural network learns representations by aggregating information from connected nodes. Its performance is compared with conventional machine learning models and graph ablation experiments.

The project combines graph construction, model training, quantitative evaluation, clustering analysis, and an interactive Streamlit dashboard in one workflow.

The processed earthquake catalog contains **7,639 events**.

### Problem Statement

Traditional machine learning models primarily learn from the feature vector of an individual observation. However, seismic data also contains relationships between events and monitoring stations, as well as spatial and temporal patterns.

This project investigates whether explicitly modelling these relationships through a heterogeneous graph provides useful representations for analysing seismic behaviour and estimating selected earthquake characteristics.

### Objectives

- Analyse earthquake magnitude, depth, temporal distribution, and spatial patterns.
- Construct a heterogeneous graph of earthquake events and seismic stations.
- Train and evaluate a GraphSAGE model against classical machine learning baselines.
- Investigate the contribution of temporal and station relationships through ablation studies.
- Explore learned graph embeddings, event clusters, and station relationships.
- Present model outputs and exploratory analyses through an interactive dashboard.

---

## Key Features

| Dashboard section | Functionality |
|---|---|
| **Catalog** | Explore earthquake events by magnitude, depth, time period, and cluster; inspect temporal trends and Gutenberg–Richter relationships. |
| **Graph** | Visualise earthquake events, seismic stations, graph connections, station neighbourhoods, and spatial-temporal relationships. |
| **Embeddings** | Explore low-dimensional representations learned by the GNN, visualise clusters, and compare learned representations with raw features. |
| **Stations** | Investigate station relationships, embedding-based station groups, and similar station pairs. |
| **Models** | Compare machine learning baselines, the full GNN, and ablation models using evaluation metrics and diagnostic plots. |
| **Predict** | Explore statistical estimates of magnitude and depth class, including uncertainty ranges, and examine a regional activity outlook with historical back-testing. |

Additional dashboard capabilities include interactive filtering, cross-filtered visualisations, and CSV downloads of filtered event data.

---

## System Architecture

The project follows a data-processing, graph-learning, analysis, and visualisation pipeline.

```text
USGS Earthquake Catalog + IRIS Station Data
                    |
                    v
           Data Collection
                    |
                    v
       Cleaning and Preprocessing
                    |
                    v
       Heterogeneous Graph Builder
                    |
                    v
       Feature and Graph Generation
                    |
                    v
       Model Training and Evaluation
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Baselines  GraphSAGE  Ablations
          |         |         |
          +---------+---------+
                    |
                    v
        Metrics and Embeddings
                    |
                    v
       Clustering and Analysis
                    |
                    v
        Interactive Streamlit App
```

**Technology stack**

- **Language:** Python
- **Machine learning:** scikit-learn, PyTorch
- **Graph learning:** GraphSAGE and graph-based message passing
- **Data processing:** pandas, NumPy
- **Visualisation:** Plotly and other plotting libraries used by the project
- **Dashboard:** Streamlit
- **Testing:** pytest
- **Data sources:** USGS ComCat and IRIS station services

---

## Heterogeneous Graph Design

The graph represents two node types:

- **Event nodes:** earthquake observations.
- **Station nodes:** seismic monitoring stations.

Four directed relationship types are used to represent spatial and temporal context.

| Relationship | Direction | Construction |
|---|---|---|
| `recorded_by` | Event → Station | Connects each event to its five nearest stations. |
| `rev_recorded_by` | Station → Event | Reverse relationship that supports information flow in the opposite direction. |
| `near` | Station ↔ Station | Connects nearby stations using five nearest neighbours, with symmetric relationships. |
| `precedes` | Earlier Event → Later Event | Connects events within 50 km and seven days, directed forward in time. |

These relationships allow the GNN to aggregate information from different node types and neighbourhoods.

### Design decisions

- Earthquake magnitude and depth are prediction targets, not input features.
- Event–station relationships are based on geographical proximity rather than magnitude-based selection.
- Temporal event edges point from earlier to later events.
- A chronological split is used to evaluate performance on later observations.
- Feature scaling is fitted using the training data only.
- Validation and test procedures follow the graph visibility rules implemented in the training pipeline.
- Ablation experiments examine the effect of removing temporal edges or station relationships.

The station links are modelling assumptions based on proximity. They do not represent verified records of which stations actually detected individual earthquakes.

---

## Machine Learning Models

The project compares conventional machine learning methods with graph-based learning.

| Model | Purpose |
|---|---|
| Mean / majority baseline | Establishes a simple reference for regression and classification tasks. |
| Linear / logistic model | Provides a simple, interpretable machine learning baseline. |
| Random Forest | Captures nonlinear relationships between input features. |
| Multilayer Perceptron (MLP) | Learns nonlinear patterns from feature vectors. |
| GraphSAGE — full model | Learns node representations by aggregating neighbourhood information. |
| GraphSAGE without temporal edges | Evaluates the contribution of temporal event relationships. |
| GraphSAGE without station relationships | Evaluates the contribution of station-related graph information. |

The final comparison should use the models and metrics actually produced by the current training code.

### Prediction tasks

**1. Magnitude estimation**

A regression task evaluated using error-based metrics.

**2. Depth classification**

A classification task evaluated using accuracy and macro-F1.

**3. Graph representation learning**

Node embeddings are used for exploratory clustering and the analysis of event and station relationships.

**4. Regional activity analysis**

The dashboard includes a statistical activity outlook and historical back-testing. These results should be interpreted as exploratory estimates, not reliable forecasts of individual earthquake events.

---

## Data and Preprocessing

### Data sources

The project uses publicly available earthquake catalog data and seismic station metadata.

- Earthquake events are obtained from the USGS ComCat catalog.
- Seismic station information is obtained from the IRIS station service for the Southern California seismic network.

### Preprocessing workflow

1. Retrieve earthquake catalog and station information.
2. Clean and prepare the event and station tables.
3. Process temporal and geographical attributes.
4. Construct event–station and station–station relationships.
5. Generate temporal relationships between qualifying earthquake events.
6. Prepare model features and prediction targets.
7. Apply chronological train, validation, and test splits.
8. Fit feature transformations using training data only.

The processed event and station data are stored under `data/processed/`.

---

## Experimental Setup and Evaluation

### Data split

The chronological dataset split is:

| Subset | Proportion | Purpose |
|---|---:|---|
| Training | 70% | Fit model parameters. |
| Validation | 15% | Monitor performance and support model selection or early stopping. |
| Testing | 15% | Evaluate performance on a held-out future period. |

The model training pipeline uses multiple random seeds to examine variation in performance. Early stopping is based on validation performance, subject to the configuration in the training script.

### Evaluation metrics

| Task | Metrics | Interpretation |
|---|---|---|
| Magnitude regression | MAE, RMSE, R² | Measures prediction error and variance explained. |
| Depth classification | Accuracy, macro-F1 | Measures overall correctness and class-balanced performance. |
| Clustering | Applicable clustering metrics and cluster profiles | Examines the structure and interpretability of learned representations. |

**Metric interpretation**

- **MAE:** Average absolute difference between predicted and actual values.
- **RMSE:** Penalises larger prediction errors more heavily.
- **R²:** Measures how much target variance is explained relative to a mean-prediction baseline.
- **Macro-F1:** Computes F1 for each class and averages the class scores equally.

Model performance should be assessed against simple baselines, not just against other complex models.

---

## Results

Final numerical results should be generated by running the training pipeline. They are intentionally not included here until the corresponding output files have been produced and verified.

The training script is expected to write model metrics and comparison outputs under `results/`.

### Model comparison

| Model | Magnitude MAE ↓ | Magnitude RMSE ↓ | Magnitude R² ↑ | Depth Accuracy ↑ | Depth Macro-F1 ↑ |
|---|---:|---:|---:|---:|---:|
| Mean / majority baseline | Pending | Pending | Pending | Pending | Pending |
| Linear / logistic model | Pending | Pending | Pending | Pending | Pending |
| Random Forest | Pending | Pending | Pending | Pending | Pending |
| MLP | Pending | Pending | Pending | Pending | Pending |
| GraphSAGE — full model | Pending | Pending | Pending | Pending | Pending |
| GraphSAGE without temporal edges | Pending | Pending | Pending | Pending | Pending |
| GraphSAGE without station relationships | Pending | Pending | Pending | Pending | Pending |

*Arrows indicate the preferred direction for each metric. Replace pending entries with actual values from the generated results, and use N/A where a metric does not apply to a model.*

### Generated outputs

The training and analysis pipeline can generate:

- `metrics_all.csv` — per-run model metrics.
- `metrics_summary.csv` — aggregated model comparison metrics.
- `model_comparison.png` — model comparison visualisation.
- `event_embeddings.npy` — learned event embeddings.
- `station_embeddings.npy` — learned station embeddings.
- `gnn_predictions.csv` — GNN prediction outputs.
- Cluster assignments, cluster profiles, station relationship outputs, and related figures.

These outputs are created by the relevant scripts; they may not be present in a fresh checkout until the pipeline has been run.

---

## Installation and Quick Start

### Prerequisites

- Python installed on your system.
- Git.
- A compatible environment for the project's pinned Python dependencies.

### Run the dashboard

Clone the repository using the correct URL:

```bash
git clone https://github.com/Ta-nishka/Seismic-Behaviour-Predictor.git
cd Seismic-Behaviour-Predictor
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the dashboard dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

**Note:** Dashboard sections that depend on generated model results or embeddings may require the training and analysis pipeline to be run first.

---

## Reproducing the Experiments

To rebuild the processed data, train the models, and generate analysis outputs, install the pipeline dependencies:

```bash
pip install -r requirements-pipeline.txt
```

Run the scripts in the following order:

```bash
python src/data_loader.py
python src/graph_builder.py
python src/train.py
python src/analysis.py
```

Then launch the dashboard:

```bash
streamlit run app.py
```

For a shorter training run, the project supports the following command:

```bash
python src/train.py --seeds 1 --epochs 50
```

This reduced run is intended for a quick experiment or smoke test, not as a substitute for the full evaluation.

Exact results can depend on the dataset retrieved, dependency versions, random seeds, and configuration. For reproducible comparisons, retain the generated metrics and document the environment and training settings.

---

## Repository Structure

```text
Seismic-Behaviour-Predictor/
│
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-pipeline.txt
│
├── .streamlit/
│   └── Streamlit configuration
│
├── data/
│   └── processed/
│       └── Processed earthquake and station data
│
├── src/
│   ├── data_loader.py
│   ├── graph_builder.py
│   ├── models.py
│   ├── train.py
│   └── analysis.py
│
├── tests/
│   └── Project tests
│
└── results/
    └── Generated metrics, predictions, embeddings, and figures
```

The `results/` directory is generated by the training and analysis workflow and may not exist in a fresh checkout. The descriptions above summarise the main project components; refer to the repository for the exact current file listing.

---

## Limitations

This project has several important limitations.

1. **Proximity-based station relationships:** The public catalog does not provide verified station-detection links for every event. The event–station edges are geographical modelling assumptions.
2. **Magnitude estimation:** Magnitude may be weakly predictable from the available input features. Performance should be judged against simple baselines.
3. **Temporal distribution shift:** The 2019 Ridgecrest earthquake sequence falls within the training period, so the held-out test period does not provide a complete assessment of generalisation to all unusual seismic sequences.
4. **Regional activity outlook:** The Poisson-based baseline does not explicitly model aftershock clustering and may not capture complex seismic dependencies.
5. **Graph assumptions:** The selected distance and time thresholds influence which event relationships are included.
6. **Retrospective evaluation:** Historical model performance does not establish that the system can reliably predict future earthquakes.
7. **Data and environment dependence:** Reproducing results requires compatible dependencies and access to the relevant source data.

These limitations should be considered when interpreting predictions, clusters, and model comparisons.

---

## Future Improvements

Potential extensions include:

- Compare alternative graph construction thresholds and neighbourhood sizes.
- Investigate temporal graph neural networks for evolving seismic relationships.
- Evaluate stronger temporal validation strategies and separate historical earthquake sequences.
- Add uncertainty calibration and more detailed error analysis.
- Compare graph embeddings against feature-only baselines under identical splits.
- Improve reproducibility with recorded dependency versions, experiment configurations, and automated tests.
- Expand the dashboard with clearer model diagnostics and explanations of prediction uncertainty.

---

## Data Sources

- [USGS ComCat Earthquake Catalog](https://earthquake.usgs.gov/data/comcat/) — earthquake event data.
- [IRIS Station Service](https://service.iris.edu/) — seismic station metadata and station information.

The data is used for academic and exploratory analysis. Refer to the original providers for data documentation and usage conditions.

---

## License

This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.
