"""
SeismoNet - models.py
 1) classical ML baselines (scikit-learn), trained on the same features + split
 2) SeismoGNN: heterogeneous GraphSAGE with two heads
      - magnitude regression
      - depth-class classification (shallow / mid / deep)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import HeteroConv, SAGEConv


# ======================================================== ML baselines
def baseline_models(seed: int = 0) -> dict:
    """name -> (regressor for magnitude, classifier for depth class)"""
    return {
        "Mean/Majority baseline": (DummyRegressor(), DummyClassifier(strategy="most_frequent")),
        "Linear / Logistic": (
            make_pipeline(StandardScaler(), LinearRegression()),
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        ),
        "MLP": (
            make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                                         early_stopping=True, random_state=seed)),
            make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                                          early_stopping=True, random_state=seed)),
        ),
    }


# ======================================================== GNN
ET_RECORDED = ("event", "recorded_by", "station")
ET_REVERSE = ("station", "rev_recorded_by", "event")
ET_STATION = ("station", "near", "station")
ET_TEMPORAL = ("event", "precedes", "event")
ALL_EDGE_TYPES = [ET_RECORDED, ET_REVERSE, ET_STATION, ET_TEMPORAL]


class SeismoGNN(nn.Module):
    def __init__(self, n_event_feats: int, n_station_feats: int, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.2, edge_types=ALL_EDGE_TYPES):
        super().__init__()
        self.edge_types = list(edge_types)
        self.dropout = dropout
        self.lin_event = nn.Linear(n_event_feats, hidden)
        self.lin_station = nn.Linear(n_station_feats, hidden)
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in self.edge_types}, aggr="sum")
            for _ in range(layers)
        ])
        self.mag_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.depth_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, data):
        x = {"event": F.relu(self.lin_event(data["event"].x)),
             "station": F.relu(self.lin_station(data["station"].x))}
        edge_index = {et: data[et].edge_index for et in self.edge_types}
        for conv in self.convs:
            out = conv(x, edge_index)
            # node types that received no messages keep their previous embedding
            x = {k: F.dropout(F.relu(out[k]), self.dropout, self.training) + x[k] if k in out else x[k]
                 for k in x}
        h = x["event"]
        return self.mag_head(h).squeeze(-1), self.depth_head(h), x