"""
Seismic Behaviour Predictor - interactive dashboard
Run from the project root:   streamlit run app.py

Layout: section navigation on top, filter popover with presets, chart cards with a live insight under every graph.
Every chart uses the FULL catalog (all events, all stations, every graph edge); filters only subset it.
Interactions: click/box-select years to cross-filter everything, play the map by year, spotlight any station.
Catalog + Graph need only data/processed/*.csv. Embeddings / Stations / Models light up after train.py + analysis.py.
"""
from pathlib import Path

from html import escape as esc

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

APP_NAME = "Seismic Behaviour Predictor"
PROC, RES = Path("data/processed"), Path("results")
st.set_page_config(page_title=APP_NAME, page_icon="🌋", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ theme
BG, PANEL, LINE, TXT, MUTED, ACC = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e", "#f59e0b"
DEPTH_ORDER = ["shallow", "mid", "deep"]
DEPTH_COLORS = {"shallow": "#2dd4bf", "mid": "#fbbf24", "deep": "#f87171"}
MAGTYPE_COLORS = {"ML": "#f59e0b", "MW": "#ef4444", "MD": "#5b8def", "MLR": "#2dd4bf", "MH": "#a78bfa", "MWR": "#f472b6"}
SPLIT_COLORS = {"train": "#5b8def", "val": "#fbbf24", "test": "#f87171"}
PALETTE = ["#5b8def", "#f59e0b", "#2dd4bf", "#f87171", "#a78bfa", "#84cc16", "#f472b6", "#22d3ee", "#fb923c", "#94a3b8"]
MAP_STYLE, CENTER, ZOOM = "carto-darkmatter", dict(lat=34.5, lon=-117.5), 5.1
EE_KM, EE_HOURS, EE_MAX, K_ES, K_SS = 50.0, 7 * 24.0, 10, 5, 5      # same rules as graph_builder.py
SECTIONS = ["Catalog", "Graph", "Embeddings", "Stations", "Models", "Predict", "About"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
.block-container {{ padding-top: 4.2rem; max-width: 1560px; }}
header[data-testid="stHeader"] {{ background: {BG}; border-bottom: 1px solid {LINE}; }}
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stToolbar"] {{ display: none; }}
[data-testid="stButtonGroup"] button {{ font-size: .95rem; font-weight: 500; padding: .45rem 1.1rem; min-height: 2.6rem; }}
[data-testid="stButtonGroup"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; padding: 3px; }}
.brand {{ font-size: 1.35rem; font-weight: 600; letter-spacing: -.01em; }}
.brand span {{ display: block; color: {MUTED}; font-weight: 400; font-size: .78rem; }}
.kpi {{ border-left: 3px solid {ACC}; padding: .15rem .9rem; }}
.kpi .v {{ font-size: 1.55rem; font-weight: 600; line-height: 1.25; }}
.kpi .l {{ color: {MUTED}; font-size: .78rem; }}
.kpi .s {{ color: {MUTED}; font-size: .7rem; }}
.lead {{ margin: 1.4rem 0 .6rem; }}
.lead h2 {{ font-size: 1.35rem; font-weight: 600; margin: 0; padding: 0; }}
.lead p {{ color: {MUTED}; margin: .2rem 0 0; max-width: 900px; font-size: .92rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {PANEL}; border-color: {LINE} !important; border-radius: 8px; }}
.ct {{ font-weight: 600; font-size: 1rem; }}
.cs {{ color: {MUTED}; font-size: .78rem; margin-bottom: .2rem; }}
.insight {{ border-left: 3px solid #5b8def; background: {BG}; padding: .55rem .8rem; margin-top: .3rem;
            color: #c9d1d9; font-size: .84rem; line-height: 1.5; border-radius: 0 4px 4px 0; }}
.insight b {{ color: {TXT}; }}
.chip {{ display: inline-block; background: {PANEL}; border: 1px solid {LINE}; border-radius: 999px; padding: 2px 12px;
         font-size: .78rem; color: {MUTED}; margin-right: 6px; }}
.chip.hot {{ border-color: {ACC}; color: {ACC}; }}
</style>""", unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{esc(str(sub))}</div></div>',
                 unsafe_allow_html=True)


def lead(title, text):
    st.markdown(f'<div class="lead"><h2>{title}</h2><p>{text}</p></div>', unsafe_allow_html=True)


def style(fig, h=420, legend=True, top=10):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans", color=TXT, size=12), showlegend=legend, margin=dict(l=8, r=8, t=top, b=8),
                      legend=dict(bgcolor="rgba(22,27,34,.8)", bordercolor=LINE, borderwidth=1))
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor=LINE)
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor=LINE)
    return fig


def card(title, fig, insight, sub="", key=None, **kw):
    """Chart card: title, chart, then a plain-language insight computed from the data on screen."""
    with st.container(border=True):
        st.markdown(f'<div class="ct">{title}</div>' + (f'<div class="cs">{sub}</div>' if sub else ""), unsafe_allow_html=True)
        out = st.plotly_chart(fig, width="stretch", key=key, **kw)
        st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)
    return out


def has(*names):
    return all((RES / n).exists() for n in names)


def need(*names):
    if has(*names):
        return True
    st.info("This section needs results from `train.py` / `analysis.py`. Missing: "
            + ", ".join(f"`results/{n}`" for n in names if not (RES / n).exists()))
    return False


# ------------------------------------------------------------------ data
for p in (PROC / "features.csv", PROC / "events.csv", PROC / "stations.csv"):
    if not p.exists():
        st.error(f"Missing `{p}`. Run data_loader.py and graph_builder.py first (from the project root).")
        st.stop()


@st.cache_data(show_spinner="Loading catalog...")
def load_data():
    f = pd.read_csv(PROC / "features.csv")
    ev = pd.read_csv(PROC / "events.csv")[["id", "depth", "magType", "place"]]
    df = f.merge(ev, on="id", how="left")
    if has("event_clusters.csv"):
        df = df.merge(pd.read_csv(RES / "event_clusters.csv"), on="id", how="left")
        df["cluster_str"] = "Cluster " + df["cluster"].fillna(-1).astype(int).astype(str)
    if has("gnn_predictions.csv"):
        df = df.merge(pd.read_csv(RES / "gnn_predictions.csv")[["id", "mag_pred", "depth_pred"]], on="id", how="left")
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = df.sort_values("time").reset_index(drop=True)          # row order == node index in the graph
    df["year"] = df["time"].dt.year
    df["magType"] = df["magType"].fillna("n/a").str.upper()
    df["place"] = df["place"].fillna("unknown")
    df["region"] = df["place"].str.split(" of ").str[-1]
    df["size_val"] = (df["mag"] - 2.0) ** 2
    return df


@st.cache_data
def load_stations():
    return pd.read_csv(PROC / "stations.csv")


def to_xy_km(lat, lon, lat0=34.5, lon0=-117.5):
    return (np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * 111.0


@st.cache_data(show_spinner="Building the full graph (all nodes and edges)...")
def build_edges(lat, lon, t_hours, slat, slon):
    """Re-creates every edge of graph_builder.py so the whole graph can be drawn."""
    ex, ey = to_xy_km(lat, lon)
    sx, sy = to_xy_km(slat, slon)
    tree = cKDTree(np.c_[sx, sy])
    d, idx = tree.query(np.c_[ex, ey], k=K_ES)
    es = dict(src=np.repeat(np.arange(len(lat)), K_ES), dst=idx.ravel(), dist=d.ravel())
    _, nn = tree.query(np.c_[sx, sy], k=K_SS + 1)
    a, b = np.repeat(np.arange(len(sx)), K_SS), nn[:, 1:].ravel()
    key = np.unique(np.sort(np.c_[a, b], axis=1), axis=0)                 # undirected, de-duplicated
    ss = dict(a=key[:, 0], b=key[:, 1], dist=np.hypot(sx[key[:, 0]] - sx[key[:, 1]], sy[key[:, 0]] - sy[key[:, 1]]))
    src, dst, dk, dt, dens = [], [], [], [], np.zeros(len(lat), dtype=int)
    for i in range(len(lat)):
        lo = np.searchsorted(t_hours, t_hours[i] - EE_HOURS, side="left")
        if lo >= i:
            continue
        cand = np.arange(lo, i)
        dist = np.hypot(ex[cand] - ex[i], ey[cand] - ey[i])
        keep = dist <= EE_KM
        cand, dist = cand[keep], dist[keep]
        dens[i] = len(cand)
        if len(cand) > EE_MAX:
            o = np.argsort(dist)[:EE_MAX]
            cand, dist = cand[o], dist[o]
        src += cand.tolist(); dst += [i] * len(cand); dk += dist.tolist(); dt += (t_hours[i] - t_hours[cand]).tolist()
    ee = dict(src=np.array(src, dtype=int), dst=np.array(dst, dtype=int), dist=np.array(dk), dt=np.array(dt))
    return es, ss, ee, dens



@st.cache_resource(show_spinner="Training the estimator on the training period...")
def train_estimator():
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    feats = ["latitude", "longitude", "nst", "gap", "dmin", "rms", "log_dt_prev", "log_density_past"]
    tr, va, te = (df[df["split"] == s_] for s_ in ("train", "val", "test"))
    reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(tr[feats], tr["mag"])
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(tr[feats], tr["depth_class"])
    res = va["mag"].to_numpy() - reg.predict(va[feats])
    pm = reg.predict(te[feats])
    met = dict(q10=float(np.quantile(res, 0.1)), q90=float(np.quantile(res, 0.9)), n_test=len(te),
               mae=float(np.abs(pm - te["mag"]).mean()), mae_base=float(np.abs(tr["mag"].mean() - te["mag"]).mean()),
               acc=float((clf.predict(te[feats]) == te["depth_class"]).mean()),
               acc_base=float((te["depth_class"] == tr["depth_class"].mode()[0]).mean()))
    return reg, clf, met

df, stations = load_data(), load_stations()
t_hours = ((df["time"] - df["time"].min()).dt.total_seconds() / 3600).to_numpy()
ES, SS, EE, DENS = build_edges(df["latitude"].to_numpy(), df["longitude"].to_numpy(), t_hours,
                               stations["latitude"].to_numpy(), stations["longitude"].to_numpy())
HAS_CLUSTERS, HAS_PRED = "cluster_str" in df.columns, "mag_pred" in df.columns
LA, LO = df["latitude"].to_numpy(), df["longitude"].to_numpy()
SLA, SLO = stations["latitude"].to_numpy(), stations["longitude"].to_numpy()


# ------------------------------------------------------------------ map helpers
def edge_trace(la, lo, lb, lob, name, color, width, opacity):
    n = len(la)
    a, b = np.full(3 * n, np.nan), np.full(3 * n, np.nan)
    a[0::3], a[1::3], b[0::3], b[1::3] = la, lb, lo, lob
    return go.Scattermap(lat=a, lon=b, mode="lines", line=dict(width=width, color=color), opacity=opacity, name=name, hoverinfo="skip")


def map_layout(fig, h):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TXT),
                      legend=dict(bgcolor="rgba(22,27,34,.85)", bordercolor=LINE, borderwidth=1, x=0.01, y=0.99))
    return fig


def base_map(h=650):
    fig = go.Figure()
    fig.update_layout(map=dict(style=MAP_STYLE, center=CENTER, zoom=ZOOM))
    return map_layout(fig, h)


def color_spec(col):
    if col == "magType": return dict(color_discrete_map=MAGTYPE_COLORS)
    if col == "depth_class": return dict(color_discrete_map=DEPTH_COLORS, category_orders={"depth_class": DEPTH_ORDER})
    if col == "split": return dict(color_discrete_map=SPLIT_COLORS, category_orders={"split": ["train", "val", "test"]})
    if col == "cluster_str": return dict(color_discrete_sequence=PALETTE, category_orders={"cluster_str": sorted(df["cluster_str"].unique())})
    return dict(color_continuous_scale="YlOrRd" if col == "mag" else "Turbo")


def event_map(d, col, h=650, stations_on=False, anim=False):
    d = d.sort_values("year") if anim else d
    fig = px.scatter_map(d, lat="latitude", lon="longitude", color=col, size="size_val", size_max=15, opacity=0.8,
                         hover_name="place", hover_data={"mag": ":.2f", "depth": ":.1f", "magType": True, "time": True,
                                                         "size_val": False, "latitude": False, "longitude": False, "year": False},
                         animation_frame="year" if anim else None, center=CENTER, zoom=ZOOM, height=h, map_style=MAP_STYLE,
                         **color_spec(col))
    if stations_on:
        fig.add_trace(go.Scattermap(lat=SLA, lon=SLO, mode="markers", name="stations", marker=dict(size=5, color=TXT, opacity=0.9),
                                    text=stations["station_id"], hoverinfo="text"))
    return map_layout(fig, h)


# ------------------------------------------------------------------ state, filters, presets
d_min, d_max = df["time"].min().date(), df["time"].max().date()
mg_min, mg_max = float(df["mag"].min()), float(df["mag"].max())
MT_ALL = sorted(df["magType"].unique())
DEFAULTS = dict(f_date=(d_min, d_max), f_mag=(mg_min, mg_max), f_depth=DEPTH_ORDER, f_mt=MT_ALL, f_split=["train", "val", "test"])
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("yr_ver", 0)


def preset(**kw):
    st.session_state.update({**DEFAULTS, **kw})
    st.session_state["yr_ver"] += 1


def selected_years():
    s = st.session_state.get(f"yr_chart_{st.session_state['yr_ver']}")
    try:
        return sorted({int(p["x"]) for p in s["selection"]["points"]})
    except Exception:
        return []


def clear_years():
    st.session_state["yr_ver"] += 1


top = st.columns([1.9, 3.9, 1.1], vertical_alignment="center")
top[0].markdown(f'<div class="brand">🌋 {APP_NAME}<span>Graph neural network analysis of Southern California seismicity</span></div>', unsafe_allow_html=True)
nav = top[1].segmented_control("Section", SECTIONS, default=SECTIONS[0], key="nav", label_visibility="collapsed") or SECTIONS[0]
with top[2].popover("Filters & presets", icon=":material/tune:", use_container_width=True):
    st.slider("Date range", d_min, d_max, key="f_date")
    st.slider("Magnitude", mg_min, mg_max, step=0.1, key="f_mag")
    st.multiselect("Depth class", DEPTH_ORDER, key="f_depth")
    st.multiselect("Magnitude type", MT_ALL, key="f_mt")
    st.multiselect("Period", ["train", "val", "test"], key="f_split")
    st.caption("Quick presets")
    p1, p2 = st.columns(2)
    p1.button("Ridgecrest 2019", on_click=preset, kwargs=dict(f_date=(pd.Timestamp("2019-07-01").date(), pd.Timestamp("2019-08-31").date())), width="stretch")
    p2.button("M4+ only", on_click=preset, kwargs=dict(f_mag=(4.0, mg_max)), width="stretch")
    p3, p4 = st.columns(2)
    p3.button("Deep events", on_click=preset, kwargs=dict(f_depth=["deep"]), width="stretch")
    p4.button("Test period", on_click=preset, kwargs=dict(f_split=["test"]), width="stretch")
    st.button("Reset everything", on_click=preset, width="stretch", type="primary")

d0, d1 = st.session_state.f_date
m_lo, m_hi = st.session_state.f_mag
BASE = ((df["time"] >= pd.Timestamp(d0, tz="UTC")) & (df["time"] < pd.Timestamp(d1, tz="UTC") + pd.Timedelta(days=1))
        & df["mag"].between(m_lo, m_hi) & df["depth_class"].isin(st.session_state.f_depth)
        & df["magType"].isin(st.session_state.f_mt) & df["split"].isin(st.session_state.f_split)).to_numpy()
ysel = selected_years()
VM = BASE & (df["year"].isin(ysel).to_numpy() if ysel else True)
view, base_view = df[VM], df[BASE]
FILTERED = nav in ("Catalog", "Graph", "Embeddings")      # other sections ignore the filters
if FILTERED and view.empty:
    st.warning("No events match the current filters. Open **Filters & presets** and press Reset everything.")
    st.stop()

ee_m, es_m = VM[EE["src"]] & VM[EE["dst"]], VM[ES["src"]]
if FILTERED:
    chips = f'<span class="chip">{len(view):,} of {len(df):,} events</span>'
    if ysel:
        chips += f'<span class="chip hot">years: {", ".join(map(str, ysel))}</span>'
    if (d0, d1) != (d_min, d_max) or (m_lo, m_hi) != (mg_min, mg_max) or len(st.session_state.f_depth) < 3 or len(st.session_state.f_split) < 3:
        chips += '<span class="chip hot">filters active</span>'
    cc = st.columns([5, 1.3, 1.3])
    cc[0].markdown(chips, unsafe_allow_html=True)
    if ysel:
        cc[1].button("Clear year selection", on_click=clear_years, width="stretch")
    EXPORT = ["id", "time", "latitude", "longitude", "depth", "mag", "magType", "depth_class", "place", "split"]
    cc[2].download_button("Download CSV", view[EXPORT].to_csv(index=False), "filtered_events.csv", "text/csv", width="stretch")

    k = st.columns(6)
    kpi(k[0], "Events", f"{len(view):,}", f"of {len(df):,} in catalog")
    kpi(k[1], "Stations", f"{len(stations):,}", "CI network")
    kpi(k[2], "Graph edges", f"{es_m.sum() + len(SS['a']) + ee_m.sum():,}", "event-station, station-station, event-event")
    kpi(k[3], "Largest event", f"M{view['mag'].max():.1f}", str(view.loc[view['mag'].idxmax(), 'place'])[:34])
    kpi(k[4], "Median depth", f"{view['depth'].median():.1f} km", f"max {view['depth'].max():.0f} km")
    kpi(k[5], "Time span", f"{view['year'].min()}-{view['year'].max()}", f"{(view['time'].max() - view['time'].min()).days:,} days")


# ================================================================== CATALOG
if nav == "Catalog":
    lead("Where and when the ground has moved",
         "Every recorded event in the catalog. Select bars in the yearly chart to cross-filter the whole dashboard, "
         "or press play to watch activity move year by year.")
    c = st.columns([1.2, 1, 1, 2])
    opts = {"Magnitude type": "magType", "Depth class": "depth_class", "Magnitude": "mag", "Period": "split", "Year": "year"}
    if HAS_CLUSTERS:
        opts["GNN cluster"] = "cluster_str"
    pick = c[0].selectbox("Color by", list(opts))
    st_on = c[1].toggle("Stations", True)
    anim = c[2].toggle("Play by year", False)
    L, R = st.columns([2.1, 1], gap="medium")
    with L:
        top_mt = view["magType"].value_counts()
        top_reg = view["region"].value_counts()
        big = view.loc[view["mag"].idxmax()]
        card("Earthquake map", event_map(view, opts[pick], 700, st_on, anim),
             f"<b>{len(view):,} events</b> plotted, none sampled. {top_mt.index[0]} is the most common magnitude type "
             f"({top_mt.iloc[0] / len(view):.0%}). The busiest area is <b>{esc(top_reg.index[0])}</b> ({top_reg.iloc[0]:,} events). "
             f"The largest event is <b>M{big['mag']:.1f}</b> near {esc(big['place'])}. Marker size grows with magnitude; "
             f"clear linear belts mark fault systems.",
             sub="Marker size = magnitude. Drag to pan, scroll to zoom, click legend items to hide groups.")
    with R:
        yc = base_view.groupby(["year", "depth_class"]).size().reset_index(name="events")
        f = px.bar(yc, x="year", y="events", color="depth_class", color_discrete_map=DEPTH_COLORS,
                   category_orders={"depth_class": DEPTH_ORDER})
        f.update_layout(dragmode="select", barmode="stack")
        yt = base_view["year"].value_counts()
        card("Events per year", style(f, 330), f"Peak year is <b>{yt.idxmax()}</b> with {yt.max():,} events, versus a median of "
             f"{int(yt.median()):,}. Spikes are usually aftershock sequences. <b>Click or box-select bars</b> to filter every chart.",
             sub="Interactive: acts as a cross-filter", key=f"yr_chart_{st.session_state['yr_ver']}",
             on_select="rerun", selection_mode=("points", "box"))
        mc = view["magType"].value_counts().reset_index()
        f = px.pie(mc, names="magType", values="count", hole=0.55, color="magType", color_discrete_map=MAGTYPE_COLORS)
        f.update_traces(textinfo="label+percent", marker=dict(line=dict(color=BG, width=2)))
        card("Magnitude type share", style(f, 330, False),
             "Types are different scales: ML local, MW moment, MD duration. Moment magnitude (MW) is most reliable for large events, "
             f"and makes up {(view['magType'] == 'MW').mean():.1%} of this view.")

    mo = view.set_index("time").resample("MS").size()
    pk = mo.idxmax()
    f = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.38, 0.38, 0.24],
                      subplot_titles=("Magnitude", "Depth (km)", "Events per month"))
    for dc in DEPTH_ORDER:
        s = view[view["depth_class"] == dc]
        f.add_trace(go.Scattergl(x=s["time"], y=s["mag"], mode="markers", name=dc, legendgroup=dc,
                                 marker=dict(size=4 + 2 * (s["mag"] - 2.5), color=DEPTH_COLORS[dc], opacity=0.7), hovertext=s["place"]), row=1, col=1)
        f.add_trace(go.Scattergl(x=s["time"], y=s["depth"], mode="markers", name=dc, legendgroup=dc, showlegend=False,
                                 marker=dict(size=4, color=DEPTH_COLORS[dc], opacity=0.6), hovertext=s["place"]), row=2, col=1)
    f.add_trace(go.Bar(x=mo.index, y=mo.values, marker_color="#5b8def", showlegend=False, name="events"), row=3, col=1)
    f.update_yaxes(autorange="reversed", row=2, col=1)
    card("Magnitude and depth over the years", style(f, 740, top=44),
         f"The busiest month is <b>{pk:%B %Y}</b> with {mo.max():,} events, {mo.max() / max(mo.median(), 1):.0f}x the monthly median. "
         "Vertical stacks of dots mark bursts of aftershocks. Depth is plotted downwards, so deeper events sit lower. Drag on any panel to zoom all three.",
         sub="Dot size = magnitude, color = depth class")

    g = st.columns(3, gap="medium")
    mags = np.arange(2.5, view["mag"].max() + 0.05, 0.1)
    cum = np.array([(view["mag"] >= m).sum() for m in mags])
    ok = cum > 0
    bval = np.log10(np.e) / (view["mag"].mean() - 2.495)                 # Aki maximum-likelihood b-value (Mc = 2.5)
    f = go.Figure(go.Scatter(x=mags[ok], y=cum[ok], mode="markers+lines", line=dict(color=ACC), name="N(>=M)"))
    f.update_yaxes(type="log", title="cumulative count"); f.update_xaxes(title="magnitude")
    with g[0]:
        card("Gutenberg-Richter law", style(f, 360, False),
             f"Each step up in magnitude is rarer by a constant factor. Estimated <b>b-value {bval:.2f}</b>; values near 1.0 are typical, "
             "so the catalog behaves like natural seismicity.", sub="Cumulative count of events at or above each magnitude")
    f = px.histogram(view, x="depth", color="depth_class", nbins=60, color_discrete_map=DEPTH_COLORS, category_orders={"depth_class": DEPTH_ORDER})
    sh = view["depth_class"].value_counts(normalize=True)
    with g[1]:
        card("Depth distribution", style(f, 360),
             f"Median depth is <b>{view['depth'].median():.1f} km</b>. Shallow (≤5 km) {sh.get('shallow', 0):.0%}, mid (5-10 km) {sh.get('mid', 0):.0%}, "
             f"deep (>10 km) {sh.get('deep', 0):.0%}. These classes are the targets for the depth classifier.", sub="Events by depth in km")
    f = px.density_heatmap(view, x="mag", y="depth", nbinsx=35, nbinsy=40, color_continuous_scale="Magma")
    f.update_yaxes(autorange="reversed"); f.update_coloraxes(showscale=False)
    with g[2]:
        card("Magnitude vs depth", style(f, 360, False),
             f"Correlation between magnitude and depth is <b>{view['mag'].corr(view['depth']):.2f}</b>, so depth alone says little about size. "
             "Brighter cells hold more events.", sub="Density of events")

    h = st.columns([1.4, 1, 1.2], gap="medium")
    ym = view.assign(month=view["time"].dt.month).groupby(["year", "month"]).size().unstack(fill_value=0)
    f = px.imshow(ym.T, aspect="auto", color_continuous_scale="Inferno", labels=dict(x="year", y="month", color="events"))
    yy, mm = np.unravel_index(ym.to_numpy().argmax(), ym.shape)
    with h[0]:
        card("Year and month heatmap", style(f, 400, False),
             f"The brightest cell is <b>{ym.index[yy]}, month {ym.columns[mm]}</b> ({ym.to_numpy().max():,} events). "
             "Bright single cells indicate sequences, not seasonality; earthquakes do not follow the calendar.", sub="Events per month")
    place = view["region"].value_counts().head(15).reset_index()
    f = px.bar(place, x="count", y="region", orientation="h", color_discrete_sequence=["#5b8def"])
    f.update_yaxes(autorange="reversed", title="")
    with h[1]:
        card("Most active locations", style(f, 400, False),
             f"The top 15 places hold <b>{place['count'].sum() / len(view):.0%}</b> of events; <b>{esc(place.iloc[0]['region'])}</b> leads.",
             sub="Nearest town named in the catalog")
    with h[2]:
        with st.container(border=True):
            st.markdown('<div class="ct">Ten largest events</div><div class="cs">In the current view</div>', unsafe_allow_html=True)
            t = view.nlargest(10, "mag")[["time", "mag", "magType", "depth", "place"]].copy()
            t["time"] = t["time"].dt.strftime("%Y-%m-%d")
            st.dataframe(t, width="stretch", hide_index=True, height=340)
            st.markdown(f'<div class="insight">The largest is <b>M{t.iloc[0]["mag"]:.1f}</b> on {t.iloc[0]["time"]}. '
                        'Big events are rare but dominate the energy released.</div>', unsafe_allow_html=True)

# ================================================================== GRAPH
elif nav == "Graph":
    n_es, n_ss, n_ee = int(es_m.sum()), len(SS["a"]), int(ee_m.sum())
    lead("The graph the model learns from",
         "Events and stations are nodes; links join stations to nearby events, stations to neighbouring stations, and earlier events to later ones. "
         "Toggle layers, then spotlight a single station to see its local neighbourhood.")
    deg = np.bincount(ES["dst"][es_m], minlength=len(stations))
    hubs = pd.DataFrame({"station": stations["station_id"], "events_linked": deg}).sort_values("events_linked", ascending=False)

    schema = go.Figure()
    for nm, x, col, n in [("Event", 0.14, ACC, len(view)), ("Station", 0.86, "#2dd4bf", len(stations))]:
        schema.add_trace(go.Scatter(x=[x], y=[0.5], mode="markers+text", text=[f"<b>{nm}</b><br>{n:,}"], hoverinfo="skip",
                                    marker=dict(size=88, color=col), textfont=dict(color=BG, size=13), showlegend=False))
    th = np.linspace(0.4, 2 * np.pi - 0.4, 40)
    for x, col, lab in [(0.14, ACC, f"precedes · {n_ee:,}"), (0.86, "#2dd4bf", f"near · {n_ss:,}")]:
        schema.add_trace(go.Scatter(x=x + 0.045 * np.sin(th), y=0.78 - 0.16 * np.cos(th), mode="lines", line=dict(color=col, width=2),
                                    hoverinfo="skip", showlegend=False))
        schema.add_annotation(x=x, y=1.02, text=lab, showarrow=False, font=dict(color=col, size=12))
    for y, ax_, x_, lab in [(0.58, 0.22, 0.78, f"recorded_by · {n_es:,}"), (0.42, 0.78, 0.22, f"rev_recorded_by · {n_es:,}")]:
        schema.add_annotation(x=x_, y=y, ax=ax_, ay=y, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                              arrowwidth=2, arrowcolor=MUTED)
        schema.add_annotation(x=0.5, y=y + (0.07 if y > 0.5 else -0.07), text=lab, showarrow=False, font=dict(color=MUTED, size=11))
    schema.update_xaxes(visible=False, range=[0, 1]); schema.update_yaxes(visible=False, range=[0.1, 1.1])
    a, b = st.columns([1, 2.2], gap="medium")
    with a:
        card("Graph schema", style(schema, 300, False),
             f"Two node types and four edge types, <b>{n_es + n_ss + n_ee:,}</b> unique links in this view. Information flows both ways between events "
             "and stations, and only forward in time between events, so the model never sees the future.", sub="Edge counts follow the filters")
    f = px.bar(hubs.head(20), x="station", y="events_linked", color="events_linked", color_continuous_scale="Viridis")
    f.update_coloraxes(showscale=False)
    with b:
        card("Hub stations", style(f, 300, False),
             f"<b>{esc(hubs.iloc[0]['station'])}</b> is the busiest hub with {hubs.iloc[0]['events_linked']:,} linked events. The top 20 stations carry "
             f"<b>{hubs.head(20)['events_linked'].sum() / max(deg.sum(), 1):.0%}</b> of all event-station links, because events cluster along faults.",
             sub="Events linked per station (top 20)")

    t = st.columns(4)
    show_ss, show_es = t[0].toggle("Station-station edges", True), t[1].toggle("Event-station edges", True)
    show_ee, show_ev = t[2].toggle("Event-event edges", True), t[3].toggle("Event nodes", True)
    fig = base_map(720)
    if show_ss:
        fig.add_trace(edge_trace(SLA[SS["a"]], SLO[SS["a"]], SLA[SS["b"]], SLO[SS["b"]], f"station-station ({n_ss:,})", "#2dd4bf", 1.2, 0.5))
    if show_es:
        fig.add_trace(edge_trace(LA[ES["src"][es_m]], LO[ES["src"][es_m]], SLA[ES["dst"][es_m]], SLO[ES["dst"][es_m]],
                                 f"event-station ({n_es:,})", "#5b8def", 0.5, 0.18))
    if show_ee:
        fig.add_trace(edge_trace(LA[EE["src"][ee_m]], LO[EE["src"][ee_m]], LA[EE["dst"][ee_m]], LO[EE["dst"][ee_m]],
                                 f"event-event ({n_ee:,})", ACC, 1.1, 0.55))
    if show_ev:
        fig.add_trace(go.Scattermap(lat=view["latitude"], lon=view["longitude"], mode="markers", name=f"events ({len(view):,})",
                                    marker=dict(size=4 + 2.2 * (view["mag"] - 2.5).clip(lower=0), color=ACC, opacity=0.85),
                                    text=view["place"] + " | M" + view["mag"].round(1).astype(str), hoverinfo="text"))
    fig.add_trace(go.Scattermap(lat=SLA, lon=SLO, mode="markers", name=f"stations ({len(stations):,})",
                                marker=dict(size=6 + 20 * np.sqrt(deg / max(deg.max(), 1)), color=deg, colorscale="Viridis", showscale=True,
                                            colorbar=dict(title="events<br>linked", x=0.99, len=0.4, y=0.3)),
                                text=stations["station_id"] + " | " + pd.Series(deg).astype(str) + " events", hoverinfo="text"))
    card("Full network on the map", fig,
         f"The whole graph: <b>{len(stations):,} stations</b>, <b>{len(view):,} events</b> and <b>{n_es + n_ss + n_ee:,} unique edges</b>. Larger, brighter stations are hubs. "
         "Orange event-event links trace aftershock chains; blue spokes show which stations each event connects to.",
         sub="Turn layers on and off above to isolate each edge type")

    sp = st.selectbox("Spotlight a station (defaults to the busiest hub)", hubs["station"].tolist(), index=0)
    si = int(stations.index[stations["station_id"] == sp][0])
    nb = np.r_[SS["b"][SS["a"] == si], SS["a"][SS["b"] == si]]
    nbd = np.r_[SS["dist"][SS["a"] == si], SS["dist"][SS["b"] == si]]
    linked = ES["src"][(ES["dst"] == si) & es_m]
    f = base_map(520)
    f.update_layout(map=dict(center=dict(lat=float(SLA[si]), lon=float(SLO[si])), zoom=7.4))
    f.add_trace(edge_trace(LA[linked], LO[linked], np.full(len(linked), SLA[si]), np.full(len(linked), SLO[si]), "linked events", "#5b8def", 0.6, 0.3))
    f.add_trace(edge_trace(np.full(len(nb), SLA[si]), np.full(len(nb), SLO[si]), SLA[nb], SLO[nb], "station neighbours", "#2dd4bf", 2, 0.9))
    f.add_trace(go.Scattermap(lat=LA[linked], lon=LO[linked], mode="markers", name="events", text=df["place"].to_numpy()[linked], hoverinfo="text",
                              marker=dict(size=4 + 2 * (df["mag"].to_numpy()[linked] - 2.5).clip(min=0), color=ACC)))
    f.add_trace(go.Scattermap(lat=SLA[nb], lon=SLO[nb], mode="markers", name="neighbours", marker=dict(size=10, color="#2dd4bf"),
                              text=stations["station_id"].to_numpy()[nb], hoverinfo="text"))
    f.add_trace(go.Scattermap(lat=[SLA[si]], lon=[SLO[si]], mode="markers", name=sp, marker=dict(size=16, color="#ef4444"), text=[sp], hoverinfo="text"))
    card(f"Station spotlight: {esc(sp)}", f,
         f"<b>{esc(sp)}</b> is connected to <b>{len(linked):,} events</b> (mean M{df['mag'].to_numpy()[linked].mean() if len(linked) else 0:.2f}) and "
         f"{len(nb)} neighbouring stations, the nearest {nbd.min():.0f} km away. This is the neighbourhood the GNN aggregates over when it builds this station's embedding.",
         sub="Red = selected station, teal = its graph neighbours, orange = events that link to it")

    g = st.columns(3, gap="medium")
    sel = ES["dist"][es_m]
    f = px.histogram(x=sel, nbins=60, color_discrete_sequence=["#5b8def"], labels={"x": "event-to-station distance (km)"})
    with g[0]:
        card("Event-station edge length", style(f, 340, False),
             f"The median link is <b>{np.median(sel):.0f} km</b> and 90% are under {np.percentile(sel, 90):.0f} km, so links stay local and "
             "reflect real network geometry.", sub="All 5 nearest-station links per event")
    ee_d, ee_t = EE["dist"][ee_m], EE["dt"][ee_m]
    f = go.Figure(go.Scattergl(x=ee_t, y=ee_d, mode="markers", marker=dict(size=3, opacity=0.35, color=df["mag"].to_numpy()[EE["dst"][ee_m]],
                                                                            colorscale="YlOrRd", showscale=True, colorbar=dict(title="mag"))))
    f.update_xaxes(type="log", title="time lag (hours)"); f.update_yaxes(title="distance (km)")
    with g[1]:
        card("Event-event links", style(f, 340, False),
             (f"<b>{(ee_t <= 24).mean():.0%}</b> of links connect events less than a day apart (median lag {np.median(ee_t):.0f} h), "
              "the signature of aftershock-style clustering." if len(ee_t) else "No event-event links in this view."),
             sub="Time lag vs distance; color = magnitude of the later event")
    dd = DENS[VM]
    f = px.histogram(x=np.minimum(dd, 60), nbins=60, color_discrete_sequence=[ACC], labels={"x": "earlier neighbours within 50 km / 7 d (capped at 60)"})
    f.update_yaxes(type="log")
    with g[2]:
        card("Space-time clustering", style(f, 340, False),
             f"<b>{(dd == 0).mean():.0%}</b> of events have no recent neighbour (isolated background activity), while "
             f"<b>{(dd >= 10).mean():.0%}</b> have 10 or more, sitting inside an active sequence.", sub="Number of earlier events nearby, per event")

# ================================================================== EMBEDDINGS
elif nav == "Embeddings":
    lead("What the graph model learned", "Each event gets a learned vector. Nearby points in the projection are events the GNN treats as similar; "
         "clusters group them without ever seeing magnitude or depth as inputs.")
    if need("event_clusters.csv", "cluster_profiles.csv", "cluster_comparison.csv") and HAS_CLUSTERS:
        cv = view.dropna(subset=["proj_x"])
        copts = {"Cluster": "cluster_str", "Magnitude": "mag", "Depth class": "depth_class", "Magnitude type": "magType", "Year": "year"}
        cc_ = st.radio("Color embedding by", list(copts), horizontal=True)
        csz = cv["cluster_str"].value_counts()
        a, b = st.columns(2, gap="medium")
        f = px.scatter(cv, x="proj_x", y="proj_y", color=copts[cc_], hover_name="place", render_mode="webgl",
                       labels={"proj_x": "dim 1", "proj_y": "dim 2"}, **color_spec(copts[cc_]))
        f.update_traces(marker=dict(size=4, opacity=0.8))
        with a:
            card("Embedding space", style(f, 560),
                 f"{len(cv):,} events in <b>{len(csz)} clusters</b>; the largest ({csz.index[0]}) holds {csz.iloc[0] / len(cv):.0%}. "
                 "Recolor by depth or magnitude: if colors separate, the embedding captured that property.", sub="2D projection of all event embeddings")
        with b:
            card("Same events, geographic view", event_map(cv, copts[cc_], 560),
                 "Compare with the left plot: clusters that are also spatially compact reflect fault regions; mixed ones capture behaviour beyond location.",
                 sub="Linked to the selector above")
        r = st.columns(3, gap="medium")
        comp = cv.groupby(["cluster_str", "depth_class"]).size().reset_index(name="events")
        f = px.bar(comp, x="cluster_str", y="events", color="depth_class", color_discrete_map=DEPTH_COLORS, category_orders={"depth_class": DEPTH_ORDER})
        dshare = comp.pivot(index="cluster_str", columns="depth_class", values="events").fillna(0)
        dshare = dshare.div(dshare.sum(axis=1), axis=0)
        with r[0]:
            card("Depth mix per cluster", style(f, 340),
                 f"<b>{dshare['deep'].idxmax() if 'deep' in dshare else '-'}</b> is the most deep-dominated cluster "
                 f"({dshare['deep'].max():.0%} deep)." if "deep" in dshare else "Depth mix per cluster.", sub="Events by depth class")
        f = px.violin(cv, x="cluster_str", y="mag", color="cluster_str", box=True, color_discrete_sequence=PALETTE,
                      category_orders={"cluster_str": sorted(cv["cluster_str"].unique())})
        med = cv.groupby("cluster_str")["mag"].median()
        with r[1]:
            card("Magnitude per cluster", style(f, 340, False),
                 f"Median magnitude spans {med.min():.2f} to {med.max():.2f} (highest in <b>{med.idxmax()}</b>). Small differences confirm that "
                 "magnitude is only weakly encoded in this graph.", sub="Distribution and box per cluster")
        cy = cv.groupby(["cluster_str", "year"]).size().unstack(fill_value=0)
        conc = (cy.max(axis=1) / cy.sum(axis=1))
        f = px.imshow(cy, aspect="auto", color_continuous_scale="Viridis", labels=dict(color="events"))
        with r[2]:
            card("Cluster activity by year", style(f, 340, False),
                 f"<b>{conc.idxmax()}</b> is the most time-concentrated: {conc.max():.0%} of its events fall in a single year, "
                 "which points to a sequence such as an aftershock swarm.", sub="Events per cluster and year")
        with st.container(border=True):
            st.markdown('<div class="ct">Cluster profiles</div>', unsafe_allow_html=True)
            st.dataframe(pd.read_csv(RES / "cluster_profiles.csv"), width="stretch", hide_index=True)
            st.markdown('<div class="insight">Centre, size, typical magnitude and depth of each cluster, plus the nearest town that appears most often.</div>',
                        unsafe_allow_html=True)
        cmp_ = pd.read_csv(RES / "cluster_comparison.csv")
        mcols = [c_ for c_ in cmp_.columns if "NMI" in c_ or "eta" in c_]
        m = cmp_.melt(id_vars="space", value_vars=mcols)
        f = px.bar(m, x="variable", y="value", color="space", barmode="group", color_discrete_sequence=[ACC, "#5b8def", "#2dd4bf"])
        f.update_xaxes(title="")
        nmi = cmp_.set_index("space")[mcols[0]] if mcols else None
        card("GNN embeddings vs raw features", style(f, 340),
             (f"On held-out events <b>{nmi.idxmax()}</b> scores highest on depth-class agreement (NMI {nmi.max():.3f}); the GNN scores {nmi.get('GNN embeddings', float('nan')):.3f}. "
              "Silhouette is not comparable across spaces, so judge on these test metrics.") if nmi is not None else "",
             sub="Higher is better; evaluated on test events")

# ================================================================== STATIONS
elif nav == "Stations":
    lead("Station relationships learned by the GNN", "Stations that behave alike end up with similar embeddings, even when far apart. "
         "Colors are groups; lines join the most similar distant pairs.")
    if need("station_clusters.csv", "station_similar_pairs.csv"):
        sc = pd.read_csv(RES / "station_clusters.csv")
        sc["group"] = "Group " + sc["cluster"].astype(str)
        pairs = pd.read_csv(RES / "station_similar_pairs.csv")
        n_pairs = st.slider("Similar distant pairs to draw", 0, len(pairs), len(pairs))
        fig = px.scatter_map(sc, lat="latitude", lon="longitude", color="group", hover_name="station_id", center=CENTER, zoom=ZOOM,
                             height=680, map_style=MAP_STYLE, color_discrete_sequence=PALETTE)
        if n_pairs:
            co, p = sc.set_index("station_id")[["latitude", "longitude"]], pairs.head(n_pairs)
            fig.add_trace(edge_trace(co.loc[p["station_a"], "latitude"].to_numpy(), co.loc[p["station_a"], "longitude"].to_numpy(),
                                     co.loc[p["station_b"], "latitude"].to_numpy(), co.loc[p["station_b"], "longitude"].to_numpy(),
                                     "most similar pairs", TXT, 1, 0.5))
        gs = sc["group"].value_counts()
        card("Station groups", map_layout(fig, 680),
             f"<b>{len(gs)} groups</b> over {len(sc):,} stations; the largest ({gs.index[0]}) has {gs.iloc[0]}. "
             f"The {n_pairs} lines link stations averaging {pairs.head(max(n_pairs, 1))['dist_km'].mean():.0f} km apart that the model still treats as alike, "
             "a relationship that proximity alone would not reveal.", sub="Drag the slider above to add or remove pair links")
        a, b, c3 = st.columns(3, gap="medium")
        f = px.bar(gs.sort_index().reset_index(), x="group", y="count", color="group", color_discrete_sequence=PALETTE)
        with a:
            card("Stations per group", style(f, 340, False), f"Group sizes range from {gs.min()} to {gs.max()} stations.", sub="Count of stations")
        f = px.scatter(pairs, x="dist_km", y="cosine_sim", color="cosine_sim", color_continuous_scale="Viridis", hover_data=["station_a", "station_b"])
        with b:
            card("Similarity vs distance", style(f, 340, False),
                 f"Top pair similarity reaches <b>{pairs['cosine_sim'].max():.2f}</b> at {pairs.loc[pairs['cosine_sim'].idxmax(), 'dist_km']:.0f} km. "
                 f"Correlation with distance is {pairs['cosine_sim'].corr(pairs['dist_km']):.2f}.", sub="Each dot is a station pair")
        heat = RES / "figures" / "station_similarity_heatmap.png"
        with c3:
            with st.container(border=True):
                st.markdown('<div class="ct">Similarity matrix</div><div class="cs">Stations sorted by group</div>', unsafe_allow_html=True)
                if heat.exists():
                    st.image(str(heat))
                st.markdown('<div class="insight">Bright blocks on the diagonal mean stations inside a group look alike.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="ct">Most similar distant pairs</div>', unsafe_allow_html=True)
            st.dataframe(pairs, width="stretch", hide_index=True, height=300)
            st.markdown('<div class="insight">Pairs more than 50 km apart, ranked by cosine similarity of their embeddings.</div>', unsafe_allow_html=True)

# ================================================================== MODELS
elif nav == "Models":
    lead("GNN vs classical machine learning", "Evaluated on the held-out final 15% of the timeline, averaged over seeds, with ablations that remove "
         "temporal edges or station nodes.")
    if need("metrics_all.csv"):
        mm = pd.read_csv(RES / "metrics_all.csv")
        cols = [c_ for c_ in mm.columns if c_ not in ("model", "seed")]
        g = mm.groupby("model", sort=False)[cols]
        mean, std = g.mean(), g.std().fillna(0)
        kind = np.where(mean.index.str.startswith("GNN"), "GNN", "ML baseline")
        f = make_subplots(rows=1, cols=4, shared_yaxes=True,
                          subplot_titles=("Magnitude MAE ↓", "Magnitude RMSE ↓", "Magnitude R² ↑", "Depth macro-F1 ↑"))
        for i, met in enumerate(["mag_MAE", "mag_RMSE", "mag_R2", "depth_MacroF1"], 1):
            if met in mean:
                f.add_trace(go.Bar(y=mean.index, x=mean[met], error_x=dict(array=std[met]), orientation="h", showlegend=False,
                                   marker_color=np.where(kind == "GNN", ACC, "#5b8def")), row=1, col=i)
        f.update_yaxes(autorange="reversed")
        f.update_annotations(font=dict(size=13, color=TXT))
        gn, bl = mean[kind == "GNN"], mean[kind != "GNN"]
        bl_f1 = bl["depth_MacroF1"].idxmax()
        card("Model comparison", style(f, 420, False, top=50),
             f"Best GNN variant on depth macro-F1 is <b>{gn['depth_MacroF1'].idxmax()}</b> ({gn['depth_MacroF1'].max():.3f}) vs best baseline "
             f"<b>{bl_f1}</b> ({bl['depth_MacroF1'].max():.3f}). For magnitude MAE the best GNN scores {gn['mag_MAE'].min():.3f} against "
             f"{bl['mag_MAE'].min():.3f}. Orange = GNN variants, blue = baselines; whiskers are the std over seeds.", sub="Mean over seeds on the test period")
        with st.container(border=True):
            st.markdown('<div class="ct">Full metric table</div>', unsafe_allow_html=True)
            st.dataframe(mean.round(3).astype(str) + " ± " + std.round(3).astype(str), width="stretch")
            st.markdown('<div class="insight">Rows with "w/o" are ablations; a drop versus GNN (full) shows how much that part of the graph helps.</div>',
                        unsafe_allow_html=True)
    if HAS_PRED and need("gnn_predictions.csv"):
        te = df[df["split"] == "test"].copy()
        te["err"] = te["mag_pred"] - te["mag"]
        a, b = st.columns(2, gap="medium")
        f = px.density_heatmap(te, x="mag", y="mag_pred", nbinsx=30, nbinsy=30, color_continuous_scale="Magma",
                               labels={"mag": "actual magnitude", "mag_pred": "GNN estimate"})
        lo, hi = float(te["mag"].min()), float(te["mag"].max())
        f.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color=TXT))
        with a:
            card(f"Magnitude: estimate vs actual ({len(te):,} test events)", style(f, 400, False),
                 f"Correlation <b>{te['mag'].corr(te['mag_pred']):.2f}</b>, mean absolute error {te['err'].abs().mean():.2f}. Predictions spread "
                 f"{te['mag_pred'].std() / te['mag'].std():.0%} as widely as reality, so large events are pulled toward the average. The dashed line is a perfect model.",
                 sub="Every held-out event")
        cm = pd.crosstab(te["depth_class"], te["depth_pred"]).reindex(index=DEPTH_ORDER, columns=DEPTH_ORDER, fill_value=0)
        rec = (np.diag(cm) / cm.sum(axis=1).replace(0, np.nan)).fillna(0)
        f = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="GNN estimate", y="actual", color="events"))
        with b:
            card("Depth class confusion matrix", style(f, 400, False),
                 f"Accuracy <b>{np.trace(cm.to_numpy()) / cm.to_numpy().sum():.0%}</b>. Recall per class: shallow {rec.iloc[0]:.0%}, mid {rec.iloc[1]:.0%}, "
                 f"deep {rec.iloc[2]:.0%}. Off-diagonal cells show which depths get confused.", sub="Rows = actual, columns = predicted")
        a, b = st.columns(2, gap="medium")
        mo = te.set_index("time")["err"].abs().resample("MS").mean().reset_index()
        f = px.line(mo, x="time", y="err", markers=True, color_discrete_sequence=[ACC], labels={"err": "MAE"})
        with a:
            card("Error over the test period", style(f, 340, False),
                 f"Monthly error peaks in <b>{mo.loc[mo['err'].idxmax(), 'time']:%B %Y}</b> ({mo['err'].max():.2f}); a rising trend would indicate drift.",
                 sub="Mean absolute magnitude error per month")
        f = px.box(te, x="depth_class", y="err", color="depth_class", color_discrete_map=DEPTH_COLORS,
                   category_orders={"depth_class": DEPTH_ORDER}, labels={"err": "estimate − actual"})
        bias = te.groupby("depth_class")["err"].mean()
        with b:
            card("Residuals by depth class", style(f, 340, False),
                 f"Mean bias: shallow {bias.get('shallow', 0):+.2f}, mid {bias.get('mid', 0):+.2f}, deep {bias.get('deep', 0):+.2f}. "
                 "Values away from zero mean the model systematically over- or under-estimates that group.", sub="Estimate minus actual")

# ================================================================== PREDICT
elif nav == "Predict":
    lead("Estimate and outlook",
         "Two statistical tools built from the full catalog. They describe likely behaviour given past data. "
         "They cannot say when or where a specific earthquake will happen.")
    mode = st.radio("Tool", ["Estimate an event", "Regional activity outlook"], horizontal=True, label_visibility="collapsed")
    regs = (df.groupby("region").agg(n=("id", "size"), lat=("latitude", "median"), lon=("longitude", "median"))
            .sort_values("n", ascending=False).head(30))
    ex_all, ey_all = to_xy_km(LA, LO)
    I, O = st.columns([1, 2.4], gap="medium")

    if mode == "Estimate an event":
        try:
            reg, clf, MET = train_estimator()
        except ImportError:
            st.error("This tool needs scikit-learn. Run `pip install scikit-learn`.")
            st.stop()
        with I:
            with st.container(border=True):
                st.markdown('<div class="ct">Describe the event</div><div class="cs">Pick an area, then adjust the values</div>', unsafe_allow_html=True)
                rn = st.selectbox("Start from area", regs.index.tolist())
                sub = df[df["region"] == rn]
                lat = st.number_input("Latitude", 31.0, 38.0, float(np.clip(regs.loc[rn, "lat"], 31, 38)), 0.01, key=f"lat_{rn}")
                lon = st.number_input("Longitude", -122.0, -113.0, float(np.clip(regs.loc[rn, "lon"], -122, -113)), 0.01, key=f"lon_{rn}")
                nst = st.slider("Stations used (nst)", int(df["nst"].min()), int(df["nst"].max()), int(sub["nst"].median()), key=f"nst_{rn}")
                gap = st.slider("Azimuthal gap (degrees)", 0, 360, int(min(sub["gap"].median(), 360)), key=f"gap_{rn}")
                dmin = st.number_input("Nearest station distance, dmin (degrees)", 0.0, float(df["dmin"].max()),
                                       float(round(sub["dmin"].median(), 3)), 0.01, key=f"dmin_{rn}")
                rms = st.number_input("RMS residual (s)", 0.0, float(df["rms"].max()), float(round(sub["rms"].median(), 2)), 0.01, key=f"rms_{rn}")
                hrs = st.number_input("Hours since the previous event", 0.0, 5000.0, float(round(np.expm1(sub["log_dt_prev"].median()), 1)), 1.0, key=f"hrs_{rn}")
                dens = st.slider("Earlier events within 50 km / 7 days", 0, 60, int(min(np.expm1(sub["log_density_past"].median()), 60)), key=f"dens_{rn}")
        X = pd.DataFrame([dict(latitude=lat, longitude=lon, nst=nst, gap=gap, dmin=dmin, rms=rms,
                               log_dt_prev=np.log1p(hrs), log_density_past=np.log1p(dens))])
        mag = float(reg.predict(X)[0])
        lo_m, hi_m = mag + MET["q10"], mag + MET["q90"]
        pr = pd.Series(clf.predict_proba(X)[0], index=clf.classes_).reindex(DEPTH_ORDER).fillna(0)
        cx, cy = to_xy_km(lat, lon)
        dloc = np.hypot(ex_all - cx, ey_all - cy)
        local = df[dloc <= 30]
        with O:
            kc = st.columns(3)
            kpi(kc[0], "Estimated magnitude", f"M{mag:.2f}", f"80% range M{lo_m:.1f} to M{hi_m:.1f}")
            kpi(kc[1], "Most likely depth class", pr.idxmax().title(), f"{pr.max():.0%} probability")
            kpi(kc[2], "Similar past events", f"{len(local):,}", "within 30 km of this point")
            st.write("")
            a, b = st.columns(2, gap="medium")
            f = px.bar(x=pr.index, y=pr.values, color=pr.index, color_discrete_map=DEPTH_COLORS, labels={"x": "depth class", "y": "probability"})
            f.update_yaxes(tickformat=".0%", range=[0, 1])
            with a:
                card("Depth class probabilities", style(f, 330, False),
                     f"The model favours <b>{pr.idxmax()}</b> ({pr.max():.0%}). When the top two classes are close, treat the estimate as undecided. "
                     "Classes: shallow 5 km or less, mid 5 to 10 km, deep over 10 km.", sub="Random forest trained on the training period")
            f = px.histogram(local if len(local) else df, x="mag", nbins=30, color_discrete_sequence=["#5b8def"])
            f.add_vline(x=mag, line_color=ACC, line_width=3, annotation_text="estimate")
            f.add_vrect(x0=lo_m, x1=hi_m, fillcolor=ACC, opacity=0.12, line_width=0)
            ref = local if len(local) else df
            with b:
                card("Where the estimate sits among past events", style(f, 330, False),
                     f"Past events near this point have a median of <b>M{ref['mag'].median():.2f}</b> and a maximum of M{ref['mag'].max():.1f}. "
                     "The shaded band is the 80% range measured on the validation period.", sub="Magnitudes within 30 km (or whole catalog if none)")
            ts = cKDTree(np.c_[to_xy_km(SLA, SLO)[0], to_xy_km(SLA, SLO)[1]])
            _, si = ts.query([cx, cy], k=5)
            f = base_map(520)
            f.update_layout(map=dict(center=dict(lat=lat, lon=lon), zoom=7.6))
            f.add_trace(edge_trace(np.full(5, lat), np.full(5, lon), SLA[si], SLO[si], "5 nearest stations (graph links)", "#2dd4bf", 2, 0.9))
            f.add_trace(go.Scattermap(lat=local["latitude"], lon=local["longitude"], mode="markers", name="past events", text=local["place"] + " | M" + local["mag"].round(1).astype(str),
                                      hoverinfo="text", marker=dict(size=4 + 2 * (local["mag"] - 2.5).clip(lower=0), color=ACC, opacity=0.7)))
            f.add_trace(go.Scattermap(lat=SLA[si], lon=SLO[si], mode="markers", name="stations", marker=dict(size=10, color="#2dd4bf"), text=stations["station_id"].to_numpy()[si], hoverinfo="text"))
            f.add_trace(go.Scattermap(lat=[lat], lon=[lon], mode="markers", name="your event", marker=dict(size=16, color="#ef4444")))
            card("Graph context for this location", f,
                 f"In the graph this event would link to its 5 nearest stations (closest {np.hypot(*(np.array(to_xy_km(SLA[si[0]], SLO[si[0]])) - np.array([cx, cy]))):.0f} km) "
                 f"and to {len(local):,} past events nearby. Move the latitude and longitude to see the context change.", sub="Red = your event, teal = stations, orange = past events")
            f = make_subplots(rows=1, cols=2, subplot_titles=("Magnitude error (MAE, lower is better)", "Depth accuracy (higher is better)"))
            f.add_trace(go.Bar(x=["Predict the mean", "Random forest"], y=[MET["mae_base"], MET["mae"]], marker_color=["#5b8def", ACC], showlegend=False), row=1, col=1)
            f.add_trace(go.Bar(x=["Most common class", "Random forest"], y=[MET["acc_base"], MET["acc"]], marker_color=["#5b8def", ACC], showlegend=False), row=1, col=2)
            gain = 1 - MET["mae"] / MET["mae_base"]
            card("How reliable is this estimator?", style(f, 340, False, top=50),
                 f"Checked on {MET['n_test']:,} held-out future events. Magnitude error improves on the naive mean by <b>{gain:.0%}</b> and depth accuracy is "
                 f"<b>{MET['acc']:.0%}</b> vs {MET['acc_base']:.0%} for always guessing the common class. Gains are modest, "
                 "so use these as estimates of what is typical, not exact values.", sub="Trained on the first 70% of the timeline, scored on the last 15%")

    else:
        with I:
            with st.container(border=True):
                st.markdown('<div class="ct">Choose the outlook</div><div class="cs">Area, size and time window</div>', unsafe_allow_html=True)
                rn = st.selectbox("Area", regs.index.tolist())
                radius = st.slider("Radius (km)", 10, 200, 50, 5)
                H = st.select_slider("Time window (days)", [7, 30, 90, 365], value=30)
                mmin = st.slider("Minimum magnitude", 2.5, 6.0, 3.0, 0.1)
        cx, cy = to_xy_km(regs.loc[rn, "lat"], regs.loc[rn, "lon"])
        sub = df[np.hypot(ex_all - cx, ey_all - cy) <= radius]
        days_all = max((df["time"].max() - df["time"].min()).days, 1)

        def rate(d, m, days):
            return (d["mag"] >= m).sum() / days

        lam = rate(sub, mmin, days_all)
        expn = lam * H
        prob = 1 - np.exp(-expn)
        with O:
            kc = st.columns(3)
            kpi(kc[0], f"Expected M{mmin:.1f}+ events", f"{expn:.2f}", f"in the next {H} days")
            kpi(kc[1], "Chance of at least one", f"{prob:.0%}", "Poisson base rate")
            kpi(kc[2], "Typical wait between events", f"{1 / lam:.0f} days" if lam > 0 else "none seen", f"{int((sub['mag'] >= mmin).sum()):,} events in {days_all:,} days")
            st.write("")
            a, b = st.columns(2, gap="medium")
            ths = np.round(np.arange(2.5, 6.6, 0.5), 1)
            pt = [1 - np.exp(-rate(sub, t, days_all) * H) for t in ths]
            f = px.bar(x=ths.astype(str), y=pt, labels={"x": "minimum magnitude", "y": f"chance within {H} days"}, color=pt, color_continuous_scale="YlOrRd")
            f.update_yaxes(tickformat=".0%", range=[0, 1]); f.update_coloraxes(showscale=False)
            with a:
                card("Chance of at least one event, by size", style(f, 360, False),
                     f"Around <b>{esc(rn)}</b> (within {radius} km), the chance falls from {pt[0]:.0%} at M2.5+ to {pt[-1]:.0%} at M6.5+. "
                     "Larger events are rare, following the Gutenberg-Richter pattern seen on the Catalog page.", sub=f"Based on every event in the catalog, {days_all:,} days")
            th_ = np.linspace(0, 2 * np.pi, 80)
            clat = float(regs.loc[rn, "lat"]); clon = float(regs.loc[rn, "lon"])
            f = base_map(360)
            f.update_layout(map=dict(center=dict(lat=clat, lon=clon), zoom=6.3 - np.log2(max(radius, 30) / 50)))
            f.add_trace(go.Scattermap(lat=clat + radius / 111 * np.sin(th_), lon=clon + radius / (111.32 * np.cos(np.radians(clat))) * np.cos(th_), mode="lines",
                                      line=dict(color=ACC, width=2), name="area", hoverinfo="skip"))
            sm = sub[sub["mag"] >= mmin]
            f.add_trace(go.Scattermap(lat=sm["latitude"], lon=sm["longitude"], mode="markers", name=f"M{mmin:.1f}+ events", text=sm["place"] + " | M" + sm["mag"].round(1).astype(str),
                                      hoverinfo="text", marker=dict(size=4 + 2 * (sm["mag"] - 2.5).clip(lower=0), color="#ef4444", opacity=0.75)))
            with b:
                card("Area used for the outlook", f,
                     f"{len(sub):,} events of any size and <b>{len(sm):,}</b> at M{mmin:.1f}+ fall inside the circle. "
                     "Increase the radius for a stabler rate, decrease it for a more local one.", sub="Orange circle = selected area")
            tr, te = df[df["split"] == "train"], df[df["split"] == "test"]

            def in_area(d):
                x_, y_ = to_xy_km(d["latitude"], d["longitude"])
                return d[np.hypot(x_ - cx, y_ - cy) <= radius]

            lam_tr = rate(in_area(tr), mmin, max((tr["time"].max() - tr["time"].min()).days, 1))
            edges = pd.date_range(te["time"].min().normalize(), te["time"].max(), freq=f"{H}D")
            tt = in_area(te)
            tt = tt[tt["mag"] >= mmin]
            if len(edges) >= 3:
                cnt = pd.cut(tt["time"], edges).value_counts(sort=False).to_numpy()
                lab = [f"{d:%b %Y}" for d in edges[:-1]]
                exp_tr = lam_tr * H
                p_tr = 1 - np.exp(-exp_tr)
                f = go.Figure(go.Bar(x=lab, y=cnt, name="actual events", marker_color="#5b8def"))
                f.add_hline(y=exp_tr, line_color=ACC, line_dash="dash", annotation_text="expected from training rate")
                f.update_yaxes(title=f"M{mmin:.1f}+ events per {H} days")
                disp = cnt.var() / cnt.mean() if cnt.mean() > 0 else float("nan")
                st.write("")
                card("Back-test on the held-out period", style(f, 340, False),
                     f"Using only the training-period rate, the model predicted a <b>{p_tr:.0%}</b> chance of at least one event per {H}-day window. "
                     f"In reality <b>{(cnt > 0).mean():.0%}</b> of the {len(cnt)} test windows had one. "
                     + (f"Counts vary {disp:.1f}x more than a plain Poisson process would, because earthquakes cluster in sequences. " if disp == disp else "")
                     + "This is why the outlook is a base rate, not a warning.", sub="Orange dashed line = the rate learned from training years")
            else:
                st.info("The held-out period is too short for this time window. Choose a shorter window to see the back-test.")

# ================================================================== ABOUT
else:
    lead(f"About {APP_NAME}", "A graph machine-learning study of Southern California seismicity.")
    with st.container(border=True):
        st.markdown("""
**Problem.** Seismic networks produce event and station data, but classic analysis treats each event independently.
Seismic Behaviour Predictor models the region as a heterogeneous graph of stations and events (spatial proximity, event-station links and
earlier-to-later event links) and learns embeddings to estimate event characteristics, group events and relate stations.

**Tasks.** Magnitude estimation, depth-class classification, event clustering, station relationship discovery, plus the Predict section: a what-if event estimator and a regional activity outlook, both validated on held-out future data.

**Evaluation.** Chronological split (train, validation, test), several seeds and ablations. Magnitude and depth are never
inputs. Event-station edges use the nearest stations rather than a magnitude-based radius, to avoid leakage.

**Limitations.** Event-station links are a modeling assumption (the public catalog does not list which stations detected each
event). Magnitude is only weakly predictable from these inputs. The 2019 Ridgecrest sequence falls in the training period, so
test results reflect distribution shift. This is **not** an earthquake prediction system.
""")

st.markdown(f'<div class="note" style="margin-top:2.5rem;border-top:1px solid {LINE};padding-top:.8rem">'
            'Data: USGS ComCat earthquake catalog and IRIS station metadata (CI network), Southern California, 2015-2025. '
            'Retrospective analysis for research and education; not for safety decisions or earthquake forecasting.</div>',
            unsafe_allow_html=True)
