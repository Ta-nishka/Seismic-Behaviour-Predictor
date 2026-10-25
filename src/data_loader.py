"""
SeismoNet - data_loader.py
Downloads the USGS earthquake catalog (events) and IRIS station metadata,
caches them to disk, and cleans them for graph construction.

Usage:
    python src/data_loader.py
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------- config
USGS_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_COUNT = "https://earthquake.usgs.gov/fdsnws/event/1/count"
IRIS_STATION = "https://service.iris.edu/fdsnws/station/1/query"

# Southern California bounding box
REGION = dict(minlatitude=32.0, maxlatitude=37.0,
              minlongitude=-121.0, maxlongitude=-114.0)

START, END = "2015-01-01", "2025-12-31"
MIN_MAG = 2.5
MAX_PER_REQUEST = 20_000          # USGS hard limit
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")


# ----------------------------------------------------------------- helpers
def _get(url: str, params: dict, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {e}")
            time.sleep(2 * (attempt + 1))


def _fetch_events(start: pd.Timestamp, end: pd.Timestamp, min_mag: float) -> list[pd.DataFrame]:
    """Fetch events in [start, end). Splits the window in half if it would
    exceed the 20,000-event API limit."""
    params = dict(starttime=start.strftime("%Y-%m-%dT%H:%M:%S"),
                  endtime=end.strftime("%Y-%m-%dT%H:%M:%S"),
                  minmagnitude=min_mag, **REGION)

    n = int(_get(USGS_COUNT, params).text.strip())
    if n == 0:
        return []
    if n > MAX_PER_REQUEST:
        mid = start + (end - start) / 2
        return _fetch_events(start, mid, min_mag) + _fetch_events(mid, end, min_mag)

    print(f"  {start.date()} -> {end.date()}: {n} events")
    r = _get(USGS_QUERY, {**params, "format": "csv", "orderby": "time-asc"})
    return [pd.read_csv(io.StringIO(r.text))]


# ----------------------------------------------------------------- events
def download_events(start: str = START, end: str = END, min_mag: float = MIN_MAG,
                    force: bool = False) -> pd.DataFrame:
    """Download the catalog year by year, cache to data/raw/events_raw.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / "events_raw.csv"
    if cache.exists() and not force:
        print(f"Loading cached events: {cache}")
        return pd.read_csv(cache)

    frames: list[pd.DataFrame] = []
    years = pd.date_range(start, end, freq="YS").tolist() + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    years[0] = pd.Timestamp(start)
    for a, b in zip(years[:-1], years[1:]):
        frames += _fetch_events(a, b, min_mag)

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="id")
    df.to_csv(cache, index=False)
    print(f"Saved {len(df)} raw events -> {cache}")
    return df


def clean_events(df: pd.DataFrame, min_mag: float = MIN_MAG) -> pd.DataFrame:
    """Keep real earthquakes, parse time, drop bad rows, add engineered features."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[df["type"] == "earthquake"]
    df = df.dropna(subset=["latitude", "longitude", "depth", "mag"])
    df = df[df["mag"] >= min_mag]
    df = df.drop_duplicates(subset="id").sort_values("time").reset_index(drop=True)

    # quality columns: fill missing with median so models do not break
    for col in ["gap", "rms", "dmin", "nst", "horizontalError", "depthError"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # engineered temporal feature (used by graph + baselines)
    df["dt_prev_hours"] = df["time"].diff().dt.total_seconds().div(3600).fillna(0)

    # depth class label for classification: shallow / mid / deep
    df["depth_class"] = pd.cut(df["depth"], bins=[-10, 5, 10, 1000],
                               labels=["shallow", "mid", "deep"]).astype(str)

    keep = ["id", "time", "latitude", "longitude", "depth", "mag", "magType",
            "nst", "gap", "dmin", "rms", "dt_prev_hours", "depth_class", "place"]
    return df[[c for c in keep if c in df.columns]]


# ----------------------------------------------------------------- stations
def download_stations(force: bool = False) -> pd.DataFrame:
    """Station metadata from IRIS (FDSN station service, pipe-delimited text)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / "stations_raw.csv"
    if cache.exists() and not force:
        print(f"Loading cached stations: {cache}")
        return pd.read_csv(cache)

    params = dict(net="CI", level="station", format="text",
                  minlat=REGION["minlatitude"], maxlat=REGION["maxlatitude"],
                  minlon=REGION["minlongitude"], maxlon=REGION["maxlongitude"])
    r = _get(IRIS_STATION, params)
    df = pd.read_csv(io.StringIO(r.text), sep="|")
    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
    df.to_csv(cache, index=False)
    print(f"Saved {len(df)} stations -> {cache}")
    return df


def clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Network": "network", "Station": "station",
                            "Latitude": "latitude", "Longitude": "longitude",
                            "Elevation": "elevation", "StartTime": "start_time",
                            "EndTime": "end_time"})
    df = df.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset=["network", "station"])
    df["elevation"] = df["elevation"].fillna(df["elevation"].median())
    df["station_id"] = df["network"].astype(str) + "." + df["station"].astype(str)
    cols = ["station_id", "network", "station", "latitude", "longitude", "elevation"]
    return df[cols].reset_index(drop=True)


# ----------------------------------------------------------------- main
def load_all(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    events = clean_events(download_events(force=force))
    stations = clean_stations(download_stations(force=force))
    events.to_csv(PROC_DIR / "events.csv", index=False)
    stations.to_csv(PROC_DIR / "stations.csv", index=False)
    return events, stations


if __name__ == "__main__":
    ev, st = load_all()
    print("\nEvents:", ev.shape, "| Stations:", st.shape)
    print(ev.head())
    print("\nMagnitude summary:\n", ev["mag"].describe())
    print("\nDepth classes:\n", ev["depth_class"].value_counts())