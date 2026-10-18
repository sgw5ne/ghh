"""Load hourly generation-by-fuel data into the grid_readings hypertable.

Usage:
    python ingest.py synthetic [--days 120]   # realistic fake data, no API key needed
    python ingest.py eia [--days 90]          # real PJM data from the EIA v2 API
"""
import argparse
import json
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from common import EIA_API_KEY, EIA_FUEL_MAP, EMISSION_FACTORS, REGION, TIMEZONE, get_conn

EIA_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
PAGE_SIZE = 5000  # EIA v2 maximum rows per request


# ---------------------------------------------------------------- EIA mode
def get_json_with_retry(url: str, attempts: int = 4) -> dict:
    """GET a JSON page, retrying with backoff: the EIA API often times out or 5xx's."""
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=90) as resp:
                return json.load(resp)
        except (TimeoutError, URLError) as e:   # HTTPError is a subclass of URLError
            if attempt == attempts or getattr(e, "code", 500) < 500:   # don't retry 4xx (bad key etc.)
                raise
            wait = 5 * attempt
            print(f"  EIA request failed ({e}); retrying in {wait}s...")
            time.sleep(wait)


def fetch_eia(days: int, region: str = REGION) -> pd.DataFrame:
    """Download hourly generation by fuel type, following EIA's offset pagination."""
    if not EIA_API_KEY:
        raise RuntimeError("EIA_API_KEY is not set (get one free at https://www.eia.gov/opendata/). "
                           "Or run `python ingest.py synthetic`.")
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=days)
    params = [
        ("api_key", EIA_API_KEY),
        ("frequency", "hourly"),            # hourly periods are in UTC
        ("data[0]", "value"),
        ("facets[respondent][]", region),
        ("start", start.strftime("%Y-%m-%dT%H")),
        ("end", end.strftime("%Y-%m-%dT%H")),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("length", PAGE_SIZE),
    ]

    rows, offset = [], 0
    while True:
        url = EIA_URL + "?" + urlencode(params + [("offset", offset)])
        payload = get_json_with_retry(url)
        if "response" not in payload:
            raise RuntimeError(f"EIA API error: {payload}")
        page = payload["response"]["data"]
        total = int(payload["response"].get("total", 0))
        rows.extend(page)
        offset += len(page)
        print(f"  fetched {offset}/{total} rows")
        if not page or offset >= total:
            break

    if not rows:
        raise RuntimeError("EIA returned no data. Check REGION and your API key.")

    df = pd.DataFrame(rows)
    df["fuel"] = df["fueltype"].map(EIA_FUEL_MAP)          # unknown/storage codes -> NaN
    df["mwh"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["fuel", "mwh"])
    df["mwh"] = df["mwh"].clip(lower=0)
    df["time"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)

    # Several EIA codes map to one fuel (e.g. SUN + SNB -> solar), so sum them
    df = df.groupby(["time", "fuel"], as_index=False)["mwh"].sum()

    # The newest hours are often only partially reported: drop hours whose total
    # generation is far below normal so they don't look like a sudden clean grid.
    totals = df.groupby("time")["mwh"].transform("sum")
    df = df[totals > 0.5 * totals.median()]

    df["region"] = region
    df["co2_kg"] = df["mwh"] * df["fuel"].map(EMISSION_FACTORS)
    return df[["time", "region", "fuel", "mwh", "co2_kg"]]


# ---------------------------------------------------------------- synthetic mode
def synthetic_readings(days: int = 120, region: str = REGION, end=None, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic-looking PJM-style generation mix (MWh per hour per fuel).

    - demand: daily double peak (morning + bigger evening), lower weekends,
      higher in summer and winter
    - nuclear: flat baseload
    - solar: midday bell curve, scaled by season and a random daily cloudiness
    - wind: slowly drifting random walk (AR(1)), a bit stronger at night
    - coal / gas / oil: fill whatever demand is left; coal and oil peakers
      ramp up in the evening, so evenings are dirtier
    """
    rng = np.random.default_rng(seed)
    if end is None:
        end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)  # last complete hour
    times = pd.date_range(end=end, periods=days * 24, freq="h")
    local = times.tz_convert(TIMEZONE)
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    doy = local.dayofyear.to_numpy()
    n = len(times)

    def bump(center, width):  # smooth daily bump in [0, 1] around a local hour
        return np.exp(-((hour - center) ** 2) / (2 * width ** 2))

    # Demand
    shape = 1 + 0.13 * bump(18.5, 2.5) + 0.06 * bump(9, 2) - 0.12 * bump(3.5, 2.5)
    season = 1 + 0.10 * np.abs(np.cos(2 * np.pi * (doy - 15) / 365))   # peaks in Jan and Jul
    weekend = np.where(dow >= 5, 0.92, 1.0)
    demand = 95_000 * shape * season * weekend * (1 + rng.normal(0, 0.015, n))

    # Clean sources
    nuclear = 32_500 * (1 + rng.normal(0, 0.005, n))
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 13), 0, None)
    solar_season = 0.7 + 0.3 * np.cos(2 * np.pi * (doy - 172) / 365)
    cloud = np.repeat(rng.uniform(0.35, 1.0, days + 1), 24)[:n]
    solar = 9_000 * daylight * solar_season * cloud

    drift = np.zeros(n)
    for t in range(1, n):                                  # AR(1): slow random weather
        drift[t] = 0.97 * drift[t - 1] + rng.normal(0, 0.08)
    wind = np.clip(4_500 * (1 + drift) * (1 + 0.15 * bump(3, 4)), 300, 12_000)

    hydro = 1_800 + 600 * bump(19, 2)
    other = 2_000 * (1 + rng.normal(0, 0.03, n))

    # Fossil plants fill the rest; evening peakers are coal and oil
    residual = np.clip(demand - nuclear - solar - wind - hydro - other, 5_000, None)
    evening = bump(19, 2)
    coal = residual * (0.30 + 0.12 * evening)
    oil = residual * 0.03 * evening
    gas = residual - coal - oil

    wide = pd.DataFrame({"time": times, "nuclear": nuclear, "solar": solar, "wind": wind,
                         "hydro": hydro, "other": other, "coal": coal, "gas": gas, "oil": oil})
    df = wide.melt(id_vars="time", var_name="fuel", value_name="mwh")
    df["region"] = region
    df["co2_kg"] = df["mwh"] * df["fuel"].map(EMISSION_FACTORS)
    return df[["time", "region", "fuel", "mwh", "co2_kg"]].sort_values(["time", "fuel"], ignore_index=True)


# ---------------------------------------------------------------- loading
def load(df: pd.DataFrame, replace: bool = False) -> None:
    """Bulk-load with COPY into a temp table, then upsert into the hypertable
    (so re-running ingest never creates duplicates). With replace=True the region's
    existing rows are deleted first, so data from different sources never mixes."""
    rows = zip((t.to_pydatetime() for t in df["time"]), df["region"], df["fuel"],
               df["mwh"].astype(float), df["co2_kg"].astype(float))
    with get_conn() as conn, conn.cursor() as cur:
        if replace:
            # Same transaction as the load: if anything fails, the old data is kept
            cur.execute("DELETE FROM grid_readings WHERE region = ANY(%s)", (list(df["region"].unique()),))
            print(f"Deleted {cur.rowcount} existing rows for {', '.join(df['region'].unique())}")
        cur.execute("CREATE TEMP TABLE staging (LIKE grid_readings) ON COMMIT DROP")
        with cur.copy("COPY staging (time, region, fuel, mwh, co2_kg) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
        cur.execute("""
            INSERT INTO grid_readings (time, region, fuel, mwh, co2_kg)
            SELECT time, region, fuel, mwh, co2_kg FROM staging
            ON CONFLICT (time, region, fuel)
            DO UPDATE SET mwh = EXCLUDED.mwh, co2_kg = EXCLUDED.co2_kg
        """)
        print(f"Loaded {cur.rowcount} rows into grid_readings")


def refresh_and_compress() -> None:
    """Refresh both continuous aggregates now (instead of waiting for the policy)
    and compress chunks older than 7 days so compression stats show up right away."""
    # CALL refresh_continuous_aggregate cannot run inside a transaction -> autocommit
    with get_conn(autocommit=True) as conn:
        conn.execute("CALL refresh_continuous_aggregate('intensity_hourly', NULL, NULL)")
        conn.execute("CALL refresh_continuous_aggregate('intensity_daily', NULL, NULL)")
        print("Refreshed intensity_hourly and intensity_daily")
        n = conn.execute("""
            SELECT count(compress_chunk(c, if_not_compressed => true))
            FROM show_chunks('grid_readings', older_than => INTERVAL '7 days') c
        """).fetchone()[0]
        print(f"Compressed {n} chunk(s) older than 7 days")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["eia", "synthetic"])
    parser.add_argument("--days", type=int, default=None, help="history to load (default 90 eia / 120 synthetic)")
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--replace", action="store_true",
                        help="delete the region's existing rows first (e.g. switching synthetic -> eia), "
                             "so the two sources never mix")
    args = parser.parse_args()

    if args.mode == "eia":
        print(f"Fetching {args.region} from EIA...")
        df = fetch_eia(args.days or 90, args.region)
    else:
        print(f"Generating synthetic data for {args.region}...")
        df = synthetic_readings(args.days or 120, args.region)
    print(f"{len(df)} rows, {df['time'].min()} -> {df['time'].max()}")

    load(df, replace=args.replace)
    refresh_and_compress()


if __name__ == "__main__":
    main()
