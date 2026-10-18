"""Forecast grid carbon intensity (gCO2/kWh) for the next 24 hours.

Model: scikit-learn HistGradientBoostingRegressor with calendar features, a
seasonal day-length feature and lags of 24/48/168 hours. Because every lag is
>= 24h, all 24 future hours can be predicted in one shot from data we already
have (a "direct" forecast, no recursion, so errors don't compound).

Design choices (each one tested offline against the naive baseline):
- The model predicts the *correction* to "same hour yesterday" (y - lag24)
  rather than the raw level, so it copes with level shifts it never saw.
- Season is encoded as day length (hours of daylight), not day-of-year
  sin/cos: with only a few months of history the holdout falls on days of the
  year never seen in training, and trees can't extrapolate those.
- Two quantile models give the 10th/90th percentile band, which is then
  widened by conformal calibration on a held-back slice of training data so
  that it actually covers ~80% of outcomes (raw quantile trees are overconfident).

Usage:
    python forecast.py            # train on Tiger Data, evaluate, write to forecasts table
    python forecast.py --offline  # same pipeline on synthetic data, no database needed
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from common import LATITUDE, REGION, TIMEZONE

MODEL_NAME = "hgb-quantile-v1"
HORIZON = 24          # hours ahead
HOLDOUT_HOURS = 336   # last 14 days held out for evaluation
CALIB_HOURS = 336     # last 14 days of the training data calibrate the uncertainty band
FEATURES = ["hour", "dow", "is_weekend", "day_length",
            "lag24", "lag48", "lag168", "mean24_lag24"]
# Fairly strong regularisation: we only have a few thousand hourly rows
MODEL_PARAMS = dict(max_iter=200, learning_rate=0.05, max_depth=4,
                    min_samples_leaf=50, l2_regularization=1.0, random_state=0)


# ---------------------------------------------------------------- data
def hourly_intensity(readings: pd.DataFrame) -> pd.Series:
    """Pandas version of the intensity_hourly continuous aggregate (for offline use)."""
    g = readings.groupby("time")[["co2_kg", "mwh"]].sum()
    return (g["co2_kg"] / g["mwh"]).rename("gco2_per_kwh")


def load_hourly(conn, region: str = REGION) -> pd.Series:
    """Read the hourly intensity series from the continuous aggregate."""
    rows = conn.execute(
        "SELECT bucket, gco2_per_kwh FROM intensity_hourly WHERE region = %s ORDER BY bucket",
        (region,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"No data in intensity_hourly for {region}. Run ingest.py first.")
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.Series([float(r[1]) for r in rows], index=idx, name="gco2_per_kwh")


def day_length_hours(day_of_year: np.ndarray, latitude: float = LATITUDE) -> np.ndarray:
    """Hours of daylight (standard astronomical approximation)."""
    declination = np.radians(23.44) * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    cos_h = np.clip(-np.tan(np.radians(latitude)) * np.tan(declination), -1, 1)
    return 24 / np.pi * np.arccos(cos_h)


def build_features(y: pd.Series, future_hours: int = 0) -> pd.DataFrame:
    """One row per hour (history + `future_hours` beyond the end) with features and target y."""
    y = y.sort_index().asfreq("h").interpolate(limit=6)     # regular hourly grid, fill short gaps
    idx = pd.date_range(y.index[0], periods=len(y) + future_hours, freq="h")
    y = y.reindex(idx)

    local = idx.tz_convert(TIMEZONE)                          # people's habits follow local time
    X = pd.DataFrame(index=idx)
    X["hour"] = local.hour
    X["dow"] = local.dayofweek
    X["is_weekend"] = (local.dayofweek >= 5).astype(int)
    X["day_length"] = day_length_hours(local.dayofyear.to_numpy())   # season
    X["lag24"] = y.shift(24)                                  # same hour yesterday
    X["lag48"] = y.shift(48)                                  # same hour 2 days ago
    X["lag168"] = y.shift(168)                                # same hour last week
    X["mean24_lag24"] = y.shift(24).rolling(24).mean()        # average level of the latest known day
    X["y"] = y
    return X


# ---------------------------------------------------------------- model
def _quantile_model(data: pd.DataFrame, q: float) -> HistGradientBoostingRegressor:
    """Fit a model for quantile q of the correction y - lag24."""
    model = HistGradientBoostingRegressor(loss="quantile", quantile=q, **MODEL_PARAMS)
    return model.fit(data[FEATURES], data["y"] - data["lag24"])


def fit_models(train: pd.DataFrame) -> dict:
    """Median model on all training rows, plus a conformally calibrated 10-90% band."""
    models = {"p50": _quantile_model(train, 0.50)}

    # Conformalized quantile regression: fit the band without the last CALIB_HOURS,
    # measure how far real values fall outside it there, and widen it by that much.
    fit_part, calib = train.iloc[:-CALIB_HOURS], train.iloc[-CALIB_HOURS:]
    models["p10"] = _quantile_model(fit_part, 0.10)
    models["p90"] = _quantile_model(fit_part, 0.90)
    lo = models["p10"].predict(calib[FEATURES]) + calib["lag24"]
    hi = models["p90"].predict(calib[FEATURES]) + calib["lag24"]
    miss = np.maximum(lo - calib["y"], calib["y"] - hi)       # >0 when outside the band
    models["widen"] = float(np.quantile(miss, 0.80))
    return models


def predict(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    base = X["lag24"].to_numpy()
    out = pd.DataFrame({
        "p10": models["p10"].predict(X[FEATURES]) + base - models["widen"],
        "p50": models["p50"].predict(X[FEATURES]) + base,
        "p90": models["p90"].predict(X[FEATURES]) + base + models["widen"],
    }, index=X.index)
    # Independent quantile models can occasionally cross: sort each row so p10 <= p50 <= p90
    out[["p10", "p50", "p90"]] = np.sort(out[["p10", "p50", "p90"]].to_numpy(), axis=1)
    return out


def training_rows(X: pd.DataFrame) -> pd.DataFrame:
    return X.dropna(subset=["y", "lag24", "lag168"])


def evaluate(y: pd.Series, holdout_hours: int = HOLDOUT_HOURS) -> dict:
    """Train on everything except the last `holdout_hours`, then compare against the
    naive 'same hour yesterday' baseline on the held-out hours.

    This is an honest day-ahead test: every feature of a held-out hour (lags >= 24h)
    would have been known up to 24 hours before that hour."""
    data = training_rows(build_features(y))
    if len(data) < holdout_hours + CALIB_HOURS + 24 * 14:
        raise RuntimeError(f"Need more history to train and evaluate (have {len(data)} usable hours, "
                           f"want ~{holdout_hours + CALIB_HOURS + 24 * 14}). Ingest more --days.")
    train, test = data.iloc[:-holdout_hours], data.iloc[-holdout_hours:]
    pred = predict(fit_models(train), test)

    mae_model = mean_absolute_error(test["y"], pred["p50"])
    mae_naive = mean_absolute_error(test["y"], test["lag24"])
    coverage = ((test["y"] >= pred["p10"]) & (test["y"] <= pred["p90"])).mean()
    return {
        "mae_model": mae_model,
        "mae_baseline": mae_naive,
        "improvement_pct": 100 * (1 - mae_model / mae_naive),
        "band_coverage_pct": 100 * coverage,
        "train_hours": len(train),
        "test_hours": len(test),
    }


def forecast_next_24h(y: pd.Series) -> pd.DataFrame:
    """Train on all history and forecast the 24 hours after the last observation."""
    X = build_features(y, future_hours=HORIZON)
    models = fit_models(training_rows(X))
    return predict(models, X.iloc[-HORIZON:])


def write_forecasts(conn, fc: pd.DataFrame, region: str = REGION) -> None:
    created_at = pd.Timestamp.now(tz="UTC").to_pydatetime()
    rows = [(t.to_pydatetime(), region, created_at, MODEL_NAME, float(r.p10), float(r.p50), float(r.p90))
            for t, r in fc.iterrows()]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO forecasts (time, region, created_at, model, gco2_p10, gco2_p50, gco2_p90)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            rows,
        )
    conn.commit()


# ---------------------------------------------------------------- scheduling
def job_kg_co2(power_kw: float, hours: float, gco2_per_kwh: float) -> float:
    """kW * h = kWh, times g/kWh = grams, / 1000 = kg."""
    return power_kw * hours * gco2_per_kwh / 1000


def best_window(forecast: pd.DataFrame, duration_hours: int, deadline, start_from=None) -> dict | None:
    """Find the start hour with the lowest average forecast intensity such that a job of
    `duration_hours` finishes by `deadline`. Also returns the 'start now' option to compare.

    forecast: DataFrame indexed by hourly UTC timestamps with columns p10/p50/p90.
    start_from: earliest allowed start (default: first forecast hour); rounded down to the hour.
    Returns None if the job can't fit before the deadline within the forecast horizon."""
    fc = forecast.sort_index()
    if start_from is not None:
        fc = fc[fc.index >= pd.Timestamp(start_from).floor("h")]
    d = int(duration_hours)
    if d < 1 or len(fc) < d:
        return None

    # Average over the d hours *starting* at each timestamp
    window_mean = fc.rolling(d).mean().shift(-(d - 1))
    ends = fc.index + pd.Timedelta(hours=d)
    feasible = window_mean[(ends <= pd.Timestamp(deadline)) & window_mean["p50"].notna()]
    if feasible.empty:
        return None

    best = feasible["p50"].idxmin()
    now = fc.index[0]
    return {
        "start": best,
        "end": best + pd.Timedelta(hours=d),
        "avg_gco2": feasible.loc[best, "p50"],
        "avg_gco2_p10": feasible.loc[best, "p10"],
        "avg_gco2_p90": feasible.loc[best, "p90"],
        "now_start": now,
        "now_avg_gco2": window_mean.loc[now, "p50"],
    }


# ---------------------------------------------------------------- CLI
def print_report(metrics: dict, fc: pd.DataFrame) -> None:
    print(f"\nHoldout evaluation ({metrics['test_hours']} h held out, {metrics['train_hours']} h training):")
    print(f"  Model MAE                        : {metrics['mae_model']:6.1f} gCO2/kWh")
    print(f"  Baseline MAE (same hour yesterday): {metrics['mae_baseline']:6.1f} gCO2/kWh")
    print(f"  Improvement over baseline        : {metrics['improvement_pct']:6.1f} %")
    print(f"  p10-p90 band coverage            : {metrics['band_coverage_pct']:6.1f} %  (target ~80%)")
    print("\nNext 24h forecast (local time):")
    show = fc.copy()
    show.index = show.index.tz_convert(TIMEZONE).strftime("%a %H:%M")
    print(show.round(0).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="use synthetic data, no database")
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args()

    if args.offline:
        from ingest import synthetic_readings
        y = hourly_intensity(synthetic_readings(days=120, region=args.region))
    else:
        from common import get_conn
        with get_conn() as conn:
            y = load_hourly(conn, args.region)
    print(f"{len(y)} hourly points, {y.index.min()} -> {y.index.max()}")

    metrics = evaluate(y)
    fc = forecast_next_24h(y)
    print_report(metrics, fc)

    w = best_window(fc, duration_hours=4, deadline=fc.index[-1] + pd.Timedelta(hours=1))
    if w:
        saved = job_kg_co2(100, 4, w["now_avg_gco2"]) - job_kg_co2(100, 4, w["avg_gco2"])
        print(f"\nExample: a 100 kW, 4 h job is cleanest starting "
              f"{w['start'].tz_convert(TIMEZONE):%a %H:%M} ({w['avg_gco2']:.0f} vs {w['now_avg_gco2']:.0f} "
              f"gCO2/kWh now) -> saves {saved:.1f} kg CO2")

    if not args.offline:
        with get_conn() as conn:
            write_forecasts(conn, fc, args.region)
        print(f"\nWrote {len(fc)} forecast rows to the forecasts table.")


if __name__ == "__main__":
    main()
