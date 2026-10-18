"""Offline sanity checks: no database or API key needed.

    python test_offline.py
"""
import numpy as np
import pandas as pd

from common import SCHEMA_FILE, split_sql
from forecast import best_window, evaluate, forecast_next_24h, hourly_intensity, job_kg_co2
from ingest import synthetic_readings


def test_synthetic_generator():
    df = synthetic_readings(days=60)
    assert set(df.columns) == {"time", "region", "fuel", "mwh", "co2_kg"}
    assert len(df) == 60 * 24 * 8, "8 fuels per hour"
    assert (df["mwh"] >= 0).all(), "no negative generation"
    assert not df.duplicated(["time", "region", "fuel"]).any()

    wide = df.pivot(index="time", columns="fuel", values="mwh")
    hour = wide.index.tz_convert("America/New_York").hour
    assert wide.loc[(hour < 5) | (hour > 21), "solar"].max() == 0, "no solar at night"
    assert wide.loc[(hour >= 11) & (hour <= 14), "solar"].mean() > 2000, "solar at midday"

    y = hourly_intensity(df)
    assert 150 < y.min() and y.max() < 700, f"intensity range looks off: {y.min():.0f}-{y.max():.0f}"
    by_hour = y.groupby(y.index.tz_convert("America/New_York").hour).mean()
    assert by_hour.loc[18:21].mean() > by_hour.loc[10:14].mean(), "evenings dirtier than midday"
    print(f"  generator ok: intensity {y.min():.0f}-{y.max():.0f} gCO2/kWh, mean {y.mean():.0f}")


def test_model_beats_baseline():
    for seed in (42, 7):
        y = hourly_intensity(synthetic_readings(days=120, seed=seed))
        m = evaluate(y)
        print(f"  seed {seed}: model MAE {m['mae_model']:.1f} vs baseline {m['mae_baseline']:.1f} "
              f"({m['improvement_pct']:.0f}% better), band coverage {m['band_coverage_pct']:.0f}%")
        assert m["mae_model"] < m["mae_baseline"], "model must beat 'same hour yesterday'"
        assert m["band_coverage_pct"] > 55, "p10-p90 band far too narrow"

    fc = forecast_next_24h(y)
    assert len(fc) == 24 and fc.index[0] == y.index[-1] + pd.Timedelta(hours=1)
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()


def test_best_window():
    idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    p50 = np.full(24, 400.0)
    p50[10:13] = 100.0                                    # clean 3-hour dip at 10:00-13:00
    fc = pd.DataFrame({"p10": p50 - 20, "p50": p50, "p90": p50 + 20}, index=idx)

    w = best_window(fc, duration_hours=3, deadline=idx[0] + pd.Timedelta(hours=24))
    assert w["start"] == idx[10] and w["avg_gco2"] == 100 and w["now_avg_gco2"] == 400

    # Deadline before the dip: best is no better than now
    w = best_window(fc, duration_hours=3, deadline=idx[0] + pd.Timedelta(hours=8))
    assert w["avg_gco2"] == 400

    # Job can't fit before deadline
    assert best_window(fc, duration_hours=5, deadline=idx[0] + pd.Timedelta(hours=4)) is None

    # 10 kW for 3 h at 400 vs 100 g/kWh -> 12 kg vs 3 kg
    assert job_kg_co2(10, 3, 400) == 12 and job_kg_co2(10, 3, 100) == 3
    print("  best_window ok")


def test_schema_splits():
    stmts = split_sql(SCHEMA_FILE.read_text())
    assert any(s.startswith("DO $$") and s.endswith("$$") for s in stmts), "DO block kept whole"
    print(f"  schema.sql splits into {len(stmts)} statements")


if __name__ == "__main__":
    for test in (test_synthetic_generator, test_model_beats_baseline, test_best_window, test_schema_splits):
        print(test.__name__)
        test()
    print("\nAll offline checks passed.")
