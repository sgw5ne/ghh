# ⚡ GridShift

**Run energy-hungry compute when the grid is clean.**

GridShift forecasts the carbon intensity of the electricity grid (gCO₂/kWh) for the next 24 hours and tells you the cleanest time to run a flexible job, such as an ML training run, EV charging, or a nightly data-center batch. It also shows how many kg of CO₂ you save by waiting.

The same kilowatt-hour can be noticeably cleaner a few hours later, because solar ramps up, the wind picks up, and coal/oil peakers switch off. Most compute doesn't care *when* it runs. GridShift turns that slack into emissions cuts.

## Why it matters (the giant leap)

Data centres used about **415 TWh** of electricity in 2024, roughly 1.5% of global demand. The IEA projects this will **more than double by 2030**, driven by AI (IEA, *Energy and AI*, 2025). Building clean generation takes years, but *shifting when flexible load runs* can be deployed today in software, with no new hardware.

GridShift is a building block for that:
- **Per job:** moving a 3-hour run of a 250 kW batch cluster into the cleanest window saves kilograms of CO₂ every run (about 12 kg in the synthetic demo), and it repeats every day.
- **Per fleet:** the same API call scales to every job scheduler (Kubernetes, Slurm, Airflow), EV fleet, or smart charger.
- **Per grid:** any EIA balancing authority works; change `REGION` (MISO, ERCO, CISO, ...).

## How it works

```
EIA v2 API ──┐                         ┌─> intensity_hourly (continuous aggregate)
             ├─ ingest.py ── COPY ─> grid_readings          └─> intensity_daily (hierarchical cagg)
synthetic ───┘                (hypertable, compressed)                  │
                                                                         v
                         forecast.py (gradient boosting + quantile bands) ──> forecasts (hypertable)
                                                                         │
                         app.py (Streamlit) <── best_window() <──────────┘ ──> jobs (relational)
```

### 🐯 Tiger Data (TimescaleDB)

| Feature | Where |
|---|---|
| **Hypertable** `grid_readings` (7-day chunks): hourly MWh + CO₂ per fuel | `schema.sql` |
| **Columnar compression**, `segmentby = region, fuel`, policy compresses chunks older than 7 days | `schema.sql`, `ingest.py` |
| **Continuous aggregate** `intensity_hourly`: gCO₂/kWh = Σco₂_kg / Σmwh, renewable % (real-time enabled) | `schema.sql` |
| **Hierarchical continuous aggregate** `intensity_daily`, built *on* `intensity_hourly` | `schema.sql` |
| **Refresh policies** for both aggregates + manual refresh after ingest | `schema.sql`, `ingest.py` |
| **Forecasts hypertable** storing every model run (p10/p50/p90) | `schema.sql`, `forecast.py` |
| **Relational table** `jobs` alongside the time series (with a generated `kg_co2_saved` column) | `schema.sql`, `app.py` |
| **COPY** bulk load into a staging table + upsert, so re-ingesting never duplicates | `ingest.py` |

The dashboard's *Under the hood* panel benchmarks the same daily query on the continuous aggregate vs. the raw hypertable (server-side `EXPLAIN ANALYZE` timing) and shows live compression stats.

### 🤖 AI/ML

- **Model:** `HistGradientBoostingRegressor` with features for hour, day of week, weekend, **day length** (season), **lags of 24/48/168 h**, and the previous day's mean.
- **Direct 24h forecast:** every lag is ≥ 24 h, so all 24 future hours are predicted at once from known data. There's no recursion, so errors don't compound.
- **Predicts the correction to "same hour yesterday"** rather than the raw level, so it copes with level shifts.
- **Uncertainty bands:** 10th/90th-percentile quantile models, **conformally calibrated** on a held-back slice so the 80% band actually covers about 80% of outcomes.
- **Honest evaluation:** holds out the last 14 days and compares against the naive *same hour yesterday* baseline.

Offline results on synthetic data (`python test_offline.py`):

| Seed | Model MAE | Baseline MAE | Improvement | p10–p90 coverage |
|---|---|---|---|---|
| 42 | 10.8 gCO₂/kWh | 14.3 | **24%** | 76% |
| 7  | 10.4 gCO₂/kWh | 15.9 | **35%** | 72% |

Design choices were tested, not assumed. An earlier version used day-of-year sin/cos features and lost to the baseline: with only a few months of history, the holdout falls on calendar days never seen in training, and trees can't extrapolate. Day length plus residual targets fixed it.

### 🌱 Sustainability

For a job of `P` kW running `H` hours, emissions = `P × H × avg gCO₂/kWh ÷ 1000` kg. GridShift compares starting **now** against the **best window** before your deadline. It shows the kg of CO₂ saved, the equivalent km of driving, and what that adds up to across 10,000 jobs a year. Saved jobs accumulate a running total.

## Setup

**1. Install** (Python 3.10+):
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**2. Test offline first** (no database or API key needed):
```bash
python test_offline.py          # generator sanity checks, model vs baseline, best_window
python forecast.py --offline    # full train/evaluate/forecast on synthetic data
```

**3. Create a database.** Create a free service at [Tiger Cloud](https://console.cloud.timescale.com), then copy `.env.example` to `.env` and paste the connection string into `DATABASE_URL`.

**4. Create the schema:**
```bash
python common.py                # or: psql "$DATABASE_URL" -f schema.sql
```

**5. Load data**, using either synthetic or real data:
```bash
python ingest.py synthetic      # 120 days of realistic fake PJM data
python ingest.py eia            # 90 days of real PJM data (needs EIA_API_KEY)
```

**6. Forecast and run the dashboard:**
```bash
python forecast.py              # trains, prints holdout metrics, writes the forecasts table
streamlit run app.py
```

The dashboard's sidebar button re-trains and refreshes the forecast at any time.

## Files

| File | Purpose |
|---|---|
| `schema.sql` | Hypertables, compression, continuous aggregates, policies, jobs table |
| `common.py` | Config from `.env`, DB connection, emission factors, schema setup |
| `ingest.py` | EIA v2 API (paginated) or synthetic data → COPY → refresh aggregates → compress |
| `forecast.py` | Features, model, holdout evaluation, 24h forecast, `best_window()` |
| `app.py` | Streamlit dashboard |
| `test_offline.py` | Offline checks for the generator, model, and scheduler |

## Notes & limitations

- Emission factors are approximate direct (stack) emissions: coal 1000, oil 800, gas 430, other 400 kg/MWh, and 0 for nuclear, solar, wind, and hydro. Lifecycle factors would be a natural extension.
- This uses *average* intensity. Marginal emissions (what the next kWh actually burns) is a better signal for load shifting but needs data that EIA doesn't publish.
- EIA data typically lags real time by a few hours. The newest, partially reported hours are dropped at ingest.
