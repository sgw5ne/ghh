-- GridShift schema for Tiger Data (TimescaleDB on Postgres).
-- Safe to re-run: every statement is idempotent.
-- Apply with:  python common.py   (or: psql "$DATABASE_URL" -f schema.sql)
-- Note: keep semicolons out of comments, common.py splits statements on them.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 1. Raw time series: hourly generation per fuel type, one row per (hour, region, fuel)
CREATE TABLE IF NOT EXISTS grid_readings (
    time    TIMESTAMPTZ      NOT NULL,
    region  TEXT             NOT NULL,
    fuel    TEXT             NOT NULL,   -- coal, gas, oil, nuclear, solar, wind, hydro, other
    mwh     DOUBLE PRECISION NOT NULL,   -- energy generated in that hour
    co2_kg  DOUBLE PRECISION NOT NULL,   -- mwh * emission factor
    UNIQUE (time, region, fuel)          -- lets ingest upsert, so re-runs never duplicate
);

-- Turn it into a hypertable, partitioned into 7-day chunks
SELECT create_hypertable('grid_readings', 'time',
                         chunk_time_interval => INTERVAL '7 days',
                         if_not_exists => TRUE);

-- 2. Native columnar compression, segmented by region/fuel so each compressed
--    segment holds one series (great ratios + fast per-series scans).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables
                   WHERE hypertable_name = 'grid_readings' AND compression_enabled) THEN
        ALTER TABLE grid_readings SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'region, fuel',
            timescaledb.compress_orderby   = 'time DESC'
        );
    END IF;
END
$$;

-- Compress chunks automatically once they are older than 7 days
SELECT add_compression_policy('grid_readings', INTERVAL '7 days', if_not_exists => TRUE);

-- 3. Continuous aggregate: hourly carbon intensity per region.
--    kg CO2 / MWh is numerically identical to g CO2 / kWh.
CREATE MATERIALIZED VIEW IF NOT EXISTS intensity_hourly
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket(INTERVAL '1 hour', time) AS bucket,
    region,
    sum(co2_kg) AS co2_kg,
    sum(mwh)    AS mwh,
    sum(CASE WHEN fuel IN ('solar', 'wind', 'hydro') THEN mwh ELSE 0 END) AS renewable_mwh,
    sum(CASE WHEN fuel IN ('solar', 'wind', 'hydro', 'nuclear') THEN mwh ELSE 0 END) AS carbon_free_mwh,
    sum(co2_kg) / NULLIF(sum(mwh), 0) AS gco2_per_kwh,
    100.0 * sum(CASE WHEN fuel IN ('solar', 'wind', 'hydro') THEN mwh ELSE 0 END)
          / NULLIF(sum(mwh), 0) AS renewable_pct
FROM grid_readings
GROUP BY bucket, region
WITH NO DATA;

SELECT add_continuous_aggregate_policy('intensity_hourly',
    start_offset      => INTERVAL '3 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists     => TRUE);

-- 4. Hierarchical continuous aggregate: daily intensity built ON TOP of the hourly one
--    (re-aggregates the sums, never touches raw rows).
CREATE MATERIALIZED VIEW IF NOT EXISTS intensity_daily
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket(INTERVAL '1 day', bucket) AS day,
    region,
    sum(co2_kg) AS co2_kg,
    sum(mwh)    AS mwh,
    sum(co2_kg) / NULLIF(sum(mwh), 0) AS gco2_per_kwh,
    100.0 * sum(renewable_mwh)   / NULLIF(sum(mwh), 0) AS renewable_pct,
    100.0 * sum(carbon_free_mwh) / NULLIF(sum(mwh), 0) AS carbon_free_pct
FROM intensity_hourly
GROUP BY day, region
WITH NO DATA;

SELECT add_continuous_aggregate_policy('intensity_daily',
    start_offset      => INTERVAL '7 days',
    end_offset        => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists     => TRUE);

-- 5. Forecasts hypertable: one row per (forecasted hour, model run)
CREATE TABLE IF NOT EXISTS forecasts (
    time         TIMESTAMPTZ      NOT NULL,   -- the hour being forecast
    region       TEXT             NOT NULL,
    created_at   TIMESTAMPTZ      NOT NULL,   -- identifies the model run
    model        TEXT             NOT NULL,
    gco2_p10     DOUBLE PRECISION,
    gco2_p50     DOUBLE PRECISION,
    gco2_p90     DOUBLE PRECISION,
    PRIMARY KEY (time, region, created_at)
);

SELECT create_hypertable('forecasts', 'time',
                         chunk_time_interval => INTERVAL '30 days',
                         if_not_exists => TRUE);

-- 6. Plain relational table: jobs users have scheduled
CREATE TABLE IF NOT EXISTS jobs (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    name              TEXT        NOT NULL,
    region            TEXT        NOT NULL,
    power_kw          DOUBLE PRECISION NOT NULL,
    duration_hours    INTEGER     NOT NULL,
    deadline          TIMESTAMPTZ NOT NULL,
    recommended_start TIMESTAMPTZ NOT NULL,
    kg_co2_now        DOUBLE PRECISION NOT NULL,   -- emissions if started immediately
    kg_co2_best       DOUBLE PRECISION NOT NULL,   -- emissions at the recommended time
    kg_co2_saved      DOUBLE PRECISION GENERATED ALWAYS AS (kg_co2_now - kg_co2_best) STORED
);
