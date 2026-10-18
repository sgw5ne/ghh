"""GridShift dashboard.  Run with:  streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st

from common import REGION, TIMEZONE, get_conn
from forecast import best_window, evaluate, forecast_next_24h, job_kg_co2, load_hourly, write_forecasts

st.set_page_config(page_title="GridShift", page_icon="⚡", layout="wide")

# Chart colors (validated categorical palette: slot 1 blue, slot 2 orange, slot 3 aqua)
ACTUAL, FORECAST, BEST = "#2a78d6", "#eb6834", "#1baf7a"

# name: (power kW, duration h)
PRESETS = {
    "ML training run (8-GPU node, ~6.5 kW)": (6.5, 6),
    "EV charging (Level 2 charger, 11 kW)": (11.0, 4),
    "Data-center batch job (250 kW cluster)": (250.0, 3),
    "Custom": (10.0, 2),
}
KG_CO2_PER_CAR_KM = 0.25   # average gasoline passenger car, ~400 g CO2/mile (US EPA)


# ---------------------------------------------------------------- database helpers
@st.cache_resource
def _conn() -> psycopg.Connection:
    # autocommit: each dashboard query stands alone, and one failed query can't
    # leave the shared connection stuck in an aborted transaction
    return get_conn(autocommit=True)


def run(sql: str, params=None):
    """Execute on the shared connection, reconnecting once if it dropped."""
    for attempt in range(2):
        try:
            return _conn().execute(sql, params)
        except psycopg.OperationalError:
            _conn.clear()
            if attempt:
                raise


def query(sql: str, params=None) -> pd.DataFrame:
    cur = run(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def local(ts):
    """UTC timestamps -> naive local time for display."""
    if isinstance(ts, pd.Series):
        return pd.to_datetime(ts, utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    return pd.Timestamp(ts).tz_convert(TIMEZONE).tz_localize(None)


def server_ms(sql: str, params, repeats: int = 3) -> float:
    """Best-of-N server-side execution time from EXPLAIN ANALYZE (excludes network latency)."""
    times = []
    for _ in range(repeats):
        plan = run("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params).fetchone()[0]
        times.append(plan[0]["Execution Time"])
    return min(times)


def refresh_forecast(region: str) -> None:
    with get_conn() as conn:
        y = load_hourly(conn, region)
        write_forecasts(conn, forecast_next_24h(y), region)


@st.cache_data(ttl=3600, show_spinner="Evaluating model on held-out data...")
def model_metrics(region: str) -> dict:
    with get_conn() as conn:
        return evaluate(load_hourly(conn, region))


# ---------------------------------------------------------------- load data
st.title("⚡ GridShift")
st.caption("Carbon-aware scheduling: run energy-hungry jobs when the grid is cleanest.")

try:
    _conn()
except Exception as e:  # noqa: BLE001 - show any connection problem in the UI
    st.error(f"Can't connect to the database: {e}\n\nCheck DATABASE_URL in your .env file.")
    st.stop()

region = st.sidebar.text_input("Grid region (EIA balancing authority)", REGION)
st.sidebar.caption(f"Times shown in {TIMEZONE}")
if st.sidebar.button("🔄 Re-train model & refresh forecast"):
    with st.spinner("Training gradient-boosting models..."):
        refresh_forecast(region)
    model_metrics.clear()

history = query("""
    SELECT bucket, gco2_per_kwh, renewable_pct
    FROM intensity_hourly
    WHERE region = %s AND bucket > now() - INTERVAL '8 days'
    ORDER BY bucket
""", (region,))
if history.empty:
    st.warning(f"No data for {region} yet. Run `python ingest.py synthetic` (or `eia`) first.")
    st.stop()

fc = query("""
    SELECT time, gco2_p10 AS p10, gco2_p50 AS p50, gco2_p90 AS p90
    FROM forecasts
    WHERE region = %s AND created_at = (SELECT max(created_at) FROM forecasts WHERE region = %s)
    ORDER BY time
""", (region, region))
if fc.empty:
    st.info("No forecast yet: generating one now...")
    refresh_forecast(region)
    st.rerun()

history["bucket"] = pd.to_datetime(history["bucket"], utc=True)
fc["time"] = pd.to_datetime(fc["time"], utc=True)
fc_indexed = fc.set_index("time")[["p10", "p50", "p90"]]

# ---------------------------------------------------------------- headline metrics
latest = history.iloc[-1]
day_ago = history[history["bucket"] == latest["bucket"] - pd.Timedelta(hours=24)]
cleanest = fc.loc[fc["p50"].idxmin()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Grid intensity now", f"{latest['gco2_per_kwh']:.0f} gCO₂/kWh",
          delta=f"{latest['gco2_per_kwh'] - day_ago['gco2_per_kwh'].iloc[0]:+.0f} vs 24h ago"
          if not day_ago.empty else None, delta_color="inverse",
          help=f"Latest complete hour: {local(latest['bucket']):%a %H:%M}")
c2.metric("Renewable share", f"{latest['renewable_pct']:.0f} %", help="Solar + wind + hydro")
c3.metric("Cleanest hour (next 24h)", f"{local(cleanest['time']):%a %H:%M}",
          f"{cleanest['p50']:.0f} gCO₂/kWh forecast", delta_color="off")
try:
    m = model_metrics(region)
    c4.metric("Forecast error (MAE)", f"{m['mae_model']:.1f} gCO₂/kWh",
              f"{m['improvement_pct']:.0f}% better than naive", delta_color="normal",
              help=f"Held-out last {m['test_hours'] // 24} days. Naive baseline = same hour yesterday "
                   f"(MAE {m['mae_baseline']:.1f}). p10-p90 band covered {m['band_coverage_pct']:.0f}% "
                   "of actual values.")
except RuntimeError as e:
    c4.metric("Forecast error (MAE)", "n/a", help=str(e))

chart_slot = st.container()   # chart drawn after the scheduler so it can show the chosen window

# ---------------------------------------------------------------- job scheduler
st.subheader("🗓️ Schedule a job")
preset = st.selectbox("Job type", list(PRESETS))
default_kw, default_h = PRESETS[preset]
col_a, col_b, col_c = st.columns(3)
# keys include the preset so switching presets resets the defaults
power_kw = col_a.number_input("Power draw (kW)", 0.1, 100_000.0, default_kw, key=f"kw-{preset}")
duration = col_b.slider("Duration (hours)", 1, 12, default_h, key=f"h-{preset}")
finish_within = col_c.slider("Must finish within (hours from now)", duration, 24, 24, key=f"dl-{preset}")
job_name = st.text_input("Job name", preset.split(" (")[0])

# Earliest start is the current hour; if the forecast is stale, anchor to its start instead
now = pd.Timestamp.now(tz="UTC").floor("h")
anchor = max(now, fc_indexed.index[0])
if (fc_indexed.index >= anchor).sum() < duration:
    st.warning("The forecast doesn't cover the coming hours (stale data?). Showing results relative "
               "to the forecast's first hour. Re-run ingest + forecast for live results.")
    anchor = fc_indexed.index[0]
deadline = anchor + pd.Timedelta(hours=finish_within)
window = best_window(fc_indexed, duration, deadline, start_from=anchor)

if window is None:
    st.error("This job doesn't fit before the deadline within the 24h forecast.")
else:
    kg_now = job_kg_co2(power_kw, duration, window["now_avg_gco2"])
    kg_best = job_kg_co2(power_kw, duration, window["avg_gco2"])
    saved = kg_now - kg_best
    pct = 100 * saved / kg_now if kg_now else 0
    wait_h = (window["start"] - window["now_start"]) / pd.Timedelta(hours=1)

    r1, r2, r3 = st.columns(3)
    r1.metric("Best start", f"{local(window['start']):%a %H:%M}",
              "start now" if wait_h == 0 else f"in {wait_h:.0f} h", delta_color="off")
    r2.metric("Avg intensity in window", f"{window['avg_gco2']:.0f} gCO₂/kWh",
              f"{window['avg_gco2'] - window['now_avg_gco2']:+.0f} vs starting now", delta_color="inverse",
              help=f"80% band for this window: {window['avg_gco2_p10']:.0f}-{window['avg_gco2_p90']:.0f}")
    r3.metric("CO₂ saved vs starting now", f"{saved:,.1f} kg", f"{pct:.0f}% less", delta_color="normal")

    st.markdown(
        f"Starting now emits **{kg_now:,.1f} kg CO₂**; shifting to the best window emits "
        f"**{kg_best:,.1f} kg**. The saving equals about **{saved / KG_CO2_PER_CAR_KM:,.0f} km** "
        f"of driving an average gasoline car.  \n"
        f"🚀 **At scale:** if 10,000 jobs like this shifted every day for a year, that's "
        f"**{saved * 10_000 * 365 / 1000:,.0f} tonnes of CO₂** avoided, with no new hardware, "
        f"just better timing."
    )

    if st.button("💾 Save this job"):
        run("""INSERT INTO jobs (name, region, power_kw, duration_hours, deadline, recommended_start,
                                 kg_co2_now, kg_co2_best)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (job_name, region, power_kw, duration, deadline.to_pydatetime(),
             window["start"].to_pydatetime(), kg_now, kg_best))
        st.success(f"Saved '{job_name}': start at {local(window['start']):%a %H:%M}.")

# ---------------------------------------------------------------- chart
with chart_slot:
    st.subheader("Carbon intensity: last 7 days + 24h forecast")
    hist = history[history["bucket"] > history["bucket"].max() - pd.Timedelta(days=7)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=local(hist["bucket"]), y=hist["gco2_per_kwh"], name="Actual",
                             line=dict(color=ACTUAL, width=2)))
    # p10-p90 band: draw p90 invisibly, then fill down to p10
    fig.add_trace(go.Scatter(x=local(fc["time"]), y=fc["p90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=local(fc["time"]), y=fc["p10"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(235,104,52,0.18)", name="80% band (p10-p90)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=local(fc["time"]), y=fc["p50"], name="Forecast",
                             line=dict(color=FORECAST, width=2, dash="dash")))
    if window is not None:
        fig.add_vrect(x0=local(window["start"]), x1=local(window["end"]), fillcolor=BEST, opacity=0.18,
                      line_width=0, annotation_text="Best window", annotation_position="top left")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title="gCO₂/kWh", legend=dict(orientation="h", y=1.08, x=0))
    fig.update_traces(hovertemplate="%{y:.0f} gCO₂/kWh")
    st.plotly_chart(fig)

# ---------------------------------------------------------------- saved jobs
st.subheader("📋 Saved jobs")
jobs = query("""
    SELECT name, power_kw, duration_hours, recommended_start, deadline,
           kg_co2_now, kg_co2_best, kg_co2_saved
    FROM jobs WHERE region = %s ORDER BY created_at DESC LIMIT 50
""", (region,))
if jobs.empty:
    st.caption("No saved jobs yet.")
else:
    st.metric("Total CO₂ saved by scheduled jobs", f"{jobs['kg_co2_saved'].sum():,.1f} kg")
    for col in ("recommended_start", "deadline"):
        jobs[col] = local(jobs[col]).dt.strftime("%a %d %b %H:%M")
    st.dataframe(jobs.round(1), hide_index=True)

# ---------------------------------------------------------------- Tiger Data panel
with st.expander("🐯 Under the hood: Tiger Data (TimescaleDB)"):
    st.markdown(
        "- **`grid_readings`** hypertable: raw hourly MWh + CO₂ per fuel, in 7-day chunks\n"
        "- **Columnar compression** segmented by `region, fuel`, auto-applied after 7 days\n"
        "- **`intensity_hourly`** continuous aggregate: gCO₂/kWh and renewable % per hour\n"
        "- **`intensity_daily`**: a *hierarchical* continuous aggregate built on the hourly one\n"
        "- **`forecasts`** hypertable for model output, plus a plain relational **`jobs`** table"
    )

    st.markdown("#### Continuous aggregate vs raw data")
    agg_sql = """SELECT time_bucket('1 day', bucket) AS day, sum(co2_kg) / sum(mwh) AS gco2
                 FROM intensity_hourly WHERE region = %s GROUP BY day ORDER BY day"""
    raw_sql = """SELECT time_bucket('1 day', time) AS day, sum(co2_kg) / sum(mwh) AS gco2
                 FROM grid_readings WHERE region = %s GROUP BY day ORDER BY day"""
    st.caption("Same question (daily average intensity over all history), asked two ways. "
               "Times are server-side execution times from EXPLAIN ANALYZE, best of 3.")
    if st.button("⏱️ Run benchmark"):
        agg_ms, raw_ms = server_ms(agg_sql, (region,)), server_ms(raw_sql, (region,))
        b1, b2, b3 = st.columns(3)
        b1.metric("Continuous aggregate", f"{agg_ms:.1f} ms")
        b2.metric("Raw hypertable", f"{raw_ms:.1f} ms")
        b3.metric("Speed-up", f"{raw_ms / agg_ms:.1f}×")
    st.code(agg_sql, language="sql")

    st.markdown("#### Compression")
    try:
        stats = query("""SELECT total_chunks, number_compressed_chunks,
                                before_compression_total_bytes AS before, after_compression_total_bytes AS after
                         FROM hypertable_compression_stats('grid_readings')""").iloc[0]
        rows = run("SELECT count(*) FROM grid_readings").fetchone()[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Raw rows", f"{rows:,}")
        k2.metric("Chunks compressed", f"{stats['number_compressed_chunks']} / {stats['total_chunks']}")
        if pd.notna(stats["before"]) and stats["after"]:
            k3.metric("Size before → after", f"{stats['before'] / 1e6:.2f} → {stats['after'] / 1e6:.2f} MB")
            k4.metric("Compression ratio", f"{stats['before'] / stats['after']:.1f}×")
        else:
            k3.caption("No compressed chunks yet (they compress once older than 7 days).")
    except psycopg.Error as e:
        st.caption(f"Compression stats unavailable: {e}")

    st.markdown("#### Daily intensity (hierarchical continuous aggregate)")
    daily = query("""SELECT day, gco2_per_kwh, carbon_free_pct FROM intensity_daily
                     WHERE region = %s ORDER BY day DESC LIMIT 30""", (region,))
    if not daily.empty:
        daily = daily.sort_values("day")
        dfig = go.Figure(go.Bar(x=local(daily["day"]), y=daily["gco2_per_kwh"], marker_color=ACTUAL,
                                customdata=daily["carbon_free_pct"],
                                hovertemplate="%{y:.0f} gCO₂/kWh<br>%{customdata:.0f}% carbon-free"
                                              "<extra></extra>"))
        dfig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="gCO₂/kWh")
        st.plotly_chart(dfig)
