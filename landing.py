"""GridShift landing page: what it is, why it matters, and a live look at the grid."""
import pandas as pd
import streamlit as st

from charts import intensity_chart, local
from common import REGION, get_conn

DASHBOARD = "dashboard.py"
N_BEST_HOURS = 3   # how many of the cleanest upcoming hours to mark on the chart


@st.cache_data(ttl=300, show_spinner=False)
def live_data(region: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Last 8 days of hourly intensity + the latest forecast run. None if the DB isn't
    reachable or empty, so the landing page always loads."""
    try:
        with get_conn() as conn:
            cur = conn.execute("""SELECT bucket, gco2_per_kwh, renewable_pct FROM intensity_hourly
                                  WHERE region = %s AND bucket > now() - INTERVAL '8 days'
                                  ORDER BY bucket""", (region,))
            history = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
            cur = conn.execute("""SELECT time, gco2_p10 AS p10, gco2_p50 AS p50, gco2_p90 AS p90 FROM forecasts
                                  WHERE region = %s
                                    AND created_at = (SELECT max(created_at) FROM forecasts WHERE region = %s)
                                  ORDER BY time""", (region, region))
            fc = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    except Exception:  # noqa: BLE001 - landing page must never crash on DB problems
        return None
    if history.empty:
        return None
    history["bucket"] = pd.to_datetime(history["bucket"], utc=True)
    fc["time"] = pd.to_datetime(fc["time"], utc=True)
    return history, fc


# ---------------------------------------------------------------- hero
st.title("⚡ GridShift")
st.markdown("### Run energy-hungry compute when the grid is cleanest.")
st.markdown(
    "GridShift forecasts the carbon intensity of the electricity grid for the next 24 hours "
    "and tells you the best time to run flexible jobs, such as **ML training, EV charging and "
    "data-center batch work**. It shows exactly how much CO₂ you save by waiting."
)
if st.button("Schedule a job →", type="primary", key="cta-top"):
    st.switch_page(DASHBOARD)

# ---------------------------------------------------------------- live teaser
data = live_data(REGION)
if data:
    history, fc = data
    latest = history.iloc[-1]
    # Only hours that haven't happened yet count as "ahead"
    ahead = fc[fc["time"] >= pd.Timestamp.now(tz="UTC").floor("h")]
    best_hours = ahead.nsmallest(N_BEST_HOURS, "p50")

    st.markdown(f"#### Live from the {REGION} grid")
    st.caption(f"Latest reported hour: {local(latest['bucket']):%a %d %b %H:%M} "
               "(the EIA publishes grid data several hours behind real time)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Carbon intensity", f"{latest['gco2_per_kwh']:.0f} gCO₂/kWh")
    c2.metric("Renewable share", f"{latest['renewable_pct']:.0f} %", help="Solar + wind + hydro")
    if not best_hours.empty:
        best = best_hours.iloc[0]
        c3.metric("Cleanest hour ahead (forecast)", f"{local(best['time']):%a %H:%M}",
                  f"{best['p50'] - latest['gco2_per_kwh']:+.0f} gCO₂/kWh vs latest", delta_color="inverse")

    # 3 days of history (not 7) so the forecast gets a useful share of the width
    st.plotly_chart(intensity_chart(history, fc, best_hours=best_hours, history_days=3, height=340))
    if best_hours.empty:
        st.caption("The latest forecast is entirely in the past: run `python forecast.py` to refresh it.")
    else:
        st.caption(f"Blue: measured carbon intensity over the past 3 days. Orange: our 24-hour forecast with "
                   f"its 80% uncertainty band. Green dots: the {len(best_hours)} cleanest hours still ahead, "
                   "the best times to run flexible jobs.")

st.divider()

# ---------------------------------------------------------------- the problem
st.markdown("#### Why timing matters")
p1, p2 = st.columns(2)
with p1:
    st.markdown(
        "**Compute is becoming a major electricity load.** Data centres used about 415 TWh "
        "in 2024, roughly 1.5% of global electricity, and the IEA expects that to more than "
        "double by 2030 as AI grows."
    )
with p2:
    st.markdown(
        "**The grid's carbon intensity changes by the hour.** Solar ramps up at midday, wind "
        "comes and goes, and coal and oil peakers switch on for the evening rush. Most batch "
        "work doesn't care *when* it runs, so that flexibility is free emissions cuts."
    )

# ---------------------------------------------------------------- how it works
st.markdown("#### How it works")
h1, h2, h3 = st.columns(3)
# height="stretch" makes each card fill its column, so all three match the tallest one
with h1.container(border=True, height="stretch"):
    st.markdown("**1 · Measure**")
    st.caption("Hourly generation by fuel type from the EIA lands in a Tiger Data hypertable. "
               "Continuous aggregates turn it into gCO₂/kWh, and compression keeps history cheap.")
with h2.container(border=True, height="stretch"):
    st.markdown("**2 · Forecast**")
    st.caption("A gradient-boosting model predicts the next 24 hours with a calibrated 80% "
               "uncertainty band, tested against a 'same hour yesterday' baseline.")
with h3.container(border=True, height="stretch"):
    st.markdown("**3 · Shift**")
    st.caption("Tell it your job's power, duration and deadline. GridShift finds the cleanest "
               "window that still finishes in time and reports the kg of CO₂ saved.")

# ---------------------------------------------------------------- use cases
st.markdown("#### Built for flexible loads")
u1, u2, u3 = st.columns(3)
u1.markdown("🧠 **ML training**  \nGPU runs that can start a few hours later.")
u2.markdown("🚗 **EV charging**  \nCars plugged in overnight, ready by morning.")
u3.markdown("🗄️ **Data-center batch**  \nNightly ETL, backups, rendering and reports.")

st.divider()
if st.button("Open the scheduler →", type="primary", key="cta-bottom"):
    st.switch_page(DASHBOARD)
