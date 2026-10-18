"""Shared Plotly chart for the landing page and the dashboard."""
import pandas as pd
import plotly.graph_objects as go

from common import TIMEZONE

# Chart colors (validated categorical palette: slot 1 blue, slot 2 orange, slot 3 aqua)
ACTUAL, FORECAST, BEST = "#2a78d6", "#eb6834", "#1baf7a"


def local(ts):
    """UTC timestamps -> naive local time for display."""
    if isinstance(ts, pd.Series):
        return pd.to_datetime(ts, utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    return pd.Timestamp(ts).tz_convert(TIMEZONE).tz_localize(None)


def intensity_chart(history: pd.DataFrame, fc: pd.DataFrame, window: dict | None = None,
                    best_hours: pd.DataFrame | None = None, history_days: int = 7,
                    height: int = 380) -> go.Figure:
    """Recent actual intensity + the forecast with its 80% band, and a "now" line.

    history: columns bucket, gco2_per_kwh.  fc: columns time, p10, p50, p90.
    window: optional best_window() result, shaded as a band.
    best_hours: optional subset of fc rows sorted cleanest first, marked as dots;
    the cleanest one gets a text label."""
    hist = history[history["bucket"] > history["bucket"].max() - pd.Timedelta(days=history_days)]
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
    if best_hours is not None and not best_hours.empty:
        fig.add_trace(go.Scatter(
            x=local(best_hours["time"]), y=best_hours["p50"], mode="markers", name="Cleanest hours ahead",
            marker=dict(color=BEST, size=11, line=dict(color="white", width=2))))
        # Label only the single cleanest hour: the cleanest hours are often adjacent and labels would collide
        top = best_hours.iloc[0]
        fig.add_annotation(x=local(top["time"]), y=top["p50"], text=f"Cleanest: {local(top['time']):%a %H:%M}",
                           showarrow=True, arrowhead=0, ax=0, ay=40)
    if window is not None:
        fig.add_vrect(x0=local(window["start"]), x1=local(window["end"]), fillcolor=BEST, opacity=0.18,
                      line_width=0, annotation_text="Best window", annotation_position="top left")

    # "Now" divider: EIA data lags real time, so part of the forecast can already be in the past
    now = local(pd.Timestamp.now(tz="UTC"))
    fig.add_vline(x=now, line=dict(color="#898781", width=1, dash="dot"))
    fig.add_annotation(x=now, y=1, yref="paper", text="now", showarrow=False, xanchor="left",
                       yanchor="top", font=dict(color="#898781"))

    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title="gCO₂/kWh",
                      legend=dict(orientation="h", y=1.08, x=0, traceorder="normal"))
    fig.update_traces(hovertemplate="%{y:.0f} gCO₂/kWh")
    return fig
