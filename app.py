"""GridShift entry point.  Run with:  streamlit run app.py

Opens on the landing page; the scheduler dashboard is one click away.
"""
import streamlit as st

st.set_page_config(page_title="GridShift", page_icon="⚡", layout="wide")

page = st.navigation(
    [
        st.Page("landing.py", title="Home", icon="🏠", default=True),
        st.Page("dashboard.py", title="Scheduler", icon="⚡"),
    ],
    position="top",
)
page.run()
