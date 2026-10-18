"""Shared config, database helpers and emission factors for GridShift.

Run `python common.py` to apply schema.sql to the database in DATABASE_URL.
"""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
EIA_API_KEY = os.getenv("EIA_API_KEY", "")
REGION = os.getenv("REGION", "PJM")

# Local timezone of each balancing authority (used for hour-of-day features and display)
REGION_TZ = {"PJM": "America/New_York", "MISO": "America/Chicago", "ERCO": "America/Chicago",
             "CISO": "America/Los_Angeles", "NYIS": "America/New_York", "ISNE": "America/New_York"}
TIMEZONE = REGION_TZ.get(REGION, "UTC")
# Rough latitude of each region's centre (used for the day-length seasonal feature)
REGION_LAT = {"PJM": 40.0, "MISO": 42.0, "ERCO": 31.0, "CISO": 37.0, "NYIS": 42.5, "ISNE": 43.0}
LATITUDE = REGION_LAT.get(REGION, 40.0)

# Approximate direct emission factors, kg CO2 per MWh generated.
# (kg/MWh is numerically equal to g/kWh, the usual unit for grid carbon intensity.)
EMISSION_FACTORS = {
    "coal": 1000.0,
    "gas": 430.0,
    "oil": 800.0,
    "other": 400.0,   # mostly biomass / waste / misc. in PJM: rough middle value
    "nuclear": 0.0,
    "solar": 0.0,
    "wind": 0.0,
    "hydro": 0.0,
}

# EIA fuel-type codes -> our fuel names. Storage (BAT, PS) is skipped because it
# just shifts energy that was generated elsewhere (and can be negative while charging).
EIA_FUEL_MAP = {
    "COL": "coal", "NG": "gas", "OIL": "oil", "NUC": "nuclear",
    "SUN": "solar", "SNB": "solar", "WND": "wind", "WAT": "hydro",
    "OTH": "other", "GEO": "other", "UNK": "other",
}

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def get_conn(autocommit: bool = False) -> psycopg.Connection:
    """Open a new connection to Tiger Data / TimescaleDB."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    return psycopg.connect(DATABASE_URL, autocommit=autocommit)


def split_sql(script: str) -> list[str]:
    """Split a SQL script on semicolons, ignoring those inside $$ ... $$ blocks."""
    lines = [ln for ln in script.splitlines() if not ln.strip().startswith("--")]
    text = "\n".join(lines)
    statements, current, in_dollar = [], [], False
    i = 0
    while i < len(text):
        if text.startswith("$$", i):
            in_dollar = not in_dollar
            current.append("$$")
            i += 2
            continue
        ch = text[i]
        if ch == ";" and not in_dollar:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def init_db() -> None:
    """Apply schema.sql one statement at a time (autocommit: some TimescaleDB DDL
    refuses to run inside a transaction block)."""
    with get_conn(autocommit=True) as conn:
        for stmt in split_sql(SCHEMA_FILE.read_text()):
            first_line = stmt.splitlines()[0][:70]
            print(f"  -> {first_line}")
            conn.execute(stmt)
    print("Schema ready.")


if __name__ == "__main__":
    init_db()
