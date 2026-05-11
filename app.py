"""Streamlit dashboard for the AI-infra breakout/momentum screener."""
from __future__ import annotations

import os
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

# Bridge Streamlit Community Cloud secrets → OS env vars before any module
# that reads POLYGON_API_KEY from os.environ runs. Handles both flat
# (`POLYGON_API_KEY = "..."`) and one level of nesting (`[any]\nPOLYGON_API_KEY = "..."`).
def _find_secret(secrets_obj, target: str):
    try:
        if target in secrets_obj:
            return secrets_obj[target]
    except Exception:
        pass
    try:
        for k in list(secrets_obj.keys()):
            try:
                inner = secrets_obj[k]
                if hasattr(inner, "keys") and target in inner:
                    return inner[target]
            except Exception:
                continue
    except Exception:
        pass
    return None

try:
    _val = _find_secret(st.secrets, "POLYGON_API_KEY") if hasattr(st, "secrets") else None
    if _val:
        os.environ["POLYGON_API_KEY"] = str(_val).strip().strip('"').strip("'")
except Exception:
    pass

# Surface a helpful error in the UI if the key is still missing, instead of
# crashing inside the data layer. Lists which secret keys ARE visible so we
# can tell whether the secret was saved under the wrong name.
if not os.environ.get("POLYGON_API_KEY"):
    try:
        _keys = list(st.secrets.keys()) if hasattr(st, "secrets") else []
    except Exception as _e:
        _keys = [f"<error reading st.secrets: {type(_e).__name__}>"]
    st.error(
        "**POLYGON_API_KEY not found.** "
        f"\n\nSecrets keys currently visible in this app: `{_keys}`.\n\n"
        "Fix: in this app's **Manage app → Settings → Secrets**, paste exactly:\n"
        "```toml\n"
        'POLYGON_API_KEY = "your_polygon_key_here"\n'
        "```\n"
        "Save (the app will reboot automatically)."
    )
    st.stop()

from detector import scan, latest_bar_date
from patterns import scan_patterns
from polygon_adapter import warm_up as _polygon_warm_up

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False


_ET = ZoneInfo("America/New_York")


def _is_market_open(now: datetime | None = None) -> bool:
    """US equity regular session: Mon-Fri 9:30 AM - 4:00 PM Eastern. (Holidays not handled.)"""
    now = (now or datetime.now(_ET)).astimezone(_ET)
    if now.weekday() >= 5:  # Sat=5, Sun=6
        return False
    return dtime(9, 30) <= now.time() <= dtime(16, 0)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

st.set_page_config(page_title="AI Infra Screener", layout="wide")

# ── Category mapping for the four AI-infra theme buckets ─────────────
CATEGORIES = {
    # Semiconductors
    "NVDA": "Semis", "AMD": "Semis", "AVGO": "Semis", "TSM": "Semis",
    "ASML": "Semis", "MU": "Semis", "INTC": "Semis", "MRVL": "Semis",
    "QCOM": "Semis", "AMAT": "Semis", "LRCX": "Semis", "KLAC": "Semis",
    "ARM": "Semis", "KLIC": "Semis", "TER": "Semis", "MKSI": "Semis",
    "ENTG": "Semis", "ON": "Semis", "TXN": "Semis", "ADI": "Semis",
    "MCHP": "Semis", "SMCI": "Semis", "ALAB": "Semis", "CRDO": "Semis",
    # Hyperscalers
    "MSFT": "Hyperscalers", "AMZN": "Hyperscalers", "GOOGL": "Hyperscalers",
    "META": "Hyperscalers", "ORCL": "Hyperscalers",
    # Neoclouds
    "CRWV": "Neoclouds", "NBIS": "Neoclouds", "APLD": "Neoclouds",
    "IREN": "Neoclouds", "WULF": "Neoclouds",
    # Power AI Infra
    "VRT": "Power", "ETN": "Power", "GEV": "Power", "CEG": "Power",
    "VST": "Power", "TLN": "Power", "SMR": "Power", "OKLO": "Power",
    "LEU": "Power", "BWXT": "Power", "PWR": "Power", "FIX": "Power",
    "PH": "Power", "MPWR": "Power", "NVT": "Power",
    # Optics for AI
    "COHR": "Optics", "LITE": "Optics", "CIEN": "Optics", "FN": "Optics",
    "AAOI": "Optics", "GLW": "Optics", "ANET": "Optics", "POET": "Optics",
}

CATEGORY_ORDER = ["Semis", "Hyperscalers", "Neoclouds", "Power", "Optics"]

CATEGORY_ACCENT = {
    "Semis": "#00ff8c",        # phosphor green
    "Hyperscalers": "#00d9ff", # cyan
    "Neoclouds": "#c792ea",    # violet
    "Power": "#ffb000",        # amber
    "Optics": "#22d3ee",       # bright cyan
}

# Momentum score — higher = better setup/trend right now. Used to rank within categories.
MOMENTUM_SCORE = {
    "BREAKOUT": 100, "PRIMED": 90, "TESTING ↑": 80,
    "EXTENDED": 70, "PULLBACK": 65, "TRENDING UP": 60,
    "BOUNCE": 50, "CONSOLIDATING": 30, "NEUTRAL": 20,
    "TRENDING DOWN": 10, "AT RISK": 5, "TESTING ↓": 5,
    "WEAK": 0, "BREAKDOWN": 0, "NO DATA": -1,
}

# Status color palette — terminal phosphor + amber/cyan accents on near-black.
STATUS_COLORS = {
    "BREAKOUT":       "#00ff8c",  # phosphor green (confirmed)
    "PRIMED":         "#ffb000",  # amber
    "TESTING ↑":      "#6b7a8f",  # cool slate (unconfirmed break up — watchlist)
    "EXTENDED":       "#9aff5a",  # lime phosphor
    "CONSOLIDATING":  "#00d9ff",  # cyan
    "PULLBACK":       "#ffa726",  # amber-orange
    "TRENDING UP":    "#22d3ee",  # cyan
    "BOUNCE":         "#c792ea",  # soft violet
    "NEUTRAL":        "#4a5568",  # muted slate
    "TRENDING DOWN":  "#ff6b6b",  # rose
    "AT RISK":        "#ff8a3d",  # orange
    "TESTING ↓":      "#6b7a8f",  # cool slate
    "WEAK":           "#ef4444",  # red
    "BREAKDOWN":      "#ff3b3b",  # deep red
}

st.markdown(
    """
    <style>
    /* ── Bloomberg-terminal aesthetic ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-0: #050608;
        --bg-1: #0c0e12;
        --bg-2: #14171c;
        --border: rgba(255,255,255,0.06);
        --border-strong: rgba(255,255,255,0.12);
        --text-0: #e6e8eb;
        --text-1: #9098a3;
        --text-2: #5b6470;
        --amber: #ffb000;
        --cyan: #00d9ff;
        --phosphor: #00ff8c;
        --warning: #ff3b3b;
        --mono: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
        --sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp { background: var(--bg-0); }
    .block-container { padding-top: 3.25rem !important; padding-bottom: 2rem; max-width: 1500px; }
    /* Make Streamlit's top toolbar blend into the page so the title isn't clipped */
    [data-testid="stHeader"] { background: var(--bg-0); height: 2.5rem; }
    [data-testid="stToolbar"] { right: 1rem; }
    body, .stApp, p, span, div, label { color: var(--text-0); font-family: var(--sans); }

    /* All headings — uppercase mono, letter-spaced */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--mono) !important;
        font-weight: 500 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase;
        color: var(--text-0);
    }
    h1 { font-size: 1.6rem !important; font-weight: 600 !important; letter-spacing: 0.18em !important; }
    h3 { font-size: 0.85rem !important; color: var(--text-1) !important; margin-top: 0.5rem !important; }
    h4 { font-size: 0.75rem !important; color: var(--text-2) !important; }

    /* Captions */
    [data-testid="stCaptionContainer"], .caption {
        font-family: var(--mono);
        font-size: 0.72rem !important;
        color: var(--text-2);
        letter-spacing: 0.05em;
    }

    /* Metrics — terminal cards */
    [data-testid="stMetric"] {
        background: var(--bg-1);
        border: 1px solid var(--border);
        border-left: 2px solid var(--amber);
        padding: 14px 16px !important;
        border-radius: 2px;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--mono) !important;
        font-size: 1.75rem !important;
        font-weight: 500 !important;
        color: var(--text-0) !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--mono) !important;
        font-size: 0.65rem !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-2) !important;
        font-weight: 500 !important;
    }

    /* Buttons */
    .stButton button {
        font-family: var(--mono) !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        background: var(--bg-2) !important;
        border: 1px solid var(--border-strong) !important;
        color: var(--text-0) !important;
        border-radius: 2px !important;
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        border-color: var(--amber) !important;
        color: var(--amber) !important;
    }
    .stButton button[kind="primary"] {
        background: var(--amber) !important;
        color: var(--bg-0) !important;
        border-color: var(--amber) !important;
        font-weight: 600 !important;
    }
    .stButton button[kind="primary"]:hover {
        background: #ffc933 !important;
        color: var(--bg-0) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-1);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] textarea, [data-testid="stSidebar"] input {
        font-family: var(--mono) !important;
        font-size: 0.78rem !important;
        background: var(--bg-0) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-0) !important;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 2px;
    }
    .stDataFrame, [data-testid="stDataFrame"] * {
        font-family: var(--mono) !important;
    }

    /* Dividers — thin amber accent */
    hr { border: none; height: 1px; background: var(--border); margin: 1.5rem 0 !important; }

    /* Status badges (used in legend) */
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        font-family: var(--mono);
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--bg-0);
        border-radius: 1px;
        margin-right: 4px;
    }

    /* Custom header with accent rule */
    .terminal-header {
        font-family: var(--mono);
        font-size: 1.7rem;
        font-weight: 600;
        letter-spacing: 0.22em;
        color: var(--text-0);
        text-transform: uppercase;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--amber);
        margin-bottom: 0.4rem;
    }
    .terminal-sub {
        font-family: var(--mono);
        font-size: 0.72rem;
        color: var(--text-2);
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* Section labels — small uppercase with accent bar */
    .section-label {
        font-family: var(--mono);
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--text-1);
        padding: 12px 0 8px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 12px;
    }
    .section-label::before {
        content: "";
        display: inline-block;
        width: 4px;
        height: 10px;
        background: var(--accent, var(--amber));
        margin-right: 10px;
        vertical-align: middle;
    }

    /* Selectbox + multiselect */
    [data-baseweb="select"] {
        font-family: var(--mono) !important;
        font-size: 0.78rem !important;
    }

    /* Plotly chart container */
    .js-plotly-plot { background: var(--bg-0) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Warming Polygon connection pool…")
def _warm_polygon_once():
    _polygon_warm_up()
    return True


_warm_polygon_once()


def run_scan(tickers: tuple[str, ...], params: dict):
    # No caching: every click of "Run scan" pulls fresh data from Polygon so the
    # Chart Patterns and candlestick sections always reflect the latest pull.
    return scan(list(tickers), params)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_universe(tickers: list[str], params: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        f.write("# Universe of tickers to scan. Edit in the sidebar of the app or here.\n\n")
        yaml.safe_dump(
            {"universe": tickers, "params": params},
            f, sort_keys=False, default_flow_style=False,
        )


def parse_tickers(raw: str) -> list[str]:
    seen, out = set(), []
    for line in raw.replace(",", "\n").splitlines():
        t = line.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


cfg = load_config()
params: dict = cfg["params"]

with st.sidebar:
    st.markdown('<div class="section-label">Scanner</div>', unsafe_allow_html=True)
    raw = st.text_area(
        "Universe — one ticker per line",
        value="\n".join(cfg["universe"]),
        height=380,
        key="universe_input",
        label_visibility="collapsed",
    )
    edited = parse_tickers(raw)
    st.caption(f"{len(edited)} TICKERS LOADED")
    c1, c2 = st.columns(2)
    with c1:
        save_clicked = st.button("SAVE", use_container_width=True)
    with c2:
        clear_clicked = st.button("CLEAR", use_container_width=True)
    if save_clicked:
        save_universe(edited, params)
        st.cache_data.clear()
        st.success("Saved.")
    if clear_clicked:
        st.cache_data.clear()
        st.success("Cache cleared.")

    st.divider()
    st.markdown('<div class="section-label">Live Monitor</div>', unsafe_allow_html=True)
    live_on = st.toggle(
        "Auto-refresh during market hours",
        value=False,
        help="When ON, the scan re-runs automatically on the chosen interval (Mon-Fri 9:30-16:00 ET).",
        disabled=not _HAS_AUTOREFRESH,
    )
    if not _HAS_AUTOREFRESH:
        st.caption("Install `streamlit-autorefresh` to enable auto-refresh.")
    refresh_minutes = st.select_slider(
        "Interval",
        options=[1, 5, 15, 30],
        value=15,
        format_func=lambda m: f"{m} min",
        disabled=not live_on,
        label_visibility="collapsed",
    )
    market_open_now = _is_market_open()
    st.caption(
        f"MARKET {'OPEN' if market_open_now else 'CLOSED'} · "
        f"NY {datetime.now(_ET).strftime('%H:%M ET')}"
    )

    st.divider()
    st.markdown('<div class="section-label">Parameters</div>', unsafe_allow_html=True)
    st.caption(
        f"DONCHIAN  {params['donchian_period']}d  \n"
        f"SQUEEZE   atr & bbw pctile < {params['atr_pct_max']}  \n"
        f"STRONG    vol≥{params['strong_vol_mult']}× & loc≥{params['strong_close_loc']}  \n"
        f"MODERATE  vol≥{params['moderate_vol_mult']}× & loc≥{params['moderate_close_loc']}  \n"
        f"MAGNITUDE move≥{params['magnitude_pct']}% & loc≥{params['magnitude_close_loc']}  \n"
        f"_Edit config.yaml to tune._"
    )

universe: list[str] = edited

st.markdown('<div class="terminal-header">AI Infra Screener</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="terminal-sub">'
    f'Semis · Hyperscalers · Neoclouds · Power · Optics &nbsp;·&nbsp; '
    f'{len(universe)} tickers · Polygon · '
    f'session {datetime.now().strftime("%H:%M:%S")}'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

run = st.button("► RUN SCAN", type="primary")

# Auto-refresh tick. The ticker runs whenever Live is on (so the page wakes up
# at market open even if it's currently closed); the *scan trigger* is gated
# on market_open_now so off-hours ticks are silent rerenders only.
auto_tick = 0
if live_on and _HAS_AUTOREFRESH:
    auto_tick = st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_tick")
last_auto_tick = st.session_state.get("last_auto_tick", -1)
auto_triggered = live_on and _HAS_AUTOREFRESH and market_open_now and auto_tick != last_auto_tick
st.session_state["last_auto_tick"] = auto_tick

scan_now = run or auto_triggered

if scan_now or "last_table" in st.session_state:
    if scan_now:
        # Snapshot the previous statuses for transition tracking
        prev = st.session_state.get("last_table")
        if prev is not None and "ticker" in prev.columns and "status" in prev.columns:
            st.session_state["prev_statuses"] = dict(zip(prev["ticker"], prev["status"]))
        else:
            st.session_state["prev_statuses"] = {}
        with st.spinner(f"Pulling {len(universe)} tickers from Polygon…"):
            table, data = run_scan(tuple(universe), params)
        st.session_state["last_table"] = table
        st.session_state["last_data"] = data
        st.session_state["last_run_at"] = datetime.now()
    table = st.session_state["last_table"]
    data = st.session_state["last_data"]

    if "close_loc" not in table.columns or "confirm_up" not in table.columns or "rs_21d" not in table.columns:
        st.cache_data.clear()
        del st.session_state["last_table"]
        del st.session_state["last_data"]
        st.warning("Detector schema changed. Click **Run scan** to refresh.")
        st.stop()

    breakouts = table[table["status"] == "BREAKOUT"]
    breakdowns = table[table["status"] == "BREAKDOWN"]
    testing_up = table[table["status"] == "TESTING ↑"]
    testing_dn = table[table["status"] == "TESTING ↓"]
    primed = table[table["status"] == "PRIMED"]
    consolidating = table[table["status"] == "CONSOLIDATING"]
    extended = table[table["status"] == "EXTENDED"]
    at_risk = table[table["status"] == "AT RISK"]
    pullbacks = table[table["status"] == "PULLBACK"]
    bounces = table[table["status"] == "BOUNCE"]
    trending_up = table[table["status"] == "TRENDING UP"]

    bar_date = latest_bar_date(data)
    now_et = datetime.now()  # local time, good enough for staleness display
    bar_str = bar_date.strftime('%a %b %-d, %Y') if bar_date is not None else 'unknown'
    age_days = (now_et.date() - bar_date.date()).days if bar_date is not None else None
    staleness = ""
    if age_days is not None:
        if age_days == 0:
            staleness = " · including today's bar"
        elif age_days == 1:
            staleness = " · last completed session"
        else:
            staleness = f" · {age_days} days old"
    auto_tag = " · AUTO" if auto_triggered else ""
    st.markdown(
        f"<div style='font-family:var(--mono); color:var(--text-2); font-size:0.72rem; "
        f"letter-spacing:0.06em; text-transform:uppercase; margin: 0.25rem 0 1rem 0;'>"
        f"DATA THROUGH <span style='color:var(--amber)'>{bar_str.upper()}</span>{staleness.upper()} · "
        f"SCAN @ {st.session_state['last_run_at'].strftime('%H:%M:%S')}{auto_tag}"
        f"</div>", unsafe_allow_html=True,
    )

    # ─── Transitions since last scan ─────────────────────────────
    prev_statuses: dict = st.session_state.get("prev_statuses", {}) or {}
    transitions = []
    if prev_statuses:
        cur_statuses = dict(zip(table["ticker"], table["status"]))
        for tkr, new_status in cur_statuses.items():
            old_status = prev_statuses.get(tkr)
            if old_status and old_status != new_status and new_status not in ("NO DATA",) and old_status not in ("NO DATA",):
                transitions.append({"ticker": tkr, "from": old_status, "to": new_status})
    transition_tickers = {t["ticker"] for t in transitions}

    if transitions:
        st.markdown(
            '<div class="section-label" style="--accent:#ffb000">'
            f'Transitions Since Last Scan <span style="color:var(--text-2); font-weight:400; '
            f'letter-spacing:0.1em; margin-left:8px">{len(transitions)} changed</span></div>',
            unsafe_allow_html=True,
        )
        trans_df = pd.DataFrame(transitions).rename(
            columns={"ticker": "Ticker", "from": "From", "to": "To"}
        )

        def _color_transition_cell(val):
            c = STATUS_COLORS.get(val)
            if not c:
                return ""
            light = val in {"BREAKOUT", "PRIMED", "EXTENDED", "PULLBACK", "TRENDING UP"}
            return f"background-color: {c}; color: {'#050608' if light else '#e6e8eb'}; font-weight: 600;"

        st.dataframe(
            trans_df.style.map(_color_transition_cell, subset=["From", "To"]),
            hide_index=True, use_container_width=True, height=min(280, 40 + 35 * len(transitions)),
        )
        st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BREAKOUTS", len(breakouts), help="Fresh cross above 50d high, confirmed by volume + close location")
    m2.metric("TESTING ↑", len(testing_up), help="Fresh cross above 50d high, NOT yet confirmed — watchlist")
    m3.metric("PRIMED", len(primed))
    m4.metric("EXTENDED", len(extended))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("TRENDING UP", len(trending_up))
    m6.metric("PULLBACK", len(pullbacks))
    m7.metric("BOUNCE", len(bounces))
    m8.metric("BREAKDOWNS", len(breakdowns))

    st.divider()

    # ─── Top Momentum by Category ─────────────────────────────────
    # Surface the strongest setup in each AI-infra bucket. Ranked by
    # momentum_score (status-derived) with rs_21d as the tiebreaker.
    cat_table = table.copy()
    cat_table["category"] = cat_table["ticker"].map(CATEGORIES).fillna("Other")
    cat_table["momentum_score"] = cat_table["status"].map(MOMENTUM_SCORE).fillna(-1)
    cat_table = cat_table[cat_table["status"] != "NO DATA"]

    st.markdown(
        '<div class="section-label" style="--accent:#00ff8c">'
        'Top Momentum by Category'
        '</div>',
        unsafe_allow_html=True,
    )

    cat_cols = st.columns(len(CATEGORY_ORDER))
    for col, cat_name in zip(cat_cols, CATEGORY_ORDER):
        accent = CATEGORY_ACCENT.get(cat_name, "#ffb000")
        sub = cat_table[cat_table["category"] == cat_name]
        if sub.empty:
            with col:
                st.markdown(
                    f'<div style="font-family:var(--mono);font-size:0.72rem;'
                    f'letter-spacing:0.1em;text-transform:uppercase;'
                    f'color:{accent};border-left:2px solid {accent};'
                    f'padding-left:8px;margin-bottom:6px">{cat_name}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<span style='color:var(--text-2);font-family:var(--mono);font-size:0.75rem'>— no data —</span>", unsafe_allow_html=True)
            continue
        # Sort: highest momentum first, then RS, then 1d %
        sort_cols = ["momentum_score"]
        if "rs_21d" in sub.columns:
            sort_cols.append("rs_21d")
        if "pct_1d" in sub.columns:
            sort_cols.append("pct_1d")
        sub = sub.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        top = sub.head(4)
        with col:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.72rem;'
                f'letter-spacing:0.1em;text-transform:uppercase;'
                f'color:{accent};border-left:2px solid {accent};'
                f'padding-left:8px;margin-bottom:6px">{cat_name} '
                f'<span style="color:var(--text-2);font-weight:400">'
                f'· {len(sub)}</span></div>',
                unsafe_allow_html=True,
            )
            rows_html = []
            for _, r in top.iterrows():
                status = r["status"]
                status_color = STATUS_COLORS.get(status, "#9098a3")
                rs_val = r.get("rs_21d")
                rs_txt = f"{rs_val:+.1f}" if pd.notna(rs_val) else "—"
                pct_val = r.get("pct_1d")
                pct_txt = f"{pct_val:+.2f}%" if pd.notna(pct_val) else ""
                pct_color = "var(--phosphor)" if pd.notna(pct_val) and pct_val >= 0 else "var(--warning)"
                rows_html.append(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;font-family:var(--mono);font-size:0.78rem;'
                    f'padding:4px 0;border-bottom:1px solid var(--border)">'
                    f'<span style="font-weight:600;color:var(--text-0);min-width:54px">{r["ticker"]}</span>'
                    f'<span style="color:{status_color};font-size:0.66rem;letter-spacing:0.06em;'
                    f'flex:1;padding:0 8px;text-align:left">{status}</span>'
                    f'<span style="color:{pct_color};min-width:60px;text-align:right">{pct_txt}</span>'
                    f'<span style="color:var(--text-1);min-width:54px;text-align:right;'
                    f'font-size:0.72rem">RS {rs_txt}</span>'
                    f'</div>'
                )
            st.markdown("".join(rows_html), unsafe_allow_html=True)

    st.divider()

    n = params["donchian_period"]
    rename_map = {
        "ticker": "Ticker",
        "status": "Status",
        "close": "Price",
        "pct_1d": "1d %",
        "donchian_high": f"{n}d High",
        "donchian_low": f"{n}d Low",
        "pct_to_breakout": "→ Breakout",
        "pct_to_breakdown": "→ Breakdown",
        "atr_contraction": "ATR ratio",
        "atr_pct": "ATR pctile",
        "bbw_pct": "BBW pctile",
        "range_10d_pct": "10d range",
        "vol_ratio": "Vol vs avg",
        "close_loc": "Close loc",
        "confirm_up": "Confirm ↑",
        "confirm_dn": "Confirm ↓",
        "slope_20d": "20d slope",
        "vs_ma50": "vs 50d MA",
        "pct_from_52w_high": "From 52w high",
        "rs_21d": "RS 21d",
    }
    col_cfg = {
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "1d %": st.column_config.NumberColumn(format="%+.2f%%", help="Today's close vs yesterday"),
        f"{n}d High": st.column_config.NumberColumn(format="$%.2f"),
        f"{n}d Low": st.column_config.NumberColumn(format="$%.2f"),
        "→ Breakout": st.column_config.NumberColumn(format="%+.2f%%", help="% move needed to break above the N-day high. Negative = already broken out."),
        "→ Breakdown": st.column_config.NumberColumn(format="%+.2f%%", help="% drop needed to break below the N-day low. Negative = already broken down."),
        "ATR pctile": st.column_config.NumberColumn(format="%.0f", help="Where today's ATR ratio sits in this stock's last 252 days. Lower = more compressed volatility."),
        "BBW pctile": st.column_config.NumberColumn(format="%.0f", help="Where today's Bollinger Band Width sits in this stock's last 252 days. Lower = tighter squeeze."),
        "Vol vs avg": st.column_config.NumberColumn(format="%.2fx"),
        "Close loc": st.column_config.NumberColumn(format="%.2f", help="Where today's close sits in today's range. 1.0 = closed at high (buyers won), 0.0 = closed at low (sellers won)."),
        "Confirm ↑": st.column_config.TextColumn(help="Up-side confirmation tier — Strong (vol≥2× AND close loc≥0.75) / Moderate (vol≥1.3× AND close loc≥0.65) / Magnitude (move≥6% AND close loc≥0.85, self-confirming on lighter vol) / Unconfirmed"),
        "Confirm ↓": st.column_config.TextColumn(help="Down-side confirmation tier — mirror image of ↑"),
        "20d slope": st.column_config.NumberColumn(format="%+.2f%%", help="% change over the last 20 trading days"),
        "vs 50d MA": st.column_config.NumberColumn(format="%+.2f%%", help="Distance from the 50-day moving average"),
        "From 52w high": st.column_config.NumberColumn(format="%+.2f%%"),
        "RS 21d": st.column_config.NumberColumn(format="%+.2f%%", help="21-day return minus SPY's 21-day return, in percentage points. Positive = outperforming the index."),
    }

    # ─── Hero callouts: today's actionable signals ─────────────────
    if len(breakouts) or len(breakdowns) or len(testing_up) or len(testing_dn):
        h1, h2 = st.columns(2)
        with h1:
            st.markdown('<div class="section-label">Breakouts Today</div>', unsafe_allow_html=True)
            if len(breakouts):
                hero = (
                    breakouts[["ticker", "close", "pct_1d", "donchian_high", "vol_ratio", "close_loc", "confirm_up"]]
                    .rename(columns=rename_map)
                )
                st.dataframe(hero, hide_index=True, use_container_width=True, column_config=col_cfg)
            else:
                st.markdown("<span style='color:var(--text-2);font-family:var(--mono);font-size:0.78rem'>— none today —</span>", unsafe_allow_html=True)
            if len(testing_up):
                st.markdown('<div class="section-label" style="margin-top:1rem">Testing ↑ (Unconfirmed)</div>', unsafe_allow_html=True)
                hero = (
                    testing_up[["ticker", "close", "pct_1d", "donchian_high", "vol_ratio", "close_loc"]]
                    .rename(columns=rename_map)
                )
                st.dataframe(hero, hide_index=True, use_container_width=True, column_config=col_cfg)
        with h2:
            st.markdown('<div class="section-label">Breakdowns Today</div>', unsafe_allow_html=True)
            if len(breakdowns):
                hero = (
                    breakdowns[["ticker", "close", "pct_1d", "donchian_low", "vol_ratio", "close_loc", "confirm_dn"]]
                    .rename(columns=rename_map)
                )
                st.dataframe(hero, hide_index=True, use_container_width=True, column_config=col_cfg)
            else:
                st.markdown("<span style='color:var(--text-2);font-family:var(--mono);font-size:0.78rem'>— none today —</span>", unsafe_allow_html=True)
            if len(testing_dn):
                st.markdown('<div class="section-label" style="margin-top:1rem">Testing ↓ (Unconfirmed)</div>', unsafe_allow_html=True)
                hero = (
                    testing_dn[["ticker", "close", "pct_1d", "donchian_low", "vol_ratio", "close_loc"]]
                    .rename(columns=rename_map)
                )
                st.dataframe(hero, hide_index=True, use_container_width=True, column_config=col_cfg)
        st.divider()

    # ─── Watch lists: closest to breakout / breakdown ─────────────
    watch = table[~table["status"].isin(["NO DATA"]) & table["pct_to_breakout"].notna()]
    near_breakout = (
        watch[watch["pct_to_breakout"] > 0]
        .nsmallest(5, "pct_to_breakout")[["ticker", "close", "donchian_high", "pct_to_breakout"]]
        .rename(columns=rename_map)
    )
    near_breakdown = (
        watch[watch["pct_to_breakdown"] > 0]
        .nsmallest(5, "pct_to_breakdown")[["ticker", "close", "donchian_low", "pct_to_breakdown"]]
        .rename(columns=rename_map)
    )

    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="section-label">Closest to Breakout</div>', unsafe_allow_html=True)
        st.dataframe(near_breakout, hide_index=True, use_container_width=True, column_config=col_cfg)
    with b2:
        st.markdown('<div class="section-label">Closest to Breakdown</div>', unsafe_allow_html=True)
        st.dataframe(near_breakdown, hide_index=True, use_container_width=True, column_config=col_cfg)

    st.divider()

    # ─── Full results ─────────────────────────────────────────────
    st.markdown('<div class="section-label">Full Scan</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        cat_filter = st.multiselect(
            "Filter by category",
            options=CATEGORY_ORDER,
            default=[],
            placeholder="All categories (default)",
            label_visibility="collapsed",
        )
    with f2:
        status_filter = st.multiselect(
            "Filter by status",
            options=list(STATUS_COLORS.keys()) + ["—"],
            default=[],
            placeholder="All statuses (default)",
            label_visibility="collapsed",
        )
    with f3:
        min_rs = st.number_input(
            "Min RS 21d (pp)",
            value=None,
            placeholder="Min RS 21d",
            step=1.0,
            help="Only show tickers whose 21d return beats SPY by at least this many percentage points. Leave blank for no filter.",
            label_visibility="collapsed",
        )

    display_cols = [
        "ticker", "status", "close", "pct_1d", "rs_21d", "close_loc", "vol_ratio",
        "vs_ma50", "slope_20d",
        "donchian_high", "donchian_low",
        "pct_to_breakout", "pct_to_breakdown",
        "atr_pct", "bbw_pct", "pct_from_52w_high",
    ]
    show = (
        table[[c for c in display_cols if c in table.columns]]
        .rename(columns=rename_map)
        .copy()
    )
    # Δ column: amber "→" mark for rows that just transitioned status.
    show.insert(0, "Δ", show["Ticker"].map(lambda t: "→" if t in transition_tickers else ""))
    show.insert(2, "Category", show["Ticker"].map(CATEGORIES).fillna("Other"))
    if cat_filter:
        show = show[show["Category"].isin(cat_filter)]
    if status_filter:
        show = show[show["Status"].isin(status_filter)]
    if min_rs is not None and "RS 21d" in show.columns:
        show = show[show["RS 21d"].fillna(-1e9) >= min_rs]

    # Dark colors get white text, light/phosphor colors get black for legibility
    _LIGHT_STATUS = {"BREAKOUT", "PRIMED", "EXTENDED", "PULLBACK", "TRENDING UP"}

    def color_status(val):
        c = STATUS_COLORS.get(val)
        if not c:
            return ""
        text_color = "#050608" if val in _LIGHT_STATUS else "#e6e8eb"
        return f"background-color: {c}; color: {text_color}; font-weight: 600; letter-spacing: 0.04em;"

    def color_pct(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #00ff8c;"
        if val < 0:
            return "color: #ff3b3b;"
        return ""

    def _color_delta(val):
        return "color: #ffb000; font-weight: 700; text-align: center;" if val else ""

    styled = show.style.map(color_status, subset=["Status"])
    if "1d %" in show.columns:
        styled = styled.map(color_pct, subset=["1d %"])
    if "RS 21d" in show.columns:
        styled = styled.map(color_pct, subset=["RS 21d"])
    if "Δ" in show.columns:
        styled = styled.map(_color_delta, subset=["Δ"])

    st.dataframe(styled, use_container_width=True, hide_index=True, column_config=col_cfg, height=560)

    st.divider()

    # ─── Chart Patterns ───────────────────────────────────────────
    st.markdown('<div class="section-label">Chart Patterns</div>', unsafe_allow_html=True)
    st.caption("Classic technical patterns detected across the universe — separate from the Donchian breakout signal.")

    pat_table, pat_results = scan_patterns(data)

    if len(pat_table) == 0:
        st.info("No patterns detected in the current scan.")
    else:
        # Direction: target above the breakout level = bullish, below = bearish.
        # Works for every detector including Symmetrical Triangle, which sets
        # target relative to whichever side it's leaning toward at detection.
        pat_table = pat_table.copy()
        pat_table["_dir"] = (pat_table["target"] > pat_table["breakout_level"]).map(
            {True: "bull", False: "bear"}
        )
        bull_tbl = pat_table[pat_table["_dir"] == "bull"].drop(columns="_dir")
        bear_tbl = pat_table[pat_table["_dir"] == "bear"].drop(columns="_dir")

        rename_pat = {
            "pattern": "Pattern", "ticker": "Ticker", "status": "Status",
            "breakout_level": "Breakout @", "target": "Target",
            "confidence": "Confidence", "notes": "Notes",
            "start_date": "Start", "end_date": "End",
        }
        pat_cfg = {
            "Breakout @": st.column_config.NumberColumn(format="$%.2f"),
            "Target": st.column_config.NumberColumn(format="$%.2f"),
            "Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "Start": st.column_config.DateColumn(format="MMM D"),
            "End": st.column_config.DateColumn(format="MMM D"),
        }

        def color_pattern_status(val):
            return {
                "Confirmed":     "background-color: #00ff8c; color: #050608; font-weight: 600; letter-spacing: 0.04em;",
                "Breaking out":  "background-color: #ffb000; color: #050608; font-weight: 600; letter-spacing: 0.04em;",
                "Forming":       "background-color: #00d9ff; color: #050608; font-weight: 600; letter-spacing: 0.04em;",
            }.get(val, "")

        def _render_pat(tbl, label, accent):
            st.markdown(
                f'<div class="section-label" style="--accent:{accent}">{label} '
                f'<span style="color:var(--text-2); font-weight:400; letter-spacing:0.1em; margin-left:8px">'
                f'{len(tbl)} detected</span></div>',
                unsafe_allow_html=True,
            )
            if len(tbl) == 0:
                st.markdown(
                    "<div style='font-family:var(--mono); color:var(--text-2); "
                    "font-size:0.78rem; padding:8px 0 16px 0'>— none in current scan —</div>",
                    unsafe_allow_html=True,
                )
                return
            show = tbl.rename(columns=rename_pat)
            st.dataframe(
                show.style.map(color_pattern_status, subset=["Status"]),
                hide_index=True, use_container_width=True, column_config=pat_cfg, height=280,
            )

        def _render_pat_detail(tbl, label, accent, key_suffix):
            if len(tbl) == 0:
                return
            st.markdown(
                f'<div class="section-label" style="--accent:{accent}; margin-top:1rem">{label}</div>',
                unsafe_allow_html=True,
            )
            options = [f"{r['ticker']} — {r['pattern']} ({r['status']})" for _, r in tbl.iterrows()]
            pick = st.selectbox(
                f"Select {key_suffix} pattern",
                options,
                key=f"pat_pick_{key_suffix}",
                label_visibility="collapsed",
            )
            sel = tbl.iloc[options.index(pick)]
            sel_result = pat_results[(sel["ticker"], sel["pattern"])]

            pdf = data[sel["ticker"]].loc[sel["start_date"]:].tail(150)
            pfig = go.Figure()
            pfig.add_trace(go.Candlestick(
                x=pdf.index, open=pdf["Open"], high=pdf["High"],
                low=pdf["Low"], close=pdf["Close"], name=sel["ticker"],
                increasing_line_color="#00ff8c", decreasing_line_color="#ff3b3b",
                increasing_fillcolor="#00ff8c", decreasing_fillcolor="#ff3b3b",
                showlegend=False,
            ))
            pfig.add_hline(
                y=sel_result.breakout_level, line_dash="dash", line_color="#ffb000",
                annotation_text=f"BRK ${sel_result.breakout_level}", annotation_position="right",
                annotation_font=dict(color="#ffb000", size=10, family="JetBrains Mono"),
            )
            pfig.add_hline(
                y=sel_result.target, line_dash="dot", line_color="#00d9ff",
                annotation_text=f"TGT ${sel_result.target}", annotation_position="right",
                annotation_font=dict(color="#00d9ff", size=10, family="JetBrains Mono"),
            )
            kp_x = [k[0] for k in sel_result.key_points]
            kp_y = [k[1] for k in sel_result.key_points]
            kp_text = [k[2] for k in sel_result.key_points]
            pfig.add_trace(go.Scatter(
                x=kp_x, y=kp_y, mode="markers+text",
                marker=dict(size=9, color=accent, line=dict(color="#050608", width=1)),
                text=kp_text, textposition="top center",
                textfont=dict(color="#e6e8eb", size=10, family="JetBrains Mono"), showlegend=False,
            ))
            pfig.add_trace(go.Scatter(
                x=kp_x, y=kp_y, mode="lines",
                line=dict(color=accent, width=1, dash="dot"), showlegend=False,
            ))
            pfig.update_layout(
                height=480, xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=80, t=40, b=10),
                template="plotly_dark",
                paper_bgcolor="#050608", plot_bgcolor="#050608",
                font=dict(family="JetBrains Mono", color="#9098a3", size=11),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
                title=dict(
                    text=f"{sel['ticker']}  ·  {sel['pattern'].upper()}  ·  {sel_result.notes}",
                    font=dict(family="JetBrains Mono", color="#e6e8eb", size=12),
                    x=0.01, xanchor="left",
                ),
            )
            st.plotly_chart(pfig, use_container_width=True, key=f"pat_chart_{key_suffix}")

        _render_pat(bull_tbl, "Bullish Patterns", "#00ff8c")
        _render_pat_detail(bull_tbl, "Bullish — Pattern Detail", "#00ff8c", "bull")
        _render_pat(bear_tbl, "Bearish Patterns", "#ff3b3b")
        _render_pat_detail(bear_tbl, "Bearish — Pattern Detail", "#ff3b3b", "bear")

    st.divider()

    # ─── Chart ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Chart</div>', unsafe_allow_html=True)
    valid = [t for t in table["ticker"] if t in data]
    if valid:
        default_ticker = (
            breakouts["ticker"].iloc[0] if len(breakouts)
            else primed["ticker"].iloc[0] if len(primed)
            else valid[0]
        )
        pick = st.selectbox("Ticker", valid, index=valid.index(default_ticker), label_visibility="collapsed")
        df = data[pick].tail(180)
        donchian_h = df["High"].rolling(params["donchian_period"]).max().shift(1)
        donchian_l = df["Low"].rolling(params["donchian_period"]).min().shift(1)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=pick,
            increasing_line_color="#00ff8c", decreasing_line_color="#ff3b3b",
            increasing_fillcolor="#00ff8c", decreasing_fillcolor="#ff3b3b",
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=donchian_h,
            line=dict(color="#ffb000", width=1.25, dash="dash"),
            name=f"{params['donchian_period']}d high",
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=donchian_l,
            line=dict(color="#00d9ff", width=1.25, dash="dash"),
            name=f"{params['donchian_period']}d low",
        ))
        fig.update_layout(
            height=560, xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_dark",
            paper_bgcolor="#050608",
            plot_bgcolor="#050608",
            font=dict(family="JetBrains Mono", color="#9098a3", size=11),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
            legend=dict(
                orientation="h", y=1.05, x=0,
                font=dict(family="JetBrains Mono", color="#9098a3", size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div class="section-label" style="margin-top:1.5rem">Status Legend</div>
        <div style='font-family:var(--mono); color:var(--text-1); font-size:0.74rem; line-height:1.9; letter-spacing:0.02em;'>
        <span class='status-badge' style='background:{STATUS_COLORS["BREAKOUT"]}'>BREAKOUT</span> fresh cross above {n}d high, confirmed (Strong vol≥{params['strong_vol_mult']}× & loc≥{params['strong_close_loc']} · Moderate vol≥{params['moderate_vol_mult']}× & loc≥{params['moderate_close_loc']} · Magnitude move≥{params['magnitude_pct']}% & loc≥{params['magnitude_close_loc']}) &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["TESTING ↑"]};color:#e6e8eb'>TESTING ↑</span> fresh cross above {n}d high but unconfirmed &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["PRIMED"]}'>PRIMED</span> in squeeze, ≤ 3% from breakout &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["EXTENDED"]}'>EXTENDED</span> already above {n}d high &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["CONSOLIDATING"]}'>CONSOLIDATING</span> in squeeze, no clear edge &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["PULLBACK"]}'>PULLBACK</span> uptrend, down 5–15% from 20d high &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["TRENDING UP"]}'>TRENDING UP</span> above 50d MA, +20d slope &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["BOUNCE"]}'>BOUNCE</span> below 50d MA but up 5%+ from 20d low &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["NEUTRAL"]};color:#e6e8eb'>NEUTRAL</span> sideways / no edge &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["TRENDING DOWN"]}'>TRENDING DOWN</span> below 50d MA, –20d slope &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["AT RISK"]}'>AT RISK</span> in squeeze, ≤ 3% from breakdown &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["TESTING ↓"]};color:#e6e8eb'>TESTING ↓</span> fresh cross below {n}d low but unconfirmed &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["WEAK"]}'>WEAK</span> already below {n}d low &nbsp;·&nbsp;
        <span class='status-badge' style='background:{STATUS_COLORS["BREAKDOWN"]}'>BREAKDOWN</span> fresh cross below {n}d low, confirmed.
        <br><span style="color:var(--text-2)">Squeeze: ATR ratio & BB-width both in bottom {params['atr_pct_max']}% of stock's own 252-day history.</span>
        </div>
        """, unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="font-family:var(--mono); color:var(--text-1); font-size:0.85rem; '
        'padding:2rem 0; letter-spacing:0.05em;">'
        'Edit the universe in the sidebar, then press <span style="color:var(--amber)">RUN SCAN</span>.'
        '</div>',
        unsafe_allow_html=True,
    )
