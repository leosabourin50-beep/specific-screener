"""Breakout / consolidation detection on OHLCV data."""
from __future__ import annotations

import os
import pandas as pd
import yfinance as yf


def _has_polygon_key() -> bool:
    if os.environ.get("POLYGON_API_KEY"):
        return True
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            return any(line.strip().startswith("POLYGON_API_KEY=") for line in f)
    return False


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift()
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _fetch_polygon(tickers: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from polygon_adapter import fetch_data
    days = max(lookback_days, 365)
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_data, t, days): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                df = fut.result()
            except Exception:
                continue
            if not df.empty:
                out[t] = df
    return out


def _fetch_yfinance(tickers: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    period = f"{max(lookback_days, 252)}d"
    raw = yf.download(
        tickers,
        period=period,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = raw[t] if len(tickers) > 1 else raw
        except KeyError:
            continue
        df = df.dropna()
        if not df.empty:
            out[t] = df
    return out


def fetch_universe(tickers: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    if _has_polygon_key():
        return _fetch_polygon(tickers, lookback_days)
    return _fetch_yfinance(tickers, lookback_days)


def _percentile_rank(series: pd.Series, value: float) -> float:
    """Where does `value` sit in `series`? 0 = lowest, 100 = highest."""
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    return float((s <= value).sum() / len(s) * 100)


def _return_n_days(close: pd.Series, n: int) -> float | None:
    """Simple % return over the last n trading days. Returns None if insufficient bars."""
    if len(close) < n + 1:
        return None
    return float(close.iloc[-1] / close.iloc[-n - 1] - 1) * 100


def analyze(df: pd.DataFrame, params: dict, spy_ret_21d: float | None = None) -> dict:
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    tr = _true_range(high, low, close)
    atr_short = tr.rolling(params["atr_short"]).mean()
    atr_long = tr.rolling(params["atr_long"]).mean()
    atr_ratio_s = atr_short / atr_long

    bb_ma = close.rolling(params["bb_period"]).mean()
    bb_sd = close.rolling(params["bb_period"]).std()
    bbw_s = (2 * params["bb_std"] * bb_sd) / bb_ma  # Bollinger Band Width

    range_window = params["range_window"]
    recent_range = (high.rolling(range_window).max() - low.rolling(range_window).min()) / close

    donchian = high.rolling(params["donchian_period"]).max().shift(1)
    donchian_lo = low.rolling(params["donchian_period"]).min().shift(1)
    vol_avg = vol.rolling(params["volume_avg_period"]).mean()
    ma_50 = close.rolling(50).mean()
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()

    last = -1
    last_close = float(close.iloc[last])
    last_high = float(high.iloc[last])
    last_low = float(low.iloc[last])
    prev_close = float(close.iloc[-2])
    atr_ratio = float(atr_ratio_s.iloc[last])
    bbw = float(bbw_s.iloc[last])
    range_pct = float(recent_range.iloc[last])
    donchian_high = float(donchian.iloc[last])
    donchian_low = float(donchian_lo.iloc[last])
    prev_donchian_high = float(donchian.iloc[-2])
    prev_donchian_low = float(donchian_lo.iloc[-2])
    vol_ratio = float(vol.iloc[last] / vol_avg.iloc[last])
    pct_1d = (last_close / prev_close - 1) * 100

    # Close location within today's bar: 1.0 = closed at the high (buyers won),
    # 0.0 = closed at the low (sellers won), 0.5 = middle.
    bar_range = last_high - last_low
    close_loc = (last_close - last_low) / bar_range if bar_range > 0 else 0.5

    pct_window = params["pct_window"]
    atr_pct = _percentile_rank(atr_ratio_s.iloc[-pct_window:], atr_ratio)
    bbw_pct = _percentile_rank(bbw_s.iloc[-pct_window:], bbw)

    squeezing = atr_pct < params["atr_pct_max"] and bbw_pct < params["bbw_pct_max"]

    fresh_cross_up = last_close > donchian_high and prev_close <= prev_donchian_high
    fresh_cross_down = last_close < donchian_low and prev_close >= prev_donchian_low

    # Tiered confirmation: AND-logic on volume + close location.
    # Up-side: buyers won the bar (high close_loc) + heavy volume.
    strong_up = vol_ratio >= params["strong_vol_mult"] and close_loc >= params["strong_close_loc"]
    moderate_up_volume = vol_ratio >= params["moderate_vol_mult"] and close_loc >= params["moderate_close_loc"]
    # Magnitude override: a big move that closes near the high is self-confirming
    # even on lighter volume — sellers couldn't push it back off the high.
    magnitude_up = pct_1d >= params["magnitude_pct"] and close_loc >= params["magnitude_close_loc"]
    moderate_up = moderate_up_volume or magnitude_up
    # Down-side: sellers won the bar (low close_loc) + heavy volume.
    strong_dn = vol_ratio >= params["strong_vol_mult"] and (1 - close_loc) >= params["strong_close_loc"]
    moderate_dn_volume = vol_ratio >= params["moderate_vol_mult"] and (1 - close_loc) >= params["moderate_close_loc"]
    magnitude_dn = pct_1d <= -params["magnitude_pct"] and (1 - close_loc) >= params["magnitude_close_loc"]
    moderate_dn = moderate_dn_volume or magnitude_dn

    if strong_up:
        confirm_up_tier = "Strong"
    elif moderate_up_volume:
        confirm_up_tier = "Moderate"
    elif magnitude_up:
        confirm_up_tier = "Magnitude"
    else:
        confirm_up_tier = "Unconfirmed"

    if strong_dn:
        confirm_dn_tier = "Strong"
    elif moderate_dn_volume:
        confirm_dn_tier = "Moderate"
    elif magnitude_dn:
        confirm_dn_tier = "Magnitude"
    else:
        confirm_dn_tier = "Unconfirmed"

    breakout = fresh_cross_up and (strong_up or moderate_up)
    breakdown = fresh_cross_down and (strong_dn or moderate_dn)
    testing_up = fresh_cross_up and not breakout      # unconfirmed break
    testing_dn = fresh_cross_down and not breakdown

    high_52w = float(high.tail(252).max())
    pct_from_52w = last_close / high_52w - 1

    # Trend / regime metrics
    ma50 = float(ma_50.iloc[last]) if pd.notna(ma_50.iloc[last]) else last_close
    above_ma = last_close > ma50
    close_20d_ago = float(close.iloc[-21]) if len(close) >= 21 else last_close
    slope_20 = (last_close / close_20d_ago - 1) * 100 if close_20d_ago else 0
    h20 = float(high_20.iloc[last])
    l20 = float(low_20.iloc[last])
    pct_from_20d_high = (last_close / h20 - 1) * 100  # negative = below recent high
    pct_from_20d_low = (last_close / l20 - 1) * 100   # positive = above recent low

    if breakout:
        status = "BREAKOUT"
    elif breakdown:
        status = "BREAKDOWN"
    elif testing_up:
        status = "TESTING ↑"
    elif testing_dn:
        status = "TESTING ↓"
    elif squeezing and (donchian_high - last_close) / last_close < 0.03:
        status = "PRIMED"
    elif squeezing and (last_close - donchian_low) / last_close < 0.03:
        status = "AT RISK"
    elif last_close > donchian_high:
        status = "EXTENDED"
    elif last_close < donchian_low:
        status = "WEAK"
    elif squeezing:
        status = "CONSOLIDATING"
    elif above_ma and -15 < pct_from_20d_high < -5:
        status = "PULLBACK"           # constructive dip in uptrend
    elif (not above_ma) and pct_from_20d_low > 5 and not magnitude_dn and not strong_dn and pct_1d > -2:
        status = "BOUNCE"             # turning up off recent low — guard: today wasn't a strong-down crash bar
    elif above_ma and slope_20 > 2:
        status = "TRENDING UP"
    elif (not above_ma) and slope_20 < -2:
        status = "TRENDING DOWN"
    else:
        status = "NEUTRAL"

    # Relative strength vs SPY over 21 trading days (~1 month).
    # RS = stock 21d return - SPY 21d return, in percentage points.
    # Positive = outperforming the index; negative = underperforming.
    stock_ret_21d = _return_n_days(close, 21)
    if stock_ret_21d is not None and spy_ret_21d is not None:
        rs_21d = round(stock_ret_21d - spy_ret_21d, 2)
    else:
        rs_21d = None

    return {
        "close": round(last_close, 2),
        "pct_1d": round(pct_1d, 2),
        "donchian_high": round(donchian_high, 2),
        "donchian_low": round(donchian_low, 2),
        "pct_to_breakout": round((donchian_high / last_close - 1) * 100, 2),
        "pct_to_breakdown": round((last_close / donchian_low - 1) * 100, 2),
        "atr_contraction": round(atr_ratio, 2),
        "atr_pct": round(atr_pct, 1),
        "bbw_pct": round(bbw_pct, 1),
        "range_10d_pct": round(range_pct * 100, 2),
        "vol_ratio": round(vol_ratio, 2),
        "close_loc": round(close_loc, 2),
        "confirm_up": confirm_up_tier,
        "confirm_dn": confirm_dn_tier,
        "slope_20d": round(slope_20, 2),
        "vs_ma50": round((last_close / ma50 - 1) * 100, 2),
        "pct_from_52w_high": round(pct_from_52w * 100, 2),
        "rs_21d": rs_21d,
        "status": status,
        "squeezing": squeezing,
        "breakout": breakout,
        "breakdown": breakdown,
    }


def latest_bar_date(data: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    """Most recent bar date across the universe (for staleness display)."""
    if not data:
        return None
    return max(df.index[-1] for df in data.values() if len(df))


def scan(tickers: list[str], params: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    # Always fetch SPY (the RS benchmark) alongside the user's universe.
    # Tracked separately so it doesn't appear as a row unless the user explicitly added it.
    fetch_list = list(tickers)
    spy_added = "SPY" not in {t.upper() for t in fetch_list}
    if spy_added:
        fetch_list.append("SPY")

    data = fetch_universe(fetch_list, params["lookback_days"])

    spy_df = data.get("SPY")
    spy_ret_21d = _return_n_days(spy_df["Close"], 21) if spy_df is not None and len(spy_df) >= 22 else None

    rows = []
    for t in tickers:
        df = data.get(t)
        if df is None or len(df) < params["atr_long"] + 5:
            rows.append({"ticker": t, "status": "NO DATA"})
            continue
        try:
            result = analyze(df, params, spy_ret_21d=spy_ret_21d)
        except Exception as e:
            rows.append({"ticker": t, "status": f"ERROR: {e}"})
            continue
        rows.append({"ticker": t, **result})

    # SPY stays in `data` for any future chart overlay, but it's not in `rows`
    # because we iterate `tickers` (the user's universe), not `fetch_list`.

    table = pd.DataFrame(rows)
    status_order = {
        "BREAKOUT": 0, "PRIMED": 1, "TESTING ↑": 2, "EXTENDED": 3,
        "CONSOLIDATING": 4, "PULLBACK": 5, "TRENDING UP": 6, "BOUNCE": 7,
        "NEUTRAL": 8, "TRENDING DOWN": 9, "AT RISK": 10, "TESTING ↓": 11,
        "WEAK": 12, "BREAKDOWN": 13, "NO DATA": 14,
    }
    table["_order"] = table["status"].map(lambda s: status_order.get(s, 99))
    table = table.sort_values(["_order", "pct_to_breakout"], na_position="last").drop(columns="_order")
    return table.reset_index(drop=True), data
