# ─────────────────────────────────────────────────────────────────────────────
#  NSE Breakout Scanner — Core Engine v3
#  Fixes: lower thresholds | individual fetch fallback | multi-signal ranking
#  Patterns: 20+ chart patterns + candlestick patterns + DMA signals
# ─────────────────────────────────────────────────────────────────────────────

import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
#  DATA FETCH  (batch first, individual fallback per stock)
# ──────────────────────────────────────────────────────────────

def fetch_one(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    """Fetch a single stock — used as fallback."""
    try:
        df = yf.download(
            f"{symbol}.NS",
            period=period, interval=interval,
            auto_adjust=True, progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how="all")
        return df if len(df) >= 10 else None
    except Exception:
        return None


def fetch_batch(symbols: list, period: str = "6mo", interval: str = "1d") -> dict:
    """
    Download all symbols in one call; fall back to individual fetch on failure.
    Returns dict: {symbol: DataFrame}
    """
    if not symbols:
        return {}

    ns_map = {f"{s}.NS": s for s in symbols}
    ns_list = list(ns_map.keys())

    result = {}

    # Single ticker edge-case
    if len(symbols) == 1:
        df = fetch_one(symbols[0], period, interval)
        if df is not None:
            result[symbols[0]] = df
        return result

    # Batch attempt
    try:
        raw = yf.download(
            tickers=ns_list,
            period=period, interval=interval,
            group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )

        for ns_sym, sym in ns_map.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    # Try both level orderings (yfinance changed in v0.2.45+)
                    lvl0 = raw.columns.get_level_values(0).unique().tolist()
                    lvl1 = raw.columns.get_level_values(1).unique().tolist()

                    if ns_sym in lvl0:
                        df = raw[ns_sym].copy()
                    elif ns_sym in lvl1:
                        df = raw.xs(ns_sym, axis=1, level=1).copy()
                    else:
                        # Strip suffix and try
                        short = sym
                        if short in lvl0:
                            df = raw[short].copy()
                        elif short in lvl1:
                            df = raw.xs(short, axis=1, level=1).copy()
                        else:
                            continue
                else:
                    df = raw.copy()

                df = df.dropna(how="all")
                if len(df) >= 10:
                    result[sym] = df
            except Exception:
                continue
    except Exception:
        pass

    # Individual fallback for any missing stocks
    missing = [s for s in symbols if s not in result]
    for sym in missing:
        try:
            df = fetch_one(sym, period, interval)
            if df is not None:
                result[sym] = df
            time.sleep(0.05)
        except Exception:
            continue

    return result


# ──────────────────────────────────────────────────────────────
#  INDICATORS
# ──────────────────────────────────────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _sma(s, n):
    return s.rolling(n, min_periods=max(1, n//2)).mean()

def _atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=2).mean()

def _rsi(close, n=14):
    d    = close.diff()
    gain = d.clip(lower=0).rolling(n, min_periods=2).mean()
    loss = (-d.clip(upper=0)).rolling(n, min_periods=2).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))

def _macd(close):
    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    macd   = ema12 - ema26
    signal = _ema(macd, 9)
    hist   = macd - signal
    return macd, signal, hist

def _bollinger(close, n=20, dev=2):
    ma    = close.rolling(n, min_periods=5).mean()
    sd    = close.rolling(n, min_periods=5).std()
    upper = ma + dev * sd
    lower = ma - dev * sd
    width = (upper - lower) / (ma.abs() + 1e-10)
    return upper, lower, width


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["EMA20"]  = _ema(c, 20)
    df["EMA50"]  = _ema(c, 50)
    df["SMA200"] = _sma(c, 200)
    df["ATR"]    = _atr(h, l, c, 14)
    df["RSI"]    = _rsi(c, 14)

    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = _macd(c)

    df["Vol_MA20"]  = _sma(v, 20)
    df["Vol_Ratio"] = v / (df["Vol_MA20"] + 1)

    df["BB_Upper"], df["BB_Lower"], df["BB_Width"] = _bollinger(c)

    # Support / Resistance (rolling 20-bar)
    df["Resistance20"] = h.rolling(20, min_periods=5).max()
    df["Support20"]    = l.rolling(20, min_periods=5).min()

    return df


# ──────────────────────────────────────────────────────────────
#  CONDITIONS LAYER
# ──────────────────────────────────────────────────────────────

def check_conditions(df: pd.DataFrame) -> dict:
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row

    close      = float(row["Close"])
    vol_ratio  = float(row["Vol_Ratio"]) if not pd.isna(row["Vol_Ratio"]) else 1.0
    resistance = float(row["Resistance20"]) if not pd.isna(row["Resistance20"]) else close * 1.02
    support    = float(row["Support20"])    if not pd.isna(row["Support20"])    else close * 0.98

    # 20-bar range
    recent_range_pct = (df["High"].tail(20).max() - df["Low"].tail(20).min()) / (close + 1e-10)

    bb_width     = float(row["BB_Width"])          if not pd.isna(row["BB_Width"])          else 0.05
    bb_width_avg = float(df["BB_Width"].tail(60).mean()) if len(df) >= 20 else bb_width

    # Trend slope
    n_slope = min(20, len(df) - 1)
    recent  = df["Close"].iloc[-n_slope:].values
    x       = np.arange(n_slope)
    try:
        slope = float(np.polyfit(x, recent, 1)[0])
    except Exception:
        slope = 0.0

    return {
        "PRICE_ABOVE_RESISTANCE" : close > resistance * 1.001,
        "PRICE_BELOW_SUPPORT"    : close < support * 0.999,
        "NEAR_RESISTANCE"        : resistance * 0.96 <= close <= resistance * 1.005,
        "NEAR_SUPPORT"           : support * 0.995 <= close <= support * 1.06,

        "HIGHER_HIGH_HIGHER_LOW" : slope > 0,
        "LOWER_HIGH_LOWER_LOW"   : slope < 0,

        "LOW_VOLATILITY_RANGE"   : recent_range_pct < 0.12,
        "TIGHT_CONSOLIDATION"    : recent_range_pct < 0.07,
        "VOLATILITY_SQUEEZE"     : bb_width < bb_width_avg * 0.75 if bb_width_avg > 0 else False,
        "EXPANSION_MOVE"         : bb_width > bb_width_avg * 1.40 if bb_width_avg > 0 else False,

        "VOLUME_GREATER_THAN_AVG": vol_ratio >= 1.0,
        "VOLUME_SPIKE_1_5X"      : vol_ratio >= 1.5,
        "VOLUME_SPIKE_2X"        : vol_ratio >= 2.0,
        "VOLUME_SPIKE_3X"        : vol_ratio >= 3.0,

        # DMA
        "ABOVE_EMA20"            : close > float(row["EMA20"])  if not pd.isna(row["EMA20"])  else False,
        "ABOVE_EMA50"            : close > float(row["EMA50"])  if not pd.isna(row["EMA50"])  else False,
        "ABOVE_SMA200"           : close > float(row["SMA200"]) if not pd.isna(row["SMA200"]) else False,

        # Raw values
        "_close"      : close,
        "_vol_ratio"  : vol_ratio,
        "_resistance" : resistance,
        "_support"    : support,
        "_slope"      : slope,
        "_ema20"      : float(row["EMA20"])  if not pd.isna(row["EMA20"])  else None,
        "_ema50"      : float(row["EMA50"])  if not pd.isna(row["EMA50"])  else None,
        "_sma200"     : float(row["SMA200"]) if not pd.isna(row["SMA200"]) else None,
        "_rsi"        : float(row["RSI"])    if not pd.isna(row["RSI"])    else 50.0,
        "_atr"        : float(row["ATR"])    if not pd.isna(row["ATR"])    else 0.0,
    }


# ──────────────────────────────────────────────────────────────
#  TREND
# ──────────────────────────────────────────────────────────────

def detect_trend(df: pd.DataFrame) -> tuple:
    try:
        ema20  = df["EMA20"].iloc[-1]
        ema50  = df["EMA50"].iloc[-1]
        close  = df["Close"].iloc[-1]
        slope5 = df["EMA20"].diff().iloc[-5:].mean()

        if close > ema20 and ema20 > ema50 and slope5 > 0:
            return "UPTREND", 1
        elif close < ema20 and ema20 < ema50 and slope5 < 0:
            return "DOWNTREND", -1
        else:
            return "SIDEWAYS", 0
    except Exception:
        return "SIDEWAYS", 0


# ──────────────────────────────────────────────────────────────
#  CANDLESTICK PATTERNS
# ──────────────────────────────────────────────────────────────

def detect_candlestick(df: pd.DataFrame) -> list:
    """
    Returns list of detected candlestick patterns (bullish focus).
    Each: {pattern, signal, conf, type='candle'}
    """
    found = []
    if len(df) < 3:
        return found

    c0 = df.iloc[-1]   # today
    c1 = df.iloc[-2]   # yesterday
    c2 = df.iloc[-3]   # day before

    o0, h0, l0, cl0 = float(c0["Open"]), float(c0["High"]), float(c0["Low"]), float(c0["Close"])
    o1, h1, l1, cl1 = float(c1["Open"]), float(c1["High"]), float(c1["Low"]), float(c1["Close"])
    o2, h2, l2, cl2 = float(c2["Open"]), float(c2["High"]), float(c2["Low"]), float(c2["Close"])

    body0 = abs(cl0 - o0)
    body1 = abs(cl1 - o1)
    rng0  = h0 - l0 + 1e-10
    rng1  = h1 - l1 + 1e-10

    def _add(name, conf, signal="CANDLE_SIGNAL"):
        found.append({"pattern": name, "signal": signal, "conf": conf, "type": "candle"})

    # ── Bullish Engulfing
    if cl1 < o1 and cl0 > o0 and o0 <= cl1 and cl0 >= o1 and body0 > body1 * 1.0:
        _add("BULLISH_ENGULFING", 72, "REVERSAL_SIGNAL")

    # ── Hammer
    lower_wick0 = o0 - l0 if cl0 > o0 else cl0 - l0
    upper_wick0 = h0 - cl0 if cl0 > o0 else h0 - o0
    if lower_wick0 >= body0 * 2 and upper_wick0 <= body0 * 0.3 and cl0 > o0:
        _add("HAMMER", 65, "REVERSAL_SIGNAL")

    # ── Inverted Hammer (bullish after downtrend)
    upper_wick_inv = h0 - max(o0, cl0)
    lower_wick_inv = min(o0, cl0) - l0
    if upper_wick_inv >= body0 * 2 and lower_wick_inv <= body0 * 0.3:
        _add("INVERTED_HAMMER", 58, "REVERSAL_SIGNAL")

    # ── Doji (indecision → possible reversal)
    if body0 <= rng0 * 0.08:
        _add("DOJI", 50, "REVERSAL_SIGNAL")

    # ── Morning Star (3-candle bullish reversal)
    body2 = abs(cl2 - o2)
    if cl2 < o2 and body1 <= min(body2, body0) * 0.4 and cl0 > o0 and cl0 > (o2 + cl2) / 2:
        _add("MORNING_STAR", 78, "REVERSAL_SIGNAL")

    # ── Three White Soldiers
    if (cl2 > o2 and cl1 > o1 and cl0 > o0 and
            cl1 > cl2 and cl0 > cl1 and
            o1 > o2 and o0 > o1):
        _add("THREE_WHITE_SOLDIERS", 80, "BREAKOUT")

    # ── Bullish Marubozu (strong bullish candle)
    if cl0 > o0 and body0 >= rng0 * 0.85:
        _add("BULLISH_MARUBOZU", 70, "BREAKOUT")

    # ── Piercing Line
    if cl1 < o1 and cl0 > o0 and o0 < cl1 and cl0 > (o1 + cl1) / 2:
        _add("PIERCING_LINE", 63, "REVERSAL_SIGNAL")

    # ── Bullish Harami
    if cl1 < o1 and cl0 > o0 and o0 > cl1 and cl0 < o1 and body0 < body1:
        _add("BULLISH_HARAMI", 58, "REVERSAL_SIGNAL")

    # ── Dragonfly Doji
    if (h0 - max(o0, cl0)) <= rng0 * 0.05 and (min(o0, cl0) - l0) >= rng0 * 0.65 and body0 <= rng0 * 0.1:
        _add("DRAGONFLY_DOJI", 66, "REVERSAL_SIGNAL")

    # ── Long Lower Wick (buying pressure)
    if lower_wick0 >= rng0 * 0.6 and cl0 > o0:
        _add("LONG_LOWER_WICK", 55, "REVERSAL_SIGNAL")

    # ── Shooting Star (bearish — warn)
    if upper_wick0 >= body0 * 2.5 and lower_wick_inv <= body0 * 0.2 and cl0 < o0:
        _add("SHOOTING_STAR_WARN", 55, "REVERSAL_DOWN")

    # ── Bearish Engulfing (warn)
    if cl1 > o1 and cl0 < o0 and o0 >= cl1 and cl0 <= o1 and body0 > body1:
        _add("BEARISH_ENGULFING_WARN", 60, "REVERSAL_DOWN")

    return found


# ──────────────────────────────────────────────────────────────
#  DMA SIGNALS
# ──────────────────────────────────────────────────────────────

def detect_dma_signals(df: pd.DataFrame, c: dict) -> list:
    found = []
    if len(df) < 5:
        return found

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    cl_now  = float(curr["Close"])
    cl_prev = float(prev["Close"])

    # Golden Cross: EMA20 crosses above EMA50
    e20_now  = float(curr["EMA20"]) if not pd.isna(curr["EMA20"]) else None
    e20_prev = float(prev["EMA20"]) if not pd.isna(prev["EMA20"]) else None
    e50_now  = float(curr["EMA50"]) if not pd.isna(curr["EMA50"]) else None
    e50_prev = float(prev["EMA50"]) if not pd.isna(prev["EMA50"]) else None
    s200     = float(curr["SMA200"]) if not pd.isna(curr["SMA200"]) else None

    if e20_now and e50_now and e20_prev and e50_prev:
        if e20_now > e50_now and e20_prev <= e50_prev:
            found.append({"pattern": "GOLDEN_CROSS", "signal": "BREAKOUT", "conf": 80, "type": "dma"})

        # Death Cross
        if e20_now < e50_now and e20_prev >= e50_prev:
            found.append({"pattern": "DEATH_CROSS", "signal": "BREAKDOWN", "conf": 75, "type": "dma"})

        # Price near 200 DMA
        if s200 and abs(cl_now - s200) / s200 < 0.015:
            if cl_now > s200:
                found.append({"pattern": "NEAR_200_DMA_SUPPORT", "signal": "NEAR_BREAKOUT", "conf": 62, "type": "dma"})
            else:
                found.append({"pattern": "NEAR_200_DMA_RESISTANCE", "signal": "NEAR_BREAKOUT", "conf": 55, "type": "dma"})

        # Price bouncing off 50 EMA
        if e50_now and abs(cl_now - e50_now) / e50_now < 0.01 and cl_now > e20_now:
            found.append({"pattern": "50_EMA_BOUNCE", "signal": "NEAR_BREAKOUT", "conf": 60, "type": "dma"})

        # Price bouncing off 20 EMA
        if e20_now and abs(cl_now - e20_now) / e20_now < 0.008:
            found.append({"pattern": "20_EMA_BOUNCE", "signal": "NEAR_BREAKOUT", "conf": 57, "type": "dma"})

        # All three aligned bullish
        if s200 and cl_now > e20_now > e50_now > s200:
            found.append({"pattern": "FULL_BULLISH_ALIGNMENT", "signal": "BREAKOUT", "conf": 72, "type": "dma"})

    return found


# ──────────────────────────────────────────────────────────────
#  CHART PATTERN DETECTORS
# ──────────────────────────────────────────────────────────────

def _det_horizontal_breakout(df, c):
    close = c["_close"]
    res   = c["_resistance"]
    pct   = (close - res) / (res + 1e-10) * 100

    if c["PRICE_ABOVE_RESISTANCE"]:
        high_52w = float(df["High"].rolling(min(252, len(df))).max().iloc[-2]) if len(df) > 20 else res
        pat   = "52W_HIGH_BREAKOUT" if close >= high_52w * 0.98 else "HORIZONTAL_BREAKOUT"
        conf  = 62
        if c["VOLUME_SPIKE_1_5X"]: conf += 13
        if c["VOLUME_SPIKE_2X"]:   conf +=  7
        if pat == "52W_HIGH_BREAKOUT": conf += 8
        return dict(pattern=pat, signal="BREAKOUT", conf=min(conf, 95),
                    breakout_level=res, pct_from_level=pct, type="chart")

    if c["NEAR_RESISTANCE"]:
        conf = 50
        if c["VOLUME_GREATER_THAN_AVG"]: conf += 8
        if c["HIGHER_HIGH_HIGHER_LOW"]:  conf += 5
        return dict(pattern="HORIZONTAL_RESISTANCE", signal="NEAR_BREAKOUT", conf=conf,
                    breakout_level=res, pct_from_level=pct, type="chart")
    return None


def _det_consolidation_breakout(df, c):
    if not c["LOW_VOLATILITY_RANGE"]:
        return None
    close     = c["_close"]
    cons_high = float(df["High"].tail(25).max())
    pct       = (close - cons_high) / (cons_high + 1e-10) * 100

    if close > cons_high * 1.001:
        conf = 62
        if c["VOLUME_SPIKE_1_5X"]:   conf += 14
        if c["TIGHT_CONSOLIDATION"]: conf +=  6
        if c["VOLATILITY_SQUEEZE"]:  conf +=  6
        return dict(pattern="CONSOLIDATION", signal="CONSOLIDATION_BREAKOUT",
                    conf=min(conf, 93), breakout_level=cons_high,
                    pct_from_level=pct, type="chart")

    if close >= cons_high * 0.97:
        conf = 52
        if c["VOLATILITY_SQUEEZE"]:  conf += 10
        if c["TIGHT_CONSOLIDATION"]: conf +=  5
        return dict(pattern="CONSOLIDATION", signal="NEAR_BREAKOUT", conf=conf,
                    breakout_level=cons_high, pct_from_level=pct, type="chart")
    return None


def _det_cup_and_handle(df, c):
    if len(df) < 40:
        return None
    data   = df.tail(min(90, len(df)))
    closes = data["Close"].values
    n      = len(closes)
    t1, t2 = n // 3, 2 * n // 3

    left_peak  = closes[:t1].max()
    cup_bottom = closes[t1:t2].min()
    right_sec  = closes[t2:]

    if len(right_sec) < 5:
        return None

    right_peak = right_sec[:-2].max() if len(right_sec) > 3 else right_sec.max()
    depth_pct  = (left_peak - cup_bottom) / (left_peak + 1e-10) * 100
    rim_diff   = abs(right_peak - left_peak) / (left_peak + 1e-10) * 100

    if not (10 <= depth_pct <= 60) or rim_diff > 8:
        return None

    handle      = closes[-6:]
    handle_low  = handle.min()
    pullback    = (right_peak - handle_low) / (right_peak + 1e-10) * 100
    current     = closes[-1]
    pct         = (current - right_peak) / (right_peak + 1e-10) * 100

    if 1 <= pullback <= 25 and current >= right_peak * 0.96:
        signal = "BREAKOUT" if current > right_peak else "NEAR_BREAKOUT"
        conf   = 65
        if c["VOLUME_GREATER_THAN_AVG"]: conf += 8
        if signal == "BREAKOUT":          conf += 8
        return dict(pattern="CUP_AND_HANDLE", signal=signal, conf=min(conf, 90),
                    breakout_level=round(right_peak, 2), pct_from_level=pct, type="chart")
    return None


def _det_bull_flag(df, c):
    if len(df) < 18:
        return None
    closes  = df["Close"].values
    volumes = df["Volume"].values

    pole_start = closes[-20] if len(closes) >= 20 else closes[0]
    pole_peak  = closes[-15:-5].max() if len(closes) >= 15 else closes[-5:-1].max()
    pole_gain  = (pole_peak - pole_start) / (pole_start + 1e-10) * 100

    if pole_gain < 5:
        return None

    flag      = closes[-7:]
    flag_rng  = (flag.max() - flag.min()) / (np.mean(flag) + 1e-10) * 100
    if flag_rng > 8:
        return None

    flag_vol  = volumes[-7:].mean()
    pole_vol  = volumes[-20:-7].mean() if len(volumes) >= 20 else flag_vol
    vol_dryup = flag_vol < pole_vol * 0.90

    flag_high = flag[:-1].max()
    current   = closes[-1]
    pct       = (current - flag_high) / (flag_high + 1e-10) * 100

    if current >= flag_high * 0.97:
        signal = "BREAKOUT" if current > flag_high else "NEAR_BREAKOUT"
        conf   = 64
        if c["VOLUME_SPIKE_1_5X"] and signal == "BREAKOUT": conf += 14
        if vol_dryup:                                        conf +=  5
        return dict(pattern="BULL_FLAG", signal=signal, conf=min(conf, 88),
                    breakout_level=round(flag_high, 2), pct_from_level=pct, type="chart")
    return None


def _det_double_bottom(df, c):
    if len(df) < 30:
        return None
    data  = df.tail(min(60, len(df)))
    lows  = data["Low"].values
    cls   = data["Close"].values
    n     = len(lows)

    b1     = lows[: n // 2].min()
    b1_idx = lows[: n // 2].argmin()

    between  = cls[b1_idx: n // 2 + 6]
    if len(between) < 2:
        return None
    neckline = between.max()

    b2   = lows[n // 2:].min()
    sim  = abs(b1 - b2) / (b1 + 1e-10) * 100
    depth= (neckline - b1) / (neckline + 1e-10) * 100

    if sim > 6 or depth < 6:
        return None

    current = cls[-1]
    pct     = (current - neckline) / (neckline + 1e-10) * 100

    if current >= neckline * 0.96:
        signal = "BREAKOUT" if current > neckline else "NEAR_BREAKOUT"
        conf   = 63
        if c["VOLUME_SPIKE_1_5X"] and signal == "BREAKOUT": conf += 14
        if c["HIGHER_HIGH_HIGHER_LOW"]:                      conf +=  5
        return dict(pattern="DOUBLE_BOTTOM", signal=signal, conf=min(conf, 88),
                    breakout_level=round(neckline, 2), pct_from_level=pct, type="chart")
    return None


def _det_double_top(df, c):
    if len(df) < 30:
        return None
    data  = df.tail(min(60, len(df)))
    highs = data["High"].values
    cls   = data["Close"].values
    n     = len(highs)

    h1  = highs[: n // 2].max()
    h2  = highs[n // 2:].max()
    sim = abs(h1 - h2) / (h1 + 1e-10) * 100

    if sim > 6:
        return None

    between  = cls[highs[:n//2].argmax(): n//2 + 6]
    neckline = between.min() if len(between) > 0 else cls[-3:].min()

    current = cls[-1]
    pct     = (current - neckline) / (neckline + 1e-10) * 100

    if current <= neckline * 1.02:
        signal = "BREAKDOWN" if current < neckline else "NEAR_BREAKDOWN"
        conf   = 62
        if c["VOLUME_SPIKE_1_5X"] and signal == "BREAKDOWN": conf += 12
        return dict(pattern="DOUBLE_TOP", signal=signal, conf=min(conf, 85),
                    breakout_level=round(neckline, 2), pct_from_level=pct, type="chart")
    return None


def _det_head_and_shoulders(df, c):
    if len(df) < 40:
        return None
    data  = df.tail(min(80, len(df)))
    highs = data["High"].values
    cls   = data["Close"].values
    n     = len(highs)

    t1, t2 = n // 3, 2 * n // 3

    left   = highs[:t1].max()
    head   = highs[t1:t2].max()
    right  = highs[t2:].max()

    if not (head > left * 1.02 and head > right * 1.02):
        return None
    if abs(left - right) / (head + 1e-10) > 0.06:
        return None

    neckline = min(cls[highs[:t1].argmax()], cls[t2 + highs[t2:].argmax()])
    current  = cls[-1]
    pct      = (current - neckline) / (neckline + 1e-10) * 100

    if current <= neckline * 1.02:
        signal = "BREAKDOWN" if current < neckline else "NEAR_BREAKDOWN"
        conf   = 70
        if c["VOLUME_SPIKE_1_5X"] and signal == "BREAKDOWN": conf += 10
        return dict(pattern="HEAD_AND_SHOULDERS", signal=signal, conf=min(conf, 88),
                    breakout_level=round(neckline, 2), pct_from_level=pct, type="chart")
    return None


def _det_inv_head_and_shoulders(df, c):
    if len(df) < 40:
        return None
    data = df.tail(min(80, len(df)))
    lows = data["Low"].values
    cls  = data["Close"].values
    n    = len(lows)
    t1, t2 = n // 3, 2 * n // 3

    left = lows[:t1].min()
    head = lows[t1:t2].min()
    right= lows[t2:].min()

    if not (head < left * 0.98 and head < right * 0.98):
        return None
    if abs(left - right) / (abs(head) + 1e-10) > 0.06:
        return None

    neckline = max(cls[lows[:t1].argmin()], cls[t2 + lows[t2:].argmin()])
    current  = cls[-1]
    pct      = (current - neckline) / (neckline + 1e-10) * 100

    if current >= neckline * 0.96:
        signal = "BREAKOUT" if current > neckline else "NEAR_BREAKOUT"
        conf   = 72
        if c["VOLUME_SPIKE_1_5X"] and signal == "BREAKOUT": conf += 12
        return dict(pattern="INV_HEAD_AND_SHOULDERS", signal=signal, conf=min(conf, 90),
                    breakout_level=round(neckline, 2), pct_from_level=pct, type="chart")
    return None


def _det_ascending_triangle(df, c):
    if len(df) < 20:
        return None
    data  = df.tail(min(40, len(df)))
    highs = data["High"].values
    lows  = data["Low"].values
    cls   = data["Close"].values

    top_h = highs[highs >= np.percentile(highs, 78)]
    if len(top_h) < 3:
        return None
    flat_top = (top_h.std() / (top_h.mean() + 1e-10) * 100) < 3.0
    if not flat_top:
        return None

    slope = float(np.polyfit(np.arange(len(lows)), lows, 1)[0])
    if slope <= 0:
        return None

    resistance = float(top_h.mean())
    current    = float(cls[-1])
    pct        = (current - resistance) / (resistance + 1e-10) * 100

    if current >= resistance * 0.96:
        signal = "BREAKOUT" if current > resistance else "NEAR_BREAKOUT"
        conf   = 66
        if c["VOLUME_SPIKE_1_5X"] and signal == "BREAKOUT": conf += 12
        return dict(pattern="ASCENDING_TRIANGLE", signal=signal, conf=min(conf, 88),
                    breakout_level=round(resistance, 2), pct_from_level=pct, type="chart")
    return None


def _det_descending_triangle(df, c):
    if len(df) < 20:
        return None
    data  = df.tail(min(40, len(df)))
    highs = data["High"].values
    lows  = data["Low"].values
    cls   = data["Close"].values

    bot_l = lows[lows <= np.percentile(lows, 22)]
    if len(bot_l) < 3:
        return None
    flat_bot = (bot_l.std() / (abs(bot_l.mean()) + 1e-10) * 100) < 3.0
    if not flat_bot:
        return None

    slope = float(np.polyfit(np.arange(len(highs)), highs, 1)[0])
    if slope >= 0:
        return None

    support = float(bot_l.mean())
    current = float(cls[-1])
    pct     = (current - support) / (support + 1e-10) * 100

    if current <= support * 1.02:
        signal = "BREAKDOWN" if current < support else "NEAR_BREAKDOWN"
        conf   = 62
        if c["VOLUME_SPIKE_1_5X"] and signal == "BREAKDOWN": conf += 12
        return dict(pattern="DESCENDING_TRIANGLE", signal=signal, conf=min(conf, 85),
                    breakout_level=round(support, 2), pct_from_level=pct, type="chart")
    return None


def _det_symmetrical_triangle(df, c):
    if len(df) < 20:
        return None
    data   = df.tail(min(35, len(df)))
    highs  = data["High"].values
    lows   = data["Low"].values
    cls    = data["Close"].values
    x      = np.arange(len(highs))

    try:
        h_slope = float(np.polyfit(x, highs, 1)[0])
        l_slope = float(np.polyfit(x, lows, 1)[0])
    except Exception:
        return None

    if not (h_slope < -0.001 and l_slope > 0.001):
        return None

    resistance = float(highs[-1])
    current    = float(cls[-1])
    pct        = (current - resistance) / (resistance + 1e-10) * 100

    if current >= resistance * 0.97:
        signal = "BREAKOUT" if current > resistance else "NEAR_BREAKOUT"
        conf   = 60
        if c["VOLUME_SPIKE_1_5X"] and signal == "BREAKOUT": conf += 12
        return dict(pattern="SYMMETRICAL_TRIANGLE", signal=signal, conf=min(conf, 85),
                    breakout_level=round(resistance, 2), pct_from_level=pct, type="chart")
    return None


def _det_rounding_bottom(df, c):
    if len(df) < 35:
        return None
    data   = df.tail(min(60, len(df)))
    closes = data["Close"].values
    n      = len(closes)
    mid    = n // 2
    left_mean  = closes[:mid // 2].mean()
    bottom_mean= closes[mid - mid//4: mid + mid//4].mean()
    right_mean = closes[-mid//2:].mean()

    if left_mean > bottom_mean and right_mean > bottom_mean:
        resistance = float(data["High"].max())
        current    = float(closes[-1])
        pct        = (current - resistance) / (resistance + 1e-10) * 100
        if current >= resistance * 0.95:
            signal = "BREAKOUT" if current > resistance else "NEAR_BREAKOUT"
            conf   = 58
            if c["VOLUME_GREATER_THAN_AVG"]: conf += 8
            return dict(pattern="ROUNDING_BOTTOM", signal=signal, conf=min(conf, 82),
                        breakout_level=round(resistance, 2), pct_from_level=pct, type="chart")
    return None


def _det_volatility_squeeze(df, c):
    if not c["VOLATILITY_SQUEEZE"]:
        return None
    res   = c["_resistance"]
    close = c["_close"]
    pct   = (close - res) / (res + 1e-10) * 100
    conf  = 52
    if c["TIGHT_CONSOLIDATION"]:      conf += 10
    if c["VOLUME_GREATER_THAN_AVG"]:  conf +=  5
    return dict(pattern="VOLATILITY_SQUEEZE", signal="NEAR_BREAKOUT", conf=conf,
                breakout_level=res, pct_from_level=pct, type="chart")


def _det_support_bounce(df, c):
    if not c["NEAR_SUPPORT"]:
        return None
    if c["HIGHER_HIGH_HIGHER_LOW"] and c["VOLUME_GREATER_THAN_AVG"]:
        sup   = c["_support"]
        close = c["_close"]
        pct   = (close - sup) / (sup + 1e-10) * 100
        conf  = 55
        if c["VOLUME_SPIKE_1_5X"]: conf += 10
        return dict(pattern="SUPPORT_BOUNCE", signal="RETEST_BREAKOUT", conf=conf,
                    breakout_level=sup, pct_from_level=pct, type="chart")
    return None


def _det_w_pattern(df, c):
    """W pattern = two lows with a peak in between"""
    if len(df) < 25:
        return None
    cls = df["Close"].tail(30).values
    n   = len(cls)
    mid = n // 2

    b1  = cls[:mid].min()
    pk  = cls[mid // 2: mid + mid // 2].max()
    b2  = cls[mid:].min()

    depth = (pk - min(b1, b2)) / (pk + 1e-10) * 100
    sim   = abs(b1 - b2) / (b1 + 1e-10) * 100

    if sim < 7 and depth > 6:
        neckline = pk
        current  = float(cls[-1])
        pct      = (current - neckline) / (neckline + 1e-10) * 100
        if current >= neckline * 0.96:
            signal = "BREAKOUT" if current > neckline else "NEAR_BREAKOUT"
            conf   = 62
            if c["VOLUME_GREATER_THAN_AVG"]: conf += 8
            return dict(pattern="W_PATTERN", signal=signal, conf=conf,
                        breakout_level=round(neckline, 2), pct_from_level=pct, type="chart")
    return None


def _det_rectangle_range(df, c):
    if not c["LOW_VOLATILITY_RANGE"]:
        return None
    data  = df.tail(min(40, len(df)))
    close = c["_close"]
    res   = c["_resistance"]
    sup   = c["_support"]
    rng   = (res - sup) / (sup + 1e-10) * 100

    if rng < 3 or rng > 15:
        return None
    pct = (close - res) / (res + 1e-10) * 100

    if close >= res * 0.97:
        signal = "BREAKOUT" if close > res else "NEAR_BREAKOUT"
        conf   = 56
        if c["VOLUME_SPIKE_1_5X"] and signal == "BREAKOUT": conf += 14
        return dict(pattern="RECTANGLE_RANGE", signal=signal, conf=min(conf, 85),
                    breakout_level=round(res, 2), pct_from_level=pct, type="chart")
    return None


# All chart detectors
_CHART_DETECTORS = [
    _det_horizontal_breakout,
    _det_consolidation_breakout,
    _det_cup_and_handle,
    _det_bull_flag,
    _det_double_bottom,
    _det_double_top,
    _det_head_and_shoulders,
    _det_inv_head_and_shoulders,
    _det_ascending_triangle,
    _det_descending_triangle,
    _det_symmetrical_triangle,
    _det_rounding_bottom,
    _det_volatility_squeeze,
    _det_support_bounce,
    _det_w_pattern,
    _det_rectangle_range,
]


# ──────────────────────────────────────────────────────────────
#  MAIN SCANNER
# ──────────────────────────────────────────────────────────────

def scan_stock(symbol: str, df: pd.DataFrame) -> dict | None:
    try:
        if df is None or len(df) < 10:
            return None

        df    = add_indicators(df)
        cond  = check_conditions(df)
        trend, trend_dir = detect_trend(df)
        rsi   = cond["_rsi"]

        # ── Collect all signals ──────────────────────────
        all_signals = []

        # Chart patterns
        for det in _CHART_DETECTORS:
            try:
                r = det(df, cond)
                if r and r.get("conf", 0) >= 40:
                    all_signals.append(r)
            except Exception:
                continue

        # Candlestick patterns
        try:
            candles = detect_candlestick(df)
            all_signals.extend(candles)
        except Exception:
            pass

        # DMA signals
        try:
            dmas = detect_dma_signals(df, cond)
            all_signals.extend(dmas)
        except Exception:
            pass

        if not all_signals:
            return None

        # ── Pick primary signal (highest conf) ──────────
        best = max(all_signals, key=lambda x: x["conf"])

        # ── Confidence adjustments ───────────────────────
        conf = best["conf"]
        if trend == "UPTREND"   and "BREAKOUT" in best["signal"]:  conf = min(conf + 5, 95)
        if trend == "DOWNTREND" and best["signal"] == "BREAKOUT":  conf = max(conf - 15, 20)
        if rsi > 80:                                                conf = max(conf - 8,  20)
        if rsi < 40 and "REVERSAL" in best.get("signal", ""):      conf = min(conf + 5, 95)

        # False breakout flag
        if rsi > 75 and cond["_vol_ratio"] < 0.9 and best["signal"] == "BREAKOUT":
            best["signal"] = "FALSE_BREAKOUT"
            conf = max(conf - 20, 20)

        best["conf"] = conf

        # ── All patterns list (for ranking) ──────────────
        pattern_list   = [s["pattern"] for s in all_signals]
        signal_types   = list(set(s["signal"] for s in all_signals))
        signal_count   = len(all_signals)
        breakout_count = sum(1 for s in all_signals if "BREAKOUT" in s.get("signal",""))

        # Breakout level
        bl = best.get("breakout_level", cond["_resistance"])
        pct= best.get("pct_from_level", 0.0)

        return {
            "symbol"         : symbol,
            "cmp"            : round(cond["_close"], 2),
            "pattern"        : best["pattern"],
            "signal"         : best["signal"],
            "confidence"     : conf,
            "conf_label"     : "HIGH" if conf >= 70 else ("MEDIUM" if conf >= 55 else "LOW"),
            "vol_ratio"      : round(cond["_vol_ratio"], 2),
            "breakout_level" : bl,
            "pct_from_level" : round(pct, 2),
            "trend"          : trend,
            "rsi"            : round(rsi, 1),
            "ema20"          : cond["_ema20"],
            "ema50"          : cond["_ema50"],
            "sma200"         : cond["_sma200"],
            "support"        : round(cond["_support"], 2),
            "resistance"     : round(cond["_resistance"], 2),
            # Multi-signal ranking
            "signal_count"   : signal_count,
            "breakout_count" : breakout_count,
            "all_patterns"   : pattern_list,
            "all_signals"    : signal_types,
            "rank_score"     : conf + signal_count * 3 + breakout_count * 5,
            "_df"            : df,
        }

    except Exception:
        return None
