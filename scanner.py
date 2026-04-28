# ─────────────────────────────────────────────────────────────────────────────
#  NSE Breakout Scanner — Core Engine v6
#  New: Trade Grades A+/A/B/C · Strong Breakout · Swing Points
#       ATR Trade Setup · Backtest Engine · Breakout Strength Score
# ─────────────────────────────────────────────────────────────────────────────

import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ──────────────────────────────────────────────────────────────
#  DATA FETCH
# ──────────────────────────────────────────────────────────────

def fetch_one(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        df = yf.download(f"{symbol}.NS", period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how="all")
        return df if len(df) >= 10 else None
    except Exception:
        return None


def fetch_batch(symbols: list, period: str = "6mo", interval: str = "1d") -> dict:
    if not symbols:
        return {}

    ns_map = {f"{s}.NS": s for s in symbols}
    result = {}

    if len(symbols) == 1:
        df = fetch_one(symbols[0], period, interval)
        if df is not None:
            result[symbols[0]] = df
        return result

    try:
        raw = yf.download(tickers=list(ns_map.keys()), period=period,
                          interval=interval, group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)

        for ns_sym, sym in ns_map.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl0 = raw.columns.get_level_values(0).unique().tolist()
                    lvl1 = raw.columns.get_level_values(1).unique().tolist()
                    if ns_sym in lvl0:
                        df = raw[ns_sym].copy()
                    elif ns_sym in lvl1:
                        df = raw.xs(ns_sym, axis=1, level=1).copy()
                    elif sym in lvl0:
                        df = raw[sym].copy()
                    elif sym in lvl1:
                        df = raw.xs(sym, axis=1, level=1).copy()
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
#  NIFTY 50 MARKET CONTEXT
# ──────────────────────────────────────────────────────────────

def get_market_context() -> dict:
    """Fetch Nifty 50 trend. Adjusts signal confidence up/down."""
    try:
        df = yf.download("^NSEI", period="60d", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how="all")

        if len(df) < 10:
            return {"trend": "UNKNOWN", "change_5d": 0, "nifty_cmp": None}

        close     = df["Close"]
        ema20     = close.ewm(span=20, adjust=False).mean()
        current   = float(close.iloc[-1])
        ema_val   = float(ema20.iloc[-1])
        change_5d = round((current - float(close.iloc[-5])) / float(close.iloc[-5]) * 100, 2)
        change_1d = round((current - float(close.iloc[-2])) / float(close.iloc[-2]) * 100, 2)

        if current > ema_val and change_5d > 0:
            trend = "UPTREND"
        elif current < ema_val and change_5d < 0:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"

        return {
            "trend"    : trend,
            "change_5d": change_5d,
            "change_1d": change_1d,
            "nifty_cmp": round(current, 2),
            "ema20"    : round(ema_val, 2),
        }
    except Exception:
        return {"trend": "UNKNOWN", "change_5d": 0, "nifty_cmp": None}


# ──────────────────────────────────────────────────────────────
#  INDICATORS
# ──────────────────────────────────────────────────────────────

def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def _sma(s, n):  return s.rolling(n, min_periods=max(1, n//2)).mean()

def _atr(high, low, close, n=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=2).mean()

def _rsi(close, n=14):
    d    = close.diff()
    gain = d.clip(lower=0).rolling(n, min_periods=2).mean()
    loss = (-d.clip(upper=0)).rolling(n, min_periods=2).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))

def _macd(close):
    m = _ema(close,12) - _ema(close,26)
    s = _ema(m, 9)
    return m, s, m - s

def _bollinger(close, n=20, dev=2):
    ma = close.rolling(n, min_periods=5).mean()
    sd = close.rolling(n, min_periods=5).std()
    up = ma + dev*sd; lo = ma - dev*sd
    return up, lo, (up-lo)/(ma.abs()+1e-10)


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
    df["Resistance20"] = h.rolling(20, min_periods=5).max()
    df["Support20"]    = l.rolling(20, min_periods=5).min()
    return df


# ──────────────────────────────────────────────────────────────
#  SWING HIGH / LOW  (scipy or custom zigzag)
# ──────────────────────────────────────────────────────────────

def detect_swing_points(df: pd.DataFrame, order: int = 5) -> dict:
    """
    Detect swing highs and lows using find_peaks or custom rolling window.
    Returns indices of swing highs and lows.
    """
    highs = df["High"].values
    lows  = df["Low"].values
    n     = len(highs)

    if HAS_SCIPY and n >= order * 2 + 1:
        peak_idx,  _ = find_peaks(highs, distance=order, prominence=highs.std() * 0.3)
        trough_idx,_ = find_peaks(-lows, distance=order, prominence=lows.std()  * 0.3)
    else:
        # Custom: local max/min in rolling window of (2*order+1)
        peak_idx   = []
        trough_idx = []
        for i in range(order, n - order):
            window_h = highs[i-order:i+order+1]
            window_l = lows [i-order:i+order+1]
            if highs[i] == window_h.max():
                peak_idx.append(i)
            if lows[i]  == window_l.min():
                trough_idx.append(i)
        peak_idx   = np.array(peak_idx)
        trough_idx = np.array(trough_idx)

    return {
        "swing_highs"   : peak_idx,
        "swing_lows"    : trough_idx,
        "sh_prices"     : highs[peak_idx]   if len(peak_idx)   > 0 else np.array([]),
        "sl_prices"     : lows [trough_idx] if len(trough_idx) > 0 else np.array([]),
    }


# ──────────────────────────────────────────────────────────────
#  CONDITIONS LAYER
# ──────────────────────────────────────────────────────────────

def check_conditions(df: pd.DataFrame) -> dict:
    row   = df.iloc[-1]
    close = float(row["Close"])

    vol_ratio  = float(row["Vol_Ratio"])  if not pd.isna(row["Vol_Ratio"])  else 1.0
    resistance = float(row["Resistance20"]) if not pd.isna(row["Resistance20"]) else close*1.02
    support    = float(row["Support20"])    if not pd.isna(row["Support20"])    else close*0.98

    recent_range_pct = (df["High"].tail(20).max() - df["Low"].tail(20).min()) / (close+1e-10)
    bb_width     = float(row["BB_Width"])        if not pd.isna(row["BB_Width"])        else 0.05
    bb_width_avg = float(df["BB_Width"].tail(60).mean()) if len(df)>=20 else bb_width

    n_slope  = min(20, len(df)-1)
    recent   = df["Close"].iloc[-n_slope:].values
    try:    slope = float(np.polyfit(np.arange(n_slope), recent, 1)[0])
    except: slope = 0.0

    # ── New v6: Strong Breakout condition ──────────────────────
    # 1. Close > resistance
    # 2. Previous 5 candles did NOT close above resistance
    # 3. Volume > 1.8x
    # 4. Candle body > 60% of range
    last_body  = abs(float(row["Close"]) - float(row["Open"]))
    last_range = float(row["High"]) - float(row["Low"]) + 1e-10
    body_pct   = last_body / last_range

    prev5_closed_above = (df["Close"].iloc[-6:-1] > resistance).any() if len(df) >= 6 else False

    strong_breakout = (
        close > resistance * 1.001
        and not prev5_closed_above
        and vol_ratio >= 1.8
        and body_pct >= 0.6
    )

    return {
        "PRICE_ABOVE_RESISTANCE" : close > resistance * 1.001,
        "PRICE_BELOW_SUPPORT"    : close < support  * 0.999,
        "NEAR_RESISTANCE"        : resistance * 0.96 <= close <= resistance * 1.005,
        "NEAR_SUPPORT"           : support * 0.995 <= close <= support * 1.06,
        "HIGHER_HIGH_HIGHER_LOW" : slope > 0,
        "LOWER_HIGH_LOWER_LOW"   : slope < 0,
        "LOW_VOLATILITY_RANGE"   : recent_range_pct < 0.12,
        "TIGHT_CONSOLIDATION"    : recent_range_pct < 0.07,
        "VOLATILITY_SQUEEZE"     : bb_width < bb_width_avg * 0.75 if bb_width_avg>0 else False,
        "EXPANSION_MOVE"         : bb_width > bb_width_avg * 1.40 if bb_width_avg>0 else False,
        "VOLUME_GREATER_THAN_AVG": vol_ratio >= 1.0,
        "VOLUME_SPIKE_1_5X"      : vol_ratio >= 1.5,
        "VOLUME_SPIKE_2X"        : vol_ratio >= 2.0,
        "VOLUME_SPIKE_3X"        : vol_ratio >= 3.0,
        "ABOVE_EMA20"            : close > float(row["EMA20"])  if not pd.isna(row["EMA20"])  else False,
        "ABOVE_EMA50"            : close > float(row["EMA50"])  if not pd.isna(row["EMA50"])  else False,
        "ABOVE_SMA200"           : close > float(row["SMA200"]) if not pd.isna(row["SMA200"]) else False,
        "STRONG_BREAKOUT"        : strong_breakout,
        "_close"      : close,
        "_vol_ratio"  : vol_ratio,
        "_resistance" : resistance,
        "_support"    : support,
        "_slope"      : slope,
        "_body_pct"   : body_pct,
        "_ema20"      : float(row["EMA20"])  if not pd.isna(row["EMA20"])  else None,
        "_ema50"      : float(row["EMA50"])  if not pd.isna(row["EMA50"])  else None,
        "_sma200"     : float(row["SMA200"]) if not pd.isna(row["SMA200"]) else None,
        "_rsi"        : float(row["RSI"])    if not pd.isna(row["RSI"])    else 50.0,
        "_atr"        : float(row["ATR"])    if not pd.isna(row["ATR"])    else 0.0,
        "_bb_width"   : bb_width,
    }


# ──────────────────────────────────────────────────────────────
#  TRADE GRADE  (A+ / A / B / C)
# ──────────────────────────────────────────────────────────────

def assign_trade_grade(signal: str, vol_ratio: float, trend: str, rsi: float,
                       strong_breakout: bool) -> str:
    """
    A+: Elite — Strong breakout + 2x volume + Uptrend + RSI not overbought
    A : Strong — Breakout + 1.5x volume + Uptrend
    B : Moderate — Near breakout / weak volume / sideways
    C : Avoid — False breakout / Downtrend / Breakdown
    """
    is_bo   = signal in ("BREAKOUT", "CONSOLIDATION_BREAKOUT", "52W_HIGH_BREAKOUT")
    is_near = "NEAR" in signal
    is_bad  = "FALSE" in signal or "BREAKDOWN" in signal or trend == "DOWNTREND"

    if is_bad:
        return "C"

    if is_bo and (strong_breakout or (vol_ratio >= 2.0 and trend == "UPTREND" and rsi <= 75)):
        return "A+"
    if is_bo and vol_ratio >= 1.5 and trend == "UPTREND":
        return "A"
    if is_bo or is_near or signal in ("REVERSAL_SIGNAL", "RETEST_BREAKOUT"):
        return "B"

    return "C"


# ──────────────────────────────────────────────────────────────
#  BREAKOUT STRENGTH SCORE  (0-100)
# ──────────────────────────────────────────────────────────────

def breakout_strength_score(df: pd.DataFrame, c: dict, trend: str) -> int:
    score = 0

    if c["PRICE_ABOVE_RESISTANCE"]: score += 20
    if c["STRONG_BREAKOUT"]:        score += 15   # v6 strong breakout

    if   c["VOLUME_SPIKE_3X"]:  score += 20
    elif c["VOLUME_SPIKE_2X"]:  score += 15
    elif c["VOLUME_SPIKE_1_5X"]:score += 8

    if trend == "UPTREND":   score += 12
    elif trend == "SIDEWAYS": score += 4

    if c["_body_pct"] >= 0.7: score += 8
    elif c["_body_pct"] >= 0.5: score += 4

    if c["ABOVE_EMA20"]: score += 5
    if c["ABOVE_EMA50"]: score += 5
    if c["ABOVE_SMA200"]:score += 5

    rsi = c["_rsi"]
    if 45 <= rsi <= 68: score += 8    # Sweet spot
    elif rsi > 80:       score -= 10
    elif rsi > 75:       score -= 5

    if c["VOLATILITY_SQUEEZE"]: score += 5

    return max(0, min(100, score))


# ──────────────────────────────────────────────────────────────
#  ATR TRADE SETUP
# ──────────────────────────────────────────────────────────────

def atr_trade_setup(cmp: float, atr: float) -> dict:
    """Generate entry / SL / targets based on ATR."""
    if atr <= 0:
        return {}
    atr_pct = round(atr / cmp * 100, 2)
    entry   = round(cmp, 2)
    sl      = round(entry - 1.5 * atr, 2)
    t1      = round(entry + 1.0 * atr, 2)
    t2      = round(entry + 2.0 * atr, 2)
    t3      = round(entry + 3.0 * atr, 2)
    sl_pct  = round((sl - entry) / entry * 100, 2)
    t1_pct  = round((t1 - entry) / entry * 100, 2)
    t2_pct  = round((t2 - entry) / entry * 100, 2)
    t3_pct  = round((t3 - entry) / entry * 100, 2)
    rr_t2   = round(abs(t2_pct / sl_pct), 2) if sl_pct != 0 else 0

    return {
        "entry"  : entry,
        "sl"     : sl,
        "sl_pct" : sl_pct,
        "t1"     : t1, "t1_pct": t1_pct,
        "t2"     : t2, "t2_pct": t2_pct,
        "t3"     : t3, "t3_pct": t3_pct,
        "rr"     : f"1:{rr_t2}",
        "atr"    : round(atr, 2),
        "atr_pct": atr_pct,
    }


# ──────────────────────────────────────────────────────────────
#  TREND DETECTOR
# ──────────────────────────────────────────────────────────────

def detect_trend(df: pd.DataFrame) -> tuple:
    try:
        ema20  = df["EMA20"].iloc[-1]
        ema50  = df["EMA50"].iloc[-1]
        close  = df["Close"].iloc[-1]
        slope5 = df["EMA20"].diff().iloc[-5:].mean()
        if close > ema20 and ema20 > ema50 and slope5 > 0: return "UPTREND", 1
        if close < ema20 and ema20 < ema50 and slope5 < 0: return "DOWNTREND", -1
        return "SIDEWAYS", 0
    except Exception:
        return "SIDEWAYS", 0


# ──────────────────────────────────────────────────────────────
#  CANDLESTICK PATTERNS
# ──────────────────────────────────────────────────────────────

def detect_candlestick(df: pd.DataFrame) -> list:
    found = []
    if len(df) < 3: return found

    c0, c1, c2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    o0,h0,l0,cl0 = float(c0["Open"]),float(c0["High"]),float(c0["Low"]),float(c0["Close"])
    o1,h1,l1,cl1 = float(c1["Open"]),float(c1["High"]),float(c1["Low"]),float(c1["Close"])
    o2,h2,l2,cl2 = float(c2["Open"]),float(c2["High"]),float(c2["Low"]),float(c2["Close"])

    body0, body1 = abs(cl0-o0), abs(cl1-o1)
    rng0 = h0-l0+1e-10

    def _add(n, conf, sig="REVERSAL_SIGNAL"):
        found.append({"pattern":n,"signal":sig,"conf":conf,"type":"candle"})

    lw0 = (o0-l0) if cl0>o0 else (cl0-l0)
    uw0 = (h0-cl0) if cl0>o0 else (h0-o0)
    if cl1<o1 and cl0>o0 and o0<=cl1 and cl0>=o1 and body0>body1: _add("BULLISH_ENGULFING",72,"REVERSAL_SIGNAL")
    if lw0>=body0*2 and uw0<=body0*0.3 and cl0>o0:                _add("HAMMER",65,"REVERSAL_SIGNAL")
    if (h0-max(o0,cl0))>=body0*2 and (min(o0,cl0)-l0)<=body0*0.3:_add("INVERTED_HAMMER",58,"REVERSAL_SIGNAL")
    if body0<=rng0*0.08:                                            _add("DOJI",50,"REVERSAL_SIGNAL")

    body2=abs(cl2-o2)
    if cl2<o2 and abs(cl1-o1)<=min(body2,body0)*0.4 and cl0>o0 and cl0>(o2+cl2)/2:
        _add("MORNING_STAR",78,"REVERSAL_SIGNAL")
    if cl2>o2 and cl1>o1 and cl0>o0 and cl1>cl2 and cl0>cl1 and o1>o2 and o0>o1:
        _add("THREE_WHITE_SOLDIERS",80,"BREAKOUT")
    if cl0>o0 and body0>=rng0*0.85:        _add("BULLISH_MARUBOZU",70,"BREAKOUT")
    if cl1<o1 and cl0>o0 and o0<cl1 and cl0>(o1+cl1)/2: _add("PIERCING_LINE",63,"REVERSAL_SIGNAL")
    if cl1<o1 and cl0>o0 and o0>cl1 and cl0<o1 and body0<body1: _add("BULLISH_HARAMI",58,"REVERSAL_SIGNAL")
    uw_inv = h0-max(o0,cl0); lw_inv = min(o0,cl0)-l0
    if (h0-max(o0,cl0))<=rng0*0.05 and (min(o0,cl0)-l0)>=rng0*0.65 and body0<=rng0*0.1:
        _add("DRAGONFLY_DOJI",66,"REVERSAL_SIGNAL")
    if cl0>o0 and uw0>=body0*2.5 and lw_inv<=body0*0.2 and cl0<o0: pass  # shooting star
    if cl1>o1 and cl0<o0 and o0>=cl1 and cl0<=o1 and body0>body1:
        _add("BEARISH_ENGULFING_WARN",60,"REVERSAL_DOWN")

    return found


# ──────────────────────────────────────────────────────────────
#  DMA SIGNALS
# ──────────────────────────────────────────────────────────────

def detect_dma_signals(df: pd.DataFrame, c: dict) -> list:
    found = []
    if len(df) < 5: return found

    curr, prev = df.iloc[-1], df.iloc[-2]
    e20n = float(curr["EMA20"]) if not pd.isna(curr["EMA20"]) else None
    e20p = float(prev["EMA20"]) if not pd.isna(prev["EMA20"]) else None
    e50n = float(curr["EMA50"]) if not pd.isna(curr["EMA50"]) else None
    e50p = float(prev["EMA50"]) if not pd.isna(prev["EMA50"]) else None
    s200 = float(curr["SMA200"])if not pd.isna(curr["SMA200"])else None
    cln  = c["_close"]

    if e20n and e50n and e20p and e50p:
        if e20n>e50n and e20p<=e50p: found.append({"pattern":"GOLDEN_CROSS","signal":"BREAKOUT","conf":80,"type":"dma"})
        if e20n<e50n and e20p>=e50p: found.append({"pattern":"DEATH_CROSS","signal":"BREAKDOWN","conf":75,"type":"dma"})
        if s200 and abs(cln-s200)/s200<0.015:
            sig = "NEAR_BREAKOUT" if cln>s200 else "NEAR_BREAKDOWN"
            found.append({"pattern":"NEAR_200_DMA","signal":sig,"conf":60,"type":"dma"})
        if e50n and abs(cln-e50n)/e50n<0.01 and cln>e20n:
            found.append({"pattern":"50_EMA_BOUNCE","signal":"NEAR_BREAKOUT","conf":60,"type":"dma"})
        if s200 and cln>e20n and e20n>e50n and e50n>s200:
            found.append({"pattern":"FULL_BULLISH_ALIGNMENT","signal":"BREAKOUT","conf":72,"type":"dma"})

    return found


# ──────────────────────────────────────────────────────────────
#  CHART PATTERN DETECTORS (swing-point enhanced where applicable)
# ──────────────────────────────────────────────────────────────

def _det_horizontal_breakout(df, c):
    close, res = c["_close"], c["_resistance"]
    pct = (close-res)/(res+1e-10)*100
    if c["PRICE_ABOVE_RESISTANCE"]:
        h52 = float(df["High"].rolling(min(252,len(df))).max().iloc[-2]) if len(df)>20 else res
        pat = "52W_HIGH_BREAKOUT" if close>=h52*0.98 else "HORIZONTAL_BREAKOUT"
        conf = 62
        if c["STRONG_BREAKOUT"]:     conf += 18
        elif c["VOLUME_SPIKE_2X"]:   conf += 13
        elif c["VOLUME_SPIKE_1_5X"]: conf +=  8
        if pat == "52W_HIGH_BREAKOUT": conf += 7
        return dict(pattern=pat, signal="BREAKOUT", conf=min(conf,95),
                    breakout_level=res, pct_from_level=pct, type="chart")
    if c["NEAR_RESISTANCE"]:
        conf = 50
        if c["VOLUME_GREATER_THAN_AVG"]: conf += 8
        if c["HIGHER_HIGH_HIGHER_LOW"]:  conf += 5
        return dict(pattern="HORIZONTAL_RESISTANCE", signal="NEAR_BREAKOUT", conf=conf,
                    breakout_level=res, pct_from_level=pct, type="chart")
    return None


def _det_consolidation_breakout(df, c):
    if not c["LOW_VOLATILITY_RANGE"]: return None
    close     = c["_close"]
    cons_high = float(df["High"].tail(25).max())
    pct       = (close-cons_high)/(cons_high+1e-10)*100
    if close > cons_high*1.001:
        conf = 62
        if c["STRONG_BREAKOUT"]:     conf += 18
        elif c["VOLUME_SPIKE_2X"]:   conf += 14
        elif c["VOLUME_SPIKE_1_5X"]: conf +=  8
        if c["TIGHT_CONSOLIDATION"]: conf +=  5
        if c["VOLATILITY_SQUEEZE"]:  conf +=  5
        return dict(pattern="CONSOLIDATION", signal="CONSOLIDATION_BREAKOUT",
                    conf=min(conf,93), breakout_level=cons_high, pct_from_level=pct, type="chart")
    if close >= cons_high*0.97:
        conf = 52
        if c["VOLATILITY_SQUEEZE"]:  conf += 10
        return dict(pattern="CONSOLIDATION", signal="NEAR_BREAKOUT", conf=conf,
                    breakout_level=cons_high, pct_from_level=pct, type="chart")
    return None


def _det_cup_and_handle(df, c):
    if len(df) < 40: return None
    data   = df.tail(min(90,len(df)))
    closes = data["Close"].values
    n = len(closes); t1,t2 = n//3, 2*n//3
    left_peak  = closes[:t1].max()
    cup_bottom = closes[t1:t2].min()
    right_sec  = closes[t2:]
    if len(right_sec) < 5: return None
    right_peak = right_sec[:-2].max() if len(right_sec)>3 else right_sec.max()
    depth_pct  = (left_peak-cup_bottom)/(left_peak+1e-10)*100
    rim_diff   = abs(right_peak-left_peak)/(left_peak+1e-10)*100
    if not (10<=depth_pct<=60) or rim_diff>8: return None
    handle     = closes[-6:]
    pullback   = (right_peak-handle.min())/(right_peak+1e-10)*100
    current    = closes[-1]
    pct        = (current-right_peak)/(right_peak+1e-10)*100
    if 1<=pullback<=25 and current>=right_peak*0.96:
        signal = "BREAKOUT" if current>right_peak else "NEAR_BREAKOUT"
        conf   = 65
        if c["VOLUME_GREATER_THAN_AVG"]: conf += 8
        if signal=="BREAKOUT":           conf += 8
        return dict(pattern="CUP_AND_HANDLE", signal=signal, conf=min(conf,90),
                    breakout_level=round(right_peak,2), pct_from_level=pct, type="chart")
    return None


def _det_bull_flag(df, c):
    if len(df) < 18: return None
    closes  = df["Close"].values
    volumes = df["Volume"].values
    ps = closes[-20] if len(closes)>=20 else closes[0]
    pp = closes[-15:-5].max() if len(closes)>=15 else closes[-5:-1].max()
    pg = (pp-ps)/(ps+1e-10)*100
    if pg<5: return None
    flag = closes[-7:]
    if (flag.max()-flag.min())/(np.mean(flag)+1e-10)*100 > 8: return None
    fv = volumes[-7:].mean()
    pv = volumes[-20:-7].mean() if len(volumes)>=20 else fv
    vol_dryup = fv < pv*0.90
    fh = flag[:-1].max()
    current = closes[-1]
    pct = (current-fh)/(fh+1e-10)*100
    if current >= fh*0.97:
        signal = "BREAKOUT" if current>fh else "NEAR_BREAKOUT"
        conf   = 64
        if c["VOLUME_SPIKE_1_5X"] and signal=="BREAKOUT": conf += 14
        if vol_dryup: conf += 5
        return dict(pattern="BULL_FLAG", signal=signal, conf=min(conf,88),
                    breakout_level=round(fh,2), pct_from_level=pct, type="chart")
    return None


def _det_double_bottom_swing(df, c, swings):
    """Enhanced double bottom using swing lows."""
    sl_prices = swings["sl_prices"]
    sl_idx    = swings["swing_lows"]
    cls = df["Close"].values

    if len(sl_prices) < 2: return None
    b1, b2 = sl_prices[-2], sl_prices[-1]
    sim = abs(b1-b2)/(b1+1e-10)*100
    if sim > 6: return None

    # Neckline = highest close between the two troughs
    i1, i2 = sl_idx[-2], sl_idx[-1]
    between = cls[i1:i2]
    if len(between) < 2: return None
    neckline = between.max()
    depth    = (neckline-min(b1,b2))/(neckline+1e-10)*100
    if depth < 6: return None

    current = cls[-1]
    pct     = (current-neckline)/(neckline+1e-10)*100
    if current >= neckline*0.96:
        signal = "BREAKOUT" if current>neckline else "NEAR_BREAKOUT"
        conf   = 65
        if c["VOLUME_SPIKE_1_5X"] and signal=="BREAKOUT": conf += 14
        if c["HIGHER_HIGH_HIGHER_LOW"]: conf += 5
        return dict(pattern="DOUBLE_BOTTOM", signal=signal, conf=min(conf,90),
                    breakout_level=round(neckline,2), pct_from_level=pct, type="chart")
    return None


def _det_double_top_swing(df, c, swings):
    """Enhanced double top using swing highs."""
    sh_prices = swings["sh_prices"]
    sh_idx    = swings["swing_highs"]
    cls = df["Close"].values

    if len(sh_prices) < 2: return None
    h1, h2 = sh_prices[-2], sh_prices[-1]
    sim = abs(h1-h2)/(h1+1e-10)*100
    if sim > 6: return None

    i1, i2 = sh_idx[-2], sh_idx[-1]
    between = cls[i1:i2]
    if len(between) < 2: return None
    neckline = between.min()

    current = cls[-1]
    pct     = (current-neckline)/(neckline+1e-10)*100
    if current <= neckline*1.02:
        signal = "BREAKDOWN" if current<neckline else "NEAR_BREAKDOWN"
        conf   = 65
        if c["VOLUME_SPIKE_1_5X"] and signal=="BREAKDOWN": conf += 12
        return dict(pattern="DOUBLE_TOP", signal=signal, conf=min(conf,88),
                    breakout_level=round(neckline,2), pct_from_level=pct, type="chart")
    return None


def _det_inv_head_and_shoulders_swing(df, c, swings):
    """Inv H&S using 3 swing lows."""
    sl_prices = swings["sl_prices"]
    sl_idx    = swings["swing_lows"]
    cls = df["Close"].values

    if len(sl_prices) < 3: return None
    ls, head, rs = sl_prices[-3], sl_prices[-2], sl_prices[-1]
    if not (head < ls*0.98 and head < rs*0.98): return None
    if abs(ls-rs)/(abs(head)+1e-10) > 0.08: return None

    i_ls, i_hd, i_rs = sl_idx[-3], sl_idx[-2], sl_idx[-1]
    neckline = max(cls[i_ls:i_hd].max() if i_hd>i_ls else 0,
                   cls[i_hd:i_rs].max() if i_rs>i_hd else 0)

    current = cls[-1]
    pct     = (current-neckline)/(neckline+1e-10)*100
    if current >= neckline*0.96:
        signal = "BREAKOUT" if current>neckline else "NEAR_BREAKOUT"
        conf   = 72
        if c["VOLUME_SPIKE_1_5X"] and signal=="BREAKOUT": conf += 12
        return dict(pattern="INV_HEAD_AND_SHOULDERS", signal=signal, conf=min(conf,90),
                    breakout_level=round(neckline,2), pct_from_level=pct, type="chart")
    return None


def _det_ascending_triangle(df, c):
    if len(df) < 20: return None
    data  = df.tail(min(40,len(df)))
    highs, lows, cls = data["High"].values, data["Low"].values, data["Close"].values
    top_h = highs[highs>=np.percentile(highs,78)]
    if len(top_h) < 3: return None
    if (top_h.std()/(top_h.mean()+1e-10)*100) >= 3.0: return None
    if float(np.polyfit(np.arange(len(lows)), lows, 1)[0]) <= 0: return None
    resistance = float(top_h.mean())
    current    = float(cls[-1])
    pct        = (current-resistance)/(resistance+1e-10)*100
    if current >= resistance*0.96:
        signal = "BREAKOUT" if current>resistance else "NEAR_BREAKOUT"
        conf   = 66
        if c["VOLUME_SPIKE_1_5X"] and signal=="BREAKOUT": conf += 12
        return dict(pattern="ASCENDING_TRIANGLE", signal=signal, conf=min(conf,88),
                    breakout_level=round(resistance,2), pct_from_level=pct, type="chart")
    return None


def _det_symmetrical_triangle(df, c):
    if len(df) < 20: return None
    data = df.tail(min(35,len(df)))
    highs,lows,cls = data["High"].values,data["Low"].values,data["Close"].values
    x = np.arange(len(highs))
    try:
        hs = float(np.polyfit(x,highs,1)[0])
        ls = float(np.polyfit(x,lows, 1)[0])
    except: return None
    if not (hs<-0.001 and ls>0.001): return None
    resistance = float(highs[-1]); current = float(cls[-1])
    pct = (current-resistance)/(resistance+1e-10)*100
    if current >= resistance*0.97:
        signal = "BREAKOUT" if current>resistance else "NEAR_BREAKOUT"
        conf   = 60
        if c["VOLUME_SPIKE_1_5X"] and signal=="BREAKOUT": conf += 12
        return dict(pattern="SYMMETRICAL_TRIANGLE", signal=signal, conf=min(conf,85),
                    breakout_level=round(resistance,2), pct_from_level=pct, type="chart")
    return None


def _det_rounding_bottom(df, c):
    if len(df)<35: return None
    data = df.tail(min(60,len(df)))
    cls = data["Close"].values; n=len(cls); mid=n//2
    lm = cls[:mid//2].mean(); bm = cls[mid-mid//4:mid+mid//4].mean(); rm=cls[-mid//2:].mean()
    if lm>bm and rm>bm:
        resistance = float(data["High"].max()); current=float(cls[-1])
        pct=(current-resistance)/(resistance+1e-10)*100
        if current>=resistance*0.95:
            signal = "BREAKOUT" if current>resistance else "NEAR_BREAKOUT"
            conf   = 58
            if c["VOLUME_GREATER_THAN_AVG"]: conf += 8
            return dict(pattern="ROUNDING_BOTTOM", signal=signal, conf=min(conf,82),
                        breakout_level=round(resistance,2), pct_from_level=pct, type="chart")
    return None


def _det_volatility_squeeze(df, c):
    if not c["VOLATILITY_SQUEEZE"]: return None
    conf=52
    if c["TIGHT_CONSOLIDATION"]:     conf+=10
    if c["VOLUME_GREATER_THAN_AVG"]: conf+=5
    return dict(pattern="VOLATILITY_SQUEEZE", signal="NEAR_BREAKOUT", conf=conf,
                breakout_level=c["_resistance"], pct_from_level=0, type="chart")


def _det_support_bounce(df, c):
    if not c["NEAR_SUPPORT"]: return None
    if c["HIGHER_HIGH_HIGHER_LOW"] and c["VOLUME_GREATER_THAN_AVG"]:
        conf=55
        if c["VOLUME_SPIKE_1_5X"]: conf+=10
        return dict(pattern="SUPPORT_BOUNCE", signal="RETEST_BREAKOUT", conf=conf,
                    breakout_level=c["_support"],
                    pct_from_level=(c["_close"]-c["_support"])/(c["_support"]+1e-10)*100,
                    type="chart")
    return None


def _det_w_pattern(df, c):
    if len(df)<25: return None
    cls = df["Close"].tail(30).values; n=len(cls); mid=n//2
    b1,pk,b2 = cls[:mid].min(), cls[mid//2:mid+mid//2].max(), cls[mid:].min()
    if abs(b1-b2)/(b1+1e-10)*100<7 and (pk-min(b1,b2))/(pk+1e-10)*100>6:
        neckline=pk; current=float(cls[-1])
        pct=(current-neckline)/(neckline+1e-10)*100
        if current>=neckline*0.96:
            signal = "BREAKOUT" if current>neckline else "NEAR_BREAKOUT"
            conf   = 62
            if c["VOLUME_GREATER_THAN_AVG"]: conf += 8
            return dict(pattern="W_PATTERN", signal=signal, conf=conf,
                        breakout_level=round(neckline,2), pct_from_level=pct, type="chart")
    return None


# ──────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ──────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, min_bars: int = 50) -> dict:
    """
    In-sample backtest: At each historical bar where a breakout occurs,
    check next 10 candles for T2 (2xATR) hit vs SL (1.5xATR) hit.
    Returns: win_rate, total_trades, avg_return, max_drawdown
    """
    if len(df) < min_bars + 10:
        return {"win_rate": None, "total": 0, "message": "Not enough data"}

    df = df.copy()
    df = add_indicators(df)
    results = []
    returns = []

    for i in range(min_bars, len(df)-10):
        sub = df.iloc[:i+1]
        try:
            cond = check_conditions(sub)
        except Exception:
            continue

        # Only test at breakout bars
        if not cond["PRICE_ABOVE_RESISTANCE"]: continue
        if not cond["VOLUME_SPIKE_1_5X"]:      continue

        entry = float(sub["Close"].iloc[-1])
        atr   = float(sub["ATR"].iloc[-1]) if not pd.isna(sub["ATR"].iloc[-1]) else 0
        if atr <= 0: continue

        sl = entry - 1.5 * atr
        t2 = entry + 2.0 * atr

        future  = df.iloc[i+1:i+11]
        hit_sl  = (future["Low"]  <= sl).any()
        hit_t2  = (future["High"] >= t2).any()

        if hit_t2 and hit_sl:
            t2_bar = (future["High"] >= t2).to_numpy().argmax()
            sl_bar = (future["Low"]  <= sl).to_numpy().argmax()
            win    = t2_bar <= sl_bar
        elif hit_t2:
            win = True
        elif hit_sl:
            win = False
        else:
            continue

        results.append(win)
        if win:
            returns.append((t2 - entry) / entry * 100)
        else:
            returns.append((sl - entry) / entry * 100)

    if not results:
        return {"win_rate": None, "total": 0, "message": "No signals found in history"}

    win_rate    = round(sum(results) / len(results) * 100, 1)
    avg_return  = round(np.mean(returns), 2)
    max_dd      = round(min(returns), 2)
    total_wins  = sum(results)

    return {
        "win_rate"  : win_rate,
        "total"     : len(results),
        "wins"      : total_wins,
        "losses"    : len(results) - total_wins,
        "avg_return": avg_return,
        "max_dd"    : max_dd,
        "message"   : f"Tested {len(results)} historical signals on this stock's data",
    }


# ──────────────────────────────────────────────────────────────
#  MAIN SCANNER
# ──────────────────────────────────────────────────────────────

def scan_stock(symbol: str, df: pd.DataFrame,
               market_trend: str = "UPTREND") -> dict | None:
    try:
        if df is None or len(df) < 10: return None
        df    = add_indicators(df)
        cond  = check_conditions(df)
        trend, _ = detect_trend(df)
        rsi   = cond["_rsi"]
        atr   = cond["_atr"]
        swings= detect_swing_points(df)

        # ── Collect all signals ────────────────────────────
        all_signals = []

        static_dets = [
            _det_horizontal_breakout, _det_consolidation_breakout,
            _det_cup_and_handle, _det_bull_flag, _det_ascending_triangle,
            _det_symmetrical_triangle, _det_rounding_bottom,
            _det_volatility_squeeze, _det_support_bounce, _det_w_pattern,
        ]
        for det in static_dets:
            try:
                r = det(df, cond)
                if r and r.get("conf",0) >= 40: all_signals.append(r)
            except Exception: continue

        # Swing-enhanced detectors
        for det in [_det_double_bottom_swing, _det_double_top_swing, _det_inv_head_and_shoulders_swing]:
            try:
                r = det(df, cond, swings)
                if r and r.get("conf",0) >= 40: all_signals.append(r)
            except Exception: continue

        # Candlestick + DMA
        try: all_signals.extend(detect_candlestick(df))
        except Exception: pass
        try: all_signals.extend(detect_dma_signals(df, cond))
        except Exception: pass

        if not all_signals: return None

        best = max(all_signals, key=lambda x: x["conf"])

        # ── Confidence adjustments ──────────────────────────
        conf = best["conf"]
        if trend=="UPTREND"   and "BREAKOUT" in best["signal"]: conf = min(conf+5, 95)
        if trend=="DOWNTREND" and best["signal"]=="BREAKOUT":   conf = max(conf-15,20)
        if rsi > 80: conf = max(conf-8, 20)
        if rsi < 40 and "REVERSAL" in best.get("signal",""): conf = min(conf+5,95)

        # Market context adjustment (v6)
        if market_trend == "DOWNTREND" and "BREAKOUT" in best["signal"]:
            conf = max(conf - 10, 20)
        if market_trend == "UPTREND" and "BREAKOUT" in best["signal"]:
            conf = min(conf + 3, 95)

        # False breakout flag
        if rsi>75 and cond["_vol_ratio"]<0.9 and best["signal"]=="BREAKOUT":
            best["signal"] = "FALSE_BREAKOUT"
            conf = max(conf-20, 20)

        best["conf"] = conf

        # ── Trade Grade (v6) ────────────────────────────────
        grade = assign_trade_grade(
            best["signal"], cond["_vol_ratio"], trend, rsi,
            cond["STRONG_BREAKOUT"]
        )

        # ── Breakout Strength Score (v6) ────────────────────
        strength = breakout_strength_score(df, cond, trend)

        # ── ATR Trade Setup (v6) ────────────────────────────
        setup = atr_trade_setup(cond["_close"], atr)

        # ── Breakout Age ────────────────────────────────────
        # How many candles ago resistance was first breached
        bo_age = 0
        res = cond["_resistance"]
        for j in range(1, min(6, len(df))):
            if float(df["Close"].iloc[-(j+1)]) > res:
                bo_age += 1
            else:
                break

        pattern_list = [s["pattern"] for s in all_signals]
        signal_types = list(set(s["signal"] for s in all_signals))
        n_bo         = sum(1 for s in all_signals if "BREAKOUT" in s.get("signal",""))
        rank_score   = conf + len(all_signals)*3 + n_bo*5 + (15 if grade=="A+" else 8 if grade=="A" else 0)

        return {
            "symbol"         : symbol,
            "cmp"            : round(cond["_close"], 2),
            "pattern"        : best["pattern"],
            "signal"         : best["signal"],
            "confidence"     : conf,
            "conf_label"     : "HIGH" if conf>=70 else ("MEDIUM" if conf>=55 else "LOW"),
            "trade_grade"    : grade,
            "strength_score" : strength,
            "atr_setup"      : setup,
            "breakout_age"   : bo_age,
            "vol_ratio"      : round(cond["_vol_ratio"], 2),
            "breakout_level" : best.get("breakout_level", cond["_resistance"]),
            "pct_from_level" : round(best.get("pct_from_level", 0), 2),
            "trend"          : trend,
            "rsi"            : round(rsi, 1),
            "ema20"          : cond["_ema20"],
            "ema50"          : cond["_ema50"],
            "sma200"         : cond["_sma200"],
            "support"        : round(cond["_support"], 2),
            "resistance"     : round(cond["_resistance"], 2),
            "strong_breakout": cond["STRONG_BREAKOUT"],
            "signal_count"   : len(all_signals),
            "breakout_count" : n_bo,
            "all_patterns"   : pattern_list,
            "all_signals"    : signal_types,
            "rank_score"     : rank_score,
            "_df"            : df,
            "_atr"           : round(atr, 2),
        }

    except Exception:
        return None
