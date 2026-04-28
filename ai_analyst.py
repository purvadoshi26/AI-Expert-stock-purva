# ─────────────────────────────────────────────────────────────────────────────
#  AI Analyst v6
#  New: Rule-based analysis (no API key needed) · Google News RSS · Telegram
# ─────────────────────────────────────────────────────────────────────────────

import yfinance as yf
import pandas as pd
import numpy as np
import json
import requests
import anthropic
from datetime import datetime, timezone, timedelta
from xml.etree import ElementTree as ET
import warnings
warnings.filterwarnings("ignore")

IST = timezone(timedelta(hours=5, minutes=30))


# ──────────────────────────────────────────────────────────────
#  NEWS  (Google News RSS — free, no API key)
# ──────────────────────────────────────────────────────────────

def get_news(symbol: str, company_name: str = "") -> list:
    """Try Google News RSS first, fall back to yfinance."""
    news = _google_news_rss(symbol, company_name)
    if not news:
        news = _yfinance_news(symbol)
    return news


def _google_news_rss(symbol: str, company_name: str = "", limit: int = 6) -> list:
    """Scrape Google News RSS — completely free."""
    query = company_name if company_name else f"{symbol} NSE stock India"
    query = query.replace(" ", "+")
    url   = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        resp = requests.get(url, timeout=6,
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        if resp.status_code != 200:
            return []

        root  = ET.fromstring(resp.content)
        items = root.findall(".//item")[:limit]
        news  = []

        for item in items:
            title  = item.findtext("title", "")  or "No title"
            link   = item.findtext("link",  "#") or "#"
            pub    = item.findtext("pubDate", "") or ""
            source_el = item.find("source")
            source = source_el.text if source_el is not None else "Google News"

            # Parse date + freshness tag
            fresh_tag, pub_str = _parse_pub_date(pub)

            news.append({
                "title"  : title,
                "link"   : link,
                "source" : source,
                "date"   : pub_str,
                "fresh"  : fresh_tag,
            })
        return news
    except Exception:
        return []


def _parse_pub_date(pub_str: str) -> tuple:
    """Returns (fresh_tag, formatted_date)."""
    try:
        # Format: "Tue, 22 Apr 2025 10:30:00 GMT"
        from email.utils import parsedate_to_datetime
        dt  = parsedate_to_datetime(pub_str)
        now = datetime.now(timezone.utc)
        diff_h = (now - dt).total_seconds() / 3600

        if diff_h < 6:
            tag = "🟢 FRESH"
        elif diff_h < 24:
            tag = "🟡 TODAY"
        else:
            tag = ""

        return tag, dt.strftime("%d %b %Y %H:%M")
    except Exception:
        return "", pub_str[:16] if pub_str else "Recent"


def _yfinance_news(symbol: str, limit: int = 6) -> list:
    """Fallback: yfinance news."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        news   = ticker.news or []
        results = []
        for item in news[:limit]:
            content = item.get("content", {})
            title   = content.get("title", item.get("title", "No title"))
            link    = content.get("canonicalUrl", {}).get("url", item.get("link", "#"))
            source  = content.get("provider", {}).get("displayName", "News")
            pub_ts  = content.get("pubDate", item.get("providerPublishTime", None))
            if isinstance(pub_ts, (int, float)):
                dt  = datetime.utcfromtimestamp(pub_ts).replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                diff_h = (now - dt).total_seconds() / 3600
                fresh = "🟢 FRESH" if diff_h < 6 else ("🟡 TODAY" if diff_h < 24 else "")
                pub_str = dt.strftime("%d %b %Y")
            else:
                fresh, pub_str = "", str(pub_ts)[:10] if pub_ts else "Recent"
            results.append({"title": title, "link": link,
                             "source": source, "date": pub_str, "fresh": fresh})
        return results
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
#  FINANCIALS
# ──────────────────────────────────────────────────────────────

def get_financials(symbol: str) -> dict:
    try:
        info = yf.Ticker(f"{symbol}.NS").info

        def safe(key, fmt="num"):
            val = info.get(key)
            if val is None or (isinstance(val, float) and val != val): return None
            if fmt == "cr":  return round(val / 1e7, 2)
            if fmt == "pct": return round(val * 100, 2)
            if fmt == "2f":  return round(float(val), 2)
            return val

        return {
            "company_name"    : info.get("longName") or info.get("shortName") or symbol,
            "sector"          : info.get("sector", "N/A"),
            "industry"        : info.get("industry", "N/A"),
            "market_cap_cr"   : safe("marketCap", "cr"),
            "pe_trailing"     : safe("trailingPE",  "2f"),
            "pe_forward"      : safe("forwardPE",   "2f"),
            "pb_ratio"        : safe("priceToBook", "2f"),
            "debt_equity"     : safe("debtToEquity","2f"),
            "roe_pct"         : safe("returnOnEquity","pct"),
            "revenue_growth"  : safe("revenueGrowth", "pct"),
            "earnings_growth" : safe("earningsGrowth","pct"),
            "dividend_yield"  : safe("dividendYield", "pct"),
            "week52_high"     : safe("fiftyTwoWeekHigh", "2f"),
            "week52_low"      : safe("fiftyTwoWeekLow",  "2f"),
            "avg50d"          : safe("fiftyDayAverage",       "2f"),
            "avg200d"         : safe("twoHundredDayAverage",  "2f"),
            "eps_ttm"         : safe("trailingEps", "2f"),
            "book_value"      : safe("bookValue",   "2f"),
            "beta"            : safe("beta",        "2f"),
            "free_cashflow_cr": safe("freeCashflow","cr"),
        }
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────
#  RULE-BASED ANALYSIS  (no API key needed — v6 new feature)
# ──────────────────────────────────────────────────────────────

def rule_based_analysis(scan_result: dict, financials: dict) -> dict:
    """
    Generates structured expert analysis purely from rules.
    Used when no Anthropic API key is set.
    """
    grade    = scan_result.get("trade_grade", "B")
    signal   = scan_result.get("signal", "")
    trend    = scan_result.get("trend", "SIDEWAYS")
    rsi      = scan_result.get("rsi", 50)
    vol      = scan_result.get("vol_ratio", 1.0)
    cmp      = scan_result.get("cmp", 0)
    atr_val  = scan_result.get("_atr", 0)
    pattern  = scan_result.get("pattern", "")
    strength = scan_result.get("strength_score", 50)
    sym      = scan_result.get("symbol", "")
    setup    = scan_result.get("atr_setup", {})

    # --- Overall recommendation ---
    if grade == "A+":
        rec, conviction, sentiment = "BUY", "HIGH", "BULLISH"
        summary = (
            f"{sym} is showing an elite A+ breakout setup — strong volume confirmation "
            f"({vol:.1f}x average), {trend.lower()}, pattern: {pattern.replace('_',' ')}. "
            f"This is a high-quality signal with {strength}/100 strength score."
        )
        key_risk = f"RSI at {rsi:.0f} — watch for overbought signals above 75. Market reversal risk."
    elif grade == "A":
        rec, conviction, sentiment = "BUY", "MEDIUM", "BULLISH"
        summary = (
            f"{sym} shows a solid A-grade breakout with volume ({vol:.1f}x), "
            f"pattern: {pattern.replace('_',' ')}, trend: {trend}. "
            f"Breakout strength score: {strength}/100. Suitable for swing trades."
        )
        key_risk = "Volume not at 2x level. Monitor for follow-through in next 2-3 sessions."
    elif grade == "B":
        rec, conviction, sentiment = "WAIT", "LOW", "NEUTRAL"
        summary = (
            f"{sym} has a {pattern.replace('_',' ')} setup forming but signal not fully confirmed. "
            f"Volume at {vol:.1f}x average, RSI: {rsi:.0f}. Wait for clearer breakout confirmation."
        )
        key_risk = "Pattern not confirmed. Entry now carries higher risk of false signal."
    else:
        rec, conviction, sentiment = "AVOID", "LOW", "BEARISH"
        summary = (
            f"{sym} is showing weakness — {signal.replace('_',' ')} with {trend}. "
            f"RSI: {rsi:.0f}, Volume: {vol:.1f}x. Not recommended for long trades."
        )
        key_risk = "Downtrend or false breakout detected. Capital preservation priority."

    # --- Intraday (high risk) ---
    entry = setup.get("entry", cmp)
    sl_i  = setup.get("sl",    round(cmp - 1.0 * atr_val, 2)) if atr_val else round(cmp * 0.99, 2)
    t1_i  = setup.get("t1",    round(cmp + 0.7 * atr_val, 2)) if atr_val else round(cmp * 1.01, 2)
    t2_i  = setup.get("t2",    round(cmp + 1.3 * atr_val, 2)) if atr_val else round(cmp * 1.02, 2)

    suitable_intraday = grade in ("A+", "A") and rsi < 75
    intraday_note = (
        "Strong volume breakout — good intraday momentum play. Exit before 3 PM." if grade == "A+"
        else "Moderate setup for intraday. Use tight SL and take partial profits early." if grade == "A"
        else "Not ideal for intraday. Pattern needs more confirmation."
    )

    # --- Swing (medium risk) ---
    sl_s  = setup.get("sl", round(cmp - 1.5 * atr_val, 2)) if atr_val else round(cmp * 0.97, 2)
    t1_s  = setup.get("t1", round(cmp + 1.0 * atr_val, 2)) if atr_val else round(cmp * 1.02, 2)
    t2_s  = setup.get("t2", round(cmp + 2.0 * atr_val, 2)) if atr_val else round(cmp * 1.04, 2)
    t3_s  = setup.get("t3", round(cmp + 3.0 * atr_val, 2)) if atr_val else round(cmp * 1.06, 2)

    suitable_swing = grade in ("A+", "A", "B") and trend != "DOWNTREND"
    swing_note = (
        f"Hold for 5-10 trading days. Trail SL to breakeven once T1 hit. "
        f"Pattern {pattern.replace('_',' ')} historically works well over swing timeframe."
    )

    # --- Long Term (lower risk if fundamentals strong) ---
    pe   = financials.get("pe_trailing")
    roe  = financials.get("roe_pct")
    de   = financials.get("debt_equity")

    fund_ok = (pe and pe < 35) or (roe and roe > 12) or True  # default allow

    sl_lt = float(scan_result.get("sma200") or cmp * 0.90)
    t1_lt = round(cmp * 1.10, 2)
    t2_lt = round(cmp * 1.20, 2)
    t3_lt = round(cmp * 1.35, 2)

    fund_view = ""
    if pe:   fund_view += f"P/E {pe} — {'reasonable' if pe<30 else 'stretched'}. "
    if roe:  fund_view += f"ROE {roe}% — {'strong' if roe>15 else 'moderate'}. "
    if de:   fund_view += f"D/E {de} — {'low debt' if de<1 else 'watch debt levels'}."
    if not fund_view:
        fund_view = "Fundamental data not available. Rely on technicals for this trade."

    suitable_lt = grade in ("A+", "A") and fund_ok

    # --- Risk Ranking ---
    risk_ranking = [
        {"timeframe": "Long Term", "risk": "LOW",    "reason": "SL below 200 DMA. Wide target. Fits investors."},
        {"timeframe": "Swing",     "risk": "MEDIUM", "reason": "ATR-based SL. 5-10 day hold. Better than intraday."},
        {"timeframe": "Intraday",  "risk": "HIGH",   "reason": "Price action volatile intraday. Tight SL. Experience needed."},
    ]

    # RSI commentary
    rsi_note = ""
    if rsi > 75: rsi_note = f"⚠️ RSI at {rsi:.0f} — overbought. Risk of short-term pullback before next leg up."
    elif rsi < 35: rsi_note = f"✅ RSI at {rsi:.0f} — oversold bounce signal. Risk/reward favors buyers here."
    elif 50 <= rsi <= 65: rsi_note = f"✅ RSI at {rsi:.0f} — healthy momentum zone. Not overbought."

    news_impact = "Load news to see market sentiment. News analysis helps confirm or deny the technical setup."

    def pct_from(target, base):
        return round((target - base) / (base + 1e-10) * 100, 2)

    return {
        "overall": {
            "recommendation": rec,
            "conviction"    : conviction,
            "summary"       : summary,
            "key_risk"      : key_risk,
            "sentiment"     : sentiment,
        },
        "intraday": {
            "suitable"     : suitable_intraday,
            "view"         : "LONG" if rec == "BUY" else "AVOID",
            "entry_zone"   : f"₹{entry} - ₹{round(entry*1.005,2)}",
            "buy_price"    : entry,
            "sl"           : sl_i,
            "target1"      : t1_i,
            "target2"      : t2_i,
            "sl_pct"       : pct_from(sl_i, entry),
            "t1_pct"       : pct_from(t1_i, entry),
            "t2_pct"       : pct_from(t2_i, entry),
            "risk_reward"  : f"1:{round(abs(pct_from(t2_i,entry)/pct_from(sl_i,entry)),2) if pct_from(sl_i,entry)!=0 else 'N/A'}",
            "time_in_trade": "2-4 hours",
            "risk_level"   : "HIGH",
            "note"         : intraday_note,
        },
        "swing": {
            "suitable"    : suitable_swing,
            "view"        : "LONG" if rec in ("BUY","WAIT") else "AVOID",
            "entry_zone"  : f"₹{entry} - ₹{round(entry*1.01,2)}",
            "buy_price"   : entry,
            "sl"          : sl_s,
            "target1"     : t1_s,
            "target2"     : t2_s,
            "target3"     : t3_s,
            "sl_pct"      : pct_from(sl_s, entry),
            "t1_pct"      : pct_from(t1_s, entry),
            "t2_pct"      : pct_from(t2_s, entry),
            "t3_pct"      : pct_from(t3_s, entry),
            "risk_reward" : f"1:{round(abs(pct_from(t2_s,entry)/pct_from(sl_s,entry)),2) if pct_from(sl_s,entry)!=0 else 'N/A'}",
            "holding_period": "5-10 trading days",
            "risk_level"  : "MEDIUM",
            "note"        : swing_note,
        },
        "longterm": {
            "suitable"         : suitable_lt,
            "view"             : "ACCUMULATE" if rec == "BUY" else ("HOLD" if rec == "WAIT" else "AVOID"),
            "entry_zone"       : f"₹{entry} - ₹{round(entry*1.02,2)}",
            "buy_price"        : entry,
            "sl"               : round(sl_lt, 2),
            "target1"          : t1_lt,
            "target2"          : t2_lt,
            "target3"          : t3_lt,
            "sl_pct"           : pct_from(round(sl_lt,2), entry),
            "t1_pct"           : pct_from(t1_lt, entry),
            "t2_pct"           : pct_from(t2_lt, entry),
            "t3_pct"           : pct_from(t3_lt, entry),
            "risk_reward"      : f"1:{round(abs(pct_from(t2_lt,entry)/pct_from(round(sl_lt,2),entry)),2) if pct_from(round(sl_lt,2),entry)!=0 else 'N/A'}",
            "holding_period"   : "3-12 months",
            "risk_level"       : "LOW",
            "fundamental_view" : fund_view,
            "note"             : f"SL placed near 200 DMA (₹{round(sl_lt,2)}). {rsi_note}",
        },
        "risk_ranking"  : risk_ranking,
        "news_impact"   : news_impact,
        "rsi_note"      : rsi_note,
        "source"        : "rule_based",
        "disclaimer"    : "⚠️ Rule-based AI analysis for educational purposes. Not SEBI registered investment advice. Always do your own research and consult a financial advisor before investing.",
    }


# ──────────────────────────────────────────────────────────────
#  CLAUDE API ANALYSIS  (enhanced v6 with grade awareness)
# ──────────────────────────────────────────────────────────────

_EXPERT_PROMPT = """
You are a senior equity analyst with 20 years of Indian stock market experience.

=== STOCK: {symbol} ===
Trade Grade: {trade_grade} (A+=Elite, A=Strong, B=Moderate, C=Avoid)
Breakout Strength Score: {strength}/100
CMP: ₹{cmp}
Pattern: {pattern} | Signal: {signal} | Confidence: {confidence}%
Trend: {trend} | RSI: {rsi} | Volume: {vol_ratio}x avg
EMA20: {ema20} | EMA50: {ema50} | SMA200: {sma200}
ATR(14): {atr}
Support: ₹{support} | Resistance: ₹{resistance} | Breakout Level: ₹{breakout_level}
Strong Breakout (v6): {strong_breakout}

=== LAST 15 SESSIONS ===
{ohlcv_table}

=== FUNDAMENTALS ===
{fundamental_data}

=== NEWS ===
{news_summary}

Return ONLY valid JSON with this structure (no markdown, no explanation):
{{
  "overall": {{"recommendation":"BUY|WAIT|AVOID","conviction":"HIGH|MEDIUM|LOW","summary":"2-3 lines","key_risk":"1 line","sentiment":"BULLISH|NEUTRAL|BEARISH"}},
  "intraday": {{"suitable":true,"view":"LONG|SHORT|AVOID","entry_zone":"₹X-₹Y","buy_price":N,"sl":N,"target1":N,"target2":N,"sl_pct":N,"t1_pct":N,"t2_pct":N,"risk_reward":"1:X","time_in_trade":"2-4 hours","risk_level":"HIGH|MEDIUM|LOW","note":"..."}},
  "swing": {{"suitable":true,"view":"LONG|SHORT|AVOID","entry_zone":"₹X-₹Y","buy_price":N,"sl":N,"target1":N,"target2":N,"target3":N,"sl_pct":N,"t1_pct":N,"t2_pct":N,"t3_pct":N,"risk_reward":"1:X","holding_period":"5-10 days","risk_level":"HIGH|MEDIUM|LOW","note":"..."}},
  "longterm": {{"suitable":true,"view":"ACCUMULATE|HOLD|AVOID","entry_zone":"₹X-₹Y","buy_price":N,"sl":N,"target1":N,"target2":N,"target3":N,"sl_pct":N,"t1_pct":N,"t2_pct":N,"t3_pct":N,"risk_reward":"1:X","holding_period":"6-12 months","risk_level":"HIGH|MEDIUM|LOW","fundamental_view":"...","note":"..."}},
  "risk_ranking":[{{"timeframe":"Intraday","risk":"HIGH","reason":"..."}},{{"timeframe":"Swing","risk":"MEDIUM","reason":"..."}},{{"timeframe":"Long Term","risk":"LOW","reason":"..."}}],
  "news_impact":"...",
  "disclaimer":"AI-generated. Not SEBI registered advice. Do your own research."
}}

SL rules: Intraday=0.5-1x ATR. Swing=1.5-2x ATR. LongTerm=below SMA200 or major support.
Min R:R = 1:1.5. Be realistic, not optimistic.
"""


def get_ai_analysis(symbol: str, scan_result: dict, financials: dict,
                    news: list, api_key: str) -> dict:
    """Call Claude API for expert analysis. Falls back to rule-based if fails."""
    if not api_key:
        return rule_based_analysis(scan_result, financials)

    try:
        df  = scan_result.get("_df")
        atr = scan_result.get("_atr", 0)

        ohlcv_str = "Date | Open | High | Low | Close | Volume\n"
        if df is not None:
            for idx, row in df.tail(15).iterrows():
                try:
                    d = idx.strftime("%d-%b") if hasattr(idx,"strftime") else str(idx)[:10]
                    ohlcv_str += f"{d} | {row['Open']:.1f} | {row['High']:.1f} | {row['Low']:.1f} | {row['Close']:.1f} | {int(row['Volume']):,}\n"
                except: continue

        fund_parts = []
        for k, lbl in [("company_name","Name"),("sector","Sector"),("market_cap_cr","MCap ₹Cr"),
                        ("pe_trailing","PE"),("pb_ratio","P/B"),("roe_pct","ROE%"),
                        ("debt_equity","D/E"),("revenue_growth","RevGrowth%"),("eps_ttm","EPS"),
                        ("beta","Beta"),("week52_high","52W Hi"),("week52_low","52W Lo")]:
            v = financials.get(k)
            if v: fund_parts.append(f"{lbl}: {v}")
        fund_str = " | ".join(fund_parts) if fund_parts else "N/A"

        news_str = "\n".join(
            [f"- [{n['date']}] {n['title']} ({n['source']})" for n in news[:5]]
        ) if news else "No news available"

        prompt = _EXPERT_PROMPT.format(
            symbol         = symbol,
            trade_grade    = scan_result.get("trade_grade","B"),
            strength       = scan_result.get("strength_score",50),
            cmp            = scan_result.get("cmp","N/A"),
            pattern        = scan_result.get("pattern","N/A"),
            signal         = scan_result.get("signal","N/A"),
            confidence     = scan_result.get("confidence","N/A"),
            trend          = scan_result.get("trend","N/A"),
            rsi            = scan_result.get("rsi","N/A"),
            vol_ratio      = scan_result.get("vol_ratio","N/A"),
            ema20          = scan_result.get("ema20") or "N/A",
            ema50          = scan_result.get("ema50") or "N/A",
            sma200         = scan_result.get("sma200") or "N/A",
            atr            = atr or "N/A",
            support        = scan_result.get("support","N/A"),
            resistance     = scan_result.get("resistance","N/A"),
            breakout_level = scan_result.get("breakout_level","N/A"),
            strong_breakout= scan_result.get("strong_breakout", False),
            ohlcv_table    = ohlcv_str,
            fundamental_data = fund_str,
            news_summary   = news_str,
        )

        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=2000,
            messages=[{"role":"user","content":prompt}]
        )
        raw = response.content[0].text.strip().replace("```json","").replace("```","").strip()
        result = json.loads(raw)
        result["source"] = "claude_api"
        return result

    except Exception as e:
        # Graceful fallback
        result = rule_based_analysis(scan_result, financials)
        result["_fallback_reason"] = str(e)
        result["source"] = "rule_based_fallback"
        return result


# ──────────────────────────────────────────────────────────────
#  TELEGRAM ALERTS  (free — just needs bot token)
# ──────────────────────────────────────────────────────────────

def send_telegram_alert(bot_token: str, chat_id: str, scan_result: dict) -> bool:
    """
    Send A+ breakout alert to Telegram.
    Setup: Create bot via @BotFather → get token.
           Get your chat_id via @userinfobot.
    """
    if not bot_token or not chat_id:
        return False

    sym   = scan_result.get("symbol","")
    grade = scan_result.get("trade_grade","")
    sig   = scan_result.get("signal","")
    cmp   = scan_result.get("cmp",0)
    vol   = scan_result.get("vol_ratio",0)
    trend = scan_result.get("trend","")
    rsi   = scan_result.get("rsi",0)
    conf  = scan_result.get("confidence",0)
    pat   = scan_result.get("pattern","").replace("_"," ")
    bl    = scan_result.get("breakout_level",0)
    setup = scan_result.get("atr_setup", {})
    atr   = scan_result.get("_atr",0)

    grade_emoji = {"A+":"🌟","A":"🔥","B":"⚡","C":"⚠️"}.get(grade,"📊")

    msg = f"""
{grade_emoji} <b>NSE BREAKOUT ALERT</b>

<b>{sym}</b> | Grade: <b>{grade}</b>
📊 Pattern: {pat}
🎯 Signal: {sig.replace('_',' ')}
💰 CMP: ₹{cmp:,.2f}

📈 Trend: {trend}
📉 RSI: {rsi}
📦 Volume: {vol:.1f}x average
🔥 Breakout Zone: ₹{bl:,.2f}
⭐ Confidence: {conf}%

"""
    if setup:
        msg += f"""💡 <b>TRADE SETUP</b>
Entry: ₹{setup.get('entry',cmp):,.2f}
SL   : ₹{setup.get('sl',0):,.2f} ({setup.get('sl_pct',0):.1f}%)
T1   : ₹{setup.get('t1',0):,.2f} (+{setup.get('t1_pct',0):.1f}%)
T2   : ₹{setup.get('t2',0):,.2f} (+{setup.get('t2_pct',0):.1f}%)
R:R  : {setup.get('rr','N/A')}

"""

    from datetime import datetime
    from datetime import timezone, timedelta
    IST = timezone(timedelta(hours=5, minutes=30))
    msg += f"<i>🕐 {datetime.now(IST).strftime('%d %b %Y %H:%M IST')}</i>\n"
    msg += "<i>⚠️ Not SEBI advice. DYOR.</i>"

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, data={
            "chat_id"   : chat_id,
            "text"      : msg.strip(),
            "parse_mode": "HTML",
        }, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
