# ─────────────────────────────────────────────────────────────────────────────
#  AI Analyst Module
#  Uses Anthropic API → Expert buy/target/SL for intraday, swing, long-term
#  News from yfinance · Financials from yfinance
# ─────────────────────────────────────────────────────────────────────────────

import yfinance as yf
import pandas as pd
import numpy as np
import json
import anthropic
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
#  NEWS
# ──────────────────────────────────────────────────────────────

def get_news(symbol: str, limit: int = 6) -> list:
    """Fetch recent news for a stock from yfinance."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        news   = ticker.news or []
        results = []
        for item in news[:limit]:
            content = item.get("content", {})
            title   = content.get("title", item.get("title", "No title"))
            link    = content.get("canonicalUrl", {}).get("url", item.get("link", "#"))
            source  = content.get("provider", {}).get("displayName", item.get("publisher", "Unknown"))
            pub_ts  = content.get("pubDate", item.get("providerPublishTime", None))
            if isinstance(pub_ts, (int, float)):
                from datetime import datetime
                pub_str = datetime.utcfromtimestamp(pub_ts).strftime("%d %b %Y")
            elif isinstance(pub_ts, str):
                pub_str = pub_ts[:10]
            else:
                pub_str = "Recent"
            results.append({"title": title, "link": link, "source": source, "date": pub_str})
        return results
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
#  FINANCIAL DATA
# ──────────────────────────────────────────────────────────────

def get_financials(symbol: str) -> dict:
    """Fetch key financial metrics from yfinance."""
    try:
        info = yf.Ticker(f"{symbol}.NS").info
        def safe(key, fmt="num"):
            val = info.get(key)
            if val is None or val != val:
                return None
            if fmt == "cr":
                return round(val / 1e7, 2)       # ₹ crores
            if fmt == "pct":
                return round(val * 100, 2)
            if fmt == "2f":
                return round(float(val), 2)
            return val

        mktcap = safe("marketCap", "cr")
        return {
            "company_name"    : info.get("longName") or info.get("shortName") or symbol,
            "sector"          : info.get("sector", "N/A"),
            "industry"        : info.get("industry", "N/A"),
            "market_cap_cr"   : mktcap,
            "pe_trailing"     : safe("trailingPE", "2f"),
            "pe_forward"      : safe("forwardPE", "2f"),
            "pb_ratio"        : safe("priceToBook", "2f"),
            "debt_equity"     : safe("debtToEquity", "2f"),
            "roe_pct"         : safe("returnOnEquity", "pct"),
            "revenue_growth"  : safe("revenueGrowth", "pct"),
            "earnings_growth" : safe("earningsGrowth", "pct"),
            "dividend_yield"  : safe("dividendYield", "pct"),
            "week52_high"     : safe("fiftyTwoWeekHigh", "2f"),
            "week52_low"      : safe("fiftyTwoWeekLow", "2f"),
            "avg50d"          : safe("fiftyDayAverage", "2f"),
            "avg200d"         : safe("twoHundredDayAverage", "2f"),
            "eps_ttm"         : safe("trailingEps", "2f"),
            "book_value"      : safe("bookValue", "2f"),
            "beta"            : safe("beta", "2f"),
            "free_cashflow_cr": safe("freeCashflow", "cr"),
        }
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────
#  AI EXPERT ANALYSIS  (Claude API)
# ──────────────────────────────────────────────────────────────

_EXPERT_PROMPT = """
You are a senior equity analyst with 20 years of experience in Indian stock markets (NSE/BSE).
A trader has asked for your expert analysis on {symbol}.

You have this data:

=== TECHNICAL DATA ===
Current Price (CMP): ₹{cmp}
Pattern Detected: {pattern}
Signal: {signal}
Confidence: {confidence}%
Trend: {trend}
RSI: {rsi}
Volume Ratio: {vol_ratio}x average
EMA 20: {ema20}
EMA 50: {ema50}
SMA 200: {sma200}
ATR (14): {atr}
Support Zone: ₹{support}
Resistance Zone: ₹{resistance}
Breakout Level: ₹{breakout_level}

=== RECENT OHLCV (last 15 sessions) ===
{ohlcv_table}

=== FUNDAMENTAL DATA ===
{fundamental_data}

=== RECENT NEWS ===
{news_summary}

=== YOUR TASK ===
Provide a structured expert analysis. You MUST respond with ONLY valid JSON (no markdown, no explanation outside JSON).

Return this exact structure:
{{
  "overall": {{
    "recommendation": "BUY" | "WAIT" | "AVOID",
    "conviction": "HIGH" | "MEDIUM" | "LOW",
    "summary": "2-3 line expert summary of the stock situation right now",
    "key_risk": "Single biggest risk for this stock right now",
    "sentiment": "BULLISH" | "NEUTRAL" | "BEARISH"
  }},
  "intraday": {{
    "suitable": true | false,
    "view": "LONG" | "SHORT" | "AVOID",
    "entry_zone": "₹X - ₹Y",
    "buy_price": number,
    "sl": number,
    "target1": number,
    "target2": number,
    "sl_pct": number,
    "t1_pct": number,
    "t2_pct": number,
    "risk_reward": "1:X",
    "time_in_trade": "2-4 hours",
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "note": "Short expert note for intraday"
  }},
  "swing": {{
    "suitable": true | false,
    "view": "LONG" | "SHORT" | "AVOID",
    "entry_zone": "₹X - ₹Y",
    "buy_price": number,
    "sl": number,
    "target1": number,
    "target2": number,
    "target3": number,
    "sl_pct": number,
    "t1_pct": number,
    "t2_pct": number,
    "t3_pct": number,
    "risk_reward": "1:X",
    "holding_period": "5-10 days",
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "note": "Short expert note for swing"
  }},
  "longterm": {{
    "suitable": true | false,
    "view": "ACCUMULATE" | "HOLD" | "AVOID",
    "entry_zone": "₹X - ₹Y",
    "buy_price": number,
    "sl": number,
    "target1": number,
    "target2": number,
    "target3": number,
    "sl_pct": number,
    "t1_pct": number,
    "t2_pct": number,
    "t3_pct": number,
    "risk_reward": "1:X",
    "holding_period": "6-12 months",
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "fundamental_view": "Based on fundamentals, is this worth holding long-term?",
    "note": "Short expert note for long-term"
  }},
  "risk_ranking": [
    {{"timeframe": "Intraday", "risk": "HIGH", "reason": "..."}},
    {{"timeframe": "Swing", "risk": "MEDIUM", "reason": "..."}},
    {{"timeframe": "Long Term", "risk": "LOW", "reason": "..."}}
  ],
  "news_impact": "How recent news impacts this stock (2 lines)",
  "disclaimer": "This is AI-generated analysis for educational purposes only. Not SEBI registered advice. Do your own research before investing."
}}

Be realistic. Use ATR for SL calculation. SL should be meaningful — not too tight, not too wide.
For intraday: SL = 0.5-1x ATR. For swing: SL = 1.5-2x ATR. For long-term: SL = below 200 DMA or major support.
Targets should use support/resistance zones and R:R of minimum 1:1.5.
"""


def get_ai_analysis(symbol: str, scan_result: dict, financials: dict, news: list, api_key: str) -> dict:
    """
    Call Claude API to get expert buy/target/SL analysis.
    Returns parsed dict or error dict.
    """
    try:
        df  = scan_result.get("_df")
        atr = None

        if df is not None and "ATR" in df.columns:
            atr = round(float(df["ATR"].iloc[-1]), 2)

        # Build OHLCV table (last 15 rows)
        ohlcv_str = "Date | Open | High | Low | Close | Volume\n"
        if df is not None:
            tail = df.tail(15)
            for idx, row in tail.iterrows():
                try:
                    date_str = idx.strftime("%d-%b") if hasattr(idx, "strftime") else str(idx)[:10]
                    ohlcv_str += f"{date_str} | {row['Open']:.1f} | {row['High']:.1f} | {row['Low']:.1f} | {row['Close']:.1f} | {int(row['Volume']):,}\n"
                except Exception:
                    continue

        # Fundamental data string
        fund_str = "Not available"
        if financials:
            parts = []
            if financials.get("company_name"): parts.append(f"Name: {financials['company_name']}")
            if financials.get("sector"):        parts.append(f"Sector: {financials['sector']}")
            if financials.get("market_cap_cr"): parts.append(f"Market Cap: ₹{financials['market_cap_cr']:,.0f} Cr")
            if financials.get("pe_trailing"):   parts.append(f"PE (TTM): {financials['pe_trailing']}")
            if financials.get("pe_forward"):    parts.append(f"PE (Fwd): {financials['pe_forward']}")
            if financials.get("pb_ratio"):      parts.append(f"P/B: {financials['pb_ratio']}")
            if financials.get("roe_pct"):       parts.append(f"ROE: {financials['roe_pct']}%")
            if financials.get("debt_equity"):   parts.append(f"Debt/Equity: {financials['debt_equity']}")
            if financials.get("revenue_growth"):parts.append(f"Revenue Growth: {financials['revenue_growth']}%")
            if financials.get("eps_ttm"):       parts.append(f"EPS: ₹{financials['eps_ttm']}")
            if financials.get("beta"):          parts.append(f"Beta: {financials['beta']}")
            if financials.get("week52_high"):   parts.append(f"52W High: ₹{financials['week52_high']}")
            if financials.get("week52_low"):    parts.append(f"52W Low: ₹{financials['week52_low']}")
            fund_str = " | ".join(parts) if parts else "Not available"

        # News string
        news_str = "No recent news available"
        if news:
            news_str = "\n".join([f"- [{n['date']}] {n['title']} ({n['source']})" for n in news[:5]])

        prompt = _EXPERT_PROMPT.format(
            symbol        = symbol,
            cmp           = scan_result.get("cmp", "N/A"),
            pattern       = scan_result.get("pattern", "N/A"),
            signal        = scan_result.get("signal", "N/A"),
            confidence    = scan_result.get("confidence", "N/A"),
            trend         = scan_result.get("trend", "N/A"),
            rsi           = scan_result.get("rsi", "N/A"),
            vol_ratio     = scan_result.get("vol_ratio", "N/A"),
            ema20         = scan_result.get("ema20") or "N/A",
            ema50         = scan_result.get("ema50") or "N/A",
            sma200        = scan_result.get("sma200") or "N/A",
            atr           = atr or "N/A",
            support       = scan_result.get("support", "N/A"),
            resistance    = scan_result.get("resistance", "N/A"),
            breakout_level= scan_result.get("breakout_level", "N/A"),
            ohlcv_table   = ohlcv_str,
            fundamental_data = fund_str,
            news_summary  = news_str,
        )

        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 2000,
            messages   = [{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        # Strip any markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {str(e)}", "raw": raw if "raw" in dir() else ""}
    except Exception as e:
        return {"error": str(e)}
