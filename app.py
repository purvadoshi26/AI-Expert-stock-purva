# ─────────────────────────────────────────────────────────────────────────────
#  NSE Breakout Scanner v5 — with AI Expert Analysis
#  New: Expert Buy/Target/SL · Stock Lookup · News · Financials
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timezone, timedelta

from stocks      import UNIVERSES
from scanner     import fetch_batch, scan_stock, fetch_one, add_indicators, check_conditions, detect_trend
from ai_analyst  import get_news, get_financials, get_ai_analysis

# ── IST ───────────────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))
def now_ist():
    return datetime.now(IST)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Breakout Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #080c18 !important;
    color: #dde3f0 !important;
}

/* ── Cards ── */
.card {
    background: linear-gradient(135deg,#0f1629 0%,#141e35 100%);
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    padding: 18px 20px;
    margin: 8px 0;
}
.card-green  { border-left: 4px solid #10b981; }
.card-orange { border-left: 4px solid #f59e0b; }
.card-purple { border-left: 4px solid #8b5cf6; }
.card-red    { border-left: 4px solid #ef4444; }
.card-blue   { border-left: 4px solid #3b82f6; }

/* ── Badges ── */
.badge { border-radius: 20px; padding: 3px 12px; font-size:0.74rem; font-weight:700; display:inline-block; margin:2px; }
.b-green  { background:#052e16; color:#34d399; border:1px solid #059669; }
.b-orange { background:#451a03; color:#fb923c; border:1px solid #ea580c; }
.b-red    { background:#2d0a0a; color:#f87171; border:1px solid #dc2626; }
.b-purple { background:#1c1b4b; color:#a78bfa; border:1px solid #7c3aed; }
.b-blue   { background:#0f2040; color:#60a5fa; border:1px solid #2563eb; }
.b-teal   { background:#022020; color:#2dd4bf; border:1px solid #0d9488; }
.b-yellow { background:#261900; color:#fbbf24; border:1px solid #d97706; }

/* ── KPI tiles ── */
.kpi-tile {
    background: linear-gradient(135deg,#0f1629,#141e35);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
}
.kpi-val { font-size:1.8rem; font-weight:800; line-height:1; }
.kpi-lab { font-size:0.7rem; color:#4b5563; margin-top:4px; }

/* ── Expert Analysis ── */
.expert-header {
    font-size:1.05rem; font-weight:800; color:#f59e0b;
    border-bottom:1px solid #1e2d4a; padding-bottom:6px; margin-bottom:12px;
}
.tf-card {
    background:#0d1420;
    border:1px solid #1e2d4a;
    border-radius:12px;
    padding:14px;
    height:100%;
}
.price-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:5px 0; border-bottom:1px solid #0f1a2e;
}
.price-label { font-size:0.72rem; color:#4b5563; }
.price-val   { font-size:0.95rem; font-weight:800; }

/* ── Confidence bar ── */
.cbar-outer { background:#1e2840; border-radius:99px; height:6px; width:100%; margin-top:3px; }
.cbar-inner { border-radius:99px; height:6px; }

/* ── News ── */
.news-item {
    background:#0d1420; border:1px solid #1e2d4a;
    border-radius:10px; padding:12px 14px; margin:5px 0;
}
.news-title  { font-size:0.88rem; font-weight:600; color:#dde3f0; }
.news-meta   { font-size:0.7rem; color:#4b5563; margin-top:3px; }

/* ── Main title ── */
.main-title {
    font-size:2rem; font-weight:800; line-height:1.1;
    background: linear-gradient(90deg,#34d399,#60a5fa,#a78bfa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background:#070b14 !important;
    border-right:1px solid #121d30;
}

/* ── Table ── */
.stDataFrame { border-radius:10px; overflow:hidden; }

/* Metrics */
div[data-testid="stMetricValue"] { font-size:1.3rem; font-weight:800; color:#34d399 !important; }
div[data-testid="stMetricLabel"] { font-size:0.7rem; color:#4b5563; }

/* Tab headers */
.stTabs [data-baseweb="tab"] { font-weight:600; color:#4b5563; }
.stTabs [aria-selected="true"] { color:#60a5fa !important; }

.disclaimer {
    background:#1a0f0a; border:1px solid #451a03;
    border-radius:8px; padding:10px 14px;
    font-size:0.72rem; color:#fb923c; margin-top:10px;
}
</style>
""", unsafe_allow_html=True)


# ── API KEY from Streamlit secrets ─────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return ""

API_KEY = get_api_key()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build Chart
# ══════════════════════════════════════════════════════════════════════════════

def build_chart(sel, cdf, title_override=None):
    chart_title = title_override or (
        f"{sel['symbol']}  ·  {sel['pattern']}  ·  {sel['signal']}  ·  Conf {sel['confidence']}%"
    )
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.54, 0.17, 0.145, 0.145],
        subplot_titles=[chart_title, "Volume", "RSI (14)", "MACD"],
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=cdf.index,
        open=cdf["Open"], high=cdf["High"],
        low=cdf["Low"],   close=cdf["Close"],
        name="Price",
        increasing=dict(line=dict(color="#10b981"), fillcolor="rgba(16,185,129,0.25)"),
        decreasing=dict(line=dict(color="#ef4444"), fillcolor="rgba(239,68,68,0.25)"),
    ), row=1, col=1)

    for col_name, color, dash, name in [
        ("EMA20",  "#60a5fa", "solid", "EMA 20"),
        ("EMA50",  "#fb923c", "solid", "EMA 50"),
        ("SMA200", "#a78bfa", "dot",   "SMA 200"),
    ]:
        if col_name in cdf.columns and not cdf[col_name].isna().all():
            fig.add_trace(go.Scatter(
                x=cdf.index, y=cdf[col_name], name=name,
                line=dict(color=color, width=1.3, dash=dash)
            ), row=1, col=1)

    if "BB_Upper" in cdf.columns:
        fig.add_trace(go.Scatter(
            x=cdf.index, y=cdf["BB_Upper"],
            line=dict(color="rgba(148,163,184,0.2)", width=0.8), showlegend=False, name="BB Upper"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=cdf.index, y=cdf["BB_Lower"],
            line=dict(color="rgba(148,163,184,0.2)", width=0.8),
            fill="tonexty", fillcolor="rgba(148,163,184,0.03)", showlegend=False, name="BB Lower"
        ), row=1, col=1)

    # Levels
    for level, color, label, pos in [
        (sel.get("breakout_level"), "#f59e0b", "Breakout Zone", "top right"),
        (sel.get("support"),        "#10b981", "Support",       "bottom right"),
        (sel.get("resistance"),     "#ef4444", "Resistance",    "top right"),
    ]:
        if level:
            fig.add_hline(y=level, line_dash="dash", line_color=color, line_width=1.3,
                          annotation_text=f"{label} ₹{level:,.2f}",
                          annotation_font_color=color,
                          annotation_position=pos, row=1, col=1)

    # Volume
    vol_colors = ["#10b981" if float(c) >= float(o) else "#ef4444"
                  for c, o in zip(cdf["Close"], cdf["Open"])]
    fig.add_trace(go.Bar(x=cdf.index, y=cdf["Volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.75), row=2, col=1)
    if "Vol_MA20" in cdf.columns:
        fig.add_trace(go.Scatter(x=cdf.index, y=cdf["Vol_MA20"], name="Vol MA20",
                                 line=dict(color="#fb923c", width=1.1)), row=2, col=1)

    # RSI
    if "RSI" in cdf.columns:
        fig.add_trace(go.Scatter(x=cdf.index, y=cdf["RSI"], name="RSI",
                                 line=dict(color="#60a5fa", width=1.3)), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.06)", line_width=0, row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(16,185,129,0.06)", line_width=0, row=3, col=1)
        fig.add_hline(y=70, line_color="#f87171", line_width=0.8, line_dash="dot", row=3, col=1)
        fig.add_hline(y=30, line_color="#34d399", line_width=0.8, line_dash="dot", row=3, col=1)

    # MACD
    if "MACD" in cdf.columns and "MACD_Signal" in cdf.columns:
        fig.add_trace(go.Scatter(x=cdf.index, y=cdf["MACD"], name="MACD",
                                 line=dict(color="#60a5fa", width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=cdf["MACD_Signal"], name="Signal",
                                 line=dict(color="#fb923c", width=1.2)), row=4, col=1)
        if "MACD_Hist" in cdf.columns:
            hist_colors = ["#10b981" if float(v) >= 0 else "#ef4444"
                           for v in cdf["MACD_Hist"].fillna(0)]
            fig.add_trace(go.Bar(x=cdf.index, y=cdf["MACD_Hist"], name="Histogram",
                                 marker_color=hist_colors, opacity=0.65), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=750,
        paper_bgcolor="#080c18",
        plot_bgcolor="#0d1220",
        showlegend=True,
        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=8, t=55, b=8),
        font=dict(family="Space Grotesk"),
    )
    fig.update_yaxes(gridcolor="#121d30", zeroline=False)
    fig.update_xaxes(gridcolor="#121d30", zeroline=False, showspikes=True, spikecolor="#1e2d4a")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Expert Analysis UI
# ══════════════════════════════════════════════════════════════════════════════

def show_expert_analysis(ai: dict, symbol: str):
    if "error" in ai:
        st.error(f"AI analysis error: {ai['error']}")
        return

    overall = ai.get("overall", {})
    rec     = overall.get("recommendation", "WAIT")
    sent    = overall.get("sentiment", "NEUTRAL")
    conv    = overall.get("conviction", "LOW")

    rec_color  = {"BUY": "#10b981", "WAIT": "#f59e0b", "AVOID": "#ef4444"}.get(rec, "#94a3b8")
    sent_color = {"BULLISH": "#10b981", "NEUTRAL": "#94a3b8", "BEARISH": "#ef4444"}.get(sent, "#94a3b8")

    # ── Overall Card ──────────────────────────────────────────
    st.markdown(f"""
    <div class="card card-{'green' if rec=='BUY' else ('orange' if rec=='WAIT' else 'red')}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:0.72rem; color:#4b5563; margin-bottom:4px;">🤖 AI EXPERT RECOMMENDATION  ·  {symbol}</div>
                <div style="font-size:1.8rem; font-weight:800; color:{rec_color};">{rec}</div>
                <div style="font-size:0.88rem; color:{sent_color}; margin-top:2px;">{sent} · Conviction: {conv}</div>
            </div>
            <div style="text-align:right; font-size:0.75rem; color:#4b5563; max-width:60%;">
                {overall.get('summary', '')}
            </div>
        </div>
        <div style="margin-top:10px; font-size:0.78rem; color:#f87171;">
            ⚠️ Key Risk: {overall.get('key_risk', 'N/A')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Timeframe Cards ───────────────────────────────────────
    col_i, col_s, col_l = st.columns(3)

    def tf_card(col, tf_key, icon, title, color):
        tf = ai.get(tf_key, {})
        if not tf:
            return
        view       = tf.get("view", "N/A")
        suitable   = tf.get("suitable", False)
        risk       = tf.get("risk_level", "MEDIUM")
        risk_color = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}.get(risk, "#94a3b8")
        view_color = {"LONG": "#10b981", "ACCUMULATE": "#10b981", "SHORT": "#ef4444", "AVOID": "#4b5563"}.get(view, "#94a3b8")

        buy   = tf.get("buy_price", 0)
        sl    = tf.get("sl", 0)
        t1    = tf.get("target1", 0)
        t2    = tf.get("target2", 0)
        t3    = tf.get("target3")
        rr    = tf.get("risk_reward", "N/A")
        hold  = tf.get("holding_period", tf.get("time_in_trade", "N/A"))
        note  = tf.get("note", "")
        t1pct = tf.get("t1_pct", 0)
        t2pct = tf.get("t2_pct", 0)
        slpct = tf.get("sl_pct", 0)
        fview = tf.get("fundamental_view", "")

        with col:
            st.markdown(f"""
            <div class="tf-card">
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <div style="font-size:1.05rem; font-weight:800; color:{color};">{icon} {title}</div>
                    <div>
                        <span class="badge b-{'green' if view in ('LONG','ACCUMULATE') else ('red' if view=='SHORT' else 'purple')}">{view}</span>
                        <span class="badge b-{'green' if risk=='LOW' else ('yellow' if risk=='MEDIUM' else 'red')}">{risk} RISK</span>
                    </div>
                </div>

                <div style="font-size:0.7rem; color:#4b5563; margin-bottom:6px;">{'✅ Suitable' if suitable else '⚠️ Not ideal for this timeframe'}</div>

                <div class="price-row">
                    <span class="price-label">📍 Buy Zone</span>
                    <span class="price-val" style="color:#fbbf24;">{tf.get('entry_zone', f'₹{buy:,.2f}')}</span>
                </div>
                <div class="price-row">
                    <span class="price-label">🎯 Target 1</span>
                    <span class="price-val" style="color:#34d399;">₹{t1:,.2f} <span style="font-size:0.72rem; color:#4b5563;">(+{t1pct:.1f}%)</span></span>
                </div>
                <div class="price-row">
                    <span class="price-label">🎯 Target 2</span>
                    <span class="price-val" style="color:#10b981;">₹{t2:,.2f} <span style="font-size:0.72rem; color:#4b5563;">(+{t2pct:.1f}%)</span></span>
                </div>
                {'<div class="price-row"><span class="price-label">🎯 Target 3</span><span class="price-val" style="color:#059669;">₹' + f'{t3:,.2f}' + '</span></div>' if t3 and t3 > 0 else ''}
                <div class="price-row">
                    <span class="price-label">🛑 Stop Loss</span>
                    <span class="price-val" style="color:#f87171;">₹{sl:,.2f} <span style="font-size:0.72rem; color:#4b5563;">(-{abs(slpct):.1f}%)</span></span>
                </div>
                <div class="price-row" style="border:none;">
                    <span class="price-label">⚖️ Risk:Reward</span>
                    <span class="price-val" style="color:#a78bfa;">{rr}</span>
                </div>

                <div style="background:#080c18; border-radius:8px; padding:8px; margin-top:8px;">
                    <div style="font-size:0.7rem; color:#4b5563;">📅 Holding Period</div>
                    <div style="font-size:0.82rem; font-weight:700; color:#dde3f0;">{hold}</div>
                </div>

                {f'<div style="background:#080c18; border-radius:8px; padding:8px; margin-top:6px; font-size:0.72rem; color:#6b7280;">{fview}</div>' if fview else ''}

                <div style="font-size:0.72rem; color:#6b7280; margin-top:8px; border-top:1px solid #1e2d4a; padding-top:6px;">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    tf_card(col_i, "intraday",  "⚡", "Intraday",  "#60a5fa")
    tf_card(col_s, "swing",     "📊", "Swing",     "#fb923c")
    tf_card(col_l, "longterm",  "📈", "Long Term", "#10b981")

    # ── Risk Ranking ──────────────────────────────────────────
    rr = ai.get("risk_ranking", [])
    if rr:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Risk Ranking (Low → High Risk)")
        rcols = st.columns(len(rr))
        for i, item in enumerate(rr):
            risk = item.get("risk", "MEDIUM")
            rcolor = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}.get(risk, "#94a3b8")
            rcols[i].markdown(f"""
            <div class="card" style="text-align:center; border-left: 4px solid {rcolor}; padding:12px;">
                <div style="font-size:1.4rem; font-weight:800; color:{rcolor};">#{i+1}</div>
                <div style="font-size:0.88rem; font-weight:700; color:#dde3f0;">{item.get('timeframe','')}</div>
                <div style="font-size:0.78rem; color:{rcolor}; font-weight:700; margin-top:2px;">{risk} RISK</div>
                <div style="font-size:0.7rem; color:#4b5563; margin-top:4px;">{item.get('reason','')}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── News Impact ───────────────────────────────────────────
    ni = ai.get("news_impact", "")
    if ni:
        st.markdown(f"""
        <div class="card card-blue" style="margin-top:10px;">
            <div style="font-size:0.72rem; color:#60a5fa; font-weight:700;">📰 NEWS IMPACT ON STOCK</div>
            <div style="font-size:0.85rem; color:#dde3f0; margin-top:6px;">{ni}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown(f"""
    <div class="disclaimer">
        ⚠️ {ai.get('disclaimer', 'AI-generated analysis for educational purposes only. Not SEBI registered investment advice. Do your own research.')}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: News UI
# ══════════════════════════════════════════════════════════════════════════════

def show_news(news: list):
    if not news:
        st.info("No recent news found for this stock.")
        return
    for n in news:
        st.markdown(f"""
        <div class="news-item">
            <a href="{n['link']}" target="_blank" style="text-decoration:none;">
                <div class="news-title">{n['title']}</div>
            </a>
            <div class="news-meta">📰 {n['source']}  ·  🗓️ {n['date']}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Financials UI
# ══════════════════════════════════════════════════════════════════════════════

def show_financials(fin: dict):
    if not fin:
        st.info("Financial data not available.")
        return

    def val(key, suffix="", prefix=""):
        v = fin.get(key)
        return f"{prefix}{v:,.2f}{suffix}" if v is not None else "N/A"

    st.markdown(f"""
    <div class="card card-blue">
        <div style="font-size:1.0rem; font-weight:800; color:#dde3f0;">{fin.get('company_name', '')}</div>
        <div style="font-size:0.78rem; color:#4b5563;">{fin.get('sector','')} · {fin.get('industry','')}</div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Market Cap",    val('market_cap_cr', ' Cr', '₹'))
    f2.metric("P/E (TTM)",     val('pe_trailing'))
    f3.metric("P/E (Forward)", val('pe_forward'))
    f4.metric("P/B Ratio",     val('pb_ratio'))

    f5, f6, f7, f8 = st.columns(4)
    f5.metric("ROE",           val('roe_pct', '%'))
    f6.metric("Debt/Equity",   val('debt_equity'))
    f7.metric("Rev Growth",    val('revenue_growth', '%'))
    f8.metric("EPS (TTM)",     val('eps_ttm', '', '₹'))

    f9, f10, f11, f12 = st.columns(4)
    f9.metric("52W High",      val('week52_high', '', '₹'))
    f10.metric("52W Low",      val('week52_low',  '', '₹'))
    f11.metric("Beta",         val('beta'))
    f12.metric("Div Yield",    val('dividend_yield', '%'))


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Scanner Settings")

    universe_keys = [k for k, v in UNIVERSES.items() if v]
    universe_choice = st.selectbox("Stock Universe", universe_keys, index=0)
    symbols = UNIVERSES[universe_choice]

    timeframe_choice = st.selectbox("Timeframe", [
        "Daily  (6 months)",
        "Weekly (2 years)",
        "1-Hour (60 days) ⚡",
    ], index=0)

    if "Daily"  in timeframe_choice: interval, period, tf_label = "1d",  "6mo", "Daily"
    elif "Weekly" in timeframe_choice: interval, period, tf_label = "1wk", "2y",  "Weekly"
    else:                               interval, period, tf_label = "1h",  "60d", "1-Hour"

    st.markdown("---")
    st.markdown("### 🔎 Filters")

    min_conf = st.slider("Min Confidence %", 40, 85, 55, 5)

    signal_filter = st.multiselect("Signals", [
        "BREAKOUT", "CONSOLIDATION_BREAKOUT", "NEAR_BREAKOUT",
        "RETEST_BREAKOUT", "BREAKDOWN", "NEAR_BREAKDOWN",
        "REVERSAL_SIGNAL", "FALSE_BREAKOUT",
    ], default=["BREAKOUT", "CONSOLIDATION_BREAKOUT", "NEAR_BREAKOUT"])

    trend_filter = st.multiselect("Trend", [
        "UPTREND", "SIDEWAYS", "DOWNTREND"
    ], default=["UPTREND", "SIDEWAYS"])

    pattern_filter = st.multiselect("Patterns (empty=all)", [
        "HORIZONTAL_BREAKOUT","52W_HIGH_BREAKOUT","CONSOLIDATION",
        "CUP_AND_HANDLE","BULL_FLAG","DOUBLE_BOTTOM","DOUBLE_TOP",
        "HEAD_AND_SHOULDERS","INV_HEAD_AND_SHOULDERS",
        "ASCENDING_TRIANGLE","DESCENDING_TRIANGLE","SYMMETRICAL_TRIANGLE",
        "ROUNDING_BOTTOM","VOLATILITY_SQUEEZE","SUPPORT_BOUNCE",
        "W_PATTERN","RECTANGLE_RANGE","GOLDEN_CROSS","BULLISH_ENGULFING",
        "HAMMER","MORNING_STAR","THREE_WHITE_SOLDIERS","BULLISH_MARUBOZU",
    ], default=[])

    st.markdown("---")

    # AI key status
    if API_KEY:
        st.markdown('<span class="badge b-green">🤖 AI Analysis: ON</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge b-red">🤖 AI Analysis: OFF</span>', unsafe_allow_html=True)
        st.caption("Add ANTHROPIC_API_KEY to Streamlit secrets to enable AI Expert View")

    auto_refresh = st.toggle("⏰ Auto Refresh", value=False)
    refresh_mins = st.select_slider("Refresh every", [5, 10, 15, 30], value=10) if auto_refresh else None

    st.markdown("---")
    st.markdown("Built by [Purva Doshi](https://linkedin.com/in/purvadoshi26)  \nFree · NSE Equity only")


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-title">📈 NSE Breakout Scanner</div>', unsafe_allow_html=True)
    st.caption("Patterns · Candlesticks · DMA · AI Expert Analysis · News · Financials")
with col_h2:
    ist_now = now_ist()
    st.markdown(f"""
    <div style="text-align:right; margin-top:10px;">
        <div style="font-size:1.4rem; font-weight:800; color:#34d399;">{ist_now.strftime('%H:%M:%S')}</div>
        <div style="font-size:0.72rem; color:#4b5563;">{ist_now.strftime('%d %b %Y')} IST</div>
        <div style="font-size:0.65rem; color:#1f2a40;">Yahoo Finance · 15-min delay</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #121d30; margin:8px 0;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE TABS
# ══════════════════════════════════════════════════════════════════════════════

main_tab1, main_tab2 = st.tabs(["🔍 Sector Scanner", "🔎 Stock Lookup"])


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1: SECTOR SCANNER
# ──────────────────────────────────────────────────────────────────────────────

with main_tab1:
    scan_col, i1, i2, i3 = st.columns([2.5, 1, 1, 1])
    with scan_col:
        scan_btn = st.button("🔍  RUN SCANNER", type="primary",
                             use_container_width=True, key="scan_sector_btn")
    with i1: st.metric("Universe", f"{len(symbols)} stocks")
    with i2: st.metric("Timeframe", tf_label)
    with i3: st.metric("Min Conf",  f"{min_conf}%")

    # ── SCAN ──────────────────────────────────────────────────
    BATCH_SIZE = 20

    def run_scan(syms, period, interval):
        results, failed = [], []
        n       = len(syms)
        prog    = st.progress(0, text="Starting…")
        status  = st.empty()
        t0      = time.time()
        batches = [syms[i:i+BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]

        for b_idx, batch in enumerate(batches):
            status.markdown(f"**Batch {b_idx+1}/{len(batches)}** — {', '.join(batch[:4])}…")
            raw = fetch_batch(batch, period=period, interval=interval)
            for sym in batch:
                df = raw.get(sym)
                r  = scan_stock(sym, df) if df is not None else None
                if r:
                    results.append(r)
                else:
                    failed.append(sym)
            done = min((b_idx + 1) * BATCH_SIZE, n)
            elapsed = time.time() - t0
            spd  = done / elapsed if elapsed > 0 else 1
            rem  = int((n - done) / spd) if (n - done) > 0 else 0
            prog.progress(done / n, text=f"{done}/{n} scanned · ~{rem}s remaining")
            time.sleep(0.05)

        prog.empty(); status.empty()
        return results, failed


    if scan_btn:
        results, failed = run_scan(symbols, period, interval)
        st.session_state["results"] = results
        st.session_state["failed"]  = failed
        st.session_state["scan_ts"] = now_ist().strftime("%H:%M:%S IST")

    if "results" not in st.session_state:
        st.markdown("""
        <div style='text-align:center; padding:50px 0; color:#1e2d4a;'>
            <div style='font-size:4rem;'>📊</div>
            <div style='font-size:1.2rem; font-weight:600; color:#374151; margin-top:10px;'>
                Select universe & press Run Scanner
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Filter ────────────────────────────────────────────────
    all_results = st.session_state["results"]
    filtered = [
        r for r in all_results
        if r["confidence"] >= min_conf
        and r["signal"]    in (signal_filter if signal_filter else [r["signal"]])
        and r["trend"]     in (trend_filter  if trend_filter  else [r["trend"]])
        and (not pattern_filter
             or r["pattern"] in pattern_filter
             or any(p in r.get("all_patterns", []) for p in pattern_filter))
    ]
    filtered.sort(key=lambda x: x["rank_score"], reverse=True)

    # ── KPIs ──────────────────────────────────────────────────
    n_bo   = sum(1 for r in filtered if "BREAKOUT" in r["signal"] and "NEAR" not in r["signal"] and "FALSE" not in r["signal"])
    n_near = sum(1 for r in filtered if "NEAR" in r["signal"])
    n_rev  = sum(1 for r in filtered if "REVERSAL" in r["signal"])
    n_high = sum(1 for r in filtered if r["confidence"] >= 70)

    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    def kpi(col, val, label, color):
        col.markdown(f'<div class="kpi-tile"><div class="kpi-val" style="color:{color};">{val}</div><div class="kpi-lab">{label}</div></div>', unsafe_allow_html=True)

    kpi(k1, len(all_results), "Scanned",        "#60a5fa")
    kpi(k2, len(filtered),    "Signals Found",   "#34d399")
    kpi(k3, n_bo,             "🔥 Breakouts",    "#10b981")
    kpi(k4, n_near,           "⚡ Near Breakout", "#fb923c")
    kpi(k5, n_rev,            "🔄 Reversals",    "#a78bfa")
    kpi(k6, n_high,           "⭐ High Conf",    "#fbbf24")

    st.markdown(f"<div style='text-align:right; font-size:0.72rem; color:#374151; margin-top:4px;'>Last scan: {st.session_state.get('scan_ts','')}</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #121d30;'>", unsafe_allow_html=True)

    if not filtered:
        st.warning("No stocks matched filters. Lower confidence or add more signal types.")
        st.stop()

    # ── Signal Tabs ───────────────────────────────────────────
    bo_rows   = [r for r in filtered if "BREAKOUT" in r["signal"] and "NEAR" not in r["signal"] and "FALSE" not in r["signal"]]
    near_rows = [r for r in filtered if "NEAR" in r["signal"]]
    all_rows  = filtered

    t1, t2, t3 = st.tabs([
        f"🔥 Breakouts ({len(bo_rows)})",
        f"⚡ Near Breakout ({len(near_rows)})",
        f"📋 All ({len(all_rows)})",
    ])

    # Signal badge helpers
    def sbadge(sig):
        if "FALSE"    in sig: return f'<span class="badge b-red">❌ {sig}</span>'
        if "BREAKDOWN" in sig: return f'<span class="badge b-red">📉 {sig}</span>'
        if "NEAR_BREAKOUT" in sig: return f'<span class="badge b-orange">⚡ {sig}</span>'
        if "BREAKOUT" in sig: return f'<span class="badge b-green">🔥 {sig}</span>'
        if "REVERSAL" in sig: return f'<span class="badge b-yellow">🔄 {sig}</span>'
        if "CONSOLIDATION" in sig: return f'<span class="badge b-purple">📦 {sig}</span>'
        return f'<span class="badge b-teal">📊 {sig}</span>'

    def tbadge(t):
        c = {"UPTREND":"#34d399","DOWNTREND":"#f87171","SIDEWAYS":"#94a3b8"}.get(t,"#94a3b8")
        e = {"UPTREND":"📈","DOWNTREND":"📉","SIDEWAYS":"➡️"}.get(t,"")
        return f'<span style="color:{c}; font-size:0.8rem;">{e} {t}</span>'

    def conf_bar_html(pct):
        c = "#10b981" if pct >= 70 else ("#f59e0b" if pct >= 55 else "#ef4444")
        lbl = "HIGH" if pct >= 70 else ("MEDIUM" if pct >= 55 else "LOW")
        return f'<span style="color:{c}; font-weight:800;">{pct}%</span> <span style="color:{c}; font-size:0.7rem;">{lbl}</span><div class="cbar-outer"><div class="cbar-inner" style="width:{pct}%; background:{c};"></div></div>'

    def ptag(p):
        return f'<span style="background:#0f2040; color:#60a5fa; border-radius:5px; padding:2px 7px; font-size:0.68rem; margin:2px; display:inline-block;">{p.replace("_"," ")}</span>'

    def show_result_rows(rows, tab_key):
        if not rows:
            st.info("No signals in this category.")
            return
        df_exp = pd.DataFrame([{
            "Rank": i+1, "Stock": r["symbol"], "CMP ₹": r["cmp"],
            "Pattern": r["pattern"], "Signal": r["signal"],
            "Confidence %": r["confidence"], "Conf Level": r["conf_label"],
            "Vol Ratio": r["vol_ratio"], "Breakout ₹": r["breakout_level"],
            "% from Level": r["pct_from_level"], "Trend": r["trend"],
            "RSI": r["rsi"], "Support ₹": r["support"], "Resistance ₹": r["resistance"],
            "Signal Count": r["signal_count"], "Rank Score": r["rank_score"],
        } for i, r in enumerate(rows)])
        csv_data = df_exp.to_csv(index=False)
        st.download_button("📥 Export CSV", csv_data,
                           f"nse_{tab_key}_{now_ist().strftime('%Y%m%d_%H%M')}.csv",
                           "text/csv", key=f"dl_{tab_key}_{id(rows)}")

        for i, r in enumerate(rows):
            c1, c2, c3, c4, c5 = st.columns([1.1, 1.7, 1.5, 1.4, 1.1])
            with c1:
                st.markdown(f"""
                <div style="font-size:0.65rem; color:#4b5563;">#{i+1} · Score {r['rank_score']}</div>
                <div style="font-size:1.1rem; font-weight:800; color:#dde3f0;">{r['symbol']}</div>
                <div style="font-size:1.3rem; font-weight:800; color:#34d399;">₹{r['cmp']:,.2f}</div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(sbadge(r["signal"]) + "<br>" + ptag(r["pattern"]), unsafe_allow_html=True)
                extras = [p for p in r.get("all_patterns",[]) if p != r["pattern"]][:2]
                if extras:
                    st.markdown(" ".join(ptag(p) for p in extras), unsafe_allow_html=True)
            with c3:
                st.markdown(conf_bar_html(r["confidence"]) + "<br>" + tbadge(r["trend"]), unsafe_allow_html=True)
            with c4:
                pct_c = "#34d399" if r["pct_from_level"] <= 2 else ("#f59e0b" if r["pct_from_level"] <= 5 else "#f87171")
                st.markdown(f"""
                <div style="font-size:0.7rem; color:#4b5563;">Breakout Zone</div>
                <div style="font-size:1rem; font-weight:700;">₹{r['breakout_level']:,.2f}</div>
                <div style="font-size:0.82rem; font-weight:700; color:{pct_c};">{r['pct_from_level']:+.2f}% from level</div>
                <div style="font-size:0.7rem; color:#4b5563; margin-top:2px;">Sup ₹{r['support']:,.0f} · Res ₹{r['resistance']:,.0f}</div>
                """, unsafe_allow_html=True)
            with c5:
                rsi_c = "#f87171" if r["rsi"]>70 else ("#34d399" if r["rsi"]<35 else "#94a3b8")
                st.markdown(f"""
                <div style="font-size:0.7rem; color:#4b5563;">RSI · Vol · Signals</div>
                <div style="font-size:0.88rem; font-weight:700;">
                    <span style="color:{rsi_c};">{r['rsi']}</span>
                    &nbsp;·&nbsp;<span style="color:#60a5fa;">{r['vol_ratio']}x</span>
                    &nbsp;·&nbsp;<span style="color:#fbbf24;">{r['signal_count']}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<hr style='border:1px solid #121d30; margin:6px 0;'>", unsafe_allow_html=True)

    with t1: show_result_rows(bo_rows,   "breakouts")
    with t2: show_result_rows(near_rows, "near_breakout")
    with t3: show_result_rows(all_rows,  "all_signals")

    # ── Chart + Expert Analysis ───────────────────────────────
    st.markdown("<hr style='border:1px solid #121d30;'>", unsafe_allow_html=True)
    st.markdown("### 📊 Chart & Expert Analysis")

    chart_opts = [r["symbol"] for r in filtered]
    sel_sym    = st.selectbox("Select stock:", chart_opts, key="sector_chart_sel")
    sel        = next((r for r in filtered if r["symbol"] == sel_sym), None)

    if sel and "_df" in sel:
        cdf = sel["_df"]

        # Chart
        fig = build_chart(sel, cdf)
        st.plotly_chart(fig, use_container_width=True)

        # Expert Analysis tabs
        ea1, ea2, ea3 = st.tabs(["🤖 Expert AI View", "📰 News", "💹 Financials"])

        with ea1:
            if not API_KEY:
                st.warning("Add ANTHROPIC_API_KEY to Streamlit secrets to enable AI Expert View. Go to app settings → Secrets → add `ANTHROPIC_API_KEY = 'your-key'`")
            else:
                ai_key = f"ai_{sel_sym}"
                if st.button(f"🤖 Get Expert Analysis for {sel_sym}", key=f"ai_btn_sector_{sel_sym}"):
                    with st.spinner("Fetching news & financials, then running AI analysis…"):
                        news_data = get_news(sel_sym)
                        fin_data  = get_financials(sel_sym)
                        ai_result = get_ai_analysis(sel_sym, sel, fin_data, news_data, API_KEY)
                    st.session_state[ai_key] = ai_result
                    st.session_state[ai_key+"_news"] = news_data
                    st.session_state[ai_key+"_fin"]  = fin_data

                if ai_key in st.session_state:
                    show_expert_analysis(st.session_state[ai_key], sel_sym)

        with ea2:
            news_key = f"ai_{sel_sym}_news"
            if news_key in st.session_state:
                show_news(st.session_state[news_key])
            else:
                if st.button(f"📰 Load News for {sel_sym}", key=f"news_btn_{sel_sym}"):
                    with st.spinner("Fetching news…"):
                        news_data = get_news(sel_sym)
                    st.session_state[news_key] = news_data
                    show_news(news_data)

        with ea3:
            fin_key = f"ai_{sel_sym}_fin"
            if fin_key in st.session_state:
                show_financials(st.session_state[fin_key])
            else:
                if st.button(f"💹 Load Financials for {sel_sym}", key=f"fin_btn_{sel_sym}"):
                    with st.spinner("Fetching financial data…"):
                        fin_data = get_financials(sel_sym)
                    st.session_state[fin_key] = fin_data
                    show_financials(fin_data)


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 2: STOCK LOOKUP
# ──────────────────────────────────────────────────────────────────────────────

with main_tab2:
    st.markdown("### 🔎 Analyse Any NSE Stock")
    st.caption("Type any NSE symbol (e.g. TATAMOTORS, RELIANCE, HDFCBANK) — get full chart + expert AI view")

    lookup_col1, lookup_col2, lookup_col3 = st.columns([2, 1, 1])
    with lookup_col1:
        lookup_sym = st.text_input("Enter NSE Symbol", placeholder="e.g. TATAMOTORS",
                                   key="lookup_input").strip().upper().replace(".NS", "")
    with lookup_col2:
        lookup_tf = st.selectbox("Timeframe", ["Daily (6mo)", "Weekly (2y)", "1-Hour (60d)"],
                                 key="lookup_tf")
    with lookup_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        lookup_btn = st.button("🔍 Analyse", type="primary", key="lookup_btn", use_container_width=True)

    if lookup_tf == "Daily (6mo)":     l_interval, l_period = "1d", "6mo"
    elif lookup_tf == "Weekly (2y)":   l_interval, l_period = "1wk", "2y"
    else:                               l_interval, l_period = "1h", "60d"

    if lookup_btn and lookup_sym:
        with st.spinner(f"Fetching data for {lookup_sym}…"):
            l_df = fetch_one(lookup_sym, l_period, l_interval)

        if l_df is None or len(l_df) < 10:
            st.error(f"Could not fetch data for {lookup_sym}. Check the symbol and try again.")
        else:
            l_df   = add_indicators(l_df)
            l_scan = scan_stock(lookup_sym, l_df)

            if l_scan is None:
                # Build minimal result manually for lookup
                cond    = check_conditions(l_df)
                trend,_ = detect_trend(l_df)
                l_scan  = {
                    "symbol"         : lookup_sym,
                    "cmp"            : round(cond["_close"], 2),
                    "pattern"        : "MANUAL_LOOKUP",
                    "signal"         : "ANALYSING",
                    "confidence"     : 50,
                    "conf_label"     : "MEDIUM",
                    "vol_ratio"      : round(cond["_vol_ratio"], 2),
                    "breakout_level" : cond["_resistance"],
                    "pct_from_level" : 0.0,
                    "trend"          : trend,
                    "rsi"            : round(cond["_rsi"], 1),
                    "ema20"          : cond["_ema20"],
                    "ema50"          : cond["_ema50"],
                    "sma200"         : cond["_sma200"],
                    "support"        : round(cond["_support"], 2),
                    "resistance"     : round(cond["_resistance"], 2),
                    "signal_count"   : 0,
                    "all_patterns"   : [],
                    "rank_score"     : 50,
                    "_df"            : l_df,
                }

            st.session_state["lookup_result"] = l_scan
            st.session_state["lookup_sym"]    = lookup_sym

    if "lookup_result" in st.session_state:
        lsel = st.session_state["lookup_result"]
        lsym = st.session_state["lookup_sym"]
        lcdf = lsel["_df"]

        # ── Quick stats ────────────────────────────────────────
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Symbol",      lsym)
        q2.metric("CMP",         f"₹{lsel['cmp']:,.2f}")
        q3.metric("Pattern",     lsel["pattern"].replace("_"," "))
        q4.metric("Signal",      lsel["signal"].replace("_"," "))
        q5.metric("RSI",         lsel["rsi"])

        # ── Chart ─────────────────────────────────────────────
        fig_l = build_chart(lsel, lcdf,
                            title_override=f"{lsym} — Pattern: {lsel['pattern']} · Trend: {lsel['trend']}")
        st.plotly_chart(fig_l, use_container_width=True)

        # ── Analysis Tabs ─────────────────────────────────────
        la1, la2, la3 = st.tabs(["🤖 Expert AI View", "📰 News", "💹 Financials"])

        with la1:
            if not API_KEY:
                st.warning("Add ANTHROPIC_API_KEY to Streamlit secrets to enable AI Expert View.")
            else:
                l_ai_key = f"lookup_ai_{lsym}"
                if st.button(f"🤖 Get Expert Analysis for {lsym}", key=f"l_ai_btn_{lsym}"):
                    with st.spinner("Fetching news & financials, then running AI analysis…"):
                        l_news = get_news(lsym)
                        l_fin  = get_financials(lsym)
                        l_ai   = get_ai_analysis(lsym, lsel, l_fin, l_news, API_KEY)
                    st.session_state[l_ai_key]         = l_ai
                    st.session_state[l_ai_key+"_news"] = l_news
                    st.session_state[l_ai_key+"_fin"]  = l_fin

                if l_ai_key in st.session_state:
                    show_expert_analysis(st.session_state[l_ai_key], lsym)

        with la2:
            l_nkey = f"lookup_ai_{lsym}_news"
            if l_nkey in st.session_state:
                show_news(st.session_state[l_nkey])
            else:
                if st.button(f"📰 Load News", key=f"l_news_btn_{lsym}"):
                    with st.spinner("Fetching news…"):
                        l_news_d = get_news(lsym)
                    st.session_state[l_nkey] = l_news_d
                    show_news(l_news_d)

        with la3:
            l_fkey = f"lookup_ai_{lsym}_fin"
            if l_fkey in st.session_state:
                show_financials(st.session_state[l_fkey])
            else:
                if st.button(f"💹 Load Financials", key=f"l_fin_btn_{lsym}"):
                    with st.spinner("Fetching financials…"):
                        l_fin_d = get_financials(lsym)
                    st.session_state[l_fkey] = l_fin_d
                    show_financials(l_fin_d)

    elif not lookup_sym:
        st.markdown("""
        <div style='text-align:center; padding:40px; color:#1e2d4a;'>
            <div style='font-size:3rem;'>🔍</div>
            <div style='font-size:1rem; color:#374151; margin-top:8px;'>
                Type any NSE symbol above and click Analyse
            </div>
            <div style='font-size:0.8rem; color:#1e2d4a; margin-top:4px;'>
                Works for any listed NSE equity — RELIANCE, TATAMOTORS, HDFCBANK, ZOMATO…
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Failed stocks (sidebar area, non-intrusive) ───────────────────────────────
if "failed" in st.session_state and st.session_state["failed"]:
    with st.expander(f"⚠️ {len(st.session_state['failed'])} stocks unavailable"):
        st.write(", ".join(st.session_state["failed"]))
        st.caption("These may be delisted or temporarily unavailable from Yahoo Finance.")


# ── Auto refresh ──────────────────────────────────────────────────────────────
if auto_refresh and refresh_mins:
    ph = st.empty()
    for remaining in range(refresh_mins * 60, 0, -1):
        ph.caption(f"🔄 Auto-refresh in {remaining//60}m {remaining%60:02d}s")
        time.sleep(1)
    ph.empty()
    st.rerun()
