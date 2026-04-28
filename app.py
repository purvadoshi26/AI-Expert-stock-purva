# ─────────────────────────────────────────────────────────────────────────────
#  NSE Breakout Scanner v6 — Decision-Making Trading Assistant
#  New: Trade Grades · Top Picks · AND Filters · Caching · Telegram · Nifty Context
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timezone, timedelta

from stocks     import UNIVERSES
from scanner    import (fetch_batch, fetch_one, scan_stock, add_indicators,
                         check_conditions, detect_trend, get_market_context,
                         run_backtest)
from ai_analyst import (get_news, get_financials, get_ai_analysis,
                         send_telegram_alert, rule_based_analysis)

IST = timezone(timedelta(hours=5, minutes=30))
def now_ist(): return datetime.now(IST)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="NSE Breakout Scanner v6",
                   page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important;background:#07080f!important;color:#dde3f0!important;}

/* Grade badges */
.grade-aplus{font-size:1.6rem;font-weight:900;color:#fff;background:linear-gradient(135deg,#f59e0b,#ef4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;}
.grade-a    {font-size:1.6rem;font-weight:900;color:#10b981;}
.grade-b    {font-size:1.6rem;font-weight:900;color:#f59e0b;}
.grade-c    {font-size:1.6rem;font-weight:900;color:#6b7280;}

/* Cards */
.card{background:linear-gradient(135deg,#0e1526,#131f38);border:1px solid #1a2d47;border-radius:14px;padding:16px 18px;margin:6px 0;}
.card-gold  {border-left:5px solid #f59e0b!important;}
.card-green {border-left:5px solid #10b981!important;}
.card-orange{border-left:5px solid #fb923c!important;}
.card-red   {border-left:5px solid #ef4444!important;}
.card-blue  {border-left:5px solid #3b82f6!important;}

/* Top Picks */
.top-pick-card{
  background:linear-gradient(135deg,#1a1200,#1a2200);
  border:2px solid #f59e0b;border-radius:16px;padding:18px;
  position:relative;overflow:hidden;
}
.top-pick-card::before{
  content:'⭐';position:absolute;top:-10px;right:-5px;
  font-size:5rem;opacity:0.08;
}

/* Signal badges */
.badge{border-radius:20px;padding:3px 11px;font-size:.73rem;font-weight:700;display:inline-block;margin:2px;}
.b-green {background:#052e16;color:#34d399;border:1px solid #059669;}
.b-orange{background:#451a03;color:#fb923c;border:1px solid #ea580c;}
.b-red   {background:#2d0a0a;color:#f87171;border:1px solid #dc2626;}
.b-blue  {background:#0f2040;color:#60a5fa;border:1px solid #2563eb;}
.b-yellow{background:#261900;color:#fbbf24;border:1px solid #d97706;}
.b-purple{background:#1c1b4b;color:#a78bfa;border:1px solid #7c3aed;}
.b-gray  {background:#111827;color:#6b7280;border:1px solid #374151;}
.b-teal  {background:#022020;color:#2dd4bf;border:1px solid #0d9488;}

/* Conf bar */
.cb-out{background:#1e2840;border-radius:99px;height:5px;width:100%;margin-top:3px;}
.cb-in {border-radius:99px;height:5px;}

/* KPI */
.kpi{background:linear-gradient(135deg,#0e1526,#131f38);border:1px solid #1a2d47;border-radius:12px;padding:13px 10px;text-align:center;}
.kv{font-size:1.7rem;font-weight:800;line-height:1;}
.kl{font-size:.68rem;color:#4b5563;margin-top:3px;}

/* ATR Setup table */
.setup-row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #0f1a2e;}
.setup-label{font-size:.72rem;color:#4b5563;}
.setup-val  {font-size:.92rem;font-weight:800;}

/* News */
.news-card{background:#0a0f1e;border:1px solid #1a2d47;border-radius:10px;padding:11px 14px;margin:5px 0;}
.news-title{font-size:.87rem;font-weight:600;color:#dde3f0;}
.news-meta {font-size:.68rem;color:#4b5563;margin-top:3px;}

/* Strength bar */
.str-bar-out{background:#1e2840;border-radius:99px;height:8px;width:100%;}
.str-bar-in {border-radius:99px;height:8px;}

/* Main title */
.main-title{font-size:1.9rem;font-weight:800;background:linear-gradient(90deg,#34d399,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}

section[data-testid="stSidebar"]{background:#060810!important;border-right:1px solid #111d2e;}
div[data-testid="stMetricValue"]{font-size:1.25rem!important;font-weight:800!important;color:#34d399!important;}
div[data-testid="stMetricLabel"]{font-size:.68rem!important;color:#4b5563!important;}
.stTabs [data-baseweb="tab"]{font-weight:600;color:#4b5563;}
.stTabs [aria-selected="true"]{color:#60a5fa!important;}
.disclaimer{background:#1a0f0a;border:1px solid #451a03;border-radius:8px;padding:10px 14px;font-size:.72rem;color:#fb923c;margin-top:10px;}
</style>
""", unsafe_allow_html=True)


# ── Secrets ───────────────────────────────────────────────────
def _secret(k, default=""):
    try: return st.secrets.get(k, default)
    except: return default

API_KEY       = _secret("ANTHROPIC_API_KEY")
TG_BOT_TOKEN  = _secret("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID    = _secret("TELEGRAM_CHAT_ID")


# ══════════════════════════════════════════════════════════════
#  CACHING  (v6 — speeds up repeat scans significantly)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_batch(symbols_tuple: tuple, period: str, interval: str) -> dict:
    """Cache OHLCV fetch for 15 minutes."""
    return fetch_batch(list(symbols_tuple), period, interval)

@st.cache_data(ttl=300, show_spinner=False)
def cached_market_context() -> dict:
    return get_market_context()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_financials(symbol: str) -> dict:
    return get_financials(symbol)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_news(symbol: str) -> list:
    fin = cached_financials(symbol)
    return get_news(symbol, fin.get("company_name",""))


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

GRADE_COLORS = {"A+":"#f59e0b","A":"#10b981","B":"#f59e0b","C":"#6b7280"}
GRADE_BG     = {"A+":"#1a1200","A":"#052e16","B":"#261900","C":"#111827"}

def grade_badge(g):
    c = GRADE_COLORS.get(g,"#6b7280"); bg = GRADE_BG.get(g,"#111827")
    em= {"A+":"🌟","A":"🔥","B":"⚡","C":"⚠️"}.get(g,"")
    return f'<span style="background:{bg};color:{c};border:1.5px solid {c};border-radius:20px;padding:3px 12px;font-size:.78rem;font-weight:900;">{em} {g}</span>'

def signal_badge(sig):
    if "FALSE"    in sig: return f'<span class="badge b-gray">❌ {sig}</span>'
    if "BREAKDOWN" in sig: return f'<span class="badge b-red">📉 {sig}</span>'
    if "NEAR_BREAKOUT" in sig: return f'<span class="badge b-orange">⚡ {sig}</span>'
    if "BREAKOUT" in sig: return f'<span class="badge b-green">🔥 {sig}</span>'
    if "REVERSAL" in sig: return f'<span class="badge b-yellow">🔄 {sig}</span>'
    if "CONSOLIDATION" in sig: return f'<span class="badge b-blue">📦 {sig}</span>'
    return f'<span class="badge b-teal">📊 {sig}</span>'

def trend_badge(t):
    c = {"UPTREND":"#34d399","DOWNTREND":"#f87171","SIDEWAYS":"#94a3b8"}.get(t,"#94a3b8")
    e = {"UPTREND":"📈","DOWNTREND":"📉","SIDEWAYS":"➡️"}.get(t,"")
    return f'<span style="color:{c};font-size:.78rem;font-weight:700;">{e} {t}</span>'

def conf_bar(pct):
    c = "#10b981" if pct>=70 else ("#f59e0b" if pct>=55 else "#ef4444")
    lbl = "HIGH" if pct>=70 else ("MEDIUM" if pct>=55 else "LOW")
    return f'<span style="color:{c};font-weight:800;">{pct}%</span> <span style="color:{c};font-size:.7rem;">{lbl}</span><div class="cb-out"><div class="cb-in" style="width:{pct}%;background:{c};"></div></div>'

def strength_bar(s):
    c = "#10b981" if s>=70 else ("#f59e0b" if s>=45 else "#ef4444")
    return f'<div style="font-size:.68rem;color:#4b5563;margin-bottom:2px;">Strength {s}/100</div><div class="str-bar-out"><div class="str-bar-in" style="width:{s}%;background:{c};"></div></div>'

def ptag(p, ptype="chart"):
    clr={"chart":"#60a5fa","candle":"#c084fc","dma":"#2dd4bf"}.get(ptype,"#60a5fa")
    bg ={"chart":"#0f2040","candle":"#1a0f2e","dma":"#022020"}.get(ptype,"#0f2040")
    return f'<span style="background:{bg};color:{clr};border-radius:5px;padding:2px 7px;font-size:.67rem;margin:2px;display:inline-block;">{p.replace("_"," ")}</span>'

def rsi_color(r):
    if r>=70: return "#f87171"
    if r<=35: return "#34d399"
    return "#94a3b8"

def kpi(col, val, label, color="#34d399"):
    col.markdown(f'<div class="kpi"><div class="kv" style="color:{color};">{val}</div><div class="kl">{label}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CHART BUILDER
# ══════════════════════════════════════════════════════════════

def build_chart(sel, cdf, title_override=None):
    t = title_override or f"{sel['symbol']}  ·  {sel['pattern']}  ·  Grade {sel['trade_grade']}  ·  Conf {sel['confidence']}%"
    fig = make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=0.025,
                        row_heights=[0.54,0.17,0.145,0.145],subplot_titles=[t,"Volume","RSI (14)","MACD"])

    fig.add_trace(go.Candlestick(x=cdf.index,open=cdf["Open"],high=cdf["High"],
        low=cdf["Low"],close=cdf["Close"],name="Price",
        increasing=dict(line=dict(color="#10b981"),fillcolor="rgba(16,185,129,.25)"),
        decreasing=dict(line=dict(color="#ef4444"),fillcolor="rgba(239,68,68,.25)")),row=1,col=1)

    for col_n,color,dash,name in [("EMA20","#60a5fa","solid","EMA 20"),("EMA50","#fb923c","solid","EMA 50"),("SMA200","#a78bfa","dot","SMA 200")]:
        if col_n in cdf.columns and not cdf[col_n].isna().all():
            fig.add_trace(go.Scatter(x=cdf.index,y=cdf[col_n],name=name,line=dict(color=color,width=1.3,dash=dash)),row=1,col=1)

    if "BB_Upper" in cdf.columns:
        fig.add_trace(go.Scatter(x=cdf.index,y=cdf["BB_Upper"],line=dict(color="rgba(148,163,184,.2)",width=.8),showlegend=False,name="BB Upper"),row=1,col=1)
        fig.add_trace(go.Scatter(x=cdf.index,y=cdf["BB_Lower"],line=dict(color="rgba(148,163,184,.2)",width=.8),fill="tonexty",fillcolor="rgba(148,163,184,.03)",showlegend=False,name="BB Lower"),row=1,col=1)

    for level,color,label,pos in [(sel.get("breakout_level"),"#f59e0b","Breakout","top right"),(sel.get("support"),"#10b981","Support","bottom right"),(sel.get("resistance"),"#ef4444","Resistance","top right")]:
        if level:
            fig.add_hline(y=level,line_dash="dash",line_color=color,line_width=1.3,
                          annotation_text=f"{label} ₹{level:,.2f}",
                          annotation_font_color=color,annotation_position=pos,row=1,col=1)

    # ATR setup levels
    setup = sel.get("atr_setup",{})
    if setup.get("t1"): fig.add_hline(y=setup["t1"],line_dash="dot",line_color="#34d399",line_width=1,annotation_text=f"T1 ₹{setup['t1']:,.2f}",annotation_font_color="#34d399",annotation_position="right",row=1,col=1)
    if setup.get("t2"): fig.add_hline(y=setup["t2"],line_dash="dot",line_color="#059669",line_width=1,annotation_text=f"T2 ₹{setup['t2']:,.2f}",annotation_font_color="#059669",annotation_position="right",row=1,col=1)
    if setup.get("sl"): fig.add_hline(y=setup["sl"],line_dash="dot",line_color="#f87171",line_width=1,annotation_text=f"SL ₹{setup['sl']:,.2f}",annotation_font_color="#f87171",annotation_position="right",row=1,col=1)

    vc = ["#10b981" if float(c)>=float(o) else "#ef4444" for c,o in zip(cdf["Close"],cdf["Open"])]
    fig.add_trace(go.Bar(x=cdf.index,y=cdf["Volume"],name="Volume",marker_color=vc,opacity=.75),row=2,col=1)
    if "Vol_MA20" in cdf.columns: fig.add_trace(go.Scatter(x=cdf.index,y=cdf["Vol_MA20"],name="Vol MA20",line=dict(color="#fb923c",width=1.1)),row=2,col=1)

    if "RSI" in cdf.columns:
        fig.add_trace(go.Scatter(x=cdf.index,y=cdf["RSI"],name="RSI",line=dict(color="#60a5fa",width=1.3)),row=3,col=1)
        fig.add_hrect(y0=70,y1=100,fillcolor="rgba(239,68,68,.06)",line_width=0,row=3,col=1)
        fig.add_hrect(y0=0,y1=30,fillcolor="rgba(16,185,129,.06)",line_width=0,row=3,col=1)
        fig.add_hline(y=70,line_color="#f87171",line_width=.8,line_dash="dot",row=3,col=1)
        fig.add_hline(y=30,line_color="#34d399",line_width=.8,line_dash="dot",row=3,col=1)

    if "MACD" in cdf.columns and "MACD_Signal" in cdf.columns:
        fig.add_trace(go.Scatter(x=cdf.index,y=cdf["MACD"],name="MACD",line=dict(color="#60a5fa",width=1.2)),row=4,col=1)
        fig.add_trace(go.Scatter(x=cdf.index,y=cdf["MACD_Signal"],name="Signal",line=dict(color="#fb923c",width=1.2)),row=4,col=1)
        if "MACD_Hist" in cdf.columns:
            hc=["#10b981" if float(v)>=0 else "#ef4444" for v in cdf["MACD_Hist"].fillna(0)]
            fig.add_trace(go.Bar(x=cdf.index,y=cdf["MACD_Hist"],name="Histogram",marker_color=hc,opacity=.65),row=4,col=1)

    fig.update_layout(template="plotly_dark",height=760,paper_bgcolor="#07080f",
                      plot_bgcolor="#0d1220",showlegend=True,
                      legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10)),
                      xaxis_rangeslider_visible=False,margin=dict(l=8,r=8,t=55,b=8),
                      font=dict(family="Space Grotesk"))
    fig.update_yaxes(gridcolor="#111d2e",zeroline=False)
    fig.update_xaxes(gridcolor="#111d2e",zeroline=False,showspikes=True,spikecolor="#1a2d47")
    return fig


# ══════════════════════════════════════════════════════════════
#  EXPERT ANALYSIS UI
# ══════════════════════════════════════════════════════════════

def show_expert_analysis(ai: dict, symbol: str):
    if "error" in ai:
        st.error(f"Analysis error: {ai['error']}")
        return

    overall = ai.get("overall",{})
    rec   = overall.get("recommendation","WAIT")
    sent  = overall.get("sentiment","NEUTRAL")
    conv  = overall.get("conviction","LOW")
    src   = ai.get("source","rule_based")

    rc  = {"BUY":"#10b981","WAIT":"#f59e0b","AVOID":"#ef4444"}.get(rec,"#94a3b8")
    sc  = {"BULLISH":"#10b981","NEUTRAL":"#94a3b8","BEARISH":"#ef4444"}.get(sent,"#94a3b8")
    src_label = "🤖 Claude AI" if src=="claude_api" else "⚡ Rule-Based Analysis"

    st.markdown(f"""
    <div class="card card-{'green' if rec=='BUY' else 'orange' if rec=='WAIT' else 'red'}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <div style="font-size:.7rem;color:#4b5563;">{src_label}  ·  {symbol}</div>
          <div style="font-size:1.9rem;font-weight:900;color:{rc};">{rec}</div>
          <div style="font-size:.82rem;color:{sc};">{sent} · Conviction: {conv}</div>
        </div>
        <div style="max-width:58%;font-size:.78rem;color:#94a3b8;text-align:right;">
          {overall.get('summary','')}
        </div>
      </div>
      <div style="margin-top:8px;font-size:.76rem;color:#f87171;">⚠️ Key Risk: {overall.get('key_risk','N/A')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ci, cs, cl = st.columns(3)

    def _tf_col(col, key, icon, title, color):
        tf = ai.get(key,{})
        if not tf: return
        view  = tf.get("view","N/A")
        risk  = tf.get("risk_level","MEDIUM")
        rc2   = {"LONG":"#10b981","ACCUMULATE":"#10b981","SHORT":"#ef4444","AVOID":"#6b7280"}.get(view,"#94a3b8")
        risc  = {"LOW":"#10b981","MEDIUM":"#f59e0b","HIGH":"#ef4444"}.get(risk,"#94a3b8")
        t3    = tf.get("target3","")

        with col:
            st.markdown(f"""
            <div style="background:#0a0f1e;border:1px solid #1a2d47;border-radius:14px;padding:15px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <div style="font-size:1rem;font-weight:800;color:{color};">{icon} {title}</div>
                <div><span class="badge {'b-green' if view in ('LONG','ACCUMULATE') else 'b-red' if view=='SHORT' else 'b-gray'}">{view}</span>
                     <span class="badge {'b-green' if risk=='LOW' else 'b-yellow' if risk=='MEDIUM' else 'b-red'}">{risk}</span></div>
              </div>
              <div class="setup-row"><span class="setup-label">📍 Entry Zone</span><span class="setup-val" style="color:#fbbf24;">{tf.get('entry_zone','N/A')}</span></div>
              <div class="setup-row"><span class="setup-label">🎯 Target 1</span><span class="setup-val" style="color:#34d399;">₹{tf.get('target1',0):,.2f} <span style="font-size:.7rem;color:#4b5563;">(+{tf.get('t1_pct',0):.1f}%)</span></span></div>
              <div class="setup-row"><span class="setup-label">🎯 Target 2</span><span class="setup-val" style="color:#10b981;">₹{tf.get('target2',0):,.2f} <span style="font-size:.7rem;color:#4b5563;">(+{tf.get('t2_pct',0):.1f}%)</span></span></div>
              {'<div class="setup-row"><span class="setup-label">🎯 Target 3</span><span class="setup-val" style="color:#059669;">₹' + f"{t3:,.2f}" + '</span></div>' if t3 and isinstance(t3,(int,float)) and t3>0 else ''}
              <div class="setup-row"><span class="setup-label">🛑 Stop Loss</span><span class="setup-val" style="color:#f87171;">₹{tf.get('sl',0):,.2f} <span style="font-size:.7rem;color:#4b5563;">({tf.get('sl_pct',0):.1f}%)</span></span></div>
              <div class="setup-row" style="border:none;"><span class="setup-label">⚖️ Risk:Reward</span><span class="setup-val" style="color:#a78bfa;">{tf.get('risk_reward','N/A')}</span></div>
              <div style="background:#07080f;border-radius:8px;padding:7px;margin-top:8px;">
                <div style="font-size:.68rem;color:#4b5563;">📅 Hold</div>
                <div style="font-size:.82rem;font-weight:700;">{tf.get('holding_period',tf.get('time_in_trade','N/A'))}</div>
              </div>
              {f'<div style="font-size:.7rem;color:#4b5563;margin-top:6px;border-top:1px solid #1a2d47;padding-top:5px;">{tf.get("fundamental_view","")}</div>' if tf.get("fundamental_view") else ''}
              <div style="font-size:.7rem;color:#6b7280;margin-top:6px;border-top:1px solid #0f1a2e;padding-top:5px;">{tf.get('note','')}</div>
            </div>""", unsafe_allow_html=True)

    _tf_col(ci, "intraday", "⚡", "Intraday",  "#60a5fa")
    _tf_col(cs, "swing",    "📊", "Swing",     "#fb923c")
    _tf_col(cl, "longterm", "📈", "Long Term", "#10b981")

    rr = ai.get("risk_ranking",[])
    if rr:
        st.markdown("<br>#### 📊 Risk Ranking (Lowest → Highest Risk)", unsafe_allow_html=True)
        rcols = st.columns(len(rr))
        for i, item in enumerate(rr):
            risk  = item.get("risk","MEDIUM")
            rc3   = {"LOW":"#10b981","MEDIUM":"#f59e0b","HIGH":"#ef4444"}.get(risk,"#94a3b8")
            rcols[i].markdown(f'<div class="card" style="text-align:center;border-left:4px solid {rc3};padding:12px;"><div style="font-size:1.3rem;font-weight:800;color:{rc3};">#{i+1}</div><div style="font-size:.88rem;font-weight:700;">{item.get("timeframe","")}</div><div style="font-size:.76rem;color:{rc3};font-weight:700;">{risk} RISK</div><div style="font-size:.68rem;color:#4b5563;margin-top:3px;">{item.get("reason","")}</div></div>', unsafe_allow_html=True)

    ni = ai.get("news_impact","")
    if ni:
        st.markdown(f'<div class="card card-blue" style="margin-top:8px;"><div style="font-size:.7rem;color:#60a5fa;font-weight:700;">📰 NEWS IMPACT</div><div style="font-size:.83rem;margin-top:5px;">{ni}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer">⚠️ {ai.get("disclaimer","AI analysis for educational purposes. Not SEBI registered advice.")}</div>', unsafe_allow_html=True)


def show_news(news):
    if not news:
        st.info("No news found."); return
    for n in news:
        fresh = n.get("fresh","")
        st.markdown(f'<div class="news-card"><a href="{n["link"]}" target="_blank" style="text-decoration:none;"><div class="news-title">{n["title"]}</div></a><div class="news-meta">📰 {n["source"]}  ·  🗓️ {n["date"]}  {fresh}</div></div>', unsafe_allow_html=True)


def show_financials(fin):
    if not fin: st.info("Financial data not available."); return
    def v(k,sfx="",pfx=""):
        val=fin.get(k); return f"{pfx}{val:,.2f}{sfx}" if val is not None else "N/A"
    st.markdown(f'<div class="card card-blue"><div style="font-size:1rem;font-weight:800;">{fin.get("company_name","")}</div><div style="font-size:.76rem;color:#4b5563;">{fin.get("sector","")} · {fin.get("industry","")}</div></div>', unsafe_allow_html=True)
    r1=st.columns(4); r1[0].metric("Market Cap",v("market_cap_cr"," Cr","₹")); r1[1].metric("P/E TTM",v("pe_trailing")); r1[2].metric("P/E Fwd",v("pe_forward")); r1[3].metric("P/B",v("pb_ratio"))
    r2=st.columns(4); r2[0].metric("ROE",v("roe_pct","%")); r2[1].metric("Debt/Equity",v("debt_equity")); r2[2].metric("Rev Growth",v("revenue_growth","%")); r2[3].metric("EPS TTM",v("eps_ttm","","₹"))
    r3=st.columns(4); r3[0].metric("52W High",v("week52_high","","₹")); r3[1].metric("52W Low",v("week52_low","","₹")); r3[2].metric("Beta",v("beta")); r3[3].metric("Div Yield",v("dividend_yield","%"))


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    universe_keys = [k for k,v in UNIVERSES.items() if v]
    universe_choice = st.selectbox("Stock Universe", universe_keys, index=0)
    symbols = UNIVERSES[universe_choice]

    tf_choice = st.selectbox("Timeframe", ["Daily (6mo)","Weekly (2y)","1-Hour (60d) ⚡"], index=0)
    if "Daily"   in tf_choice: interval,period,tf_label = "1d","6mo","Daily"
    elif "Weekly" in tf_choice: interval,period,tf_label = "1wk","2y","Weekly"
    else:                        interval,period,tf_label = "1h","60d","1-Hour"

    st.markdown("---")
    st.markdown("### 🎯 Grade Filter")
    min_grade = st.selectbox("Minimum Grade", ["All (A+/A/B/C)","A+ only","A+ and A","B and above"], index=0)

    st.markdown("---")
    st.markdown("### 🔎 Filters")

    filter_mode = st.radio("Filter Logic", ["OR (any match)", "AND (all must match)"], index=0, horizontal=True)
    use_and = "AND" in filter_mode

    min_conf  = st.slider("Min Confidence %", 40, 85, 50, 5)

    signal_filter = st.multiselect("Signals", [
        "BREAKOUT","CONSOLIDATION_BREAKOUT","NEAR_BREAKOUT",
        "RETEST_BREAKOUT","BREAKDOWN","REVERSAL_SIGNAL","FALSE_BREAKOUT",
    ], default=["BREAKOUT","CONSOLIDATION_BREAKOUT","NEAR_BREAKOUT"])

    trend_filter = st.multiselect("Trend", ["UPTREND","SIDEWAYS","DOWNTREND"],
                                  default=["UPTREND","SIDEWAYS"])

    st.markdown("#### ✅ Must-Have Conditions (AND mode)")
    must_vol   = st.checkbox("Volume Spike (≥1.5x)",    value=False)
    must_up    = st.checkbox("Uptrend only",             value=False)
    must_rsi   = st.checkbox("RSI < 70",                 value=False)
    must_bo    = st.checkbox("Confirmed Breakout",       value=False)
    must_strong= st.checkbox("Strong Breakout (v6)",     value=False)

    st.markdown("---")
    st.markdown("### 📲 Telegram Alerts")
    if TG_BOT_TOKEN and TG_CHAT_ID:
        tg_enabled = st.toggle("Send A+ alerts to Telegram", value=False)
        st.caption("Alerts sent for A+ breakouts automatically")
    else:
        tg_enabled = False
        st.caption("Add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in secrets to enable")

    st.markdown("---")
    if API_KEY:
        st.markdown('<span class="badge b-green">🤖 Claude AI: ON</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge b-orange">⚡ Rule-Based Analysis: ON</span>', unsafe_allow_html=True)
        st.caption("Add ANTHROPIC_API_KEY for Claude AI analysis")

    auto_refresh  = st.toggle("⏰ Auto Refresh", value=False)
    refresh_mins  = st.select_slider("Every", [5,10,15,30], value=10) if auto_refresh else None

    st.markdown("---")
    st.markdown("Built by [Purva Doshi](https://linkedin.com/in/purvadoshi26)  \nFree · NSE Equity Only")


# ══════════════════════════════════════════════════════════════
#  HEADER + MARKET CONTEXT
# ══════════════════════════════════════════════════════════════

h1, h2 = st.columns([3,1])
with h1:
    st.markdown('<div class="main-title">📈 NSE Breakout Scanner v6</div>', unsafe_allow_html=True)
    st.caption("Trade Grades · Strong Breakout · ATR Setups · Swing Detection · Backtest · Rule-Based AI")
with h2:
    ist_now = now_ist()
    st.markdown(f'<div style="text-align:right;margin-top:10px;"><div style="font-size:1.3rem;font-weight:800;color:#34d399;">{ist_now.strftime("%H:%M:%S")}</div><div style="font-size:.7rem;color:#4b5563;">{ist_now.strftime("%d %b %Y")} IST</div></div>', unsafe_allow_html=True)

# Nifty 50 Market Context Bar
with st.spinner("Checking market…"):
    mkt = cached_market_context()

mkt_color = {"UPTREND":"#10b981","DOWNTREND":"#ef4444","SIDEWAYS":"#f59e0b","UNKNOWN":"#6b7280"}.get(mkt.get("trend","UNKNOWN"),"#6b7280")
mkt_emoji = {"UPTREND":"📈","DOWNTREND":"📉","SIDEWAYS":"➡️","UNKNOWN":"❓"}.get(mkt.get("trend","UNKNOWN"),"❓")
mkt_adj   = {"UPTREND":"Confidence +3% for breakouts","DOWNTREND":"⚠️ Confidence -10% for breakouts","SIDEWAYS":"Neutral adjustment","UNKNOWN":""}. get(mkt.get("trend","UNKNOWN"),"")
nifty_cmp = mkt.get("nifty_cmp")
nifty_str = f"₹{nifty_cmp:,.2f}" if nifty_cmp else "N/A"

st.markdown(f"""
<div style="background:#0a0f1e;border:1px solid #1a2d47;border-radius:12px;padding:10px 16px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;">
  <div>
    <span style="font-size:.7rem;color:#4b5563;">NIFTY 50 MARKET CONTEXT</span>&nbsp;
    <span style="font-weight:800;color:{mkt_color};">{mkt_emoji} {mkt.get('trend','UNKNOWN')}</span>&nbsp;
    <span style="font-size:.8rem;color:{mkt_color};">{nifty_str}</span>&nbsp;
    <span style="font-size:.72rem;color:#4b5563;">5D: {mkt.get('change_5d',0):+.2f}%  1D: {mkt.get('change_1d',0):+.2f}%</span>
  </div>
  <div style="font-size:.68rem;color:#4b5563;">{mkt_adj}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #111d2e;margin:8px 0;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════

main_tab1, main_tab2 = st.tabs(["🔍 Sector Scanner", "🔎 Stock Lookup"])


# ──────────────────────────────────────────────────────────────
#  TAB 1: SECTOR SCANNER
# ──────────────────────────────────────────────────────────────

with main_tab1:
    sc1, si1, si2, si3 = st.columns([2.5,1,1,1])
    with sc1: scan_btn = st.button("🔍  RUN SCANNER", type="primary", use_container_width=True, key="scan_btn_main")
    with si1: st.metric("Universe", f"{len(symbols)} stocks")
    with si2: st.metric("Timeframe", tf_label)
    with si3: st.metric("Min Conf",  f"{min_conf}%")

    BATCH_SIZE = 20

    def run_full_scan(syms, period, interval, mkt_trend):
        results, failed = [], []
        n       = len(syms)
        prog    = st.progress(0, text="Starting scan…")
        status  = st.empty()
        t0      = time.time()
        batches = [syms[i:i+BATCH_SIZE] for i in range(0,n,BATCH_SIZE)]

        for b_idx, batch in enumerate(batches):
            status.markdown(f"**Batch {b_idx+1}/{len(batches)}** — {', '.join(batch[:4])}…")
            raw = cached_fetch_batch(tuple(sorted(batch)), period, interval)
            for sym in batch:
                df = raw.get(sym)
                r  = scan_stock(sym, df, market_trend=mkt_trend) if df is not None else None
                if r: results.append(r)
                else:  failed.append(sym)
            done = min((b_idx+1)*BATCH_SIZE, n)
            elapsed = time.time()-t0
            spd = done/elapsed if elapsed>0 else 1
            rem = int((n-done)/spd) if (n-done)>0 else 0
            prog.progress(done/n, text=f"{done}/{n} · ~{rem}s left")
            time.sleep(0.05)

        prog.empty(); status.empty()
        return results, failed

    if scan_btn:
        results, failed = run_full_scan(symbols, period, interval, mkt.get("trend","SIDEWAYS"))
        st.session_state["results"] = results
        st.session_state["failed"]  = failed
        st.session_state["scan_ts"] = now_ist().strftime("%H:%M:%S IST")

        # Telegram — send A+ alerts
        if tg_enabled and TG_BOT_TOKEN and TG_CHAT_ID:
            aplus = [r for r in results if r.get("trade_grade")=="A+"]
            for r in aplus[:5]:
                send_telegram_alert(TG_BOT_TOKEN, TG_CHAT_ID, r)
            if aplus:
                st.success(f"📲 Sent {len(aplus)} A+ Telegram alert(s)")

    if "results" not in st.session_state:
        st.markdown('<div style="text-align:center;padding:50px;color:#1e2d4a;"><div style="font-size:3.5rem;">📊</div><div style="font-size:1.1rem;color:#374151;margin-top:10px;">Select universe and press Run Scanner</div></div>', unsafe_allow_html=True)
        st.stop()

    # ── Apply filters ─────────────────────────────────────────
    all_res = st.session_state["results"]

    def apply_filters(rows):
        out = []
        for r in rows:
            if r["confidence"] < min_conf: continue

            # Grade filter
            g = r.get("trade_grade","C")
            if min_grade == "A+ only"    and g != "A+":            continue
            if min_grade == "A+ and A"   and g not in ("A+","A"): continue
            if min_grade == "B and above" and g == "C":             continue

            # Must-have conditions
            if must_vol    and r.get("vol_ratio",0)   < 1.5:  continue
            if must_up     and r.get("trend","") != "UPTREND": continue
            if must_rsi    and r.get("rsi",100)       >= 70:   continue
            if must_bo     and "BREAKOUT" not in r.get("signal",""): continue
            if must_strong and not r.get("strong_breakout",False):   continue

            # Signal / Trend (OR or AND)
            if use_and:
                if signal_filter and r["signal"] not in signal_filter: continue
                if trend_filter  and r["trend"]  not in trend_filter:  continue
            else:
                if signal_filter and r["signal"] not in signal_filter: continue
                if trend_filter  and r["trend"]  not in trend_filter:  continue

            out.append(r)
        return out

    filtered = apply_filters(all_res)
    filtered.sort(key=lambda x: x["rank_score"], reverse=True)

    # ── KPIs ──────────────────────────────────────────────────
    n_aplus  = sum(1 for r in filtered if r.get("trade_grade")=="A+")
    n_a      = sum(1 for r in filtered if r.get("trade_grade")=="A")
    n_b      = sum(1 for r in filtered if r.get("trade_grade")=="B")
    n_c      = sum(1 for r in filtered if r.get("trade_grade")=="C")
    n_strong = sum(1 for r in filtered if r.get("strong_breakout"))
    n_bo     = sum(1 for r in filtered if "BREAKOUT" in r.get("signal","") and "NEAR" not in r.get("signal","") and "FALSE" not in r.get("signal",""))

    st.markdown("<br>", unsafe_allow_html=True)
    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    kpi(k1, len(all_res),   "Scanned",      "#60a5fa")
    kpi(k2, len(filtered),  "Signals",      "#34d399")
    kpi(k3, f"🌟 {n_aplus}","Grade A+",     "#f59e0b")
    kpi(k4, f"🔥 {n_a}",    "Grade A",      "#10b981")
    kpi(k5, f"⚡ {n_b}",    "Grade B",      "#f59e0b")
    kpi(k6, f"⚠️ {n_c}",    "Grade C",      "#6b7280")
    kpi(k7, f"💪 {n_strong}","Strong B/O",  "#a78bfa")

    st.markdown(f"<div style='text-align:right;font-size:.7rem;color:#374151;'>Last scan: {st.session_state.get('scan_ts','')}</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #111d2e;'>", unsafe_allow_html=True)

    if not filtered:
        st.warning("No stocks matched. Lower confidence or loosen filters.")
        st.stop()

    # ── TOP PICKS (A+ only, top 5) ────────────────────────────
    top_picks = [r for r in filtered if r.get("trade_grade")=="A+"][:5]
    if top_picks:
        st.markdown("### 🌟 Top Picks — A+ Grade Only")
        tp_cols = st.columns(min(len(top_picks), 5))
        for i, r in enumerate(top_picks):
            setup = r.get("atr_setup",{})
            with tp_cols[i]:
                pct_c = "#34d399" if r["pct_from_level"]<=2 else "#f59e0b"
                st.markdown(f"""
                <div class="top-pick-card">
                  <div style="font-size:.65rem;color:#f59e0b;font-weight:700;">🌟 A+ PICK #{i+1}</div>
                  <div style="font-size:1.2rem;font-weight:800;color:#fff;margin:4px 0;">{r['symbol']}</div>
                  <div style="font-size:1.5rem;font-weight:900;color:#34d399;">₹{r['cmp']:,.2f}</div>
                  <div style="margin:6px 0;">{signal_badge(r['signal'])}</div>
                  <div style="font-size:.72rem;color:#4b5563;">Pattern: {r['pattern'].replace('_',' ')}</div>
                  <div style="font-size:.72rem;color:#4b5563;">Vol: <span style="color:#60a5fa;font-weight:700;">{r['vol_ratio']}x</span>  RSI: <span style="color:{rsi_color(r['rsi'])};font-weight:700;">{r['rsi']}</span></div>
                  <div style="margin-top:6px;font-size:.8rem;">
                    {'<span style="color:#34d399;font-size:.7rem;">💪 STRONG BREAKOUT</span>' if r.get('strong_breakout') else ''}
                  </div>
                  {f'<div style="margin-top:8px;background:#07080f;border-radius:8px;padding:7px;font-size:.72rem;"><div style="color:#4b5563;">ATR Setup</div><div style="color:#fbbf24;">Entry ₹{setup.get("entry",r["cmp"]):,.2f}</div><div style="color:#f87171;">SL ₹{setup.get("sl",0):,.2f}</div><div style="color:#34d399;">T2 ₹{setup.get("t2",0):,.2f}</div><div style="color:#a78bfa;">R:R {setup.get("rr","N/A")}</div></div>' if setup else ''}
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Avoid List (C grade) ──────────────────────────────────
    avoid_list = [r for r in filtered if r.get("trade_grade")=="C"]
    if avoid_list:
        with st.expander(f"🚫 Avoid List — {len(avoid_list)} C-Grade stocks (high risk / false signals)"):
            avoid_df = pd.DataFrame([{
                "Stock": r["symbol"], "CMP": f"₹{r['cmp']:,.2f}",
                "Signal": r["signal"], "Trend": r["trend"],
                "RSI": r["rsi"], "Vol": f"{r['vol_ratio']}x", "Reason": r["pattern"],
            } for r in avoid_list])
            st.dataframe(avoid_df, use_container_width=True, hide_index=True)

    # ── Signal Tabs ───────────────────────────────────────────
    bo_rows   = [r for r in filtered if "BREAKOUT" in r.get("signal","") and "NEAR" not in r.get("signal","") and "FALSE" not in r.get("signal","")]
    near_rows = [r for r in filtered if "NEAR" in r.get("signal","")]
    all_rows  = filtered

    t1, t2, t3 = st.tabs([f"🔥 Breakouts ({len(bo_rows)})", f"⚡ Near Breakout ({len(near_rows)})", f"📋 All ({len(all_rows)})"])

    def show_rows(rows, tab_key):
        if not rows: st.info("No signals in this category."); return

        df_exp = pd.DataFrame([{
            "Rank":i+1,"Stock":r["symbol"],"CMP ₹":r["cmp"],
            "Grade":r.get("trade_grade","B"),"Pattern":r["pattern"],
            "Signal":r["signal"],"Confidence %":r["confidence"],
            "Vol Ratio":r["vol_ratio"],"Breakout ₹":r["breakout_level"],
            "% from Level":r["pct_from_level"],"Trend":r["trend"],
            "RSI":r["rsi"],"Strong B/O":r.get("strong_breakout",False),
            "Strength Score":r.get("strength_score",0),
            "ATR Entry":r.get("atr_setup",{}).get("entry",""),
            "ATR SL":r.get("atr_setup",{}).get("sl",""),
            "ATR T1":r.get("atr_setup",{}).get("t1",""),
            "ATR T2":r.get("atr_setup",{}).get("t2",""),
            "R:R":r.get("atr_setup",{}).get("rr",""),
        } for i,r in enumerate(rows)])

        csv_data = df_exp.to_csv(index=False)
        st.download_button("📥 Export CSV", csv_data,
                           f"nse_{tab_key}_{now_ist().strftime('%Y%m%d_%H%M')}.csv",
                           "text/csv", key=f"dl_{tab_key}_{len(rows)}")

        for i, r in enumerate(rows):
            setup = r.get("atr_setup",{})
            c1,c2,c3,c4,c5 = st.columns([1.1,1.8,1.5,1.6,1.2])

            with c1:
                st.markdown(f"""
                <div style="font-size:.65rem;color:#4b5563;">#{i+1} · Score {r['rank_score']}</div>
                {grade_badge(r.get('trade_grade','B'))}
                <div style="font-size:1.1rem;font-weight:800;margin-top:4px;">{r['symbol']}</div>
                <div style="font-size:1.25rem;font-weight:800;color:#34d399;">₹{r['cmp']:,.2f}</div>
                {'<div style="font-size:.65rem;color:#a78bfa;margin-top:2px;">💪 STRONG BREAKOUT</div>' if r.get('strong_breakout') else ''}
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(signal_badge(r["signal"]) + "<br>" + ptag(r["pattern"]), unsafe_allow_html=True)
                extras = [p for p in r.get("all_patterns",[]) if p!=r["pattern"]][:2]
                if extras: st.markdown(" ".join(ptag(p) for p in extras), unsafe_allow_html=True)
                st.markdown(strength_bar(r.get("strength_score",0)), unsafe_allow_html=True)

            with c3:
                st.markdown(conf_bar(r["confidence"]) + "<br>" + trend_badge(r["trend"]), unsafe_allow_html=True)
                rc_i = rsi_color(r["rsi"])
                st.markdown(f'<div style="font-size:.72rem;margin-top:3px;"><span style="color:{rc_i};">RSI {r["rsi"]}</span>  ·  <span style="color:#60a5fa;">{r["vol_ratio"]}x vol</span>  ·  <span style="color:#fbbf24;">{r["signal_count"]} sig</span></div>', unsafe_allow_html=True)

            with c4:
                pc = "#34d399" if r["pct_from_level"]<=2 else ("#f59e0b" if r["pct_from_level"]<=5 else "#f87171")
                st.markdown(f"""
                <div style="font-size:.7rem;color:#4b5563;">Breakout Zone</div>
                <div style="font-size:.95rem;font-weight:700;">₹{r['breakout_level']:,.2f}</div>
                <div style="font-size:.8rem;font-weight:700;color:{pc};">{r['pct_from_level']:+.2f}%</div>
                """, unsafe_allow_html=True)

                if setup:
                    age_txt = f"Age: {r.get('breakout_age',0)} bars" if r.get("breakout_age",0) > 0 else "🆕 Fresh breakout"
                    st.markdown(f"""
                    <div style="background:#07080f;border-radius:8px;padding:7px;margin-top:5px;font-size:.7rem;">
                      <div style="color:#4b5563;">ATR Setup · {age_txt}</div>
                      <div style="display:grid;grid-template-columns:1fr 1fr;gap:2px;margin-top:3px;">
                        <div><span style="color:#4b5563;">SL</span> <span style="color:#f87171;font-weight:700;">₹{setup.get('sl',0):,.2f}</span></div>
                        <div><span style="color:#4b5563;">T1</span> <span style="color:#34d399;font-weight:700;">₹{setup.get('t1',0):,.2f}</span></div>
                        <div><span style="color:#4b5563;">T2</span> <span style="color:#10b981;font-weight:700;">₹{setup.get('t2',0):,.2f}</span></div>
                        <div><span style="color:#4b5563;">R:R</span> <span style="color:#a78bfa;font-weight:700;">{setup.get('rr','N/A')}</span></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            with c5:
                s200_line = f'<div style="font-size:.67rem;color:#4b5563;">SMA200: ₹{r["sma200"]:,.0f}</div>' if r.get("sma200") else ""
                e20_line  = f'<div style="font-size:.67rem;color:#4b5563;">EMA20: ₹{r["ema20"]:,.0f}</div>'  if r.get("ema20")  else ""
                st.markdown(f"""
                <div style="font-size:.7rem;color:#4b5563;">Support / Resistance</div>
                <div style="font-size:.72rem;">Sup ₹{r['support']:,.0f} · Res ₹{r['resistance']:,.0f}</div>
                {e20_line}{s200_line}
                """, unsafe_allow_html=True)

            st.markdown("<hr style='border:1px solid #0f1a2e;margin:5px 0;'>", unsafe_allow_html=True)

    with t1: show_rows(bo_rows,   "breakouts")
    with t2: show_rows(near_rows, "near_breakout")
    with t3: show_rows(all_rows,  "all_signals")

    # ── Chart + Analysis ──────────────────────────────────────
    st.markdown("<hr style='border:1px solid #111d2e;'>", unsafe_allow_html=True)
    st.markdown("### 📊 Chart & Expert Analysis")

    chart_opts = [r["symbol"] for r in filtered]
    sel_sym    = st.selectbox("Select stock:", chart_opts, key="sector_chart_sel")
    sel        = next((r for r in filtered if r["symbol"]==sel_sym), None)

    if sel and "_df" in sel:
        cdf = sel["_df"]

        # Quick metrics row
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Grade",     sel.get("trade_grade","B"))
        m2.metric("Strength",  f"{sel.get('strength_score',0)}/100")
        m3.metric("CMP",       f"₹{sel['cmp']:,.2f}")
        m4.metric("Signal",    sel["signal"].replace("_"," "))
        m5.metric("Conf",      f"{sel['confidence']}%")
        m6.metric("Breakout Age", f"{sel.get('breakout_age',0)} bars")

        fig = build_chart(sel, cdf)
        st.plotly_chart(fig, use_container_width=True)

        # Detail tabs
        ea1,ea2,ea3,ea4 = st.tabs(["🤖 Expert View","📰 News","💹 Financials","📈 Backtest"])

        with ea1:
            ai_key = f"ai_{sel_sym}"
            btn_label = "🤖 Get Claude AI Analysis" if API_KEY else "⚡ Get Rule-Based Analysis"
            if st.button(btn_label, key=f"ai_btn_{sel_sym}"):
                with st.spinner("Fetching data & running analysis…"):
                    news_d = cached_news(sel_sym)
                    fin_d  = cached_financials(sel_sym)
                    ai_r   = get_ai_analysis(sel_sym, sel, fin_d, news_d, API_KEY)
                st.session_state[ai_key] = ai_r
                st.session_state[ai_key+"_news"] = news_d
                st.session_state[ai_key+"_fin"]  = fin_d
            if ai_key in st.session_state:
                show_expert_analysis(st.session_state[ai_key], sel_sym)

        with ea2:
            nk = f"ai_{sel_sym}_news"
            if nk in st.session_state: show_news(st.session_state[nk])
            else:
                if st.button(f"📰 Load News", key=f"news_{sel_sym}"):
                    nd = cached_news(sel_sym)
                    st.session_state[nk] = nd
                    show_news(nd)

        with ea3:
            fk = f"ai_{sel_sym}_fin"
            if fk in st.session_state: show_financials(st.session_state[fk])
            else:
                if st.button(f"💹 Load Financials", key=f"fin_{sel_sym}"):
                    fd = cached_financials(sel_sym)
                    st.session_state[fk] = fd
                    show_financials(fd)

        with ea4:
            st.markdown("#### 📈 Historical Backtest")
            st.caption("Tests ATR-based breakout strategy on this stock's own historical data (in-sample). Entry = breakout bar. SL = 1.5x ATR. Target = 2x ATR. Lookahead = 10 candles.")
            if st.button(f"▶️ Run Backtest on {sel_sym}", key=f"bt_{sel_sym}"):
                with st.spinner("Running backtest on historical data…"):
                    bt = run_backtest(cdf.copy())
                st.session_state[f"bt_{sel_sym}"] = bt

            if f"bt_{sel_sym}" in st.session_state:
                bt = st.session_state[f"bt_{sel_sym}"]
                if bt.get("win_rate") is not None:
                    wr = bt["win_rate"]
                    wr_color = "#10b981" if wr>=55 else ("#f59e0b" if wr>=45 else "#ef4444")
                    b1,b2,b3,b4 = st.columns(4)
                    b1.metric("Win Rate",    f"{wr}%")
                    b2.metric("Total Trades",bt["total"])
                    b3.metric("Avg Return",  f"{bt.get('avg_return',0):+.2f}%")
                    b4.metric("Max Drawdown",f"{bt.get('max_dd',0):.2f}%")
                    st.markdown(f'<div class="card" style="border-left:4px solid {wr_color};"><div style="font-size:.72rem;color:#4b5563;">📊 Backtest Result</div><div style="font-size:.88rem;margin-top:5px;">{bt.get("message","")}</div><div style="font-size:.72rem;color:#f59e0b;margin-top:6px;">⚠️ In-sample test only. Past performance ≠ future results.</div></div>', unsafe_allow_html=True)
                else:
                    st.warning(bt.get("message","Not enough data for backtest."))


# ──────────────────────────────────────────────────────────────
#  TAB 2: STOCK LOOKUP
# ──────────────────────────────────────────────────────────────

with main_tab2:
    st.markdown("### 🔎 Analyse Any NSE Stock")
    st.caption("Enter any NSE symbol → Full chart, Trade Grade, ATR setup, Expert analysis, News, Financials")

    lc1,lc2,lc3 = st.columns([2,1,1])
    with lc1:
        lookup_sym = st.text_input("NSE Symbol", placeholder="e.g. TATAMOTORS",
                                   key="lookup_inp").strip().upper().replace(".NS","")
    with lc2:
        ltf = st.selectbox("Timeframe",["Daily (6mo)","Weekly (2y)","1-Hour (60d)"],key="ltf")
    with lc3:
        st.markdown("<br>",unsafe_allow_html=True)
        lookup_btn = st.button("🔍 Analyse",type="primary",key="lookup_btn",use_container_width=True)

    li,lp = ("1d","6mo") if "Daily" in ltf else (("1wk","2y") if "Weekly" in ltf else ("1h","60d"))

    if lookup_btn and lookup_sym:
        with st.spinner(f"Fetching {lookup_sym}…"):
            l_df = fetch_one(lookup_sym, lp, li)

        if l_df is None or len(l_df)<10:
            st.error(f"Could not fetch {lookup_sym}. Verify the NSE symbol and try again.")
        else:
            l_df  = add_indicators(l_df)
            l_scan= scan_stock(lookup_sym, l_df, market_trend=mkt.get("trend","SIDEWAYS"))
            if l_scan is None:
                cond = check_conditions(l_df); trend_l,_=detect_trend(l_df)
                l_scan = {"symbol":lookup_sym,"cmp":round(cond["_close"],2),"pattern":"MANUAL_LOOKUP","signal":"ANALYSING","confidence":50,"conf_label":"MEDIUM","trade_grade":"B","strength_score":50,"atr_setup":{},"breakout_age":0,"vol_ratio":round(cond["_vol_ratio"],2),"breakout_level":cond["_resistance"],"pct_from_level":0.0,"trend":trend_l,"rsi":round(cond["_rsi"],1),"ema20":cond["_ema20"],"ema50":cond["_ema50"],"sma200":cond["_sma200"],"support":round(cond["_support"],2),"resistance":round(cond["_resistance"],2),"strong_breakout":False,"signal_count":0,"all_patterns":[],"rank_score":50,"_df":l_df,"_atr":round(cond["_atr"],2)}
            st.session_state["lookup_result"] = l_scan
            st.session_state["lookup_sym"]    = lookup_sym

    if "lookup_result" in st.session_state:
        lsel = st.session_state["lookup_result"]
        lsym = st.session_state["lookup_sym"]
        lcdf = lsel["_df"]

        lm1,lm2,lm3,lm4,lm5,lm6 = st.columns(6)
        lm1.metric("Symbol",   lsym)
        lm2.metric("Grade",    lsel.get("trade_grade","B"))
        lm3.metric("CMP",      f"₹{lsel['cmp']:,.2f}")
        lm4.metric("Signal",   lsel["signal"].replace("_"," "))
        lm5.metric("Strength", f"{lsel.get('strength_score',0)}/100")
        lm6.metric("RSI",      lsel["rsi"])

        fig_l = build_chart(lsel, lcdf, f"{lsym} — Grade {lsel.get('trade_grade','B')} · {lsel['pattern']} · {lsel['trend']}")
        st.plotly_chart(fig_l, use_container_width=True)

        la1,la2,la3,la4 = st.tabs(["🤖 Expert View","📰 News","💹 Financials","📈 Backtest"])

        with la1:
            l_ai_key = f"lookup_ai_{lsym}"
            btn_lbl  = "🤖 Get Claude AI Analysis" if API_KEY else "⚡ Get Rule-Based Analysis"
            if st.button(btn_lbl, key=f"l_ai_{lsym}"):
                with st.spinner("Running analysis…"):
                    l_news = cached_news(lsym)
                    l_fin  = cached_financials(lsym)
                    l_ai   = get_ai_analysis(lsym, lsel, l_fin, l_news, API_KEY)
                st.session_state[l_ai_key]         = l_ai
                st.session_state[l_ai_key+"_news"] = l_news
                st.session_state[l_ai_key+"_fin"]  = l_fin
            if l_ai_key in st.session_state:
                show_expert_analysis(st.session_state[l_ai_key], lsym)

        with la2:
            lnk = f"lookup_ai_{lsym}_news"
            if lnk in st.session_state: show_news(st.session_state[lnk])
            else:
                if st.button("📰 Load News", key=f"l_news_{lsym}"):
                    nd = cached_news(lsym); st.session_state[lnk]=nd; show_news(nd)

        with la3:
            lfk = f"lookup_ai_{lsym}_fin"
            if lfk in st.session_state: show_financials(st.session_state[lfk])
            else:
                if st.button("💹 Load Financials", key=f"l_fin_{lsym}"):
                    fd=cached_financials(lsym); st.session_state[lfk]=fd; show_financials(fd)

        with la4:
            st.markdown("#### 📈 Historical Backtest")
            if st.button(f"▶️ Run Backtest on {lsym}", key=f"l_bt_{lsym}"):
                with st.spinner("Running backtest…"):
                    lbt = run_backtest(lcdf.copy())
                st.session_state[f"l_bt_{lsym}"] = lbt
            if f"l_bt_{lsym}" in st.session_state:
                lbt = st.session_state[f"l_bt_{lsym}"]
                if lbt.get("win_rate") is not None:
                    lb1,lb2,lb3,lb4 = st.columns(4)
                    lb1.metric("Win Rate",f"{lbt['win_rate']}%"); lb2.metric("Total",lbt["total"])
                    lb3.metric("Avg Return",f"{lbt.get('avg_return',0):+.2f}%"); lb4.metric("Max DD",f"{lbt.get('max_dd',0):.2f}%")
                    st.caption(lbt.get("message",""))
                else:
                    st.warning(lbt.get("message","Not enough data."))

    else:
        st.markdown('<div style="text-align:center;padding:40px;color:#1e2d4a;"><div style="font-size:3rem;">🔍</div><div style="color:#374151;margin-top:8px;">Type any NSE symbol and click Analyse</div></div>', unsafe_allow_html=True)


# ── Auto Refresh ──────────────────────────────────────────────
if auto_refresh and refresh_mins:
    ph = st.empty()
    for rem in range(refresh_mins*60, 0, -1):
        ph.caption(f"🔄 Auto-refresh in {rem//60}m {rem%60:02d}s")
        time.sleep(1)
    ph.empty()
    st.rerun()
