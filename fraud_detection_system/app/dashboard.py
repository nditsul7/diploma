"""Дашборд аналітика (Streamlit): перевірка транзакцій, моделі, аналітика масивів даних."""
from __future__ import annotations

import hashlib
import io
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from app import auth, datasets, evalutils, history, notify, userprefs
except ImportError:  # streamlit run app/dashboard.py додає app/ у sys.path
    import auth, datasets, evalutils, history, notify, userprefs  # type: ignore[no-redef]

roc_pr_curves, confusion, f1_sweep, thin_pair = (
    evalutils.roc_pr_curves, evalutils.confusion, evalutils.f1_sweep, evalutils.thin_pair)

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = Path(os.getenv("DATA_PATH", "data/raw_transactions.csv"))
MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "50000"))
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
SESSION_HOURS = float(os.getenv("SESSION_HOURS", "8"))
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 30
ALLOW_SIGNUP = os.getenv("ALLOW_SIGNUP", "1") != "0"
ROLE_UA = {"admin": "Адміністратор", "analyst": "Аналітик"}
V_COLS = [f"V{i}" for i in range(1, 29)]
INPUT_COLS = ["Time", "Amount", *V_COLS]

COLORS = {"ALLOW": "#2f7d5b", "REVIEW": "#d98a1f", "BLOCK": "#c8453b"}
ACTION_UA = {"ALLOW": "Дозволити", "REVIEW": "Перевірити", "BLOCK": "Заблокувати"}
MODEL_UA = {
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost_without_anomaly_score": "XGBoost (без Isolation Forest)",
    "hybrid": "Гібридна модель (Isolation Forest + XGBoost)",
    "isolation_forest_only": "Isolation Forest (окремо)",
}
MODEL_ORDER = list(MODEL_UA)
MODEL_COLORS = {
    "random_forest": "#8a8f8c",
    "lightgbm": "#7b5ea7",
    "xgboost_without_anomaly_score": "#3b6ea5",
    "hybrid": "#2f6f55",
    "isolation_forest_only": "#d98a1f",
}
NAV = ["Перевірка транзакцій", "Моделі", "Історія перевірок", "Налаштування"]
NAV_ICONS = {"Перевірка транзакцій": ":material/fact_check:", "Моделі": ":material/monitoring:",
             "Історія перевірок": ":material/history:", "Налаштування": ":material/tune:",
             "Користувачі": ":material/group:"}
SOURCE_UA = {"dataset": "Масив", "single": "Одна транзакція"}
SHORT_UA = {"random_forest": "Random Forest", "lightgbm": "LightGBM",
            "xgboost_without_anomaly_score": "XGBoost", "hybrid": "Гібрид", "isolation_forest_only": "Isolation Forest"}
MONTHS_UA = ["січ.", "лют.", "бер.", "квіт.", "трав.", "черв.", "лип.", "серп.", "вер.", "жовт.", "лист.", "груд."]

st.set_page_config(page_title="Захист · Fraud Monitor", page_icon="🛡️", layout="wide")

def icon_button(label: str, icon: str, **kwargs) -> bool:
    try:
        return st.button(label, icon=icon, **kwargs)
    except TypeError:  # старі версії Streamlit без параметра icon
        return st.button(label, **kwargs)


def columns_bottom(spec):
    try:
        return st.columns(spec, vertical_alignment="bottom")
    except TypeError:  # старі версії Streamlit
        return st.columns(spec)


CSS = """
<style>
:root{--green:#2f6f55;--ink:#1c2b27;--muted:#76827e;--line:#e4e8e6;}
.stApp{background:#f6f7f6;color:var(--ink);}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.1rem;max-width:1320px;}
[data-testid="stSidebar"]{background:#1f2e2a;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label{color:#dfe8e4;}
[data-testid="stSidebar"] button{border-radius:10px;justify-content:flex-start;text-align:left;padding:.55rem .8rem;}
[data-testid="stSidebar"] button[kind="secondary"],[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{background:transparent;border:1px solid transparent;color:#dfe8e4;}
[data-testid="stSidebar"] button[kind="secondary"]:hover,[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover{background:#27403a;border-color:#27403a;color:#fff;}
[data-testid="stSidebar"] button[kind="primary"],[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]{background:#2d4a40;border:1px solid #2d4a40;color:#fff;}
.st-key-logout button{border:1px solid #3a4f47 !important;justify-content:center;}
.brand{display:flex;gap:.7rem;align-items:center;margin:.2rem 0 1.4rem;}
.brand .logo{width:42px;height:42px;border-radius:12px;background:#3d6b5a;display:flex;align-items:center;justify-content:center;font-size:1.3rem;}
.brand b{font-size:1.05rem;color:#fff;display:block;}
.brand small{letter-spacing:.14em;color:#8ea39b;font-size:.62rem;}
.nav-label{font-size:.65rem;letter-spacing:.14em;color:#8ea39b;margin:.5rem 0 .4rem .3rem;}
.side-note{border:1px solid #3a4f47;border-radius:12px;padding:.9rem;font-size:.8rem;color:#c6d3ce;margin-top:1.5rem;}
.side-status{margin-top:1rem;font-size:.78rem;color:#c6d3ce;}
.side-status b{display:block;color:#fff;font-size:.9rem;}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.2rem;font-size:.85rem;color:var(--muted);}
.topbar b{color:var(--ink);font-weight:600;}
.topbar-right{display:flex;gap:1rem;align-items:center;}
.pill{border:1px solid var(--line);background:#fff;border-radius:999px;padding:.25rem .8rem;font-size:.78rem;}
.pill.ok{color:var(--green);} .pill.bad{color:#c8453b;}
.user i{display:inline-block;width:26px;height:26px;border-radius:50%;background:#e3ede8;color:var(--green);text-align:center;line-height:26px;font-style:normal;margin-right:.4rem;font-weight:600;}
.eyebrow{font-size:.68rem;letter-spacing:.16em;color:var(--muted);margin:.6rem 0 .4rem;text-transform:uppercase;}
.page-title{font-size:2.3rem;font-weight:700;margin:.1rem 0 .2rem;}
.page-sub{color:var(--muted);margin-bottom:1.2rem;}
.card-head{display:flex;gap:.8rem;align-items:center;margin-bottom:.4rem;}
.card-head .num{width:30px;height:30px;border-radius:50%;background:#e3ede8;color:var(--green);display:flex;align-items:center;justify-content:center;font-size:.72rem;font-weight:600;}
.card-head b{font-size:1.05rem;display:block;} .card-head small{color:var(--muted);}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1rem 1.2rem;margin-bottom:1rem;}
.card-title{font-weight:600;margin-bottom:.6rem;display:flex;justify-content:space-between;align-items:center;}
.state-row{display:flex;justify-content:space-between;font-size:.9rem;padding:.3rem 0;}
.state-row .lbl::before{content:"●";margin-right:.5rem;color:var(--c,#999);}
.state-row b{font-variant-numeric:tabular-nums;}
.howto-row{display:flex;gap:.7rem;margin:.6rem 0;font-size:.88rem;}
.howto-row .ic{font-weight:700;} .howto-row small{display:block;color:var(--muted);}
.decision{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1rem 1.3rem;margin:1rem 0;}
.decision .t{font-size:1.5rem;font-weight:700;} .decision .s{color:var(--muted);}
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-radius:14px;border-color:var(--line);}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:12px;padding:.7rem 1rem;}
button[kind="primary"]{background:var(--green);border-color:var(--green);}
button[kind="primary"]:hover{background:#265a46;border-color:#265a46;}
.login-brand{text-align:center;margin:8vh 0 1rem;}
.login-brand .logo{width:56px;height:56px;border-radius:16px;background:#2f6f55;display:inline-flex;align-items:center;justify-content:center;font-size:1.7rem;}
.login-brand h1{font-size:1.7rem;margin:.6rem 0 .1rem;} .login-brand p{color:var(--muted);margin:0;}
.side-user{margin-top:1rem;font-size:.8rem;color:#c6d3ce;}
.side-user b{display:block;color:#fff;font-size:.95rem;}
.login-hero{background:#1f2e2a;color:#fff;border-radius:18px;padding:2rem;min-height:470px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;}
.login-hero .logo{width:72px;height:72px;border-radius:20px;background:#3d6b5a;display:flex;align-items:center;justify-content:center;font-size:2.2rem;margin-bottom:1.2rem;}
.login-hero h1{color:#fff;font-size:2.6rem;font-weight:700;margin:0;padding:0;}
.threshold-note{color:var(--muted);font-size:.8rem;margin:.2rem 0 .6rem;}
[data-testid="stSidebar"] button div[data-testid="stMarkdownContainer"]{flex:1;width:auto;}
[data-testid="stSidebar"] button p{text-align:left;font-size:.95rem;}
[class*="st-key-nav_"] button::after{content:"›";font-size:1.4rem;line-height:1;opacity:.55;margin-left:auto;padding-left:.5rem;}
[class*="st-key-nav_"] button[kind="primary"]::after,[class*="st-key-nav_"] [data-testid="stBaseButton-primary"]::after{opacity:1;}
.st-key-logout button div[data-testid="stMarkdownContainer"]{flex:none;}
.st-key-logout button p{text-align:center;}
</style>
"""


# ------------------------------------------------------------------ дані та API
@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None


@st.cache_data(ttl=60)
def get_metrics() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/metrics", headers=HEADERS, timeout=10)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def init_state() -> None:
    for col in INPUT_COLS:
        st.session_state.setdefault(col, 450.0 if col == "Amount" else 14200.0 if col == "Time" else 0.0)


def load_sample(is_fraud: int) -> None:
    df = load_dataset()
    if df is None:
        st.session_state["flash"] = f"Датасет {DATA_PATH} не знайдено"
        return
    row = df[df["Class"] == is_fraud].sample(1).iloc[0]
    for col in INPUT_COLS:
        st.session_state[col] = float(row[col])
    st.session_state["flash"] = None
    st.session_state.pop("single_result", None)


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        d = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{d.day} {MONTHS_UA[d.month - 1]} {d.year} р."


def fmt_int(n: int | float) -> str:
    return f"{int(n):,}".replace(",", " ")


def fmt_money(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ")


def effective_th(health: dict | None) -> dict:
    """Пороги поточного користувача: власні збережені поверх серверних (+ прапорець автоблокування)."""
    cfg = userprefs.load(st.session_state["user"]["username"])
    return userprefs.effective_thresholds(health["thresholds"] if health else {}, cfg)


# ------------------------------------------------------------------ графіки
def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=13),
        legend=dict(orientation="h", y=-0.22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def roc_figure(items: list[tuple]) -> go.Figure:
    """items: (назва, fpr, tpr, auc, колір)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Випадкове вгадування",
                             line=dict(dash="dash", color="#b5beba"), hoverinfo="skip"))
    for name, fpr, tpr, auc, color in items:
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})",
                                 line=dict(color=color, width=2.6)))
    fig.update_xaxes(title="False Positive Rate", range=[0, 1])
    fig.update_yaxes(title="True Positive Rate (Recall)", range=[0, 1.02])
    fig.update_layout(title="ROC-крива")
    return style(fig)


def pr_figure(items: list[tuple], baseline: float | None = None) -> go.Figure:
    """items: (назва, recall, precision, ap, колір)."""
    fig = go.Figure()
    if baseline is not None:
        fig.add_hline(y=baseline, line_dash="dash", line_color="#b5beba",
                      annotation_text=f"Базовий рівень {baseline:.3%}", annotation_position="top right")
    for name, rec, prec, ap, color in items:
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{name} (AP {ap:.3f})",
                                 line=dict(color=color, width=2.6)))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    fig.update_layout(title="Precision–Recall крива")
    return style(fig)


def f1_figure(sweep: dict, review: float | None = None, block: float | None = None) -> go.Figure:
    t = sweep["thresholds"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sweep["f1"], name="F1", line=dict(color="#2f6f55", width=3)))
    fig.add_trace(go.Scatter(x=t, y=sweep["precision"], name="Precision", line=dict(color="#3b6ea5", dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=sweep["recall"], name="Recall", line=dict(color="#d98a1f", dash="dot")))
    best = int(np.argmax(sweep["f1"]))
    fig.add_trace(go.Scatter(x=[t[best]], y=[sweep["f1"][best]], mode="markers+text", name="Максимум F1",
                             marker=dict(size=11, color="#2f6f55"),
                             text=[f"F1 {sweep['f1'][best]:.3f} @ {t[best]:.2f}"], textposition="top center"))
    if review is not None:
        fig.add_vline(x=review, line_dash="dash", line_color=COLORS["REVIEW"], annotation_text="перевірка")
    if block is not None:
        fig.add_vline(x=block, line_dash="dash", line_color=COLORS["BLOCK"], annotation_text="блок")
    fig.update_xaxes(title="Поріг ймовірності", range=[0, 1])
    fig.update_yaxes(title="Значення метрики", range=[0, 1.05])
    fig.update_layout(title="F1, Precision та Recall залежно від порога")
    return style(fig)


def cm_figure(m: dict, title: str) -> go.Figure:
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig = go.Figure(go.Heatmap(
        z=np.log1p(z), text=[[fmt_int(v) for v in row] for row in z], texttemplate="%{text}",
        x=["Прогноз: легітимна", "Прогноз: шахрайська"], y=["Факт: легітимна", "Факт: шахрайська"],
        colorscale=[[0, "#f1f5f3"], [1, "#2f6f55"]], showscale=False, hoverinfo="skip",
        textfont=dict(size=18),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title)
    return style(fig, 320)


def explanation_figure(expl: list[dict]) -> go.Figure:
    d = pd.DataFrame(expl).sort_values("contribution")
    labels = [f"{r.feature} = {r.value:g}" for r in d.itertuples()]
    colors = [COLORS["BLOCK"] if c > 0 else COLORS["ALLOW"] for c in d["contribution"]]
    fig = go.Figure(go.Bar(x=d["contribution"], y=labels, orientation="h", marker_color=colors,
                           hovertemplate="%{y}<br>вклад: %{x:+.3f}<extra></extra>"))
    fig.update_xaxes(title="← знижує ризик · підвищує ризик →", zeroline=True)
    fig.update_layout(title="Вклад ознак у log-odds шахрайства", showlegend=False)
    return style(fig, 120 + 30 * len(d))


# ------------------------------------------------------------------ елементи інтерфейсу
def brand() -> None:
    st.markdown(
        '<div class="brand"><div class="logo">🛡️</div><div><b>Захист</b><small>FRAUD MONITOR</small></div></div>'
        '<div class="nav-label">РОБОЧИЙ ПРОСТІР</div>',
        unsafe_allow_html=True,
    )


def topbar(page: str, health: dict | None, user: dict) -> None:
    pill = ('<span class="pill ok">● Сервіс готовий</span>' if health
            else '<span class="pill bad">● API недоступний</span>')
    name = user["username"]
    st.markdown(
        f'<div class="topbar"><div>Моніторинг ризику / <b>{page}</b></div>'
        f'<div class="topbar-right">{pill}<span class="user"><i>{name[:1].upper()}</i>{name}</span></div></div>',
        unsafe_allow_html=True,
    )


def page_head(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div>'
        f'<div class="page-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def card_head(num: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="card-head"><div class="num">{num}</div><div><b>{title}</b><small>{subtitle}</small></div></div>',
        unsafe_allow_html=True,
    )


def state_card(health: dict | None) -> None:
    th = effective_th(health)
    pct = lambda k: f"{th[k] * 100:.0f}%" if k in th else "—"  # noqa: E731
    anomaly = f"{th['anomaly_review']:.4f}".replace(".", ",") if "anomaly_review" in th else "—"
    ok = health is not None
    st.markdown(
        '<div class="card"><div class="card-title"><span>📈 Стан моделі</span>'
        f'<span style="color:{"#2f7d5b" if ok else "#c8453b"}">●</span></div>'
        f'<div style="font-weight:600">{"Ok" if ok else "Недоступно"}</div>'
        '<div style="color:#76827e;font-size:.8rem;margin-bottom:.8rem">fraud-detection-api</div>'
        '<div class="eyebrow">Пороги рівнів</div>'
        f'<div class="state-row"><span class="lbl" style="--c:{COLORS["REVIEW"]}">Перевірка</span><b>{pct("review")}</b></div>'
        f'<div class="state-row"><span class="lbl" style="--c:{COLORS["BLOCK"]}">Блокування</span>'
        f'<b>{pct("block") if th.get("block_enabled", True) else "вимкнено"}</b></div>'
        f'<div class="state-row"><span class="lbl" style="--c:#3b6ea5">Поріг аномалії</span><b>{anomaly}</b></div>'
        '<hr style="margin:.6rem 0;border:none;border-top:1px solid #e4e8e6">'
        '<div class="state-row"><span style="color:#76827e">Останнє навчання</span>'
        f'<b>{fmt_date(health.get("trained_at")) if health else "—"}</b></div></div>',
        unsafe_allow_html=True,
    )


def howto_card() -> None:
    st.markdown(
        '<div class="card"><div class="card-title">ⓘ Як читати рішення</div>'
        f'<div class="howto-row"><span class="ic" style="color:{COLORS["ALLOW"]}">✓</span><div>Дозволити<small>Низький ризик за оцінкою моделі.</small></div></div>'
        f'<div class="howto-row"><span class="ic" style="color:{COLORS["REVIEW"]}">!</span><div>Перевірити<small>Потрібна увага аналітика.</small></div></div>'
        f'<div class="howto-row"><span class="ic" style="color:{COLORS["BLOCK"]}">✕</span><div>Заблокувати<small>Високий ризик шахрайства.</small></div></div>'
        '<div style="background:#f1f5f3;border-radius:10px;padding:.6rem .8rem;font-size:.8rem;color:#4b5c56">'
        'Модель показує ймовірність та фактори, що вплинули на рішення.</div></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ сторінка: перевірка
def gauge_figure(prob: float, th: dict, action: str | None = None) -> go.Figure:
    review, block = th.get("review", 0.5) * 100, th.get("block", 0.9) * 100
    pct = max(0.0, min(prob * 100, 100.0))
    if action is None:
        action = "BLOCK" if pct >= block else "REVIEW" if pct >= review else "ALLOW"
    color = COLORS[action]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={"suffix": "%", "valueformat": ".1f", "font": {"color": color, "size": 44}},
        title={"text": "Імовірність шахрайства"},
        gauge={"axis": {"range": [0, 100], "tickvals": [0, review, block, 100], "ticksuffix": "%"},
               "bar": {"color": color, "thickness": 0.62},
               "bgcolor": "#eef1ef", "borderwidth": 0,
               "steps": [{"range": [0, review], "color": "#dcefe4"},
                         {"range": [review, block], "color": "#fbe9cc"},
                         {"range": [block, 100], "color": "#f6d3cf"}]},
    ))
    return style(fig, 260)


def explanation_words(expl: list[dict]) -> list[str]:
    ups = [e for e in expl if e["contribution"] > 0][:3]
    downs = [e for e in expl if e["contribution"] < 0][:3]
    lines = [f"🔺 **{e['feature']}** = {e['value']:g} підвищує ризик ({e['contribution']:+.2f})" for e in ups]
    lines += [f"🔻 **{e['feature']}** = {e['value']:g} знижує ризик ({e['contribution']:+.2f})" for e in downs]
    return lines


def render_single_result(res: dict, th: dict) -> None:
    prob, expl = res["fraud_probability"], res.get("explanation")
    action = userprefs.decide(res["action"], prob, th)  # з урахуванням власних порогів
    color = COLORS[action]
    show = st.session_state.get("show_factors", True) and bool(expl)
    left, right = st.columns([1, 1.5], gap="large") if show else (st.container(), None)
    with left:
        st.markdown(
            f'<div class="decision" style="border-left:6px solid {color};margin-top:0">'
            f'<div class="t" style="color:{color}">{ACTION_UA[action]}</div>'
            f'<div class="s">Показник аномальності {res["anomaly_score"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(gauge_figure(prob, th, action), use_container_width=True)
    if right is not None:
        with right:
            st.markdown("**Чому модель так вирішила (XAI)**")
            st.caption("Вклад ознак у log-odds шахрайства (значення SHAP): червоні підвищують ризик, зелені знижують.")
            st.plotly_chart(explanation_figure(expl), use_container_width=True)
            for line in explanation_words(expl):
                st.markdown(line)
            with st.expander("Таблиця факторів"):
                st.dataframe(pd.DataFrame(expl), hide_index=True, use_container_width=True)
    elif not st.session_state.get("show_factors", True):
        st.caption("Фактори рішення приховано: увімкніть «Показати фактори рішення» у формі.")


def result_section(th: dict) -> None:
    res = st.session_state.get("single_result")
    with st.container(border=True):
        card_head("02", "Результат і пояснення моделі (XAI)", "Чому система ухвалила саме таке рішення")
        if not res:
            st.info("Натисніть «Перевірити транзакцію»: тут з'являться рішення, шкала ймовірності шахрайства "
                    "та графік вкладу ознак (SHAP).")
        else:
            render_single_result(res, th)


def register_check(df: pd.DataFrame, label: str, source: str, th: dict, health: dict | None = None,
                   elapsed: float | None = None) -> None:
    """Зберігає перевірку в історію користувача (SQLite) і надсилає сповіщення, якщо є BLOCK."""
    user = st.session_state["user"]["username"]
    trained = ((health or {}).get("trained_at") or "")[:10]
    try:
        history.record_check(user, df, label=label, source=source, review_th=th.get("review"),
                             block_th=th.get("block") if th.get("block_enabled", True) else None,
                             model_version=f"hybrid-{trained}" if trained else "hybrid", elapsed=elapsed)
    except Exception as exc:  # noqa: BLE001 — історія не повинна ламати перевірку
        st.warning(f"Не вдалося зберегти перевірку в історію: {exc}")
    blocked = df["action"] == "BLOCK"
    if blocked.any():
        for msg in notify.notify_block(userprefs.load(user), label=label, n_block=int(blocked.sum()),
                                       amount=float(df.loc[blocked, "Amount"].sum()), rows=len(df)):
            st.toast(msg)


def single_mode(health: dict | None) -> None:
    st.markdown("<div class='eyebrow'>Обов'язкові параметри</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    a.number_input("Час від початку вибірки (сек)", min_value=0.0, step=60.0, key="Time")
    b.number_input("Сума транзакції", min_value=0.0, step=10.0, key="Amount")
    with st.expander("Додати ознаки моделі (V1–V28, необов'язково)"):
        grid = st.columns(4)
        for i, col in enumerate(V_COLS):
            grid[i % 4].number_input(col, step=0.1, format="%.3f", key=col)

    st.markdown("<div class='eyebrow'>Синтетичні приклади</div>", unsafe_allow_html=True)
    s1, s2, _ = st.columns([1, 1, 2])
    s1.button("✓ Легітимна", on_click=load_sample, args=(0,), use_container_width=True)
    s2.button("! Підозріла", on_click=load_sample, args=(1,), use_container_width=True)
    if st.session_state.get("flash"):
        st.warning(st.session_state["flash"])

    f1, f2 = st.columns([1.2, 1])
    f1.checkbox("Показати фактори рішення", value=True, key="show_factors")
    run = f2.button("Перевірити транзакцію", type="primary", use_container_width=True)
    if run:
        payload = {c: float(st.session_state[c]) for c in INPUT_COLS}
        try:
            resp = requests.post(f"{API_URL}/predict", params={"explain": True, "top_k": 10}, json=payload,
                                 headers=HEADERS, timeout=10)
        except requests.RequestException as exc:
            st.error(f"Помилка з'єднання з API: {exc}")
        else:
            if resp.status_code != 200:
                st.error(f"API повернув {resp.status_code}: {resp.text}")
            else:
                res = resp.json()
                st.session_state["single_result"] = res
                th = effective_th(health)
                one = pd.DataFrame([{"Time": payload["Time"], "Amount": payload["Amount"],
                                     "fraud_probability": res["fraud_probability"],
                                     "anomaly_score": res["anomaly_score"],
                                     "action": userprefs.decide(res["action"], res["fraud_probability"], th)}])
                register_check(one, f"Time {payload['Time']:.0f} · Amount {payload['Amount']:.2f}", "single", th,
                               health, resp.elapsed.total_seconds())


def score_batch(data: pd.DataFrame) -> pd.DataFrame:
    payload_df = data[INPUT_COLS]
    n, results = len(payload_df), []
    bar = st.progress(0.0, text="Оцінюємо транзакції…")
    for start in range(0, n, 1000):
        chunk = payload_df.iloc[start : start + 1000].to_dict("records")
        r = requests.post(f"{API_URL}/predict/batch", json={"transactions": chunk}, headers=HEADERS, timeout=120)
        r.raise_for_status()
        results.extend(r.json()["results"])
        bar.progress(min((start + 1000) / n, 1.0), text=f"Оброблено {min(start + 1000, n)} з {n}")
    bar.empty()
    res_df = pd.DataFrame(results).drop(columns="explanation", errors="ignore")
    return pd.concat([data.reset_index(drop=True), res_df], axis=1)


MODE_SINGLE, MODE_ARRAY = "Одна транзакція", "Масив даних"
SRC_SAVED, SRC_UPLOAD, SRC_MANUAL = "Раніше завантажені", "Завантажити CSV", "Ввести вручну"


def df_signature(df: pd.DataFrame) -> str:
    """Короткий відбиток вмісту: щоб не показувати результати від іншого масиву."""
    return f"{len(df)}-{int(pd.util.hash_pandas_object(df, index=False).sum()) % 10**12}"


@st.cache_data(show_spinner=False, max_entries=8)
def load_saved_array(username: str, name: str, saved_at: str) -> pd.DataFrame:
    return datasets.load_dataset(username, name)


def limit_rows(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > MAX_BATCH_ROWS:
        st.info(f"Масив містить {fmt_int(len(df))} рядків. Для швидкості беремо випадкову вибірку "
                f"з {fmt_int(MAX_BATCH_ROWS)} (змінюється змінною MAX_BATCH_ROWS).")
        return df.sample(MAX_BATCH_ROWS, random_state=42).reset_index(drop=True)
    return df


def array_source(prefix: str) -> tuple[pd.DataFrame | None, str | None]:
    """Вибір масиву: раніше збережений, новий CSV або ручне введення (Time, V1–V28, Amount, Class)."""
    username = st.session_state["user"]["username"]
    saved = datasets.list_datasets(username)
    options = ([SRC_SAVED] if saved else []) + [SRC_UPLOAD, SRC_MANUAL]
    source = st.radio("Джерело даних", options, horizontal=True, key=f"{prefix}_source")

    if source == SRC_SAVED:
        labels = {m["name"]: f"{m['name']} · {fmt_int(m['rows'])} рядків"
                           + (f" · шахрайських {fmt_int(m['frauds'])}" if m.get("frauds") is not None else "")
                  for m in saved}
        c1, c2 = columns_bottom([5, 1])
        name = c1.selectbox("Масив", list(labels), format_func=labels.get, key=f"{prefix}_saved")
        meta = next(m for m in saved if m["name"] == name)
        with c2:
            deleted = icon_button("Видалити", ":material/delete:", key=f"{prefix}_delete", use_container_width=True)
        if deleted:
            datasets.delete_dataset(username, name)
            st.rerun()
        origin = "CSV" if meta.get("source") == "csv" else "ручне введення"
        st.caption(f"Збережено {meta['saved_at'][:16].replace('T', ' ')} UTC · джерело: {origin}")
        return limit_rows(load_saved_array(username, name, meta["saved_at"])), name

    if source == SRC_UPLOAD:
        upload = st.file_uploader("CSV-файл", type="csv", key=f"{prefix}_upload", label_visibility="collapsed")
        st.caption("Колонки: Time, Amount, (V1–V28), Class (необов'язково). Файл автоматично зберігається у ваші масиви; "
                   "файл із тією ж назвою оновлюється.")
        if upload is None:
            return None, None
        raw = upload.getvalue()
        sha = hashlib.md5(raw).hexdigest()
        name = datasets.find_by_sha(username, sha)
        try:
            if name is None:
                name = datasets.save_dataset(username, Path(upload.name).stem, pd.read_csv(io.BytesIO(raw)), "csv", sha)
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            st.error(f"Не вдалося прочитати файл: {exc}")
            return None, None
        meta = next(m for m in datasets.list_datasets(username) if m["name"] == name)
        st.caption(f"Збережено як «{name}», надалі його можна вибрати у списку «{SRC_SAVED}».")
        return limit_rows(load_saved_array(username, name, meta["saved_at"])), name

    st.caption("Введіть дані так само, як у тренувальному наборі: Time, V1–V28, Amount (та за потреби Class). "
               "Додайте рядки кнопкою «+» внизу таблиці або вставте їх з Excel (Ctrl+V). "
               "Time: секунди від початку спостереження (година доби = (Time mod 86400) / 3600). "
               "Незаповнені V1–V28 вважаються нулями. Class (0/1) заповнюйте, лише якщо знаєте факт: "
               "тоді можна порахувати метрики якості.")
    columns = ["Time", *V_COLS, "Amount", "Class"]
    base = load_dataset()
    if base is not None and set(columns) <= set(base.columns):
        template = pd.concat([base[base["Class"] == 0].head(2), base[base["Class"] == 1].head(1)])[columns]
        template = template.reset_index(drop=True)
    else:
        template = pd.DataFrame({c: [0.0, 0.0, 0.0] for c in columns})
        template["Time"] = [3600.0, 50400.0, 82800.0]
        template["Amount"] = [25.0, 450.0, 1800.0]
    template["Class"] = np.nan
    column_config = {c: st.column_config.NumberColumn(c, step=0.01, format="%.4f") for c in V_COLS}
    column_config["Time"] = st.column_config.NumberColumn("Time", min_value=0.0, step=60.0, format="%.0f")
    column_config["Amount"] = st.column_config.NumberColumn("Amount", min_value=0.0, step=1.0, format="%.2f")
    column_config["Class"] = st.column_config.NumberColumn("Class", min_value=0, max_value=1, step=1, format="%d")
    edited = st.data_editor(template, num_rows="dynamic", hide_index=True, use_container_width=True,
                            key=f"{prefix}_editor", column_config=column_config).copy()
    edited[V_COLS] = edited[V_COLS].fillna(0.0)
    df = edited.dropna(subset=["Time", "Amount"]).reset_index(drop=True)
    if df.empty:
        st.info("Додайте хоча б один рядок із часом і сумою.")
        return None, None
    if df["Class"].isna().all():
        df = df.drop(columns="Class")
    elif df["Class"].isna().any():
        st.warning("Клас заповнено не в усіх рядках, тому колонку Class ігноруємо (метрики якості недоступні).")
        df = df.drop(columns="Class")
    try:
        df = datasets.normalize(df)
    except ValueError as exc:
        st.error(str(exc))
        return None, None
    c1, c2 = columns_bottom([3, 1])
    name = c1.text_input("Назва масиву", value="Ручне введення", key=f"{prefix}_manual_name")
    with c2:
        save = icon_button("Зберегти", ":material/save:", key=f"{prefix}_manual_save", use_container_width=True)
    if save:
        try:
            st.success(f"Збережено як «{datasets.save_dataset(username, name, df, 'manual')}».")
        except ValueError as exc:
            st.error(str(exc))
    return df, name.strip() or "Ручне введення"


PICK_LIMIT = 2000


def predict_one(row: pd.Series) -> dict | None:
    payload = {c: float(row[c]) for c in INPUT_COLS}
    try:
        resp = requests.post(f"{API_URL}/predict", params={"explain": True, "top_k": 10}, json=payload,
                             headers=HEADERS, timeout=10)
    except requests.RequestException as exc:
        st.error(f"Помилка з'єднання з API: {exc}")
        return None
    if resp.status_code != 200:
        st.error(f"API повернув {resp.status_code}: {resp.text}")
        return None
    return resp.json()


def pick_transaction_section(df: pd.DataFrame, th: dict) -> None:
    """Вибір однієї транзакції із завантаженого масиву (за часом і сумою) та її перевірка з поясненням."""
    sig = df_signature(df)
    with st.container(border=True):
        card_head("02", "Перевірити конкретну транзакцію з масиву",
                  "Відфільтруйте за часом і сумою та клацніть рядок у таблиці")
        f = st.columns(4)
        t_from = f[0].number_input("Time від (сек)", value=float(df["Time"].min()), step=60.0, format="%.0f", key=f"pick_tf_{sig}")
        t_to = f[1].number_input("Time до (сек)", value=float(df["Time"].max()), step=60.0, format="%.0f", key=f"pick_tt_{sig}")
        a_from = f[2].number_input("Amount від", value=float(df["Amount"].min()), step=10.0, key=f"pick_af_{sig}")
        a_to = f[3].number_input("Amount до", value=float(df["Amount"].max()), step=10.0, key=f"pick_at_{sig}")

        mask = df["Time"].between(t_from, t_to) & df["Amount"].between(a_from, a_to)
        view = df.loc[mask].copy()
        if view.empty:
            st.info("За цим фільтром транзакцій немає. Розширте діапазон часу або суми.")
            return
        view.insert(0, "№", view.index + 1)
        sec = view["Time"] % 86400
        view["Година доби"] = ((sec // 3600).astype(int).astype(str).str.zfill(2) + ":"
                               + ((sec % 3600) // 60).astype(int).astype(str).str.zfill(2))
        cols = ["№", "Time", "Година доби", "Amount"] + (["Class"] if "Class" in view.columns else [])
        shown = view[cols].head(PICK_LIMIT).reset_index(drop=True)
        note = f" Показано перші {fmt_int(PICK_LIMIT)}: звузьте фільтр." if len(view) > PICK_LIMIT else ""
        st.caption(f"Знайдено {fmt_int(len(view))} транзакцій.{note}")

        try:
            event = st.dataframe(shown, hide_index=True, use_container_width=True, height=260,
                                 on_select="rerun", selection_mode="single-row", key=f"pick_table_{sig}")
            rows = list(event.selection.rows)
        except TypeError:  # старі версії Streamlit без вибору рядків у таблиці
            i = st.selectbox("Транзакція", range(min(len(shown), 500)), key=f"pick_sel_{sig}",
                             format_func=lambda k: f"№{shown.iloc[k]['№']} · Time {shown.iloc[k]['Time']:.0f} · Amount {shown.iloc[k]['Amount']:.2f}")
            rows = [i]
        if not rows or rows[0] >= len(shown):
            st.info("Клацніть рядок у таблиці: нижче з'являться рішення та пояснення моделі.")
            return

        idx = int(shown.iloc[rows[0]]["№"]) - 1
        row = df.iloc[idx]
        key = f"{sig}-{idx}"
        cached = st.session_state.get("pick_result")
        if not cached or cached["key"] != key:
            res = predict_one(row)
            if res is None:
                return
            cached = st.session_state["pick_result"] = {"key": key, "res": res}

        st.markdown(f"**Транзакція №{idx + 1}:** Time {row['Time']:.0f} · Amount {row['Amount']:.2f}")
        if "Class" in df.columns and pd.notna(row["Class"]):
            st.caption("Фактична мітка в масиві: " + ("шахрайська" if int(row["Class"]) == 1 else "легітимна"))
        render_single_result(cached["res"], th)


def batch_mode(health: dict | None) -> tuple[pd.DataFrame | None, str | None]:
    """Вибір масиву та запуск перевірки. Повертає (дані, назва)."""
    df, name = array_source("check")
    if df is None:
        return None, None
    info = f"«{name}»: {fmt_int(len(df))} транзакцій"
    if "Class" in df.columns:
        info += f", шахрайських {fmt_int(pd.to_numeric(df['Class'], errors='coerce').eq(1).sum())}"
    st.caption(info)
    if st.button("Запустити перевірку", type="primary", key="run_batch"):
        try:
            started = time.perf_counter()
            scored = score_batch(df)
            elapsed = time.perf_counter() - started
        except requests.RequestException as exc:
            st.error(f"Помилка API: {exc}")
        else:
            st.session_state["batch"] = {"sig": df_signature(df), "name": name, "df": scored}
            th = effective_th(health)
            register_check(userprefs.reclassify(scored, th), name, "dataset", th, health, elapsed)
    return df, name


# ------------------------------------------------------------------ PDF-звіти
def make_pdf(**kwargs) -> bytes | None:
    try:
        try:
            from app import report
        except ImportError:
            import report  # type: ignore[no-redef]
    except ImportError as exc:
        st.error(f"Для PDF-звіту встановіть бібліотеки: pip install reportlab matplotlib ({exc})")
        return None
    try:
        return report.build_report(**kwargs)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не вдалося сформувати PDF: {exc}")
        return None


def pdf_download(key: str, label: str, filename: str, **report_kwargs) -> None:
    """Двоетапно: кнопка формує звіт, після чого з'являється кнопка завантаження."""
    if icon_button(label, ":material/picture_as_pdf:", key=f"mk_{key}"):
        with st.spinner("Формуємо PDF-звіт…"):
            pdf = make_pdf(**report_kwargs)
        if pdf:
            st.session_state[f"pdf_{key}"] = pdf
    if st.session_state.get(f"pdf_{key}"):
        st.download_button("Завантажити PDF", st.session_state[f"pdf_{key}"], filename, "application/pdf",
                           key=f"dl_{key}")


def report_meta(name: str, n: int, health: dict | None, th: dict | None = None) -> dict[str, str]:
    th = th if th is not None else (health["thresholds"] if health else {})
    meta = {"Масив даних": name, "Кількість транзакцій": fmt_int(n),
            "Користувач": st.session_state["user"]["username"],
            "Моделі навчено": fmt_date(health.get("trained_at")) if health else "—"}
    if "review" in th and "block" in th:
        block = f"{th['block']:.0%}" if th.get("block_enabled", True) else "вимкнено"
        meta["Пороги рішень"] = f"перевірка {th['review']:.0%}, блокування {block}"
    return meta


# ------------------------------------------------------------------ пороги рішень
def _clip_th(x: float, default: float) -> float:
    return float(min(max(round(x if x is not None else default, 2), 0.01), 0.99))


def _save_thresholds(prefix: str, user: str) -> None:
    review, block = st.session_state[f"{prefix}_th_review"], st.session_state[f"{prefix}_th_block"]
    cfg = userprefs.load(user)
    cfg["thresholds"] = {"custom": True, "review": review, "block": max(block, review),
                         "block_enabled": bool(st.session_state[f"{prefix}_th_on"])}
    userprefs.save(user, cfg)
    st.session_state["th_flash"] = "Пороги збережено: вони діятимуть для наступних перевірок."


def _reset_thresholds(prefix: str, user: str, server: dict) -> None:
    cfg = userprefs.load(user)
    cfg["thresholds"] = dict(userprefs.DEFAULTS["thresholds"])
    userprefs.save(user, cfg)
    st.session_state[f"{prefix}_th_review"] = _clip_th(server.get("review"), 0.5)
    st.session_state[f"{prefix}_th_block"] = _clip_th(server.get("block"), 0.9)
    st.session_state[f"{prefix}_th_on"] = True
    st.session_state["th_flash"] = "Повернуто типові пороги моделі."


def threshold_panel(df: pd.DataFrame | None, health: dict | None, prefix: str) -> dict:
    """Повзунки порогів «Перевірити» / «Заблокувати» з підрахунком кількості та суми в кожній категорії.
    df: результати оцінки (fraud_probability, Amount, action від API). Повертає чинні пороги."""
    user = st.session_state["user"]["username"]
    server = health["thresholds"] if health else {}
    saved = effective_th(health)
    kr, kb, ko = f"{prefix}_th_review", f"{prefix}_th_block", f"{prefix}_th_on"
    st.session_state.setdefault(kr, _clip_th(saved.get("review"), 0.5))
    st.session_state.setdefault(kb, _clip_th(saved.get("block"), 0.9))
    st.session_state.setdefault(ko, bool(saved.get("block_enabled", True)))

    with st.container(border=True):
        card_head("%", "Пороги рішень", "Підберіть пороги та одразу побачте, скільки транзакцій і грошей потрапить у кожну категорію")
        c1, c2 = st.columns(2)
        c1.slider("Поріг «Перевірити»", 0.01, 0.99, step=0.01, key=kr,
                  help="Імовірність шахрайства від цього значення — на ручну перевірку")
        c2.slider("Поріг «Заблокувати»", 0.01, 0.99, step=0.01, key=kb,
                  help="Імовірність шахрайства від цього значення — блокування")
        st.toggle("Автоматичне блокування", key=ko,
                  help="Вимкнено: транзакції з високим ризиком не блокуються, а йдуть у «Перевірити».")
        review, block, enabled = st.session_state[kr], st.session_state[kb], st.session_state[ko]
        if block < review:
            st.warning("Поріг блокування не може бути нижчим за поріг перевірки: використано значення порога перевірки.")
            block = review
        # якщо значення збігається з серверним порогом (з точністю до 0,01) — лишаємо точне серверне
        if "review" in server and round(server["review"], 2) == review:
            review = server["review"]
        if "block" in server and round(server["block"], 2) == block:
            block = server["block"]
        live = {**server, "review": review, "block": block, "block_enabled": enabled,
                "custom": userprefs.differs(server, review, block)}

        if df is None or df.empty:
            st.caption("Запустіть перевірку масиву даних, щоб побачити кількість і суму транзакцій у кожній категорії.")
        else:
            now, before = userprefs.reclassify(df, live), userprefs.reclassify(df, saved)
            cols = st.columns(3)
            for col, a in zip(cols, ("ALLOW", "REVIEW", "BLOCK")):
                m, m0 = now["action"] == a, before["action"] == a
                col.metric(ACTION_UA[a], fmt_int(m.sum()),
                           f"{int(m.sum() - m0.sum()):+d} відносно збережених" if m.sum() != m0.sum() else None,
                           delta_color="off")
                col.caption(f"Сума: {fmt_money(float(now.loc[m, 'Amount'].sum()))} · {m.mean():.2%} транзакцій")

        b1, b2, _ = st.columns([1, 1, 2])
        b1.button("Зберегти пороги", type="primary", key=f"{prefix}_th_save", use_container_width=True,
                  on_click=_save_thresholds, args=(prefix, user))
        b2.button("Типові пороги", key=f"{prefix}_th_reset", use_container_width=True,
                  on_click=_reset_thresholds, args=(prefix, user, server))
        if saved.get("custom"):
            st.caption(f"Збережені власні пороги: перевірка {saved['review']:.0%}, блокування {saved['block']:.0%}"
                       + ("" if saved.get("block_enabled", True) else " (автоблокування вимкнено)"))
        elif not saved.get("block_enabled", True):
            st.caption("Збережено: автоблокування вимкнено.")
        flash = st.session_state.pop("th_flash", None)
        if flash:
            st.success(flash)
    return live


# ------------------------------------------------------------------ сторінка: налаштування
def notification_card() -> None:
    user = st.session_state["user"]["username"]
    n = userprefs.load(user)["notify"]
    with st.container(border=True):
        card_head("@", "Сповіщення про BLOCK", "E-mail або push на телефон, коли перевірка знаходить заблоковані транзакції")
        with st.form("notify_form"):
            on = st.toggle("Сповіщати, коли знайдено BLOCK", value=n["on_block"])
            c1, c2 = st.columns(2)
            email_on = c1.checkbox("E-mail", value=n["email_enabled"])
            email = c1.text_input("Адреса e-mail", value=n["email"], placeholder="name@example.com")
            push_on = c2.checkbox("Push (ntfy)", value=n["push_enabled"])
            topic = c2.text_input("Тема ntfy", value=n["push_topic"], placeholder="fraud-monitor-xxxxxx",
                                  help="Підпишіться на цю тему в застосунку ntfy. Тема працює як пароль каналу: "
                                       "візьміть довгу випадкову (6–64 символи: латиниця, цифри, _ -).")
            save = st.form_submit_button("Зберегти сповіщення", type="primary")
        if save:
            if email_on and not notify.EMAIL_RE.match(email.strip()):
                st.error("Некоректна адреса e-mail.")
            elif push_on and not notify.TOPIC_RE.match(topic.strip()):
                st.error("Тема push: 6–64 символи, лише латиниця, цифри, _ та -.")
            else:
                cfg = userprefs.load(user)
                cfg["notify"] = {"on_block": on, "email_enabled": email_on, "email": email.strip(),
                                 "push_enabled": push_on, "push_topic": topic.strip()}
                userprefs.save(user, cfg)
                st.success("Налаштування сповіщень збережено.")
        if (n["email_enabled"]) and not notify.email_configured():
            st.info("На сервері не задано SMTP_HOST, тому листи не надсилатимуться (див. змінні SMTP_* у README).")
        if st.button("Надіслати тестове сповіщення", key="notify_test"):
            res = notify.notify_block(userprefs.load(user), label="тест", n_block=1, amount=0.0, rows=1)
            for msg in res or ["Увімкніть e-mail або push, вкажіть адресу чи тему та збережіть налаштування."]:
                st.write(msg)


def page_settings(health: dict | None) -> None:
    page_head("Налаштування", "Налаштування", "Пороги рішень і сповіщення про заблоковані транзакції.")
    batch = st.session_state.get("batch")
    threshold_panel(batch["df"] if batch else None, health, "st")
    notification_card()


# ------------------------------------------------------------------ сторінка: історія перевірок
def page_history() -> None:
    user = st.session_state["user"]["username"]
    page_head("Журнал", "Історія перевірок", "Усі ваші перевірки: підсумки, пороги та заблоковані транзакції.")
    checks = history.list_checks(user)
    if checks.empty:
        st.info("Перевірок ще немає. Запустіть перевірку транзакції або масиву: результат з'явиться тут.")
        return

    view = pd.DataFrame({
        "№": checks["id"],
        "Дата (UTC)": checks["created_at"].str.replace("T", " ").str[:16],
        "Назва": checks["label"],
        "Тип": checks["source"].map(SOURCE_UA).fillna(checks["source"]),
        "Рядків": checks["rows"],
        "Дозволено": checks["n_allow"], "Перевірити": checks["n_review"], "Заблоковано": checks["n_block"],
        "Знято блок": checks["n_released"], "Сума BLOCK": checks["amt_block"].round(2),
    })
    with st.container(border=True):
        st.dataframe(view, hide_index=True, use_container_width=True)

    options = {f"№{r.id} · {r.created_at[:16].replace('T', ' ')} · {r.label}": int(r.id) for r in checks.itertuples()}
    chosen = st.selectbox("Деталі перевірки", list(options), key="hist_sel")
    cid = options[chosen]
    row = checks[checks["id"] == cid].iloc[0]
    k = st.columns(4)
    for col, (label, n, amt) in zip(k, (("Дозволено", row.n_allow, row.amt_allow),
                                        ("Перевірити", row.n_review, row.amt_review),
                                        ("Заблоковано", row.n_block, row.amt_block))):
        col.metric(label, fmt_int(n))
        col.caption(f"Сума: {fmt_money(amt)}")
    k[3].metric("Середній ризик", f"{row.mean_prob:.2%}")
    if pd.notna(row.review_th):
        block_txt = f"{row.block_th:.0%}" if pd.notna(row.block_th) else "вимкнено"
        st.caption(f"Пороги цієї перевірки: перевірка ≥ {row.review_th:.0%}, блокування ≥ {block_txt}")

    blocked = history.list_blocked(user, cid)
    if not blocked.empty:
        st.markdown("<div class='eyebrow'>Заблоковані транзакції</div>", unsafe_allow_html=True)
        table = pd.DataFrame({
            "Зняти блок": blocked["status"] == "released",
            "Time": blocked["time"], "Amount": blocked["amount"], "Ймовірність": blocked["prob"].round(3),
            "Статус": blocked["status"].map({"blocked": "Заблоковано", "released": "Блок знято"}),
        })
        edited = st.data_editor(
            table, hide_index=True, use_container_width=True, key=f"hist_blocked_{cid}",
            disabled=["Time", "Amount", "Ймовірність", "Статус"],
            column_config={"Зняти блок": st.column_config.CheckboxColumn(help="Позначте, щоб зняти блок; зніміть позначку, щоб повернути")})
        if icon_button("Застосувати зміни", ":material/lock_open:", key=f"hist_apply_{cid}"):
            want = edited["Зняти блок"].to_numpy()
            now = (blocked["status"] == "released").to_numpy()
            release, reblock = blocked.loc[want & ~now, "id"].tolist(), blocked.loc[~want & now, "id"].tolist()
            history.set_blocked_status(user, release, "released")
            history.set_blocked_status(user, reblock, "blocked")
            st.session_state["hist_flash"] = f"Блок знято: {len(release)}, блок повернуто: {len(reblock)}."
            st.rerun()
        if int(row.n_block) > len(blocked):
            st.caption(f"Показано {fmt_int(len(blocked))} із {fmt_int(row.n_block)} найризикованіших заблокованих транзакцій.")
    if st.session_state.get("hist_flash"):
        st.success(st.session_state.pop("hist_flash"))

    with st.expander("Усі транзакції цієї перевірки (оцінки ризику)"):
        tx = history.list_transactions(user, cid)
        if tx.empty:
            st.caption("Транзакції не збережено.")
        else:
            st.dataframe(tx.rename(columns={
                "transaction_id": "ID транзакції", "time_sec": "Time", "amount": "Amount", "log_amount": "log(Amount)",
                "anomaly_score": "Аномальність", "fraud_probability": "Ймовірність", "action_taken": "Рішення",
                "is_fraud_actual": "Class", "model_version": "Версія моделі"}),
                hide_index=True, use_container_width=True)
            st.caption(f"Показано до 1000 найризикованіших із {fmt_int(row.rows)}.")

    d1, d2, _ = st.columns([1, 1, 2])
    if icon_button("Видалити запис", ":material/delete:", key=f"hist_del_{cid}", use_container_width=True):
        history.delete_check(user, cid)
        st.rerun()
    if icon_button("Очистити історію", ":material/delete_sweep:", key="hist_clear", use_container_width=True):
        history.clear_history(user)
        st.rerun()


# ------------------------------------------------------------------ аналітика масиву
def render_analytics(out: pd.DataFrame, health: dict | None, file_name: str, sig: str) -> None:
    th = threshold_panel(out, health, "an")  # повзунки порогів; повертає чинні (живі) пороги
    out = userprefs.reclassify(out, th)
    df = out.copy()
    df["hour"] = (df["Time"] % 86400) / 3600
    total = len(df)
    counts = df["action"].value_counts()
    flagged = df[df["action"] != "ALLOW"]

    st.markdown("<div class='page-title' style='font-size:1.6rem'>Аналітика масиву даних</div>", unsafe_allow_html=True)
    k = st.columns(5)
    k[0].metric("Транзакцій", fmt_int(total))
    k[1].metric("Заблоковано", fmt_int(counts.get("BLOCK", 0)), f"{counts.get('BLOCK', 0) / total:.2%}", delta_color="off")
    k[2].metric("На перевірку", fmt_int(counts.get("REVIEW", 0)), f"{counts.get('REVIEW', 0) / total:.2%}", delta_color="off")
    k[3].metric("Середній ризик", f"{df['fraud_probability'].mean() * 100:.2f}%")
    k[4].metric("Сума під ризиком", fmt_int(flagged["Amount"].sum()), help="Сума транзакцій зі статусом REVIEW + BLOCK")

    qb = st.session_state.get("quality_batch")
    cmp_view = qb["view"] if qb and qb["sig"] == sig else None  # порівняння моделей, якщо масив уже оцінювали
    pdf_download(f"array_{sig}", "Сформувати PDF-звіт", f"report_{file_name}.pdf",
                 title="Звіт про перевірку масиву транзакційних даних", meta=report_meta(file_name, total, health, th),
                 df=out, thresholds=th, comparison=cmp_view["comparison"] if cmp_view else None,
                 curves=cmp_view["curves"] if cmp_view else None,
                 fraud_rate=cmp_view["fraud_rate"] if cmp_view else None)

    t_over, t_quality, t_top, t_table = st.tabs(["Огляд", "Якість (за міткою Class)", "Найризикованіші", "Усі результати"])

    with t_over:
        c1, c2 = st.columns(2)
        pie = go.Figure(go.Pie(labels=[ACTION_UA[a] for a in counts.index], values=counts.values, hole=0.6,
                               marker_colors=[COLORS[a] for a in counts.index], sort=False))
        pie.update_layout(title="Розподіл рішень")
        c1.plotly_chart(style(pie), use_container_width=True)

        hist = go.Figure(go.Histogram(x=df["fraud_probability"], nbinsx=50, marker_color="#2f6f55"))
        if "review" in th:
            hist.add_vline(x=th["review"], line_dash="dash", line_color=COLORS["REVIEW"], annotation_text="перевірка")
        if "block" in th:
            hist.add_vline(x=th["block"], line_dash="dash", line_color=COLORS["BLOCK"], annotation_text="блок")
        hist.update_yaxes(type="log", title="Кількість (лог. шкала)")
        hist.update_xaxes(title="Імовірність шахрайства", range=[0, 1])
        hist.update_layout(title="Розподіл імовірності шахрайства")
        c2.plotly_chart(style(hist), use_container_width=True)

        c3, c4 = st.columns(2)
        box = go.Figure()
        for action in ("ALLOW", "REVIEW", "BLOCK"):
            sub = df[(df["action"] == action) & (df["Amount"] > 0)]
            if len(sub):
                box.add_trace(go.Box(y=sub["Amount"], name=ACTION_UA[action], marker_color=COLORS[action]))
        box.update_yaxes(type="log", title="Сума (лог. шкала)")
        box.update_layout(title="Суми транзакцій за рішенням", showlegend=False)
        c3.plotly_chart(style(box), use_container_width=True)

        by_hour = pd.crosstab(df["hour"].astype(int), df["action"]).reindex(range(24), fill_value=0)
        bars = go.Figure()
        for action in ("ALLOW", "REVIEW", "BLOCK"):
            if action in by_hour:
                bars.add_trace(go.Bar(x=by_hour.index, y=by_hour[action], name=ACTION_UA[action], marker_color=COLORS[action]))
        bars.update_layout(barmode="stack", title="Транзакції за годиною доби")
        bars.update_xaxes(title="Година", dtick=2)
        bars.update_yaxes(title="Кількість")
        c4.plotly_chart(style(bars), use_container_width=True)

        sample = df.sample(min(len(df), 5000), random_state=1)
        sc = go.Figure()
        for action in ("ALLOW", "REVIEW", "BLOCK"):
            sub = sample[(sample["action"] == action) & (sample["Amount"] > 0)]
            if len(sub):
                sc.add_trace(go.Scattergl(x=sub["Amount"], y=sub["fraud_probability"], mode="markers", name=ACTION_UA[action],
                                          marker=dict(color=COLORS[action], size=6, opacity=0.6)))
        sc.update_xaxes(type="log", title="Сума (лог. шкала)")
        sc.update_yaxes(title="Імовірність шахрайства")
        sc.update_layout(title="Сума vs ймовірність шахрайства (вибірка до 5000 точок)")
        st.plotly_chart(style(sc, 420), use_container_width=True)

    with t_quality:
        if "Class" not in df.columns:
            st.info("У файлі немає колонки Class (0 = легітимна, 1 = шахрайська), тому порівняти прогноз з фактом неможливо. "
                    "Додайте її, щоб побачити ROC, F1 та матрицю помилок.")
        else:
            y = df["Class"].astype(int).to_numpy()
            if y.sum() == 0 or y.sum() == len(y):
                st.info("У колонці Class лише один клас, тому метрики якості порахувати неможливо.")
            else:
                p = df["fraud_probability"].to_numpy()
                fpr, tpr, auc, prec, rec, ap = roc_pr_curves(y, p)
                flag = (df["action"] != "ALLOW").to_numpy()
                m = confusion(y, flag)
                q = st.columns(5)
                q[0].metric("ROC-AUC", f"{auc:.3f}")
                q[1].metric("PR-AUC", f"{ap:.3f}")
                q[2].metric("F1", f"{m['f1']:.3f}", help="Для рішень REVIEW + BLOCK як «позитивних»")
                q[3].metric("Precision", f"{m['precision']:.3f}")
                q[4].metric("Recall", f"{m['recall']:.3f}")
                st.caption("Якщо цей файл збігається з навчальними даними, метрики будуть завищені. "
                           f"Частка шахрайства у файлі: {y.mean():.3%}.")
                st.caption("Усі моделі (Random Forest, LightGBM, XGBoost, гібрид) на цьому масиві: "
                           "сторінка «Моделі» → «Мій масив даних».")
                g1, g2 = st.columns(2)
                g1.plotly_chart(roc_figure([("Гібрид", fpr, tpr, auc, MODEL_COLORS["hybrid"])]), use_container_width=True)
                g2.plotly_chart(pr_figure([("Гібрид", rec, prec, ap, MODEL_COLORS["hybrid"])], baseline=float(y.mean())),
                                use_container_width=True)
                g3, g4 = st.columns(2)
                g3.plotly_chart(f1_figure(f1_sweep(y, p), th.get("review"), th.get("block")), use_container_width=True)
                g4.plotly_chart(cm_figure(m, "Матриця помилок (REVIEW + BLOCK)"), use_container_width=True)

    with t_top:
        n = st.slider("Скільки показати", 10, 200, 50, step=10)
        cols = [c for c in ("Time", "Amount", "fraud_probability", "anomaly_score", "action", "Class") if c in df.columns]
        top = df.sort_values("fraud_probability", ascending=False).head(n)[cols]
        st.dataframe(top, hide_index=True, use_container_width=True,
                     column_config={"fraud_probability": st.column_config.ProgressColumn(
                         "fraud_probability", min_value=0.0, max_value=1.0, format="%.3f")})
        st.download_button("Завантажити REVIEW + BLOCK (CSV)", flagged.to_csv(index=False).encode("utf-8"),
                           "flagged_transactions.csv", "text/csv")

    with t_table:
        st.dataframe(out.sort_values("fraud_probability", ascending=False), hide_index=True, use_container_width=True)
        st.download_button("Завантажити результати (CSV)", out.to_csv(index=False).encode("utf-8"),
                           f"scored_{file_name}", "text/csv")


def page_check(health: dict | None) -> None:
    page_head("Контроль транзакцій", "Перевірка ризику",
              "Оцініть одну транзакцію або перевірте масив транзакційних даних.")
    left, right = st.columns([2.1, 1], gap="large")
    with right:
        state_card(health)
        howto_card()
    array_df = None
    with left:
        with st.container(border=True):
            h1, h2 = st.columns([1.2, 1.5])
            with h1:
                card_head("01", "Дані для перевірки", "Введіть параметри або оберіть масив")
            mode = h2.radio("Режим", [MODE_SINGLE, MODE_ARRAY], horizontal=True, label_visibility="collapsed", key="mode")
            if mode == MODE_SINGLE:
                single_mode(health)
            else:
                array_df, _ = batch_mode(health)
    if mode == MODE_SINGLE:
        result_section(effective_th(health))
    if mode == MODE_ARRAY and array_df is not None:
        pick_transaction_section(array_df, effective_th(health))
    batch = st.session_state.get("batch")
    if mode == MODE_ARRAY and batch and array_df is not None and batch["sig"] == df_signature(array_df):
        st.divider()
        render_analytics(batch["df"], health, batch["name"], batch["sig"])


# ------------------------------------------------------------------ сторінка: моделі
def comparison_table(comparison: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Модель класифікації": r["name"], "Precision": r["precision"], "Recall": r["recall"],
          "F1-Score": r["f1"], "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"]} for r in comparison]
    )


def view_from_metrics(metrics: dict) -> dict:
    return {
        "comparison": metrics["comparison"], "curves": metrics.get("curves") or {}, "cms": None,
        "fraud_rate": metrics["fraud_rate"],
        "caption": f"Тест: {fmt_int(metrics['test_size'])} транзакцій, з них шахрайських "
                   f"{fmt_int(metrics['test_frauds'])} (частка у всьому наборі {metrics['fraud_rate']:.3%}).",
    }


def build_batch_view(y: np.ndarray, probs: dict[str, np.ndarray], thresholds: dict[str, float]) -> dict:
    """Метрики та криві всіх моделей на масиві (пороги беруться з навчання)."""
    comparison, curves, cms = [], {}, {}
    for key in MODEL_ORDER:
        if key not in probs:
            continue
        p = probs[key]
        fpr, tpr, auc, prec, rec, ap = roc_pr_curves(y, p)
        f, t = thin_pair(fpr, tpr)
        pr, rc = thin_pair(prec, rec)
        curves[key] = {"roc_auc": auc, "pr_auc": ap, "roc": {"fpr": f, "tpr": t},
                       "pr": {"precision": pr, "recall": rc}}
        if key in thresholds:
            thr = thresholds[key]
            m = confusion(y, p >= thr)
            cms[key] = m
            curves[key]["f1_by_threshold"] = f1_sweep(y, p)
            comparison.append({"key": key, "name": MODEL_UA[key], "threshold": thr, "precision": m["precision"],
                               "recall": m["recall"], "f1": m["f1"], "roc_auc": auc, "pr_auc": ap})
    n_pos = int(y.sum())
    return {
        "comparison": comparison, "curves": curves, "cms": cms, "fraud_rate": float(y.mean()), "n": len(y),
        "caption": f"Масив: {fmt_int(len(y))} транзакцій, з них шахрайських {fmt_int(n_pos)} "
                   f"(частка {y.mean():.3%}). Пороги взято з навчання (максимум F1 на валідації).",
    }


def evaluate_on_batch(data: pd.DataFrame, y: np.ndarray) -> dict:
    payload = data[INPUT_COLS]
    n, probs, thresholds = len(payload), {}, {}
    bar = st.progress(0.0, text="Оцінюємо всі моделі…")
    for start in range(0, n, 1000):
        chunk = payload.iloc[start : start + 1000].to_dict("records")
        r = requests.post(f"{API_URL}/predict/compare", json={"transactions": chunk}, headers=HEADERS, timeout=120)
        r.raise_for_status()
        js = r.json()
        thresholds = js["thresholds"]
        for key, values in js["probabilities"].items():
            probs.setdefault(key, []).extend(values)
        bar.progress(min((start + 1000) / n, 1.0), text=f"Оброблено {min(start + 1000, n)} з {n}")
    bar.empty()
    return build_batch_view(y, {k: np.asarray(v) for k, v in probs.items()}, thresholds)


def get_labels(data: pd.DataFrame) -> np.ndarray | None:
    if "Class" not in data.columns:
        st.error("У файлі немає колонки Class (0 = легітимна, 1 = шахрайська): без неї метрики порахувати неможливо.")
        return None
    y = pd.to_numeric(data["Class"], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        st.error("Колонка Class має містити лише значення 0 та 1.")
        return None
    y = y.astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        st.error("У колонці Class лише один клас, тому метрики якості порахувати неможливо.")
        return None
    return y


def batch_quality_view() -> dict | None:
    st.caption("Потрібен масив із колонкою Class (0/1). Усі моделі оцінять його, пороги беруться з навчання.")
    data, name = array_source("quality_main")
    if data is None:
        return None
    y = get_labels(data)
    if y is None:
        return None
    sig = df_signature(data)
    st.caption(f"«{name}»: {fmt_int(len(y))} транзакцій, шахрайських {fmt_int(y.sum())} ({y.mean():.3%}).")
    if st.button("Оцінити моделі на масиві", type="primary", key="run_quality"):
        try:
            view = evaluate_on_batch(data, y)
            view.update(name=name, sig=sig)
            st.session_state["quality_batch"] = {"sig": sig, "view": view}
        except requests.RequestException as exc:
            st.error(f"Помилка API: {exc}. Перезапустіть API (потрібен ендпоінт /predict/compare) "
                     "і перенавчіть моделі: python -m src.train.")
    saved = st.session_state.get("quality_batch")
    return saved["view"] if saved and saved["sig"] == sig else None


def render_quality(view: dict, metrics: dict, th: dict, is_test: bool, pdf_key: str, meta: dict) -> None:
    comparison, curves = view["comparison"], view["curves"]
    main = next((r for r in comparison if r["key"] == "hybrid"), comparison[-1])
    st.markdown(f"**{main['name']}**")
    k = st.columns(5)
    for col, (label, key) in zip(k, [("Precision", "precision"), ("Recall", "recall"), ("F1-Score", "f1"),
                                     ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc")]):
        col.metric(label, f"{main[key]:.3f}")
    st.caption(view["caption"])
    pdf_download(pdf_key, "Сформувати PDF-звіт", "models_report.pdf", title="Звіт про якість моделей", meta=meta,
                 comparison=comparison, curves=curves, fraud_rate=view["fraud_rate"])

    names = ["Порівняння моделей", "ROC", "Precision–Recall", "F1 і поріг", "Матриці помилок"]
    has_imp = is_test and bool(metrics.get("feature_importance"))
    if has_imp:
        names.append("Ознаки")
    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Порівняння моделей"]:
        table = comparison_table(comparison)
        cols = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
        styled = (table.style.format({c: "{:.3f}" for c in cols})
                  .highlight_max(subset=cols, axis=0, props="font-weight:700;background-color:#e3ede8;"))
        st.markdown("**Порівняльний аналіз точності алгоритмів класифікації шахрайських транзакцій**")
        st.dataframe(styled, hide_index=True, use_container_width=True)
        st.caption("Найкраще значення в кожному стовпці виділено. Precision, Recall і F1 пораховані при порозі, "
                   "який максимізує F1 на валідації (окремо для кожної моделі). ROC-AUC та PR-AUC від порога не залежать.")
        fig = go.Figure()
        for r in comparison:
            fig.add_trace(go.Bar(x=cols, y=[r["precision"], r["recall"], r["f1"], r["roc_auc"], r["pr_auc"]],
                                 name=r["name"], marker_color=MODEL_COLORS.get(r["key"])))
        fig.update_yaxes(range=[0, 1.05], title="Значення метрики")
        fig.update_layout(barmode="group", title="Порівняння моделей за метриками")
        st.plotly_chart(style(fig, 420), use_container_width=True)
        st.download_button("Завантажити таблицю (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                           "model_comparison.csv" if is_test else "model_comparison_batch.csv", "text/csv")

    with tabs["ROC"]:
        items = [(MODEL_UA[k], curves[k]["roc"]["fpr"], curves[k]["roc"]["tpr"], curves[k]["roc_auc"], MODEL_COLORS[k])
                 for k in MODEL_ORDER if k in curves]
        if items:
            st.plotly_chart(roc_figure(items), use_container_width=True)
            st.caption("Isolation Forest оцінюється за -score (чим аномальніше, тим вищий ризик).")
        else:
            st.info("Немає кривих у metrics.json. Перенавчіть моделі.")

    with tabs["Precision–Recall"]:
        items = [(MODEL_UA[k], curves[k]["pr"]["recall"], curves[k]["pr"]["precision"], curves[k]["pr_auc"], MODEL_COLORS[k])
                 for k in MODEL_ORDER if k in curves]
        if items:
            st.plotly_chart(pr_figure(items, baseline=view["fraud_rate"]), use_container_width=True)
            st.caption("За сильного дисбалансу класів PR-крива інформативніша за ROC.")
        else:
            st.info("Немає кривих у metrics.json. Перенавчіть моделі.")

    with tabs["F1 і поріг"]:
        options = [k for k in MODEL_ORDER if k in curves and "f1_by_threshold" in curves[k]]
        if options:
            key = st.selectbox("Модель", options, index=options.index("hybrid") if "hybrid" in options else 0,
                               format_func=lambda k: MODEL_UA[k], key=f"f1_model_{is_test}")
            st.plotly_chart(f1_figure(curves[key]["f1_by_threshold"], th.get("review"), th.get("block")),
                            use_container_width=True)
        else:
            st.info("Немає кривих у metrics.json. Перенавчіть моделі.")

    with tabs["Матриці помилок"]:
        if is_test:
            c1, c2 = st.columns(2)
            c1.plotly_chart(cm_figure(metrics["at_review_threshold"], "Поріг перевірки (REVIEW)"), use_container_width=True)
            c2.plotly_chart(cm_figure(metrics["at_block_threshold"], "Поріг блокування (BLOCK)"), use_container_width=True)
            st.caption("Матриці для робочих порогів гібридної моделі (REVIEW враховує й аномальність).")
        else:
            for i in range(0, len(comparison), 2):
                for col, r in zip(st.columns(2), comparison[i : i + 2]):
                    col.plotly_chart(cm_figure(view["cms"][r["key"]], f"{SHORT_UA[r['key']]} (поріг {r['threshold']:.2f})"),
                                     use_container_width=True)

    if has_imp:
        with tabs["Ознаки"]:
            d = pd.DataFrame(metrics["feature_importance"]).sort_values("importance")
            fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h", marker_color="#2f6f55"))
            fig.update_layout(title="Важливість ознак XGBoost (топ-15)")
            st.plotly_chart(style(fig, 460), use_container_width=True)


def page_quality(health: dict | None) -> None:
    page_head("Аналіз моделей", "Моделі",
              "Precision, Recall, F1-Score, ROC-AUC та PR-AUC: на тестовій вибірці навчання і на ваших масивах даних.")
    metrics = get_metrics()
    if not metrics:
        st.error("Не вдалося отримати метрики з API (/metrics). Перевірте, що API запущено і моделі навчено.")
        return
    th = health["thresholds"] if health else {}
    source = st.radio("Джерело оцінки", ["Тестова вибірка (з навчання)", "Мій масив даних"],
                      horizontal=True, key="quality_source")
    is_test = source.startswith("Тестова")
    base = {"Користувач": st.session_state["user"]["username"],
            "Моделі навчено": fmt_date(health.get("trained_at")) if health else "—"}
    if is_test:
        if not metrics.get("comparison"):
            st.info("У metrics.json немає порівняльної таблиці. Перенавчіть моделі: `python -m src.train`.")
            return
        meta = {"Джерело оцінки": "Тестова вибірка навчання", "Транзакцій у тесті": fmt_int(metrics["test_size"]),
                "Шахрайських у тесті": fmt_int(metrics["test_frauds"]), **base}
        render_quality(view_from_metrics(metrics), metrics, th, True, "models_test", meta)
        return
    view = batch_quality_view()
    if view is None:
        return
    if not view["comparison"]:
        st.error("API не повернув порогів моделей. Перенавчіть моделі (`python -m src.train`) і перезапустіть API.")
        return
    meta = {"Джерело оцінки": f"Масив даних «{view['name']}»", "Транзакцій": fmt_int(view["n"]), **base}
    render_quality(view, metrics, th, False, f"models_{view['sig']}", meta)


# ------------------------------------------------------------------ сторінка: користувачі (адмін)
def page_users(current: dict) -> None:
    page_head("Адміністрування", "Користувачі", "Керування доступом до дашборда.")
    if auth.uses_default_password(current["username"]):
        st.warning("Ви використовуєте типовий пароль. Змініть його на вкладці «Змінити пароль».")
    t_add, t_pw, t_del = st.tabs(["Додати", "Змінити пароль", "Видалити"])
    names = list(auth.load_users())

    with t_add:
        with st.form("add_user", clear_on_submit=True):
            u = st.text_input("Логін")
            p = st.text_input("Пароль", type="password")
            role = st.selectbox("Роль", auth.ROLES, format_func=lambda r: ROLE_UA[r])
            if st.form_submit_button("Створити", type="primary"):
                try:
                    auth.add_user(u, p, role)
                    st.success(f"Користувача «{u.strip().lower()}» створено")
                except ValueError as exc:
                    st.error(str(exc))
    with t_pw:
        with st.form("set_pw", clear_on_submit=True):
            u = st.selectbox("Користувач", names)
            p = st.text_input("Новий пароль", type="password")
            if st.form_submit_button("Змінити", type="primary"):
                try:
                    auth.set_password(u, p)
                    st.success("Пароль змінено")
                except ValueError as exc:
                    st.error(str(exc))
    with t_del:
        with st.form("del_user"):
            u = st.selectbox("Користувач", [n for n in names if n != current["username"]] or ["—"])
            if st.form_submit_button("Видалити"):
                try:
                    auth.delete_user(u)
                    st.success(f"Користувача «{u}» видалено")
                except ValueError as exc:
                    st.error(str(exc))

    users = auth.load_users()
    st.markdown("<div class='eyebrow'>Усі користувачі</div>", unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame([{"Логін": n, "Роль": ROLE_UA[d["role"]], "Створено": d.get("created_at", "")[:10]}
                      for n, d in users.items()]),
        hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ вхід у систему
def _login_form() -> None:
    locked = st.session_state.get("locked_until", 0) - time.time()
    with st.form("login"):
        username = st.text_input("Логін")
        password = st.text_input("Пароль", type="password")
        submit = st.form_submit_button("Увійти", type="primary", use_container_width=True)
    if not submit:
        return
    if locked > 0:
        st.error(f"Забагато спроб. Спробуйте через {int(locked) + 1} с.")
        return
    user = auth.authenticate(username, password)
    if user:
        st.session_state["user"] = {**user, "login_at": time.time()}
        st.session_state["failed"] = 0
        st.rerun()
    failed = st.session_state.get("failed", 0) + 1
    st.session_state["failed"] = failed
    if failed >= MAX_LOGIN_ATTEMPTS:
        st.session_state["locked_until"] = time.time() + LOCKOUT_SECONDS
        st.session_state["failed"] = 0
        st.error(f"Забагато невдалих спроб. Вхід заблоковано на {LOCKOUT_SECONDS} с.")
    else:
        st.error("Невірний логін або пароль.")


def _signup_form() -> None:
    with st.form("signup"):
        username = st.text_input("Логін", help="3–32 символи: латиниця, цифри, _ . -")
        p1 = st.text_input("Пароль", type="password", help=f"Щонайменше {auth.MIN_PASSWORD_LEN} символів")
        p2 = st.text_input("Повторіть пароль", type="password")
        submit = st.form_submit_button("Створити акаунт", type="primary", use_container_width=True)
    if not submit:
        return
    if p1 != p2:
        st.error("Паролі не збігаються.")
        return
    try:
        auth.add_user(username, p1, "analyst")
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.success("Акаунт створено (роль «Аналітик»). Перейдіть на вкладку «Вхід».")


def login_page() -> None:
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']{display:none;}</style>",
                unsafe_allow_html=True)
    auth.ensure_default_admin()
    _, mid, _ = st.columns([0.4, 3, 0.4])
    with mid:
        st.markdown('<div style="height:4vh"></div>', unsafe_allow_html=True)
        hero, form = st.columns(2, gap="large")
        with hero:
            st.markdown(
                '<div class="login-hero"><div class="logo">🛡️</div><h1>Fraud Monitor</h1></div>',
                unsafe_allow_html=True,
            )
        with form:
            st.markdown('<div class="page-title" style="font-size:1.8rem;margin-top:1.5rem">Вхід у систему</div>'
                        '<div class="page-sub">Увійдіть, щоб працювати з моніторингом ризику.</div>',
                        unsafe_allow_html=True)
            if st.session_state.pop("expired", False):
                st.info("Сеанс завершено через тривалу неактивність. Увійдіть знову.")
            if ALLOW_SIGNUP:
                t_in, t_up = st.tabs(["Вхід", "Реєстрація"])
                with t_in:
                    _login_form()
                with t_up:
                    _signup_form()
            else:
                _login_form()


# ------------------------------------------------------------------ точка входу
init_state()
st.markdown(CSS, unsafe_allow_html=True)

user = st.session_state.get("user")
if user and time.time() - user["login_at"] > SESSION_HOURS * 3600:
    st.session_state.clear()
    st.session_state["expired"] = True
    user = None
if not user:
    login_page()
    st.stop()

health = api_health()
nav = NAV + (["Користувачі"] if user["role"] == "admin" else [])

if st.session_state.get("page") not in nav:
    st.session_state["page"] = nav[0]


def go_to(target: str) -> None:
    st.session_state["page"] = target


with st.sidebar:
    brand()
    for i, item in enumerate(nav):
        icon_button(item, NAV_ICONS[item], key=f"nav_{i}", use_container_width=True, on_click=go_to, args=(item,),
                    type="primary" if st.session_state["page"] == item else "secondary")
    st.markdown(f'<div class="side-user">Ви увійшли як<b>{user["username"]}</b>{ROLE_UA[user["role"]]}</div>',
                unsafe_allow_html=True)
    if icon_button("Вийти", ":material/logout:", key="logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.markdown('<div style="height:24vh"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-status">Статус системи<b>{"Сервіс готовий" if health else "API недоступний"}</b></div>',
        unsafe_allow_html=True,
    )

page = st.session_state["page"]
topbar(page, health, user)
if page == NAV[0]:
    page_check(health)
elif page == NAV[1]:
    page_quality(health)
elif page == NAV[2]:
    page_history()
elif page == NAV[3]:
    page_settings(health)
else:
    page_users(user)