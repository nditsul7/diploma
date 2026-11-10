"""PDF-звіт (reportlab + matplotlib): підсумок перевірки масиву даних і якість моделей.

Кирилицю забезпечує шрифт DejaVu Sans, який постачається разом із matplotlib.
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

try:
    from app import evalutils
except ImportError:  # запуск через streamlit run app/dashboard.py
    import evalutils  # type: ignore[no-redef]

GREEN, INK, MUTED, LINE = "#2f6f55", "#1c2b27", "#76827e", "#d7ddda"
ACTIONS = ("ALLOW", "REVIEW", "BLOCK")
ACTION_UA = {"ALLOW": "Дозволити", "REVIEW": "Перевірити", "BLOCK": "Заблокувати"}
ACTION_COLOR = {"ALLOW": "#2f7d5b", "REVIEW": "#d98a1f", "BLOCK": "#c8453b"}
MODEL_ORDER = ["random_forest", "lightgbm", "xgboost_without_anomaly_score", "hybrid", "isolation_forest_only"]
MODEL_NAME = {
    "random_forest": "Random Forest", "lightgbm": "LightGBM",
    "xgboost_without_anomaly_score": "XGBoost (без Isolation Forest)",
    "hybrid": "Гібрид (Isolation Forest + XGBoost)", "isolation_forest_only": "Isolation Forest",
}
MODEL_COLOR = {
    "random_forest": "#8a8f8c", "lightgbm": "#7b5ea7", "xgboost_without_anomaly_score": "#3b6ea5",
    "hybrid": "#2f6f55", "isolation_forest_only": "#d98a1f",
}
PAGE_W = A4[0] - 36 * mm  # ширина робочої області


def _register_fonts() -> None:
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts / file)))


def _num(x: float, digits: int = 0) -> str:
    return f"{x:,.{digits}f}".replace(",", " ")


def _style(name: str, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, fontName=kw.pop("fontName", "DejaVu"), textColor=colors.HexColor(kw.pop("color", INK)), **kw)


def _image(fig: Figure, width: float = PAGE_W) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    buf.seek(0)
    iw, ih = ImageReader(buf).getSize()
    buf.seek(0)
    return Image(buf, width=width, height=width * ih / iw)


# ------------------------------------------------------------------ графіки
def _clean(ax, title: str) -> None:
    ax.set_title(title, fontsize=9, color=INK)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _fig_overview(df: pd.DataFrame, th: dict) -> Figure:
    fig = Figure(figsize=(9, 3.1))
    a1, a2 = fig.subplots(1, 2)
    counts = df["action"].value_counts().reindex(ACTIONS).fillna(0)
    a1.bar([ACTION_UA[a] for a in ACTIONS], counts.values, color=[ACTION_COLOR[a] for a in ACTIONS])
    for i, v in enumerate(counts.values):
        a1.text(i, v, _num(v), ha="center", va="bottom", fontsize=7)
    _clean(a1, "Розподіл рішень")
    a2.hist(df["fraud_probability"], bins=40, color=GREEN)
    a2.set_yscale("log")
    a2.set_xlabel("Імовірність шахрайства", fontsize=7)
    for key, color in (("review", ACTION_COLOR["REVIEW"]), ("block", ACTION_COLOR["BLOCK"])):
        if key in th:
            a2.axvline(th[key], color=color, ls="--", lw=1)
    _clean(a2, "Розподіл імовірності (лог. шкала)")
    fig.tight_layout()
    return fig


def _fig_hours(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(9, 2.6))
    ax = fig.subplots()
    hours = ((df["Time"] % 86400) / 3600).astype(int)
    table = pd.crosstab(hours, df["action"]).reindex(range(24), fill_value=0)
    bottom = np.zeros(24)
    for a in ACTIONS:
        if a in table:
            ax.bar(table.index, table[a], bottom=bottom, color=ACTION_COLOR[a], label=ACTION_UA[a])
            bottom += table[a].to_numpy()
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Година доби", fontsize=7)
    ax.legend(fontsize=7, frameon=False)
    _clean(ax, "Транзакції за годиною доби")
    fig.tight_layout()
    return fig


def _confusion_axes(ax, m: dict, title: str) -> None:
    z = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    shade = np.log1p(z)
    ax.imshow(shade, cmap="Greens", vmin=0)
    for (i, j), v in np.ndenumerate(z):
        ax.text(j, i, _num(v), ha="center", va="center", fontsize=9,
                color="white" if shade[i, j] > shade.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["прогноз: 0", "прогноз: 1"], fontsize=7)
    ax.set_yticks([0, 1], ["факт: 0", "факт: 1"], fontsize=7)
    ax.set_title(title, fontsize=9, color=INK)


def _fig_quality(y: np.ndarray, p: np.ndarray, m: dict) -> Figure:
    fpr, tpr, auc, prec, rec, ap = evalutils.roc_pr_curves(y, p)
    fpr, tpr = evalutils.thin_pair(fpr, tpr)
    rec, prec = evalutils.thin_pair(rec, prec)
    fig = Figure(figsize=(9, 3.0))
    a1, a2, a3 = fig.subplots(1, 3)
    a1.plot([0, 1], [0, 1], "--", color="#b5beba", lw=1)
    a1.plot(fpr, tpr, color=GREEN, lw=2)
    a1.set_xlabel("FPR", fontsize=7)
    a1.set_ylabel("TPR", fontsize=7)
    _clean(a1, f"ROC (AUC {auc:.3f})")
    a2.plot(rec, prec, color=GREEN, lw=2)
    a2.set_xlabel("Recall", fontsize=7)
    a2.set_ylabel("Precision", fontsize=7)
    a2.set_ylim(0, 1.02)
    _clean(a2, f"Precision–Recall (AP {ap:.3f})")
    _confusion_axes(a3, m, "Матриця помилок")
    fig.tight_layout()
    return fig


def _fig_models(curves: dict, fraud_rate: float | None) -> Figure:
    fig = Figure(figsize=(9, 3.7))
    a1, a2 = fig.subplots(1, 2)
    a1.plot([0, 1], [0, 1], "--", color="#b5beba", lw=1)
    for key in MODEL_ORDER:
        c = curves.get(key)
        if not c:
            continue
        a1.plot(c["roc"]["fpr"], c["roc"]["tpr"], color=MODEL_COLOR[key], lw=1.8,
                label=f"{MODEL_NAME[key]} ({c['roc_auc']:.3f})")
        a2.plot(c["pr"]["recall"], c["pr"]["precision"], color=MODEL_COLOR[key], lw=1.8)
    if fraud_rate:
        a2.axhline(fraud_rate, color="#b5beba", ls="--", lw=1)
    a1.set_xlabel("FPR", fontsize=7)
    a1.set_ylabel("TPR", fontsize=7)
    a2.set_xlabel("Recall", fontsize=7)
    a2.set_ylabel("Precision", fontsize=7)
    a2.set_ylim(0, 1.02)
    a1.legend(fontsize=6, loc="lower right", frameon=False)
    _clean(a1, "ROC-криві")
    _clean(a2, "Precision–Recall криві")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ таблиці
def _table(rows: list[list], widths: list[float] | None = None, *, header: bool = True, right_from: int = 1,
           highlight: list[tuple[int, int]] | None = None) -> Table:
    cell = _style("cell", fontSize=8, leading=10)
    data = [[Paragraph(str(c), cell) if (j == 0 and not (header and i == 0)) else c for j, c in enumerate(r)]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (right_from, 0), (-1, -1), "RIGHT"),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(GREEN)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                 ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f5")])]
    for col, row in highlight or []:
        cmds += [("BACKGROUND", (col, row), (col, row), colors.HexColor("#e3ede8")),
                 ("FONTNAME", (col, row), (col, row), "DejaVu-Bold")]
    t.setStyle(TableStyle(cmds))
    return t


def _meta_table(meta: dict[str, str]) -> Table:
    key = _style("k", fontSize=8, color=MUTED)
    val = _style("v", fontSize=9)
    t = Table([[Paragraph(k, key), Paragraph(str(v), val)] for k, v in meta.items()], colWidths=[PAGE_W * 0.32, PAGE_W * 0.68])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(18 * mm, 10 * mm, "Захист · Fraud Monitor")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Сторінка {doc.page}")
    canvas.restoreState()


# ------------------------------------------------------------------ звіт
def build_report(*, title: str, meta: dict[str, str], df: pd.DataFrame | None = None, thresholds: dict | None = None,
                 comparison: list[dict] | None = None, curves: dict | None = None,
                 fraud_rate: float | None = None) -> bytes:
    """Формує PDF і повертає його байти.

    df: результати перевірки (колонки Time, Amount, fraud_probability, anomaly_score, action, за наявності Class).
    comparison/curves: порівняння моделей (таблиця 3.2) та їхні ROC/PR-криві.
    """
    _register_fonts()
    th = thresholds or {}
    h1 = _style("h1", fontName="DejaVu-Bold", fontSize=20, leading=24, color=GREEN)
    h2 = _style("h2", fontName="DejaVu-Bold", fontSize=12, leading=16, spaceBefore=12, spaceAfter=6)
    muted = _style("muted", fontSize=8, color=MUTED, leading=11)
    body = _style("body", fontSize=9, leading=13)

    story: list = [Paragraph(title, h1), Spacer(1, 2 * mm),
                   Paragraph(f"Сформовано {datetime.now():%d.%m.%Y %H:%M}", muted), Spacer(1, 4 * mm), _meta_table(meta)]
    section = 0

    def heading(text: str) -> Paragraph:
        nonlocal section
        section += 1
        return Paragraph(f"{section}. {text}", h2)

    if df is not None and len(df):
        total = len(df)
        counts = df["action"].value_counts()
        rows = [["Рішення", "Кількість", "Частка", "Сума транзакцій"]]
        for a in ACTIONS:
            n = int(counts.get(a, 0))
            rows.append([ACTION_UA[a], _num(n), f"{n / total:.2%}", _num(df.loc[df["action"] == a, "Amount"].sum())])
        flagged_amount = df.loc[df["action"] != "ALLOW", "Amount"].sum()
        story += [heading("Підсумок рішень"),
                  _table(rows, [PAGE_W * 0.34, PAGE_W * 0.22, PAGE_W * 0.18, PAGE_W * 0.26]), Spacer(1, 3 * mm),
                  Paragraph(f"Середня ймовірність шахрайства: <b>{df['fraud_probability'].mean():.2%}</b>. "
                            f"Сума під ризиком (перевірка + блокування): <b>{_num(flagged_amount)}</b>.", body),
                  Spacer(1, 3 * mm), _image(_fig_overview(df, th)), _image(_fig_hours(df))]

        top = df.sort_values("fraud_probability", ascending=False).head(20)
        has_class = "Class" in top.columns
        head = ["№", "Time", "Amount", "Ймовірність", "Аномальність", "Рішення"] + (["Class"] if has_class else [])
        trows = [head]
        for i, r in enumerate(top.itertuples(index=False), 1):
            trows.append([str(i), _num(r.Time), _num(r.Amount, 2), f"{r.fraud_probability:.3f}", f"{r.anomaly_score:.3f}",
                          ACTION_UA[r.action]] + ([str(int(r.Class))] if has_class and pd.notna(r.Class) else []))
        story += [heading("Найризикованіші транзакції (топ-20)"), _table(trows, right_from=1)]

        labels = pd.to_numeric(df["Class"], errors="coerce") if "Class" in df.columns else None
        if labels is not None and labels.isin([0, 1]).all() and 0 < labels.sum() < len(labels):
            y = labels.astype(int).to_numpy()
            p = df["fraud_probability"].to_numpy()
            m = evalutils.confusion(y, (df["action"] != "ALLOW").to_numpy())
            _, _, auc, _, _, ap = evalutils.roc_pr_curves(y, p)
            story += [KeepTogether([
                heading("Якість гібридної моделі на цьому масиві"),
                Paragraph("Позитивними вважаються рішення «Перевірити» та «Заблокувати»; "
                          f"частка шахрайства у масиві {y.mean():.3%}.", body), Spacer(1, 2 * mm),
                _table([["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
                        [f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", f"{auc:.3f}", f"{ap:.3f}"]],
                       right_from=0),
                Spacer(1, 3 * mm), _image(_fig_quality(y, p, m))])]

    if comparison:
        cols = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
        rows = [["Модель класифікації", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]]
        for r in comparison:
            rows.append([r["name"]] + [f"{r[c]:.3f}" for c in cols])
        best = [(j + 1, 1 + int(np.argmax([r[c] for r in comparison]))) for j, c in enumerate(cols)]
        block = [heading("Порівняння моделей"),
                 Paragraph("Precision, Recall і F1 розраховано при порозі, що максимізує F1 на валідації "
                           "(окремо для кожної моделі); ROC-AUC та PR-AUC від порога не залежать. "
                           "Найкраще значення в кожному стовпці виділено.", body), Spacer(1, 2 * mm),
                 _table(rows, [PAGE_W * 0.40] + [PAGE_W * 0.12] * 5, highlight=best)]
        story.append(KeepTogether(block))
        if curves:
            story += [Spacer(1, 3 * mm), _image(_fig_models(curves, fraud_rate))]

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Звіт сформовано автоматично системою Fraud Monitor.", muted))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                      title=title, author="Захист · Fraud Monitor").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()