"""Собирает report/cutoff.ipynb — подбор порога отбора (отсечки) для выбранной модели.

    python report/build_cutoff.py                    # пересобрать ноутбук после правки этого файла
    python report/run_report.py --notebook cutoff    # выполнить и получить report/out/cutoff.html

Ноутбук читает оценки моделей по договорам, которые сохраняет отчёт (out/<имя>_scores.csv),
поэтому считается за секунды: порог можно менять и перезапускать сколько угодно раз.
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
> **Настройки — в следующей ячейке.** Меняйте `CUTOFF` и перезапускайте ноутбук: он считается за
> секунды. Те же переменные можно задать в файле `cutoff_local.py` рядом с ноутбуком.
""")
code("""
# ======================= НАСТРОЙКИ =======================
SCORES_CSV = "out/rtk_report_scores.csv"   # оценки моделей по договорам — файл сохраняет отчёт
MODEL = None             # имя модели из отчёта; None — первая (основная)
CUTOFF = None            # порог оценки: включаемся, если оценка >= CUTOFF; None — рекомендованный
SELECT_ON = ["val"]      # на каких выборках подбирается рекомендованный порог (test — только для проверки)
SAMPLES = None           # None — выборки как в отчёте; или по датам включения в РТК (включительно):
                         # {"train": ("2023-01-01", "2023-09-30"), "val": ("2023-10-01", "2023-12-31"),
                         #  "test": ("2024-01-01", "2024-12-31")}
PRODUCTS = None          # список продуктов; None — все, что есть в файле
EXTRA_COST = 0           # ₽ на договор сверх пошлины (подготовка и подача заявления)
CAUTION = 1.0            # осторожность правила: 0 — чистый максимум; 1 — самый строгий порог, результат
                         # которого статистически не отличим от максимума (рекомендуется)
N_BOOT = 500             # число повторов бутстрепа

import os
from pathlib import Path

if Path("cutoff_local.py").exists():
    exec(Path("cutoff_local.py").read_text(encoding="utf-8"))
    print("настройки дополнены из cutoff_local.py")
""")

code("""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import HTML, Markdown, display
from plotly.subplots import make_subplots

pio.renderers.default = "notebook"

INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e9e8e4", "#c9c8c2"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
FONT = "Segoe UI, Arial, sans-serif"
SAMPLE_RU = {"train": "обучение", "val": "валидация", "test": "тест"}
SAMPLE_COLOR = {"train": AQUA, "val": ORANGE, "test": BLUE}


def num(x):
    return f"{x:,.0f}".replace(",", " ").replace("-", "−")


def dec(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def pct(x, d=1):
    return dec(x * 100, d) + " %"


def signed(x, d=1):
    return ("+" if x > 0 else "−" if x < 0 else "") + dec(abs(x) / 1e6, d) + " млн ₽"


def sc(t):
    return "—" if not np.isfinite(t) else dec(t, 4)


def layout(fig, title, subtitle=None, height=420, top=84, **kw):
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:13px;color:{INK2}'>{subtitle}</span>"
    fig.update_layout(
        template="plotly_white", height=height, font={"family": FONT, "size": 13, "color": INK2},
        title={"text": text, "x": 0, "xanchor": "left", "font": {"size": 17, "color": INK}},
        margin={"l": 64, "r": 36, "t": top, "b": 56}, plot_bgcolor="white", paper_bgcolor="white",
        separators=", ", hoverlabel={"bgcolor": "white", "font": {"family": FONT, "color": INK}}, **kw)
    fig.update_xaxes(showgrid=False, linecolor=MUTED, ticks="outside", tickcolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show_tiles(tiles):
    cards = "".join(
        f'<div style="flex:1 1 200px;border:1px solid {GRID};border-radius:10px;padding:16px 18px;background:#fff">'
        f'<div style="font-size:30px;font-weight:650;color:{INK};line-height:1.1">{v}</div>'
        f'<div style="font-size:14px;color:{INK};margin-top:6px">{label}</div>'
        f'<div style="font-size:12.5px;color:{INK2};margin-top:4px">{note}</div></div>'
        for v, label, note in tiles)
    display(HTML(f'<div style="display:flex;gap:14px;flex-wrap:wrap;font-family:{FONT};margin:8px 0 4px">{cards}</div>'))


# ---------- данные
path = Path(SCORES_CSV)
if not path.exists():
    raise FileNotFoundError(f"Нет файла {SCORES_CSV}: сначала выполните отчёт (python report/run_report.py) — "
                            "он сохраняет оценки моделей по договорам")
df = pd.read_csv(path)
score_cols = {c[len("score__"):]: c for c in df.columns if c.startswith("score__")}
MODEL = MODEL or next(iter(score_cols))
if MODEL not in score_cols:
    raise ValueError(f"Модели {MODEL!r} нет в файле; есть: {', '.join(score_cols)}")
df["score"] = df[score_cols[MODEL]]
df["rtk_send_date"] = pd.to_datetime(df["rtk_send_date"])
if PRODUCTS:
    df = df[df["product"].isin(PRODUCTS)]
if SAMPLES:
    df["sample"] = None
    for name, (a, b) in SAMPLES.items():
        df.loc[df["rtk_send_date"].between(pd.Timestamp(a), pd.Timestamp(b)), "sample"] = name
    df = df[df["sample"].notna()]
df = df.reset_index(drop=True)
df["cost"] = df["duty"] + EXTRA_COST
df["profit"] = df["money"] - df["cost"]
df["payoff"] = (df["profit"] > 0).astype(int)
NAMES = [s for s in ("train", "val", "test") if (df["sample"] == s).any()]
NAMES += [s for s in df["sample"].unique() if s not in NAMES]
PARTS = {s: df[df["sample"] == s] for s in NAMES}
SELECT_ON = [s for s in SELECT_ON if s in PARTS]
if not SELECT_ON:
    raise ValueError("В SELECT_ON нет ни одной выборки из файла")
CHECK = [s for s in NAMES if s not in SELECT_ON and s != "train"] or [NAMES[-1]]
ru = lambda s: SAMPLE_RU.get(s, s)                                # noqa: E731
rng = np.random.default_rng(0)


def curve(d):
    # договоры по убыванию оценки: оценка, накопленный результат, накопленное число включений
    o = np.argsort(-d["score"].to_numpy(), kind="stable")
    return d["score"].to_numpy()[o], np.cumsum(d["profit"].to_numpy()[o])


def at(d, t):
    # что получается на выборке d при пороге t
    m = d["score"] >= t
    k = int(m.sum())
    return {"n": len(d), "k": k, "share": k / max(len(d), 1), "cost": d.loc[m, "cost"].sum(),
            "money": d.loc[m, "money"].sum(), "profit": d.loc[m, "profit"].sum(),
            "payoff": d.loc[m, "payoff"].mean() if k else np.nan,
            "all": d["profit"].sum(), "per1000": d.loc[m, "profit"].sum() / max(len(d), 1) * 1000}


def ci(d, t, q=(5, 95)):
    # разброс результата при пороге t: бутстреп по договорам выборки
    p = np.where(d["score"] >= t, d["profit"], 0.0)
    w = rng.poisson(1.0, size=(N_BOOT, len(p)))
    return np.percentile(w @ p, q)


def recommend(d, caution=CAUTION, n_boot=N_BOOT):
    # порог на выборке d: максимум результата и «осторожный» порог (см. методику)
    s, cum = curve(d)
    ends = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1] + 1            # число включений на границах оценок
    if len(ends) > 400:
        ends = np.unique(ends[np.linspace(0, len(ends) - 1, 400).astype(int)])
    ks = np.r_[0, ends]                                                # 0 — не включаться вовсе
    profit = np.r_[0.0, cum[ends - 1]]
    j_max = int(np.argmax(profit))
    o = np.argsort(-d["score"].to_numpy(), kind="stable")
    p_sorted = d["profit"].to_numpy()[o]
    boot = np.zeros((n_boot, len(ks)))
    for b in range(n_boot):
        c = np.cumsum(rng.poisson(1.0, len(p_sorted)) * p_sorted)
        boot[b, 1:] = c[ends - 1]
    se = boot[:, j_max].std()                                          # стандартная ошибка результата в максимуме
    ok = profit >= profit[j_max] - caution * se - 1e-9                 # правило одной стандартной ошибки
    j_rec = int(np.nonzero(ok)[0].min())                               # самый строгий из неотличимых
    thr = lambda j: np.inf if ks[j] == 0 else float(s[ks[j] - 1])      # noqa: E731
    return {"t_max": thr(j_max), "t_rec": thr(j_rec), "profit_max": profit[j_max], "profit_rec": profit[j_rec],
            "share_max": ks[j_max] / len(s), "share_rec": ks[j_rec] / len(s),
            "band": (thr(int(np.nonzero(ok)[0].max())), thr(j_rec))}


sel = pd.concat([PARTS[s] for s in SELECT_ON])
rec = recommend(sel)
T = float(CUTOFF) if CUTOFF is not None else rec["t_rec"]
T_SOURCE = "задан вручную (CUTOFF)" if CUTOFF is not None else "рекомендованный"
RES = {s: at(d, T) for s, d in PARTS.items()}
SEL_RU = " + ".join(ru(s) for s in SELECT_ON)
""")

code("""
main_check = CHECK[0]
r = RES[main_check]
lo, hi = ci(PARTS[main_check], T)
display(Markdown(f'''
# Выбор порога отбора: модель «{MODEL}»

**Правило решения:** включаемся в реестр по договору, если оценка модели не ниже порога.
Порог в этом расчёте — **{sc(T)}** ({T_SOURCE}). Договоры — {num(len(df))}
({", ".join(f"{ru(s)}: {num(len(d))}" for s, d in PARTS.items())}); затраты на включение — пошлина
{"+ " + num(EXTRA_COST) + " ₽ на договор" if EXTRA_COST else "без дополнительных расходов"}.
'''))
show_tiles([
    (sc(T), "порог оценки", T_SOURCE),
    (pct(r["share"], 0), f"договоров проходят порог ({ru(main_check)})", f"{num(r['k'])} из {num(r['n'])}"),
    (signed(r["profit"]), f"результат ({ru(main_check)})", f"с вероятностью 90 % — от {signed(lo)} до {signed(hi)}"),
    (signed(r["profit"] - r["all"], 0), "к включению по всем", f"по всем: {signed(r['all'], 0)}"),
])
""")

md("""
## Методика: как выбирается порог

1. **Порог — это значение оценки, а не доля договоров.** В работе решение принимается по каждому
   договору отдельно: оценка не ниже порога — включаемся. Доля включений получается сама и может
   меняться от месяца к месяцу.
2. **Что максимизируем.** Результат = поступления за 2 года − затраты на включение (пошлина и расходы
   на подачу), сумма по включённым договорам. Чем ниже порог, тем больше договоров: сначала
   добавляются окупаемые, потом убыточные — у результата есть максимум.
3. **Где подбираем и где проверяем.** Порог подбирается на выборках из настройки `SELECT_ON`
   (по умолчанию — валидация: модель на ней не обучалась). Тест в подборе не участвует — на нём
   смотрим, что получится на новых договорах. Порог, подобранный на самом тесте, даёт завышенный
   результат: это лучшая точка «задним числом».
4. **Не чистый максимум, а осторожный порог.** Вблизи максимума кривая результата почти плоская, а
   сама точка максимума случайна: на другой выборке она сдвинется. Поэтому берём **самый строгий
   порог, результат которого статистически не отличим от максимального** (уступает максимуму меньше
   одной стандартной ошибки самого максимума, оценённой бутстрепом по договорам). Строгий порог — меньше включений и меньше
   уплаченной пошлины при том же ожидаемом результате; это запас на то, что новые договоры
   возвращают меньше прежних. Настройка `CAUTION = 0` даёт чистый максимум.
5. **Проверки перед решением.** Результат при выбранном пороге на каждой выборке отдельно (раздел 1),
   форма кривой — нет ли обрыва рядом с порогом (раздел 2), окупаемость по группам оценки — где
   договоры перестают окупаться (раздел 3), устойчивость по кварталам (раздел 5) и к росту затрат
   (раздел 6).
""")

code("""
display(Markdown("## 1. Что даёт выбранный порог на каждой выборке"))
rows = {}
for s, d in PARTS.items():
    r = RES[s]
    lo, hi = ci(d, T)
    rows[ru(s)] = {
        "период включения в РТК": f"{d['rtk_send_date'].min():%d.%m.%Y} – {d['rtk_send_date'].max():%d.%m.%Y}",
        "договоров": num(r["n"]), "проходят порог": num(r["k"]), "доля включений": pct(r["share"]),
        "из них окупают затраты": "—" if not r["k"] else pct(r["payoff"], 0),
        "затраты, млн ₽": dec(r["cost"] / 1e6, 1), "поступления, млн ₽": dec(r["money"] / 1e6, 1),
        "результат, млн ₽": signed(r["profit"]).replace(" млн ₽", ""),
        "результат с вероятностью 90 %, млн ₽": f"{signed(lo)} … {signed(hi)}".replace(" млн ₽", ""),
        "на 1 000 рассмотренных договоров, тыс ₽": dec(r["per1000"] / 1e3, 0),
        "на 1 ₽ затрат возвращается, ₽": "—" if not r["cost"] else dec(r["money"] / r["cost"], 2),
        "включаться по всем, млн ₽": signed(r["all"]).replace(" млн ₽", ""),
    }
display(pd.DataFrame(rows))
display(Markdown(f'''
**Подбор порога на выборке «{SEL_RU}»:** максимум результата — при пороге {sc(rec["t_max"])}
({pct(rec["share_max"], 0)} договоров, {signed(rec["profit_max"])}); осторожный порог — {sc(rec["t_rec"])}
({pct(rec["share_rec"], 0)} договоров, {signed(rec["profit_rec"])}). Пороги от {sc(rec["band"][0])} до
{sc(rec["band"][1])} дают результат, не отличимый от максимального.
'''))
""")

code("""
display(Markdown("## 2. Как результат зависит от порога"))
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09, subplot_titles=[
    "по порогу оценки", "по доле включений"])
y_all = []
for s, d in PARTS.items():
    sc_, cum = curve(d)
    step = max(1, len(sc_) // 600)
    i = np.r_[np.arange(0, len(sc_), step), len(sc_) - 1]
    y = cum[i] / len(d) * 1000 / 1e3                         # тыс ₽ на 1 000 рассмотренных договоров
    share = (i + 1) / len(d)
    y_all.append(y[share <= 0.7])
    hov = "порог %{customdata[0]:.4f}<br>включаемся по %{customdata[1]:.0%}<br>%{y:,.0f} тыс ₽ на 1 000 договоров<extra>" + ru(s) + "</extra>"
    cd = np.c_[sc_[i], share]
    fig.add_scatter(x=sc_[i], y=y, mode="lines", name=ru(s), legendgroup=s, customdata=cd, hovertemplate=hov,
                    line={"color": SAMPLE_COLOR.get(s, INK2), "width": 2.5 if s in CHECK else 2}, row=1, col=1)
    fig.add_scatter(x=share, y=y, mode="lines", name=ru(s), legendgroup=s, showlegend=False, customdata=cd,
                    hovertemplate=hov, line={"color": SAMPLE_COLOR.get(s, INK2), "width": 2.5 if s in CHECK else 2},
                    row=1, col=2)
    r = RES[s]
    if r["k"]:
        fig.add_scatter(x=[r["share"]], y=[r["per1000"] / 1e3], mode="markers", showlegend=False, hoverinfo="skip",
                        marker={"color": SAMPLE_COLOR.get(s, INK2), "size": 10, "line": {"color": "white", "width": 2}},
                        row=1, col=2)
fig.add_hline(y=0, line={"color": INK2, "width": 1})
if np.isfinite(T):
    fig.add_vline(x=T, line={"color": INK, "width": 1.5, "dash": "dot"}, row=1, col=1,
                  annotation_text=f"порог {sc(T)}", annotation_position="top right",
                  annotation_font={"color": INK, "size": 12})
    lo_b, hi_b = rec["band"]
    if np.isfinite(lo_b) and np.isfinite(hi_b):
        fig.add_vrect(x0=lo_b, x1=hi_b, fillcolor=MUTED, opacity=0.25, line_width=0, row=1, col=1)
ys = np.concatenate(y_all)
fig.update_yaxes(range=[min(ys.min(), 0) * 1.1 - 5, ys.max() * 1.25 + 5])
q_lo = float(np.quantile(df["score"], 0.3))
fig.update_xaxes(title_text="порог оценки (правее — строже)", range=[q_lo, float(df["score"].quantile(0.999))], row=1, col=1)
fig.update_xaxes(title_text="доля включений (от самой высокой оценки)", tickformat=".0%", range=[0, 0.7], row=1, col=2)
fig.update_yaxes(title_text="тыс ₽ на 1 000 рассмотренных договоров", row=1, col=1)
layout(fig, "Результат в зависимости от порога — по каждой выборке",
       "поступления − затраты в пересчёте на 1 000 рассмотренных договоров; серая полоса — пороги, не отличимые от максимума; точки — выбранный порог",
       height=470, top=120, legend={"orientation": "h", "y": -0.2, "x": 0})
fig.update_annotations(font={"size": 13, "color": INK2})
fig.show()
display(Markdown("Наведите курсор на линию — видно порог, долю включений и результат в этой точке. Чтобы "
                 "проверить конкретное значение, задайте `CUTOFF` в настройках и перезапустите ноутбук."))
""")

code("""
display(Markdown("**Результат при разных порогах** — доли включений заданы на выборке подбора, порог один для всех выборок:"))
shares = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0]
s_sel = np.sort(sel["score"].to_numpy())[::-1]
grid = {}
for q in shares:
    t = float(s_sel[max(int(round(len(s_sel) * q)) - 1, 0)])
    row = {"порог оценки": sc(t)}
    for s, d in PARTS.items():
        r = at(d, t)
        row[f"{ru(s)}: доля включений"] = pct(r["share"], 0)
        row[f"{ru(s)}: результат, млн ₽"] = signed(r["profit"]).replace(" млн ₽", "")
    grid[pct(q, 0)] = row
row = {"порог оценки": sc(T)}
for s in PARTS:
    row[f"{ru(s)}: доля включений"] = pct(RES[s]["share"], 0)
    row[f"{ru(s)}: результат, млн ₽"] = signed(RES[s]["profit"]).replace(" млн ₽", "")
grid["выбранный порог"] = row
display(pd.DataFrame(grid).T.rename_axis(f"доля включений ({SEL_RU})"))
""")

code("""
display(Markdown('''
## 3. Где договоры перестают окупаться

Договоры разбиты на 20 равных групп по оценке (границы общие для всех выборок). Столбец — средний
результат на договор в группе: пока он выше нуля, включать договоры этой группы выгодно.
'''))
edges = np.unique(np.quantile(df["score"], np.linspace(0, 1, 21)))
edges[0], edges[-1] = -np.inf, np.inf
df["band"] = len(edges) - 1 - pd.cut(df["score"], edges, labels=False, include_lowest=True)   # 1 — высшая оценка
NB = len(edges) - 1
lows = {b: edges[NB - b] for b in range(1, NB + 1)}                                            # нижняя граница группы
fig = make_subplots(rows=1, cols=len(NAMES), horizontal_spacing=0.05, shared_yaxes=True,
                    subplot_titles=[ru(s) for s in NAMES])
last_pos = {}
for j, s in enumerate(NAMES, start=1):
    d = df[df["sample"] == s]
    g = d.groupby("band").agg(profit=("profit", "mean"), n=("profit", "size"), money=("money", "mean"),
                              cost=("cost", "mean")).reindex(range(1, NB + 1))
    pos = g["profit"].fillna(0) > 0
    last_pos[s] = int(np.argmax(~pos.to_numpy())) if (~pos).any() else NB        # сколько верхних групп подряд в плюсе
    fig.add_bar(x=g.index, y=g["profit"], row=1, col=j, showlegend=False,
                marker={"color": [BLUE if v else RED for v in pos], "cornerradius": 3},
                customdata=np.c_[[lows[b] for b in g.index], g["n"], g["money"], g["cost"]],
                hovertemplate="группа %{x}: оценка от %{customdata[0]:.4f}<br>договоров %{customdata[1]:,.0f}<br>"
                              "поступления %{customdata[2]:,.0f} ₽, затраты %{customdata[3]:,.0f} ₽<br>"
                              "результат %{y:,.0f} ₽ на договор<extra></extra>")
    if np.isfinite(T):
        k_in = sum(lows[b] >= T for b in range(1, NB + 1))
        fig.add_vline(x=k_in + 0.5, line={"color": INK, "width": 1.5, "dash": "dot"}, row=1, col=j)
fig.add_hline(y=0, line={"color": INK2, "width": 1})
layout(fig, "Средний результат на договор по группам оценки",
       "группа 1 — самая высокая оценка; пунктир — выбранный порог (левее — включаемся)", height=420, top=120,
       bargap=0.15)
fig.update_xaxes(title_text="группа по оценке")
fig.update_yaxes(title_text="₽ на договор", col=1)
fig.update_annotations(font={"size": 13, "color": INK2})
fig.show()
display(Markdown("Верхних групп подряд с положительным результатом: "
                 + ", ".join(f"{ru(s)} — {v} из {NB} (оценка от {sc(lows[v]) if v else '—'})" for s, v in last_pos.items()) + "."))
""")

code("""
display(Markdown("## 4. Насколько надёжен выбор"))
rows = {}
for s, d in PARTS.items():
    own = recommend(d, n_boot=min(N_BOOT, 200))
    rows[ru(s)] = {
        "свой максимум: порог": sc(own["t_max"]), "свой максимум: доля включений": pct(own["share_max"], 0),
        "свой максимум: результат, млн ₽": signed(own["profit_max"]).replace(" млн ₽", ""),
        "свой осторожный порог": sc(own["t_rec"]),
        "результат при выбранном пороге, млн ₽": signed(RES[s]["profit"]).replace(" млн ₽", ""),
        "потеря к своему максимуму, млн ₽": dec((own["profit_max"] - RES[s]["profit"]) / 1e6, 1),
        "доля от своего максимума": "—" if own["profit_max"] <= 0 else pct(RES[s]["profit"] / own["profit_max"], 0),
    }
display(pd.DataFrame(rows))
display(Markdown("«Свой максимум» — лучшая точка каждой выборки, найденная на ней же: достижимый потолок, а не "
                 "ожидаемый результат. Если пороги максимума на выборках близки, а потеря к максимуму мала — "
                 "порог переносится между периодами. Если максимум на тесте заметно строже — новые договоры "
                 "платят хуже, и порог стоит ужесточить."))
""")

code("""
display(Markdown("## 5. Устойчивость во времени"))
df["quarter"] = df["rtk_send_date"].dt.to_period("Q").astype(str)
df["take"] = df["score"] >= T
q = df.groupby("quarter").agg(n=("take", "size"), share=("take", "mean"), all_=("profit", "sum"),
                              sample=("sample", lambda v: v.mode().iat[0]))
tk = df[df["take"]].groupby("quarter").agg(profit=("profit", "sum"), payoff=("payoff", "mean"))
q = q.join(tk).fillna({"profit": 0.0})
q["per1000"] = q["profit"] / q["n"] * 1000 / 1e3
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=[
    "результат, тыс ₽ на 1 000 договоров", "доля договоров выше порога", "из включённых окупают затраты"])
fig.add_bar(x=q.index, y=q["per1000"], row=1, col=1, showlegend=False,
            marker={"color": [BLUE if v >= 0 else RED for v in q["per1000"]], "cornerradius": 4},
            text=[dec(v, 0) for v in q["per1000"]], textposition="outside", cliponaxis=False,
            textfont={"color": INK, "size": 11}, customdata=q["sample"].map(ru),
            hovertemplate="%{x} (%{customdata}): %{y:,.0f} тыс ₽<extra></extra>")
for j, (col, d_) in enumerate([("share", 0), ("payoff", 0)], start=2):
    fig.add_bar(x=q.index, y=q[col], row=1, col=j, showlegend=False, marker={"color": BLUE, "cornerradius": 4},
                text=["—" if pd.isna(v) else pct(v, d_) for v in q[col]], textposition="outside", cliponaxis=False,
                textfont={"color": INK, "size": 11}, hovertemplate="%{x}: %{text}<extra></extra>")
layout(fig, f"Порог {sc(T)} по кварталам включения в РТК",
       "если доля договоров выше порога или результат заметно меняются от квартала к кварталу — порог нужно пересматривать",
       height=410, top=120, bargap=0.3)
fig.update_yaxes(showticklabels=False, showgrid=False)
fig.update_yaxes(zeroline=True, zerolinecolor=MUTED, col=1)
fig.update_xaxes(tickangle=-30)
fig.update_annotations(font={"size": 13, "color": INK2})
fig.show()
display(pd.DataFrame({
    "выборка": q["sample"].map(ru), "договоров": q["n"].map(num), "доля выше порога": q["share"].map(lambda v: pct(v, 0)),
    "из включённых окупают": q["payoff"].map(lambda v: "—" if pd.isna(v) else pct(v, 0)),
    "результат, млн ₽": q["profit"].map(lambda v: signed(v).replace(" млн ₽", "")),
    "включаться по всем, млн ₽": q["all_"].map(lambda v: signed(v).replace(" млн ₽", "")),
}).rename_axis("квартал включения"))
""")

code("""
display(Markdown('''
## 6. Если затраты окажутся выше

К затратам на каждый договор добавлена сумма сверх уже учтённой. Для каждого сценария: что даёт
выбранный порог и какой порог был бы рекомендован.
'''))
rows = {}
base_profit = df["profit"].copy()
for extra in (0, 1_000, 2_000, 5_000):
    df["profit"] = base_profit - extra
    parts = {s: df[df["sample"] == s] for s in NAMES}
    r_sel = recommend(pd.concat([parts[s] for s in SELECT_ON]), n_boot=min(N_BOOT, 200))
    row = {"рекомендованный порог": sc(r_sel["t_rec"])}
    for s in CHECK:
        row[f"{ru(s)}: при выбранном пороге, млн ₽"] = signed(at(parts[s], T)["profit"]).replace(" млн ₽", "")
        row[f"{ru(s)}: при рекомендованном, млн ₽"] = signed(at(parts[s], r_sel["t_rec"])["profit"]).replace(" млн ₽", "")
        row[f"{ru(s)}: доля включений при рекомендованном"] = pct(at(parts[s], r_sel["t_rec"])["share"], 0)
    rows["как в расчёте" if extra == 0 else f"+ {num(extra)} ₽ на договор"] = row
df["profit"] = base_profit
display(pd.DataFrame(rows).T.rename_axis("затраты"))
""")

code("""
r = RES[main_check]
lo, hi = ci(PARTS[main_check], T)
own = recommend(PARTS[main_check], n_boot=min(N_BOOT, 200))
qs = q[q["sample"] == main_check]["share"]
verdict = []
verdict.append("результат на проверочной выборке положительный и интервал целиком выше нуля" if lo > 0 else
               "результат на проверочной выборке положительный, но интервал захватывает ноль — запас небольшой" if r["profit"] > 0 else
               "на проверочной выборке результат отрицательный — порог нужно ужесточать")
if own["profit_max"] > 0:
    verdict.append(f"это {pct(max(r['profit'], 0) / own['profit_max'], 0)} от максимума, достижимого на этой выборке задним числом "
                   f"(порог {sc(own['t_max'])})")
if own["t_max"] > T and np.isfinite(own["t_max"]):
    verdict.append("максимум на проверочной выборке строже выбранного порога — новые договоры платят хуже, "
                   "есть смысл сдвинуть порог вверх")
display(Markdown(f'''
## Итог

- **Порог {sc(T)}** ({T_SOURCE}; подбор — на выборке «{SEL_RU}»). Включаемся, если оценка модели
  «{MODEL}» не ниже порога.
- **На выборке «{ru(main_check)}»:** включаемся по {pct(r["share"], 0)} договоров, результат {signed(r["profit"])}
  (с вероятностью 90 % — от {signed(lo)} до {signed(hi)}), это {dec(r["per1000"] / 1e3, 0)} тыс ₽ на 1 000
  рассмотренных договоров; включение по всем дало бы {signed(r["all"], 0)}.
- **Оценка:** {"; ".join(verdict)}.
- **Диапазон без потери результата:** пороги от {sc(rec["band"][0])} до {sc(rec["band"][1])} на выборке подбора
  не отличимы от максимума — внутри него порог можно двигать из деловых соображений (бюджет на
  пошлины, загрузка юристов).

**Что делать после решения**

1. Зафиксировать порог как число и применять его к оценке каждого нового договора.
2. Раз в квартал пересчитывать этот ноутбук на договорах, у которых прошло 2 года, и следить за
   двумя цифрами из раздела 5: долей договоров выше порога и результатом на 1 000 договоров.
3. Пересматривать порог при смене тарифа пошлины или расходов на подачу (раздел 6) и при переобучении модели:
   порог привязан к шкале оценок конкретной модели.
4. По небольшой случайной доле договоров ниже порога продолжать включаться — иначе через год не на чем
   будет проверить порог и переобучить модель.

*Оговорки те же, что в отчёте: все поступления считаются результатом включения, деньги не приведены
к текущей стоимости; результат — оценка сверху.*
'''))
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["title"] = "Выбор порога отбора"
out = Path(__file__).with_name("cutoff.ipynb")
nbf.write(nb, out)
print("written", out.name)
