"""Таргет «поступления за 2 года больше госпошлины» — отдельной таблицей.

    python report/rtk_target.py                  # настройки — из settings_local.py, как у отчёта
    python report/rtk_target.py --out my.csv     # другой файл результата
    python report/rtk_target.py --all            # вместе с договорами, у которых 2 года ещё не прошло

Результат — report/out/target.csv, строка на договор:

    contract_number, financial_account_subtype_cd, rtk_send_date, rtk_balance,
    money     — поступления за 730 дней после включения в РТК (со знаком: возвраты вычитаются)
    duty      — пошлина за включение в реестр: DUTY_SHARE × имущественная пошлина + OVERHEAD
    profit    — money − duty
    payoff    — 1, если profit > 0 (таргет «окупил пошлину»)
    share5    — 1, если money / rtk_balance > 5 % (прежний таргет)
    paid      — 1, если была хотя бы одна положительная транзакция
    full_2y   — 1, если с включения прошло 730 дней (только с --all; у остальных таргеты пустые)

Функции `duty_scale` и `build_target` можно импортировать: отчёт считает пошлину этой же `duty_scale`.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
H = 730  # горизонт наблюдения, дней


def duty_scale(claim):
    """Имущественная пошлина, подп. 1 п. 1 ст. 333.21 НК РФ (ред. 259-ФЗ), ₽."""
    x = np.asarray(claim, dtype=float)
    d = np.select(
        [x <= 100_000, x <= 1_000_000, x <= 10_000_000, x <= 50_000_000],
        [10_000.0, 10_000 + 0.05 * (x - 100_000), 55_000 + 0.03 * (x - 1_000_000),
         325_000 + 0.01 * (x - 10_000_000)],
        default=725_000 + 0.005 * (x - 50_000_000))
    return np.minimum(d, 10_000_000)


def build_target(tx: pd.DataFrame, *, duty_share: float = 0.5, overhead: float = 0,
                 horizon: int = H, retro_to=None, keep_immature: bool = False) -> pd.DataFrame:
    """Таблица таргетов по договорам из таблицы транзакций.

    Parameters
    ----------
    tx : pd.DataFrame
        Строка — договор без транзакций либо одна транзакция. Колонки: ``contract_number``,
        ``financial_account_subtype_cd``, ``rtk_send_date``, ``rtk_balance``,
        ``real_transaction_dttm``, ``transaction_amt``.
    duty_share, overhead
        Затраты на включение: ``duty_share × duty_scale(rtk_balance) + overhead``.
    retro_to : date-like, optional
        Последняя дата включения в РТК; по умолчанию — все договоры, у которых прошло ``horizon`` дней.
    keep_immature : bool
        Оставить договоры, у которых срок ещё не прошёл (таргеты у них пустые, ``full_2y = 0``).
    """
    need = ["contract_number", "financial_account_subtype_cd", "rtk_send_date", "rtk_balance",
            "real_transaction_dttm", "transaction_amt"]
    missing = [c for c in need if c not in tx.columns]
    if missing:
        raise ValueError(f"В таблице транзакций нет колонок: {', '.join(missing)}")

    tx = tx[need].copy()
    tx["rtk_send_date"] = pd.to_datetime(tx["rtk_send_date"], format="mixed").dt.normalize()
    tx_date = pd.to_datetime(tx["real_transaction_dttm"], format="mixed", errors="coerce").dt.normalize()
    tx["rtk_balance"] = pd.to_numeric(tx["rtk_balance"], errors="coerce")
    tx["transaction_amt"] = pd.to_numeric(tx["transaction_amt"], errors="coerce")
    tx["tx_days"] = (tx_date - tx["rtk_send_date"]).dt.days
    last_tx = tx_date.max()
    retro_to = pd.Timestamp(retro_to) if retro_to is not None else last_tx - pd.Timedelta(days=horizon)

    out = tx.drop_duplicates("contract_number")[
        ["contract_number", "financial_account_subtype_cd", "rtk_send_date", "rtk_balance"]].copy()
    window = tx[tx["transaction_amt"].notna() & tx["tx_days"].between(0, horizon)]
    out["money"] = out["contract_number"].map(
        window.groupby("contract_number")["transaction_amt"].sum()).fillna(0.0)
    out["duty"] = duty_share * duty_scale(out["rtk_balance"]) + overhead
    out["profit"] = out["money"] - out["duty"]
    out["payoff"] = (out["profit"] > 0).astype(float)
    out["share5"] = ((out["money"] / out["rtk_balance"]).clip(lower=0) > 0.05).astype(float)
    payers = set(window.loc[window["transaction_amt"] > 0, "contract_number"])
    out["paid"] = out["contract_number"].isin(payers).astype(float)

    mature = out["rtk_send_date"] <= retro_to
    if keep_immature:
        out["full_2y"] = mature.astype(int)
        out.loc[~mature, ["money", "profit", "payoff", "share5", "paid"]] = np.nan
    else:
        out = out[mature]
    return out.reset_index(drop=True)


def load_settings(path: Path) -> dict:
    """Настройки отчёта: значения по умолчанию, поверх них — файл settings_local.py."""
    ns: dict = {
        "pd": pd, "np": np, "Path": Path, "os": os,
        "SOURCE": "csv", "TX_CSV": "data/df_1.csv", "TX_QUERY": None, "CACHE_DIR": "data_cache",
        "REFRESH": False, "RETRO_TO": None, "DUTY_SHARE": 0.5, "OVERHEAD": 0,
    }
    if path.exists():
        exec(path.read_text(encoding="utf-8"), ns)  # noqa: S102 — свой файл настроек
    return ns


def load_transactions(cfg: dict) -> pd.DataFrame:
    if cfg["SOURCE"] == "csv":
        return pd.read_csv(cfg["TX_CSV"], low_memory=False)
    cache = Path(cfg["CACHE_DIR"]) / "transactions.csv" if cfg["CACHE_DIR"] else None
    if cache is not None and cache.exists() and not cfg["REFRESH"]:
        return pd.read_csv(cache, low_memory=False)
    table = cfg["load_query"](cfg["TX_QUERY"])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache, index=False)
    return table


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Таргет «поступления больше госпошлины» по договорам")
    parser.add_argument("--settings", default=os.environ.get("RTK_REPORT_SETTINGS", "settings_local.py"),
                        help="файл настроек (по умолчанию settings_local.py рядом со скриптом)")
    parser.add_argument("--out", help="куда сохранить таблицу (по умолчанию report/out/target.csv)")
    parser.add_argument("--all", action="store_true", help="оставить договоры, у которых 2 года не прошло")
    args = parser.parse_args()

    settings = Path(args.settings)
    settings = settings.resolve() if settings.exists() else HERE / settings
    out = Path(args.out).resolve() if args.out else HERE / "out" / "target.csv"
    os.chdir(HERE)  # пути в настройках — относительно папки report/, как у отчёта
    cfg = load_settings(settings)
    target = build_target(load_transactions(cfg), duty_share=cfg["DUTY_SHARE"], overhead=cfg["OVERHEAD"],
                          retro_to=cfg["RETRO_TO"], keep_immature=args.all)
    out.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(out, index=False, encoding="utf-8-sig")

    done = target if "full_2y" not in target else target[target["full_2y"] == 1]
    print(f"сохранено: {out}")
    print(f"договоров: {len(target):,} (с полными 2 годами: {len(done):,})".replace(",", " "))
    summary = done.groupby("financial_account_subtype_cd").agg(
        договоров=("payoff", "size"), окупили_пошлину=("payoff", "mean"),
        вернули_больше_5=("share5", "mean"), с_платежом=("paid", "mean"),
        поступления_млн=("money", lambda v: v.sum() / 1e6), пошлина_млн=("duty", lambda v: v.sum() / 1e6),
    ).sort_values("договоров", ascending=False)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
