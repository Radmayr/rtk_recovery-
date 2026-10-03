"""Выполняет ноутбук из report/ и сохраняет результат в report/out/.

    python report/run_report.py                      # отчёт; настройки — из ноутбука и settings_local.py
    python report/run_report.py --settings my.py     # другой файл настроек
    python report/run_report.py --name v2            # другое имя файлов, чтобы не затереть прежние
    python report/run_report.py --notebook cutoff    # подбор порога по оценкам из отчёта

Результат: out/<имя>.html (без кода, для показа) и out/<имя>_executed.ipynb; отчёт сохраняет
ещё out/<имя>_scores.csv — оценки моделей по договорам для подбора порога.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", help="файл настроек вместо settings_local.py")
    parser.add_argument("--out", default=str(HERE / "out"), help="папка для результата")
    parser.add_argument("--name", help="имя файлов результата (по умолчанию — имя ноутбука)")
    parser.add_argument("--notebook", default="rtk_report", help="какой ноутбук выполнить: rtk_report или cutoff")
    args = parser.parse_args()
    args.name = args.name or args.notebook
    os.environ["RTK_REPORT_OUT"] = str(Path(args.out).resolve())
    os.environ["RTK_REPORT_NAME"] = args.name
    if args.settings:
        os.environ["RTK_REPORT_SETTINGS"] = str(Path(args.settings).resolve())

    nb = nbformat.read(HERE / f"{args.notebook}.ipynb", as_version=4)
    started = time.time()
    print(f"выполняю {args.notebook}…")
    ExecutePreprocessor(timeout=None, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(HERE)}})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out / f"{args.name}_executed.ipynb")
    title = nb.metadata.get("title", args.name)
    html, _ = HTMLExporter(exclude_input=True).from_notebook_node(nb, {"metadata": {"name": title}})
    (out / f"{args.name}.html").write_text(html, encoding="utf-8")
    print(f"готово за {time.time() - started:.0f} с: {out / (args.name + '.html')}")


if __name__ == "__main__":
    main()
