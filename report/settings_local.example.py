# Пример settings_local.py: скопируйте под этим именем рядом с rtk_report.ipynb и поправьте.
# Файл выполняется поверх настроек из первой ячейки ноутбука; в git не попадает.

# --- вариант 1: данные из файлов
# SOURCE = "csv"
# TX_CSV = "/workdir/RTK_model/data/df_1.csv"
# FEATURES_CSV = "/workdir/RTK_model/data/df_agg.csv"

# --- вариант 2: данные запросом
SOURCE = "query"
TX_QUERY = "select * from <схема>.<таблица транзакций>"
FEATURES_QUERY = "select * from <схема>.<таблица признаков>"
REFRESH = False          # True — перечитать из базы, а не из data_cache/


def load_query(sql):
    """Вернуть pandas.DataFrame по тексту запроса — под свою среду."""
    import os

    import tinkoffpy as tf
    from dotenv import load_dotenv

    load_dotenv()        # username / password из файла .env
    conf = tf.Configuration(username=os.getenv("username"), password=os.getenv("password"))
    return tf.Greenplum(tf.ApiClient(conf)).gp_to_df(sql, gp_service="vrcl")


# --- готовая модель вместо обучения в отчёте (по умолчанию отчёт обучает модель сам)
# MODEL_PATH = "/workdir/RTK_model/models/v_1/lgbm_final.pkl"
# MODEL_FEATURES_JSON = "/workdir/RTK_model/logs/v_1/features.json"
# SCORE_COL = "score"    # либо готовая оценка колонкой в таблице признаков
# MODEL_TARGET = "договор вернёт больше 5 % долга за 2 года"

# --- что считать
PRODUCT = "PHX"          # продукт для подробного разбора
MODEL_PRODUCTS = None    # None — все продукты, по которым есть признаки; или ["PHX", "CLA", "CAR"]
OOT_FROM = "2024-01-01"  # с этой даты включения — проверочная выборка
OVERHEAD = 0             # ₽ на договор сверх пошлины
