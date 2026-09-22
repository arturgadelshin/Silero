"""paths.py — где лежат данные.

Код примонтирован в /app, данные (кэш hub, корпус, цели, веса) в /w. Корень
переопределяется переменной SILERO_DATA — например, чтобы прогнать тесты на
копии данных.
"""
import os

DATA = os.environ.get("SILERO_DATA", "/w")


def data(*parts):
    return os.path.join(DATA, *parts)
