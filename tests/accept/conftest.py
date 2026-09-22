"""Приёмка идёт против живого сервиса и настоящих весов. Ничего этого нет на
машине разработчика, поэтому тесты пропускаются, а не падают.
"""
import os
import socket
import urllib.request

import pytest

URL = os.environ.get("TTS_URL", "http://127.0.0.1:10500")
WY = os.environ.get("TTS_WYOMING", "127.0.0.1:10501")


def alive(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:                                   # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def url():
    if not alive(URL):
        pytest.skip(f"сервис не отвечает на {URL}")
    return URL


@pytest.fixture(scope="session")
def wyoming():
    host, port = WY.split(":")
    try:
        socket.create_connection((host, int(port)), timeout=3).close()
    except OSError:
        pytest.skip(f"wyoming не отвечает на {WY}")
    return host, int(port)
