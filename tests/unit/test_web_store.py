"""Реестр моделей и история web-морды: только файлы и JSON, без модели."""
import importlib
import os

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SILERO_DATA", str(tmp_path))
    import voice.paths
    importlib.reload(voice.paths)
    import web.store as st
    importlib.reload(st)
    return st


def test_create_rename_list(store):
    a = store.create_model("Мой голос", "full")
    b = store.create_model("Второй", "probe")
    assert store.model_meta(a)["name"] == "Мой голос"
    names = [m["name"] for m in store.list_models()]
    assert names == ["Второй", "Мой голос"]              # новые сверху
    store.rename_model(a, "Другое имя")
    assert store.model_meta(a)["name"] == "Другое имя"


def test_rename_validates(store):
    mid = store.create_model("имя", "probe")
    with pytest.raises(ValueError):
        store.rename_model(mid, "   ")
    with pytest.raises(ValueError):
        store.rename_model(mid, "x" * 61)
    with pytest.raises(ValueError):
        store.create_model("", "probe")


def test_delete_model(store):
    mid = store.create_model("на удаление", "probe")
    p = store.model_dir(mid)
    assert os.path.isdir(p)
    store.delete_model(mid)
    assert not os.path.exists(p)
    assert store.model_meta(mid) is None


def test_history_name_and_delete(store):
    e = store.add_history({"ts": 1700000000.0, "text": "привет",
                           "voice_name": "Мой голос!"}, b"RIFF....")
    assert e["file"].startswith("Мой_голос_") and e["file"].endswith(".wav")
    assert os.path.exists(store.history_path(e))
    idx = store.history_index()
    assert len(idx) == 1 and idx[0]["id"] == e["id"]
    store.delete_history(e["id"])
    assert store.history_index() == []
    assert not os.path.exists(store.history_path(e))


def test_history_same_second_unique(store):
    a = store.add_history({"ts": 1700000000.0, "text": "a",
                           "voice_name": "v"}, b"x")
    b = store.add_history({"ts": 1700000000.0, "text": "b",
                           "voice_name": "v"}, b"y")
    assert a["file"] != b["file"]


def test_meta_roundtrip_survives_missing_file(store):
    mid = store.create_model("голос", "probe")
    os.remove(os.path.join(store.model_dir(mid), "meta.json"))
    assert store.model_meta(mid) is None
    m = store.save_meta(mid, name="восстановлен")
    assert m["name"] == "восстановлен"
