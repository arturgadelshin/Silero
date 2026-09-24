"""Клампы параметров генерации — без загрузки модели (web/engine импортируется,
но Engine не строится). Дефолты равны настройкам модели, как у сервиса."""
from web.engine import DEFAULTS, clamp_params


def test_empty_params_give_defaults():
    assert clamp_params({}) == DEFAULTS


def test_defaults_values():
    assert DEFAULTS["speech_rate"] == 1.0 and DEFAULTS["pitch_pct"] == 0.0
    assert DEFAULTS["pause"] == 0.5 and DEFAULTS["gap"] == 0.25
    assert DEFAULTS["numbers"] and DEFAULTS["accent"]
    assert DEFAULTS["sr"] == 48000


def test_clamps_out_of_range():
    p = clamp_params({"speech_rate": 5.0, "pitch_pct": -100.0,
                      "pause": 9.0, "gap": -1.0})
    assert p["speech_rate"] == 2.0 and p["pitch_pct"] == -60.0
    assert p["pause"] == 1.5 and p["gap"] == 0.0


def test_sr_not_in_list_falls_back():
    assert clamp_params({"sr": 44100})["sr"] == 48000
    assert clamp_params({"sr": "24000"})["sr"] == 24000


def test_types_coerced():
    p = clamp_params({"speech_rate": "1.2", "numbers": 0})
    assert p["speech_rate"] == 1.2 and p["numbers"] is False
