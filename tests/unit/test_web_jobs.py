"""Разбор stdout ступеней обучения и отбор фраз пробы.

Строки взяты из живых прогонов (SILERO.md): парсер хрупок ровно настолько,
насколько стабильны эти строки — тест фиксирует контракт.
"""
import math

from web.jobs import INIT_VOICES, nearest_voice, parse_line, probe_rows


def test_nearest_voice_by_f0():
    assert nearest_voice(104.5) == "eugene"            # artur: дефолт как было
    assert nearest_voice(109) == "eugene"
    assert nearest_voice(186) == "xenia"               # женский кластер ближе
    assert nearest_voice(250) == "baya"
    assert nearest_voice(135) == "aidar"
    assert INIT_VOICES[nearest_voice(186)] > 180       # старт точно не мужской


def test_parse_f5_progress():
    p, det, met = parse_line("f5", "10/24 {'ok': 9, 'fail': 0} | RTF 0.371, 3.2 с/текст")
    assert p == 10 / 24 and det == "10/24" and met is None


def test_parse_f5_ignores_other_steps():
    assert parse_line("targets", "10/24 {'ok': 9}")[0] is None


def test_parse_targets_part_done():
    p, det, _ = parse_line("targets", "часть 3/6: готово 191 из 191, брак {'короткая': 0}")
    assert p == 1.0 and det == "часть 4/6"


def test_parse_targets_part_progress():
    p, det, _ = parse_line("targets", "50/191 готово 49 брак {'ошибка': 1}")
    assert math.isclose(p, 50 / 191) and det == "50/191"


def test_parse_epoch():
    line = ("эпоха   7: обучение мел 0.7102 dur 0.0953 pitch 0.6010 | "
            "проверка мел 0.6980 dur 0.0919 pitch 0.5411 | 84 с")
    p, det, met = parse_line("stage1", line, 30)
    assert math.isclose(p, 7 / 30)
    assert "эпоха 7" in det
    assert met == {"epoch": 7, "mel": 0.6980, "dur": 0.0919, "pitch": 0.5411}


def test_parse_stats():
    line = ("   F0 по символам: μ 108.8 Гц, σ 16.8 Гц, μ/σ 6.49 "
            "(mean_std_coef eugene 2.76); звонких символов 75.7 %")
    p, det, met = parse_line("stats", line)
    assert p == 1.0
    assert met["msc"] == 6.49 and met["f0_mean"] == 108.8 and met["f0_std"] == 16.8


def test_parse_check():
    line = ("== свой голос: CER 0.48 % (макс 3.41), F0 медиана 107 Гц [104–109]; "
            "записи дали μ 109 Гц")
    p, _, met = parse_line("check", line)
    assert p == 1.0
    assert met == {"cer": 0.0048, "cer_max": 0.0341, "f0": 107}


def test_parse_noise_lines():
    for line in ("опора 9.51 с, 0.231 с/слог; текстов 27 (уже готово 0)",
                 "готово: принято 27, не вышло за 3 попытки 0",
                 "ступень 1: учим 3 строки голоса и 0 тензоров"):
        assert parse_line("f5", line) == (None, None, None)


def make_rows(n=50):
    return [[str(i), "val" if i % 20 == 5 else "train", "3.1", "текст"]
            for i in range(n)]


def test_probe_rows_keeps_val_and_order():
    out, tr, va = probe_rows(make_rows(), n_train=10, n_val=2)
    assert tr == 10 and va == 2
    assert sum(1 for r in out if r[1] == "val") == 2
    nums = [int(r[0]) for r in out]
    assert nums == sorted(nums)                          # порядок корпуса сохранён


def test_probe_rows_caps_train():
    out, tr, _ = probe_rows(make_rows(), n_train=7, n_val=1)
    assert tr == 7
    assert sum(1 for r in out if r[1] == "train") == 7
