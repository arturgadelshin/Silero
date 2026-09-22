"""Подготовка текста: резка на фразы, разметка вокруг RUAccent, CER."""
import pytest

from voice.text import (cer, fix_known, norm_text, outside, post_ruaccent,
                        pre_ruaccent, split_text, syllables, utf8_or_cp1251)

ALPHABET = set(" абвгдеёжзийклмнопрстуфхцчшщъыьэюя+.,!?–…:;-")


LONG1 = "Довольно длинное первое предложение про всё на свете."
LONG2 = "И второе такое же длинное предложение про всё остальное."


def test_split_by_sentences():
    assert split_text(LONG1 + " " + LONG2) == [LONG1, LONG2]


def test_short_piece_absorbs_the_next():
    # короткая первая фраза приклеивает к себе следующую
    assert split_text("Ага. " + LONG1) == ["Ага. " + LONG1]


def test_short_tail_joins_the_previous_piece():
    # и наоборот: короткий хвост не уезжает отдельной фразой со своей паузой
    assert split_text(LONG1 + " Ага.") == [LONG1 + " Ага."]


def test_glue_does_not_undo_the_comma_cut():
    # склейка не имеет права снова собрать кусок длиннее limit
    s = "а" * 240 + ", " + "б" * 20 + "."
    assert split_text(s, limit=250) == ["а" * 240 + ",", "б" * 20 + "."]


def test_long_sentence_cut_by_comma():
    s = "а" * 200 + ", " + "б" * 200 + "."
    parts = split_text(s, limit=250)
    assert len(parts) == 2 and parts[0].endswith(",")


def test_pipe_is_a_hard_break():
    assert split_text(LONG1 + " | " + LONG2) == [LONG1, LONG2]


def test_empty_text_survives():
    assert split_text("   ") == [""]


def test_cer_is_zero_on_itself():
    assert cer("привет как дела", "привет как дела") == 0.0


def test_cer_ignores_stress_and_yo():
    assert cer("ещё раз", "еще раз") == 0.0
    assert cer("ещ+ё раз", "ещё раз") == 0.0


def test_cer_counts_one_substitution():
    assert cer("дом", "том") == pytest.approx(1 / 3)


def test_cer_of_empty_reference():
    assert cer("", "что-то") == 1.0


def test_norm_text_drops_punctuation():
    assert norm_text("Привет, мир!") == "привет мир"


def test_fix_known_repairs_ruaccent():
    assert fix_known("само+ё ценное") == "с+амое ценное"
    assert fix_known("Само+ё ценное") == "С+амое ценное"


def test_pre_ruaccent_normalises_dashes_and_quotes():
    out = pre_ruaccent("«Макса» — дом… и по‑английски")
    assert "«" not in out and "»" not in out
    assert "…" not in out and "—" not in out
    assert "по-английски" in out


def test_post_ruaccent_returns_silero_symbols():
    assert post_ruaccent("дом ... и") == "дом … и"
    assert post_ruaccent("дом - это") == "дом – это"
    assert post_ruaccent("-Недавно") == "– Недавно"


def test_roundtrip_keeps_ellipsis_and_dash():
    out = post_ruaccent(pre_ruaccent("Он ушёл — молча…"))
    assert "–" in out and "…" in out


def test_outside_finds_symbols_out_of_alphabet():
    assert outside("дом 5", ALPHABET) == ["5"]
    assert outside("ДОМ", ALPHABET) == []


def test_syllables_counts_vowels():
    assert syllables("молоко") == 3
    assert syllables("всплеск") == 1


def test_cp1251_fallback():
    assert utf8_or_cp1251("Который час".encode("cp1251")) == "Который час"
    assert utf8_or_cp1251("Который час".encode("utf-8")) == "Который час"
