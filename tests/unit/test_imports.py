"""Всё импортируется и слои не перепутаны.

Сервис не должен зависеть от обучения: раньше tts_service тянул load_state из
train_voice и fix_known из say, и любой опыт с обучением мог уронить боевой
TTS. Тест держит эту границу.
"""
import importlib
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULES = ["voice.model", "voice.audio", "voice.text", "voice.numbers", "voice.asr",
           "voice.paths", "voice.phrases", "voice.targets",
           "service.tts", "train.train", "train.pipeline", "train.bake_prosody",
           "train.prep_text", "train.prep_targets", "train.gen_f5",
           "tools.say", "tools.asr_scan", "probe.probe1", "probe.probe2",
           "probe.diag_voice", "probe.prosody_try"]


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(name)


def test_voice_does_not_import_the_rest():
    for f in (ROOT / "voice").glob("*.py"):
        src = f.read_text(encoding="utf-8")
        for bad in ("import train", "import service", "import probe", "import tools"):
            assert bad not in src, f"{f.name} тянет {bad}"


def test_service_does_not_import_train():
    # путь /w/train/model_voice.pt — это данные, а не зависимость от кода
    src = (ROOT / "service" / "tts.py").read_text(encoding="utf-8")
    bad = [s for s in src.splitlines()
           if re.match(r"\s*(from|import)\s+(train|probe|tools)\b", s)]
    assert bad == []
