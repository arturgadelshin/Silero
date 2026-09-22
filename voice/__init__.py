"""voice — общий слой: модель, звук, текст, ASR.

От него зависят train, service, probe и tools; сам он не зависит ни от кого.
Раньше этот код жил в silero_probe2.py и растаскивался по файлам через
sys.path.insert — отсюда и дубли write16/cer/syllables в трёх местах.
"""
