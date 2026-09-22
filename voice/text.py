"""text.py — подготовка русского текста к синтезу.

Числа словами (voice.numbers), потом разметка ударений, потом символы Silero:
текст с «+» перед ударной гласной и без символов вне алфавита из 47 знаков.

Разметчиков два, интерфейс у них общий (вызов от строки):
  Stress — silero-stress, MIT, грузится 3.8 с, 6 мс на фразу, сеть не нужна;
  Accent — RUAccent, 26 с и докачка моделей koziev в site-packages.
На сверке с авторскими «+» RUSLAN они идут вровень (99.41 против 99.33 %),
разница в 17 словах на 1152 фразы. Подробности в SILERO.md.
"""
import re

VOWELS = "аеёиоуыэюя"
SPLIT = re.compile(r"(?<=[.!?…])\s+")

# известные ошибки RUAccent, пойманные на шаге 1 (SILERO.md, сверка с RUSLAN):
# «самое» он превращает в редкое «самоё», в «джинсы» и «баловался» бьёт мимо
FIXES = {"само+ё": "с+амое", "джинс+ы": "дж+инсы", "б+аловался": "балов+ался"}


def syllables(t):
    return sum(1 for ch in t.lower() if ch in VOWELS)


def norm_text(s):
    """для CER: без ударений, ё=е, только буквы и пробелы"""
    s = s.lower().replace("ё", "е").replace("+", "")
    return re.sub(r"\s+", " ", re.sub(r"[^а-я ]", " ", s)).strip()


def cer(ref, hyp):
    r, h = norm_text(ref), norm_text(hyp)
    if not r:
        return 1.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1] / len(r)


def fix_known(t):
    # регистр сохраняем: замена в лоб с IGNORECASE съедала заглавную в начале
    # предложения. Silero всё равно приводит текст к строчным, но размеченный
    # текст читают люди (--dump-text, format=json)
    for bad, good in FIXES.items():
        pat = re.compile(bad.replace("+", r"\+"), re.IGNORECASE)
        t = pat.sub(lambda m: good.capitalize() if m.group(0)[0].isupper() else good, t)
    return t


def pre_ruaccent(t):
    """RUAccent 1.5.8.3 молча выкидывает «…», «–», «—» и неразрывный дефис U+2011
    («по‑английски» -> «поанглийски»), а «...», « - », «-», «:», «;» сохраняет.
    Кавычки и скобки Silero всё равно выкинет — меняем их на пробел, иначе
    «(нет)» после RUAccent прилипает к соседнему слову."""
    t = t.replace("́", "").replace("’", "")   # редкое авторское ударение, апостроф
    t = t.replace("‑", "-").replace("‐", "-")
    t = t.replace("…", "...")
    t = re.sub(r"\s*[–—]\s*", " - ", t)
    t = re.sub(r"[«»„“”\"()\[\]]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def post_accent(t):
    """к символам Silero: «...» -> «…», отдельно стоящий « - » -> « – »,
    знак препинания вплотную к слову. Последнее — не косметика: silero-stress
    отбивает точку пробелом («харб+ина .»), и это лишний символ в токенах,
    со своей длительностью; в корпусе шага 1 такого нет ни разу.

    Длинное тире «—» в алфавит Silero не входит: у RUAccent его сводил к
    дефису pre_ruaccent, а silero-stress пунктуацию не трогает, и тире молча
    выпадало из речи вместе с паузой на нём."""
    t = t.replace("—", "–").replace("―", "–")
    t = re.sub(r"\.\.\.+", "…", t)
    t = re.sub(r"\s+([.,!?;:…])", r"\1", t)
    t = re.sub(r"(^|\s)-(\s|$)", r"\1–\2", t)
    # реплика диалога «–Недавно…» без пробела: дефис в начале слова — всегда тире
    t = re.sub(r"(^|\s)-(?=\S)", r"\1– ", t)
    return re.sub(r"\s+", " ", t).strip()


def stress_map(t):
    """слово без «+» -> индекс ударной гласной (по счёту гласных), только слова с «+»"""
    out = {}
    for w in re.findall(r"[а-яё+\-]+", t.lower()):
        if "+" not in w:
            continue
        pre = w.split("+")[0]
        out[w.replace("+", "").replace("ё", "е")] = sum(ch in VOWELS for ch in pre)
    return out


def split_text(t, limit=300):
    """длинный текст — по предложениям: первый кусок звучит, пока считается второй"""
    out = []
    for chunk in (s.strip() for s in t.replace("\n", " ").split("|") if s.strip()):
        start = len(out)                               # через «|» не клеим: это разрыв
        for s in SPLIT.split(chunk):
            s = s.strip()
            if not s:
                continue
            while len(s) > limit:                      # очень длинное — режем по запятой
                cut = s.rfind(",", 0, limit)
                cut = cut + 1 if cut > limit // 3 else limit
                out.append(s[:cut].strip())
                s = s[cut:].strip()
            # огрызок короче 40 знаков отдельной фразой звучит рвано: клеим его
            # к соседу — и когда короток он сам («Ага.» в хвосте), и когда
            # короток предыдущий кусок. Склейка не должна перерасти limit,
            # иначе она сведёт на нет резку по запятой выше
            if len(out) > start and min(len(out[-1]), len(s)) < 40 \
                    and len(out[-1]) + len(s) < limit:
                out[-1] = (out[-1] + " " + s).strip()
            else:
                out.append(s)
    return out or [t.strip()]


def utf8_or_cp1251(raw):
    """curl из Windows шлёт текст в кодировке консоли: «Который час» уезжает
    как cp1251 (%CA%EE%F2…), и Silero на таком мусоре бросает голый ValueError.
    Пробуем UTF-8, а если байты в него не складываются — cp1251."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1251", "replace")


def patch_token_type_ids(sess):
    """RUAccent 1.5.8.3: модель ударений незнакомых слов (nn_accent) требует
    вход token_type_ids, а CharTokenizer при transformers из kws_tts его не
    отдаёт — ValueError на первом же слове не из словаря. Подставляем нули,
    как BERT делает по умолчанию для одного сегмента."""
    import numpy as np
    need = {i.name for i in sess.get_inputs()}
    if "token_type_ids" not in need:
        return
    run = sess.run

    def run_fixed(out, feed, *args, **kw):
        if "token_type_ids" not in feed:
            feed = dict(feed, token_type_ids=np.zeros_like(feed["input_ids"]))
        return run(out, feed, *args, **kw)
    sess.run = run_fixed


class Accent:
    """RUAccent с патчем и всей обвязкой: грузится 15-25 с и тянет модели
    koziev в site-packages, поэтому контейнеру нужна сеть."""

    def __init__(self, model="turbo3.1", workdir="/ruaccent"):
        from ruaccent import RUAccent
        self.acc = RUAccent()
        try:
            self.acc.load(omograph_model_size=model, use_dictionary=True, workdir=workdir)
        except TypeError:                               # старая сигнатура load()
            self.acc.load(omograph_model_size=model, use_dictionary=True)
        patch_token_type_ids(self.acc.accent_model.session)

    def __call__(self, s):
        return fix_known(post_accent(self.acc.process_all(pre_ruaccent(s))))


class Stress:
    """silero-stress (MIT): «+» перед ударной гласной, омографы и «ё».

    Пунктуацию не трогает — ни «…», ни «—», ни U+2011, — поэтому обвязки
    pre_ruaccent ему не надо, остаётся привести «...» и «-» к символам
    Silero. Кавычки и скобки Silero всё равно выкинет, но убрать их надо
    заранее, иначе «(нет)» прилипнет к соседнему слову.

    yo_homo — угадывать «ё» в неоднозначных словах. Включено: так же ведёт
    себя RUAccent (58 раз «все» -> «всё» на корпусе у обоих), и на этом
    обучена модель. Цена — промахи вроде «Все всё поняли» -> «Всё всё…».
    """

    def __init__(self, yo_homo=True):
        from silero_stress import load_accentor
        self.acc = load_accentor()
        self.yo_homo = yo_homo

    def __call__(self, s):
        s = s.replace("́", "").replace("’", "")     # авторское ударение, апостроф
        s = s.replace("‑", "-").replace("‐", "-")  # U+2011/U+2010 Silero выкинет
        s = re.sub(r"[«»„“”\"()\[\]]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return post_accent(self.acc(s, put_yo_homo=self.yo_homo))


def accentor(engine="silero", model="turbo3.1", workdir="/ruaccent"):
    """разметчик по имени: silero | ruaccent | none"""
    if engine in ("none", "", None):
        return None
    if engine == "ruaccent":
        return Accent(model, workdir)
    if engine == "silero":
        return Stress()
    raise ValueError(f"неизвестный разметчик: {engine}")


def prepare(s, accent=None, numbers=True):
    """полный путь текста до входа Silero; accent — Stress, Accent или None"""
    if numbers:
        from voice.numbers import expand
        s = expand(s)
    return accent(s) if accent is not None else s


def outside(s, alphabet):
    """символы, которые Silero молча выкинет (сам он приводит текст к строчным)"""
    return sorted({c for c in s.lower().replace("+", "")
                   if c not in alphabet and c != " "})
