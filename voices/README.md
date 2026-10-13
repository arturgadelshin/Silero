# voices

Готовые голоса для `web/engine.py` (клон `predict` из `voice/model.py`).

## artur

Свой голос, с которого начался проект. Работает дома в боевом `silero_tts`
(10500/10501) — там он грузится не отсюда, а из `/w/train/model_voice.pt`;
копия в git нужна, чтобы голос можно было поднять с нуля на другой машине
тонким образом `docker/Dockerfile.tts`.

| | |
|---|---|
| Файл | `artur/model_voice.pt` |
| SHA256 | `20FA8D86FA421DF34580E1B4352CA5339ED0B6942BA630BE80AF0A46E5AF2CDD` |
| CER | 0.62% (макс по фразам 3.81%) на 30 отложенных |
| F0 | медиана 105 Гц, среднее 108.7 (σ 16.8), msc 6.48 |
| Обучение | F5-клон + ступени 1–2 Silero + просодия, темп `--dur-gain 1.0` |

SHA256 поставки из раздела «Где лежат веса» в `SILERO.md` (`fe455625…`) снят
с более ранней сборки, с замедлением 1.08. Здесь лежит та, что в бою.

## stalker

| | |
|---|---|
| Файл | `stalker/model_voice.pt` |
| SHA256 | `D0D497FEFFB5F78EA747E60027AA5773AAC829D863857969039D8C7088428BA2` |
| CER | 0.47% (макс по фразам 2.27%) |
| F0 | 94 Гц (σ 18.3), msc 5.18 |
| Обучение | web-морда, kind=full: F5 53.6 мин + ступени 1–2 Silero (~54 мин), опора 7.96 с |

Скопировать на сервер и зарегистрировать:

```
scp -r voices/stalker green@192.168.0.121:linkplay/tmp/silero/web/models/<id>/train/
```

 либо через `python -m web.register` (см. `web/README.md`).

## Имя голоса

`tts_voice` в Home Assistant — это строка модели (`xenia`), а подпись берётся
из имени папки: `voices/artur/model_voice.pt` → «artur». Служебному пути
(`/w/train/model_voice.pt` у боевого) подпись задаётся ключом `--voice-label`.
