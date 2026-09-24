# voices

Готовые голоса для `web/engine.py` (клон `predict` из `voice/model.py`).

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
