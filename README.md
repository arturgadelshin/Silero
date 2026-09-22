# Silero своим голосом

Silero TTS v5_5_ru, дообученный говорить моим голосом. Голос снят с одной
опорной записи ~10 с: ею через F5-TTS озвучен корпус RUSLAN на 2 часа, и на
этой синтетике обучен Silero. Весь путь, все замеры и все грабли записаны в
[SILERO.md](SILERO.md) — это журнал работы, а не справочник.

Только для личного использования: Silero CC BY-NC, корпус RUSLAN CC BY-NC-SA.

## Что где

    voice/      общий слой: модель, звук, текст, числа, ASR, цели обучения
    train/      конвейер обучения: тексты -> клоны F5 -> цели -> ступени -> просодия
    service/    всегда прогретый TTS: HTTP на 10500 и wyoming на 10501
    probe/      вскрытие модели и диагностика голоса (как всё устроено внутри)
    tools/      разовое: озвучить файл, забрать тексты корпуса, прогнать ASR
    tests/      unit/ без модели и accept/ против живого сервиса и весов
    scripts/    доставка на сервер, управление сервисом, запуск тестов
    docker/     образы: silero_ru (боевой) и silero_test (плюс pytest)

Зависимости идут в одну сторону: `voice` не знает ни про кого, а `train`,
`service`, `probe` и `tools` опираются на него. Сервис не зависит от обучения —
опыты с обучением не могут уронить боевой TTS.

Код живёт в `/app`, данные в `/w` (веса, кэш torch hub, корпус, цели). Корень
данных переопределяется переменной `SILERO_DATA`.

## Каждый день

    scripts/deploy.sh --restart     код на сервер и перезапуск сервиса
    scripts/service.sh status       жив ли TTS и что он о себе думает
    scripts/service.sh logs 40      последние строки
    scripts/test.sh                 быстрые тесты в контейнере
    scripts/test.sh -m slow         приёмка: веса, CER по gigaam

Озвучить файл:

    docker run --rm --network host -v ~/linkplay/tmp/silero/hub:/root/.cache/torch/hub \
      -v ~/linkplay/tmp/silero/ruaccent:/ruaccent -v ~/linkplay/silero:/app \
      -v ~/linkplay/tmp/silero:/w -w /app silero_ru \
      python -m tools.say --text /w/say/announce.txt --out /w/say/announce.wav

Спросить сервис:

    curl -G --data-urlencode "text=Привет, это проверка." \
      http://192.168.0.121:10500/say -o out.wav

## Обучение

Ступени идут по порядку и запускаются по одной — обучение занимает слот
RTX 3070 надолго. `train/pipeline.py` печатает план и выполняет его с `--run`:

    python -m train.pipeline --from targets --to stage2
    python -m train.pipeline --only bake --run

Рабочая точка: ступень 2 (декодер, `lin` и тела предсказателей) плюс растяжка
pitch ×1.5, вложенная в веса. Ступень 3 (текстовый кодировщик) переобучается и
отвергнута. Веса не в git — они релизом `voice-1.0`, суммы sha256 в SILERO.md.
