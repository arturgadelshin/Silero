#!/bin/sh
# test.sh — pytest в контейнере на сервере. Образ silero_test (silero_ru +
# pytest) собирается один раз.
#
#   scripts/test.sh                 быстрые тесты
#   scripts/test.sh -m slow         приёмка с весами и ASR
#   scripts/test.sh tests/unit/test_numbers.py -v
set -eu
HOST=${SILERO_HOST:-green@192.168.0.121}
DST=${SILERO_CODE_DIR:-linkplay/silero}
DATA=${SILERO_DATA_DIR:-linkplay/tmp/silero}
ssh "$HOST" "docker image inspect silero_test >/dev/null 2>&1 \
  || docker build -t silero_test -f ~/$DST/docker/Dockerfile.test ~/$DST/docker"
# PYTHONDONTWRITEBYTECODE и no:cacheprovider обязательны: контейнер работает от
# root и иначе насорит __pycache__ в примонтированный код, а deploy потом не
# сможет их удалить.
#
# Вывод собирается в файл на сервере и печатается отдельной командой: если
# отдавать его из docker run прямо в ssh, последняя строка — та самая сводка
# «N passed» — по дороге теряется. Код возврата pytest сохраняем.
ssh "$HOST" "docker run --rm --network host -e PYTHONDONTWRITEBYTECODE=1 \
  -v ~/$DST:/app -v ~/$DATA:/w -v ~/$DATA/hub:/root/.cache/torch/hub -w /app \
  silero_test python -m pytest -p no:cacheprovider ${*:--m 'not slow'} \
    >/tmp/silero_pytest.log 2>&1; rc=\$?; cat /tmp/silero_pytest.log; exit \$rc"
