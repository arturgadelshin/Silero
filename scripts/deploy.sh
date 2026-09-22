#!/bin/sh
# deploy.sh — код с рабочей машины на сервер (rsync там, tar по дороге:
# в Git Bash rsync нет). Данные в ~/linkplay/tmp/silero не трогаются.
#
#   scripts/deploy.sh            только код
#   scripts/deploy.sh --restart  и перезапустить сервис
set -eu
HOST=${SILERO_HOST:-green@192.168.0.121}
DST=${SILERO_CODE_DIR:-linkplay/silero}
cd "$(dirname "$0")/.."
tar -cz --exclude=.git --exclude=listen --exclude=__pycache__ --exclude=.pytest_cache --exclude="*.pyc" . |
  ssh "$HOST" "rm -rf ~/$DST.new && mkdir -p ~/$DST.new ~/$DST && tar -xz -C ~/$DST.new \
    && rsync -a --delete ~/$DST.new/ ~/$DST/ && rm -rf ~/$DST.new"
echo "код доставлен: $HOST:~/$DST"
if [ "${1:-}" = "--restart" ]; then
  exec "$(dirname "$0")/service.sh" restart
fi
