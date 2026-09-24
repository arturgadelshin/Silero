#!/bin/sh
# web.sh — управление веб-мордой на сервере (зеркало service.sh).
#   start | stop | restart | logs | status | build
set -eu
HOST=${SILERO_HOST:-green@192.168.0.121}
DST=${SILERO_CODE_DIR:-linkplay/silero}
case "${1:-status}" in
  start)   ssh "$HOST" "cd ~/$DST/web && docker compose up -d" ;;
  stop)    ssh "$HOST" "cd ~/$DST/web && docker compose down" ;;
  restart) ssh "$HOST" "cd ~/$DST/web && docker compose up -d --build --force-recreate" ;;
  build)   ssh "$HOST" "cd ~/$DST/web && docker compose build" ;;
  logs)    ssh "$HOST" "docker logs --tail ${2:-40} silero_web" ;;
  status)  ssh "$HOST" "docker ps --filter name=silero_web --format '{{.Status}}'; \
                        curl -s --max-time 5 localhost:10580/api/health; echo" ;;
  *) echo "start | stop | restart | build | logs [строк] | status" >&2; exit 1 ;;
esac
