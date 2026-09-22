#!/bin/sh
# service.sh — управление TTS на сервере.
#   start | stop | restart | logs | status
set -eu
HOST=${SILERO_HOST:-green@192.168.0.121}
DST=${SILERO_CODE_DIR:-linkplay/silero}
case "${1:-status}" in
  start)   ssh "$HOST" "cd ~/$DST/service && docker compose up -d" ;;
  stop)    ssh "$HOST" "cd ~/$DST/service && docker compose down" ;;
  restart) ssh "$HOST" "cd ~/$DST/service && docker compose up -d --force-recreate" ;;
  logs)    ssh "$HOST" "docker logs --tail ${2:-40} silero_tts" ;;
  status)  ssh "$HOST" "docker ps --filter name=silero_tts --format '{{.Status}}'; \
                        curl -s --max-time 5 localhost:10500/health; echo" ;;
  *) echo "start | stop | restart | logs [строк] | status" >&2; exit 1 ;;
esac
