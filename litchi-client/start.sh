#!/bin/bash
set -e

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <playerId> <host> <port>"
  exit 1
fi

PLAYER_ID="$1"
HOST="$2"
PORT="$3"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "litchi-client.jar" ]; then
  exec java -jar litchi-client.jar "$PLAYER_ID" "$HOST" "$PORT"
fi

if [ -f "target/litchi-client.jar" ]; then
  exec java -jar target/litchi-client.jar "$PLAYER_ID" "$HOST" "$PORT"
fi

echo "未找到 litchi-client.jar，请先执行: mvn -q package"
exit 1
