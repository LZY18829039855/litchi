#!/bin/bash
set -e

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <playerId> <host> <port>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTHONUNBUFFERED=1

exec python3 -u "${SCRIPT_DIR}/client.py" "$1" "$2" "$3"