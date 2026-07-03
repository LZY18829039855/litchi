"""Lychee game client entry point.

Usage:
    py -3 client.py <playerId> <host> <port>

Example:
    py -3 client.py 1001 127.0.0.1 30000
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lychee_client.game_loop import GameClient


def main():
    parser = argparse.ArgumentParser(description="Lychee game client")
    parser.add_argument("playerId", type=int, help="Player ID")
    parser.add_argument("host", type=str, help="Server host")
    parser.add_argument("port", type=int, help="Server port")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("client.log", encoding="utf-8"),
        ]
    )

    client = GameClient(args.host, args.port, args.playerId, f"client-{args.playerId}")
    try:
        client.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error("Client error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()