"""Transport layer: 5-digit length-prefixed TCP frame encoding/decoding."""

import json
import logging

logger = logging.getLogger("lychee_client.transport")

MAX_FRAME_BODY_LEN = 99999


def encode_frame(body: dict) -> bytes:
    """Encode a JSON body into a 5-digit length-prefixed TCP frame.

    Frame format: 5 ASCII digits (UTF-8 byte count of body) + UTF-8 JSON body.
    """
    body_bytes = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body_bytes) > MAX_FRAME_BODY_LEN:
        raise ValueError(f"Frame body too large: {len(body_bytes)} bytes")
    prefix = f"{len(body_bytes):05d}".encode("ascii")
    return prefix + body_bytes


def read_frames_from_buffer(data: bytes) -> tuple[list[dict], bytes]:
    """Read complete frames from a byte buffer.

    Returns (messages, remaining_bytes) where messages is a list of parsed
    JSON dicts and remaining_bytes is any incomplete data.
    """
    messages = []
    pos = 0
    while pos + 5 <= len(data):
        prefix_bytes = data[pos:pos + 5]
        try:
            prefix = prefix_bytes.decode("ascii")
        except UnicodeDecodeError:
            logger.warning("Invalid length prefix bytes at %d, resync +1", pos)
            pos += 1
            continue
        if not prefix.isdigit():
            logger.warning("Non-numeric length prefix %r at %d, resync +1", prefix, pos)
            pos += 1
            continue
        body_len = int(prefix)
        if body_len < 0 or body_len > MAX_FRAME_BODY_LEN:
            logger.warning("Length prefix out of range: %d, resync +1", body_len)
            pos += 1
            continue
        if pos + 5 + body_len > len(data):
            break
        body_bytes = data[pos + 5:pos + 5 + body_len]
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("Failed to decode frame body at %d: %s", pos, exc)
            pos += 5 + body_len
            continue
        messages.append(body)
        pos += 5 + body_len
    remaining = data[pos:]
    return messages, remaining
