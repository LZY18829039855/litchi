"""Transport layer: 5-digit length-prefixed TCP frame encoding/decoding."""

import json
import logging

logger = logging.getLogger("lychee_client.transport")

MAX_FRAME_BODY_LEN = 99999
_JSON_DECODER = json.JSONDecoder()


def encode_frame(body: dict) -> bytes:
    """Encode a JSON body into a 5-digit length-prefixed TCP frame.

    Frame format: 5 ASCII digits (UTF-8 byte count of body) + UTF-8 JSON body.
    """
    body_bytes = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body_bytes) > MAX_FRAME_BODY_LEN:
        raise ValueError(f"Frame body too large: {len(body_bytes)} bytes")
    prefix = f"{len(body_bytes):05d}".encode("ascii")
    return prefix + body_bytes


def _raw_decode_json_frame(data: bytes, start: int) -> tuple[dict, int] | None:
    """Decode one JSON object from data[start:], returning consumed byte length.

    The official protocol uses a byte-count prefix, but some servers have emitted
    frames whose prefix behaves like a lower bound. Falling back to raw_decode
    lets us recover the full JSON object instead of dropping action result frames.
    """
    try:
        text = data[start:].decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        body, end = _JSON_DECODER.raw_decode(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(body, dict):
        return None
    consumed = len(text[:end].encode("utf-8"))
    return body, consumed


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
        body_start = pos + 5
        body_bytes = data[body_start:body_start + body_len]
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            recovered = _raw_decode_json_frame(data, body_start)
            if recovered is None:
                logger.warning("Failed to decode frame body at %d: %s", pos, exc)
                break
            body, consumed = recovered
            logger.warning(
                "Recovered UTF-8 frame at %d using raw JSON length %d (prefix=%d)",
                pos, consumed, body_len,
            )
            messages.append(body)
            pos = body_start + consumed
            continue
        except json.JSONDecodeError as exc:
            recovered = _raw_decode_json_frame(data, body_start)
            if recovered is not None:
                body, consumed = recovered
                logger.warning(
                    "Recovered JSON frame at %d using raw JSON length %d (prefix=%d): %s",
                    pos, consumed, body_len, exc,
                )
                messages.append(body)
                prefix_end = body_start + body_len
                raw_end = body_start + consumed
                if consumed < body_len and data[raw_end:prefix_end].strip() == b"":
                    pos = prefix_end
                else:
                    pos = raw_end
                continue
            # A prefix that is too short often fails at the end of the slice; keep
            # the buffered bytes and wait for more data instead of dropping it.
            decoded = body_bytes.decode("utf-8", errors="ignore")
            near_end = exc.pos >= max(0, len(decoded) - 2)
            if near_end:
                logger.warning("Incomplete frame body at %d, waiting for more data: %s", pos, exc)
                break
            logger.warning("Failed to decode frame body at %d: %s", pos, exc)
            pos += 5 + body_len
            continue
        messages.append(body)
        pos += 5 + body_len
    remaining = data[pos:]
    return messages, remaining
