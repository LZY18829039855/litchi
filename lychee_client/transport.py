"""Transport layer: 5-digit length-prefixed TCP frame encoding/decoding."""

import json


def encode_frame(body: dict) -> bytes:
    """Encode a JSON body into a 5-digit length-prefixed TCP frame.

    Frame format: 5 ASCII digits (UTF-8 byte count of body) + UTF-8 JSON body.
    """
    body_bytes = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
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
        prefix = data[pos:pos + 5].decode("ascii")
        body_len = int(prefix)
        if pos + 5 + body_len > len(data):
            # Incomplete frame
            break
        body_bytes = data[pos + 5:pos + 5 + body_len]
        body = json.loads(body_bytes.decode("utf-8"))
        messages.append(body)
        pos += 5 + body_len
    remaining = data[pos:]
    return messages, remaining