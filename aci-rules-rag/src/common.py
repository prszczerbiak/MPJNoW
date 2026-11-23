"""Atomowy zapis plików, sumy kontrolne i ochrona zatwierdzonych danych."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import yaml


class ProtocolError(ValueError):
    """Naruszenie wymagań dotyczących danych lub protokołu."""


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def immutable_json(path, value):
    """Dopuszcza ponowny zapis tej samej treści; odrzuca zmiany. Wymaga jednego procesu zapisującego."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ProtocolError(f"Refusing to replace frozen artifact: {path}")
    else:
        write_json(path, value)


def require_revision(value):
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ProtocolError("An immutable 40-character commit revision is required")
