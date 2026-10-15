"""Remote-backend metadata must survive one bad value, and markers must not tear."""

import json
from datetime import date

import pytest

from mempalace.backends.base import PalaceRef
from mempalace.backends.milvus import MilvusBackend, _MilvusConfig
from mempalace.backends.milvus import _jsonable_metadata as milvus_meta
from mempalace.backends.pgvector import PgVectorBackend, _PgVectorConfig
from mempalace.backends.pgvector import _jsonable_metadata as pg_meta
from mempalace.backends.qdrant import QdrantBackend, _QdrantConfig
from mempalace.backends.qdrant import _jsonable_metadata as qdrant_meta

_META = {
    "wing": "w",
    "room": "r",
    "source_file": "a.md",
    "when": date(2026, 1, 1),
}


def test_one_unserializable_value_does_not_wipe_metadata():
    for coerce in (qdrant_meta, pg_meta, milvus_meta):
        cleaned = coerce(dict(_META))
        assert cleaned["wing"] == "w"
        assert cleaned["room"] == "r"
        assert cleaned["source_file"] == "a.md"
        assert cleaned["when"] == "2026-01-01"
        assert cleaned != {}


def test_marker_rewrite_keeps_created_at_and_skips_an_unchanged_target(tmp_path):
    backend = QdrantBackend()
    ref = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    cfg = _QdrantConfig(url="http://127.0.0.1:6333", namespace="alpha")
    backend._write_marker(ref, cfg)
    path = tmp_path / "qdrant_backend.json"
    first_text = path.read_text(encoding="utf-8")
    created = json.loads(first_text)["created_at"]

    backend._write_marker(ref, cfg)
    assert path.read_text(encoding="utf-8") == first_text

    changed = _QdrantConfig(url="http://127.0.0.1:6333", namespace="beta")
    backend._write_marker(ref, changed)
    second = backend._read_marker(ref)
    assert second["created_at"] == created
    assert second["qdrant"]["namespace"] == "beta"


def test_crash_mid_marker_write_leaves_a_readable_marker(tmp_path, monkeypatch):
    backend = PgVectorBackend()
    ref = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    cfg = _PgVectorConfig(dsn="postgresql://db.example/palace", namespace="alpha")
    backend._write_marker(ref, cfg)
    path = tmp_path / "pgvector_backend.json"
    original = path.read_bytes()

    def killed(payload, handle, **kwargs):
        handle.write("{")
        handle.flush()
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(json, "dump", killed)
    changed = _PgVectorConfig(dsn="postgresql://db.example/palace", namespace="beta")
    with pytest.raises(RuntimeError, match="killed mid-write"):
        backend._write_marker(ref, changed)

    assert path.read_bytes() == original
    marker = backend._read_marker(ref)
    assert marker["pgvector"]["namespace"] == "alpha"


def test_milvus_marker_publish_is_atomic(tmp_path, monkeypatch):
    backend = MilvusBackend()
    ref = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    cfg = _MilvusConfig(uri="http://127.0.0.1:19530", namespace="alpha")
    backend._write_marker(ref, cfg)
    path = tmp_path / "milvus_backend.json"
    original = path.read_bytes()

    def killed(payload, handle, **kwargs):
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(json, "dump", killed)
    with pytest.raises(RuntimeError, match="killed mid-write"):
        backend._write_marker(ref, _MilvusConfig(uri="http://127.0.0.1:19530", namespace="beta"))

    assert path.read_bytes() == original
    assert backend._read_marker(ref)["milvus"]["namespace"] == "alpha"
