"""Backend isolation conformance suite (RFC 001 isolation contract).

Runs the shared isolation assertions from ``_backend_conformance`` against the
built-in local backends. Server-mode backends (qdrant) run the same assertions
under their own fake/live client in ``test_qdrant_backend.py``.
"""

import pytest

from _backend_conformance import assert_partition_isolation, assert_update_skips_missing_ids

from mempalace.backends import PalaceRef
from mempalace.backends.chroma import ChromaBackend
from mempalace.backends.sqlite_exact import SQLiteExactBackend
from mempalace.backends.rust_exact import RustExactBackend

_LOCAL_BACKENDS = [
    pytest.param(ChromaBackend, id="chroma"),
    pytest.param(SQLiteExactBackend, id="sqlite_exact"),
    pytest.param(RustExactBackend, id="rust_exact"),
]


@pytest.mark.parametrize("backend_cls", _LOCAL_BACKENDS)
def test_cross_palace_isolation(backend_cls, tmp_path):
    """One backend instance must isolate two distinct palaces (PalaceRef.id)."""
    backend = backend_cls()
    try:
        cols = []
        for label in ("alpha", "beta"):
            path = tmp_path / label
            ref = PalaceRef(id=str(path), local_path=str(path))
            cols.append(
                backend.get_collection(palace=ref, collection_name="mempalace_drawers", create=True)
            )
        assert_partition_isolation(backend, cols[0], cols[1])
    finally:
        backend.close()


@pytest.mark.parametrize("backend_cls", _LOCAL_BACKENDS)
def test_update_skips_missing_ids(backend_cls, tmp_path):
    """``update`` never creates a record for an id that does not exist."""
    backend = backend_cls()
    try:
        ref = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
        col = backend.get_collection(palace=ref, collection_name="mempalace_drawers", create=True)
        assert_update_skips_missing_ids(backend, col)
    finally:
        backend.close()


@pytest.mark.parametrize("backend_cls", _LOCAL_BACKENDS)
def test_default_update_skips_missing_ids(backend_cls, tmp_path, monkeypatch):
    """The ``BaseCollection.update`` default, which pgvector, qdrant, milvus and
    third-party backends inherit, holds the same contract as the overrides."""
    from mempalace.backends.base import BaseCollection

    backend = backend_cls()
    try:
        ref = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
        col = backend.get_collection(palace=ref, collection_name="mempalace_drawers", create=True)
        monkeypatch.setattr(type(col), "update", BaseCollection.update)
        assert_update_skips_missing_ids(backend, col)
    finally:
        backend.close()


def test_local_backends_do_not_claim_namespace_isolation():
    """Local backends isolate by on-disk path, not namespace; they must not
    advertise the namespace-isolation capability (RFC 001 isolation contract)."""
    assert "supports_namespace_isolation" not in ChromaBackend.capabilities
    assert "supports_namespace_isolation" not in SQLiteExactBackend.capabilities
    assert "supports_namespace_isolation" not in RustExactBackend.capabilities


@pytest.mark.parametrize("backend_cls", _LOCAL_BACKENDS)
def test_local_backends_reject_populated_namespace(backend_cls, tmp_path):
    """RFC 001 section 4.4 no-silent-drop: non-advertising backends MUST raise when
    PalaceRef.namespace is set rather than accept-and-ignore."""
    from mempalace.backends.base import UnsupportedCapabilityError

    backend = backend_cls()
    try:
        path = tmp_path / "palace"
        path.mkdir()
        ref = PalaceRef(id=str(path), local_path=str(path), namespace="tenant-a")
        with pytest.raises(UnsupportedCapabilityError):
            backend.get_collection(palace=ref, collection_name="mempalace_drawers", create=True)
    finally:
        backend.close()
