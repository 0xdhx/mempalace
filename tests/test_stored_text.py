"""NUL and a lone surrogate round-trip as the original characters."""

import pytest

from mempalace.backends.base import PalaceRef
from mempalace.stored_text import FLAG, StoredTextError, prepare_stored_write


def _original():
    return "alpha\x00beta" + chr(0xD800) + "gamma"


def test_prepare_rejects_ids_and_escapes_text():
    original = _original()
    with pytest.raises(StoredTextError):
        prepare_stored_write([original], ["id\x00"], [{}])
    with pytest.raises(StoredTextError):
        prepare_stored_write([original], ["id" + chr(0xD800)], [{}])

    docs, ids, metas = prepare_stored_write([original], ["drawer-1"], [{"wing": "w"}])
    assert ids == ["drawer-1"]
    assert "\x00" not in docs[0]
    assert chr(0xD800) not in docs[0]
    docs[0].encode("utf-8")
    assert metas[0][FLAG] == 1
    assert metas[0]["wing"] == "w"


def test_sqlite_exact_round_trips_nul_and_surrogate(tmp_path):
    from mempalace.backends.sqlite_exact import SQLiteExactBackend

    original = _original()
    col = SQLiteExactBackend().get_collection(
        palace=PalaceRef(id=str(tmp_path), local_path=str(tmp_path)),
        collection_name="mempalace_drawers",
        create=True,
    )
    col.add(
        documents=[original],
        ids=["drawer-1"],
        metadatas=[{"note": "keep\x00me"}],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
    )
    got = col.get(ids=["drawer-1"], include=["documents", "metadatas"])
    assert got.documents == [original]
    assert got.metadatas[0]["note"] == "keep\x00me"
    assert got.metadatas[0][FLAG] == 1


def test_chroma_round_trips_nul_and_surrogate(tmp_path):
    from mempalace.backends.chroma import ChromaBackend

    original = _original()
    col = ChromaBackend().get_collection(
        palace=PalaceRef(id=str(tmp_path), local_path=str(tmp_path)),
        collection_name="mempalace_drawers",
        create=True,
    )
    col.add(
        documents=[original],
        ids=["drawer-1"],
        metadatas=[{"note": "keep\x00me"}],
    )
    got = col.get(ids=["drawer-1"], include=["documents", "metadatas"])
    assert got.documents == [original]
    assert got.metadatas[0]["note"] == "keep\x00me"
    assert got.metadatas[0][FLAG] == 1
