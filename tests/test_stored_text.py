"""NUL, lone surrogates, and escape-looking text round-trip unchanged."""

import pytest

from mempalace.backends.base import PalaceRef
from mempalace.stored_text import (
    FLAG,
    StoredTextError,
    decode_record,
    encode_record,
    prepare_stored_write,
)

PREFIX = "\x01mpesc:"
BS = chr(92)
ORIGINAL = "alpha\x00beta" + chr(0xD800) + "gamma " + BS + "0 " + BS + BS
# Already looks like an escape and holds no bad character.
LOOKALIKE = PREFIX + "path C:" + BS + "0dir " + BS + BS + "share"


@pytest.mark.parametrize("text", [ORIGINAL, LOOKALIKE, "plain", "", BS + "0"])
def test_encode_then_decode_returns_the_original(text):
    stored, meta = encode_record(text, {"note": text, "tags": [text, 3]})
    assert "\x00" not in stored
    stored.encode("utf-8")
    assert decode_record(stored, meta) == (text, {"note": text, "tags": [text, 3]})


def test_clean_records_are_stored_as_given():
    assert encode_record("plain", None) == ("plain", None)
    assert encode_record("plain", {"wing": "w"}) == ("plain", {"wing": "w"})


def test_an_unflagged_record_is_never_decoded():
    assert decode_record(LOOKALIKE, {"wing": "w"}) == (LOOKALIKE, {"wing": "w"})


def test_the_flag_is_reserved_and_hidden():
    with pytest.raises(StoredTextError, match="reserved"):
        encode_record("x", {FLAG: "mine"})
    stored, meta = encode_record(ORIGINAL, {"wing": "w"})
    assert meta[FLAG] == 1
    assert FLAG not in decode_record(stored, meta)[1]


def test_stored_form_is_copied_verbatim():
    stored, meta = encode_record(ORIGINAL, {"note": LOOKALIKE})
    assert encode_record(stored, meta) == (stored, meta)


def test_ids_and_metadata_keys_are_rejected_not_rewritten():
    for bad in ("id\x00", "id" + chr(0xD800), ""):
        with pytest.raises(StoredTextError):
            prepare_stored_write(["x"], [bad], [{}])
    with pytest.raises(StoredTextError):
        prepare_stored_write(["x"], ["ok"], [{"k\x00": "v"}])


def test_batch_without_metadata_flags_only_when_needed():
    assert prepare_stored_write(["a", "b"], ["1", "2"]) == (["a", "b"], ["1", "2"], None)
    docs, _, metas = prepare_stored_write(["a", "b\x00"], ["1", "2"])
    assert metas == [{FLAG: 1}, {FLAG: 1}]
    assert decode_record(docs[1], metas[1])[0] == "b\x00"
    assert prepare_stored_write(None, ["1"], [{"n": "x\x00"}])[0] is None


def _sqlite_exact(tmp_path):
    from mempalace.backends.sqlite_exact import SQLiteExactBackend

    return SQLiteExactBackend().get_collection(
        palace=PalaceRef(id=str(tmp_path), local_path=str(tmp_path)),
        collection_name="mempalace_drawers",
        create=True,
    )


def _chroma(tmp_path):
    from mempalace.backends.chroma import ChromaBackend

    return ChromaBackend().get_collection(
        palace=PalaceRef(id=str(tmp_path), local_path=str(tmp_path)),
        collection_name="mempalace_drawers",
        create=True,
    )


def _seed(col, vectors):
    kw = {"embeddings": [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]} if vectors else {}
    col.add(
        documents=[ORIGINAL + " kubernetes rollout", LOOKALIKE + " kubernetes notes"],
        ids=["a", "b"],
        metadatas=[{"wing": "w", "note": "keep\x00me"}, {"wing": "w", "note": LOOKALIKE}],
        **kw,
    )


def _check_reads(col, vectors):
    want = {
        "a": (ORIGINAL + " kubernetes rollout", "keep\x00me"),
        "b": (LOOKALIKE + " kubernetes notes", LOOKALIKE),
    }
    got = col.get(ids=["a", "b"], include=["documents", "metadatas"])
    assert {i: (d, m["note"]) for i, d, m in zip(got.ids, got.documents, got.metadatas)} == want
    assert all(FLAG not in m for m in got.metadatas)
    # documents alone still decode
    assert col.get(ids=["a"], include=["documents"]).documents == [want["a"][0]]
    hits = col.lexical_search(query="kubernetes", n_results=5).hits
    assert {h.id: (h.document, h.metadata["note"]) for h in hits} == want
    q = {"query_embeddings": [[0.1, 0.2, 0.3, 0.4]]} if vectors else {"query_texts": ["notes"]}
    res = col.query(n_results=2, include=["documents"], **q)
    assert sorted(res.documents[0]) == sorted(d for d, _ in want.values())


def _check_updates(col, vectors):
    kw = {"embeddings": [[0.2, 0.2, 0.2, 0.2]]} if vectors else {}
    col.update(ids=["a"], documents=["x\x00y"], **kw)
    col.update(ids=["b"], metadatas=[{"extra": "s" + chr(0xD800)}])
    got = col.get(ids=["a", "b"], include=["documents", "metadatas"])
    assert got.documents == ["x\x00y", LOOKALIKE + " kubernetes notes"]
    assert got.metadatas[1]["extra"] == "s" + chr(0xD800)
    assert got.metadatas[1]["note"] == LOOKALIKE


def test_sqlite_exact_round_trips_every_read_and_update(tmp_path):
    col = _sqlite_exact(tmp_path)
    _seed(col, vectors=True)
    _check_reads(col, vectors=True)
    _check_updates(col, vectors=True)


def test_chroma_round_trips_every_read_and_update(tmp_path):
    col = _chroma(tmp_path)
    _seed(col, vectors=False)
    _check_reads(col, vectors=False)
    _check_updates(col, vectors=False)


def test_chroma_sqlite_readers_decode(tmp_path):
    from mempalace.backends.chroma import (
        sqlite_diary_rows,
        sqlite_documents_for_ids,
        sqlite_drawer_rows,
    )

    col = _chroma(tmp_path)
    col.add(
        documents=[ORIGINAL, "plain entry"],
        ids=["d1", "d2"],
        metadatas=[
            {"room": "diary", "agent": "a", "wing": "w", "filed_at": "2026-01-02", "t": LOOKALIKE},
            {"room": "diary", "agent": "a", "wing": "w", "filed_at": "2026-01-01"},
        ],
    )
    path = str(tmp_path)
    assert sqlite_documents_for_ids(path, "mempalace_drawers", ["d1", "d2"]) == {
        "d1": ORIGINAL,
        "d2": "plain entry",
    }
    direct, rows = sqlite_drawer_rows(path, "mempalace_drawers", drawer_id="d1")
    assert direct and rows[0][1] == ORIGINAL and rows[0][2]["t"] == LOOKALIKE
    assert FLAG not in rows[0][2]
    total, rows = sqlite_diary_rows(path, "mempalace_drawers", agent_name="a", limit=5)
    assert total == 2
    assert [row[1] for row in rows] == [ORIGINAL, "plain entry"]
    meta = col.get_recent(limit=1, include=["documents", "metadatas"])
    assert meta.documents == [ORIGINAL] and FLAG not in meta.metadatas[0]
    assert all(FLAG not in m for m in col.get_all_metadata())


def test_bm25_fallback_decodes(tmp_path):
    from mempalace.searcher import _bm25_only_via_sqlite

    _seed(_chroma(tmp_path), vectors=False)
    out = _bm25_only_via_sqlite(
        "kubernetes", str(tmp_path), collection_name="mempalace_drawers", n_results=5
    )
    assert sorted(r["text"] for r in out["results"]) == sorted(
        [ORIGINAL + " kubernetes rollout", LOOKALIKE + " kubernetes notes"]
    )
