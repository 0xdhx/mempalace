"""A version-bump re-mine embeds only the text that changed.

Bumping NORMALIZE_VERSION makes every file read as not mined, so the next
mine re-files all of them. Drawers whose text comes out the same keep the
vector already stored, so that pass costs reads, not embeddings.
"""

import pytest

import mempalace.backends.embedding_wrapper as embedding_wrapper
from mempalace.palace import NORMALIZE_VERSION, file_already_mined, get_collection


def _write_chat(path, turns):
    body = "\n\n".join(f"> question {i} " + "asked " * 30 + "\n" + "answer " * 60 for i in turns)
    path.write_text(body + "\n", encoding="utf-8")


@pytest.fixture
def embedded(monkeypatch):
    texts = []
    real = embedding_wrapper._embed_texts

    def counting(batch, **kwargs):
        if not kwargs.get("query"):
            texts.extend(batch)
        return real(batch, **kwargs)

    monkeypatch.setattr(embedding_wrapper, "_embed_texts", counting)
    return texts


def _mine(src, palace):
    from mempalace import convo_miner

    convo_miner.mine_convos(str(src), palace, wing="chats", extract_mode="exchange")


def _age_stamps(palace, **extra):
    """Mark every drawer as filed by the previous NORMALIZE_VERSION."""
    col = get_collection(palace)
    got = col.get(include=["metadatas"])
    col.update(
        ids=got["ids"],
        metadatas=[
            {**meta, "normalize_version": NORMALIZE_VERSION - 1, **extra}
            for meta in got["metadatas"]
        ],
    )


def _palace_rows(palace):
    got = get_collection(palace).get(include=["documents", "metadatas", "embeddings"])
    return {
        drawer_id: (doc, meta, list(vector))
        for drawer_id, doc, meta, vector in zip(
            got["ids"], got["documents"], got["metadatas"], got["embeddings"]
        )
    }


def test_version_bump_remine_of_unchanged_conversations_embeds_nothing(tmp_path, embedded):
    src = (tmp_path / "chats").resolve()
    src.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        _write_chat(src / name, range(3))
    palace = str(tmp_path / "palace")
    _mine(src, palace)
    first = _palace_rows(palace)
    assert first and len(embedded) == len(first)

    _age_stamps(palace)
    for name in ("a.txt", "b.txt", "c.txt"):
        assert not file_already_mined(
            get_collection(palace), str(src / name), check_mtime=True, extract_mode="exchange"
        )
    embedded.clear()
    _mine(src, palace)

    assert embedded == []
    second = _palace_rows(palace)
    assert set(second) == set(first)
    for drawer_id, (doc, meta, vector) in second.items():
        assert doc == first[drawer_id][0]
        assert vector == pytest.approx(first[drawer_id][2])
        assert meta["normalize_version"] == NORMALIZE_VERSION
    for name in ("a.txt", "b.txt", "c.txt"):
        assert file_already_mined(
            get_collection(palace), str(src / name), check_mtime=True, extract_mode="exchange"
        )


def test_version_bump_remine_embeds_only_the_chunks_that_changed(tmp_path, embedded):
    src = (tmp_path / "chats").resolve()
    src.mkdir()
    chat = src / "chat.txt"
    _write_chat(chat, range(3))
    palace = str(tmp_path / "palace")
    _mine(src, palace)
    before = _palace_rows(palace)

    _age_stamps(palace)
    _write_chat(chat, range(4))
    embedded.clear()
    _mine(src, palace)

    after = _palace_rows(palace)
    changed = [
        doc for drawer_id, (doc, _, _) in after.items() if before.get(drawer_id, (None,))[0] != doc
    ]
    assert changed
    assert sorted(embedded) == sorted(changed)
    assert len(embedded) < len(after)


def test_reused_vector_still_drops_a_key_the_new_metadata_lacks(tmp_path, embedded):
    src = (tmp_path / "chats").resolve()
    src.mkdir()
    _write_chat(src / "chat.txt", range(2))
    palace = str(tmp_path / "palace")
    _mine(src, palace)

    _age_stamps(palace, stale_key="left over")
    embedded.clear()
    _mine(src, palace)

    assert embedded == []
    for _, meta, _ in _palace_rows(palace).values():
        assert "stale_key" not in meta
