"""Spell correction of user turns is opt-in and stamped (W1-09, D-01)."""

import json
from unittest.mock import patch

import chromadb
import pytest

from mempalace.config import MempalaceConfig
from mempalace.convo_miner import mine_convos

# A chat export, so the turns go through the importer that used to correct
# them by default, not only through the plain-text pass-through.
EXPORT = json.dumps(
    {
        "messages": [
            {"role": "user", "content": "i knoe the question befor"},
            {"role": "assistant", "content": "That is fine."},
            {"role": "user", "content": "why does it matter"},
            {"role": "assistant", "content": "It enables continuity."},
        ]
    }
)


class _Speller:
    def __call__(self, word):
        return {"knoe": "know", "befor": "before"}.get(word, word)


@pytest.fixture
def speller():
    """Pretend autocorrect is installed, whatever this machine has."""
    with (
        patch("mempalace.spellcheck._get_speller", return_value=_Speller()),
        patch("mempalace.spellcheck._get_system_words", return_value=set()),
        patch("mempalace.spellcheck._load_known_names", return_value=set()),
    ):
        yield


def _mine(tmp_path):
    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    (convo_dir / "chat.json").write_text(EXPORT, encoding="utf-8")
    palace_path = str(tmp_path / "palace")
    mine_convos(str(convo_dir), palace_path, wing="test")
    col = chromadb.PersistentClient(path=palace_path).get_collection("mempalace_drawers")
    rows = col.get(include=["documents", "metadatas"])
    return rows["documents"], rows["metadatas"]


def test_spellcheck_is_off_by_default(tmp_path, monkeypatch, speller):
    monkeypatch.delenv("MEMPALACE_CONVO_SPELLCHECK", raising=False)
    docs, metas = _mine(tmp_path)
    assert any("i knoe the question befor" in d for d in docs)
    assert not any("know the question before" in d for d in docs)
    assert all("transforms" not in m for m in metas)


def test_opted_in_spellcheck_stamps_the_drawers_it_changed(tmp_path, monkeypatch, speller):
    monkeypatch.setenv("MEMPALACE_CONVO_SPELLCHECK", "1")
    docs, metas = _mine(tmp_path)
    assert any("i know the question before" in d for d in docs)
    assert metas and all(m.get("transforms") == "spellcheck" for m in metas)


def test_convo_spellcheck_setting(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMPALACE_CONVO_SPELLCHECK", raising=False)
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.convo_spellcheck is False

    cfg._file_config = {"convo": {"spellcheck": True}}
    assert cfg.convo_spellcheck is True

    monkeypatch.setenv("MEMPALACE_CONVO_SPELLCHECK", "off")
    assert cfg.convo_spellcheck is False
