"""Re-mines on a real palace: what each miner leaves behind for one file.

Pins the observable contract the miners share, so the write path under them
can change without changing it: a re-mine replaces the file's drawers in its
own scope, leaves other scopes alone, and a failure part-way never leaves the
file with no drawers at all.
"""

import os

import pytest

from mempalace.palace import MineFileErrors, file_already_mined, get_collection


def _paragraphs(prefix: str, count: int) -> str:
    return "\n\n".join(f"{prefix} paragraph {i}. " + "verbatim words " * 40 for i in range(count))


def _touch(path):
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + 10))


def _rows(palace, source_file):
    got = get_collection(palace).get(
        where={"source_file": str(source_file)}, include=["documents", "metadatas"]
    )
    return list(zip(got["documents"], got["metadatas"]))


class _Project:
    def __init__(self, tmp_path):
        from mempalace import miner

        self.miner = miner
        self.src = (tmp_path / "proj").resolve()
        self.src.mkdir()
        self.doc = self.src / "notes.md"
        self.palace = str(tmp_path / "palace")

    def write(self, text):
        self.doc.write_text(text, encoding="utf-8")

    def mine(self):
        self.miner.mine(str(self.src), self.palace, wing_override="proj", agent="test")

    def rows(self):
        return _rows(self.palace, self.doc)


class _Convo:
    def __init__(self, tmp_path):
        from mempalace import convo_miner

        self.convo_miner = convo_miner
        self.src = (tmp_path / "chats").resolve()
        self.src.mkdir()
        self.doc = self.src / "chat.txt"
        self.palace = str(tmp_path / "palace")

    def write(self, turns):
        body = "\n\n".join(
            f"> question {i} " + "asked " * 30 + "\n" + "answer " * 60 for i in turns
        )
        self.doc.write_text(body + "\n", encoding="utf-8")

    def mine(self, extract_mode="exchange"):
        self.convo_miner.mine_convos(
            str(self.src), self.palace, wing="chats", extract_mode=extract_mode
        )

    def rows(self, extract_mode="exchange"):
        return [
            (doc, meta)
            for doc, meta in _rows(self.palace, self.doc)
            if meta.get("extract_mode") == extract_mode
        ]


def _fail_upserts(monkeypatch, palace):
    def fail(self, *args, **kwargs):
        raise RuntimeError("killed during the write")

    monkeypatch.setattr(type(get_collection(palace)), "upsert", fail)


def _fail_deletes(monkeypatch, palace):
    def fail(self, *args, **kwargs):
        raise RuntimeError("killed before the stale delete")

    monkeypatch.setattr(type(get_collection(palace)), "delete", fail)


# -- project miner ------------------------------------------------------------


def test_project_remine_of_a_shorter_file_leaves_only_its_new_drawers(tmp_path):
    project = _Project(tmp_path)
    project.write(_paragraphs("long", 8))
    project.mine()
    assert len(project.rows()) > 2

    project.write(_paragraphs("short", 1))
    _touch(project.doc)
    project.mine()

    rows = project.rows()
    assert len(rows) == 1
    assert rows[0][0].startswith("short paragraph 0.")
    assert rows[0][1]["chunk_total"] == 1


def test_project_remine_of_a_longer_file_files_every_chunk(tmp_path):
    project = _Project(tmp_path)
    project.write(_paragraphs("short", 1))
    project.mine()

    project.write(_paragraphs("long", 8))
    _touch(project.doc)
    project.mine()

    rows = project.rows()
    assert len(rows) > 2
    assert all("long paragraph" in doc for doc, _ in rows)
    assert {meta["chunk_total"] for _, meta in rows} == {len(rows)}
    assert file_already_mined(get_collection(project.palace), str(project.doc), check_mtime=True)


def test_project_remine_of_a_file_cut_below_the_minimum_keeps_its_drawers(tmp_path):
    project = _Project(tmp_path)
    project.write(_paragraphs("long", 3))
    project.mine()
    before = sorted(doc for doc, _ in project.rows())

    project.write("tiny")
    _touch(project.doc)
    project.mine()

    assert sorted(doc for doc, _ in project.rows()) == before


def test_project_remine_replaces_every_drawer_for_the_path(tmp_path):
    """The project miner's purge scope is the whole path, whatever filed it."""
    project = _Project(tmp_path)
    project.write(_paragraphs("long", 2))
    project.mine()
    get_collection(project.palace).upsert(
        ids=["other_scope"],
        documents=["filed by another miner"],
        metadatas=[{"source_file": str(project.doc), "wing": "elsewhere", "chunk_index": 0}],
    )

    project.write(_paragraphs("new", 2))
    _touch(project.doc)
    project.mine()

    docs = [doc for doc, _ in project.rows()]
    assert "filed by another miner" not in docs
    assert all("new paragraph" in doc for doc in docs)


def test_project_failed_write_keeps_the_previous_drawers(tmp_path, monkeypatch):
    project = _Project(tmp_path)
    project.write(_paragraphs("old", 3))
    project.mine()
    before = sorted(doc for doc, _ in project.rows())

    project.write(_paragraphs("new", 3))
    _touch(project.doc)
    _fail_upserts(monkeypatch, project.palace)
    with pytest.raises(RuntimeError, match="killed during the write"):
        project.mine()

    assert sorted(doc for doc, _ in project.rows()) == before


def test_project_failed_stale_delete_keeps_text_and_converges(tmp_path, monkeypatch):
    project = _Project(tmp_path)
    project.write(_paragraphs("old", 3))
    project.mine()

    project.write(_paragraphs("new", 1))
    _touch(project.doc)
    real_delete = type(get_collection(project.palace)).delete
    _fail_deletes(monkeypatch, project.palace)
    with pytest.raises(MineFileErrors):
        project.mine()
    assert project.rows(), "the file was left with no drawers"
    col = get_collection(project.palace)
    assert not file_already_mined(col, str(project.doc), check_mtime=True)

    monkeypatch.setattr(type(col), "delete", real_delete)
    project.mine()
    rows = project.rows()
    assert len(rows) == 1
    assert rows[0][0].startswith("new paragraph 0.")


# -- conversation miner -------------------------------------------------------


def test_convo_remine_of_a_shorter_transcript_leaves_only_its_new_drawers(tmp_path):
    convo = _Convo(tmp_path)
    convo.write(range(6))
    convo.mine()
    assert len(convo.rows()) > 2

    convo.write([0])
    _touch(convo.doc)
    convo.mine()

    rows = convo.rows()
    assert len(rows) == 1
    assert "question 0" in rows[0][0]
    assert rows[0][1]["chunk_total"] == 1


def test_convo_remine_of_a_longer_transcript_files_every_exchange(tmp_path):
    convo = _Convo(tmp_path)
    convo.write([0])
    convo.mine()

    convo.write(range(6))
    _touch(convo.doc)
    convo.mine()

    rows = convo.rows()
    assert len(rows) == 6
    assert {meta["chunk_total"] for _, meta in rows} == {6}
    col = get_collection(convo.palace)
    assert file_already_mined(col, str(convo.doc), check_mtime=True, extract_mode="exchange")


def test_convo_remine_leaves_the_other_extract_mode_alone(tmp_path):
    convo = _Convo(tmp_path)
    convo.write(range(3))
    convo.mine(extract_mode="general")
    general = sorted(doc for doc, _ in convo.rows("general"))
    assert general

    convo.mine()
    convo.write([0])
    _touch(convo.doc)
    convo.mine()

    assert len(convo.rows()) == 1
    assert sorted(doc for doc, _ in convo.rows("general")) == general


def test_convo_failed_write_keeps_the_previous_drawers(tmp_path, monkeypatch):
    convo = _Convo(tmp_path)
    convo.write(range(3))
    convo.mine()
    before = sorted(doc for doc, _ in convo.rows())

    convo.write(range(5))
    _touch(convo.doc)
    _fail_upserts(monkeypatch, convo.palace)
    with pytest.raises(MineFileErrors):
        convo.mine()

    assert sorted(doc for doc, _ in convo.rows()) == before


def test_convo_failed_stale_delete_keeps_text_and_converges(tmp_path, monkeypatch):
    convo = _Convo(tmp_path)
    convo.write(range(4))
    convo.mine()

    convo.write([0])
    _touch(convo.doc)
    real_delete = type(get_collection(convo.palace)).delete
    _fail_deletes(monkeypatch, convo.palace)
    with pytest.raises(MineFileErrors):
        convo.mine()
    assert convo.rows(), "the transcript was left with no drawers"
    col = get_collection(convo.palace)
    assert not file_already_mined(col, str(convo.doc), check_mtime=True, extract_mode="exchange")

    monkeypatch.setattr(type(col), "delete", real_delete)
    convo.mine()
    rows = convo.rows()
    assert len(rows) == 1
    assert "question 0" in rows[0][0]
