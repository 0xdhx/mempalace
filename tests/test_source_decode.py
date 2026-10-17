"""Source bytes reach drawers without U+FFFD: UTF-8, then cp1252, then latin-1.

Every ingest path decodes through ``fsio.decode_source``. A source that is not
plain UTF-8 carries ``source_encoding`` on its drawers (RFC 002 section 5.6),
so its original bytes can be recovered with ``text.encode(source_encoding)``.
"""

import json

import pytest

from mempalace.fsio import decode_source, encoding_metadata, read_source_text
from mempalace.palace import get_collection

SMART_QUOTES_CP1252 = b"\x93quoted\x94 and \x91single\x92"
SMART_QUOTES_TEXT = "“quoted” and ‘single’"


@pytest.mark.parametrize(
    ("raw", "text", "encoding"),
    [
        (b"", "", "utf-8"),
        ("café ✨".encode("utf-8"), "café ✨", "utf-8"),
        (b"\xef\xbb\xbfwith a BOM", "with a BOM", "utf-8-sig"),
        (SMART_QUOTES_CP1252, SMART_QUOTES_TEXT, "cp1252"),
        # 0x81 has no cp1252 character, so only latin-1 takes it.
        (b"a\x81b \xe9", "a\x81b é", "latin-1"),
    ],
    ids=["empty", "utf-8", "bom", "cp1252", "latin-1"],
)
def test_decode_source_picks_the_first_codec_that_fits(raw, text, encoding):
    assert decode_source(raw) == (text, encoding)


@pytest.mark.parametrize(
    "raw",
    [
        SMART_QUOTES_CP1252,
        b"a\x81b \xe9",
        b"\xef\xbb\xbfplain",
        # Mostly UTF-8 with one invalid byte in the middle.
        "naïve résumé ".encode("utf-8") * 3 + b"\xff" + b" tail",
        bytes(range(256)),
    ],
    ids=["cp1252", "latin-1", "bom", "invalid-utf-8", "every-byte"],
)
def test_decode_source_never_replaces_a_byte_and_round_trips(raw):
    text, encoding = decode_source(raw)
    assert "�" not in text
    assert text.encode(encoding) == raw


def test_decode_source_reads_newlines_like_text_mode_only_when_asked():
    assert decode_source(b"a\r\nb\rc")[0] == "a\r\nb\rc"
    assert decode_source(b"a\r\nb\rc", universal_newlines=True)[0] == "a\nb\nc"


def test_encoding_metadata_names_only_non_utf8_sources():
    assert encoding_metadata("utf-8") == {}
    assert encoding_metadata("cp1252") == {"source_encoding": "cp1252"}


def test_read_source_text_returns_the_fstat_mtime_and_encoding(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(SMART_QUOTES_CP1252 + b"\r\nnext")
    text, encoding, mtime = read_source_text(path, max_bytes=1024)
    assert (text, encoding) == (SMART_QUOTES_TEXT + "\nnext", "cp1252")
    assert mtime == path.stat().st_mtime


def test_read_source_text_refuses_a_file_over_the_cap(tmp_path):
    path = tmp_path / "big.txt"
    path.write_bytes(b"x" * 2048)
    with pytest.raises(OSError, match="too large"):
        read_source_text(path, max_bytes=1024)


def _rows(palace, source_file):
    got = get_collection(palace).get(
        where={"source_file": str(source_file)}, include=["documents", "metadatas"]
    )
    return list(zip(got["documents"], got["metadatas"]))


def _paragraphs(body: bytes) -> bytes:
    return b"\n\n".join(b"paragraph %d. " % i + body * 12 for i in range(4))


@pytest.mark.parametrize(
    ("raw", "expected", "encoding"),
    [
        (_paragraphs(SMART_QUOTES_CP1252 + b" "), SMART_QUOTES_TEXT, "cp1252"),
        (_paragraphs(b"a\x81b \xe9 "), "a\x81b é", "latin-1"),
        (b"\xef\xbb\xbf" + _paragraphs(b"plain words "), "plain words", "utf-8-sig"),
        (
            _paragraphs("résumé ".encode("utf-8")) + b"\xff tail",
            "rÃ©sumÃ©",
            "cp1252",
        ),
    ],
    ids=["cp1252", "latin-1", "bom", "invalid-utf-8"],
)
def test_project_mine_keeps_every_byte_and_stamps_the_encoding(tmp_path, raw, expected, encoding):
    from mempalace import miner

    src = (tmp_path / "proj").resolve()
    src.mkdir()
    doc = src / "notes.md"
    doc.write_bytes(raw)
    palace = str(tmp_path / "palace")
    miner.mine(str(src), palace, wing_override="proj", agent="test")

    rows = _rows(palace, doc)
    assert rows
    assert all("�" not in text and "﻿" not in text for text, _ in rows)
    assert expected in rows[0][0]
    assert {meta.get("source_encoding") for _, meta in rows} == {encoding}


def test_project_mine_stamps_nothing_for_plain_utf8(tmp_path):
    from mempalace import miner

    src = (tmp_path / "proj").resolve()
    src.mkdir()
    doc = src / "notes.md"
    doc.write_bytes(_paragraphs("café ".encode("utf-8")))
    palace = str(tmp_path / "palace")
    miner.mine(str(src), palace, wing_override="proj", agent="test")

    rows = _rows(palace, doc)
    assert rows
    assert all("source_encoding" not in meta for _, meta in rows)


def test_convo_mine_keeps_cp1252_turns_and_stamps_the_encoding(tmp_path):
    from mempalace import convo_miner

    src = (tmp_path / "chats").resolve()
    src.mkdir()
    doc = src / "chat.txt"
    turns = [
        b"> question %d " % i + SMART_QUOTES_CP1252 + b" asked" * 20 + b"\n" + b"answer " * 40
        for i in range(3)
    ]
    doc.write_bytes(b"\n\n".join(turns) + b"\n")
    palace = str(tmp_path / "palace")
    convo_miner.mine_convos(str(src), palace, wing="chats")

    rows = _rows(palace, doc)
    assert rows
    assert all("�" not in text for text, _ in rows)
    assert any(SMART_QUOTES_TEXT in text for text, _ in rows)
    assert {meta.get("source_encoding") for _, meta in rows} == {"cp1252"}


def test_sweeper_decodes_a_cp1252_line_and_names_the_encoding(tmp_path):
    from mempalace.sweeper import parse_claude_jsonl

    record = {
        "type": "user",
        "message": {"role": "user", "content": "PLACEHOLDER"},
        "timestamp": "2026-01-01T00:00:00Z",
        "uuid": "u1",
        "sessionId": "s1",
    }
    line = json.dumps(record).encode("ascii").replace(b"PLACEHOLDER", SMART_QUOTES_CP1252)
    path = tmp_path / "session.jsonl"
    path.write_bytes(line + b"\n")

    (message,) = list(parse_claude_jsonl(str(path)))
    assert message["content"] == SMART_QUOTES_TEXT
    assert message["encoding"] == "cp1252"


def test_diary_ingest_keeps_cp1252_text_and_stamps_the_encoding(tmp_path):
    from mempalace.diary_ingest import ingest_diaries

    diary_dir = tmp_path / "diary"
    diary_dir.mkdir()
    entry = b"## 09:00\n\n" + SMART_QUOTES_CP1252 + b" " + b"a day of work " * 10 + b"\n"
    (diary_dir / "2026-01-02.md").write_bytes(entry)
    palace = str(tmp_path / "palace")
    ingest_diaries(diary_dir, palace)

    got = get_collection(palace).get(include=["documents", "metadatas"])
    rows = list(zip(got["documents"], got["metadatas"]))
    assert rows
    assert all("�" not in text for text, _ in rows)
    assert any(SMART_QUOTES_TEXT in text for text, _ in rows)
    assert {meta.get("source_encoding") for _, meta in rows} == {"cp1252"}
