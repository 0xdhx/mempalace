from mempalace.revisions import append_drawer_revision, load_drawer_revisions, revisions_path


def test_append_creates_a_missing_palace_dir_and_writes_lf(tmp_path):
    palace = tmp_path / "remote-palace"

    rid = append_drawer_revision(str(palace), "d1", "line one\r\nline two", {"wing": "w"})

    raw = revisions_path(str(palace)).read_bytes()
    assert raw.endswith(b"}\n") and b"}\r\n" not in raw
    [row] = load_drawer_revisions(str(palace), "d1")
    assert row["revision_id"] == rid
    assert row["content"] == "line one\r\nline two"


def test_load_skips_a_torn_line(tmp_path):
    append_drawer_revision(str(tmp_path), "d1", "first")
    with revisions_path(str(tmp_path)).open("ab") as handle:
        handle.write(b'{"revision_id": "torn", "drawer_')
    append_drawer_revision(str(tmp_path), "d1", "second")

    assert [r["content"] for r in load_drawer_revisions(str(tmp_path), "d1")] == [
        "first",
        "second",
    ]
