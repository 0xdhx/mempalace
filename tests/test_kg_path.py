"""Which knowledge graph file each entry point opens.

One rule, in ``mempalace.paths.knowledge_graph_path``: the palace's own graph
wins; the legacy ``~/.mempalace/knowledge_graph.sqlite3`` is read only for the
legacy default palace and only while that palace has no graph of its own;
graphs are never merged.
"""

import argparse
import logging
import os

import pytest

from mempalace import knowledge_graph, paths

FILENAME = "knowledge_graph.sqlite3"


class _Opened(Exception):
    pass


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    os.makedirs(paths.default_palace_path())
    return home


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kg = knowledge_graph.KnowledgeGraph(db_path=path)
    kg.close()


def _record_opens(monkeypatch):
    opened = []

    def fake(db_path=None, **_kwargs):
        opened.append(db_path)
        raise _Opened

    monkeypatch.setattr(knowledge_graph, "KnowledgeGraph", fake)
    return opened


def _resolver(palace, monkeypatch):
    return paths.knowledge_graph_path(palace)


def _mcp(palace, monkeypatch):
    from mempalace import mcp_server

    return mcp_server._resolve_kg_path()


def _hermes(palace, monkeypatch):
    from mempalace.integrations.hermes import MempalaceProvider

    provider = object.__new__(MempalaceProvider)
    provider._palace_path = palace
    return provider._kg_db_path()


def _fact_checker(palace, monkeypatch):
    from mempalace import fact_checker

    opened = _record_opens(monkeypatch)
    fact_checker.check_text("Bob is Alice's brother")
    return opened[0]


def _audit(palace, monkeypatch):
    from mempalace import palace_audit

    opened = []

    def read_kg(path):
        opened.append(path)
        raise _Opened

    monkeypatch.setattr(palace_audit, "_read_wing_room_counts", lambda config: {})
    monkeypatch.setattr(palace_audit, "_load_tunnels", lambda config: [])
    monkeypatch.setattr(palace_audit, "list_hallways", lambda config: [])
    monkeypatch.setattr(palace_audit, "_read_kg", read_kg)
    with pytest.raises(_Opened):
        palace_audit.audit_palace()
    return opened[0]


def _kg_normalize(palace, monkeypatch):
    import mempalace.cli as cli

    opened = _record_opens(monkeypatch)
    args = argparse.Namespace(kg_action="normalize", palace=None, vocabulary=None, yes=False)
    try:
        cli.cmd_kg(args)
    except _Opened:
        return opened[0]
    except SystemExit:
        return None


ENTRY_POINTS = {
    "resolver": _resolver,
    "mcp": _mcp,
    "hermes": _hermes,
    "fact_checker": _fact_checker,
    "audit": _audit,
    "kg_normalize": _kg_normalize,
}

# (palace, graphs present, graph opened, notice expected)
CASES = {
    "palace_local_only": ("default", {"local"}, "local", False),
    "legacy_only": ("default", {"legacy"}, "legacy", False),
    "both": ("default", {"local", "legacy"}, "local", True),
    "neither": ("default", set(), "local", False),
    "env_override": ("custom", {"legacy"}, "local", True),
}


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_each_entry_point_opens_the_palace_graph(case, entry, home, tmp_path, monkeypatch):
    which, present, expected, _ = CASES[case]
    palace = paths.default_palace_path() if which == "default" else str(tmp_path / "custom")
    os.makedirs(palace, exist_ok=True)
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", palace)
    files = {
        "local": os.path.join(palace, FILENAME),
        "legacy": paths.legacy_knowledge_graph_path(),
    }
    for name in present:
        _touch(files[name])

    opened = ENTRY_POINTS[entry](palace, monkeypatch)

    if entry == "kg_normalize" and expected not in present:
        # Nothing to normalize, and the legacy graph is not touched instead.
        assert opened is None
        assert not os.path.exists(files["local"])
    else:
        assert paths.same_path(opened, files[expected]), (opened, files[expected])


@pytest.mark.parametrize("case", sorted(CASES))
def test_notice_names_the_unused_graph_and_the_import_command(case, home, tmp_path, monkeypatch):
    which, present, _, expected = CASES[case]
    palace = paths.default_palace_path() if which == "default" else str(tmp_path / "custom")
    os.makedirs(palace, exist_ok=True)
    for name in present:
        _touch(
            os.path.join(palace, FILENAME)
            if name == "local"
            else paths.legacy_knowledge_graph_path()
        )

    notice = knowledge_graph.knowledge_graph_notice(palace)

    if expected:
        assert paths.legacy_knowledge_graph_path() in notice
        assert "mempalace kg import-legacy" in notice
    else:
        assert notice is None


def test_hermes_keeps_reading_the_graph_beside_its_palace(tmp_path):
    palace = str(tmp_path / "hermes" / "palace")
    os.makedirs(palace)
    sibling = str(tmp_path / "hermes" / FILENAME)
    _touch(sibling)

    assert paths.same_path(_hermes(palace, None), sibling)
    _touch(os.path.join(palace, FILENAME))
    assert paths.same_path(_hermes(palace, None), os.path.join(palace, FILENAME))
    notice = knowledge_graph.knowledge_graph_notice(palace, legacy=sibling)
    assert sibling in notice


def test_status_prints_the_notice(home, tmp_path, monkeypatch, capsys):
    import mempalace.cli as cli

    custom = str(tmp_path / "custom")
    os.makedirs(custom)
    _touch(paths.legacy_knowledge_graph_path())
    monkeypatch.setattr("mempalace.miner.status", lambda palace_path: None)

    cli.cmd_status(argparse.Namespace(palace=custom))

    assert "mempalace kg import-legacy" in capsys.readouterr().out


def test_mcp_logs_the_notice_once(home, tmp_path, monkeypatch, caplog):
    from mempalace import mcp_server

    custom = str(tmp_path / "custom")
    os.makedirs(custom)
    _touch(paths.legacy_knowledge_graph_path())
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", custom)
    monkeypatch.setattr(mcp_server, "_kg_by_path", {})

    with caplog.at_level(logging.WARNING, logger="mempalace_mcp"):
        kg = mcp_server._get_kg()
        mcp_server._get_kg()
    kg.close()

    notices = [r for r in caplog.records if "kg import-legacy" in r.getMessage()]
    assert len(notices) == 1
    assert paths.same_path(kg.db_path, os.path.join(custom, FILENAME))


def _import(palace, yes, source=None):
    import mempalace.cli as cli

    cli.cmd_kg(argparse.Namespace(kg_action="import-legacy", palace=palace, source=source, yes=yes))


def test_import_legacy_dry_run_writes_nothing_then_copies_on_yes(home, capsys):
    palace = paths.default_palace_path()
    legacy = paths.legacy_knowledge_graph_path()
    local = os.path.join(palace, FILENAME)
    kg = knowledge_graph.KnowledgeGraph(db_path=legacy)
    kg.add_triple("alice", "likes", "tea")
    kg.close()

    _import(palace, yes=False)
    assert "Would copy 1 triples" in capsys.readouterr().out
    assert not os.path.exists(local)
    assert paths.same_path(paths.knowledge_graph_path(palace), legacy)

    _import(palace, yes=True)
    assert "Copied 1 triples" in capsys.readouterr().out
    assert paths.same_path(paths.knowledge_graph_path(palace), local)
    kg = knowledge_graph.KnowledgeGraph(db_path=local)
    try:
        assert any(f["object"] == "tea" for f in kg.query_entity("alice"))
    finally:
        kg.close()

    _import(palace, yes=True)
    out = capsys.readouterr().out
    assert "Copied 0 triples" in out and "1 already in" in out


def test_import_legacy_refuses_a_missing_source_or_the_palace_own_graph(home, tmp_path):
    palace = str(tmp_path / "custom")
    os.makedirs(palace)
    with pytest.raises(SystemExit):
        _import(palace, yes=True)
    _touch(os.path.join(palace, FILENAME))
    with pytest.raises(SystemExit):
        _import(palace, yes=True, source=os.path.join(palace, FILENAME))
