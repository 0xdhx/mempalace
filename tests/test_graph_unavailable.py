"""A palace that exists but cannot be read is not an empty graph."""

import pytest

import mempalace.palace_graph as palace_graph
from mempalace.config import MempalaceConfig
from mempalace.palace_graph import GraphUnavailable


def _config(path):
    return MempalaceConfig(config_dir=str(path.parent / "cfg"), palace_path=str(path))


def test_a_missing_palace_is_an_empty_graph(tmp_path):
    assert palace_graph._get_collection(_config(tmp_path / "missing")) is None


def test_a_palace_without_drawers_is_an_empty_graph(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()
    assert palace_graph._get_collection(_config(palace)) is None


def test_any_other_open_failure_raises_graph_unavailable(tmp_path, monkeypatch):
    def locked(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(palace_graph, "_get_palace_collection", locked)
    with pytest.raises(GraphUnavailable, match="could not be read") as info:
        palace_graph._get_collection(_config(tmp_path))
    # The client-facing message names the palace, not the backend's error text.
    assert "locked" not in str(info.value)


def test_the_graph_opens_the_palace_read_only(tmp_path, monkeypatch):
    seen = {}

    def opened(*args, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(palace_graph, "_get_palace_collection", opened)
    palace_graph._get_collection(_config(tmp_path))
    assert seen["read_only"] is True
    assert seen["create"] is False


def test_build_graph_does_not_report_an_unreadable_palace_as_empty(tmp_path, monkeypatch):
    palace_graph.invalidate_graph_cache()
    monkeypatch.setattr(palace_graph, "_try_sqlite_nodes_edges", lambda config: None)

    def locked(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(palace_graph, "_get_palace_collection", locked)
    with pytest.raises(GraphUnavailable):
        palace_graph.build_graph(config=_config(tmp_path))


@pytest.mark.parametrize("tool", ["traverse", "find_tunnels"])
def test_mcp_graph_tools_answer_an_unreadable_palace_with_an_error(monkeypatch, tool):
    import mempalace.mcp_server as mcp_server

    def unavailable(*args, **kwargs):
        raise GraphUnavailable("/some/palace")

    monkeypatch.setattr(mcp_server, "_graph_sqlite_reader", lambda: object())
    monkeypatch.setattr(mcp_server, tool, unavailable)
    if tool == "traverse":
        result = mcp_server.tool_traverse_graph("room")
    else:
        result = mcp_server.tool_find_tunnels("wing_a", "wing_b")
    assert result == {"error": "palace at /some/palace could not be read for the graph"}


def test_mcp_create_tunnel_answers_an_unreadable_palace_with_an_error(monkeypatch):
    import mempalace.mcp_server as mcp_server

    def unavailable(*args, **kwargs):
        raise GraphUnavailable()

    monkeypatch.setattr(mcp_server, "create_tunnel", unavailable)
    result = mcp_server.tool_create_tunnel("wing_a", "room_a", "wing_b", "room_b")
    assert result == {"error": "palace could not be read for the graph"}
