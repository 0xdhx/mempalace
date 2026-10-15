"""The tool capability registry is the only read/write list."""

from mempalace import mcp_server, service

# Refused in read-only mode but missing from the old hand-kept WRITE_TOOLS.
_DRIFT_FROM_WRITE_TOOLS = (
    "mempalace_artifact_put",
    "mempalace_event_ack",
    "mempalace_event_append",
    "mempalace_kg_supersede",
    "mempalace_mine",
    "mempalace_patch_submit",
    "mempalace_sync",
    "mempalace_task_create",
)

# readOnlyHint used to omit these reads.
_OMITTED_READ_ONLY_HINTS = (
    "mempalace_traverse",
    "mempalace_find_tunnels",
    "mempalace_graph_stats",
    "mempalace_list_tunnels",
    "mempalace_list_hallways",
    "mempalace_event_list",
    "mempalace_artifact_get",
    "mempalace_mesh_peers",
)

# classify_tool used to return "unknown" for these.
_FORMERLY_UNKNOWN = (
    "mempalace_event_wait",
    "mempalace_kg_supersede",
    "mempalace_mesh_peers",
    "mempalace_artifact_get",
)


def test_every_tool_has_one_capability_and_classify_matches_it():
    missing = [name for name, spec in mcp_server.TOOLS.items() if "capability" not in spec]
    assert missing == []
    assert set(mcp_server._TOOL_CAPABILITIES) == set(mcp_server.TOOLS)
    for name, spec in mcp_server.TOOLS.items():
        capability = spec["capability"]
        assert capability in {"read", "write", "maintenance"}
        assert service.classify_tool(name) == capability
        assert service.classify_tool(name) != "unknown"
        if capability == "read":
            assert spec.get("read_only") is True
            assert name not in mcp_server._READ_ONLY_REFUSED_TOOLS
        elif name != "mempalace_reconnect":
            assert name in mcp_server._READ_ONLY_REFUSED_TOOLS


def test_read_only_refusals_match_write_and_maintenance_classification():
    """Tools refused in read-only mode are writes, or maintenance that is not reconnect."""
    for name in _DRIFT_FROM_WRITE_TOOLS:
        assert name in mcp_server._READ_ONLY_REFUSED_TOOLS
        kind = service.classify_tool(name)
        assert kind in {"write", "maintenance"}
        if name in {"mempalace_mine", "mempalace_sync"}:
            assert kind == "maintenance"
            assert name not in service.WRITE_TOOLS
        else:
            assert kind == "write"
            assert name in service.WRITE_TOOLS
    assert "mempalace_reconnect" not in mcp_server._READ_ONLY_REFUSED_TOOLS
    assert service.classify_tool("mempalace_reconnect") == "maintenance"
    assert service.WRITE_TOOLS <= mcp_server._READ_ONLY_REFUSED_TOOLS
    assert mcp_server._MUTATING_TOOLS < mcp_server._READ_ONLY_REFUSED_TOOLS
    assert "mempalace_hook_settings" not in mcp_server._MUTATING_TOOLS
    assert "mempalace_memories_filed_away" not in mcp_server._MUTATING_TOOLS


def test_formerly_unknown_tools_and_omitted_hints_are_classified():
    for name in _FORMERLY_UNKNOWN:
        assert service.classify_tool(name) != "unknown"
    for name in _OMITTED_READ_ONLY_HINTS:
        assert mcp_server.TOOLS[name]["capability"] == "read"
        assert mcp_server.TOOLS[name]["read_only"] is True
        entry = mcp_server._tool_catalog_entry(name, mcp_server.TOOLS[name])
        assert entry["annotations"]["readOnlyHint"] is True


def test_derived_gate_sets_match_the_capability_flags():
    tools = mcp_server.TOOLS
    assert mcp_server._MUTATING_TOOLS == {n for n, s in tools.items() if s.get("mutating")}
    assert mcp_server._VECTOR_WRITE_TOOLS == {n for n, s in tools.items() if s.get("vector_write")}
    assert mcp_server._PEER_WRITER_EXEMPT_TOOLS == {
        n for n, s in tools.items() if s.get("peer_exempt")
    }
    assert mcp_server._SQLITE_INTEGRITY_ALLOWED_TOOLS == {
        n for n, s in tools.items() if s.get("integrity_allowed")
    }
    assert mcp_server._HTTP_LOCK_FREE_TOOLS == {n for n, s in tools.items() if s.get("lock_free")}
    assert mcp_server._HTTP_EMBEDDING_RELEASE_TOOLS == {
        n for n, s in tools.items() if s.get("embedding_release")
    }
    assert mcp_server._VECTOR_WRITE_TOOLS <= mcp_server._MUTATING_TOOLS
    assert mcp_server._PEER_WRITER_EXEMPT_TOOLS <= mcp_server._MUTATING_TOOLS
    reads = {n for n, s in tools.items() if s["capability"] == "read"}
    assert service.READ_TOOLS == reads
    assert service.WRITE_TOOLS == {n for n, s in tools.items() if s["capability"] == "write"}
    assert service.MAINTENANCE_TOOLS == {
        n for n, s in tools.items() if s["capability"] == "maintenance"
    }
