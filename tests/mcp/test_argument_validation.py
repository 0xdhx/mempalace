"""tools/call arguments are checked against the tool's input schema at dispatch."""

import pytest

from mempalace import mcp_server

_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        "threshold": {"type": "number"},
        "apply": {"type": "boolean"},
        "query": {"type": "string", "maxLength": 10},
        "strategy": {"type": "string", "enum": ["vector", "union"]},
        "ids": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
    },
}


@pytest.fixture
def received(monkeypatch):
    calls = []

    def handler(
        limit=5, threshold=None, apply=False, query="", strategy="vector", ids=None, metadata=None
    ):
        calls.append(
            {
                "limit": limit,
                "threshold": threshold,
                "apply": apply,
                "query": query,
                "strategy": strategy,
                "ids": ids,
                "metadata": metadata,
            }
        )
        return {"ok": True}

    monkeypatch.setattr(
        mcp_server, "TOOLS", {"mempalace_fake": {"handler": handler, "input_schema": _SCHEMA}}
    )
    monkeypatch.setattr(mcp_server, "_mcp_tool_preflight_refusal", lambda req_id, name: None)
    return calls


def _call(arguments, name="mempalace_fake"):
    return mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ({"limit": "5"}, {"limit": 5}),
        ({"limit": 5.0}, {"limit": 5}),
        ({"threshold": "0.5"}, {"threshold": 0.5}),
        ({"apply": "false"}, {"apply": False}),
        ({"apply": "TRUE"}, {"apply": True}),
        ({"apply": 0}, {"apply": False}),
        ({"query": 42}, {"query": "42"}),
        ({"ids": '["a", "b"]'}, {"ids": ["a", "b"]}),
        ({"metadata": '{"k": 1}'}, {"metadata": {"k": 1}}),
        ({"limit": None, "apply": None}, {"limit": 5, "apply": False}),
    ],
)
def test_values_are_coerced_to_the_declared_type(received, arguments, expected):
    response = _call(arguments)

    assert "error" not in response, response
    for key, value in expected.items():
        assert received[-1][key] == value
        assert type(received[-1][key]) is type(value)


@pytest.mark.parametrize(
    "arguments, fragment",
    [
        ({"limit": 5.5}, "'limit' must be an integer"),
        ({"limit": True}, "'limit' must be an integer"),
        ({"limit": 0}, "'limit' must be at least 1"),
        ({"limit": 1000}, "'limit' must be at most 100"),
        ({"threshold": "nan"}, "'threshold' must be a finite number"),
        ({"apply": "maybe"}, "'apply' must be true or false"),
        ({"apply": 2}, "'apply' must be true or false"),
        ({"query": "x" * 11}, "'query' must be at most 10 characters"),
        ({"query": ["a"]}, "'query' must be a string"),
        ({"strategy": "bogus"}, "'strategy' must be one of 'vector', 'union'"),
        ({"ids": "a,b"}, "'ids' must be an array"),
        ({"ids": ["a", {"b": 1}]}, "'ids[1]' must be a string"),
        ({"metadata": [1]}, "'metadata' must be an object"),
    ],
)
def test_a_value_outside_the_schema_is_refused_before_the_handler(received, arguments, fragment):
    response = _call(arguments)

    assert response["error"]["code"] == -32602
    assert fragment in response["error"]["message"]
    assert "mempalace_fake" in response["error"]["message"]
    assert received == []


def test_sync_apply_false_string_does_not_apply(monkeypatch):
    """A JSON client sending "false" used to reach tool_sync as a truthy string."""
    seen = {}

    def spy(source_dir=None, wing=None, apply=False, **kwargs):
        seen["apply"] = apply
        return {"success": True}

    monkeypatch.setitem(mcp_server.TOOLS["mempalace_sync"], "handler", spy)
    monkeypatch.setattr(mcp_server, "_mcp_tool_preflight_refusal", lambda req_id, name: None)

    response = _call({"apply": "false"}, name="mempalace_sync")

    assert "error" not in response, response
    assert seen["apply"] is False


def test_light_server_runs_the_same_check(monkeypatch):
    """palace_query reaches the handler in process; it must not skip validation."""
    from mempalace import mcp_light_server

    def search(**kwargs):  # pragma: no cover - must not run
        raise AssertionError("handler ran with an invalid limit")

    monkeypatch.setitem(mcp_server.TOOLS["mempalace_search"], "handler", search)

    result = mcp_light_server.tool_palace_query({"target": "search", "query": "x", "limit": 1000})

    assert result["success"] is False
    assert "'limit' must be at most 100" in result["error"]
