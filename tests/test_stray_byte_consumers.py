"""A drawer that holds a stray source byte survives every way out.

``fsio.decode_source`` keeps a stray byte in UTF-8 text as a lone surrogate.
Wire bodies carry it as a ``\\udcXX`` JSON escape (``stored_text.dumps_wire``),
exports write the original byte back, and services that only take valid
Unicode get U+FFFD in its place.
"""

import http.client
import json
import os
import threading

import pytest

from mempalace.stored_text import dumps_wire

STRAY = "café zebrafinch " * 10 + "stray\udcffbyte"
STRAY_BYTES = ("café zebrafinch " * 10).encode("utf-8") + b"stray\xffbyte\n"


def test_dumps_wire_escapes_only_lone_surrogates():
    text = dumps_wire({"text": "café \udcff", "n": [1]})
    assert "café" in text
    assert "\\udcff" in text
    text.encode("utf-8")
    assert json.loads(text) == {"text": "café \udcff", "n": [1]}


def test_stdio_reply_keeps_a_lone_surrogate_as_an_escape():
    from mempalace import mcp_server

    def dispatch(request):
        content = [{"type": "text", "text": STRAY}]
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"content": content}}

    reply = mcp_server._stdio_reply('{"jsonrpc": "2.0", "id": 1, "method": "x"}', dispatch)
    reply.encode("utf-8")
    assert "\\udcff" in reply
    assert json.loads(reply)["result"]["content"][0]["text"] == STRAY


def test_llm_request_body_sends_u_fffd_for_a_lone_surrogate():
    from mempalace.llm_client import json_request_body

    body = json_request_body({"messages": [{"content": "café \udcff"}]})
    assert json.loads(body.decode("utf-8")) == {"messages": [{"content": "café �"}]}


@pytest.fixture
def stray_palace(tmp_path, palace_path):
    from mempalace import miner

    src = tmp_path / "src"
    src.mkdir()
    (src / "n.txt").write_bytes(STRAY_BYTES)
    miner.mine(str(src), palace_path, wing_override="proj", agent="t")
    return palace_path


def test_export_writes_the_stray_byte_back(tmp_path, stray_palace):
    from mempalace.exporter import export_palace

    out = tmp_path / "export"
    export_palace(stray_palace, str(out))
    written = b"".join(
        open(os.path.join(root, name), "rb").read()
        for root, _, names in os.walk(out)
        for name in names
        if name.endswith(".md")
    )
    assert b"stray\xffbyte" in written


def _call(port, name, arguments):
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    ).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("POST", "/mcp", body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_http_search_and_get_drawer_carry_a_stray_byte_drawer(
    monkeypatch, config, kg, stray_palace
):
    from mempalace import mcp_server
    from mempalace.palace import get_collection

    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_get_kg", lambda: kg)
    drawer_id = get_collection(stray_palace).get()["ids"][0]

    httpd = mcp_server._build_http_server("127.0.0.1", 0)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        port = httpd.server_address[1]
        for name, arguments in (
            ("mempalace_search", {"query": "zebrafinch", "limit": 3}),
            ("mempalace_search", {"query": "nomatchwordxyz", "limit": 3}),
            ("mempalace_get_drawer", {"drawer_id": drawer_id}),
        ):
            status, raw = _call(port, name, arguments)
            assert status == 200, (name, raw[:200])
            payload = json.loads(raw)
            assert "error" not in payload, payload
            assert "stray\udcffbyte" in payload["result"]["content"][0]["text"] or (
                arguments.get("query") == "nomatchwordxyz"
            )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
