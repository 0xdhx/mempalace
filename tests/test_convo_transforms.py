"""Lossy steps of conversation parsing are stamped on the drawers (W1-10, D-02)."""

import json

import chromadb

from mempalace.convo_miner import mine_convos
from mempalace.normalize import (
    _format_tool_result,
    _try_claude_code_jsonl,
    _try_slack_json,
    recording_transforms,
    strip_noise,
)
from mempalace.sources import transforms as transform_refs
from mempalace.sources.transforms import format_transforms


def _claude_code(*entries):
    return "\n".join(json.dumps(e) for e in entries) + "\n"


def _user(content):
    return {"type": "user", "message": {"role": "user", "content": content}}


def _assistant(content):
    return {"type": "assistant", "message": {"role": "assistant", "content": content}}


def _bash_session(output_lines):
    return _claude_code(
        _user("run the test suite for the palace and tell me what failed"),
        _assistant(
            [
                {"type": "text", "text": "Running the suite now."},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest"}},
            ]
        ),
        _user(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "\n".join(f"line {i}" for i in range(output_lines)),
                }
            ]
        ),
        _assistant([{"type": "text", "text": "All of the palace tests passed this time."}]),
    )


def _mine(tmp_path, files):
    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    for name, text in files.items():
        (convo_dir / name).write_text(text, encoding="utf-8")
    palace_path = str(tmp_path / "palace")
    mine_convos(str(convo_dir), palace_path, wing="test")
    col = chromadb.PersistentClient(path=palace_path).get_collection("mempalace_drawers")
    rows = col.get(include=["metadatas"])
    return {m["source_file"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]: m for m in rows["metadatas"]}


def test_truncated_tool_output_is_stamped_on_the_drawers(tmp_path):
    metas = _mine(tmp_path, {"long.jsonl": _bash_session(100), "short.jsonl": _bash_session(3)})
    assert "tool_result_truncate" in metas["long.jsonl"]["transforms"].split(",")
    assert "tool_result_truncate" not in metas["short.jsonl"].get("transforms", "").split(",")


def test_a_plain_transcript_carries_no_stamp(tmp_path):
    plain = (
        "> what does the hall index point at\n"
        "Each hall points at the drawers filed under it.\n\n"
        "> and the closets\n"
        "Closets hold the compressed index of a room.\n\n"
        "> and tunnels\n"
        "Tunnels link rooms across wings.\n"
    )
    metas = _mine(tmp_path, {"plain.txt": plain})
    assert "transforms" not in metas["plain.txt"]


def test_each_lossy_tool_step_records_its_rfc_name():
    with recording_transforms() as run:
        _format_tool_result("a\nb", "Bash")
    assert run.applied == set()

    with recording_transforms() as run:
        _format_tool_result("file body", "Read")
        _format_tool_result("\n".join(str(i) for i in range(50)), "Grep")
    assert run.applied == {"tool_result_omitted", "tool_result_truncate"}

    with recording_transforms() as run:
        _format_tool_result("x" * 5000, "SomeTool")
    assert run.applied == {"tool_result_truncate"}


def test_tool_calls_and_dropped_blocks_are_recorded():
    session = _claude_code(
        _user("look at the config file for me"),
        _assistant(
            [
                {"type": "thinking", "thinking": "the user wants the config"},
                {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "a.py"}},
            ]
        ),
        _user([{"type": "tool_result", "tool_use_id": "r1", "content": "x = 1"}]),
        _assistant([{"type": "text", "text": "It sets x to one."}]),
    )
    with recording_transforms() as run:
        assert _try_claude_code_jsonl(session)
    assert run.applied == {
        "conversations.content_block_drop",
        "conversations.tool_input_summarize",
        "tool_result_omitted",
    }


def test_strip_noise_records_only_when_it_removes_chrome():
    with recording_transforms() as run:
        assert strip_noise("plain words\n") == "plain words"
    assert run.applied == set()

    with recording_transforms() as run:
        assert strip_noise("<system-reminder>\nnote\n</system-reminder>\nkeep") == "keep"
    assert run.applied == {"strip_tool_chrome"}


def test_slack_records_assigned_roles_and_its_footer():
    export = [
        {"type": "message", "user": "U1", "text": "morning"},
        {"type": "message", "user": "U2", "text": "hello"},
    ]
    with recording_transforms() as run:
        assert _try_slack_json(export)
    assert run.applied == {"speaker_role_assignment", "synthesized_marker"}


def test_nothing_is_recorded_outside_a_recording_block():
    assert _format_tool_result("file body", "Read") == ""


def test_transforms_compose_into_one_sorted_string():
    assert format_transforms({"tool_result_truncate", "spellcheck_user"}) == (
        "spellcheck_user,tool_result_truncate"
    )
    with recording_transforms() as run:
        pass
    assert run.transforms == ""


def test_custom_names_have_reference_implementations():
    for name in ("conversations.tool_input_summarize", "conversations.content_block_drop"):
        assert getattr(transform_refs, name.replace(".", "_"))("text") == "text"
