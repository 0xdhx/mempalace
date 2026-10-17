"""Drawers filed under an older wing slug stay reachable by project.

Before one slug function, the hooks filed diaries under ``wing_<slug>`` and
the Cursor hooks kept hyphens, while ``mempalace mine`` used the bare slug.
Existing drawers are never re-keyed, so a wing filter has to match every one
of those spellings.
"""

from _mcp_server_helpers import _patch_mcp_server

OLD_SPELLINGS = {
    "mine": "my_app",  # `mempalace mine` of ~/dev/my-app
    "hook": "wing_my_app",  # Claude Code / Antigravity hook diary
    "cursor": "my-app",  # Cursor hook kept the hyphen
}


def _seed_old_palace(collection):
    collection.add(
        ids=[f"old_{name}" for name in OLD_SPELLINGS],
        documents=[
            f"release notes for the my-app project, filed by {name}" for name in OLD_SPELLINGS
        ],
        metadatas=[
            {
                "wing": wing,
                "room": "notes",
                "source_file": f"{name}.md",
                "filed_at": "2026-01-01T00:00:00",
            }
            for name, wing in OLD_SPELLINGS.items()
        ],
    )


def test_search_and_list_find_drawers_under_every_old_spelling(
    monkeypatch, config, palace_path, collection, kg
):
    _seed_old_palace(collection)
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace import mcp_server

    written = mcp_server.tool_add_drawer(
        wing="my_app", room="notes", content="new notes for my-app"
    )
    assert written.get("success"), written

    listed = mcp_server.tool_list_drawers(wing="my_app", limit=20)
    assert listed["total"] == 4, listed
    assert {d["wing"] for d in listed["drawers"]} == {"my_app", "wing_my_app", "my-app"}

    # The old spelling a client remembers works too.
    assert mcp_server.tool_list_drawers(wing="wing_my_app", limit=20)["total"] == 4

    found = mcp_server.tool_search(
        query="release notes for the my-app project", wing="my_app", limit=10
    )
    assert {hit["wing"] for hit in found["results"]} >= {"my_app", "wing_my_app", "my-app"}, found


def test_diary_read_finds_entries_under_the_old_project_and_agent_wings(
    monkeypatch, config, palace_path, collection, kg
):
    collection.add(
        ids=["old_hook_diary", "old_agent_diary"],
        documents=["old hook checkpoint", "old agent note"],
        metadatas=[
            {
                "wing": "wing_my_app",
                "room": "diary",
                "agent": "claude",
                "topic": "checkpoint",
                "date": "2026-01-01",
                "filed_at": "2026-01-01T00:00:00",
            },
            {
                "wing": "wing_cursor-ide",
                "room": "diary",
                "agent": "cursor-ide",
                "topic": "general",
                "date": "2026-01-01",
                "filed_at": "2026-01-01T00:00:00",
            },
        ],
    )
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace import mcp_server

    # D-17: hook diaries now file into the project wing itself.
    assert mcp_server.tool_diary_write("claude", "new checkpoint", wing="my_app")["success"]
    project = mcp_server.tool_diary_read("claude", last_n=10, wing="my_app")
    assert {e["content"] for e in project["entries"]} == {"old hook checkpoint", "new checkpoint"}

    # An agent's default wing slugs like a project; the hyphenated old wing still reads.
    written = mcp_server.tool_diary_write("cursor-ide", "new agent note")
    assert written["success"] and written["entry_id"].startswith("diary_wing_cursor_ide_"), written
    agent = mcp_server.tool_diary_read("cursor-ide", last_n=10, wing="wing_cursor_ide")
    assert {e["content"] for e in agent["entries"]} == {"old agent note", "new agent note"}
