from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _hook(name):
    return (ROOT / "hooks" / name).read_text(encoding="utf-8")


def _code(name):
    return "\n".join(line for line in _hook(name).splitlines() if not line.lstrip().startswith("#"))


def test_save_hook_delegates_to_hook_run():
    body = _code("mempal_save_hook.sh")
    assert "set -euo pipefail" in body
    assert "hook run --hook stop --harness claude-code" in body
    assert "mempalace mine" not in body
    assert "parse-stop" not in body


def test_precompact_hook_delegates_to_hook_run():
    body = _code("mempal_precompact_hook.sh")
    assert "set -euo pipefail" in body
    assert "hook run --hook precompact --harness claude-code" in body
    assert "mempalace mine" not in body
    assert "parse-precompact" not in body
