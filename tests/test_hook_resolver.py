"""The shared hook resolver and the plugin copies of it.

``hooks/mempal_hook_run.sh`` picks the interpreter for every hook wrapper.
The Claude Code and Codex plugins install from their own folders, so each
ships a copy; these tests keep the copies byte-identical and check the
Codex wrapper and the resolver's failure modes.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOLVER = REPO_ROOT / "hooks" / "mempal_hook_run.sh"
RESOLVER_COPIES = [
    REPO_ROOT / ".claude-plugin" / "hooks" / "mempal_hook_run.sh",
    REPO_ROOT / ".codex-plugin" / "hooks" / "mempal_hook_run.sh",
]
CODEX_HOOK = REPO_ROOT / ".codex-plugin" / "hooks" / "mempal-hook.sh"
BASH = shutil.which("bash")

needs_bash = pytest.mark.skipif(BASH is None, reason="bash required for hook script tests")


@pytest.mark.parametrize("copy", RESOLVER_COPIES, ids=lambda p: p.parent.parent.name)
def test_plugin_copy_matches_the_shared_resolver(copy: Path) -> None:
    assert copy.read_bytes() == RESOLVER.read_bytes(), (
        f"{copy.relative_to(REPO_ROOT)} drifted from hooks/mempal_hook_run.sh; copy it over"
    )


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _recorder(args_file: Path, stdin_file: Path, *, python: bool) -> str:
    probe = 'if [ "$1" = "-c" ]; then\n  exit 0\nfi\n' if python else ""
    return (
        "#!/bin/sh\n"
        f"{probe}"
        f'printf \'%s\' "$*" > "{args_file.as_posix()}"\n'
        'payload=""\n'
        'while IFS= read -r line || [ -n "$line" ]; do\n'
        '  payload="${payload}${line}"\n'
        "done\n"
        f'printf \'%s\' "$payload" > "{stdin_file.as_posix()}"\n'
        "printf '{}\\n'\n"
    )


def _run(script: Path, args: list, payload: str, bin_dir: Path) -> subprocess.CompletedProcess:
    assert BASH is not None
    env = os.environ.copy()
    env.pop("MEMPAL_PYTHON", None)
    env["PATH"] = str(bin_dir)
    return subprocess.run(
        [BASH, script.as_posix(), *args],
        input=payload,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
    )


@needs_bash
@pytest.mark.parametrize(
    ("runner", "expected_prefix"),
    [("mempalace", ""), ("python3", "-m mempalace "), ("python", "-m mempalace ")],
)
def test_codex_hook_forwards_through_the_resolver(tmp_path, runner, expected_prefix):
    args_file = tmp_path / "args.txt"
    stdin_file = tmp_path / "stdin.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / runner, _recorder(args_file, stdin_file, python=runner != "mempalace")
    )

    payload = '{"session_id":"codex-1"}'
    result = _run(CODEX_HOOK, ["stop"], payload, bin_dir)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "{}\n"
    assert args_file.read_text(encoding="utf-8") == (
        f"{expected_prefix}hook run --hook stop --harness codex"
    )
    assert stdin_file.read_text(encoding="utf-8") == payload


@needs_bash
def test_codex_hook_reports_a_missing_mempalace(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    result = _run(CODEX_HOOK, ["stop"], "{}", bin_dir)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "could not find a runnable mempalace command or module" in result.stderr


@needs_bash
def test_resolver_replies_empty_by_default_when_nothing_runs(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    result = _run(RESOLVER, ["--hook", "stop", "--harness", "claude-code"], "{}", bin_dir)
    assert result.returncode == 0
    assert result.stdout == "{}"
    assert "MEMPAL_PYTHON" in result.stderr
