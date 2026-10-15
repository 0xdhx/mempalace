"""Legacy save/precompact scripts must not mine a transcript directory.

Mining a whole directory with no PID guard stacked miners on overlapping
hook fires. The scripts now exec ``mempalace hook run``, which mines one
transcript file behind the Python PID slot.
"""

import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = {
    "mempal_save_hook.sh": "stop",
    "mempal_precompact_hook.sh": "precompact",
}


def _code(name: str) -> str:
    text = (ROOT / "hooks" / name).read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


@pytest.mark.parametrize("name", list(HOOKS))
def test_wrapper_does_not_mine_the_transcript_directory(name):
    src = _code(name)
    assert "mempalace mine" not in src
    assert "dirname" not in src
    assert 'MEMPAL_DIR=""' not in src
    assert f"hook run --hook {HOOKS[name]} --harness claude-code" in src


@pytest.mark.skipif(sys.platform == "win32", reason="shell hooks are POSIX-only")
@pytest.mark.parametrize("name", list(HOOKS))
def test_wrapper_execs_hook_run_with_stdin(name, tmp_path):
    recorded = tmp_path / "argv"
    copied = tmp_path / "stdin"
    shim = tmp_path / "python-shim"
    shim.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f'printf \'%s\\n\' "$@" > "{recorded}"\n'
        f'cat > "{copied}"\n'
        "printf '{}'\n",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    payload = '{"session_id":"s","transcript_path":"/tmp/session.jsonl"}\n'
    env = {
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin",
        "MEMPAL_PYTHON": str(shim),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "hooks" / name)],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "{}"
    assert recorded.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "mempalace",
        "hook",
        "run",
        "--hook",
        HOOKS[name],
        "--harness",
        "claude-code",
    ]
    assert copied.read_text(encoding="utf-8") == payload
