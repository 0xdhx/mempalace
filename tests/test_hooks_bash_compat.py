"""Shell-wrapper regressions for the legacy Stop and PreCompact hooks.

The scripts are thin ``exec`` wrappers around ``mempalace hook run``.
They must stay bash 3.2-safe (no ``mapfile`` / ``readarray``) and must
not reimplement mining. SessionEnd coverage in this file is unchanged.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAVE_HOOK = REPO_ROOT / "hooks" / "mempal_save_hook.sh"
PRECOMPACT_HOOK = REPO_ROOT / "hooks" / "mempal_precompact_hook.sh"
SESSION_END_HOOK = REPO_ROOT / "hooks" / "mempal_session_end_hook.sh"
PLUGIN_SESSION_END_HOOK = REPO_ROOT / ".claude-plugin" / "hooks" / "mempal-session-end-hook.sh"

_SESSION_END_HOOKS = pytest.mark.parametrize(
    "hook",
    [SESSION_END_HOOK, PLUGIN_SESSION_END_HOOK],
    ids=["user_hook", "plugin_hook"],
)

# Re-used by every parametrize decorator that runs the same test against
# both hooks. ``ids=`` keeps pytest output readable (`...[save_hook]`
# rather than the default `hook0`/`hook1`).
_BOTH_HOOKS = pytest.mark.parametrize(
    "hook",
    [SAVE_HOOK, PRECOMPACT_HOOK],
    ids=["save_hook", "precompact_hook"],
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="bash hook scripts are POSIX-only")


def _hook_src_no_comments(hook: Path) -> str:
    return "\n".join(
        line for line in hook.read_text().splitlines() if not line.lstrip().startswith("#")
    )


def _run_hook(
    hook: Path,
    stdin: str,
    home: Path,
    *,
    expected_rc: int = 0,
    extra_env: dict | None = None,
) -> tuple[str, str]:
    """Run a hook with a controlled environment and assert its exit code.

    Returns ``(stdout, stderr)``. On unexpected exit the assertion message
    surfaces the captured stderr so CI failures are not silent. The hook's
    primary diagnostic stream is ``$STATE_DIR/hook.log`` plus the two
    sidecar dumps ``last_input.log`` / ``last_python_err.log``; the
    subprocess's stderr is typically empty on the documented exit paths
    and is captured here only to surface unexpected interpreter failures
    (e.g., a bash syntax error on a future edit).

    Forces ``umask 0o022`` in the child so the hook's own ``umask 077``
    inside the parse subshell is provably the sole reason the diagnostic
    files end up at mode 0600 — without this, a permissive ambient umask
    on the CI runner would mask a regression that drops the in-hook
    ``umask`` line.
    """
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(
        ["bash", str(hook)],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        preexec_fn=lambda: os.umask(0o022),
    )
    assert p.returncode == expected_rc, (
        f"{hook.name} exited {p.returncode} (expected {expected_rc}); "
        f"stderr={p.stderr!r}; stdout={p.stdout!r}"
    )
    return p.stdout, p.stderr


class TestNoBash4OnlyBuiltins:
    """Source-level regression: bash 4.0 array-read builtins are unavailable on macOS bash 3.2."""

    @_BOTH_HOOKS
    def test_no_mapfile(self, hook):
        code = _hook_src_no_comments(hook)
        assert "mapfile" not in code, (
            f"{hook.name} uses mapfile, unavailable on macOS /bin/bash 3.2 (#1440)"
        )
        assert "readarray" not in code, (
            f"{hook.name} uses readarray, unavailable on macOS /bin/bash 3.2 (#1440)"
        )

    @_BOTH_HOOKS
    def test_bash_syntax_clean(self, hook):
        p = subprocess.run(
            ["bash", "-n", str(hook)],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, f"{hook.name} syntax error: {p.stderr}"


class TestLegacyWrappersDelegate:
    """Stop and PreCompact wrappers exec hook run and do not mine."""

    _HOOK_NAME = {
        "mempal_save_hook.sh": "stop",
        "mempal_precompact_hook.sh": "precompact",
    }

    @_BOTH_HOOKS
    def test_source_is_a_strict_wrapper(self, hook):
        code = _hook_src_no_comments(hook)
        hook_name = self._HOOK_NAME[hook.name]
        assert "set -euo pipefail" in code
        assert f"hook run --hook {hook_name} --harness claude-code" in code
        assert 'exec "${python_bin}"' in code
        assert '"${MEMPAL_PYTHON:-}"' in code
        assert "mempalace mine" not in code
        assert "dirname" not in code
        assert 'MEMPAL_DIR=""' not in code
        assert "mapfile" not in code

    @_BOTH_HOOKS
    def test_forwards_stdin_to_hook_run(self, hook, tmp_path):
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
        shim.chmod(0o755)
        payload = '{"session_id":"abc12345","transcript_path":"/tmp/t.jsonl"}'
        out, _err = _run_hook(
            hook,
            payload,
            tmp_path,
            extra_env={"MEMPAL_PYTHON": str(shim)},
        )
        assert out == "{}"
        args = recorded.read_text(encoding="utf-8").splitlines()
        hook_name = self._HOOK_NAME[hook.name]
        assert args == [
            "-m",
            "mempalace",
            "hook",
            "run",
            "--hook",
            hook_name,
            "--harness",
            "claude-code",
        ]
        assert copied.read_text(encoding="utf-8") == payload


class TestSessionEndWrappers:
    """The SessionEnd wrappers must background their work — so the foreground
    beats Claude Code's ~1.5s SessionEnd budget (a plugin-provided per-hook
    timeout cannot raise it) — and stay bash 3.2-safe like the other hooks."""

    @_SESSION_END_HOOKS
    def test_bash_syntax_clean(self, hook):
        p = subprocess.run(["bash", "-n", str(hook)], capture_output=True, text=True)
        assert p.returncode == 0, f"{hook.name} syntax error: {p.stderr}"

    @_SESSION_END_HOOKS
    def test_dispatches_session_end_through_cli(self, hook):
        src = _hook_src_no_comments(hook)
        # The dispatcher runs ``mempalace hook run "$@"`` in run_mempalace_hook,
        # and the bottom call supplies the ``--hook session-end --harness`` args
        # — so the two halves are asserted separately, not as one contiguous string.
        assert "hook run" in src
        assert "--hook session-end --harness" in src
        assert "MEMPALACE_HOOK_HARNESS" in src
        assert "MEMPAL_PYTHON" in src

    @_SESSION_END_HOOKS
    def test_backgrounds_then_returns_empty(self, hook):
        src = _hook_src_no_comments(hook)
        assert "</dev/null &" in src, f"{hook.name} does not background the worker"
        assert "disown" in src, f"{hook.name} does not disown the backgrounded worker"
        assert src.rstrip().endswith("printf '{}'"), (
            f"{hook.name} must return the empty payload immediately"
        )

    @_SESSION_END_HOOKS
    def test_no_bash4_only_builtins(self, hook):
        src = _hook_src_no_comments(hook)
        assert "mapfile" not in src, f"{hook.name} uses mapfile (bash 4 only)"
        assert "readarray" not in src, f"{hook.name} uses readarray (bash 4 only)"
