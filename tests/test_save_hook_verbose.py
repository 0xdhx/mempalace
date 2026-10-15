"""Silent vs verbose save is the Python hook's job, not the legacy shell.

The wrapper must not grow its own MEMPAL_VERBOSE / decision:block fork.
``mempalace hook run`` reads ``hooks.silent_save``.
"""

from pathlib import Path


def test_save_hook_does_not_reimplement_verbose_mode():
    src = (Path(__file__).resolve().parents[1] / "hooks" / "mempal_save_hook.sh").read_text(
        encoding="utf-8"
    )
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "MEMPAL_VERBOSE" not in code
    assert "decision" not in code
    assert "hook run --hook stop --harness claude-code" in code
