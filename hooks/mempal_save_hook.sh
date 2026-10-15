#!/bin/bash
# Legacy Claude Code Stop hook.
#
# Mining, the per-file PID guard, write-routing, and silent-save live in
# `mempalace hook run`. This wrapper only chooses an interpreter and
# forwards stdin. The supported install is the Claude Code plugin
# (.claude-plugin/); see hooks/README.md.
set -euo pipefail

python_bin="${MEMPAL_PYTHON:-}"
if [ -z "${python_bin}" ] || [ ! -x "${python_bin}" ]; then
    python_bin="$(command -v python3 2>/dev/null || echo python3)"
fi

exec "${python_bin}" -m mempalace hook run --hook stop --harness claude-code
