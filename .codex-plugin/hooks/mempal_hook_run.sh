#!/bin/bash
# Runs `mempalace hook run "$@"`, stdin included, through the first of
# $MEMPAL_PYTHON, `mempalace` on PATH, python3, or python that can import
# mempalace.
#
# This file is shared by every hook wrapper. The Claude Code and Codex plugins
# install from their own folders, so each carries a byte-identical copy
# (.claude-plugin/hooks/, .codex-plugin/hooks/); tests/test_hook_resolver.py
# keeps the copies in step. Edit this file and copy it over.
#
# When no interpreter can import mempalace, the hook prints the empty reply
# and exits 0 so the harness carries on. A leading --on-missing=error makes it
# exit 1 with the message instead, for harnesses that should show the failure.
set -uo pipefail

on_missing=empty
if [ "${1:-}" = "--on-missing=error" ]; then
    on_missing=error
    shift
fi

if [ -n "${MEMPAL_PYTHON:-}" ] && "${MEMPAL_PYTHON}" -c "import mempalace" </dev/null >/dev/null 2>&1; then
    exec "${MEMPAL_PYTHON}" -m mempalace hook run "$@"
fi
if command -v mempalace >/dev/null 2>&1; then
    exec mempalace hook run "$@"
fi
for python_bin in python3 python; do
    if command -v "${python_bin}" >/dev/null 2>&1 \
        && "${python_bin}" -c "import mempalace" </dev/null >/dev/null 2>&1; then
        exec "${python_bin}" -m mempalace hook run "$@"
    fi
done

echo "MemPalace hook error: could not find a runnable mempalace command or module; set MEMPAL_PYTHON" >&2
if [ "$on_missing" = error ]; then
    exit 1
fi
printf '{}'
