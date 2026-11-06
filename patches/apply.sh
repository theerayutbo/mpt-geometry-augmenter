#!/usr/bin/env bash
# Apply the local compatibility patches to an MPT-Calculator checkout.
#
# Usage:  patches/apply.sh /path/to/MPT-Calculator
#
# Patches (against upstream MPT-Calculator/MPT-Calculator @ ddb99c2, Nov 2025):
#   0001  count_prismatic_elements.py hangs forever with Netgen >= 6.2.2608
#         (newer .vol files no longer contain the '# surfid' terminator line).
#   0002  NumPy >= 2.5 raises "setting an array element with a sequence" when a
#         (1,1) array is assigned to R[i, j]; add .item(). Lets you drop the
#         numpy==2.3.3 pin.
#
# Both have been submitted upstream. Once they are merged, this directory can be
# deleted and the pin in requirements.txt removed.
#
# Safe to re-run: already-applied patches are detected and skipped.

set -euo pipefail

MPT_ROOT="${1:?usage: $0 /path/to/MPT-Calculator}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$MPT_ROOT/main.py" ] || [ ! -d "$MPT_ROOT/Functions" ]; then
  echo "error: $MPT_ROOT does not look like an MPT-Calculator checkout" >&2
  exit 1
fi

cd "$MPT_ROOT"
for p in "$HERE"/*.patch; do
  name="$(basename "$p")"
  if git apply --check --reverse "$p" >/dev/null 2>&1; then
    echo "skip   $name (already applied)"
  elif git apply --check "$p" >/dev/null 2>&1; then
    git apply "$p"
    echo "applied $name"
  else
    echo "FAILED $name — upstream has changed; inspect manually:" >&2
    git apply --check "$p" || true
    exit 1
  fi
done
echo "done."
