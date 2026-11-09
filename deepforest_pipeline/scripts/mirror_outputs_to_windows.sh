#!/usr/bin/env bash
# Mirror one completed output folder from the WSL working copy to the Windows copy of this repository and verify
# every file by SHA-256. Refuses to overwrite: the Windows target must not exist yet.
#
#   bash scripts/mirror_outputs_to_windows.sh lowalt_repro_v1
set -euo pipefail
NAME="${1:?usage: mirror_outputs_to_windows.sh <folder under outputs/>}"
L="${LINUX_REPO:-/home/gabpe/src/urban-tree-vision-baselines}/outputs/$NAME"
W="${WINDOWS_REPO:-/mnt/c/Users/gabpe/OneDrive/Documents/src/urban-tree-vision-baselines}/outputs/$NAME"
[ -d "$L" ] || { echo "source $L not found" >&2; exit 1; }
[ -e "$W" ] && { echo "target $W already exists; refusing to overwrite" >&2; exit 1; }
mkdir -p "$(dirname "$W")"
cp -r --preserve=timestamps "$L" "$W"
TMP="$(mktemp -d)"
(cd "$L" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$TMP/l.sha"
(cd "$W" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$TMP/w.sha"
if cmp -s "$TMP/l.sha" "$TMP/w.sha"; then
  echo "mirror verified: $(wc -l < "$TMP/w.sha") files identical ($W)"
else
  echo "MIRROR MISMATCH between $L and $W" >&2; diff "$TMP/l.sha" "$TMP/w.sha" | head >&2; exit 1
fi
rm -rf "$TMP"
