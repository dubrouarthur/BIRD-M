#!/usr/bin/env bash
# Publish a merged checkpoint stored in a bucket to a model repo, one file at a time
# (bucket -> repo copies are not supported server-side; local disk is limited).
# Usage: scripts/publish_from_bucket.sh adubrou/bird-m-runs merged-v1 adubrou/BIRD-M model_card/README.md
set -euo pipefail
BUCKET="$1"; PREFIX="$2"; REPO="$3"; CARD="$4"
TMP="${TMPDIR:-/tmp}/bird_publish"
mkdir -p "$TMP"
files=$(hf buckets ls "$BUCKET" -R 2>/dev/null | awk -v p="$PREFIX/" 'index($NF, p) == 1 {print $NF}')
[ -n "$files" ] || { echo "no files under $BUCKET/$PREFIX" >&2; exit 1; }
for f in $files; do
  name="${f#"$PREFIX"/}"
  hf buckets cp "hf://buckets/$BUCKET/$f" "$TMP/$name" >/dev/null
  hf upload "$REPO" "$TMP/$name" "$name" --commit-message "BIRD-M: add $name" >/dev/null
  rm -f "$TMP/$name"
  echo "published $name"
done
hf upload "$REPO" "$CARD" README.md --commit-message "BIRD-M: model card" >/dev/null
echo "published README.md"
