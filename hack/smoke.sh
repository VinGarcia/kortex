#!/usr/bin/env bash
# Manual integration smoke test: runs the same trivial prompt through the real
# `claude` directly and through kortex, and checks kortex is a transparent proxy.
# Requires an authenticated `claude` CLI on PATH and Go to build kortex.
set -euo pipefail

cd "$(dirname "$0")/.."
PROMPT='Reply with exactly the single word: KORTEX_SMOKE_OK'
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

go build -o "$TMP/kortex" .

echo "== 1/3 direct claude (text output)"
direct_out="$(claude -p "$PROMPT" --output-format text)"
echo "direct: $direct_out"

echo "== 2/3 kortex -> claude (text output)"
kortex_out="$(KORTEX_LOG="$TMP/traffic.log" "$TMP/kortex" -p "$PROMPT" --output-format text)"
echo "kortex: $kortex_out"

fail=0
case "$direct_out" in *KORTEX_SMOKE_OK*) ;; *) echo "FAIL: direct claude output missing token"; fail=1 ;; esac
case "$kortex_out" in *KORTEX_SMOKE_OK*) ;; *) echo "FAIL: kortex output missing token"; fail=1 ;; esac

echo "== 3/3 kortex stream-json output is valid NDJSON with a terminal result"
stream_out="$("$TMP/kortex" -p "$PROMPT" --output-format stream-json --verbose)"
line_count=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  line_count=$((line_count + 1))
  if ! printf '%s' "$line" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; then
    echo "FAIL: non-JSON line on stream-json stdout: ${line:0:120}"
    fail=1
  fi
done <<<"$stream_out"
case "$stream_out" in *'"type":"result"'*) ;; *) echo "FAIL: no terminal result event"; fail=1 ;; esac
echo "stream-json lines: $line_count (all parsed as JSON unless failed above)"

if [ "$fail" -eq 0 ]; then
  echo "SMOKE OK"
else
  echo "SMOKE FAILED"
  exit 1
fi
