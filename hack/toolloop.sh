#!/usr/bin/env bash
# F3a spike: prove the complete tool-use loop against the real Anthropic
# Messages API (https://api.anthropic.com/v1/messages), authenticated with
# the Max-plan OAuth setup-token (so usage bills on the plan, not API
# pricing) rather than an ANTHROPIC_API_KEY.
#
# This is throwaway spike code, not production kortex code. It proves:
#   (a) the full non-streaming tool_use -> tool_result round trip
#   (b) the SSE event sequence for a tool_use turn when stream:true
#   (c) prompt caching (cache_control) layered on top of a tool-use request
#   (d) the same request against claude-haiku-4-5-20251001 and, optionally,
#       claude-fable-5
#
# SAFETY: the only tool declared is `bash`, and this script will only ever
# execute the tool call locally if the model's requested command matches an
# allow-list of read-only, side-effect-free commands (echo/pwd/date). Any
# other requested command is refused and printed, not executed.
#
# Auth: never echo the token. It is read from the secrets file into an env
# var and only ever sent as an Authorization: Bearer header.
set -euo pipefail

TOKEN_FILE="/home/vingarcia/.openclaw/secrets/kortex-oauth-token"
API="https://api.anthropic.com/v1/messages"
HAIKU_MODEL="claude-haiku-4-5-20251001"
FABLE_MODEL="claude-fable-5"

if [ ! -f "$TOKEN_FILE" ]; then
  echo "FAIL: token file not found at $TOKEN_FILE" >&2
  exit 1
fi
CODECOMPANION_OAUTH_TOKEN="$(cat "$TOKEN_FILE")"
export CODECOMPANION_OAUTH_TOKEN

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# curl_api <payload-file> <out-file>
# Calls the Messages API with the standard two headers this spike proves
# work: Authorization: Bearer <oauth token>, anthropic-version: 2023-06-01.
curl_api() {
  local payload_file="$1" out_file="$2"
  curl -sS "$API" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${CODECOMPANION_OAUTH_TOKEN}" \
    -H "anthropic-version: 2023-06-01" \
    -d @"$payload_file" \
    -o "$out_file"
}

# is_safe_command <cmd> -> 0 if the literal command is on the allow-list.
is_safe_command() {
  case "$1" in
    "echo hello-from-toolloop"|"echo "*|"pwd"|"date"|"date "*) return 0 ;;
    *) return 1 ;;
  esac
}

# run_tool_loop <model> <label>
# Runs part (a): one non-streaming request that forces a bash tool_use,
# executes it locally, and sends the tool_result back in a second request.
run_tool_loop() {
  local model="$1" label="$2"
  echo "=== (a) non-streaming tool loop: $label ($model) ==="

  cat >"$TMP/req1.json" <<EOF
{
  "model": "$model",
  "max_tokens": 1024,
  "system": "You are a helpful assistant with access to a bash tool. When asked to run a shell command, call the bash tool instead of guessing the output.",
  "tools": [
    {
      "name": "bash",
      "description": "Run a shell command and return its stdout/stderr.",
      "input_schema": {
        "type": "object",
        "properties": {
          "command": {"type": "string", "description": "The shell command to execute."}
        },
        "required": ["command"]
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "Run the shell command \`echo hello-from-toolloop\` and tell me its exact output."}
  ]
}
EOF
  curl_api "$TMP/req1.json" "$TMP/resp1.json"

  echo "--- request 1 response (trimmed) ---"
  jq '{stop_reason, content, usage}' "$TMP/resp1.json"

  local stop_reason
  stop_reason="$(jq -r '.stop_reason' "$TMP/resp1.json")"
  if [ "$stop_reason" != "tool_use" ]; then
    echo "FAIL: expected stop_reason=tool_use, got $stop_reason" >&2
    jq '.' "$TMP/resp1.json" >&2
    return 1
  fi

  local tool_id tool_name tool_input command
  tool_id="$(jq -r '.content[] | select(.type=="tool_use") | .id' "$TMP/resp1.json")"
  tool_name="$(jq -r '.content[] | select(.type=="tool_use") | .name' "$TMP/resp1.json")"
  command="$(jq -r '.content[] | select(.type=="tool_use") | .input.command' "$TMP/resp1.json")"
  echo "tool_use: id=$tool_id name=$tool_name input.command=$command"

  local tool_output is_error
  if is_safe_command "$command"; then
    tool_output="$(eval "$command" 2>&1)"
    is_error="false"
  else
    tool_output="refused: command not on allow-list (echo/pwd/date only)"
    is_error="true"
    echo "REFUSED to execute: $command"
  fi
  echo "tool_output: $tool_output"

  # Build the full assistant content block array (the model may have emitted
  # text before the tool_use block too) to echo back verbatim, plus the
  # second request's new user turn carrying the tool_result.
  jq -n \
    --arg model "$model" \
    --slurpfile assistant_content <(jq -c '.content' "$TMP/resp1.json") \
    --arg tool_id "$tool_id" \
    --arg tool_output "$tool_output" \
    --argjson is_error "$is_error" \
    '{
      model: $model,
      max_tokens: 1024,
      system: "You are a helpful assistant with access to a bash tool. When asked to run a shell command, call the bash tool instead of guessing the output.",
      tools: [
        {
          name: "bash",
          description: "Run a shell command and return its stdout/stderr.",
          input_schema: {
            type: "object",
            properties: { command: { type: "string", description: "The shell command to execute." } },
            required: ["command"]
          }
        }
      ],
      messages: [
        {role: "user", content: "Run the shell command `echo hello-from-toolloop` and tell me its exact output."},
        {role: "assistant", content: $assistant_content[0]},
        {role: "user", content: [
          {type: "tool_result", tool_use_id: $tool_id, content: $tool_output, is_error: $is_error}
        ]}
      ]
    }' > "$TMP/req2.json"

  echo "--- request 2 payload (the tool_result turn we send back) ---"
  jq '.messages[2]' "$TMP/req2.json"

  curl_api "$TMP/req2.json" "$TMP/resp2.json"

  echo "--- request 2 response (trimmed) ---"
  jq '{stop_reason, content, usage}' "$TMP/resp2.json"

  local stop_reason2
  stop_reason2="$(jq -r '.stop_reason' "$TMP/resp2.json")"
  if [ "$stop_reason2" != "end_turn" ]; then
    echo "NOTE: expected stop_reason=end_turn on final turn, got $stop_reason2" >&2
  fi
  echo
}

# run_streaming_tool_loop <model> <label>
# Part (b): same first turn, but stream:true. Captures the raw SSE and
# records the observed event-type sequence plus how input_json_delta
# fragments assemble into the final tool input JSON.
run_streaming_tool_loop() {
  local model="$1" label="$2"
  echo "=== (b) streaming tool loop: $label ($model) ==="

  cat >"$TMP/req_stream.json" <<EOF
{
  "model": "$model",
  "max_tokens": 1024,
  "stream": true,
  "system": "You are a helpful assistant with access to a bash tool. When asked to run a shell command, call the bash tool instead of guessing the output.",
  "tools": [
    {
      "name": "bash",
      "description": "Run a shell command and return its stdout/stderr.",
      "input_schema": {
        "type": "object",
        "properties": {
          "command": {"type": "string", "description": "The shell command to execute."}
        },
        "required": ["command"]
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "Run the shell command \`pwd\` and tell me its exact output."}
  ]
}
EOF
  curl -sS "$API" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${CODECOMPANION_OAUTH_TOKEN}" \
    -H "anthropic-version: 2023-06-01" \
    -d @"$TMP/req_stream.json" > "$TMP/stream.sse"

  echo "--- observed SSE event-type sequence ---"
  grep '^event: ' "$TMP/stream.sse" | sed 's/^event: //'

  echo "--- content_block_start for the tool_use block ---"
  grep '"type":"content_block_start"' "$TMP/stream.sse" | grep '"tool_use"' || true

  # Each SSE line is "data: {...json...}" (or an event: line, or blank).
  # Strip the "data: " prefix and let jq parse+unescape each JSON event
  # properly, rather than regexing the escaped string by hand (regexing it
  # leaves backslash-escapes in place and produces invalid JSON).
  echo "--- input_json_delta fragments (partial_json), in order, after proper JSON-unescaping ---"
  grep '^data: ' "$TMP/stream.sse" | sed 's/^data: //' | \
    jq -r 'select(.type=="content_block_delta" and .delta.type=="input_json_delta") | .delta.partial_json'

  echo "--- reassembled tool input JSON (concatenation of partial_json fragments) ---"
  assembled="$(grep '^data: ' "$TMP/stream.sse" | sed 's/^data: //' | \
    jq -r 'select(.type=="content_block_delta" and .delta.type=="input_json_delta") | .delta.partial_json' | tr -d '\n')"
  echo "raw concatenation: $assembled"
  printf '%s' "$assembled" | jq '.' >/dev/null && echo "parsed OK -> $assembled" || echo "FAIL: concatenated partial_json did not parse as JSON"

  echo "--- final message_delta (carries stop_reason) ---"
  grep '"type":"message_delta"' "$TMP/stream.sse"
  echo
}

# run_cache_test <model> <label>
# Part (c): pad the system prompt well past Haiku's 4096-token cache
# minimum, keep the same bash tool declared, and run the same request
# twice back to back to observe cache_creation_input_tokens on the first
# call and cache_read_input_tokens on the second.
run_cache_test() {
  local model="$1" label="$2"
  echo "=== (c) cache_control + tool use: $label ($model) ==="

  # Filler padding: ~6000 tokens of repeated text (roughly 4 chars/token),
  # comfortably over Haiku 4.5's 4096-token cacheable minimum, appended
  # after the real instructions and ahead of the cache_control breakpoint.
  local filler
  filler="$(python3 -c "print('This is filler context to exceed the prompt-cache minimum token count for Haiku four point five. ' * 500)")"

  jq -n --arg filler "$filler" '{
    model: "'"$model"'",
    max_tokens: 1024,
    system: [
      {
        type: "text",
        text: ("You are a helpful assistant with access to a bash tool. When asked to run a shell command, call the bash tool instead of guessing the output.\n\n" + $filler),
        cache_control: {type: "ephemeral"}
      }
    ],
    tools: [
      {
        name: "bash",
        description: "Run a shell command and return its stdout/stderr.",
        input_schema: {
          type: "object",
          properties: { command: { type: "string", description: "The shell command to execute." } },
          required: ["command"]
        }
      }
    ],
    messages: [
      {role: "user", content: "Run the shell command `date` and tell me its exact output."}
    ]
  }' > "$TMP/req_cache.json"

  echo "-- call 1 (expect cache_creation_input_tokens > 0, cache_read_input_tokens == 0) --"
  curl_api "$TMP/req_cache.json" "$TMP/resp_cache1.json"
  jq '{stop_reason, usage}' "$TMP/resp_cache1.json"

  echo "-- call 2, identical payload (expect cache_read_input_tokens > 0) --"
  curl_api "$TMP/req_cache.json" "$TMP/resp_cache2.json"
  jq '{stop_reason, usage}' "$TMP/resp_cache2.json"
  echo
}

echo "########## HAIKU 4.5 ##########"
run_tool_loop "$HAIKU_MODEL" "haiku"
run_streaming_tool_loop "$HAIKU_MODEL" "haiku"
run_cache_test "$HAIKU_MODEL" "haiku"

if [ "${RUN_FABLE:-0}" = "1" ]; then
  echo "########## FABLE 5 (confirmation run) ##########"
  run_tool_loop "$FABLE_MODEL" "fable"
fi

echo "DONE"
