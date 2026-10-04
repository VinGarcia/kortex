package protocol

import "testing"

// Wire fixtures copied from the F0 protocol spec (§3/§4/§5).
const (
	fixtureInitialize    = `{"type":"control_request","request_id":"req-1","request":{"subtype":"initialize","appendSystemPrompt":"You are Sylphie.","excludeDynamicSections":false,"hooks":{"UserPromptSubmit":[{"hookCallbackIds":["UserPromptSubmit"]}],"PreToolUse":[{"hookCallbackIds":["PreToolUse"]}]}}}`
	fixtureUserPrompt    = `{"type":"user","message":{"role":"user","content":"oi, tudo bem?"},"parent_tool_use_id":null,"uuid":"uuid-in-1","session_id":"sess-1"}`
	fixtureUserBlocks    = `{"type":"user","message":{"role":"user","content":[{"type":"text","text":"analisa isso"},{"type":"text","text":"por favor"}]},"uuid":"uuid-in-2"}`
	fixtureSystemInit    = `{"type":"system","subtype":"init","session_id":"sess-1","capabilities":["msg_lifecycle_v1"]}`
	fixtureAssistantText = `{"type":"assistant","message":{"id":"msg_01","role":"assistant","content":[{"type":"text","text":"Tudo ótimo!"}],"usage":{"input_tokens":10,"output_tokens":4,"cache_read_input_tokens":100,"cache_creation_input_tokens":7}},"uuid":"uuid-out-1","parent_tool_use_id":null,"session_id":"sess-1"}`
	fixtureAssistantTool = `{"type":"assistant","message":{"id":"msg_02","role":"assistant","content":[{"type":"tool_use","id":"toolu_01","name":"Bash","input":{"command":"ls"}}]},"uuid":"uuid-out-2","session_id":"sess-1"}`
	fixtureToolResult    = `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"main.go\nREADME.md"}]},"uuid":"uuid-out-3","session_id":"sess-1"}`
	fixtureUserReplay    = `{"type":"user","message":{"role":"user","content":"oi, tudo bem?"},"uuid":"uuid-in-1","session_id":"sess-1","isReplay":true}`
	fixtureStreamDelta   = `{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Tudo"}},"session_id":"sess-1"}`
	fixtureResult        = `{"type":"result","subtype":"success","is_error":false,"result":"Tudo ótimo!","num_turns":1,"stop_reason":"end_turn","session_id":"sess-1"}`
	fixtureResultError   = `{"type":"result","subtype":"error_during_execution","is_error":true,"result":"","num_turns":0,"stop_reason":null,"session_id":"sess-1"}`
	fixtureCanUseTool    = `{"type":"control_request","request_id":"req-2","request":{"subtype":"can_use_tool","tool_name":"Bash","input":{"command":"ls"},"tool_use_id":"toolu_01"}}`
	fixtureControlResp   = `{"type":"control_response","response":{"subtype":"success","request_id":"req-2","response":{"behavior":"allow"}}}`
	fixtureKeepAlive     = `{"type":"keep_alive"}`
)

func TestParse(t *testing.T) {
	tests := []struct {
		desc string
		line string
		// want covers the scalar envelope fields; Message/Result/ParseErr
		// have dedicated tests and are zeroed before comparing.
		want Event
	}{
		{
			desc: "control_request initialize surfaces the subtype and system prompt",
			line: fixtureInitialize,
			want: Event{Type: TypeControlRequest, Subtype: "initialize", RequestID: "req-1", AppendSystemPrompt: "You are Sylphie."},
		},
		{
			desc: "user prompt with string content",
			line: fixtureUserPrompt,
			want: Event{Type: TypeUser, UUID: "uuid-in-1", SessionID: "sess-1"},
		},
		{
			desc: "system init",
			line: fixtureSystemInit,
			want: Event{Type: TypeSystem, Subtype: "init", SessionID: "sess-1"},
		},
		{
			desc: "replayed user message",
			line: fixtureUserReplay,
			want: Event{Type: TypeUser, UUID: "uuid-in-1", SessionID: "sess-1", IsReplay: true},
		},
		{
			desc: "stream_event carries the inner SSE event type",
			line: fixtureStreamDelta,
			want: Event{Type: TypeStreamEvent, StreamEventType: "content_block_delta", SessionID: "sess-1"},
		},
		{
			desc: "control_response surfaces subtype and request_id from the payload",
			line: fixtureControlResp,
			want: Event{Type: TypeControlResponse, Subtype: "success", RequestID: "req-2"},
		},
		{
			desc: "control_request can_use_tool surfaces the request subtype",
			line: fixtureCanUseTool,
			want: Event{Type: TypeControlRequest, Subtype: "can_use_tool", RequestID: "req-2"},
		},
		{
			desc: "keep_alive",
			line: fixtureKeepAlive,
			want: Event{Type: TypeKeepAlive},
		},
		{
			desc: "session id accepted under camelCase alias",
			line: `{"type":"assistant","sessionId":"sess-camel"}`,
			want: Event{Type: TypeAssistant, SessionID: "sess-camel"},
		},
		{
			desc: "unknown event type is kept, not rejected",
			line: `{"type":"totally_new_thing","session_id":"sess-1"}`,
			want: Event{Type: "totally_new_thing", SessionID: "sess-1"},
		},
		{
			desc: "non-json line becomes TypeUnknown",
			line: "stray log output",
			want: Event{Type: TypeUnknown},
		},
		{
			desc: "json without type becomes TypeUnknown",
			line: `{"foo":"bar"}`,
			want: Event{Type: TypeUnknown},
		},
		{
			desc: "empty line becomes TypeUnknown",
			line: "",
			want: Event{Type: TypeUnknown},
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := Parse([]byte(test.line))
			got.Message, got.Result, got.ParseErr = nil, nil, nil
			if got != test.want {
				t.Errorf("Parse(%q) =\n%+v, want\n%+v", test.line, got, test.want)
			}
		})
	}
}

func TestParse_messageContent(t *testing.T) {
	t.Run("string content normalizes to one text block", func(t *testing.T) {
		got := Parse([]byte(fixtureUserPrompt))
		if got.Message == nil {
			t.Fatal("Message is nil")
		}
		if text := got.Message.TextContent(); text != "oi, tudo bem?" {
			t.Errorf("TextContent = %q", text)
		}
	})
	t.Run("block-array content joins text blocks", func(t *testing.T) {
		got := Parse([]byte(fixtureUserBlocks))
		if text := got.Message.TextContent(); text != "analisa isso\npor favor" {
			t.Errorf("TextContent = %q", text)
		}
	})
	t.Run("assistant usage decodes the four token counters", func(t *testing.T) {
		got := Parse([]byte(fixtureAssistantText))
		usage := got.Message.Usage
		if usage == nil {
			t.Fatal("Usage is nil")
		}
		if usage.InputTokens != 10 || usage.OutputTokens != 4 ||
			usage.CacheReadInputTokens != 100 || usage.CacheCreationInputTokens != 7 {
			t.Errorf("usage = %+v", *usage)
		}
	})
	t.Run("tool_use block keeps id, name and raw input", func(t *testing.T) {
		got := Parse([]byte(fixtureAssistantTool))
		blocks := got.Message.Content
		if len(blocks) != 1 || blocks[0].Type != "tool_use" {
			t.Fatalf("content = %+v", blocks)
		}
		if blocks[0].ID != "toolu_01" || blocks[0].Name != "Bash" || string(blocks[0].Input) != `{"command":"ls"}` {
			t.Errorf("tool_use block = %+v", blocks[0])
		}
	})
	t.Run("tool_result block keeps tool_use_id and raw content", func(t *testing.T) {
		got := Parse([]byte(fixtureToolResult))
		blocks := got.Message.Content
		if len(blocks) != 1 || blocks[0].Type != "tool_result" {
			t.Fatalf("content = %+v", blocks)
		}
		if blocks[0].ToolUseID != "toolu_01" {
			t.Errorf("ToolUseID = %q", blocks[0].ToolUseID)
		}
	})
	t.Run("malformed content sets ParseErr but keeps the type", func(t *testing.T) {
		got := Parse([]byte(`{"type":"assistant","message":{"role":"assistant","content":42},"session_id":"sess-1"}`))
		if got.Type != TypeAssistant {
			t.Errorf("Type = %q", got.Type)
		}
		if got.ParseErr == nil {
			t.Error("ParseErr is nil, want decode error")
		}
		if got.SessionID != "sess-1" {
			t.Errorf("SessionID = %q, want partial decode to keep it", got.SessionID)
		}
	})
}

func TestContentBlock_ResultText(t *testing.T) {
	tests := []struct {
		desc    string
		content string // raw JSON of the tool_result "content" field
		want    string
	}{
		{desc: "string payload", content: `"main.go\nREADME.md"`, want: "main.go\nREADME.md"},
		{desc: "nested text blocks", content: `[{"type":"text","text":"conteúdo"}]`, want: "conteúdo"},
		{desc: "empty payload", content: ``, want: ""},
		{desc: "unrecognized shape comes back verbatim", content: `{"weird":true}`, want: `{"weird":true}`},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			block := ContentBlock{Type: "tool_result", Content: []byte(test.content)}
			if got := block.ResultText(); got != test.want {
				t.Errorf("ResultText() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestParse_result(t *testing.T) {
	t.Run("success result", func(t *testing.T) {
		got := Parse([]byte(fixtureResult))
		if got.Result == nil {
			t.Fatal("Result is nil")
		}
		want := Result{IsError: false, Text: "Tudo ótimo!", NumTurns: 1, StopReason: "end_turn"}
		if *got.Result != want {
			t.Errorf("Result = %+v, want %+v", *got.Result, want)
		}
	})
	t.Run("error result with null stop_reason", func(t *testing.T) {
		got := Parse([]byte(fixtureResultError))
		if !got.Result.IsError || got.Result.StopReason != "" {
			t.Errorf("Result = %+v", *got.Result)
		}
		if got.Subtype != "error_during_execution" {
			t.Errorf("Subtype = %q", got.Subtype)
		}
	})
}
