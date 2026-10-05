package protocol

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestEmit_golden(t *testing.T) {
	tests := []struct {
		desc string
		emit func() ([]byte, error)
		want string
	}{
		{
			desc: "system/init reports the tool list and no capabilities",
			emit: func() ([]byte, error) { return EmitSystemInit("sess-1", []string{"bash", "read", "mcp__openclaw__foo"}) },
			want: `{"type":"system","subtype":"init","tools":["bash","read","mcp__openclaw__foo"],"capabilities":[],"session_id":"sess-1"}`,
		},
		{
			desc: "system/init emits an empty tool array, never null",
			emit: func() ([]byte, error) { return EmitSystemInit("sess-1", nil) },
			want: `{"type":"system","subtype":"init","tools":[],"capabilities":[],"session_id":"sess-1"}`,
		},
		{
			desc: "system/keepalive carries only identity",
			emit: func() ([]byte, error) { return EmitSystemKeepalive("sess-1") },
			want: `{"type":"system","subtype":"keepalive","session_id":"sess-1"}`,
		},
		{
			desc: "assistant text message",
			emit: func() ([]byte, error) {
				return EmitAssistant("sess-1", "uuid-out-1",
					[]ContentBlock{{Type: "text", Text: "Tudo ótimo!"}},
					Usage{InputTokens: 10, OutputTokens: 4, CacheReadInputTokens: 100, CacheCreationInputTokens: 7})
			},
			want: `{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Tudo ótimo!"}],"usage":{"input_tokens":10,"output_tokens":4,"cache_read_input_tokens":100,"cache_creation_input_tokens":7}},"uuid":"uuid-out-1","parent_tool_use_id":null,"session_id":"sess-1"}`,
		},
		{
			desc: "assistant tool_use message",
			emit: func() ([]byte, error) {
				return EmitAssistant("sess-1", "uuid-out-2",
					[]ContentBlock{{Type: "tool_use", ID: "toolu_01", Name: "Bash", Input: json.RawMessage(`{"command":"ls"}`)}},
					Usage{InputTokens: 5, OutputTokens: 2})
			},
			want: `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_01","name":"Bash","input":{"command":"ls"}}],"usage":{"input_tokens":5,"output_tokens":2,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}},"uuid":"uuid-out-2","parent_tool_use_id":null,"session_id":"sess-1"}`,
		},
		{
			desc: "user tool_result echo",
			emit: func() ([]byte, error) {
				return EmitUser("sess-1", "uuid-out-3",
					[]ContentBlock{{Type: "tool_result", ToolUseID: "toolu_01", Content: json.RawMessage(`"main.go\nREADME.md"`)}})
			},
			want: `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"main.go\nREADME.md"}]},"uuid":"uuid-out-3","parent_tool_use_id":null,"session_id":"sess-1"}`,
		},
		{
			desc: "result success",
			emit: func() ([]byte, error) {
				return EmitResult("sess-1",
					Result{Text: "Tudo ótimo!", NumTurns: 1, StopReason: "end_turn"},
					Usage{InputTokens: 10, OutputTokens: 4, CacheReadInputTokens: 100, CacheCreationInputTokens: 7})
			},
			want: `{"type":"result","subtype":"success","is_error":false,"result":"Tudo ótimo!","num_turns":1,"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":4,"cache_read_input_tokens":100,"cache_creation_input_tokens":7},"session_id":"sess-1"}`,
		},
		{
			desc: "result error leaves stop_reason null",
			emit: func() ([]byte, error) {
				return EmitResult("sess-1", Result{IsError: true, NumTurns: 0}, Usage{})
			},
			want: `{"type":"result","subtype":"error","is_error":true,"result":"","num_turns":0,"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0},"session_id":"sess-1"}`,
		},
		{
			desc: "control_response success echoes the request id",
			emit: func() ([]byte, error) { return EmitControlResponse("req-2") },
			want: `{"type":"control_response","response":{"subtype":"success","request_id":"req-2"}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got, err := test.emit()
			if err != nil {
				t.Fatalf("emit returned error: %v", err)
			}
			if string(got) != test.want+"\n" {
				t.Errorf("emit =\n%s\nwant\n%s\n", got, test.want)
			}
		})
	}
}

// TestEmit_roundTrip proves each emitted line decodes back through Parse into
// the fields the reader already consumes, so the write side cannot drift from
// the read side.
func TestEmit_roundTrip(t *testing.T) {
	// emit forwards an emitter's two-value return (its call is the sole
	// argument), failing the test on the error that these inputs never trigger.
	emit := func(line []byte, err error) []byte {
		t.Helper()
		if err != nil {
			t.Fatalf("emit returned error: %v", err)
		}
		return line
	}

	t.Run("system/init", func(t *testing.T) {
		ev := Parse(emit(EmitSystemInit("sess-1", []string{"bash"})))
		if ev.Type != TypeSystem || ev.Subtype != "init" || ev.SessionID != "sess-1" {
			t.Errorf("parsed = %+v", ev)
		}
	})

	t.Run("system/keepalive", func(t *testing.T) {
		ev := Parse(emit(EmitSystemKeepalive("sess-1")))
		if ev.Type != TypeSystem || ev.Subtype != "keepalive" || ev.SessionID != "sess-1" {
			t.Errorf("parsed = %+v", ev)
		}
	})

	t.Run("assistant", func(t *testing.T) {
		blocks := []ContentBlock{{Type: "text", Text: "Tudo ótimo!"}}
		usage := Usage{InputTokens: 10, OutputTokens: 4, CacheReadInputTokens: 100, CacheCreationInputTokens: 7}
		ev := Parse(emit(EmitAssistant("sess-1", "uuid-out-1", blocks, usage)))
		if ev.Type != TypeAssistant || ev.UUID != "uuid-out-1" || ev.SessionID != "sess-1" {
			t.Fatalf("parsed = %+v", ev)
		}
		if ev.Message == nil || ev.Message.Role != "assistant" || !reflect.DeepEqual([]ContentBlock(ev.Message.Content), blocks) {
			t.Fatalf("message = %+v", ev.Message)
		}
		if ev.Message.Usage == nil || *ev.Message.Usage != usage {
			t.Errorf("usage = %+v", ev.Message.Usage)
		}
	})

	t.Run("user tool_result", func(t *testing.T) {
		blocks := []ContentBlock{{Type: "tool_result", ToolUseID: "toolu_01", Content: json.RawMessage(`"main.go"`)}}
		ev := Parse(emit(EmitUser("sess-1", "uuid-out-3", blocks)))
		if ev.Type != TypeUser || ev.UUID != "uuid-out-3" || ev.SessionID != "sess-1" {
			t.Fatalf("parsed = %+v", ev)
		}
		if ev.Message == nil || ev.Message.Role != "user" || !reflect.DeepEqual([]ContentBlock(ev.Message.Content), blocks) {
			t.Errorf("message = %+v", ev.Message)
		}
	})

	t.Run("result success", func(t *testing.T) {
		ev := Parse(emit(EmitResult("sess-1",
			Result{Text: "Tudo ótimo!", NumTurns: 1, StopReason: "end_turn"},
			Usage{InputTokens: 10})))
		want := Result{Text: "Tudo ótimo!", NumTurns: 1, StopReason: "end_turn"}
		if ev.Type != TypeResult || ev.Subtype != "success" || ev.SessionID != "sess-1" {
			t.Fatalf("parsed = %+v", ev)
		}
		if ev.Result == nil || *ev.Result != want {
			t.Errorf("result = %+v, want %+v", ev.Result, want)
		}
	})

	t.Run("result error", func(t *testing.T) {
		ev := Parse(emit(EmitResult("sess-1", Result{IsError: true}, Usage{})))
		if ev.Type != TypeResult || ev.Subtype != "error" {
			t.Fatalf("parsed = %+v", ev)
		}
		if ev.Result == nil || !ev.Result.IsError || ev.Result.StopReason != "" {
			t.Errorf("result = %+v", ev.Result)
		}
	})

	t.Run("control_response", func(t *testing.T) {
		ev := Parse(emit(EmitControlResponse("req-2")))
		if ev.Type != TypeControlResponse || ev.Subtype != "success" || ev.RequestID != "req-2" {
			t.Errorf("parsed = %+v", ev)
		}
	})
}
