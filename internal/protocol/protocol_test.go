package protocol

import "testing"

func TestMessageType(t *testing.T) {
	tests := []struct {
		desc string
		line string
		want string
	}{
		{desc: "stream event envelope", line: `{"type":"stream_event","event":{"type":"message_start"}}`, want: "stream_event"},
		{desc: "result event", line: `{"type":"result","is_error":false}` + "\n", want: "result"},
		{desc: "line with surrounding whitespace", line: `  {"type":"assistant"}  ` + "\n", want: "assistant"},
		{desc: "json without type field", line: `{"foo":"bar"}`, want: ""},
		{desc: "non-json line", line: "some stray log output", want: ""},
		{desc: "empty line", line: "", want: ""},
		{desc: "json with non-string type", line: `{"type":42}`, want: ""},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := MessageType([]byte(test.line))
			if got != test.want {
				t.Errorf("MessageType(%q) = %q, want %q", test.line, got, test.want)
			}
		})
	}
}
