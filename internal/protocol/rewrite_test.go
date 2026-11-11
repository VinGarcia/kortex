package protocol

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func upper(text string) (string, bool) { return strings.ToUpper(text), true }

func TestRewriteUserText(t *testing.T) {
	tests := []struct {
		desc    string
		line    string
		rewrite func(string) (string, bool)
		// wantContent is the expected message.content of the result,
		// compared as canonical JSON. "" means expect no rewrite (nil, nil).
		wantContent string
		wantErr     error
	}{
		{
			desc:        "plain string content",
			line:        `{"type":"user","message":{"role":"user","content":"oi tudo"},"uuid":"u1"}`,
			rewrite:     upper,
			wantContent: `"OI TUDO"`,
		},
		{
			desc: "single text block among other blocks, others preserved",
			line: `{"type":"user","message":{"role":"user","content":[` +
				`{"type":"image","source":{"data":"aGk="}},{"type":"text","text":"oi","cache_control":{"type":"ephemeral"}}]}}`,
			rewrite:     upper,
			wantContent: `[{"type":"image","source":{"data":"aGk="}},{"cache_control":{"type":"ephemeral"},"text":"OI","type":"text"}]`,
		},
		{
			desc:    "rewrite declines",
			line:    `{"type":"user","message":{"role":"user","content":"oi"}}`,
			rewrite: func(string) (string, bool) { return "", false },
		},
		{
			desc:    "two text blocks unsupported",
			line:    `{"type":"user","message":{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}}`,
			rewrite: upper,
			wantErr: ErrUnsupportedContent,
		},
		{
			desc:    "no text block unsupported",
			line:    `{"type":"user","message":{"role":"user","content":[{"type":"image","source":{}}]}}`,
			rewrite: upper,
			wantErr: ErrUnsupportedContent,
		},
		{
			desc:    "missing content unsupported",
			line:    `{"type":"user","message":{"role":"user"}}`,
			rewrite: upper,
			wantErr: ErrUnsupportedContent,
		},
		{
			desc:    "missing message unsupported",
			line:    `{"type":"user"}`,
			rewrite: upper,
			wantErr: ErrUnsupportedContent,
		},
		{
			desc:    "non-json line unsupported",
			line:    `not json`,
			rewrite: upper,
			wantErr: ErrUnsupportedContent,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got, err := RewriteUserText([]byte(test.line+"\n"), test.rewrite)
			if test.wantErr != nil {
				if !errors.Is(err, test.wantErr) {
					t.Fatalf("want %v, got %v", test.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if test.wantContent == "" {
				if got != nil {
					t.Fatalf("want no rewrite, got %q", got)
				}
				return
			}
			if got[len(got)-1] != '\n' || strings.Count(string(got), "\n") != 1 {
				t.Fatalf("result must be one newline-terminated line: %q", got)
			}
			var envelope struct {
				Message struct {
					Content json.RawMessage `json:"content"`
				} `json:"message"`
			}
			if err := json.Unmarshal(got, &envelope); err != nil {
				t.Fatal(err)
			}
			if canon(t, envelope.Message.Content) != canon(t, json.RawMessage(test.wantContent)) {
				t.Errorf("content = %s, want %s", envelope.Message.Content, test.wantContent)
			}
		})
	}
}

func TestRewriteUserText_preservesUnknownEnvelopeFields(t *testing.T) {
	line := `{"type":"user","message":{"role":"user","content":"oi","some_future_field":7},"uuid":"u1","session_id":"s","custom":true}` + "\n"
	got, err := RewriteUserText([]byte(line), upper)
	if err != nil {
		t.Fatal(err)
	}
	var envelope map[string]any
	if err := json.Unmarshal(got, &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope["uuid"] != "u1" || envelope["session_id"] != "s" || envelope["custom"] != true {
		t.Errorf("envelope fields lost: %v", envelope)
	}
	message := envelope["message"].(map[string]any)
	if message["some_future_field"] != float64(7) || message["role"] != "user" {
		t.Errorf("message fields lost: %v", message)
	}
}

func TestNewMessageSegment(t *testing.T) {
	tests := []struct {
		desc string
		text string
		want string
	}{
		{
			desc: "no context echo returns text unchanged",
			text: "oi tudo\n\nroda o script",
			want: "oi tudo\n\nroda o script",
		},
		{
			desc: "echoed context block then new message returns only the new segment",
			text: "Recent conversation ⟦openclaw:ctx⟧\n[Sun 2026-10-04 20:33] user: contei uma novidade ontem\n" +
				"[emoções p1: investment=2 valence=positiva emotions=alegria(2)]\n\nroda o script de novo",
			want: "roda o script de novo",
		},
		{
			desc: "multiple context blocks returns text after the last",
			text: "a ⟦openclaw:ctx⟧ turn one\n\nb ⟦openclaw:ctx⟧ turn two\n\nmensagem nova",
			want: "mensagem nova",
		},
		{
			desc: "new message spanning paragraphs is kept whole",
			text: "ctx ⟦openclaw:ctx⟧ old turn\n\npar um\n\npar dois",
			want: "par um\n\npar dois",
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := NewMessageSegment(test.text); got != test.want {
				t.Errorf("NewMessageSegment = %q, want %q", got, test.want)
			}
		})
	}
}

// canon re-marshals raw JSON into a canonical string for comparison.
func canon(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("canon: %v (%s)", err, raw)
	}
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

func TestDecodeErrors_KeepCauseInChain(t *testing.T) {
	tests := []struct {
		desc string
		call func() error
		// wantPrefix is our own text before the cause; the cause text itself
		// belongs to encoding/json and is deliberately not pinned.
		wantPrefix string
		// findCause extracts the expected encoding/json cause via errors.As.
		findCause func(error) (error, bool)
	}{
		{
			desc: "envelope",
			call: func() error {
				_, err := RewriteUserText([]byte(`{bad`), upper)
				return err
			},
			wantPrefix: "unsupported message content shape: decoding envelope: ",
			findCause:  asCause[*json.SyntaxError],
		},
		{
			desc: "message",
			call: func() error {
				_, err := RewriteUserText([]byte(`{"message":5}`), upper)
				return err
			},
			wantPrefix: "unsupported message content shape: decoding message: ",
			findCause:  asCause[*json.UnmarshalTypeError],
		},
		{
			desc: "string content",
			call: func() error {
				_, _, err := contentText(json.RawMessage(`"abc`))
				return err
			},
			wantPrefix: "unsupported message content shape: decoding string content: ",
			findCause:  asCause[*json.SyntaxError],
		},
		{
			desc: "content blocks",
			call: func() error {
				_, _, err := contentText(json.RawMessage(`5`))
				return err
			},
			wantPrefix: "unsupported message content shape: decoding content blocks: ",
			findCause:  asCause[*json.UnmarshalTypeError],
		},
		{
			desc: "content block",
			call: func() error {
				_, _, err := contentText(json.RawMessage(`[5]`))
				return err
			},
			wantPrefix: "unsupported message content shape: decoding content block: ",
			findCause:  asCause[*json.UnmarshalTypeError],
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			err := test.call()
			if !errors.Is(err, ErrUnsupportedContent) {
				t.Fatalf("errors.Is(ErrUnsupportedContent) = false for %v", err)
			}
			cause, ok := test.findCause(err)
			if !ok {
				t.Fatalf("expected json cause not found via errors.As in %v", err)
			}
			if want := test.wantPrefix + cause.Error(); err.Error() != want {
				t.Errorf("Error() =\n%q\nwant\n%q", err.Error(), want)
			}
		})
	}
}

// asCause reports whether err wraps a T, returning it as the cause.
func asCause[T error](err error) (error, bool) {
	var cause T
	ok := errors.As(err, &cause)
	return cause, ok
}
