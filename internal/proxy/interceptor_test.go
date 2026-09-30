package proxy

import (
	"bytes"
	"io"
	"strings"
	"testing"

	"github.com/vingarcia/kortex/internal/protocol"
)

// rewriteUserInterceptor rewrites user-typed lines and passes everything
// else through, mimicking the facet contract of the real interceptor.
type rewriteUserInterceptor struct {
	replacement string
	seenTypes   []string
}

func (i *rewriteUserInterceptor) InterceptToBackend(line []byte, event protocol.Event) []byte {
	i.seenTypes = append(i.seenTypes, event.Type)
	if event.Type != protocol.TypeUser {
		return nil
	}
	return []byte(i.replacement + "\n")
}

func TestCopyLines_interceptorRewritesOnlyUserLines(t *testing.T) {
	input := `{"type":"control_request","request_id":"r1","request":{"subtype":"initialize"}}` + "\n" +
		`{"type":"user","message":{"role":"user","content":"original"}}` + "\n" +
		"not json\n"
	interceptor := &rewriteUserInterceptor{replacement: `{"type":"user","message":{"role":"user","content":"annotated"}}`}

	var out bytes.Buffer
	err := copyLines(strings.NewReader(input), &out, newTrafficLogger("", io.Discard), newObserver("", io.Discard), interceptor, protocol.ToBackend)
	if err != nil {
		t.Fatal(err)
	}

	want := `{"type":"control_request","request_id":"r1","request":{"subtype":"initialize"}}` + "\n" +
		`{"type":"user","message":{"role":"user","content":"annotated"}}` + "\n" +
		"not json\n"
	if out.String() != want {
		t.Errorf("got:\n%swant:\n%s", out.String(), want)
	}
	if len(interceptor.seenTypes) != 3 {
		t.Errorf("interceptor saw %d lines, want 3", len(interceptor.seenTypes))
	}
}

func TestCopyLines_nilReturnKeepsPassthroughByteIdentical(t *testing.T) {
	// An interceptor that always declines must leave the stream untouched,
	// including CRLF and non-JSON lines.
	input := "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"x\"}}\r\nstray\n"
	decline := interceptorFunc(func(line []byte, event protocol.Event) []byte { return nil })

	var out bytes.Buffer
	err := copyLines(strings.NewReader(input), &out, newTrafficLogger("", io.Discard), newObserver("", io.Discard), decline, protocol.ToBackend)
	if err != nil {
		t.Fatal(err)
	}
	if out.String() != input {
		t.Errorf("passthrough altered: got %q, want %q", out.String(), input)
	}
}

func TestCopyLines_malformedReplacementIsDiscarded(t *testing.T) {
	input := `{"type":"user","message":{"role":"user","content":"x"}}` + "\n"
	tests := []struct {
		desc        string
		replacement string
	}{
		{desc: "missing trailing newline", replacement: `{"type":"user"}`},
		{desc: "embedded newline splits the frame", replacement: "{\"type\":\"user\"}\ninjected\n"},
		{desc: "empty replacement", replacement: ""},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			bad := interceptorFunc(func(line []byte, event protocol.Event) []byte {
				return []byte(test.replacement)
			})
			var out bytes.Buffer
			err := copyLines(strings.NewReader(input), &out, newTrafficLogger("", io.Discard), newObserver("", io.Discard), bad, protocol.ToBackend)
			if err != nil {
				t.Fatal(err)
			}
			if out.String() != input {
				t.Errorf("malformed replacement must be discarded: got %q, want original %q", out.String(), input)
			}
		})
	}
}

type interceptorFunc func(line []byte, event protocol.Event) []byte

func (f interceptorFunc) InterceptToBackend(line []byte, event protocol.Event) []byte {
	return f(line, event)
}
