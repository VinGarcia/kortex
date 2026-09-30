package facet

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/protocol"
)

type fakeEvaluator struct {
	response Evaluation
	err      error
	delay    time.Duration
	calls    int
	gotSys   string
	gotUser  string
}

func (f *fakeEvaluator) Evaluate(ctx context.Context, system string, userMessage string) (Evaluation, error) {
	f.calls++
	f.gotSys = system
	f.gotUser = userMessage
	if f.delay > 0 {
		select {
		case <-ctx.Done():
			return Evaluation{}, ctx.Err()
		case <-time.After(f.delay):
		}
	}
	return f.response, f.err
}

func evalText(text string) Evaluation {
	return Evaluation{Text: text, StopReason: "end_turn", InputTokens: 100, OutputTokens: 20}
}

func newTestAnnotator(evaluator Evaluator) *InputAnnotator {
	return NewInputAnnotator(AnnotatorParams{
		Evaluator:    evaluator,
		SystemPrompt: "avalie as emoções",
		Timeout:      time.Second,
		Model:        "claude-haiku-4-5-20251001",
	})
}

func intercept(t *testing.T, a *InputAnnotator, line string) []byte {
	t.Helper()
	raw := []byte(line + "\n")
	return a.InterceptToBackend(raw, protocol.Parse(raw))
}

const oneParagraphTags = `[{"investment":2,"valence":"positiva","emotions":[{"emotion":"alegria","level":2,"about":"boa notícia"}]}]`

func oneParagraphEval() Evaluation { return evalText(oneParagraphTags) }

func TestInterceptToBackend_annotatesStringContent(t *testing.T) {
	evaluator := &fakeEvaluator{response: evalText(`[
		{"investment":4,"valence":"negativa","emotions":[{"emotion":"cobrança","level":3,"about":"contava com você"}]},
		{"investment":0,"valence":"neutra","emotions":[]}
	]`)}
	a := newTestAnnotator(evaluator)
	line := `{"type":"user","message":{"role":"user","content":"contava com você ontem\n\nroda o script"},"parent_tool_use_id":null,"uuid":"u-1","session_id":"s-1"}`

	got := intercept(t, a, line)
	if got == nil {
		t.Fatal("expected the line to be rewritten")
	}
	if got[len(got)-1] != '\n' || strings.Count(string(got), "\n") != 1 {
		t.Fatalf("rewritten line must be a single NDJSON line: %q", got)
	}

	var envelope struct {
		Type      string `json:"type"`
		UUID      string `json:"uuid"`
		SessionID string `json:"session_id"`
		Message   struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.Unmarshal(got, &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Type != "user" || envelope.UUID != "u-1" || envelope.SessionID != "s-1" || envelope.Message.Role != "user" {
		t.Errorf("envelope fields not preserved: %+v", envelope)
	}
	wantContent := "contava com você ontem\n" +
		"[emoções p1: investment=4 valence=negativa emotions=cobrança(3)]\n\n" +
		"roda o script\n" +
		"[emoções p2: investment=0 valence=neutra emotions=nenhuma]"
	if envelope.Message.Content != wantContent {
		t.Errorf("content:\n%s\nwant:\n%s", envelope.Message.Content, wantContent)
	}

	if evaluator.gotSys != "avalie as emoções" {
		t.Errorf("system prompt = %q", evaluator.gotSys)
	}
	wantUser := "<<<MENSAGEM_RECEBIDA_INICIO>>>\ncontava com você ontem\n\nroda o script\n<<<MENSAGEM_RECEBIDA_FIM>>>"
	if evaluator.gotUser != wantUser {
		t.Errorf("evaluator user message = %q, want %q", evaluator.gotUser, wantUser)
	}
}

func TestInterceptToBackend_annotatesSingleTextBlockPreservingOthers(t *testing.T) {
	a := newTestAnnotator(&fakeEvaluator{response: oneParagraphEval()})
	line := `{"type":"user","message":{"role":"user","content":[` +
		`{"type":"image","source":{"type":"base64","data":"aGk="}},` +
		`{"type":"text","text":"boa notícia hoje"}]},"uuid":"u-2"}`

	got := intercept(t, a, line)
	if got == nil {
		t.Fatal("expected the line to be rewritten")
	}
	var envelope struct {
		Message struct {
			Content []map[string]any `json:"content"`
		} `json:"message"`
	}
	if err := json.Unmarshal(got, &envelope); err != nil {
		t.Fatal(err)
	}
	if len(envelope.Message.Content) != 2 {
		t.Fatalf("want 2 blocks, got %d", len(envelope.Message.Content))
	}
	if envelope.Message.Content[0]["type"] != "image" {
		t.Errorf("image block not preserved at index 0: %v", envelope.Message.Content[0])
	}
	text, _ := envelope.Message.Content[1]["text"].(string)
	if !strings.HasPrefix(text, "boa notícia hoje\n[emoções p1: investment=2 valence=positiva") {
		t.Errorf("text block not annotated: %q", text)
	}
}

func TestInterceptToBackend_passthroughCases(t *testing.T) {
	tests := []struct {
		desc string
		line string
	}{
		{
			desc: "initialize handshake untouched",
			line: `{"type":"control_request","request_id":"r1","request":{"subtype":"initialize","appendSystemPrompt":"x"}}`,
		},
		{
			desc: "control_response untouched",
			line: `{"type":"control_response","response":{"subtype":"success","request_id":"r1"}}`,
		},
		{
			desc: "keep_alive untouched",
			line: `{"type":"keep_alive"}`,
		},
		{
			desc: "non-json line untouched",
			line: `not json at all`,
		},
		{
			desc: "replayed user message untouched",
			line: `{"type":"user","isReplay":true,"message":{"role":"user","content":"oi"}}`,
		},
		{
			desc: "tool_result carrier untouched",
			line: `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}}`,
		},
		{
			desc: "non-user role untouched",
			line: `{"type":"user","message":{"role":"assistant","content":"oi"}}`,
		},
		{
			desc: "user message without text untouched",
			line: `{"type":"user","message":{"role":"user","content":[{"type":"image","source":{}}]}}`,
		},
		{
			desc: "blank-only text untouched",
			line: `{"type":"user","message":{"role":"user","content":"\n \n"}}`,
		},
		{
			desc: "already-annotated text untouched (resume idempotence)",
			line: `{"type":"user","message":{"role":"user","content":"oi\n[emoções p1: investment=0 valence=neutra emotions=nenhuma]"}}`,
		},
		{
			desc: "two text blocks untouched (index desync risk)",
			line: `{"type":"user","message":{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			evaluator := &fakeEvaluator{response: oneParagraphEval()}
			a := newTestAnnotator(evaluator)
			if got := intercept(t, a, test.line); got != nil {
				t.Fatalf("want passthrough (nil), got %q", got)
			}
			if evaluator.calls != 0 {
				t.Errorf("evaluator called %d times for a structural passthrough", evaluator.calls)
			}
		})
	}
}

func TestInterceptToBackend_failOpen(t *testing.T) {
	userLine := `{"type":"user","message":{"role":"user","content":"mensagem real"},"uuid":"u-3"}`
	tests := []struct {
		desc      string
		evaluator *fakeEvaluator
	}{
		{desc: "api error", evaluator: &fakeEvaluator{err: errors.New("anthropic: http 529")}},
		{desc: "timeout", evaluator: &fakeEvaluator{delay: time.Minute, response: oneParagraphEval()}},
		{desc: "no json array in output", evaluator: &fakeEvaluator{response: evalText("desculpa, não posso ajudar")}},
		{desc: "malformed json array", evaluator: &fakeEvaluator{response: evalText(`[{"investment":`)}},
		{desc: "count mismatch", evaluator: &fakeEvaluator{response: evalText(`[]`)}},
		{desc: "schema violation", evaluator: &fakeEvaluator{response: evalText(`[{"investment":9,"valence":"neutra","emotions":[]}]`)}},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			a := NewInputAnnotator(AnnotatorParams{
				Evaluator:    test.evaluator,
				SystemPrompt: "s",
				Timeout:      50 * time.Millisecond,
				Model:        "m",
			})
			start := time.Now()
			if got := intercept(t, a, userLine); got != nil {
				t.Fatalf("want passthrough (nil) on failure, got %q", got)
			}
			if elapsed := time.Since(start); elapsed > 5*time.Second {
				t.Errorf("fail-open took %v; timeout not applied", elapsed)
			}
			if test.evaluator.calls != 1 {
				t.Errorf("evaluator calls = %d, want 1", test.evaluator.calls)
			}
		})
	}
}

func TestInputAnnotator_writesStructuredLog(t *testing.T) {
	var log bytes.Buffer
	a := NewInputAnnotator(AnnotatorParams{
		Evaluator:    &fakeEvaluator{response: oneParagraphEval()},
		SystemPrompt: "s",
		Timeout:      time.Second,
		Model:        "claude-haiku-4-5-20251001",
		LogSink:      &log,
	})
	if got := intercept(t, a, `{"type":"user","message":{"role":"user","content":"boa notícia hoje"}}`); got == nil {
		t.Fatal("expected annotation to succeed")
	}
	failing := NewInputAnnotator(AnnotatorParams{
		Evaluator:    &fakeEvaluator{err: errors.New("boom")},
		SystemPrompt: "s",
		Timeout:      time.Second,
		Model:        "claude-haiku-4-5-20251001",
		LogSink:      &log,
	})
	if got := intercept(t, failing, `{"type":"user","message":{"role":"user","content":"outra"}}`); got != nil {
		t.Fatal("expected fail-open")
	}

	lines := strings.Split(strings.TrimSpace(log.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 log lines, got %d: %s", len(lines), log.String())
	}
	var success, failure callLog
	if err := json.Unmarshal([]byte(lines[0]), &success); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &failure); err != nil {
		t.Fatal(err)
	}
	if !success.OK || success.Facet != "input_annotator" || success.Model != "claude-haiku-4-5-20251001" ||
		success.Paragraphs != 1 || success.Time == "" ||
		success.InputTokens != 100 || success.OutputTokens != 20 || success.StopReason != "end_turn" {
		t.Errorf("success entry: %+v", success)
	}
	if failure.OK || failure.Err != "boom" {
		t.Errorf("failure entry: %+v", failure)
	}
}

func TestLoadSystemPrompt(t *testing.T) {
	dir := t.TempDir()
	promptPath := filepath.Join(dir, "prompt.md")
	memoryPath := filepath.Join(dir, "memories.md")
	writeFile(t, promptPath, "# header nota de manutenção\n---\nCorpo do prompt.\n")
	writeFile(t, memoryPath, "E1: memória forte.\n")

	t.Run("body only", func(t *testing.T) {
		got, err := LoadSystemPrompt(promptPath, "")
		if err != nil {
			t.Fatal(err)
		}
		if got != "Corpo do prompt.\n" {
			t.Errorf("got %q", got)
		}
	})
	t.Run("memory appended as trusted context", func(t *testing.T) {
		got, err := LoadSystemPrompt(promptPath, memoryPath)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(got, "Corpo do prompt.\n") ||
			!strings.Contains(got, "CONTEXTO CONFIÁVEL — MEMÓRIA DE EMOÇÕES") ||
			!strings.Contains(got, "E1: memória forte.") {
			t.Errorf("got %q", got)
		}
	})
	t.Run("prompt without body separator fails", func(t *testing.T) {
		bare := filepath.Join(dir, "bare.md")
		writeFile(t, bare, "sem separador")
		if _, err := LoadSystemPrompt(bare, ""); err == nil {
			t.Fatal("want error for prompt without --- body")
		}
	})
	t.Run("missing memory file fails", func(t *testing.T) {
		if _, err := LoadSystemPrompt(promptPath, filepath.Join(dir, "nope.md")); err == nil {
			t.Fatal("want error for missing memory file")
		}
	})
}

func writeFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
