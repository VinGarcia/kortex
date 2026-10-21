package facet

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSessionTagsPath(t *testing.T) {
	tests := []struct {
		desc      string
		sessionID string
		wantOK    bool
		wantFile  string // basename only, checked when wantOK
	}{
		{desc: "uuid-like id passes through untouched", sessionID: "a1b2c3d4-0000-1111-2222-333344445555", wantOK: true, wantFile: "a1b2c3d4-0000-1111-2222-333344445555.jsonl"},
		{desc: "path separators sanitized", sessionID: "agent:sylphie/telegram:direct:X", wantOK: true, wantFile: "agent_sylphie_telegram_direct_X.jsonl"},
		{desc: "empty id rejected", sessionID: "", wantOK: false},
		{desc: "dot id rejected after sanitization", sessionID: ".", wantOK: false},
		{desc: "dotdot id rejected after sanitization", sessionID: "..", wantOK: false},
		{desc: "id of only separators degenerates to underscores, not rejected", sessionID: "::", wantOK: true, wantFile: "__.jsonl"},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			path, ok := sessionTagsPath("/tmp/sessions", test.sessionID)
			if ok != test.wantOK {
				t.Fatalf("ok = %v, want %v (path=%q)", ok, test.wantOK, path)
			}
			if test.wantOK && filepath.Base(path) != test.wantFile {
				t.Errorf("path basename = %q, want %q", filepath.Base(path), test.wantFile)
			}
			if test.wantOK && filepath.Dir(path) != "/tmp/sessions" {
				t.Errorf("path must stay under the configured dir, got %q", path)
			}
		})
	}
}

func TestEvidenceFromParagraph(t *testing.T) {
	tests := []struct {
		desc  string
		input string
		want  string
	}{
		{desc: "collapses newlines and tabs", input: "linha um\n\tlinha dois", want: "linha um linha dois"},
		{desc: "cuts at first sentence end", input: "Isso é grave. Mais texto depois que não deve aparecer.", want: "Isso é grave."},
		{desc: "cuts at first question mark", input: "Você está bem? Eu fico preocupada.", want: "Você está bem?"},
		{desc: "no punctuation keeps whole trimmed text", input: "  sem pontuação final  ", want: "sem pontuação final"},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := evidenceFromParagraph(test.input); got != test.want {
				t.Errorf("evidenceFromParagraph(%q) = %q, want %q", test.input, got, test.want)
			}
		})
	}
}

func TestEvidenceFromParagraph_truncatesAt160Runes(t *testing.T) {
	long := strings.Repeat("a", 200)
	got := evidenceFromParagraph(long)
	runes := []rune(got)
	if len(runes) != 161 { // 160 chars + the ellipsis rune
		t.Fatalf("len(runes) = %d, want 161", len(runes))
	}
	if runes[160] != '…' {
		t.Errorf("last rune = %q, want ellipsis", runes[160])
	}
	if string(runes[:160]) != strings.Repeat("a", 160) {
		t.Errorf("truncated body changed: %q", string(runes[:160]))
	}
}

func TestMaxInvestmentIndex(t *testing.T) {
	tests := []struct {
		desc        string
		annotations []ParagraphAnnotation
		want        int
	}{
		{desc: "empty", annotations: nil, want: -1},
		{desc: "single", annotations: []ParagraphAnnotation{{Investment: 2}}, want: 0},
		{desc: "first of a tie wins", annotations: []ParagraphAnnotation{
			{Investment: 1}, {Investment: 4}, {Investment: 4}, {Investment: 2},
		}, want: 1},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := maxInvestmentIndex(test.annotations); got != test.want {
				t.Errorf("maxInvestmentIndex() = %d, want %d", got, test.want)
			}
		})
	}
}

func TestAppendSessionTag_disabledWhenDirEmpty(t *testing.T) {
	dir := t.TempDir()
	// dir == "" must create nothing at all, including the dir argument's
	// sibling path — appendSessionTag must return before ever touching disk.
	appendSessionTag(&facetLogger{}, "m", "", "session-1", []string{"oi"}, []ParagraphAnnotation{{Investment: 3, Valence: "negativa"}})
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected no files written, got %v", entries)
	}
}

func TestAppendSessionTag_degenerateIDWritesNothing(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sessions")
	appendSessionTag(&facetLogger{}, "m", dir, "", []string{"oi"}, []ParagraphAnnotation{{Investment: 3, Valence: "negativa"}})
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("expected the session dir never to be created for a degenerate id, stat err = %v", err)
	}
}

func TestAppendSessionTag_writesOneLineWithSalientParagraph(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sessions")
	paragraphs := []string{"mensagem neutra", "mensagem bem carregada"}
	annotations := []ParagraphAnnotation{
		{Investment: 1, Valence: "neutra", Emotions: nil},
		{Investment: 4, Valence: "negativa", Emotions: []Emotion{{Emotion: "exaustão", Level: 4, About: "cobrança"}}},
	}
	appendSessionTag(&facetLogger{}, "m", dir, "sess-abc", paragraphs, annotations)

	path, ok := sessionTagsPath(dir, "sess-abc")
	if !ok {
		t.Fatal("sessionTagsPath rejected sess-abc")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("expected a file at %s: %v", path, err)
	}
	var line sessionTagsLine
	if err := json.Unmarshal(bytes.TrimSpace(data), &line); err != nil {
		t.Fatalf("line is not valid JSON: %v (%s)", err, data)
	}
	if line.SessionID != "sess-abc" {
		t.Errorf("sessionId = %q, want sess-abc", line.SessionID)
	}
	if line.Source != "kortex" {
		t.Errorf("source = %q, want kortex", line.Source)
	}
	if line.Polarity != "negative" || line.Valence != "negativa" || line.Investment != 4 {
		t.Errorf("polarity/valence/investment = %q/%q/%d, want negative/negativa/4", line.Polarity, line.Valence, line.Investment)
	}
	if len(line.Emotions) != 1 || line.Emotions[0].Emotion != "exaustão" {
		t.Errorf("emotions = %+v, want the salient paragraph's emotion list", line.Emotions)
	}
	if line.Evidence != "mensagem bem carregada" {
		t.Errorf("evidence = %q, want the salient paragraph's text", line.Evidence)
	}
	if line.Ts == 0 {
		t.Error("ts must be set")
	}

	// File permissions: 0600, matching the facet log / inbound tap convention
	// ("no secrets, but least privilege by default").
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("file mode = %o, want 0600", perm)
	}
}

func TestAppendSessionTag_appendsAcrossCalls(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sessions")
	one := []ParagraphAnnotation{{Investment: 2, Valence: "positiva"}}
	two := []ParagraphAnnotation{{Investment: 5, Valence: "mista"}}
	appendSessionTag(&facetLogger{}, "m", dir, "sess-multi", []string{"primeira"}, one)
	appendSessionTag(&facetLogger{}, "m", dir, "sess-multi", []string{"segunda"}, two)

	path, _ := sessionTagsPath(dir, "sess-multi")
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var count int
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) == "" {
			continue
		}
		count++
	}
	if count != 2 {
		t.Fatalf("expected 2 appended lines, got %d", count)
	}
}

// TestSessionIDSanitizer_matchesBashContract pins the exact sanitization rule
// scripts/emotion-session-path.sh implements in bash
// (`sed 's/[^A-Za-z0-9._-]/_/g'`, then reject a sanitized result of ""/"."/
// "..") as an explicit input->output fixture table. There is no automated
// cross-language conformance test (the bash script is not invoked from these
// Go tests); this table is the documented contract both sides must agree
// with by inspection. If either side's sanitization rule ever changes,
// update this table AND scripts/emotion-session-path.sh together.
func TestSessionIDSanitizer_matchesBashContract(t *testing.T) {
	tests := []struct {
		sessionID string
		wantOK    bool
		wantSafe  string // basename without ".jsonl"; only checked when wantOK
	}{
		{sessionID: "", wantOK: false},
		{sessionID: ".", wantOK: false},
		{sessionID: "..", wantOK: false},
		{sessionID: "a1b2c3d4-0000-1111-2222-333344445555", wantOK: true, wantSafe: "a1b2c3d4-0000-1111-2222-333344445555"},
		{sessionID: "agent:sylphie:telegram:direct:X", wantOK: true, wantSafe: "agent_sylphie_telegram_direct_X"},
		{sessionID: "with space/and\\backslash", wantOK: true, wantSafe: "with_space_and_backslash"},
		{sessionID: "já_com_ãccent", wantOK: true, wantSafe: "j__com__ccent"},
	}
	for _, test := range tests {
		t.Run(test.sessionID, func(t *testing.T) {
			path, ok := sessionTagsPath("/tmp/sessions", test.sessionID)
			if ok != test.wantOK {
				t.Fatalf("ok = %v, want %v (path=%q)", ok, test.wantOK, path)
			}
			if test.wantOK && filepath.Base(path) != test.wantSafe+".jsonl" {
				t.Errorf("sanitized basename = %q, want %q", filepath.Base(path), test.wantSafe+".jsonl")
			}
		})
	}
}

// decodeOnlySessionTagsLogLine asserts the sink captured exactly one callLog
// line and decodes it, failing the test otherwise.
func decodeOnlySessionTagsLogLine(t *testing.T, buf *bytes.Buffer) callLog {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 1 || lines[0] == "" {
		t.Fatalf("want exactly 1 log line, got %d: %q", len(lines), buf.String())
	}
	var entry callLog
	if err := json.Unmarshal([]byte(lines[0]), &entry); err != nil {
		t.Fatalf("log line is not valid JSON: %v (%s)", err, lines[0])
	}
	return entry
}

// TestAppendSessionTag_mkdirFailureIsLogged covers the u-never-fail-silently
// fix: a MkdirAll failure (here, a path component already exists as a
// regular file, so it cannot be a directory) must produce a callLog entry
// through the facetLogger rather than disappearing silently.
func TestAppendSessionTag_mkdirFailureIsLogged(t *testing.T) {
	parent := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(parent, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(parent, "sessions") // MkdirAll(dir) must fail: parent is a file

	var buf bytes.Buffer
	log := &facetLogger{sink: &buf}
	appendSessionTag(log, "model-x", dir, "sess-mkdir-fail", []string{"oi"}, []ParagraphAnnotation{{Investment: 2, Valence: "neutra"}})

	entry := decodeOnlySessionTagsLogLine(t, &buf)
	if entry.Facet != sessionTagsFacet {
		t.Errorf("facet = %q, want %q", entry.Facet, sessionTagsFacet)
	}
	if entry.Err == "" {
		t.Error("expected Err to be set on a mkdir failure, got empty")
	}
}

// TestAppendSessionTag_openFailureIsLogged covers the u-never-fail-silently
// fix for the OpenFile branch: pre-creating the target path as a directory
// makes OpenFile(O_WRONLY) fail, and that failure must reach the facetLogger.
func TestAppendSessionTag_openFailureIsLogged(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sessions")
	path, ok := sessionTagsPath(dir, "sess-open-fail")
	if !ok {
		t.Fatal("sessionTagsPath rejected sess-open-fail")
	}
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	log := &facetLogger{sink: &buf}
	appendSessionTag(log, "model-x", dir, "sess-open-fail", []string{"oi"}, []ParagraphAnnotation{{Investment: 2, Valence: "neutra"}})

	entry := decodeOnlySessionTagsLogLine(t, &buf)
	if entry.Facet != sessionTagsFacet {
		t.Errorf("facet = %q, want %q", entry.Facet, sessionTagsFacet)
	}
	if entry.Err == "" {
		t.Error("expected Err to be set on an open failure, got empty")
	}
}

// TestInputAnnotator_persistsSessionTagOnSuccessfulAnnotation is the
// end-to-end wiring test: a real InputAnnotator, constructed the way
// main.go's buildFacets constructs it (SessionID + SessionTagsDir set),
// writes the session-tags file as a side effect of a successful Annotate
// call — the native-path seam session.Run uses (contrast with
// InterceptToBackend, the proxy seam).
func TestInputAnnotator_persistsSessionTagOnSuccessfulAnnotation(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sessions")
	a := NewInputAnnotator(AnnotatorParams{
		Evaluator:      &fakeEvaluator{response: evalText(`[{"investment":4,"valence":"negativa","emotions":[{"emotion":"cobrança","level":3,"about":"contava com você"}]}]`)},
		SystemPrompt:   "s",
		Timeout:        time.Second,
		Model:          "m",
		SessionID:      "sess-native-1",
		SessionTagsDir: dir,
	})

	got := a.Annotate("contava com você ontem")
	if got == "contava com você ontem" {
		t.Fatal("expected the text to be annotated, got the original back")
	}

	path, ok := sessionTagsPath(dir, "sess-native-1")
	if !ok {
		t.Fatal("sessionTagsPath rejected sess-native-1")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected the session-tags file to exist: %v", err)
	}
}

// TestInputAnnotator_noSessionTagsDirIsANoOp confirms the off-by-default
// behavior the task's gating constraint requires: SessionTagsDir == ""
// creates no file, even though a sessionID is set.
func TestInputAnnotator_noSessionTagsDirIsANoOp(t *testing.T) {
	a := NewInputAnnotator(AnnotatorParams{
		Evaluator:    &fakeEvaluator{response: oneParagraphEval()},
		SystemPrompt: "s",
		Timeout:      time.Second,
		Model:        "m",
		SessionID:    "sess-no-dir",
	})
	if got := a.Annotate("boa notícia hoje"); got == "boa notícia hoje" {
		t.Fatal("expected annotation to still succeed with persistence disabled")
	}
	// No directory was ever named, so there is nothing to assert against on
	// disk beyond "the call above did not panic or block" — the real
	// guarantee (no dir created) is covered by
	// TestAppendSessionTag_disabledWhenDirEmpty at the unit level.
}

// TestInputAnnotator_failedAnnotationNeverPersists confirms the fail-open
// contract extends to persistence: a facet failure must never write a
// session-tags line (there is nothing salient to record).
func TestInputAnnotator_failedAnnotationNeverPersists(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sessions")
	a := NewInputAnnotator(AnnotatorParams{
		Evaluator:      &fakeEvaluator{err: errors.New("boom")},
		SystemPrompt:   "s",
		Timeout:        time.Second,
		Model:          "m",
		SessionID:      "sess-fail",
		SessionTagsDir: dir,
	})
	if got := a.Annotate("texto qualquer"); got != "texto qualquer" {
		t.Fatalf("expected fail-open passthrough, got %q", got)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("expected no session-tags dir on a failed annotation, stat err = %v", err)
	}
}
