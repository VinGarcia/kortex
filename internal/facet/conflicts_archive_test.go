package facet

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	archiveIDA = "0123456789abcdef0123456789abcdef"
	archiveIDB = "fedcba9876543210fedcba9876543210"
)

func writeConflicts(t *testing.T, lines ...string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "x.conflicts")
	if err := os.WriteFile(p, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestConflictsArchive_LoadArchive(t *testing.T) {
	lineA := `{"loopId":"` + archiveIDA + `","tail":[{"type":"draft","text":"bad draft"}]}`
	lineB := `{"loopId":"` + archiveIDB + `","note":"` + archiveIDA + `","tail":[]}`
	path := writeConflicts(t, lineB, lineA)

	for name, tc := range map[string]struct {
		path    string
		id      string
		want    string
		wantErr string
	}{
		"found":                    {path: path, id: archiveIDA, want: lineA},
		"found with #":             {path: path, id: "#" + archiveIDA, want: lineA},
		"id only in another field": {path: writeConflicts(t, lineB), id: archiveIDA, wantErr: "no superego archive found"},
		"not found":                {path: path, id: strings.Repeat("a", 32), wantErr: "no superego archive found"},
		"empty id":                 {path: path, id: "", wantErr: "invalid archive id"},
		"hash only":                {path: path, id: "#", wantErr: "invalid archive id"},
		"too short":                {path: path, id: "abc", wantErr: "invalid archive id"},
		"non-hex":                  {path: path, id: strings.Repeat("g", 32), wantErr: "invalid archive id"},
		"too long":                 {path: path, id: archiveIDA + "0", wantErr: "invalid archive id"},
		"missing file":             {path: filepath.Join(t.TempDir(), "nope"), id: archiveIDA, wantErr: "opening superego archive"},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := NewConflictsArchive(tc.path).LoadArchive(tc.id)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want containing %q", err, tc.wantErr)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Errorf("got %q err %v, want %q", got, err, tc.want)
			}
		})
	}
}

func TestConflictsArchive_OversizedLineIsBounded(t *testing.T) {
	huge := `{"loopId":"` + archiveIDB + `","tail":"` + strings.Repeat("x", maxArchiveLineBytes) + `"}`
	_, err := NewConflictsArchive(writeConflicts(t, huge)).LoadArchive(archiveIDA)
	if err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("err = %v, want line-too-long error", err)
	}
}

func TestConflictsArchive_RoundTripWithLoggerAndLoopID(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.conflicts")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	id := newLoopID()
	if err := (&conflictsLogger{sink: f}).record(ladderTailEntry{LoopID: id, Outcome: ladderOutcomeApprove}); err != nil {
		t.Fatal(err)
	}
	got, err := NewConflictsArchive(path).LoadArchive("#" + id)
	if err != nil || !strings.Contains(got, id) {
		t.Errorf("got %q err %v, want the line written by the logger", got, err)
	}
}

func TestActiveSuperego_ArchivePromptAndStrip(t *testing.T) {
	on := NewActiveSuperego(ActiveSuperegoParams{ConflictsLogSink: &strings.Builder{}})
	off := NewActiveSuperego(ActiveSuperegoParams{})
	if !strings.Contains(on.SystemPromptNote(), "load_superego_archive") {
		t.Error("archive-enabled governor must explain the archive")
	}
	if off.SystemPromptNote() != "" {
		t.Error("governor without conflicts log must not mention the archive tool")
	}
	oldFooter := "\n\n[superego conversation archived with ID #22219a3b - search the log]"
	for name, tc := range map[string]struct{ in, want string }{
		"new form":     {"hi\n\n[superego archive #abc]", "hi"},
		"old form":     {"hi" + oldFooter, "hi"},
		"mid text":     {"a [superego archive #abc] b", "a b"},
		"no reference": {"[other] text", "[other] text"},
	} {
		if got := on.StripArchiveRefs(tc.in); got != tc.want {
			t.Errorf("%s: got %q, want %q", name, got, tc.want)
		}
	}
}
