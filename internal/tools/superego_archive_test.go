package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vingarcia/kortex/internal/anthropic"
)

const (
	archiveIDA = "0123456789abcdef0123456789abcdef"
	archiveIDB = "fedcba9876543210fedcba9876543210"
)

func writeArchive(t *testing.T, lines ...string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "x.conflicts")
	if err := os.WriteFile(p, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRunLoadSuperegoArchive(t *testing.T) {
	lineA := `{"loopId":"` + archiveIDA + `","tail":[{"type":"draft","text":"bad draft"},{"type":"critique","text":"too rude"}]}`
	lineB := `{"loopId":"` + archiveIDB + `","note":"` + archiveIDA + `","tail":[]}`
	path := writeArchive(t, lineB, lineA)

	t.Run("found, with and without #", func(t *testing.T) {
		for _, id := range []string{archiveIDA, "#" + archiveIDA} {
			out, isErr := RunLoadSuperegoArchive(path, SuperegoArchiveInput{ID: id})
			if isErr || out != lineA {
				t.Errorf("id %q: out=%q isErr=%v, want line A", id, out, isErr)
			}
		}
	})
	t.Run("id only mentioned in another field is not a match", func(t *testing.T) {
		out, isErr := RunLoadSuperegoArchive(writeArchive(t, lineB), SuperegoArchiveInput{ID: archiveIDA})
		if !isErr || !strings.Contains(out, "no superego archive found") {
			t.Errorf("out=%q isErr=%v, want not-found error", out, isErr)
		}
	})
	t.Run("not found", func(t *testing.T) {
		out, isErr := RunLoadSuperegoArchive(path, SuperegoArchiveInput{ID: strings.Repeat("a", 32)})
		if !isErr || !strings.Contains(out, "no superego archive found") {
			t.Errorf("out=%q isErr=%v", out, isErr)
		}
	})
	t.Run("invalid id", func(t *testing.T) {
		for _, id := range []string{"", "#", "abc", "loop-abc123", strings.Repeat("g", 32), archiveIDA + "0"} {
			out, isErr := RunLoadSuperegoArchive(path, SuperegoArchiveInput{ID: id})
			if !isErr || !strings.Contains(out, "invalid archive id") {
				t.Errorf("id %q: out=%q isErr=%v, want invalid-id error", id, out, isErr)
			}
		}
	})
	t.Run("missing file", func(t *testing.T) {
		if _, isErr := RunLoadSuperegoArchive(filepath.Join(t.TempDir(), "nope"), SuperegoArchiveInput{ID: archiveIDA}); !isErr {
			t.Error("want error for missing file")
		}
	})
	t.Run("truncation", func(t *testing.T) {
		big := `{"loopId":"` + archiveIDA + `","tail":"` + strings.Repeat("é", superegoArchiveMaxBytes) + `"}`
		out, isErr := RunLoadSuperegoArchive(writeArchive(t, big), SuperegoArchiveInput{ID: archiveIDA})
		if isErr || !strings.Contains(out, "[truncated: showing first") {
			t.Fatalf("isErr=%v, want truncation note; out tail=%q", isErr, out[max(0, len(out)-80):])
		}
		if len(out) > superegoArchiveMaxBytes+100 {
			t.Errorf("output %d bytes, want bounded near %d", len(out), superegoArchiveMaxBytes)
		}
	})
}

func TestDispatcher_SuperegoArchiveToolOnlyWhenWired(t *testing.T) {
	has := func(d *Dispatcher) bool {
		for _, def := range d.ToolDefs() {
			if def.Name == "load_superego_archive" {
				return true
			}
		}
		return false
	}
	if has(NewDispatcher()) {
		t.Error("tool declared without a conflicts log")
	}
	path := writeArchive(t, `{"loopId":"`+archiveIDA+`"}`)
	d := NewDispatcher().WithSuperegoArchive(path)
	if !has(d) {
		t.Fatal("tool not declared when wired")
	}
	in, _ := json.Marshal(SuperegoArchiveInput{ID: "#" + archiveIDA})
	res := d.Execute(context.Background(), anthropic.ContentBlock{Type: "tool_use", ID: "t1", Name: "load_superego_archive", Input: in})
	if res.IsError || !strings.Contains(res.Content, archiveIDA) {
		t.Errorf("result = %+v", res)
	}
	res = NewDispatcher().Execute(context.Background(), anthropic.ContentBlock{Type: "tool_use", ID: "t2", Name: "load_superego_archive", Input: in})
	if !res.IsError {
		t.Error("unwired dispatcher must reject the tool")
	}
}
