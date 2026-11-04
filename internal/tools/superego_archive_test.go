package tools

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vingarcia/kortex/internal/anthropic"
)

// fakeLoader is an ArchiveLoader returning a canned result and recording the id.
type fakeLoader struct {
	content string
	err     error
	gotID   string
}

func (f *fakeLoader) LoadArchive(id string) (string, error) {
	f.gotID = id
	return f.content, f.err
}

func TestRunLoadSuperegoArchive(t *testing.T) {
	for name, tc := range map[string]struct {
		loader      *fakeLoader
		wantErr     bool
		wantContain string
		maxLen      int
	}{
		"found":      {loader: &fakeLoader{content: `{"loopId":"x"}`}, wantContain: `{"loopId":"x"}`},
		"loader err": {loader: &fakeLoader{err: errors.New("no superego archive found")}, wantErr: true, wantContain: "no superego archive found"},
		"truncation": {loader: &fakeLoader{content: strings.Repeat("\u00e9", superegoArchiveMaxBytes)}, wantContain: "[truncated: showing first", maxLen: superegoArchiveMaxBytes + 100},
	} {
		t.Run(name, func(t *testing.T) {
			out, isErr := RunLoadSuperegoArchive(tc.loader, SuperegoArchiveInput{ID: "#abc"})
			if isErr != tc.wantErr || !strings.Contains(out, tc.wantContain) {
				t.Errorf("out=%.80q isErr=%v, want isErr=%v containing %q", out, isErr, tc.wantErr, tc.wantContain)
			}
			if tc.maxLen > 0 && len(out) > tc.maxLen {
				t.Errorf("output %d bytes, want <= %d", len(out), tc.maxLen)
			}
			if tc.loader.gotID != "#abc" {
				t.Errorf("loader got id %q, want it passed through unchanged", tc.loader.gotID)
			}
		})
	}
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
		t.Error("tool declared without an archive loader")
	}
	if has(NewDispatcher().WithSuperegoArchive(nil)) {
		t.Error("tool declared with a nil loader")
	}
	d := NewDispatcher().WithSuperegoArchive(&fakeLoader{content: "archived"})
	if !has(d) {
		t.Fatal("tool not declared when wired")
	}
	in, _ := json.Marshal(SuperegoArchiveInput{ID: "abc"})
	res := d.Execute(context.Background(), anthropic.ContentBlock{Type: "tool_use", ID: "t1", Name: "load_superego_archive", Input: in})
	if res.IsError || res.Content != "archived" {
		t.Errorf("result = %+v", res)
	}
	res = NewDispatcher().Execute(context.Background(), anthropic.ContentBlock{Type: "tool_use", ID: "t2", Name: "load_superego_archive", Input: in})
	if !res.IsError {
		t.Error("unwired dispatcher must reject the tool")
	}
}
