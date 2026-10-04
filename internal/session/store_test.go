package session

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vingarcia/kortex/internal/anthropic"
)

func TestStore_RoundTrip(t *testing.T) {
	store := NewStore(filepath.Join(t.TempDir(), "sessions"))
	history := []anthropic.Message{
		{Role: "user", Content: "my favorite color is blue"},
		{Role: "assistant", Content: "noted"},
	}

	if err := store.Save("sess-1", history); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, err := store.Load("sess-1")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(got) != 2 || got[0].Content != "my favorite color is blue" || got[1].Role != "assistant" {
		t.Errorf("round-tripped history = %+v, want the saved two messages", got)
	}
}

func TestStore_LoadMissingIsNilNotError(t *testing.T) {
	store := NewStore(t.TempDir())
	got, err := store.Load("never-saved")
	if err != nil {
		t.Fatalf("Load of a missing session errored: %v", err)
	}
	if got != nil {
		t.Errorf("Load of a missing session = %+v, want nil (fresh session)", got)
	}
}

func TestStore_LoadCorruptIsError(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)
	if err := os.WriteFile(filepath.Join(dir, "sess-x.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load("sess-x"); err == nil {
		t.Error("Load of a corrupt history returned nil error, want it to fail loudly")
	}
}

func TestStore_SaveSanitizesSessionID(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)
	// A session id with path separators must not escape the state directory.
	if err := store.Save("../../etc/evil", []anthropic.Message{{Role: "user", Content: "x"}}); err != nil {
		t.Fatalf("Save: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("state dir has %d entries, want exactly 1 (no traversal)", len(entries))
	}
	// The sanitized id round-trips back through Load.
	got, err := store.Load("../../etc/evil")
	if err != nil || len(got) != 1 {
		t.Errorf("Load after sanitized Save = %+v, err %v", got, err)
	}
}
