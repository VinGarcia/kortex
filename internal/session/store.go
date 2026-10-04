// Package session runs kortex as the native stream-json producer: instead of
// execing the real claude-cli, it reads the OpenClaw NDJSON protocol from
// stdin, drives the tool-loop once per user turn, and emits stream-json on
// stdout — indistinguishable to OpenClaw from the real CLI. It persists the
// canonical conversation per session id so a turn survives a process restart
// (OpenClaw respawns kortex with --resume <id> to continue a session).
//
// v0 scope: the native path does NOT emit hook callbacks, so user-scope hooks
// (UserPromptSubmit/PreToolUse that OpenClaw declares in initialize) simply do
// not fire; it does not implement interrupts or permission prompts (kortex
// drives tools itself and never asks to use one). These are deliberate v0
// limitations, not bugs.
package session

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/vingarcia/kortex/internal/anthropic"
)

// Store persists each session's canonical history under a directory, one JSON
// file per session id, so a cross-process resume continues the same
// conversation.
type Store struct {
	dir string
}

// NewStore builds a Store rooted at dir. The directory is created lazily on the
// first Save, so an unused store touches nothing.
func NewStore(dir string) *Store {
	return &Store{dir: dir}
}

// Load returns the persisted history for sessionID, or nil when none exists yet
// (a brand-new session). A present-but-corrupt file is an error rather than a
// silent fresh start: dropping a real conversation's context would be a worse
// failure than refusing to continue it.
func (s *Store) Load(sessionID string) ([]anthropic.Message, error) {
	data, err := os.ReadFile(s.path(sessionID))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("session: reading history for %q: %w", sessionID, err)
	}
	var messages []anthropic.Message
	if err := json.Unmarshal(data, &messages); err != nil {
		return nil, fmt.Errorf("session: decoding history for %q: %w", sessionID, err)
	}
	return messages, nil
}

// Save writes history for sessionID atomically — a temp file in the same
// directory renamed over the target — so a crash mid-write cannot leave a
// truncated file that fails to load on the next resume.
func (s *Store) Save(sessionID string, messages []anthropic.Message) error {
	if err := os.MkdirAll(s.dir, 0o700); err != nil {
		return fmt.Errorf("session: creating state dir: %w", err)
	}
	data, err := json.Marshal(messages)
	if err != nil {
		return fmt.Errorf("session: encoding history for %q: %w", sessionID, err)
	}
	tmp, err := os.CreateTemp(s.dir, fileName(sessionID)+".*.tmp")
	if err != nil {
		return fmt.Errorf("session: creating temp history file: %w", err)
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("session: writing history for %q: %w", sessionID, err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("session: closing history for %q: %w", sessionID, err)
	}
	if err := os.Rename(tmpPath, s.path(sessionID)); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("session: committing history for %q: %w", sessionID, err)
	}
	return nil
}

func (s *Store) path(sessionID string) string {
	return filepath.Join(s.dir, fileName(sessionID)+".json")
}

// fileName maps a session id to a flat, traversal-safe filename: every
// character outside [A-Za-z0-9._-] becomes '_'. OpenClaw's session ids are
// already uuid-like, but sanitizing guards against a crafted id escaping the
// state directory.
func fileName(sessionID string) string {
	safe := make([]byte, 0, len(sessionID))
	for i := 0; i < len(sessionID); i++ {
		c := sessionID[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '.' || c == '-':
			safe = append(safe, c)
		default:
			safe = append(safe, '_')
		}
	}
	if len(safe) == 0 {
		return "_"
	}
	return string(safe)
}
