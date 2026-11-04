package tools

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

// superegoArchiveMaxBytes caps the archive content returned in the
// tool_result, so loading one archived loop cannot blow up the next request.
const superegoArchiveMaxBytes = 32 << 10

var superegoArchiveIDRe = regexp.MustCompile(`^[0-9a-f]{32}$`)

// SuperegoArchiveInput is the load_superego_archive tool's input_schema.
type SuperegoArchiveInput struct {
	// ID is the loop id from a "[superego archive #<id>]" reference; a leading
	// "#" is accepted.
	ID string `json:"id"`
}

// SuperegoArchiveDef is the load_superego_archive declaration for the
// Messages API "tools" field. It is only declared when the dispatcher was
// given a conflicts log (see Dispatcher.WithSuperegoArchive).
var SuperegoArchiveDef = mustToolDef(
	"load_superego_archive",
	"Load an archived superego loop by id: the rejected drafts and the superego's critiques behind one of your earlier replies. Use the id from a `[superego archive #<id>]` reference in your history.",
	map[string]any{
		"type":       "object",
		"properties": map[string]any{"id": map[string]any{"type": "string", "description": "The 32-hex-character archive id, with or without a leading '#'."}},
		"required":   []string{"id"},
	},
)

// RunLoadSuperegoArchive scans the conflicts log at path for the JSONL line
// whose "loopId" equals the requested id and returns that line's content,
// truncated to superegoArchiveMaxBytes with a note when cut.
func RunLoadSuperegoArchive(path string, input SuperegoArchiveInput) (output string, isError bool) {
	id := strings.TrimPrefix(strings.TrimSpace(input.ID), "#")
	if !superegoArchiveIDRe.MatchString(id) {
		return fmt.Sprintf("error: invalid archive id %q: want 32 lowercase hex characters (optionally prefixed with '#')", input.ID), true
	}
	f, err := os.Open(path)
	if err != nil {
		return fmt.Sprintf("error: opening superego archive: %v", err), true
	}
	defer f.Close()
	r := bufio.NewReader(f)
	for {
		line, readErr := r.ReadBytes('\n')
		// Cheap substring prefilter before paying for a JSON decode.
		if bytes.Contains(line, []byte(id)) {
			var entry struct {
				LoopID string `json:"loopId"`
			}
			if json.Unmarshal(line, &entry) == nil && entry.LoopID == id {
				line = bytes.TrimRight(line, "\n")
				cut, truncated := truncateUTF8(line, superegoArchiveMaxBytes)
				if truncated {
					return fmt.Sprintf("%s\n[truncated: showing first %d of %d bytes]", cut, len(cut), len(line)), false
				}
				return string(cut), false
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return fmt.Sprintf("error: no superego archive found with id %s", id), true
			}
			return fmt.Sprintf("error: reading superego archive: %v", readErr), true
		}
	}
}
