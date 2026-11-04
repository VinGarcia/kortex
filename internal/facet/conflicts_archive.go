package facet

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// maxArchiveLineBytes bounds the memory used per conflicts-log line while
// scanning. A line is one whole ladder loop (drafts + critiques), normally a
// few KiB; a longer line makes the scan fail with an error instead of being
// buffered without limit.
const maxArchiveLineBytes = 8 << 20

// archiveIDRe is the shape of a loop id as minted by newLoopID (16 random
// bytes, lowercase hex). It is validated here, next to the minting code, so
// the id format has one owner.
var archiveIDRe = regexp.MustCompile(`^[0-9a-f]{32}$`)

// ConflictsArchive reads loops back out of the conflicts log file. facet owns
// the on-disk shape of that file (one ladderTailEntry JSON object per line,
// keyed by "loopId"); no other package parses it. The tools package consumes
// this through its own ArchiveLoader port, wired by the composition root.
type ConflictsArchive struct {
	path string
}

// NewConflictsArchive returns a reader for the conflicts log at path.
func NewConflictsArchive(path string) *ConflictsArchive {
	return &ConflictsArchive{path: path}
}

// LoadArchive returns the raw JSONL line of the loop whose "loopId" equals id
// (a leading "#" is accepted), without its trailing newline. A line longer
// than maxArchiveLineBytes aborts the scan with an error.
func (a *ConflictsArchive) LoadArchive(id string) (string, error) {
	id = strings.TrimPrefix(strings.TrimSpace(id), "#")
	if !archiveIDRe.MatchString(id) {
		return "", fmt.Errorf("invalid archive id: want 32 lowercase hex characters (optionally prefixed with '#')")
	}
	f, err := os.Open(a.path)
	if err != nil {
		return "", fmt.Errorf("opening superego archive: %w", err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64<<10), maxArchiveLineBytes)
	needle := []byte(id)
	for sc.Scan() {
		line := sc.Bytes()
		// Cheap substring prefilter before paying for a JSON decode.
		if !bytes.Contains(line, needle) {
			continue
		}
		var entry struct {
			LoopID string `json:"loopId"`
		}
		if json.Unmarshal(line, &entry) == nil && entry.LoopID == id {
			return string(line), nil
		}
	}
	if err := sc.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			return "", fmt.Errorf("reading superego archive: a line exceeds %d bytes", maxArchiveLineBytes)
		}
		return "", fmt.Errorf("reading superego archive: %w", err)
	}
	return "", fmt.Errorf("no superego archive found with id %s", id)
}

// archiveSystemPromptNote is the one-time explanation of the archive
// reference rendered by retrievalHint, for the agent's system prompt.
const archiveSystemPromptNote = "\n\nSuperego archive: drafts you wrote and the superego's critiques of them are ephemeral and are archived outside the conversation. When that happens, your own earlier reply in this history ends with a short reference `[superego archive #<id>]` pointing to that archive; call the `load_superego_archive` tool with that id to read the rejected drafts and the superego's critiques. The reference is bookkeeping for you only: never write, copy, or imitate such a reference in your replies; the user never sees it."

// archiveRefRe matches the retrievalHint reference and the OLD long footer
// form ("[superego conversation archived with ID #<id> — ...]"), still present
// in older sessions' history and so still imitable.
var archiveRefRe = regexp.MustCompile(`\s*\[superego (?:archive|conversation archived with ID) #[^\]]*\]`)
