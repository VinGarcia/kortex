package tools

import "fmt"

// superegoArchiveMaxBytes caps the archive content returned in the
// tool_result, so loading one archived loop cannot blow up the next request.
const superegoArchiveMaxBytes = 32 << 10

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

// ArchiveLoader is the port through which load_superego_archive reads an
// archived loop. The implementation (facet.ConflictsArchive) owns the on-disk
// format and id validation; tools only caps the output size.
type ArchiveLoader interface {
	// LoadArchive returns the archived loop for id, or an error whose message
	// is safe to show the model (invalid id, not found, unreadable).
	LoadArchive(id string) (string, error)
}

// RunLoadSuperegoArchive loads the archive for the requested id through
// loader and truncates the content to superegoArchiveMaxBytes with a note
// when cut.
func RunLoadSuperegoArchive(loader ArchiveLoader, input SuperegoArchiveInput) (output string, isError bool) {
	content, err := loader.LoadArchive(input.ID)
	if err != nil {
		return fmt.Sprintf("error: %v", err), true
	}
	cut, truncated := TruncateUTF8([]byte(content), superegoArchiveMaxBytes)
	if truncated {
		return fmt.Sprintf("%s\n[truncated: showing first %d of %d bytes]", cut, len(cut), len(content)), false
	}
	return string(cut), false
}
