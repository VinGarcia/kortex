package tools

import (
	"fmt"
	"os"
	"strings"
)

// ReadInput is the read tool's input_schema.
type ReadInput struct {
	// Path is the absolute or relative file path to read.
	Path string `json:"path"`
}

// ReadDef is the read tool declaration for the Messages API "tools" field.
var ReadDef = mustToolDef(
	"read",
	"Read a file's full contents as text.",
	map[string]any{
		"type":       "object",
		"properties": map[string]any{"path": map[string]any{"type": "string", "description": "Path of the file to read."}},
		"required":   []string{"path"},
	},
)

// RunRead returns the file's contents, or an error tool_result when the
// path is empty, missing, or unreadable (e.g. a directory).
func RunRead(input ReadInput) (output string, isError bool) {
	if input.Path == "" {
		return "error: path is required", true
	}
	data, err := os.ReadFile(input.Path)
	if err != nil {
		return fmt.Sprintf("error: reading %s: %v", input.Path, err), true
	}
	return string(data), false
}

// WriteInput is the write tool's input_schema.
type WriteInput struct {
	// Path is the file to create or overwrite.
	Path string `json:"path"`
	// Content is the full file content to write.
	Content string `json:"content"`
}

// WriteDef is the write tool declaration for the Messages API "tools"
// field.
var WriteDef = mustToolDef(
	"write",
	"Write content to a file, creating it if it does not exist and overwriting it completely if it does.",
	map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":    map[string]any{"type": "string", "description": "Path of the file to write."},
			"content": map[string]any{"type": "string", "description": "Full content to write to the file."},
		},
		"required": []string{"path", "content"},
	},
)

// writeFileMode is the permission bits for a newly created file: readable
// and writable by the owner and group, readable by everyone — the same
// default os.WriteFile examples use for text/source files, with no execute
// bit since the builtin never writes scripts meant to run directly.
const writeFileMode = 0o644

// RunWrite creates or overwrites input.Path with input.Content.
func RunWrite(input WriteInput) (output string, isError bool) {
	if input.Path == "" {
		return "error: path is required", true
	}
	if err := os.WriteFile(input.Path, []byte(input.Content), writeFileMode); err != nil {
		return fmt.Sprintf("error: writing %s: %v", input.Path, err), true
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(input.Content), input.Path), false
}

// EditInput is the edit tool's input_schema: an exact string replacement,
// matching Claude Code's Edit tool semantics.
type EditInput struct {
	// Path is the file to edit.
	Path string `json:"path"`
	// OldString is the exact text to find. It must be unique in the file
	// unless ReplaceAll is set.
	OldString string `json:"old_string"`
	// NewString is the replacement text.
	NewString string `json:"new_string"`
	// ReplaceAll replaces every occurrence of OldString instead of
	// requiring exactly one.
	ReplaceAll bool `json:"replace_all,omitempty"`
}

// EditDef is the edit tool declaration for the Messages API "tools" field.
var EditDef = mustToolDef(
	"edit",
	"Replace an exact string match in a file with new text. old_string must match exactly one location in the file unless replace_all is set, otherwise the edit is rejected (no match, or ambiguous multiple matches) rather than guessing.",
	map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":        map[string]any{"type": "string", "description": "Path of the file to edit."},
			"old_string":  map[string]any{"type": "string", "description": "Exact text to find and replace."},
			"new_string":  map[string]any{"type": "string", "description": "Replacement text."},
			"replace_all": map[string]any{"type": "boolean", "description": "Replace every occurrence instead of requiring exactly one match."},
		},
		"required": []string{"path", "old_string", "new_string"},
	},
)

// RunEdit performs an exact string replacement in input.Path. Like Claude
// Code's Edit tool, it refuses (is_error:true) rather than guessing when
// old_string is absent, or when it is ambiguous (more than one match and
// ReplaceAll is false).
func RunEdit(input EditInput) (output string, isError bool) {
	if input.Path == "" {
		return "error: path is required", true
	}
	if input.OldString == "" {
		return "error: old_string is required", true
	}
	if input.OldString == input.NewString {
		return "error: old_string and new_string are identical, no edit to make", true
	}
	data, err := os.ReadFile(input.Path)
	if err != nil {
		return fmt.Sprintf("error: reading %s: %v", input.Path, err), true
	}
	content := string(data)
	count := strings.Count(content, input.OldString)
	if count == 0 {
		return fmt.Sprintf("error: old_string not found in %s", input.Path), true
	}
	if count > 1 && !input.ReplaceAll {
		return fmt.Sprintf("error: old_string matches %d locations in %s; make it unique or set replace_all", count, input.Path), true
	}

	var replaced string
	if input.ReplaceAll {
		replaced = strings.ReplaceAll(content, input.OldString, input.NewString)
	} else {
		replaced = strings.Replace(content, input.OldString, input.NewString, 1)
	}
	if err := os.WriteFile(input.Path, []byte(replaced), writeFileMode); err != nil {
		return fmt.Sprintf("error: writing %s: %v", input.Path, err), true
	}
	return fmt.Sprintf("replaced %d occurrence(s) in %s", count, input.Path), false
}
