package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestRunRead(t *testing.T) {
	dir := t.TempDir()
	okPath := filepath.Join(dir, "ok.txt")
	if err := os.WriteFile(okPath, []byte("hello world"), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		desc    string
		path    string
		wantOut string
		wantErr bool
	}{
		{desc: "ok", path: okPath, wantOut: "hello world", wantErr: false},
		{desc: "missing file", path: filepath.Join(dir, "missing.txt"), wantErr: true},
		{desc: "empty path", path: "", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			output, isError := RunRead(ReadInput{Path: test.path})
			if isError != test.wantErr {
				t.Errorf("isError = %v, want %v (output=%q)", isError, test.wantErr, output)
			}
			if !test.wantErr && output != test.wantOut {
				t.Errorf("output = %q, want %q", output, test.wantOut)
			}
		})
	}
}

// TestRunRead_capAtBoundaryIsUntouched writes a file exactly
// readMaxOutputBytes long and checks RunRead returns it byte-identical,
// with no truncation notice appended — the cap must only kick in once the
// file is strictly larger than it.
func TestRunRead_capAtBoundaryIsUntouched(t *testing.T) {
	content := strings.Repeat("a", readMaxOutputBytes)
	path := filepath.Join(t.TempDir(), "at-cap.txt")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	output, isError := RunRead(ReadInput{Path: path})
	if isError {
		t.Fatalf("unexpected error: %q", output)
	}
	if output != content {
		t.Errorf("output differs from the at-cap file content (len got=%d want=%d)", len(output), len(content))
	}
}

// TestRunRead_truncationIsValidUTF8 writes a file one byte past
// readMaxOutputBytes where the last byte within the cap is the lead byte of
// a 2-byte UTF-8 rune (0xC3 0xA9, "é") and its continuation byte falls just
// past the cap. A raw byte-offset cut would keep the lone 0xC3 and return
// invalid UTF-8; RunRead must instead back off that dangling byte.
func TestRunRead_truncationIsValidUTF8(t *testing.T) {
	data := append([]byte(strings.Repeat("a", readMaxOutputBytes-1)), 0xC3, 0xA9)
	path := filepath.Join(t.TempDir(), "over-cap.txt")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	output, isError := RunRead(ReadInput{Path: path})
	if isError {
		t.Fatalf("unexpected error: %q", output)
	}
	if !utf8.ValidString(output) {
		t.Errorf("output is not valid UTF-8: %q", output)
	}
	wantNotice := "[output truncated at 262144 bytes]"
	if !strings.Contains(output, wantNotice) {
		t.Errorf("output does not contain %q", wantNotice)
	}
	wantContent := strings.Repeat("a", readMaxOutputBytes-1)
	if !strings.HasPrefix(output, wantContent) {
		t.Errorf("output content was not the dangling-byte-free %d 'a's", readMaxOutputBytes-1)
	}
}

func TestRunWrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.txt")

	if _, isError := RunWrite(WriteInput{Path: path, Content: "first"}); isError {
		t.Fatalf("unexpected error creating file")
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != "first" {
		t.Fatalf("after create: got %q, err=%v", got, err)
	}

	if _, isError := RunWrite(WriteInput{Path: path, Content: "second"}); isError {
		t.Fatalf("unexpected error overwriting file")
	}
	got, err = os.ReadFile(path)
	if err != nil || string(got) != "second" {
		t.Fatalf("after overwrite: got %q, err=%v", got, err)
	}

	if _, isError := RunWrite(WriteInput{Path: ""}); !isError {
		t.Errorf("want error for empty path")
	}
}

func TestRunEdit(t *testing.T) {
	writeTemp := func(t *testing.T, content string) string {
		t.Helper()
		path := filepath.Join(t.TempDir(), "edit.txt")
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}

	t.Run("single match replaces", func(t *testing.T) {
		path := writeTemp(t, "foo bar baz")
		output, isError := RunEdit(EditInput{Path: path, OldString: "bar", NewString: "qux"})
		if isError {
			t.Fatalf("unexpected error: %q", output)
		}
		got, _ := os.ReadFile(path)
		if string(got) != "foo qux baz" {
			t.Errorf("content = %q, want %q", got, "foo qux baz")
		}
	})

	t.Run("no match errors and leaves file untouched", func(t *testing.T) {
		path := writeTemp(t, "foo bar baz")
		_, isError := RunEdit(EditInput{Path: path, OldString: "nope", NewString: "qux"})
		if !isError {
			t.Fatalf("want error for no match")
		}
		got, _ := os.ReadFile(path)
		if string(got) != "foo bar baz" {
			t.Errorf("file was modified despite no-match error: %q", got)
		}
	})

	t.Run("multiple matches without replace_all errors", func(t *testing.T) {
		path := writeTemp(t, "foo foo foo")
		_, isError := RunEdit(EditInput{Path: path, OldString: "foo", NewString: "bar"})
		if !isError {
			t.Fatalf("want error for ambiguous match")
		}
		got, _ := os.ReadFile(path)
		if string(got) != "foo foo foo" {
			t.Errorf("file was modified despite ambiguous-match error: %q", got)
		}
	})

	t.Run("multiple matches with replace_all replaces every occurrence", func(t *testing.T) {
		path := writeTemp(t, "foo foo foo")
		output, isError := RunEdit(EditInput{Path: path, OldString: "foo", NewString: "bar", ReplaceAll: true})
		if isError {
			t.Fatalf("unexpected error: %q", output)
		}
		got, _ := os.ReadFile(path)
		if string(got) != "bar bar bar" {
			t.Errorf("content = %q, want %q", got, "bar bar bar")
		}
	})

	t.Run("identical old and new string rejected", func(t *testing.T) {
		path := writeTemp(t, "foo bar baz")
		_, isError := RunEdit(EditInput{Path: path, OldString: "bar", NewString: "bar"})
		if !isError {
			t.Fatalf("want error for identical old/new string")
		}
	})

	t.Run("missing file errors", func(t *testing.T) {
		_, isError := RunEdit(EditInput{Path: filepath.Join(t.TempDir(), "missing.txt"), OldString: "a", NewString: "b"})
		if !isError {
			t.Fatalf("want error for missing file")
		}
	})
}
