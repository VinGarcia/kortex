package tools

import (
	"os"
	"path/filepath"
	"testing"
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

func TestRunWrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.txt")

	// create
	if _, isError := RunWrite(WriteInput{Path: path, Content: "first"}); isError {
		t.Fatalf("unexpected error creating file")
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != "first" {
		t.Fatalf("after create: got %q, err=%v", got, err)
	}

	// overwrite
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
