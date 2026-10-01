package tools

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestRunBash(t *testing.T) {
	tests := []struct {
		desc        string
		input       BashInput
		wantErr     bool
		wantContain string
	}{
		{
			desc:        "success",
			input:       BashInput{Command: "echo hello-from-bash"},
			wantErr:     false,
			wantContain: "hello-from-bash",
		},
		{
			desc:        "nonzero exit",
			input:       BashInput{Command: "exit 3"},
			wantErr:     true,
			wantContain: "exit status 3",
		},
		{
			desc:        "timeout",
			input:       BashInput{Command: "sleep 5", TimeoutSeconds: 1},
			wantErr:     true,
			wantContain: "timed out",
		},
		{
			desc:    "empty command rejected",
			input:   BashInput{Command: ""},
			wantErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			output, isError := RunBash(context.Background(), test.input)
			if isError != test.wantErr {
				t.Errorf("isError = %v, want %v (output=%q)", isError, test.wantErr, output)
			}
			if test.wantContain != "" && !strings.Contains(output, test.wantContain) {
				t.Errorf("output = %q, want it to contain %q", output, test.wantContain)
			}
		})
	}
}

func TestRunBash_cwd(t *testing.T) {
	dir := t.TempDir()
	output, isError := RunBash(context.Background(), BashInput{Command: "pwd", Cwd: dir})
	if isError {
		t.Fatalf("unexpected error: %q", output)
	}
	resolvedDir, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.TrimSpace(output), resolvedDir) {
		t.Errorf("pwd output = %q, want it to contain cwd %q", output, resolvedDir)
	}
}

// TestRunBash_truncationIsValidUTF8 produces output one byte past
// bashMaxOutputBytes where the very last byte within the cap is the lead
// byte of a 2-byte UTF-8 rune (0xC3 0xA9, "é") and its continuation byte
// falls just past the cap. A raw byte-offset cut would keep the lone 0xC3
// and return invalid UTF-8; RunBash must instead back off that dangling
// byte.
func TestRunBash_truncationIsValidUTF8(t *testing.T) {
	output, isError := RunBash(context.Background(), BashInput{
		Command: "head -c 262143 /dev/zero | tr '\\0' 'a'; printf '\\303\\251'",
	})
	if isError {
		t.Fatalf("unexpected error: %q", output)
	}
	if !utf8.ValidString(output) {
		t.Errorf("output is not valid UTF-8: %q", output)
	}
	if !strings.Contains(output, "[output truncated at 262144 bytes]") {
		t.Errorf("output = %q, want the truncation notice", output)
	}
}

func TestRunBash_capturesStderr(t *testing.T) {
	output, isError := RunBash(context.Background(), BashInput{Command: "echo oops 1>&2; exit 1"})
	if !isError {
		t.Fatalf("want isError=true, got output=%q", output)
	}
	if !strings.Contains(output, "oops") {
		t.Errorf("output = %q, want it to contain stderr text", output)
	}
}
