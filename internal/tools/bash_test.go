package tools

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
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

func TestRunBash_capturesStderr(t *testing.T) {
	output, isError := RunBash(context.Background(), BashInput{Command: "echo oops 1>&2; exit 1"})
	if !isError {
		t.Fatalf("want isError=true, got output=%q", output)
	}
	if !strings.Contains(output, "oops") {
		t.Errorf("output = %q, want it to contain stderr text", output)
	}
}
