package proxy

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeExecutable(t *testing.T, dir string, name string, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestResolveClaudeBin(t *testing.T) {
	selfDir := t.TempDir()
	self := writeExecutable(t, selfDir, "claude", "#!/bin/sh\n")

	selfLinkDir := t.TempDir()
	if err := os.Symlink(self, filepath.Join(selfLinkDir, "claude")); err != nil {
		t.Fatal(err)
	}

	realDir := t.TempDir()
	real := writeExecutable(t, realDir, "claude", "#!/bin/sh\necho real\n")

	nonExecDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(nonExecDir, "claude"), []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		desc               string
		explicit           string
		searchDirs         []string
		want               string
		expectErrToContain string
	}{
		{desc: "explicit bin wins over search dirs", explicit: real, searchDirs: []string{selfDir}, want: real},
		{desc: "explicit bin missing", explicit: filepath.Join(selfDir, "nope"), expectErrToContain: "not an executable"},
		{desc: "search skips self and finds real claude", searchDirs: []string{selfDir, realDir}, want: real},
		{desc: "search skips symlink to self", searchDirs: []string{selfLinkDir, realDir}, want: real},
		{desc: "search skips non-executable candidate", searchDirs: []string{nonExecDir, realDir}, want: real},
		{desc: "only self in search dirs", searchDirs: []string{selfDir}, expectErrToContain: "not found"},
		{desc: "no search dirs", searchDirs: nil, expectErrToContain: "not found"},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got, err := ResolveClaudeBin(test.explicit, self, test.searchDirs)
			if test.expectErrToContain != "" {
				if err == nil || !strings.Contains(err.Error(), test.expectErrToContain) {
					t.Fatalf("error = %v, want containing %q", err, test.expectErrToContain)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Errorf("resolved %q, want %q", got, test.want)
			}
		})
	}
}
