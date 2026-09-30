package proxy

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ResolveClaudeBin locates the real claude binary. An explicit path (from
// KORTEX_CLAUDE_BIN) wins; otherwise the PATH is scanned for "claude",
// skipping any candidate that is kortex itself — as a drop-in replacement
// kortex is typically installed under the name "claude", so a naive
// exec.LookPath would resolve to us and fork-bomb.
func ResolveClaudeBin(explicit string, selfPath string) (string, error) {
	if explicit != "" {
		if !isExecutableFile(explicit) {
			return "", fmt.Errorf("KORTEX_CLAUDE_BIN %q is not an executable file", explicit)
		}
		return explicit, nil
	}
	for _, dir := range filepath.SplitList(os.Getenv("PATH")) {
		if dir == "" {
			continue
		}
		candidate := filepath.Join(dir, "claude")
		if !isExecutableFile(candidate) {
			continue
		}
		if sameFile(candidate, selfPath) {
			continue
		}
		return candidate, nil
	}
	return "", fmt.Errorf("real claude binary not found in PATH (and KORTEX_CLAUDE_BIN not set)")
}

func isExecutableFile(path string) bool {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return false
	}
	return info.Mode().Perm()&0o111 != 0
}

func sameFile(a string, b string) bool {
	if b == "" {
		return false
	}
	ra, err := filepath.EvalSymlinks(a)
	if err != nil {
		ra = a
	}
	rb, err := filepath.EvalSymlinks(b)
	if err != nil {
		rb = b
	}
	if strings.TrimSpace(ra) == strings.TrimSpace(rb) {
		return true
	}
	sa, err := os.Stat(ra)
	if err != nil {
		return false
	}
	sb, err := os.Stat(rb)
	if err != nil {
		return false
	}
	return os.SameFile(sa, sb)
}
