package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return string(b)
}

func TestRotatingLog(t *testing.T) {
	tests := []struct {
		name       string
		maxBytes   int64
		writes     []string
		wantMain   string
		wantBackup string // "" = no backup expected
	}{
		{
			name:     "under cap does not rotate",
			maxBytes: 100,
			writes:   []string{"aaaa\n", "bbbb\n"},
			wantMain: "aaaa\nbbbb\n",
		},
		{
			name:       "crossing cap rotates and new write lands in fresh file",
			maxBytes:   10,
			writes:     []string{"aaaaaa\n", "bbbbbb\n"},
			wantMain:   "bbbbbb\n",
			wantBackup: "aaaaaa\n",
		},
		{
			name:       "second rotation replaces the backup",
			maxBytes:   10,
			writes:     []string{"aaaaaa\n", "bbbbbb\n", "cccccc\n"},
			wantMain:   "cccccc\n",
			wantBackup: "bbbbbb\n",
		},
		{
			name:     "single record above cap is written without rotating an empty file",
			maxBytes: 3,
			writes:   []string{"huge-record\n"},
			wantMain: "huge-record\n",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "k.native")
			var diag bytes.Buffer
			l, err := newRotatingLog(path, test.maxBytes, &diag)
			if err != nil {
				t.Fatal(err)
			}
			defer l.Close()
			for _, w := range test.writes {
				if n, err := l.Write([]byte(w)); err != nil || n != len(w) {
					t.Fatalf("Write(%q) = %d, %v", w, n, err)
				}
			}
			if got := readFile(t, path); got != test.wantMain {
				t.Errorf("main = %q, want %q", got, test.wantMain)
			}
			if test.wantBackup == "" {
				if _, err := os.Stat(path + ".1"); !os.IsNotExist(err) {
					t.Errorf("backup should not exist, stat err = %v", err)
				}
			} else {
				if got := readFile(t, path+".1"); got != test.wantBackup {
					t.Errorf("backup = %q, want %q", got, test.wantBackup)
				}
				assertMode0600(t, path+".1")
			}
			assertMode0600(t, path)
			if diag.Len() != 0 {
				t.Errorf("unexpected diag: %s", diag.String())
			}
		})
	}
}

func assertMode0600(t *testing.T, path string) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("%s perms = %o, want 600", path, perm)
	}
}

func TestRotatingLog_RenameFailureStillWrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.native")
	// A non-empty directory at the backup path makes the rename fail.
	if err := os.MkdirAll(filepath.Join(path+".1", "sub"), 0o700); err != nil {
		t.Fatal(err)
	}
	var diag bytes.Buffer
	l, err := newRotatingLog(path, 10, &diag)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	for _, w := range []string{"aaaaaa\n", "bbbbbb\n", "cccccc\n"} {
		if _, err := l.Write([]byte(w)); err != nil {
			t.Fatalf("Write must succeed despite rotation failure: %v", err)
		}
	}
	if got := readFile(t, path); got != "aaaaaa\nbbbbbb\ncccccc\n" {
		t.Errorf("main = %q, want all writes kept in current file", got)
	}
	if n := strings.Count(diag.String(), "rotation failed"); n != 1 {
		t.Errorf("rotation failure reported %d times, want exactly 1: %q", n, diag.String())
	}
}

func TestRotatingLog_FollowsRotationByAnotherProcess(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.native")
	l, err := newRotatingLog(path, 1000, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	l.Write([]byte("one\n"))
	if err := os.Rename(path, path+".1"); err != nil { // simulates a sibling process
		t.Fatal(err)
	}
	l.Write([]byte("two\n"))
	if got := readFile(t, path); got != "two\n" {
		t.Errorf("main = %q, want fresh file with only the new write", got)
	}
	if got := readFile(t, path+".1"); got != "one\n" {
		t.Errorf("backup = %q, want untouched", got)
	}
}

func TestRotatingLog_ConcurrentWrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.native")
	l, err := newRotatingLog(path, 2000, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	const writers, perWriter = 8, 100
	line := strings.Repeat("x", 49) + "\n"
	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < perWriter; j++ {
				if _, err := l.Write([]byte(line)); err != nil {
					t.Errorf("Write: %v", err)
				}
			}
		}()
	}
	wg.Wait()
	for _, p := range []string{path, path + ".1"} {
		for _, got := range strings.SplitAfter(readFile(t, p), "\n") {
			if got != "" && got != line {
				t.Fatalf("%s holds a torn record: %q", p, got)
			}
		}
	}
	info, _ := os.Stat(path)
	if info.Size() > 2000 {
		t.Errorf("current file %d bytes exceeds cap", info.Size())
	}
}

// Several instances on one path simulate several kortex processes. A backup
// must never be replaced by a tiny freshly-created file.
func TestRotatingLog_MultiProcessKeepsFullBackup(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.native")
	const maxBytes, recSize, writers, perWriter = 1000, 50, 6, 400
	line := strings.Repeat("x", recSize-1) + "\n"
	minBackup := int64(maxBytes - recSize)

	checkBackup := func() {
		if info, err := os.Stat(path + ".1"); err == nil && info.Size() < minBackup {
			t.Errorf("backup is %d bytes, want >= %d (full backup destroyed)", info.Size(), minBackup)
		}
	}

	logs := make([]*rotatingLog, writers)
	for i := range logs {
		l, err := newRotatingLog(path, maxBytes, &bytes.Buffer{})
		if err != nil {
			t.Fatal(err)
		}
		defer l.Close()
		logs[i] = l
	}

	stop := make(chan struct{})
	sampled := make(chan struct{})
	go func() {
		defer close(sampled)
		for {
			select {
			case <-stop:
				return
			default:
				checkBackup()
			}
		}
	}()

	var wg sync.WaitGroup
	for _, l := range logs {
		wg.Add(1)
		go func(l *rotatingLog) {
			defer wg.Done()
			for j := 0; j < perWriter; j++ {
				if _, err := l.Write([]byte(line)); err != nil {
					t.Errorf("Write: %v", err)
				}
			}
		}(l)
	}
	wg.Wait()
	close(stop)
	<-sampled
	checkBackup()
}

func TestNativeLogMaxBytes(t *testing.T) {
	tests := []struct {
		name     string
		env      string
		want     int64
		wantWarn bool
	}{
		{"unset uses default", "", defaultNativeLogMaxBytes, false},
		{"valid value", "4096", 4096, false},
		{"garbage falls back with warning", "lots", defaultNativeLogMaxBytes, true},
		{"zero falls back with warning", "0", defaultNativeLogMaxBytes, true},
		{"negative falls back with warning", "-5", defaultNativeLogMaxBytes, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var diag bytes.Buffer
			got := nativeLogMaxBytes(func(string) string { return test.env }, &diag)
			if got != test.want {
				t.Errorf("got %d, want %d", got, test.want)
			}
			if warned := diag.Len() > 0; warned != test.wantWarn {
				t.Errorf("warned = %v, want %v", warned, test.wantWarn)
			}
		})
	}
}
