package main

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
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

const (
	helperEnvPath  = "KORTEX_ROTLOG_TEST_PATH"
	helperEnvMax   = "KORTEX_ROTLOG_TEST_MAX"
	helperEnvID    = "KORTEX_ROTLOG_TEST_ID"
	helperEnvCount = "KORTEX_ROTLOG_TEST_COUNT"
	helperEnvStart = "KORTEX_ROTLOG_TEST_START"
	helperRecSize  = 50
)

// helperLine builds one fixed-size record "p<id>-<seq>" padded to helperRecSize.
func helperLine(id int, seq int) string {
	head := fmt.Sprintf("p%d-%04d ", id, seq)
	return head + strings.Repeat("x", helperRecSize-1-len(head)) + "\n"
}

// TestRotatingLogHelperProcess is not a test: it is the body of the child
// processes spawned by TestRotatingLog_RealProcesses (re-executing this test
// binary), and does nothing in a normal run.
func TestRotatingLogHelperProcess(t *testing.T) {
	path := os.Getenv(helperEnvPath)
	if path == "" {
		t.Skip("helper process body; only runs when spawned by TestRotatingLog_RealProcesses")
	}
	var maxBytes int64
	var id, count int
	fmt.Sscan(os.Getenv(helperEnvMax), &maxBytes)
	fmt.Sscan(os.Getenv(helperEnvID), &id)
	fmt.Sscan(os.Getenv(helperEnvCount), &count)

	l, err := newRotatingLog(path, maxBytes, os.Stderr)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	// Spin until the parent releases every child at once, to force contention.
	for start := os.Getenv(helperEnvStart); ; time.Sleep(time.Millisecond) {
		if _, err := os.Stat(start); err == nil {
			break
		}
	}
	for seq := 0; seq < count; seq++ {
		if _, err := l.Write([]byte(helperLine(id, seq))); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}
}

// Real OS processes (not instances in one process) appending to one path.
func TestRotatingLog_RealProcesses(t *testing.T) {
	const procs, perProc = 4, 300
	const total = procs * perProc * helperRecSize
	tests := []struct {
		name     string
		maxBytes int64
		wantAll  bool // cap leaves room for one rotation only, so no record may be dropped
	}{
		{name: "single rotation keeps every record", maxBytes: total * 2 / 3, wantAll: true},
		{name: "many rotations keep intact files", maxBytes: 2000},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "k.native")
			start := filepath.Join(dir, "go")

			var cmds []*exec.Cmd
			var stderrs []*bytes.Buffer
			for id := 0; id < procs; id++ {
				cmd := exec.Command(os.Args[0], "-test.run=^TestRotatingLogHelperProcess$")
				cmd.Env = append(os.Environ(),
					helperEnvPath+"="+path,
					helperEnvMax+"="+fmt.Sprint(tc.maxBytes),
					helperEnvID+"="+fmt.Sprint(id),
					helperEnvCount+"="+fmt.Sprint(perProc),
					helperEnvStart+"="+start,
				)
				var stderr bytes.Buffer
				cmd.Stderr = &stderr
				if err := cmd.Start(); err != nil {
					t.Fatal(err)
				}
				cmds = append(cmds, cmd)
				stderrs = append(stderrs, &stderr)
			}
			if err := os.WriteFile(start, nil, 0o600); err != nil {
				t.Fatal(err)
			}
			for i, cmd := range cmds {
				if err := cmd.Wait(); err != nil {
					t.Fatalf("child %d: %v\nstderr: %s", i, err, stderrs[i])
				}
				if stderrs[i].Len() > 0 {
					t.Errorf("child %d reported on stderr: %s", i, stderrs[i])
				}
			}

			seen := map[string]int{}
			for _, p := range []string{path, path + ".1"} {
				assertMode0600(t, p)
				content := readFile(t, p)
				if content != "" && !strings.HasSuffix(content, "\n") {
					t.Errorf("%s ends in a torn record", p)
				}
				sc := bufio.NewScanner(strings.NewReader(content))
				for sc.Scan() {
					line := sc.Text() + "\n"
					if len(line) != helperRecSize || !strings.HasPrefix(line, "p") {
						t.Fatalf("%s has a corrupt record %q", p, line)
					}
					seen[line]++
				}
			}
			backup, err := os.Stat(path + ".1")
			if err != nil || backup.Size() == 0 {
				t.Fatalf("want a non-empty backup, got %v (err %v)", backup, err)
			}
			if min := tc.maxBytes - helperRecSize; backup.Size() < min {
				t.Errorf("backup is %d bytes, want >= %d (full backup clobbered)", backup.Size(), min)
			}
			for line, n := range seen {
				if n != 1 {
					t.Errorf("record %q appears %d times", line, n)
				}
			}
			if tc.wantAll && len(seen) != procs*perProc {
				t.Errorf("kept %d records, want all %d", len(seen), procs*perProc)
			}
		})
	}
}

// With the lock unobtainable, rotation is skipped (never done unlocked) but
// every write must still land, and the failure must be reported exactly once.
func TestRotatingLog_LockFailureKeepsWritingAndReports(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.native")
	// A directory at the lock path makes opening the lock file fail.
	if err := os.Mkdir(path+".lock", 0o700); err != nil {
		t.Fatal(err)
	}
	var diag bytes.Buffer
	l, err := newRotatingLog(path, 10, &diag)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	want := ""
	for _, w := range []string{"aaaaaa\n", "bbbbbb\n", "cccccc\n"} {
		want += w
		if _, err := l.Write([]byte(w)); err != nil {
			t.Fatalf("Write must succeed despite lock failure: %v", err)
		}
	}
	if got := readFile(t, path); got != want {
		t.Errorf("main = %q, want %q", got, want)
	}
	if _, err := os.Stat(path + ".1"); err == nil {
		t.Error("rotated without holding the lock")
	}
	if n := strings.Count(diag.String(), "open lock"); n != 1 {
		t.Errorf("lock failure reported %d times, want exactly 1: %q", n, diag.String())
	}
}
