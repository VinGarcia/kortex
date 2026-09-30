package proxy

import (
	"bytes"
	"io"
	"os"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

func TestCopyLines(t *testing.T) {
	giantLine := strings.Repeat("x", 10<<20) + "\n"
	tests := []struct {
		desc  string
		input string
	}{
		{desc: "ndjson lines", input: `{"type":"system","subtype":"init"}` + "\n" + `{"type":"result"}` + "\n"},
		{desc: "non-json line passes untouched", input: "not json at all\n{\"type\":\"assistant\"}\n"},
		{desc: "line larger than any buffer", input: giantLine},
		{desc: "final line without trailing newline", input: `{"type":"result"}`},
		{desc: "empty lines preserved", input: "\n\n{\"type\":\"result\"}\n"},
		{desc: "crlf preserved", input: "{\"type\":\"result\"}\r\n"},
		{desc: "empty input", input: ""},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			var out bytes.Buffer
			err := copyLines(strings.NewReader(test.input), &out, newTrafficLogger("", io.Discard), ">>")
			if err != nil {
				t.Fatal(err)
			}
			if out.String() != test.input {
				t.Errorf("output differs from input (len %d vs %d)", out.Len(), len(test.input))
			}
		})
	}
}

func TestRun(t *testing.T) {
	dir := t.TempDir()
	echoBin := writeExecutable(t, dir, "fake-claude-echo", "#!/bin/sh\ncat\n")
	exit7Bin := writeExecutable(t, dir, "fake-claude-exit7", "#!/bin/sh\nexit 7\n")
	sigkillBin := writeExecutable(t, dir, "fake-claude-sigkill", "#!/bin/sh\nkill -KILL $$\n")
	argvBin := writeExecutable(t, dir, "fake-claude-argv", "#!/bin/sh\necho \"$@\"\n")

	tests := []struct {
		desc     string
		bin      string
		argv     []string
		stdin    string
		wantOut  string
		wantCode int
	}{
		{
			desc:     "stdin passthrough byte-identical incl non-json",
			bin:      echoBin,
			stdin:    "{\"type\":\"user\"}\nstray non-json line\n" + strings.Repeat("y", 9<<20) + "\n",
			wantOut:  "{\"type\":\"user\"}\nstray non-json line\n" + strings.Repeat("y", 9<<20) + "\n",
			wantCode: 0,
		},
		{
			desc:     "argv forwarded untouched",
			bin:      argvBin,
			argv:     []string{"--print", "--output-format", "stream-json"},
			wantOut:  "--print --output-format stream-json\n",
			wantCode: 0,
		},
		{desc: "exit code propagated", bin: exit7Bin, wantCode: 7},
		{desc: "death by signal maps to 128+sig", bin: sigkillBin, wantCode: 137},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			var out, errOut bytes.Buffer
			code := Run(Config{
				ClaudeBin: test.bin,
				Argv:      test.argv,
				Stdin:     strings.NewReader(test.stdin),
				Stdout:    &out,
				Stderr:    &errOut,
			})
			if code != test.wantCode {
				t.Errorf("exit code = %d, want %d (stderr: %s)", code, test.wantCode, errOut.String())
			}
			if out.String() != test.wantOut {
				t.Errorf("stdout differs from expected (len %d vs %d)", out.Len(), len(test.wantOut))
			}
		})
	}
}

// syncBuffer lets the signal test watch stdout while Run is still writing it.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func TestRun_forwardsSIGTERMToChild(t *testing.T) {
	dir := t.TempDir()
	bin := writeExecutable(t, dir, "fake-claude-trap",
		"#!/bin/sh\ntrap 'exit 3' TERM\necho ready\nwhile :; do sleep 0.05; done\n")

	out := &syncBuffer{}
	stdinR, stdinW := io.Pipe()
	defer stdinW.Close()
	done := make(chan int, 1)
	go func() {
		done <- Run(Config{ClaudeBin: bin, Stdin: stdinR, Stdout: out, Stderr: io.Discard})
	}()

	deadline := time.Now().Add(5 * time.Second)
	for !strings.Contains(out.String(), "ready") {
		if time.Now().After(deadline) {
			t.Fatal("child never became ready")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err := syscall.Kill(os.Getpid(), syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	select {
	case code := <-done:
		if code != 3 {
			t.Errorf("exit code = %d, want 3 (child's TERM trap)", code)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not return after SIGTERM")
	}
}

func TestRun_forwardsAuthFD(t *testing.T) {
	dir := t.TempDir()
	// Reads the secret from FD 13 the way claude reads the auth FD.
	bin := writeExecutable(t, dir, "fake-claude-authfd", "#!/bin/sh\ncat <&13\n")

	pipeR, pipeW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pipeW.WriteString("secret-token"); err != nil {
		t.Fatal(err)
	}
	pipeW.Close()
	// Pin the read end to FD 13 so the env announcement matches, as it does
	// when OpenClaw spawns kortex with the auth FD already inherited.
	if err := syscall.Dup2(int(pipeR.Fd()), 13); err != nil {
		t.Fatal(err)
	}
	defer syscall.Close(13)
	defer pipeR.Close()

	var out bytes.Buffer
	code := Run(Config{
		ClaudeBin: bin,
		Stdin:     strings.NewReader(""),
		Stdout:    &out,
		Stderr:    io.Discard,
		Getenv: func(key string) string {
			if key == oauthTokenFDEnv {
				return "13"
			}
			return ""
		},
	})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if out.String() != "secret-token" {
		t.Errorf("child read %q from auth FD, want %q", out.String(), "secret-token")
	}
}

func TestRun_writesTrafficLog(t *testing.T) {
	dir := t.TempDir()
	echoBin := writeExecutable(t, dir, "fake-claude-echo", "#!/bin/sh\ncat\n")
	logPath := dir + "/traffic.log"

	var out bytes.Buffer
	code := Run(Config{
		ClaudeBin: echoBin,
		LogPath:   logPath,
		Stdin:     strings.NewReader("{\"type\":\"user\"}\nstray\n"),
		Stdout:    &out,
		Stderr:    io.Discard,
	})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	logData, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	log := string(logData)
	for _, want := range []string{">> [user]", ">> [raw] stray", "<< [user]", "<< [raw] stray"} {
		if !strings.Contains(log, want) {
			t.Errorf("traffic log missing %q; log:\n%s", want, log)
		}
	}
	if strings.Contains(out.String(), "[user]") {
		t.Error("log annotations leaked into protocol stdout")
	}
}
