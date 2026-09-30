// Package proxy execs the real claude binary and plumbs its stdio
// transparently: byte-identical passthrough of the stream-json protocol,
// exit-code and signal propagation, and optional raw-traffic debug logging.
package proxy

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/vingarcia/kortex/internal/protocol"
)

// Env var names OpenClaw uses to point the backend at the auth secret it
// writes on an inherited file descriptor (spec F0 §2). Kortex must hand that
// same descriptor down to the real claude or auth breaks.
const (
	oauthTokenFDEnv = "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"
	apiKeyFDEnv     = "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"
)

type Config struct {
	ClaudeBin string
	Argv      []string
	LogPath   string
	Stdin     io.Reader
	Stdout    io.Writer
	Stderr    io.Writer
	// Getenv defaults to os.Getenv; overridable for tests.
	Getenv func(key string) string
}

// Run execs the real claude with untouched argv/env, proxies stdio until the
// child exits, and returns the exit code to propagate (128+signal when the
// child died from a signal).
func Run(cfg Config) int {
	getenv := cfg.Getenv
	if getenv == nil {
		getenv = os.Getenv
	}
	logger := newTrafficLogger(cfg.LogPath, cfg.Stderr)
	defer logger.Close()

	cmd := exec.Command(cfg.ClaudeBin, cfg.Argv...)
	cmd.Stderr = cfg.Stderr
	forwardAuthFD(cmd, getenv)

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		fmt.Fprintf(cfg.Stderr, "kortex: stdin pipe: %v\n", err)
		return 1
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		fmt.Fprintf(cfg.Stderr, "kortex: stdout pipe: %v\n", err)
		return 1
	}
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(cfg.Stderr, "kortex: exec %s: %v\n", cfg.ClaudeBin, err)
		return 1
	}

	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)
	go func() {
		for sig := range sigCh {
			_ = cmd.Process.Signal(sig)
		}
	}()

	// The stdin pump is deliberately not joined: if the child exits while the
	// caller still holds stdin open, this goroutine stays blocked on read and
	// joining it would hang the proxy past the child's death.
	go func() {
		// Write errors here mean the child went away; the exit code from
		// Wait is the meaningful outcome, not this copy error.
		_ = copyLines(cfg.Stdin, stdinPipe, logger, ">>")
		stdinPipe.Close()
	}()

	if err := copyLines(stdoutPipe, cfg.Stdout, logger, "<<"); err != nil {
		fmt.Fprintf(cfg.Stderr, "kortex: stdout copy: %v\n", err)
	}
	return exitCode(cmd.Wait())
}

// copyLines pumps NDJSON lines from r to w, byte-identical. This per-line
// plumbing (instead of a raw io.Copy) exists so F2 can intercept and rewrite
// protocol messages here; a line that does not parse as JSON passes untouched.
func copyLines(r io.Reader, w io.Writer, logger *trafficLogger, direction string) error {
	br := bufio.NewReaderSize(r, 64*1024)
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 {
			logger.Log(direction, line)
			if _, werr := w.Write(line); werr != nil {
				return werr
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// forwardAuthFD re-exposes the inherited auth file descriptor to the child
// at the same FD number OpenClaw announced in the env.
func forwardAuthFD(cmd *exec.Cmd, getenv func(string) string) {
	value := getenv(oauthTokenFDEnv)
	if value == "" {
		value = getenv(apiKeyFDEnv)
	}
	if value == "" {
		return
	}
	fd, err := strconv.Atoi(value)
	if err != nil || fd < 3 {
		return
	}
	// ExtraFiles maps index i to child FD 3+i; pad gaps so the announced
	// number lands on the same FD in the child.
	files := make([]*os.File, fd-3+1)
	for i := range files[:len(files)-1] {
		null, err := os.Open(os.DevNull)
		if err != nil {
			return
		}
		files[i] = null
	}
	files[fd-3] = os.NewFile(uintptr(fd), "kortex-auth-fd")
	cmd.ExtraFiles = files
}

func exitCode(waitErr error) int {
	if waitErr == nil {
		return 0
	}
	exitErr, ok := waitErr.(*exec.ExitError)
	if !ok {
		return 1
	}
	if status, ok := exitErr.Sys().(syscall.WaitStatus); ok && status.Signaled() {
		return 128 + int(status.Signal())
	}
	return exitErr.ExitCode()
}

// trafficLogger appends raw wire traffic to a debug file, never to the
// protocol's stdout/stderr. A nil file makes every method a no-op.
type trafficLogger struct {
	mu   sync.Mutex
	file *os.File
}

func newTrafficLogger(path string, stderr io.Writer) *trafficLogger {
	if path == "" {
		return &trafficLogger{}
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		fmt.Fprintf(stderr, "kortex: KORTEX_LOG disabled: %v\n", err)
		return &trafficLogger{}
	}
	return &trafficLogger{file: file}
}

func (l *trafficLogger) Log(direction string, line []byte) {
	if l.file == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	msgType := protocol.MessageType(line)
	if msgType == "" {
		msgType = "raw"
	}
	fmt.Fprintf(l.file, "%s %s [%s] ", time.Now().Format(time.RFC3339Nano), direction, msgType)
	l.file.Write(line)
	if line[len(line)-1] != '\n' {
		l.file.Write([]byte("\n"))
	}
}

func (l *trafficLogger) Close() {
	if l.file != nil {
		l.file.Close()
	}
}
