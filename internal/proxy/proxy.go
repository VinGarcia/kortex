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
	"sync"
	"syscall"
	"time"

	"github.com/vingarcia/kortex/internal/protocol"
)

type Config struct {
	ClaudeBin string
	Argv      []string
	// AuthFDs are inherited file descriptors carrying auth secrets (spec F0
	// §2: OpenClaw announces them via CLAUDE_CODE_*_FILE_DESCRIPTOR env vars,
	// resolved by the caller). Each must reach the child at the same number.
	AuthFDs []int
	// LogPath receives the raw wire traffic; EventLogPath the structured
	// per-event/per-turn log. Empty disables each. Both are derived from
	// KORTEX_LOG by main (the composition root owns all path decisions).
	LogPath      string
	EventLogPath string
	Stdin        io.Reader
	Stdout       io.Writer
	Stderr       io.Writer
}

// Run execs the real claude with untouched argv/env, proxies stdio until the
// child exits, and returns the exit code to propagate (128+signal when the
// child died from a signal).
func Run(cfg Config) int {
	logger := newTrafficLogger(cfg.LogPath, cfg.Stderr)
	defer logger.Close()
	obs := newObserver(cfg.EventLogPath, cfg.Stderr)
	defer obs.Close()

	cmd := exec.Command(cfg.ClaudeBin, cfg.Argv...)
	cmd.Stderr = cfg.Stderr
	if err := forwardAuthFDs(cmd, cfg.AuthFDs); err != nil {
		fmt.Fprintf(cfg.Stderr, "kortex: auth fd not forwarded: %v\n", err)
		return 1
	}

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
		_ = copyLines(cfg.Stdin, stdinPipe, logger, obs, protocol.ToBackend)
		stdinPipe.Close()
	}()

	if err := copyLines(stdoutPipe, cfg.Stdout, logger, obs, protocol.FromBackend); err != nil {
		fmt.Fprintf(cfg.Stderr, "kortex: stdout copy: %v\n", err)
	}
	return exitCode(cmd.Wait())
}

// copyLines pumps NDJSON lines from r to w, byte-identical. This per-line
// plumbing (instead of a raw io.Copy) exists so the interception layers can
// see every protocol line; a line that does not parse as JSON passes
// untouched. Each line is parsed exactly once, and the observation layers
// run as pure readers after the forwarding write, so they can never alter
// the passthrough — only add their own (small, per-line) latency to the
// pump when logging is enabled.
func copyLines(r io.Reader, w io.Writer, logger *trafficLogger, obs *observer, direction protocol.Direction) error {
	prefix := ">>"
	if direction == protocol.FromBackend {
		prefix = "<<"
	}
	br := bufio.NewReaderSize(r, 64*1024)
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 {
			if _, werr := w.Write(line); werr != nil {
				return werr
			}
			event := protocol.Parse(line)
			logger.Log(prefix, event.Type, line)
			obs.Observe(direction, event)
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// forwardAuthFDs re-exposes inherited auth file descriptors to the child at
// the same FD numbers the env announces. ExtraFiles maps index i to child FD
// 3+i, so gaps are padded with /dev/null to land each FD on its number.
func forwardAuthFDs(cmd *exec.Cmd, fds []int) error {
	maxFD := 0
	for _, fd := range fds {
		if fd < 3 {
			return fmt.Errorf("announced auth fd %d is below 3", fd)
		}
		if fd > maxFD {
			maxFD = fd
		}
	}
	if maxFD == 0 {
		return nil
	}
	files := make([]*os.File, maxFD-3+1)
	for _, fd := range fds {
		files[fd-3] = os.NewFile(uintptr(fd), "kortex-auth-fd")
	}
	for i, file := range files {
		if file != nil {
			continue
		}
		null, err := os.Open(os.DevNull)
		if err != nil {
			for _, opened := range files[:i] {
				if opened != nil {
					opened.Close()
				}
			}
			return fmt.Errorf("padding fd %d: %w", i+3, err)
		}
		files[i] = null
	}
	cmd.ExtraFiles = files
	return nil
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

func (l *trafficLogger) Log(direction string, msgType string, line []byte) {
	if l.file == nil || len(line) == 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if msgType == protocol.TypeUnknown {
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
