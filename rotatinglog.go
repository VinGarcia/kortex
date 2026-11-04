package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"sync"
	"syscall"
)

// defaultNativeLogMaxBytes caps the ".native" wire log before it is rotated to
// "<name>.1". With one backup the on-disk total stays near twice this value.
const defaultNativeLogMaxBytes int64 = 100 << 20

// nativeLogMaxBytes resolves the rotation cap from KORTEX_NATIVE_LOG_MAX_BYTES
// (a positive byte count). Unset selects the default; an invalid value warns
// on diag and also selects the default, so a typo never disables the bound.
func nativeLogMaxBytes(getenv func(string) string, diag io.Writer) int64 {
	raw := getenv("KORTEX_NATIVE_LOG_MAX_BYTES")
	if raw == "" {
		return defaultNativeLogMaxBytes
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || n <= 0 {
		fmt.Fprintf(diag, "kortex: invalid KORTEX_NATIVE_LOG_MAX_BYTES %q; using %d\n", raw, defaultNativeLogMaxBytes)
		return defaultNativeLogMaxBytes
	}
	return n
}

// rotatingLog is an append-only file writer (mode 0600) that, when a write would
// push the file past maxBytes, renames it to "<path>.1" (replacing any older
// backup) and continues in a fresh file. Rotation is best-effort: a failure is
// reported once on diag and the write still lands in the current file, so the
// log can never fail a turn. Safe for concurrent use.
//
// Several kortex processes may append to the same path. Before each write the
// file is stat'ed by path (lock-free) so a process whose file was rotated by
// another one reopens the fresh file. When rotation looks needed, the decision
// is serialized across processes with flock(2) on "<path>.lock": under the lock
// the path is re-stat'ed and renamed only if it is still r.file and still over
// the cap, so a process that lost the race never renames a freshly created file
// over the backup. A process that stat'ed just before another rotated may still
// append one record to the (already renamed) backup; this is tolerated.
type rotatingLog struct {
	path     string
	maxBytes int64
	diag     io.Writer

	mu          sync.Mutex
	file        *os.File
	reportedErr bool // true while the last rotation attempt failed and was reported
}

// newRotatingLog opens (creating if needed) the log at path. An open failure is
// returned so the caller can keep its "log disabled" degradation.
func newRotatingLog(path string, maxBytes int64, diag io.Writer) (*rotatingLog, error) {
	f, err := openAppend(path)
	if err != nil {
		return nil, err
	}
	return &rotatingLog{path: path, maxBytes: maxBytes, diag: diag, file: f}, nil
}

func openAppend(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
}

func (r *rotatingLog) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.rotateIfNeeded(int64(len(p)))
	return r.file.Write(p)
}

// Close closes the current file.
func (r *rotatingLog) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.file.Close()
}

// rotateIfNeeded never fails: on any error it reports and leaves r.file usable.
func (r *rotatingLog) rotateIfNeeded(incoming int64) {
	pathInfo, err := os.Stat(r.path)
	if err != nil {
		// Path vanished (rotated or removed elsewhere): start a fresh file.
		r.reopen()
		return
	}
	if cur, err := r.file.Stat(); err == nil && !os.SameFile(cur, pathInfo) {
		// Another process rotated; follow the fresh file.
		r.reopen()
		return
	}
	// size > 0 guards against rotating forever on a single record above the cap.
	if pathInfo.Size() == 0 || pathInfo.Size()+incoming <= r.maxBytes {
		return
	}
	r.rotateLocked(incoming)
}

// rotateLocked serializes the rotation across processes via flock on the
// sidecar lock file. If the lock cannot be taken it reports and skips rotation
// (never rotates without the lock), so the write lands in the current file.
func (r *rotatingLog) rotateLocked(incoming int64) {
	lock, err := os.OpenFile(r.path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		r.report(fmt.Errorf("open lock: %w", err))
		return
	}
	defer lock.Close() // closing releases the flock
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		r.report(fmt.Errorf("flock: %w", err))
		return
	}

	// Re-check under the lock: another process may have rotated meanwhile.
	pathInfo, err := os.Stat(r.path)
	if err != nil {
		r.reopen()
		return
	}
	if cur, err := r.file.Stat(); err == nil && !os.SameFile(cur, pathInfo) {
		r.reopen()
		return
	}
	if pathInfo.Size() == 0 || pathInfo.Size()+incoming <= r.maxBytes {
		return
	}
	if err := os.Rename(r.path, r.path+".1"); err != nil {
		r.report(fmt.Errorf("rename: %w", err))
		return
	}
	r.reopen()
}

// reopen swaps in a fresh file, keeping the old handle if the open fails.
func (r *rotatingLog) reopen() {
	f, err := openAppend(r.path)
	if err != nil {
		r.report(fmt.Errorf("reopen: %w", err))
		return
	}
	r.file.Close()
	r.file = f
	r.reportedErr = false
}

// report logs a rotation failure once until a later reopen succeeds, so a
// persistent fault cannot flood diag on every write.
func (r *rotatingLog) report(err error) {
	if r.reportedErr {
		return
	}
	r.reportedErr = true
	fmt.Fprintf(r.diag, "kortex: native wire log rotation failed (continuing in current file): %v\n", err)
}
