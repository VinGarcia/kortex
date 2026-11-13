// Package delivered reads the anti-double-send ledger that background runs
// append to after they deliver a message to a chat, and reports which of those
// deliveries a live session has not seen yet. A live session does not see
// cron/background deliveries in its own transcript, so without this it may
// repeat an already-fixed mistake or re-offer work already delivered.
//
// Ledger line format (TSV, one line per delivery, fields after the first are
// matched by KEY, not by position):
//
//	2026-11-13 13:13:32<TAB>delivered<TAB>title=...<TAB>channel=telegram<TAB>target=180614014
//
// This is the native replacement for sylphie's scripts/delivered-unseen.sh and
// keeps its semantics: a byte-offset cursor, a first read (no cursor) bounded
// to the last 3 matching deliveries of the last 24h, a shrunk ledger
// (rotation/truncation) re-read from the start, and fail-open everywhere
// (errors inject nothing and are logged to the diag writer). Unlike the script,
// reading and advancing the cursor are split: Unseen only reads, and the caller
// calls Commit after the turn succeeded, so a failed turn re-shows the entries.
package delivered

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	// firstReadMax and firstReadWindow bound the first read of a session (no
	// cursor yet): only the most recent deliveries, so a fresh session is not
	// flooded with the whole ledger history.
	firstReadMax    = 3
	firstReadWindow = 24 * time.Hour
	// maxReadBytes caps how much of the ledger one call reads: the first read
	// takes only the ledger's last maxReadBytes, and a cursor lagging further
	// behind than that skips ahead to the last maxReadBytes.
	maxReadBytes = 64 * 1024
	// maxTitleRunes caps one rendered title, so a runaway ledger line cannot
	// flood the core's context.
	maxTitleRunes   = 200
	timestampLayout = "2006-01-02 15:04:05"
)

// Feed reports the ledger deliveries a single session has not seen yet. The
// cursor (a byte offset into the ledger) is persisted at cursorPath so it
// survives a cross-process resume of the same session.
type Feed struct {
	ledgerPath    string
	cursorPath    string
	now           func() time.Time
	diag          io.Writer
	missingLogged bool
}

// NewFeed builds a Feed reading ledgerPath and persisting its cursor at
// cursorPath. Swallowed errors are logged to diag (nil discards them). now is
// injected for tests; nil means time.Now.
func NewFeed(ledgerPath string, cursorPath string, diag io.Writer, now func() time.Time) *Feed {
	if now == nil {
		now = time.Now
	}
	if diag == nil {
		diag = io.Discard
	}
	return &Feed{ledgerPath: ledgerPath, cursorPath: cursorPath, now: now, diag: diag}
}

// Unseen returns the ephemeral block listing the deliveries to chatID (an
// OpenClaw Conversation-info chat_id such as "telegram:-5447333904") appended
// to the ledger since the last committed cursor, plus the offset to pass to
// Commit once the turn succeeded. It never moves the cursor itself. ok is false
// when there is nothing to commit (malformed chatID or any read error); block
// is "" when nothing new was delivered to chatID. Fail-open: errors only log.
func (f *Feed) Unseen(chatID string) (block string, nextOffset int64, ok bool) {
	channel, target, found := strings.Cut(chatID, ":")
	if !found || channel == "" || target == "" {
		return "", 0, false
	}
	cursor, hasCursor := f.readCursor()
	data, start, err := f.readTail(cursor, hasCursor)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			if !f.missingLogged {
				f.missingLogged = true
				fmt.Fprintf(f.diag, "delivered: ledger %s not found, feed idle\n", f.ledgerPath)
			}
		} else {
			fmt.Fprintf(f.diag, "delivered: reading ledger %s: %v\n", f.ledgerPath, err)
		}
		return "", 0, false
	}
	// Only complete lines count: a trailing line still being written is left for
	// the next call instead of being skipped by the cursor.
	complete := data[:bytes.LastIndexByte(data, '\n')+1]
	nextOffset = start + int64(len(complete))

	var lines []string
	if !hasCursor {
		lines = matching(complete, channel, target, f.now().Add(-firstReadWindow), f.now())
		if len(lines) > firstReadMax {
			lines = lines[len(lines)-firstReadMax:]
		}
	} else {
		lines = matching(complete, channel, target, time.Time{}, f.now())
	}
	if len(lines) == 0 {
		return "", nextOffset, true
	}
	block = "<delivered_since_last_turn>\n" +
		"Delivered to you in this chat by background runs since your last turn " +
		"(already sent; do not repeat or re-offer them):\n" +
		strings.Join(lines, "\n") + "\n</delivered_since_last_turn>"
	return block, nextOffset, true
}

// Commit persists nextOffset (from Unseen) as the session's cursor, atomically
// (temp file + rename). Call it only after the turn that received the block
// succeeded. A failed write is logged; the next turn then repeats entries.
func (f *Feed) Commit(nextOffset int64) {
	if err := f.writeCursor(nextOffset); err != nil {
		fmt.Fprintf(f.diag, "delivered: writing cursor %s: %v\n", f.cursorPath, err)
	}
}

// readTail returns the ledger bytes to scan and the offset they start at. With
// a cursor it reads from the cursor (from 0 if the ledger shrank); without one,
// or when the cursor lags more than maxReadBytes, it reads the last
// maxReadBytes, dropping the leading partial line.
func (f *Feed) readTail(cursor int64, hasCursor bool) ([]byte, int64, error) {
	file, err := os.Open(f.ledgerPath)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, 0, err
	}
	size := info.Size()
	if !info.Mode().IsRegular() {
		return nil, 0, fmt.Errorf("not a regular file")
	}
	start := cursor
	if !hasCursor || cursor > size {
		start = 0
	}
	trimLeadingPartial := false
	if size-start > maxReadBytes {
		// One byte before the window, so a window that begins exactly at a line
		// start keeps that line (the byte read is the previous line's '\n').
		start = size - maxReadBytes - 1
		trimLeadingPartial = true
	}
	if _, err := file.Seek(start, io.SeekStart); err != nil {
		return nil, 0, err
	}
	data, err := io.ReadAll(io.LimitReader(file, size-start))
	if err != nil {
		return nil, 0, err
	}
	if trimLeadingPartial {
		cut := bytes.IndexByte(data, '\n') + 1
		data, start = data[cut:], start+int64(cut)
	}
	return data, start, nil
}

// matching returns one formatted line per ledger entry addressed to
// channel/target whose timestamp is not before since. Today's entries render as
// "HH:MM title", older ones as "YYYY-MM-DD HH:MM title".
func matching(data []byte, channel string, target string, since time.Time, now time.Time) []string {
	today := now.Format("2006-01-02")
	var out []string
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 0, 4096), maxReadBytes+1)
	for scanner.Scan() {
		fields := strings.Split(scanner.Text(), "\t")
		if len(fields) < 2 {
			continue
		}
		ts, err := time.ParseInLocation(timestampLayout, fields[0], now.Location())
		if err != nil || ts.Before(since) {
			continue
		}
		var title string
		var okChannel, okTarget bool
		for _, field := range fields[1:] {
			switch {
			case field == "channel="+channel:
				okChannel = true
			case field == "target="+target:
				okTarget = true
			case strings.HasPrefix(field, "title="):
				title = sanitizeTitle(strings.TrimPrefix(field, "title="))
			}
		}
		if !okChannel || !okTarget {
			continue
		}
		stamp := ts.Format("15:04")
		if ts.Format("2006-01-02") != today {
			stamp = ts.Format("2006-01-02 15:04")
		}
		out = append(out, stamp+" "+title)
	}
	return out
}

// sanitizeTitle neutralizes angle brackets (so a title cannot close or open the
// block's delimiter tags) and caps the title at maxTitleRunes.
func sanitizeTitle(title string) string {
	title = strings.NewReplacer("<", "‹", ">", "›").Replace(title)
	runes := []rune(title)
	if len(runes) > maxTitleRunes {
		return string(runes[:maxTitleRunes]) + "…"
	}
	return title
}

func (f *Feed) readCursor() (int64, bool) {
	raw, err := os.ReadFile(f.cursorPath)
	if err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			fmt.Fprintf(f.diag, "delivered: reading cursor %s: %v\n", f.cursorPath, err)
		}
		return 0, false
	}
	offset, err := strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
	if err != nil || offset < 0 {
		fmt.Fprintf(f.diag, "delivered: ignoring malformed cursor %s\n", f.cursorPath)
		return 0, false
	}
	return offset, true
}

func (f *Feed) writeCursor(offset int64) error {
	dir := filepath.Dir(f.cursorPath)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, filepath.Base(f.cursorPath)+".tmp-*")
	if err != nil {
		return err
	}
	_, writeErr := fmt.Fprintf(tmp, "%d\n", offset)
	closeErr := tmp.Close()
	if err := errors.Join(writeErr, closeErr); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	if err := os.Rename(tmp.Name(), f.cursorPath); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	return nil
}
