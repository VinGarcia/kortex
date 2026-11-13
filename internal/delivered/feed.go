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
// (rotation/truncation) re-read from the start, and fail-open everywhere.
package delivered

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// firstReadMax and firstReadWindow bound the first read of a session (no
// cursor yet): only the most recent deliveries, so a fresh session is not
// flooded with the whole ledger history.
const (
	firstReadMax    = 3
	firstReadWindow = 24 * time.Hour
	timestampLayout = "2006-01-02 15:04:05"
)

// Feed reports the ledger deliveries a single session has not seen yet. The
// cursor (a byte offset into the ledger) is persisted at CursorPath so it
// survives a cross-process resume of the same session.
type Feed struct {
	ledgerPath string
	cursorPath string
	now        func() time.Time
}

// NewFeed builds a Feed reading ledgerPath and persisting its cursor at
// cursorPath. now is injected for tests; nil means time.Now.
func NewFeed(ledgerPath string, cursorPath string, now func() time.Time) *Feed {
	if now == nil {
		now = time.Now
	}
	return &Feed{ledgerPath: ledgerPath, cursorPath: cursorPath, now: now}
}

// Unseen returns the ephemeral block listing the deliveries to chatID (an
// OpenClaw Conversation-info chat_id such as "telegram:-5447333904") appended
// to the ledger since the previous call, and advances the cursor to the end
// of the ledger. It returns "" when there is nothing new or on any error
// (missing/unreadable ledger, malformed chatID): the feed is fail-open and must
// never fail the turn. An unwritable cursor only means the next call may
// repeat entries.
func (f *Feed) Unseen(chatID string) string {
	channel, target, ok := strings.Cut(chatID, ":")
	if !ok || channel == "" || target == "" {
		return ""
	}
	data, err := os.ReadFile(f.ledgerPath)
	if err != nil {
		return ""
	}
	offset, hasCursor := f.readCursor()
	f.writeCursor(len(data))

	var lines []string
	if !hasCursor {
		lines = matching(data, channel, target, f.now().Add(-firstReadWindow), f.now())
		if len(lines) > firstReadMax {
			lines = lines[len(lines)-firstReadMax:]
		}
	} else {
		if offset > len(data) {
			offset = 0
		}
		lines = matching(data[offset:], channel, target, time.Time{}, f.now())
	}
	if len(lines) == 0 {
		return ""
	}
	return "<delivered_since_last_turn>\n" +
		"Delivered to you in this chat by background runs since your last turn " +
		"(already sent; do not repeat or re-offer them):\n" +
		strings.Join(lines, "\n") + "\n</delivered_since_last_turn>"
}

// matching returns one formatted line per ledger entry addressed to
// channel/target whose timestamp is not before since. Today's entries render as
// "HH:MM title", older ones as "YYYY-MM-DD HH:MM title".
func matching(data []byte, channel string, target string, since time.Time, now time.Time) []string {
	today := now.Format("2006-01-02")
	var out []string
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 0, 4096), 1<<20)
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
				title = strings.TrimPrefix(field, "title=")
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

func (f *Feed) readCursor() (int, bool) {
	file, err := os.Open(f.cursorPath)
	if err != nil {
		return 0, false
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, 64))
	if err != nil {
		return 0, false
	}
	offset, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil || offset < 0 {
		return 0, false
	}
	return offset, true
}

func (f *Feed) writeCursor(offset int) {
	if err := os.MkdirAll(filepath.Dir(f.cursorPath), 0o700); err != nil {
		return
	}
	_ = os.WriteFile(f.cursorPath, []byte(fmt.Sprintf("%d\n", offset)), 0o600)
}
