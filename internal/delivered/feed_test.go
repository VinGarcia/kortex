package delivered

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func fixedNow() time.Time {
	return time.Date(2026, 11, 13, 14, 0, 0, 0, time.Local)
}

func ledgerLine(ts string, title string, target string) string {
	return ts + "\tdelivered\ttitle=" + title + "\tchannel=telegram\ttarget=" + target + "\n"
}

func appendFile(t *testing.T, path string, s string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteString(s); err != nil {
		t.Fatal(err)
	}
}

func TestUnseen(t *testing.T) {
	t.Run("first read is bounded to the last 3 of the last 24h, then the cursor advances", func(t *testing.T) {
		dir := t.TempDir()
		ledger := filepath.Join(dir, "delivered.log")
		appendFile(t, ledger, ledgerLine("2026-11-11 10:00:00", "too old", "1")+
			ledgerLine("2026-11-13 09:00:00", "a", "1")+
			ledgerLine("2026-11-13 10:00:00", "b", "1")+
			ledgerLine("2026-11-13 11:00:00", "other chat", "2")+
			ledgerLine("2026-11-13 12:00:00", "c", "1")+
			ledgerLine("2026-11-13 13:00:00", "d", "1"))
		feed := NewFeed(ledger, filepath.Join(dir, "cur", "s1"), fixedNow)

		got := feed.Unseen("telegram:1")
		for _, want := range []string{"10:00 b", "12:00 c", "13:00 d"} {
			if !strings.Contains(got, want) {
				t.Errorf("first read missing %q:\n%s", want, got)
			}
		}
		for _, unwanted := range []string{"too old", " a\n", "other chat"} {
			if strings.Contains(got, unwanted) {
				t.Errorf("first read contains %q:\n%s", unwanted, got)
			}
		}

		if got := feed.Unseen("telegram:1"); got != "" {
			t.Errorf("second read without new deliveries = %q, want empty", got)
		}

		appendFile(t, ledger, ledgerLine("2026-11-12 23:00:00", "yesterday", "1"))
		got = feed.Unseen("telegram:1")
		if !strings.Contains(got, "2026-11-12 23:00 yesterday") {
			t.Errorf("new delivery not reported with date:\n%s", got)
		}
	})

	t.Run("cursors are isolated per session", func(t *testing.T) {
		dir := t.TempDir()
		ledger := filepath.Join(dir, "delivered.log")
		appendFile(t, ledger, ledgerLine("2026-11-13 13:00:00", "x", "-5447333904"))
		s1 := NewFeed(ledger, filepath.Join(dir, "s1"), fixedNow)
		s2 := NewFeed(ledger, filepath.Join(dir, "s2"), fixedNow)

		if got := s1.Unseen("telegram:-5447333904"); !strings.Contains(got, "13:00 x") {
			t.Fatalf("s1 = %q", got)
		}
		if got := s2.Unseen("telegram:-5447333904"); !strings.Contains(got, "13:00 x") {
			t.Errorf("s2 should still see the delivery s1 consumed, got %q", got)
		}
	})

	t.Run("shrunk ledger is re-read from the start", func(t *testing.T) {
		dir := t.TempDir()
		ledger := filepath.Join(dir, "delivered.log")
		cursor := filepath.Join(dir, "s1")
		appendFile(t, ledger, ledgerLine("2026-11-13 13:00:00", "rotated", "1"))
		if err := os.WriteFile(cursor, []byte("99999\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		if got := NewFeed(ledger, cursor, fixedNow).Unseen("telegram:1"); !strings.Contains(got, "rotated") {
			t.Errorf("got %q", got)
		}
	})

	t.Run("fail-open cases inject nothing", func(t *testing.T) {
		dir := t.TempDir()
		empty := filepath.Join(dir, "empty.log")
		appendFile(t, empty, "")
		tests := []struct {
			desc   string
			ledger string
			chatID string
		}{
			{desc: "missing ledger", ledger: filepath.Join(dir, "nope.log"), chatID: "telegram:1"},
			{desc: "empty ledger", ledger: empty, chatID: "telegram:1"},
			{desc: "ledger is a directory", ledger: dir, chatID: "telegram:1"},
			{desc: "no chat id", ledger: empty, chatID: ""},
			{desc: "chat id without channel", ledger: empty, chatID: "180614014"},
		}
		for _, tc := range tests {
			t.Run(tc.desc, func(t *testing.T) {
				if got := NewFeed(tc.ledger, filepath.Join(dir, "cur-"+tc.desc), fixedNow).Unseen(tc.chatID); got != "" {
					t.Errorf("got %q, want empty", got)
				}
			})
		}
	})
}
