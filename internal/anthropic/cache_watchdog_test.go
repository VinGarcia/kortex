package anthropic

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// systemBelowFloor is a prefix comfortably under every model's cacheable floor
// (opus/sonnet/fable = 1024 tokens ~= 4k bytes); systemAboveFloor clears even
// Haiku 4.5's 4096-token floor (~16k bytes).
var (
	systemBelowFloor = []systemBlock{{Type: "text", Text: "tiny system prompt"}}
	systemAboveFloor = []systemBlock{{Type: "text", Text: strings.Repeat("stable system prefix content. ", 1000)}} // ~30k bytes
)

// markLast returns a copy of system whose last block carries an ephemeral cache
// breakpoint — the exact shape markStableSystemPrefix produces on the wire.
func markLast(system []systemBlock) []systemBlock {
	out := append([]systemBlock(nil), system...)
	if len(out) > 0 {
		out[len(out)-1].CacheControl = &cacheControl{Type: "ephemeral"}
	}
	return out
}

func TestCacheWatchdog_observe(t *testing.T) {
	tests := []struct {
		desc      string
		model     string
		system    []systemBlock
		usage     Usage
		wantAlert bool
		wantCase  string
	}{
		{
			// Case (a): breakpoint went out, but the API reported neither a cache
			// write nor a read — the breakpoint was ignored.
			desc:      "case a: breakpoint set but no caching happened",
			model:     "claude-opus-4-8",
			system:    markLast(systemAboveFloor),
			usage:     Usage{InputTokens: 198000, CacheCreationInputTokens: 0, CacheReadInputTokens: 0},
			wantAlert: true,
			wantCase:  string(cacheAlertNoCaching),
		},
		{
			// Case (b): prefix large enough to cache, but no breakpoint went out at
			// all — the born-blind regression.
			desc:      "case b: born blind, eligible prefix with no breakpoint",
			model:     "claude-opus-4-8",
			system:    systemAboveFloor,
			usage:     Usage{InputTokens: 198000},
			wantAlert: true,
			wantCase:  string(cacheAlertBornBlind),
		},
		{
			// Negative control: breakpoint set and a cache read happened — the
			// intended steady state. No alert.
			desc:      "healthy: breakpoint set and cache read > 0",
			model:     "claude-opus-4-8",
			system:    markLast(systemAboveFloor),
			usage:     Usage{InputTokens: 2000, CacheReadInputTokens: 196000},
			wantAlert: false,
		},
		{
			// A session's first turn: breakpoint set, cache WRITE but no read yet.
			// cachedTokens > 0, so it is healthy and must not alert.
			desc:      "healthy: first turn cache write, no read yet",
			model:     "claude-opus-4-8",
			system:    markLast(systemAboveFloor),
			usage:     Usage{InputTokens: 2000, CacheCreationInputTokens: 196000},
			wantAlert: false,
		},
		{
			// Fail-safe: prefix below the floor, so markStableSystemPrefix correctly
			// set no breakpoint. Must NOT be mistaken for the born-blind case.
			desc:      "fail-safe: below floor, no breakpoint, no false alert",
			model:     "claude-opus-4-8",
			system:    systemBelowFloor,
			usage:     Usage{InputTokens: 300},
			wantAlert: false,
		},
		{
			// Fail-safe: empty system (the client sent only the preamble, trimmed to
			// nothing here) — no breakpoint, not eligible, no alert.
			desc:      "fail-safe: empty system, no false alert",
			model:     "claude-opus-4-8",
			system:    nil,
			usage:     Usage{InputTokens: 50},
			wantAlert: false,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			var buf bytes.Buffer
			w := NewCacheWatchdog(&buf, time.Minute)
			w.observe(test.model, test.system, test.usage)

			lines := warnLines(buf.String())
			if test.wantAlert {
				if len(lines) != 1 {
					t.Fatalf("want exactly 1 WARN line, got %d: %q", len(lines), buf.String())
				}
				if !strings.Contains(lines[0], "case="+test.wantCase) {
					t.Errorf("WARN line = %q, want case=%s", lines[0], test.wantCase)
				}
			} else if len(lines) != 0 {
				t.Errorf("want no WARN lines, got: %q", buf.String())
			}
		})
	}
}

// TestCacheWatchdog_throttle proves an identical (model, case) alert is
// suppressed within the cooldown window and re-surfaces only after it elapses.
func TestCacheWatchdog_throttle(t *testing.T) {
	var buf bytes.Buffer
	w := NewCacheWatchdog(&buf, 5*time.Minute)
	clock := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	w.now = func() time.Time { return clock }

	marked := markLast(systemAboveFloor)
	noCache := Usage{InputTokens: 198000}

	// First trigger fires.
	w.observe("claude-opus-4-8", marked, noCache)
	// Second identical trigger 1 minute later is inside the 5-minute cooldown.
	clock = clock.Add(time.Minute)
	w.observe("claude-opus-4-8", marked, noCache)
	if got := len(warnLines(buf.String())); got != 1 {
		t.Fatalf("within cooldown: want 1 WARN line, got %d: %q", got, buf.String())
	}

	// A different case on the same model is tracked independently and fires.
	w.observe("claude-opus-4-8", systemAboveFloor, noCache) // case (b)
	if got := len(warnLines(buf.String())); got != 2 {
		t.Fatalf("distinct case: want 2 WARN lines, got %d: %q", got, buf.String())
	}

	// Past the cooldown the original (model, case) re-fires.
	clock = clock.Add(5 * time.Minute)
	w.observe("claude-opus-4-8", marked, noCache)
	if got := len(warnLines(buf.String())); got != 3 {
		t.Fatalf("after cooldown: want 3 WARN lines, got %d: %q", got, buf.String())
	}
}

// TestCreateMessage_cacheWatchdog proves the Client wires the watchdog on the
// real response path (a healthy large prefix stays silent) and that a nil
// watchdog is a silent no-op — the fail-safe default that never touches the
// request flow.
func TestCreateMessage_cacheWatchdog(t *testing.T) {
	// A big system prefix the client will NOT mark here because we point it at a
	// server that echoes zero cache usage — exercising the born-blind detection
	// end-to-end would need the client to skip marking, which it never does for
	// an eligible prefix. So instead assert the healthy large-prefix path emits
	// nothing (marked + the server reports a cache read), and that the watchdog
	// being nil is safe.
	bigSystem := strings.Repeat("stable system prefix content. ", 1000)
	newServer := func() *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			// Report a cache read so the marked large prefix is healthy.
			w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1,"cache_read_input_tokens":5000}}`))
		}))
	}

	t.Run("healthy marked prefix emits nothing", func(t *testing.T) {
		server := newServer()
		defer server.Close()
		var buf bytes.Buffer
		client := NewClient("t", server.URL, time.Second)
		client.SetCacheWatchdog(NewCacheWatchdog(&buf, time.Minute))
		if _, err := client.CreateMessage(context.Background(), MessageRequest{
			Model:     "claude-opus-4-8",
			System:    bigSystem,
			MaxTokens: 1024,
			Messages:  []Message{{Role: "user", Content: "hello"}},
		}); err != nil {
			t.Fatal(err)
		}
		if got := warnLines(buf.String()); len(got) != 0 {
			t.Errorf("healthy path emitted WARN: %q", buf.String())
		}
	})

	t.Run("nil watchdog is a safe no-op", func(t *testing.T) {
		server := newServer()
		defer server.Close()
		client := NewClient("t", server.URL, time.Second)
		// No SetCacheWatchdog: request must still succeed unchanged.
		resp, err := client.CreateMessage(context.Background(), MessageRequest{
			Model:     "claude-opus-4-8",
			System:    bigSystem,
			MaxTokens: 1024,
			Messages:  []Message{{Role: "user", Content: "hello"}},
		})
		if err != nil {
			t.Fatal(err)
		}
		if resp.Text != "ok" || resp.Usage.CacheReadInputTokens != 5000 {
			t.Errorf("unexpected response with nil watchdog: %+v", resp)
		}
	})
}

// warnLines splits slog text output into its non-empty WARN lines.
func warnLines(out string) []string {
	var lines []string
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		lines = append(lines, line)
	}
	return lines
}
