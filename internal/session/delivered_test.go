package session

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/tools"
)

// fakeDeliveredFeed returns its queued blocks in order (one per call, "" once
// exhausted), records the chat ids it was asked about and the commits it got.
// Each call returns its call number as the offset, so commits identify turns.
type fakeDeliveredFeed struct {
	blocks  []string
	chatIDs []string
	commits []int64
}

func (f *fakeDeliveredFeed) Unseen(chatID string) (string, int64, bool) {
	f.chatIDs = append(f.chatIDs, chatID)
	offset := int64(len(f.chatIDs))
	if len(f.blocks) == 0 {
		return "", offset, true
	}
	block := f.blocks[0]
	f.blocks = f.blocks[1:]
	return block, offset, true
}

func (f *fakeDeliveredFeed) Commit(nextOffset int64) {
	f.commits = append(f.commits, nextOffset)
}

// TestRun_DeliveredFeed proves the "delivered since your last turn" block is
// appended to the ephemeral tail (after the tone digest), keyed on the inbound
// chat_id, never persisted, and that the request prefix before the tail is
// byte-identical to a run without the feed (the prompt-cache prefix is intact).
func TestRun_DeliveredFeed(t *testing.T) {
	const digest = "<<<TOM>>>"
	const block = "<delivered_since_last_turn>\n13:00 x\n</delivered_since_last_turn>"
	marker := "⟦openclaw:ctx⟧"
	withInfo := func(text string) string {
		return "Conversation info: " + marker + "\n```json\n" +
			`{"chat_id":"telegram:-5447333904","message_id":"1","sender":{"name":"V"}}` +
			"\n```\n\n" + text
	}

	run := func(t *testing.T, feed DeliveredFeed) [][]anthropic.Message {
		var calls [][]anthropic.Message
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body struct {
				Messages []anthropic.Message `json:"messages"`
			}
			json.NewDecoder(r.Body).Decode(&body)
			calls = append(calls, body.Messages)
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
		}))
		defer server.Close()
		in := strings.Join([]string{
			initializeLine("sys"),
			userLine("s", withInfo("oi")),
			userLine("s", withInfo("e agora?")),
		}, "\n") + "\n"
		var out strings.Builder
		store := NewStore(t.TempDir())
		err := Run(context.Background(), Config{
			Stdin: strings.NewReader(in), Stdout: &out, Client: anthropic.NewClient("secret", server.URL, time.Second),
			Dispatcher: tools.NewDispatcher(), Store: store, SessionID: "s", Model: "m", MaxTokens: 1024,
			NewUUID: seqUUID(), ToneDigester: &fakeToneDigester{digest: digest}, DeliveredFeed: feed,
		})
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		saved, err := store.Load("s")
		if err != nil {
			t.Fatalf("store.Load: %v", err)
		}
		for i, m := range saved {
			if s, ok := m.Content.(string); ok && strings.Contains(s, "delivered_since_last_turn") {
				t.Errorf("persisted message %d carries the delivered block", i)
			}
		}
		if len(calls) != 2 {
			t.Fatalf("captured %d core calls, want 2", len(calls))
		}
		return calls
	}

	feed := &fakeDeliveredFeed{blocks: []string{block}}
	withFeed := run(t, feed)
	withoutFeed := run(t, nil)

	if want := []string{"telegram:-5447333904", "telegram:-5447333904"}; !reflect.DeepEqual(feed.chatIDs, want) {
		t.Errorf("feed chat ids = %v, want %v", feed.chatIDs, want)
	}
	if want := []int64{1, 2}; !reflect.DeepEqual(feed.commits, want) {
		t.Errorf("commits = %v, want %v (one per successful turn)", feed.commits, want)
	}
	if last := withFeed[0][len(withFeed[0])-1]; last.Content != digest+"\n\n"+block {
		t.Errorf("turn 1 tail = %+v, want digest then delivered block", last.Content)
	}
	if last := withFeed[1][len(withFeed[1])-1]; last.Content != digest {
		t.Errorf("turn 2 tail = %+v, want digest only (nothing new delivered)", last.Content)
	}
	for turn := range withFeed {
		got, want := withFeed[turn], withoutFeed[turn]
		if !reflect.DeepEqual(got[:len(got)-1], want[:len(want)-1]) {
			t.Errorf("turn %d: request prefix differs with the feed wired; the delivered block leaked out of the ephemeral tail", turn+1)
		}
	}
}

func TestJoinEphemeral(t *testing.T) {
	tests := []struct {
		desc   string
		blocks []string
		want   string
	}{
		{desc: "all empty sends no tail", blocks: []string{"", ""}, want: ""},
		{desc: "only digest", blocks: []string{"d", ""}, want: "d"},
		{desc: "only delivered", blocks: []string{"", "x"}, want: "x"},
		{desc: "both, digest first", blocks: []string{"d", "x"}, want: "d\n\nx"},
	}
	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			if got := joinEphemeral(tc.blocks...); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// TestRun_DeliveredFeedNotCommittedOnFailedTurn proves a turn whose core call
// fails does not mark its deliveries seen.
func TestRun_DeliveredFeedNotCommittedOnFailedTurn(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusBadRequest)
	}))
	defer server.Close()
	text := "Conversation info: ⟦openclaw:ctx⟧\n```json\n" + `{"chat_id":"telegram:1"}` + "\n```\n\noi"
	feed := &fakeDeliveredFeed{blocks: []string{"<delivered_since_last_turn>x</delivered_since_last_turn>"}}
	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("s", text) + "\n"),
		Stdout: &out, Client: anthropic.NewClient("secret", server.URL, time.Second), Dispatcher: tools.NewDispatcher(),
		Store: NewStore(t.TempDir()), SessionID: "s", Model: "m", MaxTokens: 10, NewUUID: seqUUID(), DeliveredFeed: feed,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(feed.chatIDs) != 1 {
		t.Fatalf("feed asked %d times, want 1", len(feed.chatIDs))
	}
	if len(feed.commits) != 0 {
		t.Errorf("commits = %v, want none after a failed turn", feed.commits)
	}
}
