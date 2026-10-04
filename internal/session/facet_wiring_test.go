package session

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/history"
	"github.com/vingarcia/kortex/internal/tools"
)

// fakeAnnotator records every text handed to it and returns a fixed rewrite,
// standing in for the input-annotator facet.
type fakeAnnotator struct {
	seen    []string
	rewrite string
}

func (f *fakeAnnotator) Annotate(text string) string {
	f.seen = append(f.seen, text)
	return f.rewrite
}

// fakeTurnEvaluator records every completed-turn notification, standing in for
// the output-evaluator facet.
type fakeTurnEvaluator struct {
	indices   []int
	snapshots []history.Snapshot
}

func (f *fakeTurnEvaluator) EvaluateCompletedTurn(turnIndex int, snapshot history.Snapshot) {
	f.indices = append(f.indices, turnIndex)
	f.snapshots = append(f.snapshots, snapshot)
}

// TestRun_WiresFacets proves the native path runs the input annotator over the
// incoming user message (its rewrite reaches the model) and fires the turn
// evaluator on completion with the session's accumulated history — the F3d
// regression this test guards against (facets inert on the native path).
func TestRun_WiresFacets(t *testing.T) {
	server, rec := textResponder(t, "the-reply")
	client := anthropic.NewClient("secret", server.URL, time.Second)

	ann := &fakeAnnotator{rewrite: "ANNOTATED"}
	eval := &fakeTurnEvaluator{}

	in := strings.Join([]string{
		initializeLine("sys"),
		userLine("sess-1", "raw user text"),
		userLine("sess-1", "second raw"),
	}, "\n") + "\n"
	var out strings.Builder

	err := Run(context.Background(), Config{
		Stdin: strings.NewReader(in), Stdout: &out, Client: client,
		Dispatcher: tools.NewDispatcher(), Store: NewStore(t.TempDir()),
		SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		Annotator: ann, TurnEvaluator: eval,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// (a) the annotator saw both raw user messages, untouched.
	if len(ann.seen) != 2 || ann.seen[0] != "raw user text" || ann.seen[1] != "second raw" {
		t.Errorf("annotator inputs = %q, want the two raw user texts", ann.seen)
	}
	// The annotated text is what the model received: the last turn's final
	// user message must be the rewrite, not the raw text.
	if rec.lastUserText != "ANNOTATED" {
		t.Errorf("model user text = %q, want the annotator rewrite", rec.lastUserText)
	}

	// (b) the evaluator fired once per completed turn, with ascending indices.
	if len(eval.indices) != 2 || eval.indices[0] != 0 || eval.indices[1] != 1 {
		t.Fatalf("evaluator turn indices = %v, want [0 1]", eval.indices)
	}
	// Its snapshot carried the accumulated history: turn 1's snapshot holds
	// both turns, each with the annotated user text and the assistant reply.
	last := eval.snapshots[1]
	if len(last.Turns) != 2 {
		t.Fatalf("last snapshot turns = %d, want 2", len(last.Turns))
	}
	if last.Turns[0].UserText != "ANNOTATED" || last.Turns[0].AssistantText != "the-reply" {
		t.Errorf("snapshot turn 0 = %+v, want annotated user + reply assistant", last.Turns[0])
	}
	if !last.Turns[1].Completed || last.Turns[1].AssistantText != "the-reply" {
		t.Errorf("snapshot turn 1 = %+v, want completed with the reply", last.Turns[1])
	}
}

// TestRun_NoFacetsIsClean proves the native path still runs with no facets
// wired (nil Annotator/TurnEvaluator): the pre-facet behavior is untouched.
func TestRun_NoFacetsIsClean(t *testing.T) {
	server, _ := textResponder(t, "ok")
	client := anthropic.NewClient("secret", server.URL, time.Second)

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("s", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: NewStore(t.TempDir()), SessionID: "s", Model: "m", MaxTokens: 10, NewUUID: seqUUID(),
	})
	if err != nil {
		t.Fatalf("Run with no facets: %v", err)
	}
}
