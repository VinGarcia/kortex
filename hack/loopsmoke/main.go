// Command loopsmoke runs internal/tools.RunLoop against the LIVE Anthropic
// Messages API, closing F3b's caveat that RunLoop had only ever been exercised
// against httptest fakes. It builds the real OAuth client plus the Dispatcher
// with the native builtins, declares the bash tool, and asks Haiku to run one
// innocuous command so the loop actually drives
// model -> tool_use -> RunBash -> tool_result -> terminal text.
//
// The OAuth token is read from the same secrets file hack/toolloop.sh uses and
// is only ever handed to the client as a Bearer header; it is never printed.
package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/tools"
)

const (
	tokenFile = "/home/vingarcia/.openclaw/secrets/kortex-setup-token"
	// Haiku 4.5: claude-fable-5 is 429-rate-limited on this OAuth (F3a).
	model = "claude-haiku-4-5-20251001"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		os.Exit(1)
	}
}

func run() error {
	tokenBytes, err := os.ReadFile(tokenFile)
	if err != nil {
		return fmt.Errorf("reading oauth token file: %w", err)
	}
	token := strings.TrimSpace(string(tokenBytes))
	if token == "" {
		return fmt.Errorf("oauth token file %s is empty", tokenFile)
	}

	client := anthropic.NewClient(token, "", 60*time.Second)
	dispatcher := tools.NewDispatcher()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	result, err := tools.RunLoop(ctx, client, dispatcher, tools.LoopRequest{
		Model:       model,
		System:      "You are a helpful assistant with access to a bash tool. When asked to run a shell command, call the bash tool instead of guessing the output.",
		MaxTokens:   1024,
		UserMessage: "Run the shell command `echo hello-from-kortex` and tell me its exact output.",
	})
	if err != nil {
		return err
	}

	fmt.Printf("turns=%d stop_reason=%q\n", result.Turns, result.StopReason)
	fmt.Printf("final_text=%q\n", result.FinalText)

	if result.Turns < 2 {
		return fmt.Errorf("no tool_use round trip: loop finished in %d turn(s), want >= 2", result.Turns)
	}
	if result.StopReason != "end_turn" {
		return fmt.Errorf("want terminal stop_reason=end_turn, got %q", result.StopReason)
	}
	if !strings.Contains(result.FinalText, "hello-from-kortex") {
		return fmt.Errorf("final text does not reflect the tool_result output: %q", result.FinalText)
	}

	fmt.Println("PASS: RunLoop drove model -> tool_use -> RunBash -> tool_result -> end_turn live")
	return nil
}
