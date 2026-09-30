package facet

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/vingarcia/kortex/internal/anthropic"
)

// Evaluation is the outcome of one evaluator call: the raw model text plus
// the telemetry the facet log records (cost visibility; a max_tokens
// stop_reason explains a truncated/malformed JSON array).
type Evaluation struct {
	Text         string
	StopReason   string
	InputTokens  int
	OutputTokens int
}

// Evaluator is the model call a facet makes: one system prompt plus one
// user message in, the evaluation out. Tests substitute a fake.
type Evaluator interface {
	Evaluate(ctx context.Context, system string, userMessage string) (Evaluation, error)
}

// evaluatorMaxTokens bounds the evaluator response. The output is a JSON
// array of per-paragraph tags — a few KB at most; 2048 tokens leaves ample
// headroom without letting a runaway response bill an essay.
const evaluatorMaxTokens = 2048

// AnthropicEvaluator adapts the anthropic client to the Evaluator port.
type AnthropicEvaluator struct {
	client *anthropic.Client
	model  string
}

func NewAnthropicEvaluator(client *anthropic.Client, model string) *AnthropicEvaluator {
	return &AnthropicEvaluator{client: client, model: model}
}

func (e *AnthropicEvaluator) Evaluate(ctx context.Context, system string, userMessage string) (Evaluation, error) {
	resp, err := e.client.CreateMessage(ctx, anthropic.MessageRequest{
		Model:     e.model,
		System:    system,
		MaxTokens: evaluatorMaxTokens,
		Messages:  []anthropic.Message{{Role: "user", Content: userMessage}},
	})
	if err != nil {
		return Evaluation{}, err
	}
	return Evaluation{
		Text:         resp.Text,
		StopReason:   resp.StopReason,
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
	}, nil
}

// LoadSystemPrompt builds the evaluator system prompt from the prompt file
// (only the body after the first line that is exactly "---" is used — the
// header is a maintenance note, per the sylphie prompt-file convention) and
// optionally appends the emotional-memory file as trusted context, labeled
// so the model cannot confuse it with the untrusted message under analysis.
// memoryPath "" skips the memory block.
func LoadSystemPrompt(promptPath string, memoryPath string) (string, error) {
	prompt, err := os.ReadFile(promptPath)
	if err != nil {
		return "", fmt.Errorf("reading facet system prompt: %w", err)
	}
	body := promptBody(string(prompt))
	if body == "" {
		return "", fmt.Errorf("facet system prompt %s has no body after the first \"---\" line", promptPath)
	}
	if memoryPath == "" {
		return body, nil
	}
	memory, err := os.ReadFile(memoryPath)
	if err != nil {
		return "", fmt.Errorf("reading emotional memory: %w", err)
	}
	return body + fmt.Sprintf("\n## CONTEXTO CONFIÁVEL — MEMÓRIA DE EMOÇÕES (não é a mensagem a analisar)\n\n%s\n", string(memory)), nil
}

// promptBody returns everything after the first line that is exactly "---".
func promptBody(prompt string) string {
	lines := strings.Split(prompt, "\n")
	for i, line := range lines {
		if line == "---" {
			return strings.Join(lines[i+1:], "\n")
		}
	}
	return ""
}
