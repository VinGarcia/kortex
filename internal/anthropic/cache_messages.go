package anthropic

import "encoding/json"

// cacheIneligibleContentTypes are content-block types the Messages API
// rejects a cache_control marker on. Thinking blocks (emitted by models that
// think by default, e.g. Fable) can appear in an assistant turn's raw content,
// so markLastEligibleRawBlock must skip them and place the marker on the
// nearest eligible block instead of blindly using the literal last element.
var cacheIneligibleContentTypes = map[string]bool{
	"thinking":          true,
	"redacted_thinking": true,
}

// MarkMessagesCachePrefix returns messages with one ephemeral cache_control
// breakpoint added to the last eligible content block of the LAST message, so
// a follow-up call whose prefix up to that point is byte-identical reads the
// entire conversation history back from cache (~0.1x price) instead of
// re-billing it in full every turn. markStableSystemPrefix already caches
// tools+system; this closes the gap on the growing message history, which
// previously went out unmarked on every call.
//
// One moving breakpoint is enough for incremental caching across a whole
// tool-loop session: each round trip appends at most two messages (the
// assistant's turn, then the tool_result/user reply) before the next call, so
// the new breakpoint always lands well inside the Messages API's 20-position
// cache lookback window and finds the previous call's cache entry. Combined
// with the system breakpoint, this uses 2 of the API's 4-breakpoint ceiling.
//
// It never mutates messages or anything inside it — every level it touches
// (the last message, and its content slice when the content is a slice) is a
// fresh copy, so the caller's original slice is untouched. It is fail-safe
// like markStableSystemPrefix: when there is nothing eligible to mark (empty
// history, empty content, or a last message made entirely of ineligible
// blocks), it returns messages unchanged and the request proceeds uncached,
// exactly as the API would behave with no marker.
//
// Call this ONLY with canonical conversation history — never with a slice
// that already has a volatile tail appended (e.g. a per-turn tone digest),
// or every call would write a fresh, never-read cache entry for the whole
// conversation instead of reusing the last one (the "shared prefix, varying
// suffix" anti-pattern). A caller that appends such a tail must call this on
// the real history first, then append the tail to the result — see
// tools.withEphemeralContext, which wraps both steps in that order so no call
// site has to get the ordering right on its own.
//
// Observability gap (known, accepted): unlike markStableSystemPrefix, this
// breakpoint is not covered by CacheWatchdog. The API's usage counters
// (cache_read_input_tokens / cache_creation_input_tokens) are totals across
// ALL breakpoints in the request, not broken down per breakpoint, so there is
// no way to tell "the system prefix cached but the message history didn't"
// from usage alone — the system breakpoint's own healthy reads would mask a
// silently-failing message breakpoint in the aggregate. Detecting that would
// need baselining cache_creation growth across turns (whether each turn's
// write stays roughly constant at the full history size instead of shrinking
// to just the new delta), which is real new monitoring surface, not a small
// addition to the existing per-request classifier — left as a follow-up
// rather than folded into this change.
func MarkMessagesCachePrefix(messages []Message) []Message {
	if len(messages) == 0 {
		return messages
	}
	marked, ok := markLastEligibleBlock(messages[len(messages)-1])
	if !ok {
		return messages
	}
	out := make([]Message, len(messages))
	copy(out, messages)
	out[len(out)-1] = marked
	return out
}

// markLastEligibleBlock returns msg with a cache_control breakpoint placed on
// its last eligible content block, and whether one was found. msg.Content
// must be one of the shapes this client ever assigns to a history message
// (see Message's doc): a plain string, []ToolResultBlock, or []json.RawMessage
// (an echoed assistant turn from MessageResponse.RawContent). Any other shape
// — including nil or empty content — is reported ineligible rather than
// guessed at.
//
// The []json.RawMessage case is not reachable from either of tools.RunLoop's
// two call sites today: by construction, an assistant's raw-content message
// is always immediately followed by a user message in the same append (see
// RunLoop's tool_use branch), so it is never the trailing element when
// MarkMessagesCachePrefix next runs. It is kept — proven correct by its own
// unit tests — because MarkMessagesCachePrefix is an exported function
// documented to handle every shape Message.Content takes, and the
// architecture review for this change identified a near-term caller that
// WOULD exercise it (the active-governor draft-review path, which sends raw
// assistant content as its trailing message). Simplifying this away now
// would just mean re-deriving it when that caller lands.
func markLastEligibleBlock(msg Message) (Message, bool) {
	switch content := msg.Content.(type) {
	case string:
		if content == "" {
			return Message{}, false
		}
		msg.Content = []messageContentBlock{{
			Type:         "text",
			Text:         content,
			CacheControl: &cacheControl{Type: "ephemeral"},
		}}
		return msg, true

	case []ToolResultBlock:
		if len(content) == 0 {
			return Message{}, false
		}
		cloned := make([]ToolResultBlock, len(content))
		copy(cloned, content)
		cloned[len(cloned)-1].CacheControl = &cacheControl{Type: "ephemeral"}
		msg.Content = cloned
		return msg, true

	case []json.RawMessage:
		marked, ok := markLastEligibleRawBlock(content)
		if !ok {
			return Message{}, false
		}
		msg.Content = marked
		return msg, true

	default:
		return Message{}, false
	}
}

// markLastEligibleRawBlock walks blocks backward from the end, skipping any
// whose "type" the API won't accept a cache_control marker on (see
// cacheIneligibleContentTypes), and returns a copy of blocks with a marker
// injected into the first eligible one it finds. ok is false when blocks is
// empty or every block in it is ineligible or undecodable.
func markLastEligibleRawBlock(blocks []json.RawMessage) ([]json.RawMessage, bool) {
	for i := len(blocks) - 1; i >= 0; i-- {
		var typed struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(blocks[i], &typed); err != nil {
			continue
		}
		if cacheIneligibleContentTypes[typed.Type] {
			continue
		}
		withMarker, err := addCacheControl(blocks[i])
		if err != nil {
			continue
		}
		out := make([]json.RawMessage, len(blocks))
		copy(out, blocks)
		out[i] = withMarker
		return out, true
	}
	return nil, false
}

// addCacheControl decodes a raw content block into its field map, injects
// cache_control, and re-encodes it. Re-encoding through a map reorders keys
// (Go sorts map keys when marshaling), which is harmless: the API does not
// care about JSON key order, and cache_control never participates in the
// prefix-matching hash — moving it between requests does not invalidate a
// previously-cached prefix.
func addCacheControl(raw json.RawMessage) (json.RawMessage, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	cc, err := json.Marshal(cacheControl{Type: "ephemeral"})
	if err != nil {
		return nil, err
	}
	fields["cache_control"] = cc
	return json.Marshal(fields)
}

// messageContentBlock is the array-of-blocks form a plain string
// Message.Content must become to carry a cache_control marker — the API has
// no way to attach cache_control to a bare string content field.
type messageContentBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}
