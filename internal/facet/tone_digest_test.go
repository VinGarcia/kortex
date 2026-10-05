package facet

import (
	"fmt"
	"strings"
	"testing"
)

// annLine builds one canonical annotation line via the production renderer, so
// these tests parse exactly the format the input annotator emits (no drift
// between annotationLine and parseAnnotationLine).
func annLine(valence string, emotions ...Emotion) string {
	return annotationLine(0, ParagraphAnnotation{Investment: 3, Valence: valence, Emotions: emotions})
}

func emo(name string, level int) Emotion { return Emotion{Emotion: name, Level: level} }

func TestBuildToneDigest_dedupKeepsMaxLevelAndCount(t *testing.T) {
	userTexts := []string{
		"oi\n" + annLine("positiva", emo("entusiasmo", 3)),
		"segue\n" + annLine("positiva", emo("entusiasmo", 4)) + "\n" + annLine("positiva", emo("entusiasmo", 2)),
	}
	got := BuildToneDigest(userTexts, 0)

	// entusiasmo(positiva) appears 3 times, max level 4 → ONE deduped line.
	want := "- entusiasmo — nível máx 4, valência positiva, 3x"
	if !strings.Contains(got, want) {
		t.Fatalf("digest missing deduped entry %q:\n%s", want, got)
	}
	if n := strings.Count(got, "entusiasmo"); n != 1 {
		t.Errorf("entusiasmo listed %d times, want 1 (deduped)", n)
	}
}

func TestBuildToneDigest_keepsAllValences(t *testing.T) {
	userTexts := []string{
		annLine("positiva", emo("alegria", 3)),
		annLine("negativa", emo("medo", 3)),
		annLine("mista", emo("ambivalência", 3)),
		annLine("neutra", emo("curiosidade", 3)),
	}
	got := BuildToneDigest(userTexts, 0)
	for _, v := range []string{"positiva", "negativa", "mista", "neutra"} {
		if !strings.Contains(got, "valência "+v) {
			t.Errorf("digest dropped valence %q:\n%s", v, got)
		}
	}
}

func TestBuildToneDigest_sameEmotionDifferentValenceStaySeparate(t *testing.T) {
	userTexts := []string{
		annLine("positiva", emo("tensão", 3)),
		annLine("negativa", emo("tensão", 4)),
	}
	got := BuildToneDigest(userTexts, 0)
	if n := strings.Count(got, "tensão"); n != 2 {
		t.Errorf("tensão listed %d times, want 2 (different valences not collapsed):\n%s", n, got)
	}
}

func TestBuildToneDigest_emptyWhenNoAnnotations(t *testing.T) {
	// Negative control: plain conversation text with no annotation lines yields
	// no digest at all (fail-open → caller injects nothing).
	cases := [][]string{
		nil,
		{"", ""},
		{"só texto puro\nsem nenhuma tag de emoção aqui"},
		{annLine("neutra") /* emotions=nenhuma */},
	}
	for i, texts := range cases {
		if got := BuildToneDigest(texts, 0); got != "" {
			t.Errorf("case %d: want empty digest, got:\n%s", i, got)
		}
	}
}

func TestBuildToneDigest_minLevelCut(t *testing.T) {
	userTexts := []string{
		annLine("positiva", emo("leve", 2), emo("forte", 4)),
	}
	// minLevel 0 keeps both; minLevel 3 drops the level-2 emotion.
	all := BuildToneDigest(userTexts, 0)
	if !strings.Contains(all, "leve") || !strings.Contains(all, "forte") {
		t.Errorf("minLevel 0 should include both emotions:\n%s", all)
	}
	cut := BuildToneDigest(userTexts, 3)
	if strings.Contains(cut, "leve") {
		t.Errorf("minLevel 3 should drop the level-2 emotion:\n%s", cut)
	}
	if !strings.Contains(cut, "forte") {
		t.Errorf("minLevel 3 should keep the level-4 emotion:\n%s", cut)
	}
}

func TestBuildToneDigest_minLevelCutToEmpty(t *testing.T) {
	userTexts := []string{annLine("positiva", emo("leve", 2))}
	if got := BuildToneDigest(userTexts, 3); got != "" {
		t.Errorf("all emotions below cut → empty digest, got:\n%s", got)
	}
}

func TestBuildToneDigest_boundedOutput(t *testing.T) {
	// More distinct emotions than the cap: the block must bound the list and
	// summarize the remainder as an omitted count.
	var lines []string
	total := maxDigestEmotions + 5
	for i := 0; i < total; i++ {
		lines = append(lines, annLine("positiva", emo(fmt.Sprintf("emo%02d", i), 3)))
	}
	got := BuildToneDigest([]string{strings.Join(lines, "\n")}, 0)

	listed := strings.Count(got, "- emo")
	if listed != maxDigestEmotions {
		t.Errorf("listed %d emotions, want the cap %d", listed, maxDigestEmotions)
	}
	wantOmitted := fmt.Sprintf("(+%d emoção", total-maxDigestEmotions)
	if !strings.Contains(got, wantOmitted) {
		t.Errorf("missing omitted-count note %q:\n%s", wantOmitted, got)
	}
}

func TestBuildToneDigest_deterministicOrdering(t *testing.T) {
	// Highest level first, then highest count, then name. Build input where all
	// three tie-breakers are exercised and assert the exact line order is stable.
	userTexts := []string{
		annLine("positiva", emo("baixa", 2)),
		annLine("positiva", emo("alta", 5)),
		annLine("positiva", emo("media_freq", 3)),
		annLine("positiva", emo("media_freq", 3)), // count 2, beats media_rare (count 1) at same level
		annLine("positiva", emo("media_rare", 3)),
	}
	first := BuildToneDigest(userTexts, 0)
	// Stable across repeated calls (map iteration order must not leak).
	for i := 0; i < 5; i++ {
		if again := BuildToneDigest(userTexts, 0); again != first {
			t.Fatalf("digest not deterministic across calls:\n--- a ---\n%s\n--- b ---\n%s", first, again)
		}
	}
	order := []string{"alta", "media_freq", "media_rare", "baixa"}
	var lastIdx int
	for _, name := range order {
		idx := strings.Index(first, "- "+name+" ")
		if idx < 0 {
			t.Fatalf("emotion %q missing from digest:\n%s", name, first)
		}
		if idx < lastIdx {
			t.Fatalf("emotion %q out of expected order %v:\n%s", name, order, first)
		}
		lastIdx = idx
	}
}

func TestBuildToneDigest_delimitedBlock(t *testing.T) {
	got := BuildToneDigest([]string{annLine("positiva", emo("alegria", 4))}, 0)
	if !strings.HasPrefix(got, toneDigestOpen) {
		t.Errorf("digest must open with %q:\n%s", toneDigestOpen, got)
	}
	if !strings.HasSuffix(got, toneDigestClose) {
		t.Errorf("digest must close with %q:\n%s", toneDigestClose, got)
	}
}

func TestParseAnnotationLine(t *testing.T) {
	tests := []struct {
		name        string
		line        string
		wantOK      bool
		wantValence string
		wantEmotion []Emotion
	}{
		{
			name:        "normal line with emotions",
			line:        annLine("negativa", emo("exaustão", 4), emo("cobrança", 3)),
			wantOK:      true,
			wantValence: "negativa",
			wantEmotion: []Emotion{emo("exaustão", 4), emo("cobrança", 3)},
		},
		{
			name:        "nenhuma yields valid line, no emotions",
			line:        annLine("neutra"),
			wantOK:      true,
			wantValence: "neutra",
			wantEmotion: nil,
		},
		{
			name:        "leading/trailing whitespace tolerated",
			line:        "   " + annLine("positiva", emo("alegria", 2)) + "  ",
			wantOK:      true,
			wantValence: "positiva",
			wantEmotion: []Emotion{emo("alegria", 2)},
		},
		{name: "not an annotation line", line: "just prose", wantOK: false},
		{name: "marker but unknown valence", line: "[emoções p1: investment=0 valence=bogus emotions=nenhuma]", wantOK: false},
		{
			name:        "emotion name with a space",
			line:        annLine("positiva", emo("orgulho tranquilo", 3)),
			wantOK:      true,
			wantValence: "positiva",
			wantEmotion: []Emotion{emo("orgulho tranquilo", 3)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valence, emotions, ok := parseAnnotationLine(tt.line)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			if valence != tt.wantValence {
				t.Errorf("valence = %q, want %q", valence, tt.wantValence)
			}
			if len(emotions) != len(tt.wantEmotion) {
				t.Fatalf("emotions = %+v, want %+v", emotions, tt.wantEmotion)
			}
			for i := range emotions {
				if emotions[i] != tt.wantEmotion[i] {
					t.Errorf("emotion %d = %+v, want %+v", i, emotions[i], tt.wantEmotion[i])
				}
			}
		})
	}
}

func TestToneDigest_facetDigestFailOpen(t *testing.T) {
	// The facet adapter returns the same content as BuildToneDigest on the happy
	// path and "" on empty input (the fail-open contract).
	d := NewToneDigest(0)
	texts := []string{annLine("positiva", emo("alegria", 4))}
	if d.Digest(texts) != BuildToneDigest(texts, 0) {
		t.Errorf("facet Digest diverged from BuildToneDigest")
	}
	if got := d.Digest(nil); got != "" {
		t.Errorf("empty input → empty digest, got %q", got)
	}
}
