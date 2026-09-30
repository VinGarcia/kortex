package facet

import (
	"reflect"
	"strings"
	"testing"
)

func TestSegmentParagraphs(t *testing.T) {
	tests := []struct {
		desc string
		text string
		want []string
	}{
		{desc: "empty text", text: "", want: nil},
		{desc: "only blank lines", text: "\n \n\t\n", want: nil},
		{desc: "single paragraph", text: "oi tudo bem", want: []string{"oi tudo bem"}},
		{
			desc: "blank line separates paragraphs",
			text: "primeiro\n\nsegundo",
			want: []string{"primeiro", "segundo"},
		},
		{
			desc: "runs of blank lines count as one separator",
			text: "a\n\n\n\nb",
			want: []string{"a", "b"},
		},
		{
			desc: "whitespace-only line is a separator",
			text: "a\n \t \nb",
			want: []string{"a", "b"},
		},
		{
			desc: "internal newlines preserved inside a paragraph",
			text: "linha um\nlinha dois\n\noutro",
			want: []string{"linha um\nlinha dois", "outro"},
		},
		{
			desc: "leading and trailing blanks create no paragraphs",
			text: "\n\nmeio\n\n",
			want: []string{"meio"},
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := segmentParagraphs(test.text)
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("got %q, want %q", got, test.want)
			}
		})
	}
}

func TestExtractJSONArray(t *testing.T) {
	tests := []struct {
		desc string
		raw  string
		want string
	}{
		{desc: "bare array", raw: `[{"a":1}]`, want: `[{"a":1}]`},
		{desc: "code fence around array", raw: "```json\n[1,2]\n```", want: "[1,2]"},
		{desc: "prose around array", raw: "Here you go:\n[1]\nHope it helps", want: "[1]"},
		{desc: "greedy first-to-last bracket", raw: `x [1] y [2] z`, want: `[1] y [2]`},
		{desc: "no array", raw: "no brackets here", want: ""},
		{desc: "reversed brackets", raw: "] then [", want: ""},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := extractJSONArray(test.raw); got != test.want {
				t.Errorf("got %q, want %q", got, test.want)
			}
		})
	}
}

func TestParseAnnotations(t *testing.T) {
	tests := []struct {
		desc    string
		json    string
		count   int
		wantErr string
	}{
		{
			desc:  "valid two paragraphs",
			json:  `[{"investment":4,"valence":"negativa","emotions":[{"emotion":"exaustão","level":4,"about":"a noite toda"}]},{"investment":0,"valence":"neutra","emotions":[]}]`,
			count: 2,
		},
		{
			desc:    "count mismatch",
			json:    `[{"investment":0,"valence":"neutra","emotions":[]}]`,
			count:   2,
			wantErr: "1 annotations for 2 paragraphs",
		},
		{
			desc:    "not an array",
			json:    `{"investment":0}`,
			count:   1,
			wantErr: "not the expected JSON array",
		},
		{
			desc:    "investment out of range",
			json:    `[{"investment":7,"valence":"neutra","emotions":[]}]`,
			count:   1,
			wantErr: "investment 7 out of range",
		},
		{
			desc:    "non-integer investment",
			json:    `[{"investment":2.5,"valence":"neutra","emotions":[]}]`,
			count:   1,
			wantErr: "not the expected JSON array",
		},
		{
			desc:    "unknown valence",
			json:    `[{"investment":0,"valence":"feliz","emotions":[]}]`,
			count:   1,
			wantErr: `unknown valence "feliz"`,
		},
		{
			desc:    "empty emotion name",
			json:    `[{"investment":1,"valence":"positiva","emotions":[{"emotion":"","level":1,"about":"x"}]}]`,
			count:   1,
			wantErr: "emotion with empty name",
		},
		{
			desc:    "emotion level out of range",
			json:    `[{"investment":1,"valence":"positiva","emotions":[{"emotion":"alegria","level":6,"about":"x"}]}]`,
			count:   1,
			wantErr: "level 6 out of range",
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			_, err := parseAnnotations(test.json, test.count)
			if test.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("want error containing %q, got %v", test.wantErr, err)
			}
		})
	}
}

func TestInterleave(t *testing.T) {
	annotations := []ParagraphAnnotation{
		{Investment: 4, Valence: "negativa", Emotions: []Emotion{
			{Emotion: "exaustão", Level: 4},
			{Emotion: "co\nbrança", Level: 3}, // newline in name must collapse to a space
		}},
		{Investment: 0, Valence: "neutra", Emotions: nil},
	}
	got := interleave([]string{"linha um\nlinha dois", "operacional"}, annotations)
	want := "linha um\nlinha dois\n" +
		"[emoções p1: investment=4 valence=negativa emotions=exaustão(4), co brança(3)]\n\n" +
		"operacional\n" +
		"[emoções p2: investment=0 valence=neutra emotions=nenhuma]"
	if got != want {
		t.Errorf("got:\n%s\nwant:\n%s", got, want)
	}
}

func TestNumberParagraphs(t *testing.T) {
	tests := []struct {
		name       string
		paragraphs []string
		want       string
	}{
		{
			name:       "single paragraph",
			paragraphs: []string{"oi"},
			want:       "[P1]\noi",
		},
		{
			name:       "paragraph with internal newline keeps its lines under one marker",
			paragraphs: []string{"linha 1\nlinha 2", "fim"},
			want:       "[P1]\nlinha 1\nlinha 2\n\n[P2]\nfim",
		},
		{
			name:       "three blocks numbered in order",
			paragraphs: []string{"a", "b", "c"},
			want:       "[P1]\na\n\n[P2]\nb\n\n[P3]\nc",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := numberParagraphs(tt.paragraphs); got != tt.want {
				t.Errorf("numberParagraphs() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSegmentationContract(t *testing.T) {
	got := segmentationContract(7)
	for _, want := range []string{"em 7 parágrafos", "EXATAMENTE 7 elementos", `{"investment": 0, "valence": "neutra", "emotions": []}`} {
		if !strings.Contains(got, want) {
			t.Errorf("segmentationContract(7) missing %q:\n%s", want, got)
		}
	}
}
