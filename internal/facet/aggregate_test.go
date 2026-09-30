package facet

import "testing"

// para builds a ParagraphAnnotation for aggregate tests, where only
// investment and valence matter.
func para(investment int, valence string) ParagraphAnnotation {
	return ParagraphAnnotation{Investment: investment, Valence: valence}
}

// TestAggregate covers the emo_max_investment + emo_valence_of_max cases
// from sylphie's scripts/lib/emotion-eval.sh, including the divergent-tie ⇒
// mista rule (#4085).
func TestAggregate(t *testing.T) {
	tests := []struct {
		desc string
		anns []ParagraphAnnotation
		want Aggregate
	}{
		{
			desc: "single paragraph carries its own valence",
			anns: []ParagraphAnnotation{para(4, "negativa")},
			want: Aggregate{Investment: 4, Valence: "negativa"},
		},
		{
			desc: "max wins over average: one heavy paragraph among neutrals",
			anns: []ParagraphAnnotation{para(0, "neutra"), para(5, "negativa"), para(0, "neutra")},
			want: Aggregate{Investment: 5, Valence: "negativa"},
		},
		{
			desc: "valence comes from the max paragraph, not the others",
			anns: []ParagraphAnnotation{para(2, "negativa"), para(4, "positiva")},
			want: Aggregate{Investment: 4, Valence: "positiva"},
		},
		{
			desc: "tie at max with divergent valences aggregates to mista",
			anns: []ParagraphAnnotation{para(3, "positiva"), para(3, "negativa")},
			want: Aggregate{Investment: 3, Valence: "mista"},
		},
		{
			desc: "tie at max with the same valence keeps it (unique, not count)",
			anns: []ParagraphAnnotation{para(3, "negativa"), para(3, "negativa")},
			want: Aggregate{Investment: 3, Valence: "negativa"},
		},
		{
			desc: "neutra at the max is ignored when a directed valence is tied",
			anns: []ParagraphAnnotation{para(2, "neutra"), para(2, "positiva")},
			want: Aggregate{Investment: 2, Valence: "positiva"},
		},
		{
			desc: "mista at the max stays mista",
			anns: []ParagraphAnnotation{para(3, "mista")},
			want: Aggregate{Investment: 3, Valence: "mista"},
		},
		{
			desc: "mista tied with another directed valence is still mista",
			anns: []ParagraphAnnotation{para(3, "mista"), para(3, "negativa")},
			want: Aggregate{Investment: 3, Valence: "mista"},
		},
		{
			desc: "only neutra at the max aggregates to neutra",
			anns: []ParagraphAnnotation{para(1, "neutra"), para(0, "negativa")},
			want: Aggregate{Investment: 1, Valence: "neutra"},
		},
		{
			desc: "empty input aggregates to the zero charge",
			anns: nil,
			want: Aggregate{Investment: 0, Valence: "neutra"},
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := aggregate(test.anns); got != test.want {
				t.Errorf("aggregate() = %+v, want %+v", got, test.want)
			}
		})
	}
}

// TestGateWouldFire pins the tag-based triggers of the gate prompt's
// primary path: fires at min (default 4) regardless of valence, and at
// min-1 only with valence negativa or mista.
func TestGateWouldFire(t *testing.T) {
	tests := []struct {
		desc string
		agg  Aggregate
		min  int
		want bool
	}{
		{desc: "at threshold fires even with positive valence", agg: Aggregate{4, "positiva"}, min: 4, want: true},
		{desc: "above threshold fires", agg: Aggregate{5, "neutra"}, min: 4, want: true},
		{desc: "threshold-1 with negativa fires", agg: Aggregate{3, "negativa"}, min: 4, want: true},
		{desc: "threshold-1 with mista fires", agg: Aggregate{3, "mista"}, min: 4, want: true},
		{desc: "threshold-1 with positiva does not fire", agg: Aggregate{3, "positiva"}, min: 4, want: false},
		{desc: "threshold-1 with neutra does not fire", agg: Aggregate{3, "neutra"}, min: 4, want: false},
		{desc: "threshold-2 with negativa does not fire", agg: Aggregate{2, "negativa"}, min: 4, want: false},
		{desc: "custom lower threshold shifts both triggers", agg: Aggregate{1, "negativa"}, min: 2, want: true},
		{desc: "custom lower threshold still needs direction below it", agg: Aggregate{1, "positiva"}, min: 2, want: false},
		{desc: "threshold 0 fires on anything", agg: Aggregate{0, "neutra"}, min: 0, want: true},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := gateWouldFire(test.agg, test.min); got != test.want {
				t.Errorf("gateWouldFire(%+v, %d) = %v, want %v", test.agg, test.min, got, test.want)
			}
		})
	}
}
