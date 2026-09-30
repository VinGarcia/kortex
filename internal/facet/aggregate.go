package facet

// This file is the Go port of the gate-aggregate logic of sylphie's
// scripts/lib/emotion-eval.sh (emo_max_investment + emo_valence_of_max): the
// per-paragraph annotations are distilled into ONE (investment, valence)
// pair for the binary gate — the MAXIMUM investment across paragraphs (never
// the average: one heavy paragraph among ten operational ones must stay
// visible) carrying the valence of the paragraph(s) that reach that maximum.

// Aggregate is the distilled gate input for one assistant turn.
type Aggregate struct {
	// Investment is the maximum investment across all paragraphs.
	Investment int `json:"investment"`
	// Valence is the aggregated valence of the paragraphs that reach the
	// maximum investment.
	Valence string `json:"valence"`
}

// aggregate ports emo_max_investment + emo_valence_of_max. Valence rule
// (agreed with Vini 2026-09-24, #4085): among the paragraphs tied at the
// maximum investment, ignore "neutra" when any directed valence is present;
// exactly one distinct directed valence → itself; more than one (e.g. a
// positive and a negative paragraph tied at the max) → "mista"; only
// "neutra" at the max → "neutra". An empty slice aggregates to the zero
// charge {0, "neutra"}.
func aggregate(annotations []ParagraphAnnotation) Aggregate {
	if len(annotations) == 0 {
		return Aggregate{Investment: 0, Valence: "neutra"}
	}
	max := annotations[0].Investment
	for _, ann := range annotations[1:] {
		if ann.Investment > max {
			max = ann.Investment
		}
	}
	directed := map[string]bool{}
	for _, ann := range annotations {
		if ann.Investment == max && ann.Valence != "neutra" {
			directed[ann.Valence] = true
		}
	}
	valence := "neutra"
	switch len(directed) {
	case 0:
		// only neutra at the max
	case 1:
		for v := range directed {
			valence = v
		}
	default:
		valence = "mista"
	}
	return Aggregate{Investment: max, Valence: valence}
}

// gateWouldFire ports the TAG-based triggers of the primary path of
// sylphie's core/emotion-output-gate-prompt.md: "investment ≥ 4; ou
// investment = 3 com valence mista ou negativa" — generalized to a
// configurable minimum (minInvestment 4 keeps today's behavior exactly; see
// config.DefaultGateMinInvestment). The prompt's remaining OR triggers
// (severity/blast_radius/reversibility) belong to the model-backed gate and
// are out of this deterministic port's scope: kortex only has the emotion
// tags here.
func gateWouldFire(agg Aggregate, minInvestment int) bool {
	if agg.Investment >= minInvestment {
		return true
	}
	return agg.Investment == minInvestment-1 && (agg.Valence == "negativa" || agg.Valence == "mista")
}
