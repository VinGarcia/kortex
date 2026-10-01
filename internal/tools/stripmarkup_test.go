package tools

import "testing"

func TestStripToolMarkup(t *testing.T) {
	tests := []struct {
		desc string
		text string
		want string
	}{
		{
			desc: "no markup passes through unchanged",
			text: "The answer is 42.",
			want: "The answer is 42.",
		},
		{
			desc: "empty string passes through unchanged",
			text: "",
			want: "",
		},
		{
			desc: "only markup strips to empty",
			text: "<invoke name=\"Bash\">\n<parameter name=\"command\">ls</parameter>\n</invoke>",
			want: "",
		},
		{
			desc: "markup mixed with surrounding prose keeps the prose",
			text: "Running the command now.\n" +
				"<invoke name=\"Bash\">\n<parameter name=\"command\">ls</parameter>\n</invoke>\n" +
				"Done, see output above.",
			want: "Running the command now.\n\nDone, see output above.",
		},
		{
			desc: "multiple invoke blocks are both removed",
			text: "<invoke name=\"Bash\">\n<parameter name=\"command\">ls</parameter>\n</invoke>" +
				"between" +
				"<invoke name=\"Bash\">\n<parameter name=\"command\">pwd</parameter>\n</invoke>",
			want: "between",
		},
		{
			desc: "invoke with multiple parameters strips the whole block",
			text: "before " +
				"<invoke name=\"FileWrite\">\n<parameter name=\"path\">a.txt</parameter>\n<parameter name=\"content\">hi</parameter>\n</invoke>" +
				" after",
			want: "before  after",
		},
		{
			desc: "unclosed invoke tag is left untouched (conservative)",
			text: "Some prose.\n<invoke name=\"Bash\">\n<parameter name=\"command\">ls</parameter>",
			want: "Some prose.\n<invoke name=\"Bash\">\n<parameter name=\"command\">ls</parameter>",
		},
		{
			desc: "bare invoke tag with no attributes and no parameters still strips",
			text: "x<invoke></invoke>y",
			want: "xy",
		},
		{
			desc: "stray closing tag with no opening tag is left untouched",
			text: "prose </invoke> more prose",
			want: "prose </invoke> more prose",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := StripToolMarkup(test.text)
			if got != test.want {
				t.Errorf("StripToolMarkup(%q) = %q, want %q", test.text, got, test.want)
			}
		})
	}
}
