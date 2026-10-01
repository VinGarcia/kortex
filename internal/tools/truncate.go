package tools

import "unicode/utf8"

// truncateUTF8 cuts data to at most max bytes. A raw byte-offset cut can
// split a multibyte UTF-8 rune in half, leaving an invalid trailing byte
// sequence; since the cut content ends up inside a tool_result's JSON
// string, that invalid UTF-8 would risk breaking the Messages API request
// encoding. truncateUTF8 backs the cut off rune-by-rune until what remains
// decodes cleanly, so callers always get valid UTF-8 back. It reports
// whether data was long enough to need cutting at all.
func truncateUTF8(data []byte, max int) (cut []byte, truncated bool) {
	if len(data) <= max {
		return data, false
	}
	cut = data[:max]
	for len(cut) > 0 {
		r, size := utf8.DecodeLastRune(cut)
		if r != utf8.RuneError || size != 1 {
			break
		}
		cut = cut[:len(cut)-1]
	}
	return cut, true
}
