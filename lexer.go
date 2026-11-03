package wgsl

import (
	"unicode"
	"unicode/utf8"
)

// Kind is a token kind.
type Kind uint8

const (
	TEOF       Kind = iota
	TIdent          // identifier (never a keyword or reserved word)
	TKeyword        // one of the spec's keywords (including true/false)
	TInt            // integer literal
	TFloat          // floating point literal
	TPunct          // operator or punctuation; "_" is a TPunct
	TTmplStart      // '<' that starts a template list
	TTmplEnd        // '>' that ends a template list
)

// Token is a lexical token. Text is the exact source spelling.
type Token struct {
	Kind Kind
	Text string
	Off  int
	End  int
}

// Limits bounds every resource a parse may consume. Zero fields take the
// defaults of DefaultLimits.
type Limits struct {
	MaxSource int // bytes of source text
	MaxTokens int // tokens, excluding EOF
	MaxDepth  int // nesting depth of expressions, templates and statements
}

// DefaultLimits are generous for real shaders and tight against bombs.
func DefaultLimits() Limits {
	return Limits{MaxSource: 4 << 20, MaxTokens: 1 << 20, MaxDepth: 256}
}

func (l Limits) withDefaults() Limits {
	d := DefaultLimits()
	if l.MaxSource <= 0 {
		l.MaxSource = d.MaxSource
	}
	if l.MaxTokens <= 0 {
		l.MaxTokens = d.MaxTokens
	}
	if l.MaxDepth <= 0 {
		l.MaxDepth = d.MaxDepth
	}
	return l
}

var keywords = map[string]bool{}
var reserved = map[string]bool{}

func init() {
	for _, k := range []string{"alias", "break", "case", "const", "const_assert", "continue", "continuing",
		"default", "diagnostic", "discard", "else", "enable", "false", "fn", "for", "if", "let", "loop",
		"override", "requires", "return", "struct", "switch", "true", "var", "while"} {
		keywords[k] = true
	}
	for _, k := range reservedWords {
		reserved[k] = true
	}
}

// IsKeyword reports whether s is a WGSL keyword.
func IsKeyword(s string) bool { return keywords[s] }

// IsReserved reports whether s is a reserved word of the pinned spec.
func IsReserved(s string) bool { return reserved[s] }

func isBlank(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\v', '\f', '\r', 0x85, 0x200E, 0x200F, 0x2028, 0x2029:
		return true
	}
	return false
}

// UAX31 profile of the spec: Start = XID_Start + '_', Continue = Start +
// XID_Continue. The Go standard library has no XID tables, so this uses the
// ID_Start/ID_Continue approximation (letters, Nl, Other_ID_Start; plus Mn,
// Mc, Nd, Pc, Other_ID_Continue). It differs from XID only on a handful of
// NFKC-unstable code points.
func isIDStart(r rune) bool {
	if r < utf8.RuneSelf {
		return r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z'
	}
	return unicode.IsLetter(r) || unicode.Is(unicode.Nl, r) || unicode.Is(unicode.Other_ID_Start, r)
}

func isIDContinue(r rune) bool {
	if r < utf8.RuneSelf {
		return r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9'
	}
	return isIDStart(r) || unicode.In(r, unicode.Mn, unicode.Mc, unicode.Nd, unicode.Pc, unicode.Other_ID_Continue)
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }
func isHex(c byte) bool {
	return isDigit(c) || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'
}

// scanner holds the character-level scanning shared by template discovery and
// tokenization.
type scanner struct {
	src *Source
	s   string
}

// skipTrivia skips blankspace and comments. It returns the new offset or an
// error offset (>= 0 in errOff) for an unterminated block comment.
func (sc *scanner) skipTrivia(i int) (int, *Error) {
	s := sc.s
	for i < len(s) {
		c := s[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r':
			i++
		case c == '/' && i+1 < len(s) && s[i+1] == '/':
			for i < len(s) {
				r, n := utf8.DecodeRuneInString(s[i:])
				if r == '\n' || r == '\v' || r == '\f' || r == '\r' || r == 0x85 || r == 0x2028 || r == 0x2029 {
					break
				}
				i += n
			}
		case c == '/' && i+1 < len(s) && s[i+1] == '*':
			start := i
			depth := 0
			for {
				if i+1 >= len(s) {
					return i, sc.src.errAt(start, "unterminated-comment", "unterminated block comment")
				}
				if s[i] == '/' && s[i+1] == '*' {
					depth++
					i += 2
				} else if s[i] == '*' && s[i+1] == '/' {
					depth--
					i += 2
					if depth == 0 {
						break
					}
				} else {
					i++
				}
			}
		case c >= utf8.RuneSelf:
			r, n := utf8.DecodeRuneInString(s[i:])
			if r != utf8.RuneError && isBlank(r) {
				i += n
			} else {
				return i, nil
			}
		default:
			return i, nil
		}
	}
	return i, nil
}

// scanIdent returns the end of the identifier-like run at i, or i if none.
func (sc *scanner) scanIdent(i int) int {
	s := sc.s
	r, n := utf8.DecodeRuneInString(s[i:])
	if r == utf8.RuneError && n <= 1 || !isIDStart(r) {
		return i
	}
	j := i + n
	for j < len(s) {
		r, n := utf8.DecodeRuneInString(s[j:])
		if r == utf8.RuneError && n <= 1 || !isIDContinue(r) {
			break
		}
		j += n
	}
	return j
}

// scanNumber scans a numeric literal at i. It returns the end, whether it is
// a float, and ok=false (with a message) for a malformed literal. If no
// numeric literal starts at i it returns end == i.
func (sc *scanner) scanNumber(i int) (end int, float bool, msg string) {
	s := sc.s
	n := len(s)
	digits := func(j int, hex bool) int {
		for j < n && (isDigit(s[j]) || hex && isHex(s[j])) {
			j++
		}
		return j
	}
	exp := func(j int, e1, e2 byte) (int, bool) {
		if j < n && (s[j] == e1 || s[j] == e2) {
			k := j + 1
			if k < n && (s[k] == '+' || s[k] == '-') {
				k++
			}
			if k < n && isDigit(s[k]) {
				return digits(k, false), true
			}
		}
		return j, false
	}
	switch {
	case i >= n:
		return i, false, ""
	case s[i] == '0' && i+1 < n && (s[i+1] == 'x' || s[i+1] == 'X'):
		j := digits(i+2, true)
		ndig := j - (i + 2)
		dot := false
		if j < n && s[j] == '.' {
			k := digits(j+1, true)
			if ndig+(k-j-1) > 0 {
				dot = true
				ndig += k - j - 1
				j = k
			}
		}
		if ndig == 0 {
			return i + 1, false, "" // "0" then identifier x...: parser reports
		}
		j2, hasExp := exp(j, 'p', 'P')
		if hasExp {
			j = j2
			if j < n && (s[j] == 'f' || s[j] == 'h') {
				j++
			}
			return j, true, ""
		}
		if dot {
			return j, true, ""
		}
		if j < n && (s[j] == 'i' || s[j] == 'u') {
			j++
		}
		return j, false, ""
	case isDigit(s[i]) || s[i] == '.' && i+1 < n && isDigit(s[i+1]):
		j := digits(i, false)
		intDigits := j - i
		isFloat := false
		if j < n && s[j] == '.' {
			isFloat = true
			j = digits(j+1, false)
		}
		if j2, ok := exp(j, 'e', 'E'); ok {
			isFloat = true
			j = j2
		}
		if isFloat {
			if j < n && (s[j] == 'f' || s[j] == 'h') {
				j++
			}
			return j, true, ""
		}
		// integer or suffixed float without point/exponent
		if j < n && (s[j] == 'f' || s[j] == 'h') {
			if intDigits > 1 && s[i] == '0' {
				return j + 1, true, "leading zeros are not allowed"
			}
			return j + 1, true, ""
		}
		if intDigits > 1 && s[i] == '0' {
			return j, false, "leading zeros are not allowed in an integer literal"
		}
		if j < n && (s[j] == 'i' || s[j] == 'u') {
			j++
		}
		return j, false, ""
	}
	return i, false, ""
}

// discoverTemplates implements the "Template list discovery" algorithm of the
// spec (section 3.9). It returns the byte offsets of the '<' that start and of
// the '>' that end template lists.
func (sc *scanner) discoverTemplates() (starts, ends map[int]bool, err *Error) {
	s := sc.s
	starts = map[int]bool{}
	ends = map[int]bool{}
	type cand struct{ pos, depth int }
	var pending []cand
	depth := 0
	i := 0
	for {
		var e *Error
		i, e = sc.skipTrivia(i)
		if e != nil {
			return nil, nil, e
		}
		if i >= len(s) {
			return
		}
		if end, _, _ := sc.scanNumber(i); end > i {
			i = end
			continue
		}
		if end := sc.scanIdent(i); end > i {
			i, e = sc.skipTrivia(end)
			if e != nil {
				return nil, nil, e
			}
			if i < len(s) && s[i] == '<' {
				pending = append(pending, cand{i, depth})
				i++
				if i < len(s) && (s[i] == '<' || s[i] == '=') {
					pending = pending[:len(pending)-1]
					i++
				}
			}
			continue
		}
		c := s[i]
		switch {
		case c == '>':
			if len(pending) > 0 && pending[len(pending)-1].depth == depth {
				t := pending[len(pending)-1]
				starts[t.pos] = true
				ends[i] = true
				pending = pending[:len(pending)-1]
				i++
				continue
			}
			i++
			if i < len(s) && s[i] == '=' {
				i++
			}
		case c == '(' || c == '[':
			depth++
			i++
		case c == ')' || c == ']':
			for len(pending) > 0 && pending[len(pending)-1].depth >= depth {
				pending = pending[:len(pending)-1]
			}
			if depth > 0 {
				depth--
			}
			i++
		case c == '!':
			i++
			if i < len(s) && s[i] == '=' {
				i++
			}
		case c == '=':
			i++
			if i < len(s) && s[i] == '=' {
				i++
				continue
			}
			depth = 0
			pending = pending[:0]
		case c == ';' || c == '{' || c == ':':
			depth = 0
			pending = pending[:0]
			i++
		case c == '&' && i+1 < len(s) && s[i+1] == '&', c == '|' && i+1 < len(s) && s[i+1] == '|':
			for len(pending) > 0 && pending[len(pending)-1].depth >= depth {
				pending = pending[:len(pending)-1]
			}
			i += 2
		default:
			_, n := utf8.DecodeRuneInString(s[i:])
			i += n
		}
	}
}

var puncts3 = []string{"<<=", ">>="}
var puncts2 = []string{"&&", "||", "->", "==", "!=", ">=", "<=", "<<", ">>", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="}

// Lex tokenizes src. The final token is EOF. Reserved words, malformed
// literals, unterminated comments and invalid identifiers are errors.
func Lex(src *Source, lim Limits) ([]Token, error) {
	lim = lim.withDefaults()
	if len(src.Text) > lim.MaxSource {
		return nil, src.errAt(lim.MaxSource, "limit-source", "source of %d bytes exceeds the limit of %d", len(src.Text), lim.MaxSource)
	}
	if !utf8.ValidString(src.Text) {
		for i := 0; i < len(src.Text); {
			r, n := utf8.DecodeRuneInString(src.Text[i:])
			if r == utf8.RuneError && n == 1 {
				return nil, src.errAt(i, "utf8", "invalid UTF-8")
			}
			i += n
		}
	}
	sc := &scanner{src: src, s: src.Text}
	starts, ends, err := sc.discoverTemplates()
	if err != nil {
		return nil, err
	}
	s := src.Text
	var toks []Token
	i := 0
	for {
		var e *Error
		i, e = sc.skipTrivia(i)
		if e != nil {
			return nil, e
		}
		if i >= len(s) {
			toks = append(toks, Token{Kind: TEOF, Off: len(s), End: len(s)})
			return toks, nil
		}
		if len(toks) >= lim.MaxTokens {
			return nil, src.errAt(i, "limit-tokens", "more than %d tokens", lim.MaxTokens)
		}
		if end, isFloat, msg := sc.scanNumber(i); end > i {
			if msg != "" {
				return nil, src.errAt(i, "bad-literal", "%s: %q", msg, s[i:end])
			}
			k := TInt
			if isFloat {
				k = TFloat
			}
			toks = append(toks, Token{k, s[i:end], i, end})
			i = end
			continue
		}
		if end := sc.scanIdent(i); end > i {
			t := s[i:end]
			switch {
			case t == "_":
				toks = append(toks, Token{TPunct, t, i, end})
			case len(t) >= 2 && t[0] == '_' && t[1] == '_':
				return nil, src.errAt(i, "ident-double-underscore", "identifier %q must not start with two underscores", t)
			case keywords[t]:
				toks = append(toks, Token{TKeyword, t, i, end})
			case reserved[t]:
				return nil, src.errAt(i, "reserved-word", "%q is a reserved word and cannot be used", t)
			default:
				toks = append(toks, Token{TIdent, t, i, end})
			}
			i = end
			continue
		}
		if starts[i] {
			toks = append(toks, Token{TTmplStart, "<", i, i + 1})
			i++
			continue
		}
		if ends[i] {
			toks = append(toks, Token{TTmplEnd, ">", i, i + 1})
			i++
			continue
		}
		matched := ""
		for _, p := range puncts3 {
			if len(s)-i >= 3 && s[i:i+3] == p {
				matched = p
			}
		}
		if matched == "" {
			for _, p := range puncts2 {
				if len(s)-i >= 2 && s[i:i+2] == p {
					matched = p
					break
				}
			}
		}
		if matched == "" {
			switch s[i] {
			case '&', '@', '/', '!', '[', ']', '{', '}', ':', ',', '=', '>', '<', '%', '(', ')', '.', '-', '+', '|', ';', '*', '~', '^':
				matched = s[i : i+1]
			}
		}
		if matched == "" {
			r, _ := utf8.DecodeRuneInString(s[i:])
			return nil, src.errAt(i, "bad-char", "unexpected character %q", r)
		}
		toks = append(toks, Token{TPunct, matched, i, i + len(matched)})
		i += len(matched)
	}
}
