package wgsl

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// Position is a human readable source location (1-based line and column;
// the column counts code points).
type Position struct {
	Offset int
	Line   int
	Col    int
}

func (p Position) String() string { return fmt.Sprintf("%d:%d", p.Line, p.Col) }

// Source is the text of a module with a line table.
type Source struct {
	Name  string
	Text  string
	lines []int // byte offset of the start of each line
}

// NewSource indexes the line breaks of text as defined by the WGSL spec
// (LF, VT, FF, CR, CRLF, NEL, LS, PS).
func NewSource(name, text string) *Source {
	s := &Source{Name: name, Text: text, lines: []int{0}}
	for i := 0; i < len(text); {
		r, n := utf8.DecodeRuneInString(text[i:])
		switch r {
		case '\n', '\v', '\f', 0x85, 0x2028, 0x2029:
			s.lines = append(s.lines, i+n)
		case '\r':
			if i+1 < len(text) && text[i+1] == '\n' {
				n = 2
			}
			s.lines = append(s.lines, i+n)
		}
		i += n
	}
	return s
}

// Position converts a byte offset to a line and column.
func (s *Source) Position(off int) Position {
	if off < 0 {
		off = 0
	}
	if off > len(s.Text) {
		off = len(s.Text)
	}
	l := sort.Search(len(s.lines), func(i int) bool { return s.lines[i] > off }) - 1
	col := utf8.RuneCountInString(s.Text[s.lines[l]:off]) + 1
	return Position{Offset: off, Line: l + 1, Col: col}
}

// Error is a positioned diagnostic.
type Error struct {
	Pos  Position
	Name string // source name, may be empty
	Msg  string
	Rule string // optional stable rule id, e.g. "reserved-word"
}

func (e *Error) Error() string {
	name := e.Name
	if name == "" {
		name = "<wgsl>"
	}
	return fmt.Sprintf("%s:%d:%d: %s", name, e.Pos.Line, e.Pos.Col, e.Msg)
}

// ErrorList is a list of diagnostics; it is an error when non-empty.
type ErrorList []*Error

func (l ErrorList) Error() string {
	switch len(l) {
	case 0:
		return "no errors"
	case 1:
		return l[0].Error()
	}
	var b strings.Builder
	for i, e := range l {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(e.Error())
	}
	return b.String()
}

// Err returns nil for an empty list.
func (l ErrorList) Err() error {
	if len(l) == 0 {
		return nil
	}
	return l
}

func (s *Source) errAt(off int, rule, format string, a ...any) *Error {
	return &Error{Pos: s.Position(off), Name: s.Name, Msg: fmt.Sprintf(format, a...), Rule: rule}
}
