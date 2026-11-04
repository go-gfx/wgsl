package wgsl

import "testing"

// The seed corpus lives in testdata/fuzz/<Fuzz name>/ (written by
// internal/fuzzseed from the vendored real shaders plus hand-written edge
// cases; fuzzer-found inputs are added by `go test -fuzz`). `go test` runs
// every committed input as a regression test.

func fuzzLimits() Options {
	return Options{Limits: Limits{MaxSource: 1 << 16, MaxTokens: 1 << 14, MaxDepth: 64}}
}

func FuzzLex(f *testing.F) {
	f.Add(sample)
	f.Fuzz(func(t *testing.T, src string) {
		_, _ = Lex(NewSource("", src), fuzzLimits().Limits)
	})
}

func FuzzParse(f *testing.F) {
	f.Add(sample)
	f.Fuzz(func(t *testing.T, src string) {
		if _, err := Parse(src, fuzzLimits()); err != nil {
			if _, ok := err.(*Error); !ok {
				t.Fatalf("non-positioned error %T", err)
			}
		}
	})
}

// FuzzRoundTrip: whatever parses must print to text that parses to an equal
// AST, and printing must be idempotent.
func FuzzRoundTrip(f *testing.F) {
	f.Add(sample)
	f.Fuzz(func(t *testing.T, src string) {
		m, err := Parse(src, fuzzLimits())
		if err != nil {
			return
		}
		out := Print(m)
		m2, err := Parse(out, fuzzLimits())
		if err != nil {
			t.Fatalf("printed text does not parse: %v\n%s", err, out)
		}
		if !Equal(m, m2) {
			t.Fatalf("round trip changed the AST:\n%s", out)
		}
		if Print(m2) != out {
			t.Fatalf("printer not idempotent:\n%s", out)
		}
	})
}

// FuzzValidate: the checker must never panic or hang on parseable input.
func FuzzValidate(f *testing.F) {
	f.Add(sample)
	f.Fuzz(func(t *testing.T, src string) {
		_ = Validate(src, fuzzLimits())
	})
}
