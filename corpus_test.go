package wgsl

import (
	"os"
	"path/filepath"
	"testing"
)

// roundTrip checks Print(Parse(src)) re-parses to an equal AST and that
// printing is idempotent.
func roundTrip(src string) error {
	m, err := Parse(src)
	if err != nil {
		return err
	}
	out := Print(m)
	m2, err := Parse(out)
	if err != nil {
		return err
	}
	if !Equal(m, m2) {
		return &Error{Msg: "round trip changed the AST"}
	}
	if Print(m2) != out {
		return &Error{Msg: "printer not idempotent"}
	}
	return nil
}

// TestRealShaders parses and round-trips every vendored real shader
// (testdata/real/*/*.wgsl, vendored by internal/corpus).
func TestRealShaders(t *testing.T) {
	files, err := filepath.Glob("testdata/real/*/*.wgsl")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) < 30 {
		t.Fatalf("only %d real shaders vendored", len(files))
	}
	// Shaders that the host repositories' own tests feed to the GPU on
	// purpose, to check that an invalid module is reported: `meta` is a
	// reserved word (the failure that went unnoticed in the LLM work) and
	// `fn main( {` is a syntax error. They must be rejected.
	invalid := map[string]string{
		"examples__batchcompute__main.go.322.wgsl": "reserved-word",
		"examples__batchcompute__main.go.323.wgsl": "syntax",
		"gpubatch__module_test.go.619.wgsl":        "syntax",
		"batch__device_js_test.go.247.wgsl":        "reserved-word",
		"batch__device_js_test.go.249.wgsl":        "syntax",
	}
	ok, rejected := 0, 0
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if rule, bad := invalid[filepath.Base(f)]; bad {
			_, err := Parse(string(b))
			if e, isErr := err.(*Error); !isErr || e.Rule != rule {
				t.Errorf("%s: want a %s error, got %v", f, rule, err)
			}
			rejected++
			continue
		}
		if err := roundTrip(string(b)); err != nil {
			t.Errorf("%s: %v", f, err)
		}
		ok++
	}
	if rejected != len(invalid) {
		t.Errorf("expected %d invalid shaders in the corpus, found %d", len(invalid), rejected)
	}
	t.Logf("%d real shaders parse and round-trip; %d invalid-by-design are rejected", ok, rejected)
}
