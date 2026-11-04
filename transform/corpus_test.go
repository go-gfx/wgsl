package transform

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/go-gfx/wgsl"
)

// The kernels of go-wasmcontainers/llm (SmolLM2 and Qwen2.5-0.5B: tiled
// matvecs with strided loops and workgroupBarrier, the Q4_0 embedding gather,
// the Q8_0 head, NEOX rope, attention with subgroup-free reductions) are the
// real-shader corpus of the loop budget. Each must be rewritten, the result
// must re-parse to the same tree, and no loop may share a counter with an
// ancestor loop: a counter shared with an enclosing loop makes that loop's
// check non-uniform when the inner loop's trip count differs between
// invocations, and the browser then rejects a barrier in the outer loop
// (this broke 4 of the 19 Qwen modules before the per-depth counters).
func TestLoopBudgetCorpus(t *testing.T) {
	files, err := filepath.Glob("testdata/llm-qwen2/*.wgsl")
	if err != nil || len(files) < 15 {
		t.Fatalf("corpus: %d files, %v", len(files), err)
	}
	loopsSeen := 0
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		m := parse(t, string(b))
		out, rep, err := LoopBudget(m, Config{Budget: BudgetSource{Group: 3, Binding: 1}, FlagGroup: 3, FlagBinding: 0})
		if err != nil {
			t.Errorf("%s: %v", f, err)
			continue
		}
		loopsSeen += rep.Loops
		if rep.Loops == 0 {
			continue
		}
		text := wgsl.Print(out)
		if !wgsl.Equal(out, parse(t, text)) {
			t.Errorf("%s: the output does not round-trip", f)
		}
		for _, d := range out.Decls {
			if fn, ok := d.(*wgsl.Func); ok {
				if bad := sharedCounter(fn.Body.Stmts, nil); bad != "" {
					t.Errorf("%s: %s: counter %s is shared with an enclosing loop", f, fn.Name, bad)
				}
			}
		}
	}
	if loopsSeen < 20 {
		t.Errorf("only %d loops in the corpus", loopsSeen)
	}
}

// counterOf is the counter a rewritten loop checks (its body's first
// statement is `counter += 1u`).
func counterOf(body []wgsl.Stmt) string {
	if len(body) > 0 {
		if a, ok := body[0].(*wgsl.Assign); ok && a.Op == "+=" {
			if id, ok := a.LHS.(*wgsl.Ident); ok {
				return id.Name
			}
		}
	}
	return ""
}

// sharedCounter returns the first counter a loop uses that an enclosing loop
// uses too ("" if none).
func sharedCounter(ss []wgsl.Stmt, ancestors []string) string {
	for _, s := range ss {
		var body []wgsl.Stmt
		var cont []wgsl.Stmt
		loop := true
		switch s := s.(type) {
		case *wgsl.Loop:
			body = s.Body
			if s.Cont != nil {
				cont = s.Cont.Stmts
			}
		case *wgsl.For:
			body = s.Body.Stmts
		case *wgsl.While:
			body = s.Body.Stmts
		case *wgsl.Block:
			if bad := sharedCounter(s.Stmts, ancestors); bad != "" {
				return bad
			}
			loop = false
		case *wgsl.If:
			if bad := sharedCounter([]wgsl.Stmt{s.Then}, ancestors); bad != "" {
				return bad
			}
			if s.Else != nil {
				if bad := sharedCounter([]wgsl.Stmt{s.Else}, ancestors); bad != "" {
					return bad
				}
			}
			loop = false
		case *wgsl.Switch:
			for _, c := range s.Cases {
				if bad := sharedCounter([]wgsl.Stmt{c.Body}, ancestors); bad != "" {
					return bad
				}
			}
			loop = false
		default:
			loop = false
		}
		if !loop {
			continue
		}
		name := counterOf(body)
		for _, a := range ancestors {
			if a == name {
				return name
			}
		}
		if bad := sharedCounter(append(body, cont...), append(append([]string(nil), ancestors...), name)); bad != "" {
			return bad
		}
	}
	return ""
}
