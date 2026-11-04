package transform

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-gfx/wgsl"
)

const loops = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    var n = 0u;
    loop { if n >= 5u { break; } n++; }
    for (var i = 0u; i < 4u; i++) {
        var j = 0u;
        while j < 3u { j++; for (var k = 0u; k < 2u; k++) { n++; } }
    }
    switch n { case 1: { while false {} } default: { } }
    out[0] = n;
}
`

func parse(t *testing.T, src string) *wgsl.Module {
	t.Helper()
	m, err := wgsl.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestLoopBudgetStructure(t *testing.T) {
	in := parse(t, loops)
	before := wgsl.Print(in)
	out, rep, err := LoopBudget(in, Config{Budget: BudgetSource{Group: 1, Binding: 0}, FlagGroup: 1, FlagBinding: 1})
	if err != nil {
		t.Fatal(err)
	}
	if wgsl.Print(in) != before {
		t.Fatal("input module was modified")
	}
	// loop, for, while, inner for, while (in switch) = 5 loops; three
	// outermost nests of depth 1 (loop), 3 (for, while, for) and 1 (the while in
	// the switch): one counter per depth = 5 counters.
	if rep.Loops != 5 || rep.Counters != 5 || !rep.Instrumented {
		t.Fatalf("%+v", rep)
	}
	text := wgsl.Print(out)
	for _, want := range []string{
		"@group(1) @binding(0) var<uniform> wgsl_loop_budget: u32;",
		"@group(1) @binding(1) var<storage, read_write> wgsl_truncated: atomic<u32>;",
		"atomicStore(&wgsl_truncated, 1u);",
		"var wgsl_loop_count: u32 = 0u;",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q in\n%s", want, text)
		}
	}
	// the transformed text re-parses to the same tree and validates
	m2 := parse(t, text)
	if !wgsl.Equal(out, m2) {
		t.Fatal("output does not round-trip")
	}
	if err := wgsl.Validate(text); err != nil {
		t.Fatalf("transformed shader does not validate: %v\n%s", err, text)
	}
}

func TestLoopBudgetOverride(t *testing.T) {
	out, _, err := LoopBudget(parse(t, loops), Config{Budget: BudgetSource{Override: true, Default: 1000}, FlagGroup: 0, FlagBinding: 1})
	if err != nil {
		t.Fatal(err)
	}
	text := wgsl.Print(out)
	if !strings.Contains(text, "override wgsl_loop_budget: u32 = 1000u;") {
		t.Fatal(text)
	}
	if err := wgsl.Validate(text); err != nil {
		t.Fatal(err)
	}
}

func TestNoLoopsUnchanged(t *testing.T) {
	in := parse(t, "@compute @workgroup_size(1) fn main() { var a = 1; a++; }")
	out, rep, err := LoopBudget(in, Config{})
	if err != nil || rep.Instrumented || !wgsl.Equal(in, out) {
		t.Fatalf("%v %+v", err, rep)
	}
}

func TestNameCollision(t *testing.T) {
	src := "@compute @workgroup_size(1) fn main() { var wgsl_loop_budget = 1u; var wgsl_truncated = 0u; var wgsl_loop_count = 0u; while wgsl_loop_budget > 0u { wgsl_loop_budget--; } }"
	out, rep, err := LoopBudget(parse(t, src), Config{FlagBinding: 1})
	if err != nil {
		t.Fatal(err)
	}
	if rep.BudgetName == "wgsl_loop_budget" || rep.FlagName == "wgsl_truncated" {
		t.Fatalf("%+v", rep)
	}
	if err := wgsl.Validate(wgsl.Print(out)); err != nil {
		t.Fatal(err)
	}
}

// Parse-level only: a shader the checker rejects is still transformed.
func TestParseLevelOnly(t *testing.T) {
	in := parse(t, "fn f() { let x: i32 = 1.5; while true { } nosuch(); }")
	out, rep, err := LoopBudget(in, Config{})
	if err != nil || rep.Loops != 1 {
		t.Fatalf("%v %+v", err, rep)
	}
	parse(t, wgsl.Print(out))
}

// Every real shader that validates still validates after the transform.
func TestRealShadersStillValidate(t *testing.T) {
	files, _ := filepath.Glob("../testdata/real/*/*.wgsl")
	n, withLoops := 0, 0
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if wgsl.Validate(string(b)) != nil {
			continue
		}
		n++
		for _, cfg := range []Config{
			{Budget: BudgetSource{Group: 2, Binding: 0}, FlagGroup: 2, FlagBinding: 1},
			{Budget: BudgetSource{Override: true, Default: 4096}, FlagGroup: 2, FlagBinding: 1},
		} {
			out, rep, err := LoopBudget(parse(t, string(b)), cfg)
			if err != nil {
				t.Errorf("%s: %v", f, err)
				continue
			}
			text := wgsl.Print(out)
			if err := wgsl.Validate(text); err != nil {
				t.Errorf("%s: transformed shader invalid: %v", f, err)
			}
			if rep.Instrumented && cfg.Budget.Override {
				withLoops++
			}
		}
	}
	if n < 20 {
		t.Fatalf("only %d shaders", n)
	}
	t.Logf("%d valid real shaders, %d with loops: transformed output validates", n, withLoops)
}

// An inner loop whose trip count differs between invocations must not make the
// outer loop's check non-uniform (the browser would then reject the
// workgroupBarrier in the outer loop): each depth has its own counter, and the
// outer check reads only the outer one.
func TestLoopBudgetCounterPerDepth(t *testing.T) {
	src := `
var<workgroup> tile: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) {
    for (var t = 0u; t < 4u; t++) {
        var i = li;
        loop { if (i >= 100u) { break; } tile[i % 64u] = 1.0; i += 64u; }
        workgroupBarrier();
    }
}
`
	out, rep, err := LoopBudget(parse(t, src), Config{Budget: BudgetSource{Group: 1, Binding: 0}, FlagGroup: 1, FlagBinding: 1})
	if err != nil || rep.Counters != 2 {
		t.Fatalf("%v %+v", err, rep)
	}
	text := wgsl.Print(out)
	// the outer loop counts with wgsl_loop_count, the inner with the next name
	outer := strings.Index(text, "wgsl_loop_count += 1u;")
	inner := strings.Index(text, "wgsl_loop_count_0 += 1u;")
	if outer < 0 || inner < 0 || outer > inner {
		t.Errorf("counters per depth missing in\n%s", text)
	}
	if strings.Count(text, "wgsl_loop_count_0 +=") != 1 || strings.Count(text, "wgsl_loop_count +=") != 1 {
		t.Errorf("each depth counts once per loop of the depth\n%s", text)
	}
}

// A leaf loop with a small constant trip count is not checked per iteration:
// the enclosing counter is advanced by the trip count once, so the bound stays
// exact; anything else is checked as before.
func TestLoopBudgetSmallLeaf(t *testing.T) {
	cfg := Config{Budget: BudgetSource{Group: 1, Binding: 0}, FlagGroup: 1, FlagBinding: 1, SmallLoop: 16}
	src := `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    var n = 0u;
    for (var a = 0u; a < 4u; a++) { n++; }
    for (var i = 0u; i < 100u; i++) {
        for (var j = 0u; j < 4u; j++) { n++; }
        for (var k = 1u; k <= 7u; k += 2u) { n++; }
        for (var m = 0u; m < 64u; m++) { n++; }
        for (var p = 0u; p < 4u; p++) { p += 1u; }
        for (var q = 0u; q < n; q++) { n++; }
        for (var r = 0u; r < 4u; r++) { for (var s = 0u; s < 2u; s++) { n++; } }
        for (var w = 0u; w < 4u; w = w + 1u) { out[w] = 1u; }
    }
}
`
	out, rep, err := LoopBudget(parse(t, src), cfg)
	if err != nil {
		t.Fatal(err)
	}
	// skipped: a (outside any loop), j, k, w, and s (a leaf inside r); checked:
	// i, m (trip 64), p (assigns its variable), q (not constant) and r (not a leaf)
	text := wgsl.Print(out)
	if rep.Skipped != 5 || rep.Loops != 5 {
		t.Errorf("%+v\n%s", rep, text)
	}
	for _, want := range []string{"wgsl_loop_count += 4u;", "for (var j = 0u; j < 4u; j++)", "for (var a = 0u; a < 4u; a++)"} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q in\n%s", want, text)
		}
	}
	if strings.Contains(text, "var wgsl_loop_count: u32 = 0u;\n    for (var a") {
		t.Errorf("a leaf loop outside every loop got a counter\n%s", text)
	}
	if err := wgsl.Validate(text); err != nil {
		t.Fatalf("%v\n%s", err, text)
	}
	// off by default
	_, rep, _ = LoopBudget(parse(t, src), Config{Budget: BudgetSource{Group: 1, Binding: 0}, FlagGroup: 1, FlagBinding: 1})
	if rep.Skipped != 0 {
		t.Errorf("SmallLoop 0 skipped %d", rep.Skipped)
	}
}
