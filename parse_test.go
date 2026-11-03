package wgsl

import (
	"strings"
	"testing"
)

const sample = `
enable f16;
diagnostic(off, derivative_uniformity);
struct S { @align(16) a: vec4<f32>, b: array<array<f32, 2>, 2>, }
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> u: S;
override n: u32 = 4u;
const k = 1.5e3 + 0x1p3 * 2.;
alias V = vec3<f32>;
fn helper(a: f32, @builtin(position) p: vec4<f32>) -> f32 { return a * 2.0; }
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var i: u32 = 0u;
    let v = array<f32, 4>(1., 2., 3., 4.);
    loop {
        if i >= n { break; }
        buf[i] += f32(i) * k;
        continuing { i++; break if i > 100u; }
    }
    for (var j = 0; j < 4; j++) { _ = j; }
    while (i < 3u && n > 2u) || false { i = i << 1u; }
    switch i { case 1, 2: { } case 3, default { } default: { } }
    let t = (a < b) == (c > d);
    let p = &buf;
    (*p)[0] = -(-1.0);
    const_assert 1 < 2;
}
`

func TestRoundTrip(t *testing.T) {
	m, err := Parse(sample)
	if err != nil {
		t.Fatal(err)
	}
	out := Print(m)
	m2, err := Parse(out)
	if err != nil {
		t.Fatalf("reparse: %v\n%s", err, out)
	}
	if !Equal(m, m2) {
		t.Fatalf("round trip differs:\n%s", out)
	}
	if Print(m2) != out {
		t.Fatal("printer is not idempotent")
	}
}

func TestReservedWord(t *testing.T) {
	_, err := Parse("fn f() { let meta = 1; }")
	e, ok := err.(*Error)
	if !ok || e.Rule != "reserved-word" || e.Pos.Line != 1 || e.Pos.Col != 14 {
		t.Fatalf("got %#v", err)
	}
	if _, err := Parse("var<private> __x: i32;"); err == nil {
		t.Fatal("double underscore accepted")
	}
}

func TestTemplateDisambiguation(t *testing.T) {
	if _, err := Parse("fn f() { var a: array<vec4<f32>, 2>; let x = a[0].x < a[1].x; let y = 1 >> 2; }"); err != nil {
		t.Fatal(err)
	}
	if _, err := Parse("var<private> z: array<array<f32,2>,2>=array<array<f32,2>,2>();"); err != nil {
		t.Fatal(err)
	}
}

func TestSyntaxRejections(t *testing.T) {
	for _, src := range []string{
		"fn f() { let x = a & b | c; }",
		"fn f() { let x = a < b < c; }",
		"fn f() { let x = a && b || c; }",
		"fn f() { let x = a + b << c; }",
		"fn f() { let x = 01; }",
		"fn f() { let x = 1 }",
		"fn f() { /* unterminated }",
		"@workgroup_size() fn f() {}",
		"fn f() { break if true; }",
		"const x = 1;; fn",
	} {
		if _, err := Parse(src); err == nil {
			t.Errorf("accepted %q", src)
		}
	}
}

func TestDeepNestingRefused(t *testing.T) {
	for name, src := range map[string]string{
		"parens":  "fn f() { let x = " + strings.Repeat("(", 100000) + "1" + strings.Repeat(")", 100000) + "; }",
		"unary":   "fn f() { let x = " + strings.Repeat("- ", 100000) + "1; }",
		"blocks":  "fn f() " + strings.Repeat("{", 100000) + strings.Repeat("}", 100000),
		"index":   "fn f() { let x = a" + strings.Repeat("[0]", 100000) + "; }",
		"chain":   "fn f() { let x = 1" + strings.Repeat(" + 1", 100000) + "; }",
		"else-if": "fn f() { if a {} " + strings.Repeat("else if a {} ", 100000) + "}",
		"types":   "var<private> x: " + strings.Repeat("array<", 100000) + "f32" + strings.Repeat(">", 100000) + ";",
	} {
		_, err := Parse(src, Options{Limits: Limits{MaxTokens: 1 << 22, MaxSource: 1 << 24}})
		e, ok := err.(*Error)
		if !ok || e.Rule != "limit-depth" {
			t.Errorf("%s: got %v", name, err)
		}
	}
}

func TestLimits(t *testing.T) {
	if _, err := Parse(strings.Repeat(";", 100), Options{Limits: Limits{MaxSource: 50}}); err == nil {
		t.Error("source limit not enforced")
	}
	if _, err := Parse(strings.Repeat("a ", 100), Options{Limits: Limits{MaxTokens: 10}}); err == nil {
		t.Error("token limit not enforced")
	}
}
