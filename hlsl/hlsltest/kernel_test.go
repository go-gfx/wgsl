//go:build windows

package hlsltest

import (
	"os"
	"strings"
	"testing"

	"github.com/go-mswin/d3d11/d3dcompiler"
)

// gpu2d's whole raster kernel, emitted and compiled by D3DCompile for cs_5_0
// with IEEE strictness and warnings as errors. Running it against Rasterize is
// gpu2d's job (milestone D3); here it must at least compile, and use no mad.
func TestKernelCompiles(t *testing.T) {
	src, err := os.ReadFile("../testdata/kernel.wgsl")
	if err != nil {
		t.Fatal(err)
	}
	text := emitHLSL(t, string(src), "raster")
	dxbc := compile(t, text, "raster")
	dis := disassemble(t, dxbc)
	if !strings.Contains(dis, "dcl_thread_group 16, 8, 1") {
		t.Errorf("the compiled kernel is not a 16 x 8 thread group:\n%s", dis)
	}
	if hasMad(dis) {
		t.Errorf("a mad (fused multiply-add) was emitted for the kernel, which WGSL's arithmetic does not allow:\n%s", dis)
	}
	t.Logf("kernel: %d bytes of DXBC", len(dxbc))
}

func hasMad(dis string) bool {
	for _, l := range strings.Split(dis, "\n") {
		f := strings.Fields(l)
		if len(f) > 0 && (f[0] == "mad" || f[0] == "mad_sat") {
			return true
		}
	}
	return false
}

const madSrc = `
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  let a = inp[3u * g.x];
  let b = inp[3u * g.x + 1u];
  let c = inp[3u * g.x + 2u];
  outp[g.x] = a * b + c;
}
`

// The break check of rule 1. Compiled as the emitter writes it, a * b + c is a
// mul and an add; with the precise pins removed (wgsl_pf an identity and the
// variables not precise) the same compiler fuses them into a mad. A check that
// cannot fail proves nothing: this one shows the pins are what prevents it.
func TestPreciseKeepsMulAndAddApart(t *testing.T) {
	text := emitHLSL(t, madSrc, "run")
	// Without D3DCOMPILE_IEEE_STRICTNESS: the pins alone must keep them apart.
	const optimised = d3dcompiler.FlagOptimizationLevel3
	good := disassemble(t, compileWith(t, text, "run", optimised))
	if hasMad(good) {
		t.Fatalf("the emitted a*b+c was fused into a mad:\n%s", good)
	}
	if !strings.Contains(good, "mul ") || !strings.Contains(good, "add ") {
		t.Fatalf("no mul and add in:\n%s", good)
	}
	mutant := strings.ReplaceAll(text, "precise float r = x; return r;", "return x;")
	mutant = strings.ReplaceAll(mutant, "precise float", "float")
	if mutant == text {
		t.Fatal("the mutation changed nothing")
	}
	bad := disassemble(t, compileWith(t, mutant, "run", optimised))
	if !hasMad(bad) {
		t.Fatalf("without the pins the compiler did not fuse a*b+c, so the check above proves nothing:\n%s", bad)
	}
}

// The break check of the integer rules: an unguarded divisor must be caught by
// the edge cases. DXBC's udiv by zero is 0xFFFFFFFF; WGSL says the dividend.
func TestUnguardedDivisionIsCaught(t *testing.T) {
	const src = `
@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> outp: array<u32>;
@compute @workgroup_size(8)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= 4u) { return; }
  outp[g.x] = inp[2u * g.x] / inp[2u * g.x + 1u];
}
`
	in := u32s(7, 0, 100, 3, 0, 0, 0xFFFFFFFF, 0)
	want := []uint32{7, 33, 0, 0xFFFFFFFF}
	text := emitHLSL(t, src, "run")
	got := runHLSL(t, text, "run", []buf{storage(in), storage(make([]byte, 16))}, [3]uint32{1, 1, 1})
	for i, w := range want {
		if g := word(got[1], i); g != w {
			t.Errorf("%d / %d = %d, WGSL says %d", word(in, 2*i), word(in, 2*i+1), g, w)
		}
	}
	mutant := strings.Replace(text, "return a / (b == 0u ? 1u : b);", "return a / b;", 1)
	if mutant == text {
		t.Fatal("the mutation changed nothing")
	}
	bad := runHLSL(t, mutant, "run", []buf{storage(in), storage(make([]byte, 16))}, [3]uint32{1, 1, 1})
	differs := false
	for i, w := range want {
		differs = differs || word(bad[1], i) != w
	}
	if !differs {
		t.Fatal("an unguarded division gave the same answers: the edge cases cannot tell")
	}
}
