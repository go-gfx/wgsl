package msl

import (
	"errors"
	"strings"
	"testing"
)

func TestKernelEmits(t *testing.T) {
	m, info := check(t, kernelSrc)
	out, err := Module(m, info, "raster")
	if err != nil {
		t.Fatal(err)
	}
	t.Log(out[len(Preamble):])
	for _, want := range []string{
		"static uint byteAt(uint i, device const uint *arena)",
		"kernel void raster(device const Inst *insts [[buffer(0)]],",
		"device const uint *bins [[buffer(1)]]",
		"device const uint *arena [[buffer(2)]]",
		"device uint *px [[buffer(3)]]",
		"constant Frame &f [[buffer(4)]]",
		"uint3 gid [[thread_position_in_grid]]",
		"for (uint k = 0u; (k < count); k++)",
		"wgsl_f2u(",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("no %q in the emitted MSL", want)
		}
	}
}

// What a compute module may not do is refused, with a position.
func TestModuleRefusals(t *testing.T) {
	for _, tc := range []struct{ name, src, entry, want string }{
		{"vertex entry", "@vertex fn v() -> @builtin(position) vec4f { return vec4f(0.0); }", "v", "only compute"},
		{"not an entry", "fn f() {}", "f", "not an entry point"},
		{"no such function", "fn f() {}", "nosuch", "no function"},
		{"group 1", "@group(1) @binding(0) var<storage, read> a: array<u32>;\n@compute @workgroup_size(1) fn run() { _ = a[0]; }", "run", "group"},
		{"private variable", "var<private> p: u32;\n@compute @workgroup_size(1) fn run() { p = 1u; }", "run", "private"},
		{"workgroup variable", "var<workgroup> w: u32;\n@compute @workgroup_size(1) fn run() { w = 1u; }", "run", "workgroup"},
		{"fixed array", "@group(0) @binding(0) var<storage, read_write> a: array<u32, 4>;\n@compute @workgroup_size(1) fn run() { a[0] = 1u; }", "run", "fixed-size array"},
		{"uniform array", "@group(0) @binding(0) var<uniform> a: array<vec4u, 4>;\n@compute @workgroup_size(1) fn run() { _ = a[0]; }", "run", "uniform array"},
		{"local array", "@compute @workgroup_size(1) fn run() { var a: array<u32, 4>; a[0] = 1u; }", "run", "not supported"},
		{"vec3 layout", "struct V { a: vec3f, b: f32 }\n@group(0) @binding(0) var<storage, read_write> o: array<V>;\n@compute @workgroup_size(1) fn run() { o[0].b = 1.0; }", "run", "offset"},
		{"nested struct", "struct N { a: u32 }\nstruct V { n: N }\n@group(0) @binding(0) var<storage, read_write> o: array<V>;\n@compute @workgroup_size(1) fn run() { o[0].n.a = 1u; }", "run", "only scalars and vectors"},
		{"local_invocation_id", "@compute @workgroup_size(1) fn run(@builtin(local_invocation_id) l: vec3u) { _ = l; }", "run", "@builtin"},
		{"loop continuing", "@compute @workgroup_size(1) fn run() { var i = 0u; loop { if (i > 3u) { break; } continuing { i++; } } }", "run", "continuing"},
		{"switch", "@compute @workgroup_size(1) fn run() { var i = 0u; switch (i) { default { i = 1u; } } }", "run", "not supported"},
		{"f32 vector to u32 vector", "@compute @workgroup_size(1) fn run() { var v = vec2u(vec2f(1.5)); v.x = 1u; }", "run", "vector"},
		{"binding 31", "@group(0) @binding(31) var<storage, read_write> a: array<u32>;\n@compute @workgroup_size(1) fn run() { a[0] = 1u; }", "run", "31 buffer"},
	} {
		m, info := check(t, tc.src)
		_, err := Module(m, info, tc.entry)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want an error naming %q", tc.name, err, tc.want)
			continue
		}
		var e *Error
		if !errors.As(err, &e) && tc.name != "no such function" {
			t.Errorf("%s: %T is not an *Error with a position", tc.name, err)
		}
	}
}
