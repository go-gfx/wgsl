package msl

import (
	"errors"
	"strings"
	"testing"

	"github.com/go-gfx/wgsl"
)

// coverageSrc is go-gfx/gpu2d's coverage arithmetic, as its WebGPU shader
// writes it (webgpu/shader.wgsl): the case this emitter exists for.
const coverageSrc = `
struct Inst {
  x: i32, y: i32, w: i32, h: i32,
  cx0: i32, cy0: i32, cx1: i32, cy1: i32,
  color: u32, radius: i32, kind: u32,
  data: u32, stride: u32, srcW: u32, srcH: u32,
  pad: u32,
}

fn cornerCenter(x: i32, y: i32, s: Inst, rad: i32) -> vec3f {
  let left = x < s.x + rad;
  let right = x >= s.x + s.w - rad;
  let top = y < s.y + rad;
  let bottom = y >= s.y + s.h - rad;
  if (left && top) { return vec3f(f32(s.x + rad), f32(s.y + rad), 1.0); }
  if (right && top) { return vec3f(f32(s.x + s.w - rad), f32(s.y + rad), 1.0); }
  if (left && bottom) { return vec3f(f32(s.x + rad), f32(s.y + s.h - rad), 1.0); }
  if (right && bottom) { return vec3f(f32(s.x + s.w - rad), f32(s.y + s.h - rad), 1.0); }
  return vec3f(0.0);
}

fn coverage(x: i32, y: i32, s: Inst) -> f32 {
  let rad = s.radius;
  let c = cornerCenter(x, y, s, rad);
  let d = length(vec2f(f32(x) + 0.5 - c.x, f32(y) + 0.5 - c.y));
  if (s.kind == 0u) {
    if (c.z == 0.0) { return 1.0; }
    return clamp(f32(rad) + 0.5 - d, 0.0, 1.0);
  }
  if (rad > 0 && c.z != 0.0) {
    return max(1.0 - abs(d - (f32(rad) - 0.5)), 0.0);
  }
  let edge = y == s.y || y == s.y + s.h - 1 || x == s.x || x == s.x + s.w - 1;
  return select(0.0, 1.0, edge);
}
`

func check(t *testing.T, src string) (*wgsl.Module, *wgsl.Info) {
	t.Helper()
	m, err := wgsl.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	return m, info
}

func TestCoverageEmits(t *testing.T) {
	m, info := check(t, coverageSrc)
	out, err := Functions(m, info, "coverage")
	if err != nil {
		t.Fatal(err)
	}
	t.Log(out[len(Preamble):])
	for _, want := range []string{
		"struct Inst {", "static float3 cornerCenter(int x, int y, Inst s, int rad)",
		"static float coverage(int x, int y, Inst s)", "wgsl_add(s.x, rad)", "wgsl_length(float2(",
		"precise::fmin(precise::fmax(", "select(0.0f, 1.0f, edge)",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("no %q in the emitted MSL", want)
		}
	}
}

// Out of scope is refused, with a position, never emitted approximately.
func TestRefusals(t *testing.T) {
	for _, tc := range []struct{ name, src, fn, want string }{
		{"module-scope variable", "var<private> g: i32; fn f() -> i32 { return g; }", "f", "module-scope"},
		{"switch", "fn f() -> i32 { var i = 0; switch (i) { default { i = 1; } } return i; }", "f", "not supported"},
		{"entry point", "@compute @workgroup_size(1) fn main() {}", "main", "entry point"},
		{"f32 vector to i32 vector", "fn f(x: vec2f) -> vec2i { return vec2i(x); }", "f", "vector"},
		{"reserved name", "fn kernel() -> i32 { return 1; }", "kernel", "reserved"},
		{"array", "fn f(a: array<i32, 4>) -> i32 { return 0; }", "f", "not supported"},
	} {
		m, info := check(t, tc.src)
		_, err := Functions(m, info, tc.fn)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want an error naming %q", tc.name, err, tc.want)
		}
		var e *Error
		if err != nil && strings.Contains(tc.want, "") && !errors.As(err, &e) && tc.name != "entry point" {
			t.Errorf("%s: %T is not an *Error with a position", tc.name, err)
		}
	}
	m, info := check(t, coverageSrc)
	if _, err := Functions(m, info, "nosuch"); err == nil {
		t.Error("a missing function was emitted")
	}
}
