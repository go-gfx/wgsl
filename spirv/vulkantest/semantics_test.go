package vulkantest

import (
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	"github.com/go-gfx/wgsl"
)

const intsSrc = `
@group(0) @binding(0) var<storage, read> inp: array<i32>;
@group(0) @binding(1) var<storage, read_write> outp: array<i32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;

fn add(a: i32, b: i32) -> i32 { return a + b; }
fn sub(a: i32, b: i32) -> i32 { return a - b; }
fn mul(a: i32, b: i32) -> i32 { return a * b; }
fn div(a: i32, b: i32) -> i32 { return a / b; }
fn rem(a: i32, b: i32) -> i32 { return a % b; }
fn neg(a: i32) -> i32 { return -a; }
fn shl(a: i32, b: i32) -> i32 { return a << u32(b); }
fn shr(a: i32, b: i32) -> i32 { return a >> u32(b); }
fn udiv(a: i32, b: i32) -> i32 { return bitcast<i32>(u32(a) / u32(b)); }
fn urem(a: i32, b: i32) -> i32 { return bitcast<i32>(u32(a) % u32(b)); }
fn ushl(a: i32, b: i32) -> i32 { return bitcast<i32>(u32(a) << u32(b)); }
fn ushr(a: i32, b: i32) -> i32 { return bitcast<i32>(u32(a) >> u32(b)); }
fn compound(a: i32, b: i32) -> i32 {
  var x = a;
  x += b;
  x *= 3;
  if (x > 0) { x -= 1; } else if (x < 0) { x = -x; } else { x = 7; }
  return x;
}
fn absi(a: i32) -> i32 { return abs(a); }
fn mini(a: i32, b: i32) -> i32 { return min(a, b); }
fn maxi(a: i32, b: i32) -> i32 { return max(a, b); }
fn clampi(a: i32, b: i32) -> i32 { return clamp(a, -b, b); }

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.x) { return; }
  let a = inp[2u * g.x];
  let b = inp[2u * g.x + 1u];
  let o = g.x * 17u;
  outp[o + 0u] = add(a, b);   outp[o + 1u] = sub(a, b);   outp[o + 2u] = mul(a, b);
  outp[o + 3u] = div(a, b);   outp[o + 4u] = rem(a, b);   outp[o + 5u] = neg(a);
  outp[o + 6u] = shl(a, b);   outp[o + 7u] = shr(a, b);   outp[o + 8u] = udiv(a, b);
  outp[o + 9u] = urem(a, b);  outp[o + 10u] = ushl(a, b); outp[o + 11u] = ushr(a, b);
  outp[o + 12u] = compound(a, b);
  outp[o + 13u] = absi(a);    outp[o + 14u] = mini(a, b); outp[o + 15u] = maxi(a, b);
  outp[o + 16u] = clampi(a, b);
}
`

// WGSL's definitions (gpuweb wgsl/index.bs; docs/MSL.md rules 4 and 5), in Go.
var intRef = []func(a, b int32) int32{
	func(a, b int32) int32 { return int32(uint32(a) + uint32(b)) },
	func(a, b int32) int32 { return int32(uint32(a) - uint32(b)) },
	func(a, b int32) int32 { return int32(uint32(a) * uint32(b)) },
	func(a, b int32) int32 {
		if b == 0 || (a == math.MinInt32 && b == -1) {
			return a
		}
		return a / b
	},
	func(a, b int32) int32 {
		if b == 0 || (a == math.MinInt32 && b == -1) {
			return 0
		}
		return a % b
	},
	func(a, _ int32) int32 { return int32(0 - uint32(a)) },
	func(a, b int32) int32 { return int32(uint32(a) << (uint32(b) & 31)) },
	func(a, b int32) int32 { return a >> (uint32(b) & 31) },
	func(a, b int32) int32 {
		if b == 0 {
			return a
		}
		return int32(uint32(a) / uint32(b))
	},
	func(a, b int32) int32 {
		if b == 0 {
			return 0
		}
		return int32(uint32(a) % uint32(b))
	},
	func(a, b int32) int32 { return int32(uint32(a) << (uint32(b) & 31)) },
	func(a, b int32) int32 { return int32(uint32(a) >> (uint32(b) & 31)) },
	func(a, b int32) int32 {
		x := int32(uint32(a) + uint32(b))
		x = int32(uint32(x) * 3)
		switch {
		case x > 0:
			x = int32(uint32(x) - 1)
		case x < 0:
			x = int32(0 - uint32(x))
		default:
			x = 7
		}
		return x
	},
	func(a, _ int32) int32 { // abs(i32::MIN) is i32::MIN
		if a < 0 {
			return int32(0 - uint32(a))
		}
		return a
	},
	func(a, b int32) int32 { return min(a, b) },
	func(a, b int32) int32 { return max(a, b) },
	func(a, b int32) int32 { // clamp(a, -b, b) = min(max(a, -b), b), whatever the order of -b and b
		return min(max(a, int32(0-uint32(b))), b)
	},
}

var intNames = []string{"add", "sub", "mul", "div", "rem", "neg", "shl", "shr", "udiv", "urem", "ushl", "ushr", "compound", "abs", "min", "max", "clamp"}

func TestIntegerSemantics(t *testing.T) {
	vals := []int32{0, 1, -1, 2, -2, 3, -7, 7, 31, 32, 33, 64, -32, 1 << 30, math.MaxInt32, math.MinInt32, math.MinInt32 + 1, 123456789, -987654321}
	var pairs []int32
	for _, a := range vals {
		for _, b := range vals {
			pairs = append(pairs, a, b)
		}
	}
	n := len(pairs) / 2
	k := len(intNames)
	out := runModule(t, intsSrc, "run", []buf{storage(i32s(pairs...)), storage(make([]byte, 4*k*n)), uniform(u32s(uint32(n), 0, 0, 0))}, groupsFor(n, 64))
	for i := range n {
		a, b := pairs[2*i], pairs[2*i+1]
		for j, ref := range intRef {
			if want, got := ref(a, b), int32(word(out[1], i*k+j)); got != want {
				t.Errorf("%s(%d, %d) = %d on the GPU, WGSL says %d", intNames[j], a, b, got, want)
			}
		}
	}
}

const floatsSrc = `
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;

fn clampf(x: f32, lo: f32, hi: f32) -> f32 { return clamp(x, lo, hi); }
fn len2(a: f32, b: f32) -> f32 { return length(vec2f(a, b)); }
fn len3(a: f32, b: f32, c: f32) -> f32 { return length(vec3f(a, b, c)); }
fn pick(a: f32, b: f32, c: f32) -> f32 { return select(a, b, c > 0.0); }
fn absf(a: f32) -> f32 { return abs(a); }
fn minmax(a: f32, b: f32) -> f32 { return max(min(a, b), 0.25); }
fn sqrtf(a: f32) -> f32 { return sqrt(a); }
fn floorf(a: f32) -> f32 { return floor(a); }

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.x) { return; }
  let a = inp[2u * g.x];
  let b = inp[2u * g.x + 1u];
  let o = g.x * 9u;
  outp[o + 0u] = clampf(a, b, 1.5);
  outp[o + 1u] = clampf(a, 1.5, b);
  outp[o + 2u] = len2(a, b);
  outp[o + 3u] = len3(a, b, a);
  outp[o + 4u] = pick(a, b, a - b);
  outp[o + 5u] = absf(a);
  outp[o + 6u] = minmax(a, b);
  outp[o + 7u] = sqrtf(abs(a));
  outp[o + 8u] = floorf(a);
}
`

func ftz(x float32) float32 { // WGSL lets a subnormal input or output be flushed to zero
	if x != 0 && math.Abs(float64(x)) < 0x1p-126 {
		return 0
	}
	return x
}

func min32(a, b float32) float32 {
	if b < a {
		return b
	}
	return a
}

func max32(a, b float32) float32 {
	if b > a {
		return b
	}
	return a
}

func TestFloatSemantics(t *testing.T) {
	inf := float32(math.Inf(1))
	vals := []float32{0, float32(math.Copysign(0, -1)), 1, -1, 0.5, 3, -4, 1e-30, 1e30, 3.4e38, 0.1, 2.5, -7.25, 1e-45, inf, -inf}
	var pairs []float32
	for _, a := range vals {
		for _, b := range vals {
			pairs = append(pairs, a, b)
		}
	}
	n := len(pairs) / 2
	const k = 9
	out := runModule(t, floatsSrc, "run", []buf{storage(f32bits(pairs...)), storage(make([]byte, 4*k*n)), uniform(u32s(uint32(n), 0, 0, 0))}, groupsFor(n, 64))
	sum := func(xs ...float32) float32 { // left to right, each step rounded to f32
		var s float32
		for _, x := range xs {
			s = float32(s + float32(x*x))
		}
		return s
	}
	sqrt := func(x float32) float32 { return float32(math.Sqrt(float64(x))) }
	clamp := func(x, lo, hi float32) float32 { return min32(max32(x, lo), hi) }
	expect := func(a, b float32) []float32 {
		pick := a
		if a-b > 0 {
			pick = b
		}
		return []float32{clamp(a, b, 1.5), clamp(a, 1.5, b), sqrt(sum(a, b)), sqrt(sum(a, b, a)), pick,
			float32(math.Abs(float64(a))), max32(min32(a, b), 0.25), sqrt(float32(math.Abs(float64(a)))), float32(math.Floor(float64(a)))}
	}
	same := func(g, w float32) bool {
		return math.Float32bits(g) == math.Float32bits(w) || (g == 0 && w == 0) || (g != g && w != w)
	}
	// The length of a vector that holds an infinity or whose squares overflow:
	// inf, as the sum of squares is. The rounding of sqrt itself is measured in
	// TestSqrtAgainstTheReference, and is not held to exactness here.
	for i := range n {
		a, b := pairs[2*i], pairs[2*i+1]
		exact := expect(a, b)
		flushed := expect(ftz(a), ftz(b))
		for j := range exact {
			g := math.Float32frombits(word(out[1], i*k+j))
			if j == 4 {
				c := float32(a - b)
				ok := false
				for _, cond := range []bool{c > 0, ftz(c) > 0} {
					v := a
					if cond {
						v = b
					}
					ok = ok || same(g, v) || same(g, ftz(v))
				}
				if !ok {
					t.Errorf("select(%g, %g, %g > 0) = %g on the GPU: none of WGSL's results", a, b, c, g)
				}
				continue
			}
			if j == 2 || j == 3 || j == 7 {
				// sqrt: within 1 ULP of the correctly rounded value is enough here
				// (the exact count is TestSqrtAgainstTheReference's); a wrong function is far off.
				if !within(g, exact[j], 1) && !within(g, flushed[j], 1) {
					t.Errorf("function %d (%g, %g) = %g on the GPU, WGSL gives %g", j, a, b, g, exact[j])
				}
				continue
			}
			if !same(g, exact[j]) && !same(g, ftz(flushed[j])) {
				t.Errorf("function %d (%g, %g) = %g on the GPU, WGSL gives %g (or %g flushed to zero)", j, a, b, g, exact[j], ftz(flushed[j]))
			}
		}
	}
}

// within reports whether g is at most ulps units in the last place from w.
func within(g, w float32, ulps int64) bool {
	if g == w || (g != g && w != w) {
		return true
	}
	if g != g || w != w || math.IsInf(float64(g), 0) != math.IsInf(float64(w), 0) {
		return false
	}
	d := int64(int32(math.Float32bits(g))) - int64(int32(math.Float32bits(w)))
	if (g < 0) != (w < 0) {
		return g == 0 && w == 0
	}
	return d >= -ulps && d <= ulps
}

const convSrc = `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;

@compute @workgroup_size(32)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.x) { return; }
  o[2u * g.x] = u32(x[g.x]);
  o[2u * g.x + 1u] = bitcast<u32>(i32(x[g.x]));
}
`

// f32 to u32 and i32: truncate, then the nearest value of the type that f32
// can also represent (wgsl/index.bs, "Floating Point Conversion").
func TestFloatToInteger(t *testing.T) {
	inf := float32(math.Inf(1))
	vals := []float32{0, float32(math.Copysign(0, -1)), 0.5, 0.9999, -0.9, 1, 1.5, -1.5, 2.5, 255.5, 256, 65535.75, 16777216, 1e9,
		2147483520, 2147483648, 3e9, 4294967040, 4294967296, 1e10, 3.4e38, inf, -1, -2147483648, -2147483904, -3e9, -1e10, -inf, 1e-45, -1e-45,
		float32(math.NaN())}
	cnt := uint32(len(vals))
	out := runModule(t, convSrc, "run", []buf{storage(f32bits(vals...)), storage(make([]byte, 8*len(vals))), uniform(u32s(cnt, 0, 0, 0))}, groupsFor(len(vals), 32))
	for i, v := range vals {
		var u, s float64
		if v != v {
			// A NaN gives an indeterminate value in WGSL; this emitter's rule is 0.
			u, s = 0, 0
		} else {
			x := math.Trunc(float64(v))
			u = math.Min(math.Max(x, 0), 4294967040)
			s = math.Min(math.Max(x, -2147483648), 2147483520)
		}
		if g := word(out[1], 2*i); g != uint32(u) {
			t.Errorf("u32(%g) = %d on the GPU, WGSL says %d", v, g, uint32(u))
		}
		if g := int32(word(out[1], 2*i+1)); g != int32(s) {
			t.Errorf("i32(%g) = %d on the GPU, WGSL says %d", v, g, int32(s))
		}
	}
	// The checker's constant evaluator gives the same answers as the GPU (it once
	// clamped an f32 at 2^32-1 and 2^31-1, which f32 cannot hold).
	for i, v := range vals {
		if math.IsInf(float64(v), 0) || v != v {
			continue // not const-expressions
		}
		lit := strconv.FormatFloat(float64(v), 'g', -1, 64)
		if !strings.ContainsAny(lit, ".eE") {
			lit += ".0"
		}
		src := "const u = u32(f32(" + lit + ")); const s = i32(f32(" + lit + "));"
		m, err := wgsl.Parse(src)
		if err != nil {
			t.Fatal(err)
		}
		info, errs := wgsl.Check(m)
		if len(errs) > 0 {
			t.Fatalf("%s: %v", src, errs)
		}
		got := map[string]int64{}
		for _, d := range m.Decls {
			if val, ok := d.(*wgsl.Value); ok {
				got[val.Name] = info.Decls[val].(*wgsl.ValueObj).Val.Int
			}
		}
		if g := word(out[1], 2*i); int64(g) != got["u"] {
			t.Errorf("const u32(f32(%s)) = %d, the GPU gives %d", lit, got["u"], g)
		}
		if g := int32(word(out[1], 2*i+1)); int64(g) != got["s"] {
			t.Errorf("const i32(f32(%s)) = %d, the GPU gives %d", lit, got["s"], g)
		}
	}
}

const nanSrc = `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;

fn b2u(c: bool) -> u32 { return select(0u, 1u, c); }

@compute @workgroup_size(8)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= 8u) { return; }
  let v = x[g.x];
  o[g.x * 6u + 0u] = b2u(v != v);
  o[g.x * 6u + 1u] = b2u(v == v);
  o[g.x * 6u + 2u] = b2u(v < 1.0);
  o[g.x * 6u + 3u] = b2u(v >= 1.0);
  o[g.x * 6u + 4u] = b2u(v > 0.0);
  o[g.x * 6u + 5u] = b2u(v <= 0.0);
}
`

// NaN, +-0 and +-INF in comparisons (IEEE 754, which WGSL follows).
func TestComparisonsOfNaNZerosInfinities(t *testing.T) {
	nan := float32(math.NaN())
	negZero := float32(math.Copysign(0, -1))
	inf := float32(math.Inf(1))
	vals := []float32{nan, 0, negZero, inf, -inf, 1, 0.5, 1e-45}
	out := runModule(t, nanSrc, "run", []buf{storage(f32bits(vals...)), storage(make([]byte, 4*6*len(vals)))}, [3]uint32{1, 1, 1})
	b2u := func(c bool) uint32 {
		if c {
			return 1
		}
		return 0
	}
	for i, v := range vals {
		want := []uint32{b2u(v != v), b2u(v == v), b2u(v < 1), b2u(v >= 1), b2u(v > 0), b2u(v <= 0)}
		if v == 1e-45 {
			continue // a subnormal may be flushed: > 0 and <= 0 may go either way
		}
		for j, w := range want {
			if g := word(out[1], 6*i+j); g != w {
				t.Errorf("comparison %d of %g = %d on the GPU, IEEE 754 says %d", j, v, g, w)
			}
		}
	}
}

const loopsSrc = `
@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.x) { return; }
  let n = inp[g.x];
  var s = 0u;
  for (var k = 0u; k < n; k++) {
    if (k % 3u == 0u) { continue; }
    if (k > 40u) { break; }
    s += k * k;
  }
  var i = 2147483640;
  for (var t = 0u; t < n; t++) { i++; }
  var m = -2147483640;
  for (var t = 0u; t < n; t++) { m--; }
  out[g.x * 3u + 0u] = s;
  out[g.x * 3u + 1u] = bitcast<u32>(i);
  out[g.x * 3u + 2u] = bitcast<u32>(m);
}
`

func TestLoopsAndWrapping(t *testing.T) {
	var ns []uint32
	for n := range uint32(75) {
		ns = append(ns, n)
	}
	out := runModule(t, loopsSrc, "run", []buf{storage(u32s(ns...)), storage(make([]byte, 12*len(ns))), uniform(u32s(uint32(len(ns)), 0, 0, 0))}, groupsFor(len(ns), 64))
	for g, n := range ns {
		var s uint32
		for k := uint32(0); k < n; k++ {
			if k%3 == 0 {
				continue
			}
			if k > 40 {
				break
			}
			s += k * k
		}
		i, m := int32(2147483640), int32(-2147483640)
		for range n {
			i = int32(uint32(i) + 1)
			m = int32(uint32(m) - 1)
		}
		for c, w := range [3]uint32{s, uint32(i), uint32(m)} {
			if got := word(out[1], g*3+c); got != w {
				t.Errorf("n=%d result %d = %d on the GPU, WGSL says %d", n, c, got, w)
			}
		}
	}
}

const structSrc = `
struct Q { v: vec4f, w: vec2u, a: u32, b: i32 }   // 32 bytes
@group(0) @binding(0) var<storage, read> src: array<Q>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;

fn mix(q: Q, k: u32) -> u32 { return q.a * k + bitcast<u32>(q.b) + q.w.x; }

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.x) { return; }
  let q = src[g.x];
  let o = g.x * 9u;
  out[o + 0u] = bitcast<u32>(q.v.x);
  out[o + 1u] = bitcast<u32>(q.v.y);
  out[o + 2u] = bitcast<u32>(q.v.z);
  out[o + 3u] = bitcast<u32>(q.v.w);
  out[o + 4u] = q.w.x;
  out[o + 5u] = q.w.y;
  out[o + 6u] = q.a;
  out[o + 7u] = bitcast<u32>(q.b);
  out[o + 8u] = mix(q, 2654435761u);
}
`

// Whole structures in a buffer, read into a local, passed to a function and
// read member by member (the Inst of gpu2d's kernel).
func TestStructsInBuffers(t *testing.T) {
	const n = 37
	in := make([]byte, 0, 32*n)
	type q struct {
		v [4]float32
		w [2]uint32
		a uint32
		b int32
	}
	qs := make([]q, n)
	for i := range qs {
		s := uint32(i)*2654435761 + 12345
		qs[i] = q{[4]float32{float32(i) * 0.5, -1.25, float32(s % 977), 3}, [2]uint32{s, s >> 3}, s * 7, int32(s) - 1<<30}
		in = append(in, f32bits(qs[i].v[:]...)...)
		in = append(in, u32s(qs[i].w[0], qs[i].w[1], qs[i].a, uint32(qs[i].b))...)
	}
	out := runModule(t, structSrc, "run", []buf{storage(in), storage(make([]byte, 4*9*n)), uniform(u32s(n, 0, 0, 0))}, groupsFor(n, 64))
	for i, s := range qs {
		want := []uint32{math.Float32bits(s.v[0]), math.Float32bits(s.v[1]), math.Float32bits(s.v[2]), math.Float32bits(s.v[3]),
			s.w[0], s.w[1], s.a, uint32(s.b), s.a*2654435761 + uint32(s.b) + s.w[0]}
		for c, w := range want {
			if g := word(out[1], i*9+c); g != w {
				t.Fatalf("element %d word %d = %#x, want %#x", i, c, g, w)
			}
		}
	}
}

const sqrtSrc = `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.x) { return; }
  let a = x[2u * g.x];
  let b = x[2u * g.x + 1u];
  o[2u * g.x] = sqrt(a);
  o[2u * g.x + 1u] = length(vec2f(a, b));
}
`

// TestSqrtAgainstTheReference MEASURES the rounding of sqrt and length on the
// device against the correctly rounded value (the f64 square root rounded once
// to f32, which is exact for a square root of an f32). docs/NATIVE.md, "sqrt is
// the one operation no specification pins": Vulkan allows 3 ULP. It reports
// counts and fails only on a result more than 3 ULP off, the specification's
// bound. It does not wire in a corrector.
func TestSqrtAgainstTheReference(t *testing.T) {
	const n = 1 << 18
	rng := rand.New(rand.NewSource(1))
	xs := make([]float32, 2*n)
	for i := range xs {
		switch i % 3 {
		case 0: // the range of gpu2d's coverage: squares of distances up to a few thousand
			xs[i] = float32(rng.Float64() * 4096)
		case 1: // every binade
			xs[i] = math.Float32frombits(rng.Uint32() & 0x7f7fffff)
		default: // exact squares and their neighbours, where midpoints live
			r := float32(rng.Intn(1 << 11))
			xs[i] = math.Float32frombits(math.Float32bits(r*r) + uint32(rng.Intn(3)) - 1)
		}
	}
	out := runModule(t, sqrtSrc, "run", []buf{storage(f32bits(xs...)), storage(make([]byte, 8*n)), uniform(u32s(n, 0, 0, 0))}, groupsFor(n, 64))
	var sqrtOff, sqrtFar, lenOff, lenFar, sqrtTotal, lenTotal int
	worstS, worstL := int64(0), int64(0)
	for i := range n {
		a, b := xs[2*i], xs[2*i+1]
		ref := func(x float32) float32 { return float32(math.Sqrt(float64(x))) }
		if a >= 0 && !math.IsInf(float64(a), 0) && ftz(a) == a {
			sqrtTotal++
			g := math.Float32frombits(word(out[1], 2*i))
			if g != ref(a) {
				sqrtOff++
				d := ulpDist(g, ref(a))
				worstS = max(worstS, d)
				if d > 3 {
					sqrtFar++
				}
			}
		}
		if s := float32(float32(a*a) + float32(b*b)); !math.IsInf(float64(s), 0) && ftz(a) == a && ftz(b) == b && s >= 0x1p-126 {
			lenTotal++
			g := math.Float32frombits(word(out[1], 2*i+1))
			if g != ref(s) {
				lenOff++
				d := ulpDist(g, ref(s))
				worstL = max(worstL, d)
				if d > 3 {
					lenFar++
				}
			}
		}
	}
	t.Logf("sqrt:   %d of %d inputs differ from the correctly rounded value (worst %d ULP)", sqrtOff, sqrtTotal, worstS)
	t.Logf("length: %d of %d inputs differ from sqrt of the f32 sum of squares, correctly rounded (worst %d ULP)", lenOff, lenTotal, worstL)
	if sqrtFar > 0 || lenFar > 0 {
		t.Errorf("results more than 3 ULP from the reference: sqrt %d, length %d (Vulkan's bound is 3 ULP)", sqrtFar, lenFar)
	}
}

func ulpDist(a, b float32) int64 {
	x, y := int64(int32(math.Float32bits(a))), int64(int32(math.Float32bits(b)))
	if x < 0 {
		x = math.MinInt32 - x
	}
	if y < 0 {
		y = math.MinInt32 - y
	}
	if d := x - y; d < 0 {
		return -d
	} else {
		return d
	}
}
