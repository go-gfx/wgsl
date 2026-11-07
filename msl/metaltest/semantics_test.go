// Package metaltest runs MSL emitted by package msl on a Mac's GPU and holds
// every result to WGSL's definition, computed in Go: the integer cases MSL
// leaves undefined (overflow, a zero divisor, MIN / -1, shift counts of 32
// and more, >> of a negative value) and the float functions Metal would
// otherwise compute with fast math. It is a module of its own so that
// go-gfx/wgsl does not depend on go-macos/metal.
package metaltest

import (
	"errors"
	"math"
	"testing"
	"unsafe"

	"github.com/go-gfx/wgsl"
	"github.com/go-gfx/wgsl/msl"
	mtl "github.com/go-macos/metal"
)

const ints = `
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
`

const floats = `
fn clampf(x: f32, lo: f32, hi: f32) -> f32 { return clamp(x, lo, hi); }
fn len2(a: f32, b: f32) -> f32 { return length(vec2f(a, b)); }
fn len3(a: f32, b: f32, c: f32) -> f32 { return length(vec3f(a, b, c)); }
fn pick(a: f32, b: f32, c: f32) -> f32 { return select(a, b, c > 0.0); }
fn absf(a: f32) -> f32 { return abs(a); }
fn minmax(a: f32, b: f32) -> f32 { return max(min(a, b), 0.25); }
`

// The Go side: WGSL's definitions (gpuweb wgsl/index.bs, opened 2026-11-07).
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
}

var intNames = []string{"add", "sub", "mul", "div", "rem", "neg", "shl", "shr", "udiv", "urem", "ushl", "ushr", "compound"}

func emit(t *testing.T, src string, names ...string) string {
	t.Helper()
	m, err := wgsl.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	out, err := msl.Functions(m, info, names...)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func device(t *testing.T) *mtl.Device {
	t.Helper()
	d, err := mtl.Default()
	if errors.Is(err, mtl.ErrUnsupported) || errors.Is(err, mtl.ErrNoDevice) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	return d
}

func run(t *testing.T, d *mtl.Device, src, kernel string, in []byte, outLen int) []byte {
	t.Helper()
	lib, err := d.Compile(src)
	if err != nil {
		t.Fatalf("the emitted MSL does not compile: %v\n%s", err, src)
	}
	defer lib.Close()
	p, err := lib.Pipeline(kernel)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	ib, err := d.NewBuffer(len(in))
	if err != nil {
		t.Fatal(err)
	}
	defer ib.Close()
	copy(ib.Bytes(), in)
	ob, err := d.NewBuffer(outLen)
	if err != nil {
		t.Fatal(err)
	}
	defer ob.Close()
	n := len(in) / 8
	if err := d.Run(func(e *mtl.Encoder) {
		e.Use(p)
		e.Buffer(0, ib)
		e.Buffer(1, ob)
		e.Dispatch(n)
	}); err != nil {
		t.Fatal(err)
	}
	return append([]byte(nil), ob.Bytes()...)
}

func TestIntegerSemantics(t *testing.T) {
	d := device(t)
	vals := []int32{0, 1, -1, 2, -2, 3, -7, 7, 31, 32, 33, 64, -32, 1 << 30, math.MaxInt32, math.MinInt32, math.MinInt32 + 1, 123456789, -987654321}
	var pairs []int32
	for _, a := range vals {
		for _, b := range vals {
			pairs = append(pairs, a, b)
		}
	}
	n := len(pairs) / 2
	k := len(intNames)
	src := emit(t, ints, intNames...) + `
kernel void ints(device const int2 *in [[buffer(0)]], device int *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  int a = in[i].x; int b = in[i].y;
  device int *o = out + i * ` + itoa(k) + `;
  o[0] = add(a, b); o[1] = sub(a, b); o[2] = mul(a, b); o[3] = div(a, b); o[4] = rem(a, b);
  o[5] = neg(a); o[6] = shl(a, b); o[7] = shr(a, b); o[8] = udiv(a, b); o[9] = urem(a, b);
  o[10] = ushl(a, b); o[11] = ushr(a, b); o[12] = compound(a, b);
}`
	out := run(t, d, src, "ints", int32Bytes(pairs), 4*k*n)
	got := unsafe.Slice((*int32)(unsafe.Pointer(&out[0])), k*n)
	for i := range n {
		a, b := pairs[2*i], pairs[2*i+1]
		for j, ref := range intRef {
			if want := ref(a, b); got[i*k+j] != want {
				t.Errorf("%s(%d, %d) = %d on the GPU, WGSL says %d", intNames[j], a, b, got[i*k+j], want)
			}
		}
	}
}

func TestFloatSemantics(t *testing.T) {
	d := device(t)
	vals := []float32{0, -0.0, 1, -1, 0.5, 3, -4, 1e-30, 1e30, 3.4e38, 0.1, 2.5, -7.25, 1e-45}
	var pairs []float32
	for _, a := range vals {
		for _, b := range vals {
			pairs = append(pairs, a, b)
		}
	}
	n := len(pairs) / 2
	const k = 7
	src := emit(t, floats, "clampf", "len2", "len3", "pick", "absf", "minmax") + `
kernel void floats(device const float2 *in [[buffer(0)]], device float *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  float a = in[i].x; float b = in[i].y;
  device float *o = out + i * 7;
  o[0] = clampf(a, b, 1.5f); o[1] = clampf(a, 1.5f, b); o[2] = len2(a, b); o[3] = len3(a, b, a);
  o[4] = pick(a, b, a - b); o[5] = absf(a); o[6] = minmax(a, b);
}`
	in := make([]byte, 0, 4*len(pairs))
	for _, v := range pairs {
		in = append(in, float32Bytes(v)...)
	}
	out := run(t, d, src, "floats", in, 4*k*n)
	got := unsafe.Slice((*float32)(unsafe.Pointer(&out[0])), k*n)
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
			float32(math.Abs(float64(a))), max32(min32(a, b), 0.25)}
	}
	same := func(g, w float32) bool { return math.Float32bits(g) == math.Float32bits(w) || (g == 0 && w == 0) }
	for i := range n {
		a, b := pairs[2*i], pairs[2*i+1]
		exact := expect(a, b)
		// WGSL lets any input or output of these operations be flushed to
		// zero when subnormal (wgsl/index.bs, "flush to zero"): the GPU may
		// give either result, and nothing else.
		flushed := expect(ftz(a), ftz(b))
		for j := range exact {
			g := got[i*k+j]
			if j == 4 {
				// select(a, b, a - b > 0): each operation may flush on its
				// own, so the condition may be decided on c or on ftz(c), and
				// the value chosen may itself be flushed.
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
			if !same(g, exact[j]) && !same(g, ftz(flushed[j])) {
				t.Errorf("function %d (%g, %g) = %g on the GPU, WGSL gives %g (or %g flushed to zero)", j, a, b, g, exact[j], ftz(flushed[j]))
			}
		}
	}
}

// ftz flushes a subnormal to zero.
func ftz(x float32) float32 {
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

func itoa(n int) string { return string(rune('0'+n/10)) + string(rune('0'+n%10)) }

func int32Bytes(v []int32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		u := uint32(x)
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
	}
	return b
}

func float32Bytes(f float32) []byte {
	u := math.Float32bits(f)
	return []byte{byte(u), byte(u >> 8), byte(u >> 16), byte(u >> 24)}
}
