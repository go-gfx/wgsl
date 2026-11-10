package metaltest

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/go-gfx/wgsl"
	"github.com/go-gfx/wgsl/msl"
	mtl "github.com/go-macos/metal"
)

// The compute-entry constructs of msl.Module, run on the GPU and held to
// WGSL's definition computed in Go: buffers of every address space and access
// (storage read, storage read_write, uniform of a struct and of a vector),
// structures in buffers, @builtin(global_invocation_id) in one and two
// dimensions, functions that reach a buffer through a callee, for, while and
// loop with break and continue, ++ and -- on i32 and u32, and f32 to integer
// conversion.

func emitModule(t *testing.T, src string, entry string) string {
	t.Helper()
	m, err := wgsl.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	out, err := msl.Module(m, info, entry)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// runModule binds bufs at buffer indices 0, 1, ... and dispatches the kernel
// over dims threads; it returns every buffer's bytes afterwards.
func runModule(t *testing.T, d *mtl.Device, src, entry string, bufs [][]byte, dims ...int) [][]byte {
	t.Helper()
	code := emitModule(t, src, entry)
	lib, err := d.Compile(code)
	if err != nil {
		t.Fatalf("the emitted MSL does not compile: %v\n%s", err, code)
	}
	defer lib.Close()
	p, err := lib.Pipeline(entry)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	mb := make([]*mtl.Buffer, len(bufs))
	for i, b := range bufs {
		if mb[i], err = d.NewBuffer(len(b)); err != nil {
			t.Fatal(err)
		}
		defer mb[i].Close()
		copy(mb[i].Bytes(), b)
	}
	if err := d.Run(func(e *mtl.Encoder) {
		e.Use(p)
		for i, b := range mb {
			e.Buffer(i, b)
		}
		e.Dispatch(dims...)
	}); err != nil {
		t.Fatalf("%v\n%s", err, code)
	}
	out := make([][]byte, len(bufs))
	for i, b := range mb {
		out[i] = append([]byte(nil), b.Bytes()...)
	}
	return out
}

func u32s(v ...uint32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}

func f32bits(v ...float32) []byte {
	u := make([]uint32, len(v))
	for i, x := range v {
		u[i] = math.Float32bits(x)
	}
	return u32s(u...)
}

func word(b []byte, i int) uint32 { return binary.LittleEndian.Uint32(b[4*i:]) }

const buffersSrc = `
struct Q { v: vec4f, w: vec2u, a: u32, b: i32 }              // 32 bytes: vec4, vec2, two scalars
struct U { k: u32, off: i32, scale: f32, n: u32, rows: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> src: array<Q>;
@group(0) @binding(1) var<storage, read_write> dst: array<Q>;
@group(0) @binding(2) var<uniform> u: U;

fn mix(q: Q) -> u32 { return q.a * u.k + bitcast<u32>(q.b + u.off) + q.w.x; }
fn deep(q: Q) -> u32 { return mix(q) ^ 5u; }                 // reaches u through mix

@compute @workgroup_size(16, 8)
fn run(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.n || gid.y >= u.rows) { return; }
  let i = gid.y * u.n + gid.x;
  let q = src[i];
  var r = q;
  r.a = deep(q);
  r.b = q.b - 1;
  r.v = q.v * u.scale;
  r.w = q.w + vec2u(1u, 2u);
  dst[i] = r;
}
`

func TestComputeBuffersAndBuiltins(t *testing.T) {
	d := device(t)
	const n, rows = 37, 19 // not multiples of any threadgroup shape
	const qsize = 32
	const k = uint32(2654435761)
	off := int32(-7)
	const scale = float32(1.5)
	type q struct {
		v [4]float32
		w [2]uint32
		a uint32
		b int32
	}
	in := make([]byte, qsize*n*rows)
	srcQ := make([]q, n*rows)
	for i := range srcQ {
		s := uint32(i)*2654435761 + 12345
		srcQ[i] = q{[4]float32{float32(i) * 0.5, -1.25, float32(s % 977), 3}, [2]uint32{s, s >> 3}, s * 7, int32(s) - 1<<30}
		o := qsize * i
		copy(in[o:], f32bits(srcQ[i].v[:]...))
		copy(in[o+16:], u32s(srcQ[i].w[0], srcQ[i].w[1], srcQ[i].a, uint32(srcQ[i].b)))
	}
	uni := append(u32s(k, uint32(off)), f32bits(scale)...)
	uni = append(uni, u32s(n, rows, 0, 0, 0)...)
	// A guard value in dst: nothing outside the guarded grid may be touched.
	dst := make([]byte, len(in)+256)
	for i := range dst {
		dst[i] = 0xAB
	}
	out := runModule(t, d, buffersSrc, "run", [][]byte{in, dst, uni}, n, rows)
	got := out[1]
	for i, s := range srcQ {
		o := qsize * i
		mix := s.a*k + uint32(s.b) + uint32(off) + s.w[0]
		want := q{b: int32(uint32(s.b) - 1), a: mix ^ 5, w: [2]uint32{s.w[0] + 1, s.w[1] + 2}}
		for c := range 4 {
			want.v[c] = s.v[c] * scale
		}
		for c := range 4 {
			if g := word(got[o:], c); g != math.Float32bits(want.v[c]) && !(want.v[c] == 0 && g == 0) {
				t.Fatalf("element %d v[%d] = %#x, want %#x", i, c, g, math.Float32bits(want.v[c]))
			}
		}
		g := [4]uint32{word(got[o+16:], 0), word(got[o+16:], 1), word(got[o+16:], 2), word(got[o+16:], 3)}
		if w := [4]uint32{want.w[0], want.w[1], want.a, uint32(want.b)}; g != w {
			t.Fatalf("element %d w, a, b = %v, want %v", i, g, w)
		}
	}
	for i := len(in); i < len(got); i++ {
		if got[i] != 0xAB {
			t.Fatalf("byte %d past the data was written", i)
		}
	}
}

const loopsSrc = `
@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;       // u.x is the count

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
  var w = 0u;
  var j = n;
  while (j > 0u) {
    w += j;
    j -= 1u;
    if (w > 1000u) { break; }
  }
  var l = 0u;
  loop {
    l++;
    if (l >= n) { break; }
  }
  var i = 2147483640;
  for (var t = 0u; t < n; t++) { i++; }
  var m = -2147483640;
  for (var t = 0u; t < n; t++) { m--; }
  out[g.x * 5u + 0u] = s;
  out[g.x * 5u + 1u] = w;
  out[g.x * 5u + 2u] = l;
  out[g.x * 5u + 3u] = bitcast<u32>(i);
  out[g.x * 5u + 4u] = bitcast<u32>(m);
}
`

func TestComputeLoops(t *testing.T) {
	d := device(t)
	var ns []uint32
	for n := range uint32(75) {
		ns = append(ns, n)
	}
	cnt := uint32(len(ns))
	out := runModule(t, d, loopsSrc, "run", [][]byte{u32s(ns...), make([]byte, 4*5*len(ns)), u32s(cnt, 0, 0, 0)}, len(ns))
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
		var w uint32
		for j := n; j > 0; {
			w += j
			j--
			if w > 1000 {
				break
			}
		}
		l := uint32(0)
		for {
			l++
			if l >= n {
				break
			}
		}
		i := int32(2147483640)
		m := int32(-2147483640)
		for range n {
			i = int32(uint32(i) + 1)
			m = int32(uint32(m) - 1)
		}
		want := [5]uint32{s, w, l, uint32(i), uint32(m)}
		for c, wv := range want {
			if got := word(out[1], g*5+c); got != wv {
				t.Errorf("n=%d result %d = %d on the GPU, WGSL says %d", n, c, got, wv)
			}
		}
	}
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
func TestComputeFloatToInteger(t *testing.T) {
	d := device(t)
	inf := float32(math.Inf(1))
	vals := []float32{0, float32(math.Copysign(0, -1)), 0.5, 0.9999, -0.9, 1, 1.5, -1.5, 2.5, 255.5, 256, 65535.75, 16777216, 1e9,
		2147483520, 2147483648, 3e9, 4294967040, 4294967296, 1e10, 3.4e38, inf, -1, -2147483648, -2147483904, -3e9, -1e10, -inf, 1e-45, -1e-45}
	cnt := uint32(len(vals))
	out := runModule(t, d, convSrc, "run", [][]byte{f32bits(vals...), make([]byte, 8*len(vals)), u32s(cnt, 0, 0, 0)}, len(vals))
	for i, v := range vals {
		x := math.Trunc(float64(v))
		u := math.Min(math.Max(x, 0), 4294967040)
		s := math.Min(math.Max(x, -2147483648), 2147483520)
		if g := word(out[1], 2*i); g != uint32(u) {
			t.Errorf("u32(%g) = %d on the GPU, WGSL says %d", v, g, uint32(u))
		}
		if g := int32(word(out[1], 2*i+1)); g != int32(s) {
			t.Errorf("i32(%g) = %d on the GPU, WGSL says %d", v, g, int32(s))
		}
	}
	// A NaN gives an indeterminate value: any, but the kernel must finish.
	nan := float32(math.NaN())
	runModule(t, d, convSrc, "run", [][]byte{f32bits(nan), make([]byte, 8), u32s(1, 0, 0, 0)}, 1)
}

const forwardSrc = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;

fn first(i: u32) -> u32 { return second(i) + 1u; }   // second is declared below
fn second(i: u32) -> u32 { return out[i] * 3u; }       // and reaches the buffer

@compute @workgroup_size(32)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= 8u) { return; }
  out[g.x] = first(g.x);
}
`

func TestComputeCallsADeclaredLaterFunction(t *testing.T) {
	d := device(t)
	out := runModule(t, d, forwardSrc, "run", [][]byte{u32s(0, 1, 2, 3, 4, 5, 6, 7)}, 8)
	for i := range 8 {
		if got, want := word(out[0], i), uint32(i)*3+1; got != want {
			t.Errorf("out[%d] = %d, want %d", i, got, want)
		}
	}
}
