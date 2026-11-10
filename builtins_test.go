package wgsl

import (
	"os"
	"strings"
	"testing"
)

// TestBuiltinsMatchSpec checks that every built-in function listed in the
// pinned specification (testdata/spec-builtin-names.txt, written by
// `go run ./internal/specbuiltins -names wgsl/index.bs`) is known to the
// checker and has an implementation.
func TestBuiltinsMatchSpec(t *testing.T) {
	b, err := os.ReadFile("testdata/spec-builtin-names.txt")
	if err != nil {
		t.Fatal(err)
	}
	notBuiltinFns := map[string]bool{"array": true, "bool": true, "f16": true, "f32": true, "i32": true, "u32": true,
		"vec2": true, "vec3": true, "vec4": true, "bitcast": true, "Structures": true,
		// buffer_view language extension (not implemented, see package doc)
		"bufferView": true, "bufferArrayView": true, "bufferLength": true}
	n := 0
	for _, name := range strings.Fields(string(b)) {
		if strings.HasPrefix(name, "mat") || notBuiltinFns[name] {
			if !predeclared(name) && !strings.HasPrefix(name, "buffer") && name != "Structures" {
				t.Errorf("spec constructor %s is not predeclared", name)
			}
			continue
		}
		n++
		if builtinFuncs[name] == nil {
			t.Errorf("spec built-in function %s is missing", name)
		}
	}
	if n < 140 {
		t.Errorf("only %d built-in functions found in the spec list", n)
	}
	t.Logf("%d spec built-in functions covered", n)
}

func TestConstEvaluation(t *testing.T) {
	runCases(t, []checkCase{
		{"int arithmetic", "const a = 7 / 2; const b = 7 % 4; const_assert a == 3; const_assert b == 3; const_assert 2 * 3 + 1 == 7;", ""},
		{"float arithmetic", "const a = 1.5 + 2.5; const_assert a == 4.0; const_assert 0.1 + 0.2 > 0.3;", ""},
		{"wrong assert", "const_assert 1 + 1 == 3;", "const-assert"},
		{"vector const", "const v = vec3(1, 2, 3) * 2; const_assert v.y == 4; const_assert v.zyx.x == 6;", ""},
		{"array const", "const a = array(1, 2, 3); const_assert a[2] == 3;", ""},
		{"array const oob", "const a = array(1, 2, 3); const x = a[3];", "const-index-range"},
		{"struct const", "struct S { a: i32, b: f32 } const s = S(1, 2.5); const_assert s.a == 1; const_assert s.b == 2.5;", ""},
		{"u32 wrap", "const a = 0u - 1u; const_assert a == 4294967295u;", ""},
		{"i32 conversion", "const a = i32(2.9); const b = u32(-1.0); const_assert a == 2; const_assert b == 0u;", ""},
		// "Floating Point Conversion": the nearest value of the integer type that
		// is also exactly representable in the source float type. For f32 that
		// is 4294967040 and 2147483520, not 2^32-1 and 2^31-1; an AbstractFloat
		// (an f64) holds those, so it saturates at them.
		{"u32 from f32 saturates at 4294967040", "const a = u32(f32(4294967296.0)); const b = u32(f32(1e10)); const c = u32(f32(4294967040.0)); const_assert a == 4294967040u; const_assert b == 4294967040u; const_assert c == 4294967040u;", ""},
		{"u32 from f32 below the bound", "const a = u32(f32(4294966784.0)); const_assert a == 4294966784u; const b = u32(f32(-5.0)); const_assert b == 0u;", ""},
		{"i32 from f32 saturates at 2147483520", "const a = i32(f32(2147483648.0)); const b = i32(f32(1e10)); const c = i32(f32(2147483520.0)); const_assert a == 2147483520; const_assert b == 2147483520; const_assert c == 2147483520;", ""},
		{"i32 from f32 low end", "const a = i32(f32(-2147483648.0)); const b = i32(f32(-1e10)); const_assert a == -2147483648; const_assert b == -2147483648;", ""},
		{"u32 from abstract float saturates at 2^32-1", "const a = u32(4294967296.0); const b = u32(1e10); const_assert a == 4294967295u; const_assert b == 4294967295u;", ""},
		{"i32 from abstract float saturates at 2^31-1", "const a = i32(2147483648.0); const_assert a == 2147483647;", ""},
		{"i32 range", "const a = i32(3000000000);", "const-overflow"},
		{"u32 from abstract", "const a = u32(3000000000);", ""},
		{"f32 overflow conv", "const a = f32(1e39);", "const-overflow"},
		{"f16 conv", "enable f16; const a = f16(65504.0); const b = f16(70000.0);", "const-overflow"},
		{"f16 round", "enable f16; const a = f16(1.0009765625); const_assert a == 1.0009765625h;", ""},
		{"min max clamp", "const_assert min(3, 2) == 2; const_assert max(3u, 2u) == 3u; const_assert clamp(5, 1, 3) == 3;", ""},
		{"math", "const_assert abs(-2) == 2; const_assert floor(1.5) == 1.0; const_assert sqrt(16.0) == 4.0; const_assert sign(-3.0) == -1.0;", ""},
		{"sqrt domain", "const a = sqrt(-1.0);", "const-overflow"},
		{"bits", "const_assert countOneBits(7u) == 3u; const_assert reverseBits(1u) == 2147483648u; const_assert firstLeadingBit(16u) == 4u; const_assert extractBits(0xF0u, 4u, 4u) == 15u;", ""},
		{"select all any", "const_assert select(1, 2, true) == 2; const_assert all(vec2(true, true)); const_assert !any(vec2(false, false));", ""},
		{"dot cross", "const_assert dot(vec3(1, 2, 3), vec3(4, 5, 6)) == 32; const c = cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)); const_assert c.z == 1.0;", ""},
		{"matrix", "const m = mat2x2(1.0, 2.0, 3.0, 4.0); const t = transpose(m); const_assert t[0][1] == 3.0; const p = m * vec2(1.0, 1.0); const_assert p.x == 4.0;", ""},
		{"hex literals", "const_assert 0x10 == 16; const_assert 0x1p4 == 16.0; const_assert 0x1.8p1 == 3.0; const_assert 1e2 == 100.0; const_assert .5 == 0.5;", ""},
		{"hex float exact", "const a = 0x1.0000000001p0f;", "literal-range"},
		{"decimal float overflow", "const a = 1e39f;", "literal-range"},
		{"abstract int literal max", "const a = 9223372036854775807; const b = 9223372036854775808;", "literal-range"},
		{"const used in array size", "const n = 2 * 3; var<private> a: array<f32, n>; var<private> b: array<f32, 6>; fn f() { a = b; }", ""},
		{"const bool logic", "const_assert (1 < 2) && !(2 < 1); const_assert true || (1 / 0 == 0);", ""},
		{"zero array huge", "const a = array<f32, 100000000>(); const_assert a[99999999] == 0.0;", ""},
		{"pack", "const p = pack4x8unorm(vec4(1.0, 0.0, 0.0, 0.0)); const_assert p == 0u + p;", ""},
		{"abstract shift", "const a = 1 << 62; const b = 1 << 63;", "const-overflow"},
		{"workgroup size const expr", "const n = 2; @compute @workgroup_size(n * 4) fn m() {}", ""},
		{"case selector const expr", "fn f() { var a = 1; switch a { case 1 + 1 { } default { } } }", ""},
	})
}

// TestSpecExamples runs examples from the specification text.
func TestSpecExamples(t *testing.T) {
	runCases(t, []checkCase{
		// "Declaration and scope"
		{"valid and invalid declarations", `
var<private> modf: f32 = 0.0;
var<private> foo: f32 = 0.0;
var<private> bar: u32 = 0u;
fn my_func(foo: f32) {
  var bar: u32;
  {
    var foo : f32;
    var bar: u32;
    for ( var i: i32 = 0; i < 10; i++ ) { }
  }
  var early_use : i32 = later_def;
}
fn my_foo(my_foo: i32) { }
var<private> later_def : i32 = 1;`, ""},
		{"spec invalid: var shadows param", "fn my_func(foo: f32) { var foo: f32; }", "redeclaration"},
		{"spec invalid: modf resolves to module var", "var<private> modf: f32 = 0.0; fn f(foo: f32) { let res = modf(foo); }", "not-callable"},
		// type inference for literals
		{"literal inference", `
fn f() {
  var u32_1 = 1u; var i32_1 = 1i; var f32_1 = 1f;
  let some_i32 = 1;
  var i32_from_type : i32 = 1; var u32_from_type : u32 = 1;
  var f32_promotion : f32 = 1;
  var u32_from_expr = 1 + u32_1; var i32_from_expr = 1 + i32_1;
  let u32_large : u32 = 2147483649;
  let i32_min = -2147483648;
  var u32_expr1 = (1 + (1 + (1 + (1 + 1)))) + 1u;
  var u32_expr3 = (1 + (1 + (1 + (1u + 1)))) + 1;
  let i32_clamp = clamp(1, -5, 5);
  let u32_clamp = clamp(5, 0, u32_from_expr);
  let f32_clamp = clamp(0, f32_1, 1);
  let f32_promotion1 = 1.0 + 2 + 3 + 4;
  let f32_promotion3 = 1f + ((2 + 3) + 4);
  let out_and_in_again = (0x1ffffffff / 8);
}`, ""},
		{"spec invalid: u32 neg", "fn f() { var u32_neg = -1u; }", "no-matching-overload"},
		{"spec invalid: float to int", "fn f() { var i32_demotion : i32 = 1.0; }", "type-mismatch"},
		{"spec invalid: u32 too large", "fn f() { let u32_too_large : u32 = 1234567890123456890; }", "const-overflow"},
		{"spec invalid: f32 out of range", "fn f() { let x = 0x1p500; let y: f32 = x; }", "const-overflow"},
		{"spec invalid: i32 to f32", "fn f() { let some_i32 = 1; let some_f32 : f32 = some_i32; }", "type-mismatch"},
		{"spec invalid: ambiguous clamp", "fn f() { let ambiguous_clamp = clamp(1u, 0, 1i); }", "no-matching-overload"},
		{"spec invalid: u32 overflow", "fn f() { let overflow_u32 = (1 -2) + 1u; }", "const-overflow"},
		{"spec invalid: out of range", "fn f() { let out_of_range = (0x1ffffffff / 8u); }", "const-overflow"},
		// behaviors
		{"spec behaviors: dead code", "fn simple() -> i32 { var a: i32; return 0; a = 1; return 2; }", ""},
		{"spec behaviors: missing return", "fn missing_return () -> i32 { var a: i32 = 0; if a == 42 { return a; } }", "missing-return"},
		{"spec behaviors: infinite loop", "fn f() { loop { } }", "behavior"},
		{"spec behaviors: conditional continue", `
fn conditional_continue() {
  var a: i32;
  loop {
    if a == 5 { break; }
    if a % 2 == 1 { continue; }
    a = a * 2;
    continuing { a = a + 1; }
  }
}`, ""},
		{"spec behaviors: continue outside loop", "fn f () { var a: i32 = 0; if a > 0 { continue; } }", "continue"},
		{"spec behaviors: break in continuing", `fn f() { var a: i32 = 2; var i: i32 = 0; loop { let step: i32 = 1; if i % 2 == 0 { continue; } a = a * 2; continuing { i = i + step; if i >= 4 { break; } } } }`, "break"},
		{"spec behaviors: break if", `fn f() { var a: i32 = 2; var i: i32 = 0; loop { let step: i32 = 1; if i % 2 == 0 { continue; } a = a * 2; continuing { i = i + step; break if i >= 4; } } }`, ""},
		// entry points
		{"spec entry points", `
@vertex fn vert_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
@fragment fn frag_main(@builtin(position) coord_in: vec4<f32>) -> @location(0) vec4<f32> { return vec4<f32>(coord_in.x, coord_in.y, 0.0, 1.0); }
@compute @workgroup_size(1) fn comp_main() { }
@compute @workgroup_size(8,4,1) fn sorter() { }
@compute @workgroup_size(8u) fn reverser() { }
@id(42) override block_width = 12u;
@compute @workgroup_size(block_width) fn shuffler() { }`, ""},
		{"spec invalid: compute without size", "@compute fn bad_shader() { }", "workgroup-size"},
		{"spec locations", `
struct A { @location(0) x: f32, @location(1) y: f32 }
@fragment fn fragShader(in1: A, @location(2) in2: f32) -> @location(0) vec4<f32> { return vec4<f32>(); }`, ""},
		{"spec invalid locations", "struct A { @location(0) x: f32, @location(0) y: f32 }", "location-duplicate"},
		{"spec mixed io", `
struct MyInputs { @location(0) x: vec4<f32>, @builtin(front_facing) y: bool, @location(1) @interpolate(flat) z: u32 }
struct MyOutputs { @builtin(frag_depth) x: f32, @location(0) y: vec4<f32> }
@fragment fn fragShader(in1: MyInputs) -> MyOutputs { return MyOutputs(); }`, ""},
		// layout
		{"spec uniform struct stride", `
struct small_stride { a: array<f32,8> }
@group(0) @binding(0) var<uniform> invalid: small_stride;`, "layout"},
		{"spec uniform struct stride fixed", `
struct wrapped_f32 { @size(16) elem: f32 }
struct big_stride { a: array<wrapped_f32,8> }
@group(0) @binding(1) var<uniform> valid: big_stride;`, ""},
		{"spec uniform offsets invalid", `
struct S { x: f32 }
struct Invalid { a: S, b: f32 }
@group(0) @binding(0) var<uniform> invalid: Invalid;`, "layout"},
		{"spec uniform offsets valid", `
struct S { x: f32 }
struct Valid { a: S, @align(16) b: f32 }
@group(0) @binding(1) var<uniform> valid: Valid;`, ""},
		// buffers
		{"spec storage buffers", `
struct PointLight { position : vec3f, color : vec3f }
struct LightStorage { pointCount : u32, point : array<PointLight> }
@group(0) @binding(1) var<storage> lights : LightStorage;
@group(0) @binding(0) var<storage, read_write> pbuf: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> pp: PointLight;`, ""},
		{"spec workgroup override array", `
override blockSize = 16;
var<workgroup> odds: array<i32,blockSize>;
var<workgroup> evens: array<i32,blockSize>;
var<workgroup> evens_1: array<i32,(blockSize * 2 / 2)>;`, ""},
		{"spec array types", `
var<private> a: array<f32,8>;
var<private> b: array<i32,8>;
var<private> c: array<i32,8u>;
const width = 8;
const height = 8;
var<private> d: array<i32,width>;
var<private> e: array<i32,height>;
fn f() { d = e; }`, ""},
		{"spec static asserts", `
const x = 1; const y = 2;
const_assert x < y;
const_assert(y != 0);
fn foo() { const z = x + y - 2; const_assert z > 0; let a = 3; }`, ""},
		{"spec static assert runtime", "fn foo() { let a = 3; const_assert a != 0; }", "const-assert"},
		{"spec const declarations", `
const a = 4;
const b : i32 = 4;
const c : u32 = 4;
const d : f32 = 4;
const e = vec3(a, a, a);
const f = 2.0;
const g = mat2x2(a, f, a, f);
const h = array(a, f, a, f);`, ""},
		{"spec override declarations", `
@id(0)    override has_point_light: bool = true;
@id(1200) override specular_param: f32 = 2.3;
@id(1300) override gain: f32;
          override width: f32 = 0.0;
          override depth: f32;
          override height = 2 * depth;`, ""},
		{"spec override div zero", "override a : i32 = 0; override b = a / 0;", "const-div-zero"},
		{"spec pointers", `
fn bar(p : ptr<function, f32>) { }
fn baz(p : ptr<private, i32>) { }
fn bar2(p : ptr<function, f32>) { let a = &*&*(p); bar(p); bar(a); }
struct S { x : i32 }
@group(0) @binding(0) var<storage> ro_storage : f32;
var usable_priv : i32;
fn foo() {
  var usable_func : f32;
  let a_priv = &usable_priv;
  let c_priv = &*&usable_priv;
  baz(&usable_priv); baz(a_priv); baz(c_priv);
  bar(&usable_func);
}`, "address-space"},
	})
}
