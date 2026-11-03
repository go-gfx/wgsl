package wgsl

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

type checkCase struct {
	name string
	src  string
	rule string // "" = must be accepted; otherwise a diagnostic with this Rule must be reported
}

func checkSrc(t *testing.T, src string) (*Info, ErrorList) {
	t.Helper()
	s := NewSource("t.wgsl", src)
	m, err := ParseSource(s, Limits{})
	if err != nil {
		t.Fatalf("parse error: %v\n%s", err, src)
	}
	return Check(m, CheckOptions{Source: s})
}

func runCases(t *testing.T, cases []checkCase) {
	t.Helper()
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			_, errs := checkSrc(t, tc.src)
			if tc.rule == "" {
				if len(errs) > 0 {
					t.Fatalf("want accepted, got %v\n%s", errs, tc.src)
				}
				return
			}
			for _, e := range errs {
				if e.Rule == tc.rule {
					if e.Pos.Line == 0 || e.Msg == "" {
						t.Fatalf("diagnostic without position or message: %+v", e)
					}
					return
				}
			}
			t.Fatalf("want rule %q, got %v\n%s", tc.rule, errs, tc.src)
		})
	}
}

func fn(body string) string { return "fn f() {\n" + body + "\n}\n" }
func cs(body string) string {
	return "@compute @workgroup_size(1) fn main() {\n" + body + "\n}\n"
}

func TestDeclarationsAndScope(t *testing.T) {
	runCases(t, []checkCase{
		{"order independent", "fn a() -> i32 { return b(); } fn b() -> i32 { return later; } const later = 3;", ""},
		{"redeclared global", "var<private> x: i32; var<private> x: i32;", "redeclaration"},
		{"redeclared function", "fn a() {} fn a() {}", "redeclaration"},
		{"local redeclared", fn("var a: i32; var a: i32;"), "redeclaration"},
		{"param redeclared in body", "fn f(a: i32) { var a: i32; }", "redeclaration"},
		{"duplicate param", "fn f(a: i32, a: i32) {}", "redeclaration"},
		{"shadow in nested block", fn("var a: i32; { var a: u32; }"), ""},
		{"shadow predeclared", "fn min() -> u32 { return 0; } const rgba8unorm = 12;", ""},
		{"shadow type name locally", fn("let f32 = 1; let x = f32 + 1;"), ""},
		{"use before declaration", fn("let x = y; let y = 1;"), "undeclared-ident"},
		{"self reference let", fn("let x = x;"), "undeclared-ident"},
		{"for scope shared with body", fn("for (var i: i32 = 0; i < 2; i++) { var i: i32 = 1; }"), "redeclaration"},
		{"scope ends with block", fn("{ let a = 1; } let b = a;"), "undeclared-ident"},
		{"recursive function", "fn a() { b(); } fn b() { a(); }", "recursive-declaration"},
		{"self recursive function", "fn a() { a(); }", "recursive-declaration"},
		{"recursive struct", "struct S { a: S }", "recursive-declaration"},
		{"cyclic aliases", "alias A = B; alias B = A;", "recursive-declaration"},
		{"cyclic consts", "const a = b; const b = a;", "recursive-declaration"},
		{"const self reference", "const a = a + 1;", "recursive-declaration"},
		{"undeclared type", "var<private> x: Foo;", "undeclared-ident"},
		{"value used as type", "const c = 1; var<private> x: c;", "not-a-type"},
		{"type used as value", fn("let x = f32;"), "type-as-value"},
		{"function as value", "fn g() {} fn f() { let x = g; }", "function-as-value"},
	})
}

func TestTypesAndConversions(t *testing.T) {
	runCases(t, []checkCase{
		{"abstract int to i32", fn("let a: i32 = 1;"), ""},
		{"abstract int to u32", fn("let a: u32 = 1;"), ""},
		{"abstract int to f32", fn("let a: f32 = 1;"), ""},
		{"abstract float to i32", fn("let a: i32 = 1.0;"), "type-mismatch"},
		{"i32 to f32", fn("let a = 1i; let b: f32 = a;"), "type-mismatch"},
		{"u32 plus abstract float", fn("let a = 1u + 2.5;"), "no-matching-overload"},
		{"mixed abstract", fn("let a = 1 + 2.5;"), ""},
		{"i32 min literal", fn("let a = -2147483648;"), ""},
		{"i32 too large", fn("let a = 2147483648;"), "const-overflow"},
		{"u32 large ok", fn("let a: u32 = 2147483649;"), ""},
		{"i32 large bad", fn("let a: i32 = 2147483649;"), "const-overflow"},
		{"neg u32", fn("let a = -1u;"), "no-matching-overload"},
		{"suffix range", fn("let a = 4294967296u;"), "literal-range"},
		{"f16 needs enable", "var<private> x: f16;", "extension-required"},
		{"f16 enabled", "enable f16; var<private> x: f16 = 1h;", ""},
		{"f16 literal needs enable", fn("let a = 1h;"), "extension-required"},
		{"abstract wrap", fn("let a = (1 - 2) + 1u;"), "const-overflow"},
		{"i32 wrap const", fn("const a = -1 * i32(-2147483648);"), ""},
		{"clamp inference", fn("let x = clamp(1, -5, 5); let y: i32 = x;"), ""},
		{"clamp mixed sign", fn("let x = clamp(1u, 0, 1i);"), "no-matching-overload"},
		{"promotion", fn("let a = 1.0 + 2 + 3; let b: f32 = a;"), ""},
		{"vec ctor splat", fn("let v = vec3<f32>(1.0); let w = vec3(1, 2, 3); let x: vec3<i32> = w;"), ""},
		{"vec ctor mixed", fn("let v = vec4<f32>(vec2<f32>(1, 2), 3, 4);"), ""},
		{"vec ctor wrong count", fn("let v = vec3<f32>(1.0, 2.0);"), "no-matching-overload"},
		{"vec ctor wrong type", fn("let v = vec2<f32>(1i, 2i);"), "no-matching-overload"},
		{"vec ctor convert", fn("let v = vec2<f32>(vec2<i32>(1, 2));"), ""},
		{"mat ctor", fn("let m = mat2x2<f32>(1, 2, 3, 4); let n = mat2x2f(vec2f(1), vec2f(2));"), ""},
		{"mat int elem", "var<private> m: mat2x2<i32>;", "type-args"},
		{"array ctor", fn("let a = array<i32, 3>(1, 2, 3); let b = array(1, 2.5);"), ""},
		{"array ctor count", fn("let a = array<i32, 3>(1, 2);"), "arg-count"},
		{"struct ctor", "struct S { a: i32, b: f32 } fn f() { let s = S(1, 2); }", ""},
		{"struct ctor arity", "struct S { a: i32, b: f32 } fn f() { let s = S(1); }", "arg-count"},
		{"zero values", fn("let a = vec3<f32>(); let b = array<i32, 4>(); let c = mat2x2<f32>(); let d = f32();"), ""},
		{"scalar conversions", fn("let a = f32(1u); let b = u32(-1i); let c = i32(1.5); let d = bool(2);"), ""},
		{"bitcast", fn("let a = bitcast<f32>(1u); let b = bitcast<vec2<u32>>(vec2<f32>());"), ""},
		{"bitcast bad", fn("let a = bitcast<bool>(1u);"), "bitcast"},
		{"var inference", fn("var a = 1; var b = 2.5; a = 2; b = 1.0;"), ""},
		{"var assign mismatch", fn("var a = 1; a = 1.5;"), "type-mismatch"},
		{"let needs constructible", "@group(0) @binding(0) var<storage, read_write> a: atomic<u32>; fn f() { let x = a; }", "let-type"},
		{"let pointer", fn("var a: i32; let p = &a; let q = *p;"), ""},
		{"atomic type arg", "var<workgroup> a: atomic<f32>;", "type-args"},
		{"runtime array in function", fn("var a: array<i32>;"), "var-type"},
		{"zero array count", "var<private> a: array<i32, 0>;", "array-count"},
		{"negative array count", "var<private> a: array<i32, -1>;", "array-count"},
		{"float array count", "var<private> a: array<i32, 1.5>;", "array-count"},
		{"array count const", "const n = 4; var<private> a: array<i32, n>; var<private> b: array<i32, 4u>;", ""},
		{"array same type", "const n = 4; var<private> a: array<i32, n>; fn f() { var b: array<i32, 4u> = a; }", ""},
		{"array of runtime", "struct S { a: array<array<i32>, 2> } ", "array-element"},
		{"nested runtime array", "var<private> a: array<array<i32>, 2>;", "array-element"},
		{"struct runtime last", "struct S { a: i32, b: array<i32> } @group(0) @binding(0) var<storage> s: S;", ""},
		{"struct runtime not last", "struct S { b: array<i32>, a: i32 }", "struct-member-type"},
		{"struct duplicate member", "struct S { a: i32, a: i32 }", "redeclaration"},
		{"struct pointer member", "struct S { a: ptr<function, i32> }", "struct-member-type"},
		{"struct nested runtime", "struct A { a: array<i32> } struct B { a: A }", "struct-member-type"},
		{"override sized array", "override n = 4; var<workgroup> a: array<i32, n>;", ""},
		{"override array private", "override n = 4; var<private> a: array<i32, n>;", "var-type"},
		{"override nested", "override n = 4; var<workgroup> a: array<array<i32, n>, 2>;", "array-element"},
		{"ptr access mode", "fn f(p: ptr<function, i32, read_write>) {}", "type-args"},
		{"ptr storage access", "@group(0) @binding(0) var<storage, read_write> a: i32; fn g(p: ptr<storage, i32, read_write>) {} fn f() { g(&a); }", ""},
		{"texture types", "@group(0) @binding(0) var t: texture_2d<f32>; @group(0) @binding(1) var s: sampler; @group(0) @binding(2) var d: texture_depth_2d; @group(0) @binding(3) var w: texture_storage_2d<rgba8unorm, write>;", ""},
		{"texture bad sampled type", "@group(0) @binding(0) var t: texture_2d<bool>;", "type-args"},
		{"storage texture format", "@group(0) @binding(0) var t: texture_storage_2d<f32, write>;", "type-args"},
	})
}

func TestExpressions(t *testing.T) {
	runCases(t, []checkCase{
		{"index vec", fn("let v = vec3<f32>(); let a = v[1]; let b = v.x; let c = v.xyz; let d = v.rgb;"), ""},
		{"swizzle mixed", fn("let v = vec4<f32>(); let a = v.xg;"), "swizzle"},
		{"swizzle past end", fn("let v = vec2<f32>(); let a = v.z;"), "swizzle"},
		{"const index out of range", fn("let v = vec3<f32>(); let a = v[3];"), "const-index-range"},
		{"runtime index ok", fn("var i = 5; let v = vec3<f32>(); let a = v[i];"), ""},
		{"array const oob", fn("var a: array<i32, 4>; let x = a[4];"), "const-index-range"},
		{"index float", fn("var a: array<i32, 4>; let x = a[1.5];"), "index-type"},
		{"matrix column", fn("let m = mat3x2<f32>(); let c = m[2]; let e = m[1][0];"), ""},
		{"matrix oob", fn("let m = mat3x2<f32>(); let c = m[3];"), "const-index-range"},
		{"struct member", "struct S { a: i32 } fn f() { var s: S; s.a = 1; let b = s.a; }", ""},
		{"no member", "struct S { a: i32 } fn f() { var s: S; let b = s.c; }", "no-member"},
		{"arith vec scalar", fn("let v = vec2<f32>(); let a = v * 2.0; let b = 2.0 * v; let c = v + v;"), ""},
		{"mat mul", fn("let m = mat2x3<f32>(); let n = mat3x2<f32>(); let a = m * n; let b = m * vec2<f32>(); let c = vec3<f32>() * m;"), ""},
		{"mat mul bad dims", fn("let m = mat2x3<f32>(); let a = m * m;"), "no-matching-overload"},
		{"int mod", fn("let a = 7 % 3; let b = 7.5 % 2.0;"), ""},
		{"div by const zero", fn("let a = 1 / 0;"), "const-div-zero"},
		{"div by zero runtime lhs", fn("var a = 1; let b = a / 0;"), "const-div-zero"},
		{"i32 min div -1", fn("let a = i32(-2147483648) / -1;"), "const-overflow"},
		{"shift", fn("let a = 1u << 3u; let b = 1 << 3; var x = 1u; let c = x >> 2u;"), ""},
		{"shift count too big", fn("let a = 1u << 32u;"), "const-shift"},
		{"shift overflow", fn("let a = (1u << 31u) << 1u;"), "const-overflow"},
		{"shift by i32", fn("var s = 1i; let a = 1u << s;"), "no-matching-overload"},
		{"logical", fn("let a = true && false; let b = true || false; let c = !true; let d = vec2<bool>() | vec2<bool>();"), ""},
		{"logical int", fn("let a = 1 && 2;"), "type-mismatch"},
		{"short circuit skips eval error", fn("let a = false && (10i < i32(5 * 1000 * 1000 * 1000));"), ""},
		{"no short circuit error", fn("let a = true && (10i < i32(5 * 1000 * 1000 * 1000));"), "const-overflow"},
		{"comparison", fn("let a = 1 < 2; let b = vec2<f32>() == vec2<f32>(); let c = true != false;"), ""},
		{"comparison bool order", fn("let a = true < false;"), "no-matching-overload"},
		{"bitwise", fn("let a = 1u & 3u; let b = ~1; let c = 1 ^ 3; let d = 1i | 2;"), ""},
		{"float overflow const", fn("let a = 3.4e38f * 10.0f;"), "const-overflow"},
		{"abstract float huge", fn("const a = 1e308 * 10.0;"), "const-overflow"},
		{"i32 add wraps", fn("const a = 2147483647i + 1i;"), ""},
		{"abstract int overflow", fn("const a = 9223372036854775807 + 1;"), "const-overflow"},
		{"address of var", fn("var a: i32; let p = &a;"), ""},
		{"address of let", fn("let a = 1; let p = &a;"), "address-of"},
		{"address of vec component", fn("var v: vec3<f32>; let p = &v.x;"), "address-of"},
		{"deref non pointer", fn("var a = 1; let b = *a;"), "indirection"},
		{"deref ptr member", "struct S { a: i32 } fn g(p: ptr<function, S>) -> i32 { return (*p).a + p.a; } ", ""},
		{"call arg mismatch", "fn g(a: f32) {} fn f() { g(1u); }", "type-mismatch"},
		{"call arg count", "fn g(a: f32) {} fn f() { g(); }", "arg-count"},
		{"call entry point", "@compute @workgroup_size(1) fn m() {} fn f() { m(); }", "entry-point-call"},
		{"void value", "fn g() {} fn f() { let x = g(); }", "let-type"},
		{"call user pointer", "fn g(p: ptr<function, i32>) { *p = 1; } fn f() { var a = 0; g(&a); }", ""},
		{"call pointer wrong space", "var<private> a: i32; fn g(p: ptr<function, i32>) { } fn f() { g(&a); }", "type-mismatch"},
		{"must_use user", "@must_use fn g() -> i32 { return 1; } fn f() { g(); }", "must-use"},
		{"must_use builtin", fn("abs(1);"), "must-use"},
		{"builtin shadowed", "fn abs(a: i32) -> i32 { return a; } fn f() { let x = abs(1); }", ""},
		{"call non function", fn("var a = 1; a();"), "not-callable"},
		{"builtin targs", fn("let a = abs<f32>(1.0);"), "type-args"},
	})
}

func TestBuiltins(t *testing.T) {
	runCases(t, []checkCase{
		{"numeric", fn("let a = abs(-1); let b = sin(1.0); let c = max(1, 2u); let d = pow(2.0, 3.0); let e = sqrt(vec2<f32>(1.0));"), ""},
		{"int sin", fn("var i = 1; let a = sin(i);"), "no-matching-overload"},
		{"abstract int sin", fn("let a = sin(1);"), ""},
		{"vector builtins", fn("let a = dot(vec3<f32>(), vec3<f32>()); let b = cross(vec3<f32>(), vec3<f32>()); let c = length(vec2<f32>()); let d = normalize(vec4<f32>(1));"), ""},
		{"cross dim", fn("let a = cross(vec2<f32>(), vec2<f32>());"), "no-matching-overload"},
		{"mix scalar", fn("let a = mix(vec3<f32>(), vec3<f32>(), 0.5);"), ""},
		{"select", fn("let a = select(1, 2, true); let b = select(vec2<f32>(), vec2<f32>(1), vec2<bool>());"), ""},
		{"select bad", fn("let a = select(1u, 2i, true);"), "no-matching-overload"},
		{"matrix funcs", fn("let a = determinant(mat3x3<f32>()); let b = transpose(mat2x3<f32>());"), ""},
		{"bits", fn("let a = countOneBits(1u); let b = extractBits(1i, 0u, 4u); let c = insertBits(1u, 2u, 0u, 3u); let d = reverseBits(1);"), ""},
		{"extractBits const range", fn("let a = extractBits(1u, 30u, 4u);"), "const-bits"},
		{"clamp const order", fn("let a = clamp(1, 5, 2);"), "const-clamp"},
		{"frexp modf", fn("let a = frexp(1.5f); let b = a.fract + f32(a.exp); let c = modf(vec2<f32>(1.5)); let d = c.whole;"), ""},
		{"pack", fn("let a = pack4x8unorm(vec4<f32>()); let b = unpack2x16float(1u); let c = pack2x16snorm(vec2<f32>());"), ""},
		{"dot4 ext", fn("let a = dot4U8Packed(1u, 2u); let b = pack4xI8(vec4<i32>());"), ""},
		{"quantize", fn("let a = quantizeToF16(1.0f);"), ""},
		{"arrayLength", "@group(0) @binding(0) var<storage> a: array<i32>; fn f() { let n = arrayLength(&a); }", ""},
		{"arrayLength fixed", "@group(0) @binding(0) var<storage> a: array<i32, 4>; fn f() { let n = arrayLength(&a); }", "type-mismatch"},
		{"atomics", "var<workgroup> a: atomic<u32>; @compute @workgroup_size(1) fn m() { atomicAdd(&a, 1u); atomicStore(&a, 2); let x = atomicLoad(&a); let y = atomicCompareExchangeWeak(&a, 1u, 2u); let z = y.old_value; let w = y.exchanged; atomicMax(&a, 1u); }", ""},
		{"atomic wrong type", "var<workgroup> a: atomic<u32>; fn f() { atomicAdd(&a, 1i); }", "type-mismatch"},
		{"atomic non atomic", "var<workgroup> a: u32; fn f() { atomicAdd(&a, 1u); }", "type-mismatch"},
		{"atomic vertex", "var<workgroup> a: atomic<u32>; @vertex fn v() -> @builtin(position) vec4<f32> { atomicAdd(&a, 1u); return vec4<f32>(); }", "stage-builtin"},
		{"barrier compute", "@compute @workgroup_size(1) fn m() { workgroupBarrier(); storageBarrier(); }", ""},
		{"barrier fragment", "@fragment fn m() { workgroupBarrier(); }", "stage-builtin"},
		{"barrier in helper not called", "fn h() { workgroupBarrier(); }", ""},
		{"workgroupUniformLoad", "var<workgroup> a: i32; @compute @workgroup_size(1) fn m() { let x = workgroupUniformLoad(&a); }", ""},
		{"texture sample", "@group(0) @binding(0) var t: texture_2d<f32>; @group(0) @binding(1) var s: sampler; @fragment fn m() -> @location(0) vec4<f32> { return textureSample(t, s, vec2<f32>(0.5)) + textureSample(t, s, vec2(0.5, 0.5), vec2(1, 2)); }", ""},
		{"texture sample offset range", "@group(0) @binding(0) var t: texture_2d<f32>; @group(0) @binding(1) var s: sampler; @fragment fn m() { _ = textureSample(t, s, vec2<f32>(), vec2(8, 0)); }", "const-arg"},
		{"texture sample offset const", "@group(0) @binding(0) var t: texture_2d<f32>; @group(0) @binding(1) var s: sampler; @fragment fn m() { var o = vec2<i32>(); _ = textureSample(t, s, vec2<f32>(), o); }", "const-arg"},
		{"texture sample compute", "@group(0) @binding(0) var t: texture_2d<f32>; @group(0) @binding(1) var s: sampler; @compute @workgroup_size(1) fn m() { _ = textureSample(t, s, vec2<f32>()); }", "stage-builtin"},
		{"texture level compute", "@group(0) @binding(0) var t: texture_2d<f32>; @group(0) @binding(1) var s: sampler; @compute @workgroup_size(1) fn m() { _ = textureSampleLevel(t, s, vec2<f32>(), 0.0); }", ""},
		{"texture load", "@group(0) @binding(0) var t: texture_2d<f32>; fn f() { let a = textureLoad(t, vec2(1, 2), 0); let b = textureDimensions(t); let c = textureNumLevels(t); }", ""},
		{"texture store", "@group(0) @binding(0) var t: texture_storage_2d<rgba8unorm, write>; fn f() { textureStore(t, vec2(1, 2), vec4<f32>()); }", ""},
		{"texture store wrong value", "@group(0) @binding(0) var t: texture_storage_2d<rgba8uint, write>; fn f() { textureStore(t, vec2(1, 2), vec4<f32>()); }", "no-matching-overload"},
		{"texture load write only", "@group(0) @binding(0) var t: texture_storage_2d<rgba8unorm, write>; fn f() { _ = textureLoad(t, vec2(1, 2)); }", "no-matching-overload"},
		{"texture gather", "@group(0) @binding(0) var t: texture_2d<f32>; @group(0) @binding(1) var s: sampler; fn f() { _ = textureGather(1, t, s, vec2<f32>()); }", ""},
		{"texture gather component range", "@group(0) @binding(0) var t: texture_2d<f32>; @group(0) @binding(1) var s: sampler; fn f() { _ = textureGather(4, t, s, vec2<f32>()); }", "const-arg"},
		{"depth compare", "@group(0) @binding(0) var t: texture_depth_2d; @group(0) @binding(1) var s: sampler_comparison; @fragment fn m() { _ = textureSampleCompare(t, s, vec2<f32>(), 0.5); }", ""},
		{"derivative fragment", "@fragment fn m() { _ = dpdx(1.0f); }", ""},
		{"derivative compute", "@compute @workgroup_size(1) fn m() { _ = dpdx(1.0f); }", "stage-builtin"},
		{"subgroup needs enable", "@compute @workgroup_size(1) fn m() { _ = subgroupAdd(1u); }", "extension-required"},
		{"subgroup enabled", "enable subgroups; @compute @workgroup_size(1) fn m() { _ = subgroupAdd(1u); _ = subgroupBallot(true); _ = subgroupShuffle(1.0f, 2u); }", ""},
		{"subgroup vertex", "enable subgroups; @vertex fn m() -> @builtin(position) vec4<f32> { _ = subgroupAdd(1u); return vec4<f32>(); }", "stage-builtin"},
		{"discard", "@fragment fn m() { discard; }", ""},
		{"discard compute", "@compute @workgroup_size(1) fn m() { discard; }", "stage-builtin"},
		{"const builtin in const", "const a = max(1, 2); var<private> b: array<i32, a>; const c = vec3(1.0, 2.0, 3.0); const d = dot(c, c); const e = length(vec2(3.0, 4.0)); const_assert e == 5.0;", ""},
		{"const eval firstLeadingBit", "var<private> a: array<i32, firstLeadingBit(257)>;", ""},
		{"non const builtin in const", "@group(0) @binding(0) var t: texture_2d<f32>; const a = textureDimensions(t);", "const-init"},
	})
}

func TestStatements(t *testing.T) {
	runCases(t, []checkCase{
		{"if bool", fn("if true {} else if false {} else {}"), ""},
		{"if non bool", fn("if 1 {}"), "type-mismatch"},
		{"while for loop", fn("var i = 0; while i < 3 { i++; } for (var j = 0; j < 3; j++) { if j == 1 { continue; } break; }"), ""},
		{"loop break", fn("loop { break; }"), ""},
		{"loop continuing break if", fn("var i = 0; loop { continuing { i++; break if i > 3; } }"), ""},
		{"loop infinite", fn("loop { }"), "behavior"},
		{"loop discard only", "@fragment fn m() { loop { discard; } }", "behavior"},
		{"break outside", fn("break;"), "break"},
		{"continue outside", fn("continue;"), "continue"},
		{"break in continuing", fn("loop { if true { break; } continuing { break; } }"), "break"},
		{"continue in switch in loop", fn("var a = 1; loop { switch a { case 1 { continue; } default { } } if true { break; } }"), ""},
		{"continue in every switch clause", fn("var a = 1; loop { switch a { default { continue; } } if true { break; } }"), "behavior"},
		{"return in continuing", "fn f() -> i32 { var i = 0; loop { if i > 1 { break; } continuing { i++; return 1; } } return 0; }", "return"},
		{"break in switch", fn("var a = 1; switch a { default { break; } }"), ""},
		{"missing return", "fn f() -> i32 { var a = 0; if a == 42 { return a; } }", "missing-return"},
		{"return all paths", "fn f() -> i32 { var a = 0; if a == 42 { return a; } else { return 1; } }", ""},
		{"unreachable code", "fn f() -> i32 { return 0; var a = 1; return 2; }", ""},
		{"loop return", "fn f() -> i32 { loop { return 1; } }", ""},
		{"return value in void", "fn f() { return 1; }", "return-type"},
		{"return missing value", "fn f() -> i32 { return; }", "return-type"},
		{"return wrong type", "fn f() -> i32 { return 1.5; }", "type-mismatch"},
		{"switch ok", fn("var a = 1; switch a { case 0 { } case 1, 2: { } case 3, default { } }"), ""},
		{"switch no default", fn("var a = 1; switch a { case 0 { } }"), "switch"},
		{"switch two defaults", fn("var a = 1; switch a { default { } case default { } }"), "switch"},
		{"switch duplicate", fn("var a = 1; switch a { case 1 { } case 1 { } default { } }"), "switch"},
		{"switch type mismatch", fn("var a = 1; switch a { case 1u { } default { } }"), "no-matching-overload"},
		{"switch non const", fn("var a = 1; var b = 2; switch a { case b { } default { } }"), "switch"},
		{"switch const selector", fn("const c = 2; var a = 1u; switch a { case c { } default { } }"), ""},
		{"assign to let", fn("let a = 1; a = 2;"), "assign-target"},
		{"assign const", "const a = 1; fn f() { a = 2; }", "assign-target"},
		{"assign param", "fn f(a: i32) { a = 1; }", "assign-target"},
		{"assign readonly storage", "@group(0) @binding(0) var<storage, read> a: i32; fn f() { a = 1; }", "access-mode"},
		{"assign uniform", "@group(0) @binding(0) var<uniform> a: i32; fn f() { a = 1; }", "access-mode"},
		{"assign rw storage", "@group(0) @binding(0) var<storage, read_write> a: i32; fn f() { a = 1; }", ""},
		{"assign component", fn("var v: vec3<f32>; v.x = 1.0; v[1] = 2.0;"), ""},
		{"assign swizzle multi", fn("var v: vec3<f32>; v.xy = vec2<f32>();"), "assign-target"},
		{"compound assign", fn("var a = 1; a += 2; a <<= 1u; a %= 2; var f = 1.0; f *= 2.0;"), ""},
		{"compound mismatch", fn("var a = 1; a += 1.5;"), "no-matching-overload"},
		{"compound shift type", fn("var a = 1u; a <<= 1u; a >>= 1;"), ""},
		{"incdec", fn("var a = 1; a++; a--;"), ""},
		{"incdec float", fn("var a = 1.0; a++;"), "type-mismatch"},
		{"phony", fn("var a = 1; _ = a; _ = 1;"), ""},
		{"phony atomic", "var<workgroup> a: atomic<u32>; fn f() { _ = a; }", "type-mismatch"},
		{"phony ptr", "var<workgroup> a: atomic<u32>; fn f() { _ = &a; }", ""},
		{"local const", fn("const a = 1; const b: f32 = a; let c = b;"), ""},
		{"local const non const init", fn("var a = 1; const b = a;"), "const-init"},
		{"local var space", fn("var<function> a: i32; var<private> b: i32;"), "address-space"},
		{"local var type or init", fn("var a;"), "var-type"},
		{"const_assert", "const a = 1; const_assert a == 1; fn f() { const_assert a < 2; }", ""},
		{"const_assert fails", "const a = 1; const_assert a == 2;", "const-assert"},
		{"const_assert non const", "fn f() { var a = 1; const_assert a == 1; }", "const-assert"},
		{"const_assert non bool", "const_assert 1;", "type-mismatch"},
	})
}

func TestAddressSpacesAndVariables(t *testing.T) {
	runCases(t, []checkCase{
		{"private init const", "const a = 1; var<private> x: i32 = a + 1;", ""},
		{"private init runtime", "var<private> y: i32; var<private> x: i32 = y;", "var-init"},
		{"workgroup init", "var<workgroup> x: i32 = 1;", "var-init"},
		{"storage init", "@group(0) @binding(0) var<storage> x: i32 = 1;", "var-init"},
		{"missing address space", "var x: i32;", "address-space"},
		{"function space at module", "var<function> x: i32;", "address-space"},
		{"workgroup atomic ok", "var<workgroup> x: atomic<i32>;", ""},
		{"private atomic", "var<private> x: atomic<i32>;", "var-type"},
		{"ro storage atomic", "@group(0) @binding(0) var<storage> x: atomic<i32>;", "var-type"},
		{"storage bool", "@group(0) @binding(0) var<storage> x: bool;", "var-type"},
		{"storage rw runtime", "@group(0) @binding(0) var<storage, read_write> x: array<vec4<f32>>;", ""},
		{"uniform runtime array", "@group(0) @binding(0) var<uniform> x: array<vec4<f32>>;", "var-type"},
		{"uniform array stride", "@group(0) @binding(0) var<uniform> x: array<f32, 4>;", "layout"},
		{"uniform array ok", "@group(0) @binding(0) var<uniform> x: array<vec4<f32>, 4>;", ""},
		{"uniform struct gap", "struct S { x: f32 } struct I { a: S, b: f32 } @group(0) @binding(0) var<uniform> i: I;", "layout"},
		{"uniform struct align", "struct S { x: f32 } struct I { a: S, @align(16) b: f32 } @group(0) @binding(0) var<uniform> i: I;", ""},
		{"uniform vec3 f32 ok", "struct S { a: vec3<f32>, b: f32 } @group(0) @binding(0) var<uniform> s: S;", ""},
		{"missing binding", "@group(0) var<uniform> x: vec4<f32>;", "resource-binding"},
		{"binding on private", "@group(0) @binding(0) var<private> x: vec4<f32>;", "attr-placement"},
		{"negative binding", "@group(0) @binding(-1) var<uniform> x: vec4<f32>;", "attr-value"},
		{"duplicate bindings", "@group(0) @binding(0) var<uniform> a: vec4<f32>; @group(0) @binding(0) var<uniform> b: vec4<f32>; @compute @workgroup_size(1) fn m() { _ = a; _ = b; }", "binding-duplicate"},
		{"duplicate bindings unused", "@group(0) @binding(0) var<uniform> a: vec4<f32>; @group(0) @binding(0) var<uniform> b: vec4<f32>; @compute @workgroup_size(1) fn m() { _ = a; }", ""},
		{"duplicate bindings separate entries", "@group(0) @binding(0) var<uniform> a: vec4<f32>; @group(0) @binding(0) var<uniform> b: vec4<f32>; @compute @workgroup_size(1) fn m() { _ = a; } @compute @workgroup_size(1) fn n() { _ = b; }", ""},
		{"workgroup in fragment", "var<workgroup> w: i32; @fragment fn m() { _ = w; }", "stage-access"},
		{"workgroup via helper in vertex", "var<workgroup> w: i32; fn h() { _ = w; } @vertex fn m() -> @builtin(position) vec4<f32> { h(); return vec4<f32>(); }", "stage-access"},
		{"storage rw in vertex", "@group(0) @binding(0) var<storage, read_write> s: i32; @vertex fn m() -> @builtin(position) vec4<f32> { _ = s; return vec4<f32>(); }", "stage-access"},
		{"handle with space", "@group(0) @binding(0) var<private> t: texture_2d<f32>;", "var-type"},
		{"override ok", "@id(0) override a: i32 = 1; override b = a * 2; override c: bool;", ""},
		{"override id dup", "@id(1) override a: i32; @id(1) override b: i32;", "override-id"},
		{"override id range", "@id(65536) override a: i32;", "attr-value"},
		{"override type", "override a: vec2<f32>;", "override-type"},
		{"override needs type", "override a;", "override-type"},
		{"override init runtime", "var<private> p: i32; override a = p;", "override-init"},
		{"const needs const", "override o = 1; const c = o;", "const-init"},
		{"uniform standard layout opt in", "requires uniform_buffer_standard_layout; @group(0) @binding(0) var<uniform> x: array<f32, 4>;", ""},
		{"immediate", "var<immediate> x: vec4<f32>; @compute @workgroup_size(1) fn m() { _ = x; }", ""},
		{"immediate two", "var<immediate> x: vec4<f32>; var<immediate> y: vec4<f32>; @compute @workgroup_size(1) fn m() { _ = x; _ = y; }", "immediate"},
		{"struct layout attrs", "struct S { @size(16) a: f32, @align(16) b: f32 } @group(0) @binding(0) var<storage> s: S; fn f() { _ = s.b; }", ""},
		{"size too small", "struct S { @size(2) a: f32 }", "attr-value"},
		{"align not pow2", "struct S { @align(3) a: f32 }", "attr-value"},
		{"align on var", "@align(4) var<private> a: i32;", "attr-placement"},
	})
}

func TestEntryPoints(t *testing.T) {
	runCases(t, []checkCase{
		{"compute ok", "@compute @workgroup_size(8, 4, 1) fn m(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) i: u32) {}", ""},
		{"compute needs size", "@compute fn m() {}", "workgroup-size"},
		{"size on fragment", "@fragment @workgroup_size(1) fn m() {}", "attr-placement"},
		{"size zero", "@compute @workgroup_size(0) fn m() {}", "workgroup-size"},
		{"size mixed types", "@compute @workgroup_size(1i, 1u) fn m() {}", "workgroup-size"},
		{"size override", "override n = 8u; @compute @workgroup_size(n, 2) fn m() {}", ""},
		{"size runtime", "var<private> p: i32; @compute @workgroup_size(p) fn m() {}", "workgroup-size"},
		{"two stages", "@compute @fragment @workgroup_size(1) fn m() {}", "entry-point"},
		{"compute returns", "@compute @workgroup_size(1) fn m() -> @location(0) vec4<f32> { return vec4<f32>(); }", "entry-point"},
		{"vertex ok", "struct V { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> } @vertex fn m(@builtin(vertex_index) i: u32, @location(0) pos: vec3<f32>) -> V { return V(); }", ""},
		{"vertex without position", "@vertex fn m() -> @location(0) vec4<f32> { return vec4<f32>(); }", "entry-point"},
		{"fragment ok", "@fragment fn m(@builtin(position) p: vec4<f32>, @builtin(front_facing) f: bool, @location(1) @interpolate(flat) i: u32) -> @location(0) vec4<f32> { return p; }", ""},
		{"frag depth output", "struct O { @builtin(frag_depth) d: f32, @location(0) c: vec4<f32> } @fragment fn m() -> O { return O(); }", ""},
		{"builtin wrong stage", "@fragment fn m(@builtin(vertex_index) i: u32) {}", "builtin-stage"},
		{"builtin wrong type", "@compute @workgroup_size(1) fn m(@builtin(global_invocation_id) i: vec3<i32>) {}", "builtin-type"},
		{"builtin duplicate", "@compute @workgroup_size(1) fn m(@builtin(local_invocation_index) a: u32, @builtin(local_invocation_index) b: u32) {}", "builtin-duplicate"},
		{"unknown builtin", "@compute @workgroup_size(1) fn m(@builtin(nope) a: u32) {}", "attr-value"},
		{"location on compute", "@compute @workgroup_size(1) fn m(@location(0) a: f32) {}", "location-stage"},
		{"location duplicate", "@fragment fn m(@location(0) a: f32, @location(0) b: f32) {}", "location-duplicate"},
		{"location struct duplicate", "struct A { @location(0) x: f32, @location(0) y: f32 }", "location-duplicate"},
		{"location bool", "@fragment fn m(@location(0) a: bool) {}", "location-type"},
		{"location on struct", "struct D { x: vec4<f32> } @fragment fn m(@location(0) d: D) {}", "io-struct"},
		{"nested io struct", "struct B { @location(0) x: f32 } struct C { b: B } @fragment fn m(c: C) {}", "io-struct"},
		{"missing io attr", "@fragment fn m(a: f32) {}", "io-missing"},
		{"integer needs flat", "@fragment fn m(@location(0) a: u32) {}", "interpolate"},
		{"interpolate no location", "@fragment fn m(@builtin(front_facing) @interpolate(flat) a: bool) {}", "attr-placement"},
		{"interpolate flat sampling", "@fragment fn m(@location(0) @interpolate(flat, center) a: f32) {}", "attr-value"},
		{"interpolate linear first", "@fragment fn m(@location(0) @interpolate(linear, first) a: f32) {}", "attr-value"},
		{"invariant position", "@vertex fn m() -> @builtin(position) @invariant vec4<f32> { return vec4<f32>(); }", ""},
		{"invariant other", "@fragment fn m(@location(0) @invariant a: f32) {}", "attr-placement"},
		{"blend_src needs ext", "struct O { @location(0) @blend_src(0) a: vec4<f32>, @location(0) @blend_src(1) b: vec4<f32> } @fragment fn m() -> O { return O(); }", "extension-required"},
		{"blend_src ok", "enable dual_source_blending; struct O { @location(0) @blend_src(0) a: vec4<f32>, @location(0) @blend_src(1) b: vec4<f32> } @fragment fn m() -> O { return O(); }", ""},
		{"clip distances", "enable clip_distances; struct O { @builtin(position) p: vec4<f32>, @builtin(clip_distances) c: array<f32, 4> } @vertex fn m() -> O { return O(); }", ""},
		{"clip distances too many", "enable clip_distances; struct O { @builtin(position) p: vec4<f32>, @builtin(clip_distances) c: array<f32, 9> } @vertex fn m() -> O { return O(); }", "builtin-type"},
		{"clip distances needs enable", "struct O { @builtin(position) p: vec4<f32>, @builtin(clip_distances) c: array<f32, 4> } @vertex fn m() -> O { return O(); }", "extension-required"},
		{"attr on plain fn", "@location(0) fn f() {}", "attr-placement"},
		{"io attr on plain fn param", "fn f(@location(0) a: f32) {}", "attr-placement"},
		{"unknown attribute", "@nope fn f() {}", "unknown-attribute"},
		{"duplicate attribute", "@compute @compute @workgroup_size(1) fn m() {}", "attr-duplicate"},
		{"const attribute", "@const fn f() {}", "attr-placement"},
		{"entry info", "@compute @workgroup_size(4) fn m() {}", ""},
	})
}

func TestDirectives(t *testing.T) {
	runCases(t, []checkCase{
		{"enable unknown", "enable nope;", "unknown-extension"},
		{"requires unknown", "requires nope;", "unknown-extension"},
		{"requires known", "requires readonly_and_readwrite_storage_textures; requires pointer_composite_access;", ""},
		{"diagnostic ok", "diagnostic(off, derivative_uniformity); diagnostic(warning, chromium.unreachable_code);", ""},
		{"diagnostic bad severity", "diagnostic(loud, derivative_uniformity);", "diagnostic"},
		{"diagnostic conflict", "diagnostic(off, derivative_uniformity); diagnostic(error, derivative_uniformity);", "diagnostic-conflict"},
		{"range diagnostic", "fn f() { if true @diagnostic(off, derivative_uniformity) { } }", ""},
		{"storage texture rw", "@group(0) @binding(0) var t: texture_storage_2d<r32float, read_write>; fn f() { _ = textureLoad(t, vec2(0, 0)); textureStore(t, vec2(0, 0), vec4<f32>()); }", ""},
		{"tier1 format", "@group(0) @binding(0) var t: texture_storage_2d<rg8unorm, write>;", ""},
		{"buffer_view unsupported", "requires buffer_view;", "unknown-extension"},
	})
}

func TestPointersAndAccess(t *testing.T) {
	runCases(t, []checkCase{
		{"ptr member access", "struct S { a: i32, b: array<i32, 2> } fn g(p: ptr<function, S>) { (*p).a = 1; p.a = 2; p.b[1] = 3; }", ""},
		{"ptr in storage param", "@group(0) @binding(0) var<storage, read_write> a: i32; fn g(p: ptr<storage, i32, read_write>) { *p = 1; } fn f() { g(&a); }", ""},
		{"ptr read through write only", "fn f() { var a: i32; let p = &a; let b = *p + 1; }", ""},
		{"ptr to let", "fn f() { var a = array<i32, 2>(); let p = &a[1]; *p = 3; }", ""},
		{"function returns pointer", "fn f(p: ptr<function, i32>) -> ptr<function, i32> { return p; }", "return-type"},
		{"param texture", "fn g(t: texture_2d<f32>, s: sampler) -> vec4<f32> { return textureSampleLevel(t, s, vec2<f32>(), 0.0); }", ""},
		{"param atomic", "fn g(a: atomic<u32>) {}", "param-type"},
		{"param runtime array", "fn g(a: array<i32>) {}", "param-type"},
		{"texture let", "@group(0) @binding(0) var t: texture_2d<f32>; fn f() { let x = t; }", ""},
	})
}

// TestInfo checks what a back end consumes.
func TestInfo(t *testing.T) {
	src := `
struct S { a: f32, b: vec3<f32>, c: array<vec2<f32>, 3>, @size(16) d: i32 }
const N = 4;
override K: u32 = 8;
@group(1) @binding(2) var<storage, read_write> buf: array<S>;
@group(0) @binding(0) var<uniform> u: vec4<f32>;
var<workgroup> wg: array<f32, N>;
fn helper(i: u32) -> f32 { return buf[i].a + u.x; }
@compute @workgroup_size(N, 2) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  wg[0] = helper(gid.x);
  let x = 1 + 2;
}
`
	info, errs := checkSrc(t, src)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	if len(info.EntryPoints) != 1 {
		t.Fatalf("entry points: %d", len(info.EntryPoints))
	}
	ep := info.EntryPoints[0]
	if ep.Stage != StageCompute || ep.WorkgroupSize != [3]int64{4, 2, 1} {
		t.Errorf("stage/size: %v %v", ep.Stage, ep.WorkgroupSize)
	}
	if len(ep.Resources) != 2 || ep.Resources[0].Name != "u" || ep.Resources[1].Name != "buf" {
		t.Errorf("resources: %v", ep.Resources)
	}
	if len(ep.Inputs) != 1 || ep.Inputs[0].Builtin != "global_invocation_id" {
		t.Errorf("inputs: %+v", ep.Inputs)
	}
	if len(ep.Functions) != 2 {
		t.Errorf("functions: %d", len(ep.Functions))
	}
	st := info.Structs[0]
	wantOff := []int64{0, 16, 32, 56}
	for i, m := range st.Members {
		if m.Offset != wantOff[i] {
			t.Errorf("member %s offset %d want %d", m.Name, m.Offset, wantOff[i])
		}
	}
	if st.Align != 16 || st.Size != 80 {
		t.Errorf("struct align/size %d/%d want 16/80", st.Align, st.Size)
	}
	var sawConst, sawObj bool
	for e, c := range info.Consts {
		if lit, ok := e.(*IntLit); ok && lit.Text == "4" && c.Int == 4 {
			sawConst = true
		}
	}
	for id, o := range info.Objects {
		if id.Name == "helper" {
			if f, ok := o.(*FuncObj); ok && f.Name == "helper" {
				sawObj = true
			}
		}
	}
	if !sawConst || !sawObj {
		t.Errorf("const %v obj %v", sawConst, sawObj)
	}
	for e, ty := range info.Types {
		if id, ok := e.(*Ident); ok && id.Name == "buf" {
			if r, ok := ty.(*RefType); !ok || r.AS != ASStorage || r.Access != AccessReadWrite {
				t.Errorf("buf type %v", ty)
			}
		}
	}
}

func TestLayouts(t *testing.T) {
	src := `
struct A { u: f32, v: f32, w: vec2<f32>, x: f32 }
struct B { a: vec2<f32>, b: vec3<f32>, c: f32, d: f32, e: A, f: vec3<f32>, g: array<A, 3>, h: i32 }
@group(0) @binding(0) var<storage, read_write> sb: B;
fn f() { _ = sb.h; }
`
	info, errs := checkSrc(t, src)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	a, b := info.Structs[0], info.Structs[1]
	if a.Align != 8 || a.Size != 24 {
		t.Errorf("A: %d/%d", a.Align, a.Size)
	}
	// spec example "Layout of structures using implicit member sizes and alignments"
	wantOff := []int64{0, 16, 28, 32, 40, 64, 80, 152}
	for i, m := range b.Members {
		if m.Offset != wantOff[i] {
			t.Errorf("B.%s offset %d want %d", m.Name, m.Offset, wantOff[i])
		}
	}
	if b.Align != 16 || b.Size != 160 {
		t.Errorf("B: %d/%d want 16/160", b.Align, b.Size)
	}
}

func TestStringAndSame(t *testing.T) {
	cases := map[Type]string{
		vec(3, F32): "vec3<f32>", &MatType{2, 3, F16}: "mat2x3<f16>",
		&ArrayType{Elem: I32, Count: 4}: "array<i32, 4>", &ArrayType{Elem: I32, Runtime: true}: "array<i32>",
		&PtrType{AS: ASStorage, Elem: U32, Access: AccessRead}: "ptr<storage, u32, read>",
		&AtomicType{U32}: "atomic<u32>", AbstractFloat: "AbstractFloat",
		&TextureType{Kind: TexSampled, Dim: Dim2D, Arrayed: true, Sampled: F32}: "texture_2d_array<f32>",
	}
	for ty, want := range cases {
		if ty.String() != want {
			t.Errorf("%v: %q want %q", ty, ty.String(), want)
		}
	}
	if !Same(&ArrayType{Elem: I32, Count: 4}, &ArrayType{Elem: I32, Count: 4}) || Same(vec(2, F32), vec(3, F32)) {
		t.Error("Same")
	}
}

// ---- resource bounds ----

func TestAdversarialInputs(t *testing.T) {
	deepArr := strings.Repeat("array<", 300) + "i32, 2" + strings.Repeat(">", 300)
	var manyAliases strings.Builder
	manyAliases.WriteString("alias A0 = i32;\n")
	for i := 1; i < 5000; i++ {
		fmt.Fprintf(&manyAliases, "alias A%d = A%d;\n", i, i-1)
	}
	var reversed strings.Builder
	for i := 4999; i >= 1; i-- {
		fmt.Fprintf(&reversed, "alias A%d = A%d;\n", i, i-1)
	}
	reversed.WriteString("alias A0 = i32;\n")
	var cyc strings.Builder
	for i := 0; i < 3000; i++ {
		fmt.Fprintf(&cyc, "const c%d = c%d + 1;\n", i, (i+1)%3000)
	}
	var nestedStructs strings.Builder
	nestedStructs.WriteString("struct S0 { a: f32 }\n")
	for i := 1; i < 100; i++ {
		fmt.Fprintf(&nestedStructs, "struct S%d { a: S%d }\n", i, i-1)
	}
	var addChain strings.Builder
	addChain.WriteString("const x = 1")
	for i := 0; i < 5000; i++ {
		addChain.WriteString(" + 1")
	}
	addChain.WriteString(";")
	cases := []struct {
		name, src string
		wantErr   bool
	}{
		{"huge array type", "var<private> a: array<i32, 1000000000>;", false},
		{"huge array of arrays", "var<private> a: array<array<array<f32, 100000>, 100000>, 100000>;", false},
		{"huge const zero array", "const a = array<i32, 1000000000>(); const b = a[999999999];", false},
		{"huge const array index", "const a = array<i32, 1000000000>(); const b = a[1000000000];", true},
		{"overflowing array count", "var<private> a: array<i32, 9223372036854775807>;", true},
		{"deep nested array types", "var<private> a: " + deepArr + ";", true},
		{"alias chain forward", manyAliases.String(), false},
		{"alias chain reversed", reversed.String(), true},
		{"cyclic consts", cyc.String(), true},
		{"recursive struct", "struct S { a: array<S, 2> }", true},
		{"deep struct nesting", nestedStructs.String(), true},
		{"cyclic alias struct", "struct S { a: A } alias A = S;", true},
		{"long add chain", addChain.String(), true},
		{"giant array ctor", "const a = array<i32, 1000000000>(1, 2);", true},
		{"huge workgroup", "@compute @workgroup_size(2147483647, 2147483647, 2147483647) fn m() {}", true},
		{"shift const", "const a = 1 << 63;", true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			done := make(chan error, 1)
			go func() { done <- Validate(tc.src) }()
			select {
			case err := <-done:
				if (err != nil) != tc.wantErr {
					msg := ""
					if err != nil {
						msg = err.Error()
						if len(msg) > 300 {
							msg = msg[:300]
						}
					}
					t.Fatalf("wantErr=%v got %q", tc.wantErr, msg)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("did not terminate")
			}
		})
	}
}

func TestErrorsArePositioned(t *testing.T) {
	_, errs := checkSrc(t, "fn f() {\n  let a = b;\n}\n")
	if len(errs) != 1 {
		t.Fatalf("errs: %v", errs)
	}
	e := errs[0]
	if e.Pos.Line != 2 || e.Pos.Col != 11 || e.Rule != "undeclared-ident" || e.Name != "t.wgsl" {
		t.Errorf("%+v", e)
	}
}

func TestValidateAPI(t *testing.T) {
	if err := Validate("fn f() {}"); err != nil {
		t.Fatal(err)
	}
	if err := Validate("fn f() { let a = 1u + 1i; }"); err == nil {
		t.Fatal("want an error")
	}
	if err := Validate("fn f( {"); err == nil {
		t.Fatal("want a syntax error")
	}
	m, _ := Parse("fn f() { let a = 1.5 + 1; }")
	info, errs := Check(m)
	if len(errs) != 0 || len(info.Types) == 0 {
		t.Fatal("Check without Source")
	}
	_, errs = Check(mustParse(t, "fn f() { x; }"))
	_ = errs
}

func mustParse(t *testing.T, s string) *Module {
	m, err := Parse(s)
	if err != nil {
		t.Skip("not parseable")
	}
	return m
}
