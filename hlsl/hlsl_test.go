package hlsl

import (
	_ "embed"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/go-gfx/wgsl"
)

//go:embed testdata/kernel.wgsl
var kernelSrc string

func emit(t *testing.T, src string, entries ...string) (string, error) {
	t.Helper()
	m, err := wgsl.Parse(src)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		t.Fatalf("check: %v", errs)
	}
	return Module(m, info, entries...)
}

func TestKernel(t *testing.T) {
	out, err := emit(t, kernelSrc, "raster")
	if err != nil {
		t.Fatal(err)
	}
	if p := os.Getenv("HLSL_DUMP"); p != "" { // read the emitted kernel: HLSL_DUMP=/path go test ./hlsl
		if err := os.WriteFile(p, []byte(out), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, want := range []string{
		"[numthreads(16, 8, 1)]",
		"void raster(uint3 gid : SV_DispatchThreadID)",
		"StructuredBuffer<Inst> insts : register(t0);",
		"StructuredBuffer<uint> bins : register(t1);",
		"StructuredBuffer<uint> arena : register(t2);",
		"RWStructuredBuffer<uint> px : register(u3);",
		"cbuffer wgsl_cb4 : register(b4) { Frame f; };",
		"precise float d = wgsl_length(float2(",
		"wgsl_pf(",
		"wgsl_f2u(",
		"[loop] for (uint k = 0u; (k < count); k++)",
		"wgsl_add(s.x, rad)", // i32 wraps
		"((float3)(0.0f))",   // HLSL has no float3(x)
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the emitted kernel lacks %q", want)
		}
	}
	// Callees are defined before their callers.
	at := func(s string) int { return strings.Index(out, s) }
	if !(at("byteAt(uint i)") < at("void raster") && at("float3 cornerCenter(") < at("float coverage(") && at("float coverage(") < at("void raster")) {
		t.Error("functions are not in callee-first order")
	}
	again, err := emit(t, kernelSrc, "raster")
	if err != nil || again != out {
		t.Error("the emitter is not deterministic")
	}
}

func TestRefusals(t *testing.T) {
	const entry = "@compute @workgroup_size(8) fn main_(@builtin(global_invocation_id) g: vec3<u32>) {"
	cases := []struct{ name, src, want string }{
		{"private variable", "var<private> a: u32;\n" + entry + " a = 1u; }", "private variable"},
		{"group 1", "@group(1) @binding(0) var<storage, read_write> o: array<u32>;\n" + entry + " o[0] = 1u; }", "group 1"},
		{"uav register", "@group(0) @binding(8) var<storage, read_write> o: array<u32>;\n" + entry + " o[0] = 1u; }", "register u8"},
		{"cbuffer register", "@group(0) @binding(14) var<uniform> u: vec4<u32>;\n@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n" + entry + " o[0] = u.x; }", "register b14"},
		{"srv register", "@group(0) @binding(128) var<storage, read> i: array<u32>;\n@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n" + entry + " o[0] = i[0]; }", "register t128"},
		{"fixed array", "@group(0) @binding(0) var<storage, read_write> o: array<u32, 4>;\n" + entry + " o[0] = 1u; }", "fixed-size array"},
		{"storage struct", "struct S { a: u32 }\n@group(0) @binding(0) var<storage, read_write> o: S;\n" + entry + " o.a = 1u; }", "not an array"},
		{"uniform array", "@group(0) @binding(0) var<uniform> u: array<vec4<u32>, 4>;\n@group(0) @binding(1) var<storage, read_write> o: array<u32>;\n" + entry + " o[0] = u[0].x; }", "uniform array"},
		{"vec3 in storage", "struct S { a: u32, v: vec3<f32> }\n@group(0) @binding(0) var<storage, read> i: array<S>;\n@group(0) @binding(1) var<storage, read_write> o: array<f32>;\n" + entry + " o[0] = i[0].v.x; }", "offset 16 in WGSL and 4 in a StructuredBuffer"},
		{"vec2 after u32", "struct S { a: u32, v: vec2<u32> }\n@group(0) @binding(0) var<storage, read> i: array<S>;\n@group(0) @binding(1) var<storage, read_write> o: array<u32>;\n" + entry + " o[0] = i[0].v.x; }", "StructuredBuffer"},
		{"workgroup too big", "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n@compute @workgroup_size(32, 32, 2) fn big(@builtin(global_invocation_id) g: vec3<u32>) { o[0] = 1u; }", "beyond cs_5_0"},
		{"other builtin", "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n@compute @workgroup_size(8) fn k(@builtin(local_invocation_index) i: u32) { o[0] = i; }", "not a supported @builtin"},
		{"integer abs", "@group(0) @binding(0) var<storage, read_write> o: array<i32>;\n" + entry + " o[0] = abs(i32(g.x)); }", "integer abs"},
		{"float rem", "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n" + entry + " o[0] = f32(g.x) % 2.0; }", "floating-point %"},
		{"short circuit with a call", "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\nfn p(x: u32) -> bool { o[1] = x; return x > 1u; }\n" + entry + " if (g.x > 0u && p(g.x)) { o[0] = 1u; } }", "does not short-circuit"},
		{"f32 vector to integer", "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n" + entry + " let v = vec2<u32>(vec2<f32>(f32(g.x), 1.5)); o[0] = v.x; }", "f32 vector to an integer vector"},
		{"reserved function name", "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\nfn line(x: u32) -> u32 { return x; }\n" + entry + " o[0] = line(1u); }", "reserved in HLSL"},
		{"vector type name", "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n" + entry + " let float3 = 1u; o[0] = float3; }", "reserved in HLSL"},
		{"intrinsic name", "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\nfn mad(x: u32) -> u32 { return x; }\n" + entry + " o[0] = mad(1u); }", "reserved in HLSL"},
		{"wgsl_ prefix", "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\nfn wgsl_add(x: u32) -> u32 { return x; }\n" + entry + " o[0] = wgsl_add(1u); }", "reserved in HLSL"},
		{"switch", "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n" + entry + " switch g.x { case 0u: { o[0] = 1u; } default: { o[0] = 2u; } } }", "not supported yet"},
		{"bool vector", "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n" + entry + " let b = vec2<bool>(true, false); o[0] = u32(b.x); }", "vec2<bool>"},
		{"select with a vector condition", "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n" + entry + " let r = select(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0), vec2<bool>(true, false)); o[0] = r.x; }", "not supported yet"},
		{"struct constructor", "struct S { a: u32 }\n@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n" + entry + " let s = S(1u); o[0] = s.a; }", "not supported yet"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m, err := wgsl.Parse(c.src)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			info, errs := wgsl.Check(m)
			if len(errs) > 0 {
				t.Skipf("the checker refuses it first: %v", errs[0])
			}
			name := "main_"
			for _, f := range info.Funcs {
				if f.Entry != nil {
					name = f.Name
				}
			}
			_, err = Module(m, info, name)
			var he *Error
			if !errors.As(err, &he) || !strings.Contains(he.Error(), c.want) {
				t.Fatalf("got %v, want an *hlsl.Error containing %q", err, c.want)
			}
		})
	}
}

func TestModuleArguments(t *testing.T) {
	const src = "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\nfn helper() -> u32 { return 1u; }\n@compute @workgroup_size(8) fn k(@builtin(global_invocation_id) g: vec3<u32>) { o[g.x] = helper(); }\n@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"
	m, _ := wgsl.Parse(src)
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	if _, err := Module(m, info); err == nil {
		t.Error("no entry point named")
	}
	if _, err := Module(m, info, "nope"); err == nil {
		t.Error("an unknown function")
	}
	var he *Error
	if _, err := Module(m, info, "helper"); !errors.As(err, &he) || !strings.Contains(he.Error(), "not an entry point") {
		t.Errorf("a plain function: %v", err)
	}
	if _, err := Module(m, info, "vs"); !errors.As(err, &he) || !strings.Contains(he.Error(), "only compute") {
		t.Errorf("a vertex shader: %v", err)
	}
	out, err := Module(m, info, "k")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "vs(") {
		t.Error("an entry point that was not asked for was emitted")
	}
}

func TestIntegerAndFloatRules(t *testing.T) {
	const src = `
@group(0) @binding(0) var<storage, read> inp: array<i32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(8)
fn k(@builtin(global_invocation_id) g: vec3<u32>) {
  let a = inp[g.x];
  let b = inp[g.x + 1u];
  var i = a + b * 3 - 1;
  i += 2;
  i++;
  let q = a / b + a % b + (a << u32(b)) + (a >> u32(b)) + -a;
  let x = f32(q) * 0.5 + f32(a);
  var y = 2.0;
  y *= x;
  outp[g.x] = y / 3.0 + f32(u32(x)) + f32(i32(x)) + select(0.0, 1.0, x > 0.0);
}
`
	out, err := emit(t, src, "k")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"wgsl_add(wgsl_sub(a, 0)", "wgsl_mul(b, 3)", "i = wgsl_add(i, 1)", "wgsl_div(a, b)", "wgsl_rem(a, b)",
		"wgsl_shl(a, asuint(b))", "wgsl_shr(a, asuint(b))", "wgsl_neg(a)",
		"precise float x = wgsl_pf(wgsl_pf(float(q) * 0.5f) + float(a))",
		"precise float y = 2.0f", "y = wgsl_pf(y * x)", "wgsl_f2u(x)", "wgsl_f2i(x)", "((x > 0.0f) ? 1.0f : 0.0f)",
	} {
		if want == "wgsl_add(wgsl_sub(a, 0)" {
			want = "wgsl_sub(wgsl_add(a, wgsl_mul(b, 3)), 1)"
		}
		if !strings.Contains(out, want) {
			t.Errorf("the emitted code lacks %q", want)
		}
	}
}
