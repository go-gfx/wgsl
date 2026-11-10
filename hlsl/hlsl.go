// Package hlsl emits High-Level Shading Language (Shader Model 5.0, to be
// compiled for cs_5_0 by D3DCompile) for the compute entry points of a checked
// WGSL module.
//
// Module emits compute entry points with the storage and uniform buffers they
// use, loops, runtime-sized array indexing and f32 to integer conversion
// (docs/HLSL.md). Anything else is refused with its position as a *Error, never
// emitted approximately.
//
// Every rule that makes the emitted code compute what WGSL defines, rather
// than what HLSL's defaults allow, is in docs/HLSL.md with its source:
//   - every float +, -, * and / is assigned to a `precise` variable, so no
//     mul+add is fused into a mad, no operation is reordered and NaN and INF
//     are respected;
//   - signed +, -, * and << go through uint, so i32 wraps;
//   - / and % follow WGSL's definition for a zero divisor and MIN / -1;
//   - shift counts are taken modulo 32 and >> on i32 is arithmetic;
//   - f32 to integer saturates where WGSL says, and a NaN gives 0.
package hlsl

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/go-gfx/wgsl"
)

// Preamble opens every emitted file: the helpers the WGSL rules need.
const Preamble = `// Emitted by go-gfx/wgsl/hlsl for Shader Model 5.0 (cs_5_0). Do not edit.
// Compile with D3DCOMPILE_IEEE_STRICTNESS (docs/HLSL.md, rule 1).

// int division by a non-constant warns (X3556); wgsl_div guards its divisor.
#pragma warning(disable: 3556)

// Rule 1: a precise variable restricts every operation that contributes to the
// value assigned to it. Each float +, -, * and / is passed through one, so a
// mul and an add are never fused into a mad and nothing is reassociated.
float wgsl_pf(float x) { precise float r = x; return r; }
float2 wgsl_pf(float2 x) { precise float2 r = x; return r; }
float3 wgsl_pf(float3 x) { precise float3 r = x; return r; }
float4 wgsl_pf(float4 x) { precise float4 r = x; return r; }

static const int wgsl_int_min = -2147483647 - 1;

// WGSL i32 arithmetic wraps (two's complement), through uint.
int wgsl_add(int a, int b) { return asint(asuint(a) + asuint(b)); }
int wgsl_sub(int a, int b) { return asint(asuint(a) - asuint(b)); }
int wgsl_mul(int a, int b) { return asint(asuint(a) * asuint(b)); }
int wgsl_neg(int a) { return asint(0u - asuint(a)); }
// WGSL division and remainder: e2 == 0, and MIN / -1, are defined.
int wgsl_div(int a, int b) {
  bool special = b == 0 || (a == wgsl_int_min && b == -1);
  return a / (special ? 1 : b);
}
int wgsl_rem(int a, int b) {
  bool special = b == 0 || (a == wgsl_int_min && b == -1);
  int d = special ? 1 : b;
  return special ? 0 : wgsl_sub(a, wgsl_mul(a / d, d));
}
uint wgsl_udiv(uint a, uint b) { return a / (b == 0u ? 1u : b); }
uint wgsl_urem(uint a, uint b) { return b == 0u ? 0u : a % (b == 0u ? 1u : b); }
// WGSL shifts: the count modulo 32; << discards high bits; >> on i32 is
// arithmetic.
int wgsl_shl(int a, uint s) { return asint(asuint(a) << (s & 31u)); }
int wgsl_shr(int a, uint s) {
  uint u = asuint(a) >> (s & 31u);
  return asint(a < 0 ? (u | ~(0xFFFFFFFFu >> (s & 31u))) : u);
}
// WGSL f32 to integer: truncate, then the nearest value of the integer type
// that is also exactly representable in f32 (u32 saturates at 4294967040, i32
// at 2147483520); a NaN gives an indeterminate value, here 0.
uint wgsl_f2u(float v) { return v >= 4294967040.0f ? 4294967040u : (v > 0.0f ? uint(v) : 0u); }
int wgsl_f2i(float v) { return v >= 2147483520.0f ? 2147483520 : (v <= -2147483648.0f ? wgsl_int_min : (v == v ? int(v) : 0)); }
// length: the explicit sum of squares in WGSL's order, each step precise.
float wgsl_length(float2 v) {
  precise float xx = v.x * v.x;
  precise float yy = v.y * v.y;
  precise float s = xx + yy;
  return sqrt(s);
}
float wgsl_length(float3 v) {
  precise float xx = v.x * v.x;
  precise float yy = v.y * v.y;
  precise float zz = v.z * v.z;
  precise float s0 = xx + yy;
  precise float s = s0 + zz;
  return sqrt(s);
}
float wgsl_length(float4 v) {
  precise float xx = v.x * v.x;
  precise float yy = v.y * v.y;
  precise float zz = v.z * v.z;
  precise float ww = v.w * v.w;
  precise float s0 = xx + yy;
  precise float s1 = s0 + zz;
  precise float s = s1 + ww;
  return sqrt(s);
}
`

// Error is a construct the emitter refuses, with where it is.
type Error struct {
	Offset int
	What   string
}

func (e *Error) Error() string { return fmt.Sprintf("hlsl: offset %d: %s", e.Offset, e.What) }

// Module emits Preamble, the structures the named compute entry points use,
// the buffers they reach, the functions they call (callees first: HLSL, like
// C, needs a function declared before its use) and each entry point as a
// `[numthreads]` function, in that order. The module-scope variables the entry
// points reach are storage and uniform buffers of group 0: a `read` storage
// array becomes a StructuredBuffer at register t<binding>, a `read_write` one
// an RWStructuredBuffer at u<binding>, a uniform a cbuffer at b<binding>. A
// @builtin(global_invocation_id) parameter becomes SV_DispatchThreadID.
// Everything else is refused (docs/HLSL.md). m must be the module info was
// checked from.
func Module(m *wgsl.Module, info *wgsl.Info, entries ...string) (string, error) {
	e := &emitter{info: info, need: map[*wgsl.FuncObj]bool{}}
	byName := map[string]*wgsl.FuncObj{}
	for _, f := range info.Funcs {
		byName[f.Name] = f
	}
	if len(entries) == 0 {
		return "", fmt.Errorf("hlsl: no entry point named")
	}
	for _, n := range entries {
		f, ok := byName[n]
		if !ok {
			return "", fmt.Errorf("hlsl: no function %q in the module", n)
		}
		if f.Entry == nil {
			return "", &Error{f.Decl.Start, "function " + f.Name + " is not an entry point"}
		}
		if f.Entry.Stage != wgsl.StageCompute {
			return "", &Error{f.Decl.Start, "entry point " + f.Name + " is a " + f.Entry.Stage.String() + " shader; only compute is supported"}
		}
		if f.Entry.WorkgroupSizeExprs != [3]wgsl.Expr{} {
			return "", &Error{f.Decl.Start, "entry point " + f.Name + " has an override-expression workgroup size"}
		}
		ws := f.Entry.WorkgroupSize
		if ws[0] > 1024 || ws[1] > 1024 || ws[2] > 64 || ws[0]*ws[1]*ws[2] > 1024 {
			return "", &Error{f.Decl.Start, fmt.Sprintf("@workgroup_size(%d, %d, %d) is beyond cs_5_0 (1024 threads, z at most 64)", ws[0], ws[1], ws[2])}
		}
		e.entries = append(e.entries, f)
		if err := e.require(f); err != nil {
			return "", err
		}
	}
	var b strings.Builder
	b.WriteString(Preamble)
	for _, s := range info.Structs {
		if s.Predeclared || !e.structs[s] {
			continue
		}
		if err := e.emitStruct(&b, s); err != nil {
			return "", err
		}
	}
	vars := e.resources()
	if len(vars) > 0 {
		b.WriteString("\n")
	}
	for _, v := range vars {
		d, err := e.declareBuffer(v)
		if err != nil {
			return "", err
		}
		b.WriteString(d + "\n")
	}
	// Callees first.
	var order []*wgsl.FuncObj
	seen := map[*wgsl.FuncObj]bool{}
	var visit func(f *wgsl.FuncObj)
	visit = func(f *wgsl.FuncObj) {
		if seen[f] {
			return
		}
		seen[f] = true
		for _, c := range e.callees(f) {
			visit(c)
		}
		order = append(order, f)
	}
	for _, f := range info.Funcs {
		if e.need[f] && f.Entry == nil {
			visit(f)
		}
	}
	for _, f := range e.entries {
		visit(f)
	}
	for _, f := range order {
		if err := e.emitFunc(&b, f); err != nil {
			return "", err
		}
	}
	return b.String(), nil
}

type emitter struct {
	info    *wgsl.Info
	entries []*wgsl.FuncObj
	need    map[*wgsl.FuncObj]bool
	structs map[*wgsl.StructType]bool
	host    map[*wgsl.StructType]string // structures that live in a buffer: "structured" or "cbuffer"
	vars    map[*wgsl.VarObj]bool
	b       *strings.Builder
	depth   int
}

// callees lists the user functions f calls, from its body.
func (e *emitter) callees(f *wgsl.FuncObj) []*wgsl.FuncObj {
	var out []*wgsl.FuncObj
	walkStmt(f.Decl.Body, func(x wgsl.Expr) {
		if c, ok := x.(*wgsl.Call); ok {
			if g, ok := e.info.Objects[c.Fn].(*wgsl.FuncObj); ok && !slices.Contains(out, g) {
				out = append(out, g)
			}
		}
	})
	return out
}

// require marks f and its callees for emission, refusing what is out of scope
// at the level of the whole function.
func (e *emitter) require(f *wgsl.FuncObj) error {
	if e.need[f] {
		return nil
	}
	if f.Entry != nil && !slices.Contains(e.entries, f) {
		return &Error{f.Decl.Start, "function " + f.Name + " is an entry point"}
	}
	e.need[f] = true
	if f.Entry == nil {
		for _, p := range f.Params {
			if err := e.useType(p.Type, p.Decl.Start); err != nil {
				return err
			}
		}
	}
	if _, void := f.Ret.(*wgsl.VoidType); !void {
		if err := e.useType(f.Ret, f.Decl.Start); err != nil {
			return err
		}
	}
	var firstErr error
	walkStmt(f.Decl.Body, func(x wgsl.Expr) {
		if id, ok := x.(*wgsl.Ident); ok && firstErr == nil {
			if v, ok := e.info.Objects[id].(*wgsl.VarObj); ok && v.Global {
				if err := e.useBuffer(v); err != nil {
					firstErr = err
				}
			}
		}
	})
	if firstErr != nil {
		return firstErr
	}
	for _, c := range e.callees(f) {
		if err := e.require(c); err != nil {
			return err
		}
	}
	return nil
}

// walkStmt calls fn on every expression under s.
func walkStmt(s wgsl.Stmt, fn func(wgsl.Expr)) {
	switch s := s.(type) {
	case *wgsl.Block:
		for _, t := range s.Stmts {
			walkStmt(t, fn)
		}
	case *wgsl.If:
		walkExpr(s.Cond, fn)
		walkStmt(s.Then, fn)
		if s.Else != nil {
			walkStmt(s.Else, fn)
		}
	case *wgsl.Return:
		walkExpr(s.X, fn)
	case *wgsl.Assign:
		walkExpr(s.LHS, fn)
		walkExpr(s.RHS, fn)
	case *wgsl.DeclStmt:
		switch d := s.Decl.(type) {
		case *wgsl.Value:
			walkExpr(d.Init, fn)
		case *wgsl.Var:
			walkExpr(d.Init, fn)
		}
	case *wgsl.CallStmt:
		walkExpr(s.Call, fn)
	case *wgsl.For:
		if s.Init != nil {
			walkStmt(s.Init, fn)
		}
		walkExpr(s.Cond, fn)
		if s.Update != nil {
			walkStmt(s.Update, fn)
		}
		walkStmt(s.Body, fn)
	case *wgsl.While:
		walkExpr(s.Cond, fn)
		walkStmt(s.Body, fn)
	case *wgsl.Loop:
		for _, t := range s.Body {
			walkStmt(t, fn)
		}
		if s.Cont != nil {
			for _, t := range s.Cont.Stmts {
				walkStmt(t, fn)
			}
		}
	case *wgsl.IncDec:
		walkExpr(s.X, fn)
	}
}

func walkExpr(x wgsl.Expr, fn func(wgsl.Expr)) {
	if x == nil {
		return
	}
	fn(x)
	switch x := x.(type) {
	case *wgsl.Call:
		for _, a := range x.Args {
			walkExpr(a, fn)
		}
	case *wgsl.Unary:
		walkExpr(x.X, fn)
	case *wgsl.Binary:
		walkExpr(x.X, fn)
		walkExpr(x.Y, fn)
	case *wgsl.Paren:
		walkExpr(x.X, fn)
	case *wgsl.Member:
		walkExpr(x.X, fn)
	case *wgsl.Index:
		walkExpr(x.X, fn)
		walkExpr(x.I, fn)
	}
}

// useBuffer checks a module-scope variable is a group-0 storage or uniform
// buffer of a supported shape, and records it and the structure it holds.
func (e *emitter) useBuffer(v *wgsl.VarObj) error {
	if e.vars[v] {
		return nil
	}
	off := v.Decl.Start
	if v.AS != wgsl.ASStorage && v.AS != wgsl.ASUniform {
		return &Error{off, "the " + v.AS.String() + " variable " + v.Name + " is not supported yet (only storage and uniform buffers)"}
	}
	if v.Group == nil || v.Binding == nil {
		return &Error{off, "the buffer " + v.Name + " has no @group and @binding"}
	}
	if *v.Group != 0 {
		return &Error{off, fmt.Sprintf("the buffer %s is in group %d; only group 0 is supported (the binding is its register index)", v.Name, *v.Group)}
	}
	limit, class := 127, "t"
	switch {
	case v.AS == wgsl.ASUniform:
		limit, class = 13, "b"
	case v.Access != wgsl.AccessRead:
		limit, class = 7, "u"
	}
	if *v.Binding < 0 || int(*v.Binding) > limit {
		return &Error{off, fmt.Sprintf("the buffer %s is at binding %d; its register %s%d is beyond the %d registers of that class in Direct3D 11's compute stage", v.Name, *v.Binding, class, *v.Binding, limit+1)}
	}
	t := v.Store
	if a, ok := t.(*wgsl.ArrayType); ok {
		if v.AS == wgsl.ASUniform {
			return &Error{off, "the uniform array " + v.Name + " is not supported yet"}
		}
		if !a.Runtime {
			return &Error{off, "the fixed-size array " + v.Name + " is not supported yet (only runtime-sized storage arrays)"}
		}
		t = a.Elem
	} else if v.AS == wgsl.ASStorage {
		return &Error{off, "the storage variable " + v.Name + " is not an array; only runtime-sized storage arrays are supported"}
	}
	if e.vars == nil {
		e.vars = map[*wgsl.VarObj]bool{}
		e.host = map[*wgsl.StructType]string{}
	}
	e.vars[v] = true
	if s, ok := t.(*wgsl.StructType); ok {
		kind := "structured"
		if v.AS == wgsl.ASUniform {
			kind = "cbuffer"
		}
		if old, ok := e.host[s]; ok && old != kind {
			return &Error{off, "structure " + s.Name + " is used in both a structured buffer and a cbuffer"}
		}
		e.host[s] = kind
	}
	return e.useType(t, off)
}

// resources is every buffer the entry points reach, in binding order.
func (e *emitter) resources() []*wgsl.VarObj {
	var out []*wgsl.VarObj
	for v := range e.vars {
		out = append(out, v)
	}
	slices.SortFunc(out, func(a, b *wgsl.VarObj) int { return int(*a.Binding - *b.Binding) })
	return out
}

// declareBuffer is the HLSL declaration of buffer v.
func (e *emitter) declareBuffer(v *wgsl.VarObj) (string, error) {
	n, err := checkName(v.Name, v.Decl.Start)
	if err != nil {
		return "", err
	}
	if a, ok := v.Store.(*wgsl.ArrayType); ok {
		t, err := typeName(a.Elem, v.Decl.Start)
		if err != nil {
			return "", err
		}
		if v.Access == wgsl.AccessRead {
			return fmt.Sprintf("StructuredBuffer<%s> %s : register(t%d);", t, n, *v.Binding), nil
		}
		return fmt.Sprintf("RWStructuredBuffer<%s> %s : register(u%d);", t, n, *v.Binding), nil
	}
	t, err := typeName(v.Store, v.Decl.Start)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("cbuffer wgsl_cb%d : register(b%d) { %s %s; };", *v.Binding, *v.Binding, t, n), nil
}

// useType checks t is in scope and records the structures it names.
func (e *emitter) useType(t wgsl.Type, off int) error {
	if e.structs == nil {
		e.structs = map[*wgsl.StructType]bool{}
	}
	switch t := t.(type) {
	case *wgsl.ScalarType:
		if _, err := scalarName(t, off); err != nil {
			return err
		}
	case *wgsl.VecType:
		if t.Elem.Kind == wgsl.KBool {
			return &Error{off, "type " + t.String() + " is not supported yet"}
		}
		if _, err := scalarName(t.Elem, off); err != nil {
			return err
		}
	case *wgsl.StructType:
		if t.Runtime {
			return &Error{off, "structure " + t.Name + " has a runtime-sized array"}
		}
		e.structs[t] = true
		for _, m := range t.Members {
			if err := e.useType(m.Type, off); err != nil {
				return err
			}
		}
	default:
		return &Error{off, "type " + t.String() + " is not supported yet"}
	}
	return nil
}

func scalarName(s *wgsl.ScalarType, off int) (string, error) {
	switch s.Kind {
	case wgsl.KBool:
		return "bool", nil
	case wgsl.KI32:
		return "int", nil
	case wgsl.KU32:
		return "uint", nil
	case wgsl.KF32:
		return "float", nil
	}
	return "", &Error{off, "type " + s.String() + " is not supported yet"}
}

// typeName is the HLSL spelling of an in-scope type.
func typeName(t wgsl.Type, off int) (string, error) {
	switch t := t.(type) {
	case *wgsl.ScalarType:
		return scalarName(t, off)
	case *wgsl.VecType:
		s, err := scalarName(t.Elem, off)
		return s + strconv.Itoa(t.N), err
	case *wgsl.StructType:
		return checkName(t.Name, off)
	case *wgsl.RefType:
		return typeName(t.Elem, off)
	}
	return "", &Error{off, "type " + t.String() + " is not supported yet"}
}

// isFloat reports whether t is f32 or a vector of f32.
func isFloat(t wgsl.Type) bool {
	switch t := t.(type) {
	case *wgsl.ScalarType:
		return t.Kind == wgsl.KF32
	case *wgsl.VecType:
		return t.Elem.Kind == wgsl.KF32
	}
	return false
}

// reserved are the HLSL keywords, types and intrinsics a WGSL identifier may be
// but an HLSL one may not (or may not safely: a function named like an
// intrinsic would shadow it for the emitted helpers): emitted names are WGSL's
// own, so a collision is refused.
var reserved = map[string]bool{}

func init() {
	for _, w := range strings.Fields(`
		asm asm_fragment bool break buffer cbuffer centroid class column_major compile compile_fragment const
		continue discard do double else export extern false float for fxgroup groupshared half if in inline inout
		int interface line lineadj linear matrix namespace nointerpolation noperspective out packoffset pass
		pixelfragment point precise register return row_major sample sampler shared snorm stateblock static string
		struct switch tbuffer technique texture triangle triangleadj true typedef uint uniform unorm unsigned
		vector vertexfragment void volatile while main
		abs acos all any asfloat asin asint asuint atan atan2 ceil clamp clip cos cosh cross ddx ddy degrees
		determinant distance dot exp exp2 faceforward floor fmod frac frexp isfinite isinf isnan ldexp length lerp
		lit log log10 log2 mad max min modf mul normalize pow radians rcp reflect refract round rsqrt saturate sign
		sin sincos sinh smoothstep sqrt step tan tanh transpose trunc select
		Buffer StructuredBuffer RWStructuredBuffer ByteAddressBuffer RWByteAddressBuffer Texture2D SamplerState`) {
		reserved[w] = true
	}
}

func checkName(n string, off int) (string, error) {
	if reserved[n] || strings.HasPrefix(n, "wgsl_") || strings.HasPrefix(n, "__") || strings.HasPrefix(n, "SV_") || vectorTypeName(n) {
		return "", &Error{off, "the name " + n + " is reserved in HLSL"}
	}
	return n, nil
}

// vectorTypeName reports whether n spells an HLSL vector or matrix type
// (float3, uint2, int4x4, min16float, ...).
func vectorTypeName(n string) bool {
	for _, p := range []string{"bool", "int", "uint", "float", "double", "half", "dword"} {
		if rest, ok := strings.CutPrefix(n, p); ok && rest != "" && strings.Trim(rest, "1234x") == "" {
			return true
		}
	}
	return strings.HasPrefix(n, "min16") || strings.HasPrefix(n, "min10") || strings.HasPrefix(n, "min12")
}

func memberSize(s *wgsl.StructType, m *wgsl.FieldInfo) (int64, error) {
	switch t := m.Type.(type) {
	case *wgsl.ScalarType:
		return 4, nil
	case *wgsl.VecType:
		return int64(4 * t.N), nil
	}
	return 0, &Error{m.Decl.Start, "member " + m.Name + " of " + s.Name + ": only scalars and vectors can live in a buffer yet"}
}

// hlslStructuredLayout checks that a structure kept in a StructuredBuffer has,
// in HLSL, the offsets WGSL's memory layout gives it. A structured buffer packs
// members at 4-byte granularity with no register rule, so the only structures
// that match are those whose WGSL offsets are also tightly packed.
func hlslStructuredLayout(s *wgsl.StructType) error {
	var off int64
	for _, m := range s.Members {
		size, err := memberSize(s, m)
		if err != nil {
			return err
		}
		if off != m.Offset {
			return &Error{m.Decl.Start, fmt.Sprintf("member %s of %s is at offset %d in WGSL and %d in a StructuredBuffer", m.Name, s.Name, m.Offset, off)}
		}
		off += size
	}
	if off != s.Size {
		return &Error{s.Decl.Start, fmt.Sprintf("%s is %d bytes in WGSL and %d in a StructuredBuffer", s.Name, s.Size, off)}
	}
	return nil
}

// hlslCBufferLayout checks the same for a structure held in a cbuffer: members
// are packed into 16-byte registers and a vector does not straddle one.
func hlslCBufferLayout(s *wgsl.StructType) error {
	var off int64
	for _, m := range s.Members {
		size, err := memberSize(s, m)
		if err != nil {
			return err
		}
		if off%16+size > 16 {
			off = (off + 15) / 16 * 16
		}
		if off != m.Offset {
			return &Error{m.Decl.Start, fmt.Sprintf("member %s of %s is at offset %d in WGSL and %d in a cbuffer", m.Name, s.Name, m.Offset, off)}
		}
		off += size
	}
	if (off+15)/16*16 < s.Size {
		return &Error{s.Decl.Start, fmt.Sprintf("%s is %d bytes in WGSL and %d in a cbuffer", s.Name, s.Size, (off+15)/16*16)}
	}
	return nil
}

func (e *emitter) emitStruct(b *strings.Builder, s *wgsl.StructType) error {
	name, err := checkName(s.Name, s.Decl.Start)
	if err != nil {
		return err
	}
	switch e.host[s] {
	case "structured":
		if err := hlslStructuredLayout(s); err != nil {
			return err
		}
	case "cbuffer":
		if err := hlslCBufferLayout(s); err != nil {
			return err
		}
	}
	fmt.Fprintf(b, "\nstruct %s {\n", name)
	for _, m := range s.Members {
		t, err := typeName(m.Type, m.Decl.Start)
		if err != nil {
			return err
		}
		n, err := checkName(m.Name, m.Decl.Start)
		if err != nil {
			return err
		}
		fmt.Fprintf(b, "  %s %s;\n", t, n)
	}
	b.WriteString("};\n")
	return nil
}

// builtins maps an entry point's @builtin input to its HLSL type and semantic.
var builtins = map[string]struct{ typ, sem string }{
	"global_invocation_id": {"uint3", "SV_DispatchThreadID"},
}

func (e *emitter) emitFunc(b *strings.Builder, f *wgsl.FuncObj) error {
	name, err := checkName(f.Name, f.Decl.Start)
	if err != nil {
		return err
	}
	var params []string
	ret := "void"
	if _, void := f.Ret.(*wgsl.VoidType); !void {
		if f.Entry != nil {
			return &Error{f.Decl.Start, "entry point " + f.Name + " returns a value"}
		}
		if ret, err = typeName(f.Ret, f.Decl.Start); err != nil {
			return err
		}
	}
	if f.Entry != nil {
		for _, in := range f.Entry.Inputs {
			bi, ok := builtins[in.Builtin]
			if in.Member != "" || !ok {
				return &Error{f.Decl.Start, "input " + in.Name + " of " + f.Name + " is not a supported @builtin (only global_invocation_id)"}
			}
			n, err := checkName(in.Name, f.Params[in.Param].Decl.Start)
			if err != nil {
				return err
			}
			params = append(params, fmt.Sprintf("%s %s : %s", bi.typ, n, bi.sem))
		}
		ws := f.Entry.WorkgroupSize
		fmt.Fprintf(b, "\n[numthreads(%d, %d, %d)]\n", ws[0], ws[1], ws[2])
	} else {
		for _, p := range f.Params {
			t, err := typeName(p.Type, p.Decl.Start)
			if err != nil {
				return err
			}
			n, err := checkName(p.Name, p.Decl.Start)
			if err != nil {
				return err
			}
			params = append(params, t+" "+n)
		}
		b.WriteString("\n")
	}
	fmt.Fprintf(b, "%s %s(%s) ", ret, name, strings.Join(params, ", "))
	e.b, e.depth = b, 0
	if err := e.block(f.Decl.Body); err != nil {
		return err
	}
	b.WriteString("\n")
	return nil
}

func (e *emitter) line(format string, args ...any) {
	e.b.WriteString(strings.Repeat("  ", e.depth))
	fmt.Fprintf(e.b, format, args...)
	e.b.WriteString("\n")
}

func (e *emitter) block(bl *wgsl.Block) error {
	e.b.WriteString("{\n")
	e.depth++
	for _, s := range bl.Stmts {
		if err := e.stmt(s); err != nil {
			return err
		}
	}
	e.depth--
	e.b.WriteString(strings.Repeat("  ", e.depth) + "}")
	return nil
}

func (e *emitter) stmt(s wgsl.Stmt) error {
	switch s := s.(type) {
	case *wgsl.Block:
		e.b.WriteString(strings.Repeat("  ", e.depth))
		if err := e.block(s); err != nil {
			return err
		}
		e.b.WriteString("\n")
	case *wgsl.Return:
		if s.X == nil {
			e.line("return;")
			return nil
		}
		x, err := e.expr(s.X)
		if err != nil {
			return err
		}
		e.line("return %s;", x)
	case *wgsl.DeclStmt, *wgsl.Assign, *wgsl.IncDec, *wgsl.CallStmt:
		t, err := e.simple(s)
		if err != nil {
			return err
		}
		e.line("%s;", t)
	case *wgsl.Break:
		e.line("break;")
	case *wgsl.Continue:
		e.line("continue;")
	case *wgsl.For:
		var init, cond, update string
		var err error
		if s.Init != nil {
			if init, err = e.simple(s.Init); err != nil {
				return err
			}
		}
		if s.Cond != nil {
			if cond, err = e.expr(s.Cond); err != nil {
				return err
			}
		}
		if s.Update != nil {
			if update, err = e.simple(s.Update); err != nil {
				return err
			}
		}
		e.b.WriteString(strings.Repeat("  ", e.depth))
		fmt.Fprintf(e.b, "[loop] for (%s; %s; %s) ", init, cond, update)
		if err := e.block(s.Body); err != nil {
			return err
		}
		e.b.WriteString("\n")
	case *wgsl.While:
		c, err := e.expr(s.Cond)
		if err != nil {
			return err
		}
		e.b.WriteString(strings.Repeat("  ", e.depth))
		fmt.Fprintf(e.b, "[loop] while (%s) ", c)
		if err := e.block(s.Body); err != nil {
			return err
		}
		e.b.WriteString("\n")
	case *wgsl.Loop:
		if s.Cont != nil {
			return &Error{s.Start, "a loop with a continuing block is not supported yet"}
		}
		e.b.WriteString(strings.Repeat("  ", e.depth) + "[loop] while (true) ")
		if err := e.block(&wgsl.Block{Stmts: s.Body}); err != nil {
			return err
		}
		e.b.WriteString("\n")
	case *wgsl.If:
		e.b.WriteString(strings.Repeat("  ", e.depth))
		if err := e.ifStmt(s); err != nil {
			return err
		}
		e.b.WriteString("\n")
	case *wgsl.Empty:
	default:
		return &Error{s.NodeSpan().Start, fmt.Sprintf("statement %T is not supported yet", s)}
	}
	return nil
}

func (e *emitter) ifStmt(s *wgsl.If) error {
	c, err := e.expr(s.Cond)
	if err != nil {
		return err
	}
	fmt.Fprintf(e.b, "if (%s) ", c)
	if err := e.block(s.Then); err != nil {
		return err
	}
	switch el := s.Else.(type) {
	case nil:
	case *wgsl.If:
		e.b.WriteString(" else ")
		return e.ifStmt(el)
	case *wgsl.Block:
		e.b.WriteString(" else ")
		return e.block(el)
	}
	return nil
}

// simple is a statement that may head a for loop, as text without its
// semicolon: a declaration, an assignment, an increment or a call.
func (e *emitter) simple(s wgsl.Stmt) (string, error) {
	switch s := s.(type) {
	case *wgsl.DeclStmt:
		return e.decl(s)
	case *wgsl.Assign:
		return e.assign(s)
	case *wgsl.IncDec:
		return e.incdec(s)
	case *wgsl.CallStmt:
		return e.expr(s.Call)
	}
	return "", &Error{s.NodeSpan().Start, fmt.Sprintf("statement %T is not supported here", s)}
}

// incdec is x++ or x--. An i32 goes through wgsl_add so that it wraps; the
// operand is written twice, so it must not contain a call.
func (e *emitter) incdec(s *wgsl.IncDec) (string, error) {
	x, err := e.expr(s.X)
	if err != nil {
		return "", err
	}
	k, _ := elemKind(e.valueType(s.X))
	if k != wgsl.KI32 {
		return x + s.Op, nil
	}
	hasCall := false
	walkExpr(s.X, func(y wgsl.Expr) {
		if _, ok := y.(*wgsl.Call); ok {
			hasCall = true
		}
	})
	if hasCall {
		return "", &Error{s.Start, "an i32 " + s.Op + " of an operand with a call is not supported yet"}
	}
	f := map[string]string{"++": "wgsl_add", "--": "wgsl_sub"}[s.Op]
	return fmt.Sprintf("%s = %s(%s, 1)", x, f, x), nil
}

func (e *emitter) decl(s *wgsl.DeclStmt) (string, error) {
	var (
		name string
		init wgsl.Expr
		t    wgsl.Type
		off  int
	)
	switch d := s.Decl.(type) {
	case *wgsl.Value:
		name, init, off = d.Name, d.Init, d.Start
		t = e.info.DeclTypes[d]
	case *wgsl.Var:
		name, init, off = d.Name, d.Init, d.Start
		t = e.info.DeclTypes[d]
	default:
		return "", &Error{s.Start, fmt.Sprintf("declaration %T is not supported yet", d)}
	}
	if err := e.useType(t, off); err != nil {
		return "", err
	}
	tn, err := typeName(t, off)
	if err != nil {
		return "", err
	}
	n, err := checkName(name, off)
	if err != nil {
		return "", err
	}
	prefix := ""
	if isFloat(t) { // rule 1: every float variable is precise
		prefix = "precise "
	}
	if init == nil { // WGSL zero-initialises a var
		return fmt.Sprintf("%s%s %s = (%s)0", prefix, tn, n, tn), nil
	}
	x, err := e.expr(init)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s%s %s = %s", prefix, tn, n, x), nil
}

func (e *emitter) assign(s *wgsl.Assign) (string, error) {
	if s.LHS == nil {
		return "", &Error{s.Start, "a phony assignment is not supported yet"}
	}
	lhs, err := e.expr(s.LHS)
	if err != nil {
		return "", err
	}
	if s.Op == "=" {
		rhs, err := e.expr(s.RHS)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("%s = %s", lhs, rhs), nil
	}
	op := strings.TrimSuffix(s.Op, "=")
	rhs, err := e.binary(op, s.LHS, s.RHS, e.valueType(s.LHS), s.Start)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s = %s", lhs, rhs), nil
}

// valueType is the type x has as a value: its concretised type, with the
// load rule applied.
func (e *emitter) valueType(x wgsl.Expr) wgsl.Type {
	t := e.info.Types[x]
	if c, ok := e.info.Conversions[x]; ok {
		t = c
	}
	if r, ok := t.(*wgsl.RefType); ok {
		return r.Elem
	}
	return t
}

func elemKind(t wgsl.Type) (wgsl.ScalarKind, bool) {
	switch t := t.(type) {
	case *wgsl.ScalarType:
		return t.Kind, true
	case *wgsl.VecType:
		return t.Elem.Kind, false
	}
	return wgsl.KBool, false
}

func (e *emitter) expr(x wgsl.Expr) (string, error) {
	t := e.valueType(x)
	if c, ok := e.info.Consts[x]; ok && c != nil {
		if s, ok := e.constant(t, x); ok {
			return s, nil
		}
	}
	switch x := x.(type) {
	case *wgsl.Paren:
		s, err := e.expr(x.X)
		return "(" + s + ")", err
	case *wgsl.IntLit, *wgsl.FloatLit:
		return e.literalValue(x, t)
	case *wgsl.BoolLit:
		return strconv.FormatBool(x.Value), nil
	case *wgsl.Ident:
		switch o := e.info.Objects[x].(type) {
		case *wgsl.ParamObj:
			return checkName(o.Name, x.Start)
		case *wgsl.VarObj:
			return checkName(o.Name, x.Start)
		case *wgsl.ValueObj:
			if !o.Local {
				return "", &Error{x.Start, "module-scope " + o.Kind + " " + o.Name + " is not supported yet"}
			}
			return checkName(o.Name, x.Start)
		}
		return "", &Error{x.Start, "identifier " + x.Name + " is not supported yet"}
	case *wgsl.Member:
		base, err := e.expr(x.X)
		if err != nil {
			return "", err
		}
		return base + "." + x.Name, nil
	case *wgsl.Index:
		// Not bounds-checked by this emitter: an out-of-range struct index of a
		// structured buffer reads 0 and a write is dropped (docs/HLSL.md, rule 7).
		base, err := e.expr(x.X)
		if err != nil {
			return "", err
		}
		i, err := e.expr(x.I)
		if err != nil {
			return "", err
		}
		return base + "[" + i + "]", nil
	case *wgsl.Unary:
		return e.unary(x, t)
	case *wgsl.Binary:
		return e.binary(x.Op, x.X, x.Y, t, x.Start)
	case *wgsl.Call:
		return e.call(x, t)
	}
	return "", &Error{x.NodeSpan().Start, fmt.Sprintf("expression %T is not supported yet", x)}
}

// constant emits a scalar const-expression as a literal of its concrete type.
func (e *emitter) constant(t wgsl.Type, x wgsl.Expr) (string, bool) {
	switch x.(type) {
	case *wgsl.IntLit, *wgsl.FloatLit, *wgsl.Unary:
	default:
		return "", false // composite constants keep their spelling
	}
	s, err := e.literalValue(x, t)
	return s, err == nil
}

// literalValue spells a numeric literal (or a negated one) in its concrete
// type: an integer exactly, a float as its float32 value, shortest form.
func (e *emitter) literalValue(x wgsl.Expr, t wgsl.Type) (string, error) {
	neg := false
	if u, ok := x.(*wgsl.Unary); ok {
		if u.Op != "-" {
			return "", &Error{u.Start, "not a literal"}
		}
		neg, x = true, u.X
	}
	var text string
	switch l := x.(type) {
	case *wgsl.IntLit:
		text = l.Text
	case *wgsl.FloatLit:
		text = l.Text
	default:
		return "", &Error{0, "not a literal"}
	}
	k, scalar := elemKind(t)
	if !scalar {
		return "", &Error{0, "not a scalar literal"}
	}
	switch k {
	case wgsl.KI32, wgsl.KU32:
		v, err := strconv.ParseInt(strings.TrimRight(text, "iu"), 0, 64)
		if err != nil {
			return "", &Error{0, "integer literal " + text}
		}
		if neg {
			v = -v
		}
		if k == wgsl.KU32 {
			return strconv.FormatUint(uint64(uint32(v)), 10) + "u", nil
		}
		if int32(v) == -1<<31 {
			return "wgsl_int_min", nil // -2147483648 is not an HLSL literal
		}
		return strconv.FormatInt(int64(int32(v)), 10), nil
	case wgsl.KF32:
		v, err := strconv.ParseFloat(strings.TrimRight(text, "fhiu"), 64)
		if err != nil {
			return "", &Error{0, "float literal " + text}
		}
		if neg {
			v = -v
		}
		s := strconv.FormatFloat(float64(float32(v)), 'g', -1, 32)
		if !strings.ContainsAny(s, ".en") {
			s += ".0"
		}
		return s + "f", nil
	}
	return "", &Error{0, "literal of type " + t.String()}
}

func (e *emitter) unary(x *wgsl.Unary, t wgsl.Type) (string, error) {
	a, err := e.expr(x.X)
	if err != nil {
		return "", err
	}
	k, scalar := elemKind(t)
	switch x.Op {
	case "-":
		if k == wgsl.KI32 {
			if !scalar {
				return "", &Error{x.Start, "negation of an i32 vector is not supported yet"}
			}
			return "wgsl_neg(" + a + ")", nil
		}
		return "(-" + a + ")", nil
	case "!", "~":
		if !scalar && x.Op == "!" {
			return "", &Error{x.Start, "! on a vector is not supported yet"}
		}
		return "(" + x.Op + a + ")", nil
	}
	return "", &Error{x.Start, "unary " + x.Op + " is not supported yet"}
}

// hasUserCall reports whether x calls a function declared in the module.
func (e *emitter) hasUserCall(x wgsl.Expr) bool {
	found := false
	walkExpr(x, func(y wgsl.Expr) {
		if c, ok := y.(*wgsl.Call); ok {
			if _, ok := e.info.Objects[c.Fn].(*wgsl.FuncObj); ok {
				found = true
			}
		}
	})
	return found
}

func (e *emitter) binary(op string, xe, ye wgsl.Expr, t wgsl.Type, off int) (string, error) {
	a, err := e.expr(xe)
	if err != nil {
		return "", err
	}
	b, err := e.expr(ye)
	if err != nil {
		return "", err
	}
	k, scalar := elemKind(e.valueType(xe))
	if k2, _ := elemKind(e.valueType(ye)); k == wgsl.KBool && k2 != wgsl.KBool {
		k = k2
	}
	call := func(f string) (string, error) {
		if !scalar {
			return "", &Error{off, "integer " + op + " on vectors is not supported yet"}
		}
		return f + "(" + a + ", " + b + ")", nil
	}
	switch op {
	case "+", "-", "*":
		switch k {
		case wgsl.KI32:
			return call(map[string]string{"+": "wgsl_add", "-": "wgsl_sub", "*": "wgsl_mul"}[op])
		case wgsl.KF32:
			return "wgsl_pf(" + a + " " + op + " " + b + ")", nil // rule 1
		}
	case "/":
		switch k {
		case wgsl.KI32:
			return call("wgsl_div")
		case wgsl.KU32:
			return call("wgsl_udiv")
		case wgsl.KF32:
			return "wgsl_pf(" + a + " / " + b + ")", nil
		}
	case "%":
		switch k {
		case wgsl.KI32:
			return call("wgsl_rem")
		case wgsl.KU32:
			return call("wgsl_urem")
		}
		return "", &Error{off, "floating-point % is not supported yet"}
	case "<<":
		if k == wgsl.KI32 {
			return call("wgsl_shl")
		}
		if !scalar {
			return "", &Error{off, "shifts of vectors are not supported yet"}
		}
		return "(" + a + " << (" + b + " & 31u))", nil
	case ">>":
		if k == wgsl.KI32 {
			return call("wgsl_shr")
		}
		if !scalar {
			return "", &Error{off, "shifts of vectors are not supported yet"}
		}
		return "(" + a + " >> (" + b + " & 31u))", nil
	case "&&", "||":
		if !scalar {
			return "", &Error{off, op + " on vectors"}
		}
		// HLSL's && and || evaluate both operands; WGSL's do not. Equal unless
		// the right operand has an effect, which only a user function can have.
		if e.hasUserCall(ye) {
			return "", &Error{off, "a call to a function on the right of " + op + " is not supported (HLSL does not short-circuit)"}
		}
	case "&", "|":
		if k == wgsl.KBool {
			if !scalar {
				return "", &Error{off, op + " on a bool vector"}
			}
			op = map[string]string{"&": "&&", "|": "||"}[op] // both operands are evaluated in WGSL too
		}
	case "==", "!=", "<", "<=", ">", ">=", "^":
	default:
		return "", &Error{off, "operator " + op + " is not supported yet"}
	}
	if !scalar && slices.Contains([]string{"==", "!=", "<", "<=", ">", ">="}, op) {
		return "", &Error{off, "vector comparisons are not supported yet"}
	}
	return "(" + a + " " + op + " " + b + ")", nil
}

func (e *emitter) call(x *wgsl.Call, t wgsl.Type) (string, error) {
	args := make([]string, len(x.Args))
	for i, a := range x.Args {
		s, err := e.expr(a)
		if err != nil {
			return "", err
		}
		args[i] = s
	}
	all := strings.Join(args, ", ")
	if ct, ok := e.info.TypeSpecs[x.Fn]; ok { // a constructor or a conversion
		return e.construct(x, ct, args)
	}
	switch o := e.info.Objects[x.Fn].(type) {
	case *wgsl.FuncObj:
		n, err := checkName(o.Name, x.Start)
		if err != nil {
			return "", err
		}
		return n + "(" + all + ")", nil
	case *wgsl.BuiltinObj:
		at := e.valueType(x.Args[0])
		k, _ := elemKind(at)
		float := k == wgsl.KF32
		switch o.Name {
		case "abs":
			if !float {
				return "", &Error{x.Start, "integer abs is not supported yet"}
			}
			return "abs(" + all + ")", nil
		case "sqrt":
			return "sqrt(" + all + ")", nil
		case "floor":
			return "floor(" + all + ")", nil
		case "min", "max":
			return o.Name + "(" + all + ")", nil
		case "clamp": // WGSL: min(max(e, low), high), defined when low > high
			return "min(max(" + args[0] + ", " + args[1] + "), " + args[2] + ")", nil
		case "select": // select(f, t, cond): both operands are evaluated, as in WGSL
			if _, ok := e.valueType(x.Args[2]).(*wgsl.VecType); ok {
				return "", &Error{x.Start, "select with a vector condition is not supported yet"}
			}
			return "(" + args[2] + " ? " + args[1] + " : " + args[0] + ")", nil
		case "length":
			if _, ok := at.(*wgsl.VecType); !ok {
				return "abs(" + all + ")", nil
			}
			return "wgsl_length(" + all + ")", nil
		}
		return "", &Error{x.Start, "builtin " + o.Name + " is not supported yet"}
	}
	return "", &Error{x.Start, "call of " + x.Fn.Name + " is not supported yet"}
}

func (e *emitter) construct(x *wgsl.Call, ct wgsl.Type, args []string) (string, error) {
	if err := e.useType(ct, x.Start); err != nil {
		return "", err
	}
	tn, err := typeName(ct, x.Start)
	if err != nil {
		return "", err
	}
	if len(args) == 0 {
		return "((" + tn + ")0)", nil // the zero value
	}
	if x.Fn.Name == "bitcast" {
		return e.bitcast(x, ct, args[0])
	}
	if _, ok := ct.(*wgsl.StructType); ok {
		return "", &Error{x.Start, "a structure constructor is not supported yet"}
	}
	from, fromScalar := elemKind(e.valueType(x.Args[0]))
	if v, ok := ct.(*wgsl.VecType); ok {
		if len(x.Args) == 1 && fromScalar {
			return "((" + tn + ")(" + args[0] + "))", nil // a splat: HLSL has no float3(x)
		}
		isInt := func(k wgsl.ScalarKind) bool { return k == wgsl.KI32 || k == wgsl.KU32 }
		if len(x.Args) == 1 && isInt(v.Elem.Kind) && from == wgsl.KF32 {
			return "", &Error{x.Start, "conversion of an f32 vector to an integer vector is not supported yet"}
		}
		if len(x.Args) == 1 && isInt(v.Elem.Kind) && isInt(from) && from != v.Elem.Kind {
			return "", &Error{x.Start, "conversion between i32 and u32 vectors is not supported yet (use bitcast)"}
		}
		return tn + "(" + strings.Join(args, ", ") + ")", nil
	}
	s := ct.(*wgsl.ScalarType)
	if len(x.Args) == 1 {
		switch {
		case s.Kind == wgsl.KI32 && from == wgsl.KF32:
			return "wgsl_f2i(" + args[0] + ")", nil // rule 6
		case s.Kind == wgsl.KU32 && from == wgsl.KF32:
			return "wgsl_f2u(" + args[0] + ")", nil
		case s.Kind == wgsl.KI32 && from == wgsl.KU32:
			return "asint(" + args[0] + ")", nil // the bits, never a value conversion
		case s.Kind == wgsl.KU32 && from == wgsl.KI32:
			return "asuint(" + args[0] + ")", nil
		case from == wgsl.KBool && s.Kind != wgsl.KBool:
			one := map[wgsl.ScalarKind]string{wgsl.KI32: "1", wgsl.KU32: "1u", wgsl.KF32: "1.0f"}[s.Kind]
			zero := map[wgsl.ScalarKind]string{wgsl.KI32: "0", wgsl.KU32: "0u", wgsl.KF32: "0.0f"}[s.Kind]
			return "(" + args[0] + " ? " + one + " : " + zero + ")", nil
		case s.Kind == wgsl.KBool && from != wgsl.KBool:
			return "", &Error{x.Start, "conversion to bool is not supported yet"}
		}
	}
	return tn + "(" + strings.Join(args, ", ") + ")", nil
}

// bitcast is asint, asuint or asfloat: a reinterpretation of the bits.
func (e *emitter) bitcast(x *wgsl.Call, ct wgsl.Type, arg string) (string, error) {
	k, _ := elemKind(ct)
	switch k {
	case wgsl.KI32:
		return "asint(" + arg + ")", nil
	case wgsl.KU32:
		return "asuint(" + arg + ")", nil
	case wgsl.KF32:
		return "asfloat(" + arg + ")", nil
	}
	return "", &Error{x.Start, "bitcast to " + ct.String() + " is not supported"}
}
