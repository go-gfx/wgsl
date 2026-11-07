// Package msl emits Metal Shading Language for the functions of a checked
// WGSL module.
//
// It is the first version described in docs/MSL.md: plain functions (no
// entry points, no module-scope variables, no loops, arrays or textures) over
// bool, i32, u32, f32, their vectors and structures of those. Anything else is
// refused with its position, never emitted approximately.
//
// Every rule that makes the emitted code compute what WGSL defines, rather
// than what Metal's defaults allow, is in docs/MSL.md with its source:
//   - fast math is turned off for the whole file;
//   - library functions are the precise:: ones;
//   - signed +, -, * and << go through uint, because WGSL's i32 wraps and
//     C++'s signed overflow is undefined;
//   - / and % follow WGSL's definition for a zero divisor and MIN / -1;
//   - shift counts are taken modulo 32, and >> on i32 is arithmetic.
package msl

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/go-gfx/wgsl"
)

// Preamble opens every emitted file: Metal's fast math is off (§1.6.3 of the
// Metal Shading Language Specification: math_mode(safe), and contract(off),
// since safe mode alone still fuses a*b+c within a statement), then the
// helpers the WGSL integer rules need.
const Preamble = `#include <metal_stdlib>
using namespace metal;

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// WGSL i32 arithmetic wraps (two's complement); C++ signed overflow does not.
static inline int wgsl_add(int a, int b) { return as_type<int>(as_type<uint>(a) + as_type<uint>(b)); }
static inline int wgsl_sub(int a, int b) { return as_type<int>(as_type<uint>(a) - as_type<uint>(b)); }
static inline int wgsl_mul(int a, int b) { return as_type<int>(as_type<uint>(a) * as_type<uint>(b)); }
static inline int wgsl_neg(int a) { return as_type<int>(0u - as_type<uint>(a)); }
// WGSL division and remainder: e2 == 0, and MIN / -1, are defined.
static inline int wgsl_div(int a, int b) { return a / select(b, 1, b == 0 || (a == INT_MIN && b == -1)); }
static inline int wgsl_rem(int a, int b) {
  bool special = b == 0 || (a == INT_MIN && b == -1);
  int d = select(b, 1, special);
  return select(wgsl_sub(a, wgsl_mul(a / d, d)), 0, special);
}
static inline uint wgsl_udiv(uint a, uint b) { return a / select(b, 1u, b == 0u); }
static inline uint wgsl_urem(uint a, uint b) { return select(a % select(b, 1u, b == 0u), 0u, b == 0u); }
// WGSL shifts: the count modulo 32; << discards high bits; >> on i32 is
// arithmetic.
static inline int wgsl_shl(int a, uint s) { return as_type<int>(as_type<uint>(a) << (s & 31u)); }
static inline int wgsl_shr(int a, uint s) {
  uint u = as_type<uint>(a) >> (s & 31u);
  return as_type<int>(select(u, u | ~(0xFFFFFFFFu >> (s & 31u)), a < 0));
}
static inline float wgsl_length(float2 v) { return precise::sqrt(v.x * v.x + v.y * v.y); }
static inline float wgsl_length(float3 v) { return precise::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
static inline float wgsl_length(float4 v) { return precise::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w); }
`

// Error is a construct the emitter refuses, with where it is.
type Error struct {
	Offset int
	What   string
}

func (e *Error) Error() string { return fmt.Sprintf("msl: offset %d: %s", e.Offset, e.What) }

// Functions emits Preamble, the structures the named functions use, and the
// named functions with every function they call, in declaration order. m
// must be the module info was checked from.
func Functions(m *wgsl.Module, info *wgsl.Info, names ...string) (string, error) {
	e := &emitter{info: info, need: map[*wgsl.FuncObj]bool{}}
	byName := map[string]*wgsl.FuncObj{}
	for _, f := range info.Funcs {
		byName[f.Name] = f
	}
	for _, n := range names {
		f, ok := byName[n]
		if !ok {
			return "", fmt.Errorf("msl: no function %q in the module", n)
		}
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
	for _, f := range info.Funcs {
		if !e.need[f] {
			continue
		}
		if err := e.emitFunc(&b, f); err != nil {
			return "", err
		}
	}
	return b.String(), nil
}

type emitter struct {
	info    *wgsl.Info
	need    map[*wgsl.FuncObj]bool
	structs map[*wgsl.StructType]bool
	b       *strings.Builder
	depth   int
}

// require marks f and its callees for emission, refusing what is out of scope
// at the level of the whole function.
func (e *emitter) require(f *wgsl.FuncObj) error {
	if e.need[f] {
		return nil
	}
	if f.Entry != nil {
		return &Error{f.Decl.Start, "function " + f.Name + " is an entry point"}
	}
	if len(f.Globals) > 0 {
		return &Error{f.Decl.Start, fmt.Sprintf("function %s uses the module-scope %s", f.Name, f.Globals[0].ObjName())}
	}
	e.need[f] = true
	for _, p := range f.Params {
		if err := e.useType(p.Type, p.Decl.Start); err != nil {
			return err
		}
	}
	if _, void := f.Ret.(*wgsl.VoidType); !void {
		if err := e.useType(f.Ret, f.Decl.Start); err != nil {
			return err
		}
	}
	// The callees, from the body itself: the checker fills FuncObj.Calls only
	// for the functions of an entry point, and a library of functions has none.
	var callees []*wgsl.FuncObj
	walkStmt(f.Decl.Body, func(x wgsl.Expr) {
		if c, ok := x.(*wgsl.Call); ok {
			if g, ok := e.info.Objects[c.Fn].(*wgsl.FuncObj); ok {
				callees = append(callees, g)
			}
		}
	})
	for _, c := range callees {
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

// typeName is the MSL spelling of an in-scope type.
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

// reserved are the MSL and C++ words a WGSL identifier may be but an MSL one
// may not: emitted names are WGSL's own, so a collision is refused.
var reserved = map[string]bool{
	"auto": true, "bool2": true, "bool3": true, "bool4": true, "char": true, "class": true, "const": true,
	"constant": true, "default": true, "delete": true, "device": true, "double": true, "enum": true,
	"explicit": true, "extern": true, "float": true, "float2": true, "float3": true, "float4": true,
	"fragment": true, "goto": true, "half": true, "inline": true, "int": true, "int2": true, "int3": true,
	"int4": true, "kernel": true, "long": true, "metal": true, "namespace": true, "new": true,
	"operator": true, "private": true, "protected": true, "public": true, "register": true,
	"sampler": true, "short": true, "signed": true, "sizeof": true, "static": true, "template": true,
	"this": true, "thread": true, "threadgroup": true, "typedef": true, "typename": true, "uint": true,
	"uint2": true, "uint3": true, "uint4": true, "union": true, "unsigned": true, "using": true,
	"vertex": true, "virtual": true, "volatile": true, "INT_MIN": true, "select": true, "precise": true,
}

func checkName(n string, off int) (string, error) {
	if reserved[n] || strings.HasPrefix(n, "wgsl_") || strings.HasPrefix(n, "__") {
		return "", &Error{off, "the name " + n + " is reserved in MSL"}
	}
	return n, nil
}

func (e *emitter) emitStruct(b *strings.Builder, s *wgsl.StructType) error {
	name, err := checkName(s.Name, s.Decl.Start)
	if err != nil {
		return err
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

func (e *emitter) emitFunc(b *strings.Builder, f *wgsl.FuncObj) error {
	ret := "void"
	if _, void := f.Ret.(*wgsl.VoidType); !void {
		var err error
		if ret, err = typeName(f.Ret, f.Decl.Start); err != nil {
			return err
		}
	}
	name, err := checkName(f.Name, f.Decl.Start)
	if err != nil {
		return err
	}
	params := make([]string, len(f.Params))
	for i, p := range f.Params {
		t, err := typeName(p.Type, p.Decl.Start)
		if err != nil {
			return err
		}
		n, err := checkName(p.Name, p.Decl.Start)
		if err != nil {
			return err
		}
		params[i] = t + " " + n
	}
	fmt.Fprintf(b, "\nstatic %s %s(%s) ", ret, name, strings.Join(params, ", "))
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
	case *wgsl.DeclStmt:
		return e.decl(s)
	case *wgsl.Assign:
		return e.assign(s)
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

func (e *emitter) decl(s *wgsl.DeclStmt) error {
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
		return &Error{s.Start, fmt.Sprintf("declaration %T is not supported yet", d)}
	}
	if err := e.useType(t, off); err != nil {
		return err
	}
	tn, err := typeName(t, off)
	if err != nil {
		return err
	}
	n, err := checkName(name, off)
	if err != nil {
		return err
	}
	if init == nil { // WGSL zero-initialises a var
		e.line("%s %s = %s(0);", tn, n, tn)
		return nil
	}
	x, err := e.expr(init)
	if err != nil {
		return err
	}
	e.line("%s %s = %s;", tn, n, x)
	return nil
}

func (e *emitter) assign(s *wgsl.Assign) error {
	if s.LHS == nil {
		return &Error{s.Start, "a phony assignment is not supported yet"}
	}
	lhs, err := e.expr(s.LHS)
	if err != nil {
		return err
	}
	if s.Op == "=" {
		rhs, err := e.expr(s.RHS)
		if err != nil {
			return err
		}
		e.line("%s = %s;", lhs, rhs)
		return nil
	}
	op := strings.TrimSuffix(s.Op, "=")
	rhs, err := e.binary(op, s.LHS, s.RHS, e.valueType(s.LHS), s.Start)
	if err != nil {
		return err
	}
	e.line("%s = %s;", lhs, rhs)
	return nil
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
		return e.literal(x, t)
	case *wgsl.BoolLit:
		return strconv.FormatBool(x.Value), nil
	case *wgsl.Ident:
		switch o := e.info.Objects[x].(type) {
		case *wgsl.ParamObj:
			return checkName(o.Name, x.Start)
		case *wgsl.VarObj:
			if o.Global {
				return "", &Error{x.Start, "module-scope variable " + o.Name}
			}
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

func (e *emitter) literal(x wgsl.Expr, t wgsl.Type) (string, error) { return e.literalValue(x, t) }

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
			return "INT_MIN", nil // -2147483648 is not a literal in C++
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
		return "(" + x.Op + a + ")", nil
	}
	return "", &Error{x.Start, "unary " + x.Op + " is not supported yet"}
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
	call := func(f string) (string, error) {
		if !scalar {
			return "", &Error{off, "integer " + op + " on vectors is not supported yet"}
		}
		return f + "(" + a + ", " + b + ")", nil
	}
	switch op {
	case "+", "-", "*":
		if k == wgsl.KI32 {
			return call(map[string]string{"+": "wgsl_add", "-": "wgsl_sub", "*": "wgsl_mul"}[op])
		}
	case "/":
		switch k {
		case wgsl.KI32:
			return call("wgsl_div")
		case wgsl.KU32:
			return call("wgsl_udiv")
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
	case "==", "!=", "<", "<=", ">", ">=", "&&", "||", "&", "|", "^":
		if !scalar && (op == "&&" || op == "||") {
			return "", &Error{off, op + " on vectors"}
		}
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
		return n + "(" + all + ")", err
	case *wgsl.BuiltinObj:
		at := e.valueType(x.Args[0])
		k, _ := elemKind(at)
		float := k == wgsl.KF32
		switch o.Name {
		case "abs":
			if !float {
				return "", &Error{x.Start, "integer abs is not supported yet"}
			}
			return "precise::fabs(" + all + ")", nil
		case "sqrt":
			return "precise::sqrt(" + all + ")", nil
		case "floor":
			return "precise::floor(" + all + ")", nil
		case "min", "max":
			if float {
				return "precise::f" + o.Name + "(" + all + ")", nil
			}
			return o.Name + "(" + all + ")", nil
		case "clamp": // WGSL: min(max(e, low), high), defined when low > high
			if float {
				return "precise::fmin(precise::fmax(" + args[0] + ", " + args[1] + "), " + args[2] + ")", nil
			}
			return "min(max(" + args[0] + ", " + args[1] + "), " + args[2] + ")", nil
		case "select": // same argument order in both languages (MSL Table 6.3)
			return "select(" + all + ")", nil
		case "length":
			if _, ok := at.(*wgsl.VecType); !ok {
				return "precise::fabs(" + all + ")", nil
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
		return tn + "(0)", nil // the zero value
	}
	if x.Fn.Name == "bitcast" {
		// A reinterpretation of the bits, which is as_type in MSL (§2.23),
		// never a value conversion.
		return "as_type<" + tn + ">(" + args[0] + ")", nil
	}
	if s, ok := ct.(*wgsl.ScalarType); ok && len(x.Args) == 1 {
		from, _ := elemKind(e.valueType(x.Args[0]))
		switch {
		case (s.Kind == wgsl.KI32 || s.Kind == wgsl.KU32) && from == wgsl.KF32:
			// WGSL clamps an out-of-range float; C++ leaves it undefined.
			return "", &Error{x.Start, "conversion from f32 to an integer is not supported yet"}
		case (s.Kind == wgsl.KI32 && from == wgsl.KU32) || (s.Kind == wgsl.KU32 && from == wgsl.KI32):
			// WGSL reinterprets the bits between i32 and u32; C++17 makes
			// uint to int implementation-defined past INT_MAX.
			return "as_type<" + tn + ">(" + args[0] + ")", nil
		}
	}
	if _, ok := ct.(*wgsl.StructType); ok {
		return tn + "{" + strings.Join(args, ", ") + "}", nil
	}
	return tn + "(" + strings.Join(args, ", ") + ")", nil
}
