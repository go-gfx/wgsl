// Package spirv emits SPIR-V 1.5 (the Vulkan 1.2 environment) for the compute
// entry points of a checked WGSL module.
//
// It is the subset described in docs/SPIRV.md, enough for a fill kernel:
// compute entry points with @builtin inputs, storage and uniform buffers of
// group 0, scalar and vector arithmetic, comparisons, if, for, while and loop
// with break and continue, local variables and calls of other functions.
// Anything else is refused with its position (*Error), never emitted
// approximately.
//
// The module follows the logical layout of the SPIR-V specification, section
// 2.4, and its structured control flow, section 2.11 (docs/SPIRV.md names every
// section used). The instruction and enumerant numbers in ops_gen.go come from
// the Khronos grammar, not from this file.
package spirv

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/go-gfx/wgsl"
)

// Options tunes the emitted module.
type Options struct {
	// RoundingModeRTE declares the RoundingModeRTE execution mode for 32-bit
	// floats on every entry point, so that +, -, * and / are correctly rounded
	// (Vulkan: the device must report shaderRoundingModeRTEFloat32). Float
	// operations carry NoContraction either way.
	RoundingModeRTE bool
}

// Error is a construct the emitter refuses, with where it is.
type Error struct {
	Offset int
	What   string
}

func (e *Error) Error() string { return fmt.Sprintf("spirv: offset %d: %s", e.Offset, e.What) }

// Version is the SPIR-V version word of the emitted modules: 1.5.
const Version = 0x00010500

// Bytes is the module as a stream of little-endian bytes, the form a Vulkan
// implementation takes.
func Bytes(words []uint32) []byte {
	b := make([]byte, 4*len(words))
	for i, w := range words {
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(w), byte(w>>8), byte(w>>16), byte(w>>24)
	}
	return b
}

// Module emits the named compute entry points, with the functions they call
// and the buffers they reach. m must be the module info was checked from.
func Module(m *wgsl.Module, info *wgsl.Info, opts Options, entries ...string) ([]uint32, error) {
	e := &emitter{
		info: info, opts: opts, next: 1,
		types: map[string]uint32{}, consts: map[string]uint32{},
		fnID: map[*wgsl.FuncObj]uint32{}, glob: map[*wgsl.VarObj]*global{},
		locals: map[wgsl.Object]*local{},
	}
	byName := map[string]*wgsl.FuncObj{}
	for _, f := range info.Funcs {
		byName[f.Name] = f
	}
	var eps []*wgsl.FuncObj
	for _, n := range entries {
		f, ok := byName[n]
		if !ok {
			return nil, fmt.Errorf("spirv: no function %q in the module", n)
		}
		if f.Entry == nil {
			return nil, &Error{f.Decl.Start, "function " + f.Name + " is not an entry point"}
		}
		if f.Entry.Stage != wgsl.StageCompute {
			return nil, &Error{f.Decl.Start, "entry point " + f.Name + " is a " + f.Entry.Stage.String() + " shader; only compute is supported"}
		}
		if f.Entry.WorkgroupSizeExprs != [3]wgsl.Expr{} {
			return nil, &Error{f.Decl.Start, "entry point " + f.Name + " has an override-expression workgroup size"}
		}
		eps = append(eps, f)
	}
	if len(eps) == 0 {
		return nil, fmt.Errorf("spirv: no entry point named")
	}
	// The functions to emit: the entry points and everything they call.
	need := map[*wgsl.FuncObj]bool{}
	var mark func(f *wgsl.FuncObj) error
	mark = func(f *wgsl.FuncObj) error {
		if need[f] {
			return nil
		}
		if f.Entry != nil && !slices.Contains(eps, f) {
			return &Error{f.Decl.Start, "function " + f.Name + " is an entry point"}
		}
		need[f] = true
		var err error
		walkStmt(f.Decl.Body, func(x wgsl.Expr) {
			if c, ok := x.(*wgsl.Call); ok && err == nil {
				if g, ok := info.Objects[c.Fn].(*wgsl.FuncObj); ok {
					err = mark(g)
				}
			}
		})
		return err
	}
	for _, f := range eps {
		if err := mark(f); err != nil {
			return nil, err
		}
	}
	for _, f := range info.Funcs {
		if need[f] {
			e.fnID[f] = e.id()
		}
	}
	// Each function's body, then the module around them.
	for _, f := range info.Funcs {
		if !need[f] {
			continue
		}
		if err := e.function(f); err != nil {
			return nil, err
		}
	}
	e.reach = map[*wgsl.FuncObj][]uint32{}
	for _, f := range eps {
		seen := map[*wgsl.FuncObj]bool{}
		set := map[uint32]bool{}
		var visit func(g *wgsl.FuncObj)
		visit = func(g *wgsl.FuncObj) {
			if seen[g] {
				return
			}
			seen[g] = true
			for v := range e.uses[g] {
				set[v] = true
			}
			for _, c := range e.calls[g] {
				visit(c)
			}
		}
		visit(f)
		ids := make([]uint32, 0, len(set))
		for v := range set {
			ids = append(ids, v)
		}
		slices.Sort(ids)
		e.reach[f] = ids
	}
	return e.assemble(eps), nil
}

// emitter holds the sections of the module as the layout of section 2.4 orders
// them, and the caches that keep every type and constant declared once.
type emitter struct {
	info *wgsl.Info
	opts Options
	next uint32

	annot  []uint32 // decorations
	names  []uint32 // OpName
	decls  []uint32 // types, constants, global variables
	funcs  []uint32 // function definitions
	inputs []inputVar

	types  map[string]uint32
	consts map[string]uint32
	fnID   map[*wgsl.FuncObj]uint32
	glob   map[*wgsl.VarObj]*global
	locals map[wgsl.Object]*local
	fn     *fnState
	// uses and calls are recorded per function; reach is, per entry point, the
	// module-scope variables it and its callees use, which its OpEntryPoint
	// lists as the interface (section 2.4, OpEntryPoint, version 1.4 and later).
	uses  map[*wgsl.FuncObj]map[uint32]bool
	calls map[*wgsl.FuncObj][]*wgsl.FuncObj
	reach map[*wgsl.FuncObj][]uint32
}

// inputVar is a @builtin input: a module-scope variable of the Input class.
type inputVar struct {
	id      uint32
	builtin uint32
}

// global is a module-scope buffer: the variable, and what it points at.
type global struct {
	id      uint32 // the OpVariable
	inner   uint32 // type id of what the variable denotes: member 0 of the wrapper, or the structure itself
	sc      uint32 // storage class
	zero    uint32 // the constant 0, member 0's index
	wrapped bool   // the Block structure wraps one member: reach it with an access chain
}

// local is a function-scope name: an SSA value (let, parameter) or a variable.
type local struct {
	id    uint32
	ptr   bool
	typeT wgsl.Type
}

func (e *emitter) id() uint32 { e.next++; return e.next - 1 }

// inst appends an instruction to s: the word count and opcode, then operands.
func inst(s *[]uint32, op uint32, operands ...uint32) {
	*s = append(*s, uint32(len(operands)+1)<<16|op)
	*s = append(*s, operands...)
}

// str is a literal string: UTF-8, a zero byte, padded to whole words
// (section 2.2.1).
func str(s string) []uint32 {
	b := append([]byte(s), 0)
	for len(b)%4 != 0 {
		b = append(b, 0)
	}
	w := make([]uint32, len(b)/4)
	for i := range w {
		w[i] = uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24
	}
	return w
}

func (e *emitter) assemble(eps []*wgsl.FuncObj) []uint32 {
	var out []uint32
	out = append(out, Magic, Version, 0, 0, 0) // generator 0, bound and schema set below
	inst(&out, OpCapability, CapabilityShader)
	if e.opts.RoundingModeRTE {
		inst(&out, OpCapability, CapabilityRoundingModeRTE)
	}
	inst(&out, OpMemoryModel, AddressingModelLogical, MemoryModelGLSL450)
	for _, f := range eps {
		operands := append([]uint32{ExecutionModelGLCompute, e.fnID[f]}, str(f.Name)...)
		operands = append(operands, e.reach[f]...)
		inst(&out, OpEntryPoint, operands...)
	}
	for _, f := range eps {
		ws := f.Entry.WorkgroupSize
		inst(&out, OpExecutionMode, e.fnID[f], ExecutionModeLocalSize, uint32(ws[0]), uint32(ws[1]), uint32(ws[2]))
		if e.opts.RoundingModeRTE {
			inst(&out, OpExecutionMode, e.fnID[f], ExecutionModeRoundingModeRTE, 32)
		}
	}
	for _, f := range eps {
		inst(&out, OpName, append([]uint32{e.fnID[f]}, str(f.Name)...)...)
	}
	out = append(out, e.names...)
	out = append(out, e.annot...)
	out = append(out, e.decls...)
	out = append(out, e.funcs...)
	out[3] = e.next // the bound: one more than the largest id
	return out
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
	case *wgsl.IncDec:
		walkExpr(s.X, fn)
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

// ---- types and constants (section 2.4, step 9: declared before use) ----

func scalarKey(k wgsl.ScalarKind) (string, bool) {
	switch k {
	case wgsl.KBool:
		return "bool", true
	case wgsl.KI32:
		return "i32", true
	case wgsl.KU32:
		return "u32", true
	case wgsl.KF32:
		return "f32", true
	}
	return "", false
}

// typeID declares a scalar or vector type, once.
func (e *emitter) typeID(t wgsl.Type, off int) (uint32, error) {
	switch t := concrete(t).(type) {
	case *wgsl.ScalarType:
		key, ok := scalarKey(t.Kind)
		if !ok {
			return 0, &Error{off, "type " + t.String() + " is not supported yet"}
		}
		if id, ok := e.types[key]; ok {
			return id, nil
		}
		id := e.id()
		switch t.Kind {
		case wgsl.KBool:
			inst(&e.decls, OpTypeBool, id)
		case wgsl.KI32:
			inst(&e.decls, OpTypeInt, id, 32, 1)
		case wgsl.KU32:
			inst(&e.decls, OpTypeInt, id, 32, 0)
		case wgsl.KF32:
			inst(&e.decls, OpTypeFloat, id, 32)
		}
		e.types[key] = id
		return id, nil
	case *wgsl.VecType:
		el, err := e.typeID(t.Elem, off)
		if err != nil {
			return 0, err
		}
		key := fmt.Sprintf("vec%d.%d", t.N, el)
		if id, ok := e.types[key]; ok {
			return id, nil
		}
		id := e.id()
		inst(&e.decls, OpTypeVector, id, el, uint32(t.N))
		e.types[key] = id
		return id, nil
	case *wgsl.RefType:
		return e.typeID(t.Elem, off)
	}
	return 0, &Error{off, "type " + t.String() + " is not supported yet"}
}

func (e *emitter) voidID() uint32 {
	if id, ok := e.types["void"]; ok {
		return id
	}
	id := e.id()
	inst(&e.decls, OpTypeVoid, id)
	e.types["void"] = id
	return id
}

func (e *emitter) ptrID(sc, elem uint32) uint32 {
	key := fmt.Sprintf("ptr%d.%d", sc, elem)
	if id, ok := e.types[key]; ok {
		return id
	}
	id := e.id()
	inst(&e.decls, OpTypePointer, id, sc, elem)
	e.types[key] = id
	return id
}

// fnType declares the function type ret(params...).
func (e *emitter) fnType(ret uint32, params []uint32) uint32 {
	key := fmt.Sprintf("fn%d%v", ret, params)
	if id, ok := e.types[key]; ok {
		return id
	}
	id := e.id()
	inst(&e.decls, OpTypeFunction, append([]uint32{id, ret}, params...)...)
	e.types[key] = id
	return id
}

// scalarConst declares a 32-bit constant of type ty, with the bit pattern bits.
func (e *emitter) scalarConst(ty uint32, bits uint32) uint32 {
	key := fmt.Sprintf("c%d.%d", ty, bits)
	if id, ok := e.consts[key]; ok {
		return id
	}
	id := e.id()
	inst(&e.decls, OpConstant, ty, id, bits)
	e.consts[key] = id
	return id
}

func (e *emitter) boolConst(v bool) (uint32, error) {
	ty, _ := e.typeID(&wgsl.ScalarType{Kind: wgsl.KBool}, 0)
	key := fmt.Sprintf("b%v", v)
	if id, ok := e.consts[key]; ok {
		return id, nil
	}
	id := e.id()
	if v {
		inst(&e.decls, OpConstantTrue, ty, id)
	} else {
		inst(&e.decls, OpConstantFalse, ty, id)
	}
	e.consts[key] = id
	return id, nil
}

func (e *emitter) u32Const(v uint32) uint32 {
	ty, _ := e.typeID(&wgsl.ScalarType{Kind: wgsl.KU32}, 0)
	return e.scalarConst(ty, v)
}

// constant spells a const-expression value of type t (scalar or vector).
func (e *emitter) constant(c *wgsl.Const, t wgsl.Type, off int) (uint32, error) {
	switch t := t.(type) {
	case *wgsl.ScalarType:
		ty, err := e.typeID(t, off)
		if err != nil {
			return 0, err
		}
		switch t.Kind {
		case wgsl.KBool:
			return e.boolConst(c.Bool)
		case wgsl.KI32, wgsl.KU32:
			return e.scalarConst(ty, uint32(c.Int)), nil
		case wgsl.KF32:
			return e.scalarConst(ty, math.Float32bits(float32(c.Float))), nil
		}
	case *wgsl.VecType:
		ty, err := e.typeID(t, off)
		if err != nil {
			return 0, err
		}
		parts := make([]uint32, t.N)
		for i := range parts {
			el := c
			if len(c.Elems) == t.N {
				el = c.Elems[i]
			} else if len(c.Elems) != 0 || !c.Zero {
				return 0, &Error{off, "vector constant of an unexpected shape"}
			}
			if c.Zero && len(c.Elems) == 0 {
				el = &wgsl.Const{Type: t.Elem, Zero: true}
			}
			if parts[i], err = e.constant(el, t.Elem, off); err != nil {
				return 0, err
			}
		}
		key := fmt.Sprintf("cc%d%v", ty, parts)
		if id, ok := e.consts[key]; ok {
			return id, nil
		}
		id := e.id()
		inst(&e.decls, OpConstantComposite, append([]uint32{ty, id}, parts...)...)
		e.consts[key] = id
		return id, nil
	}
	return 0, &Error{off, "constant of type " + t.String() + " is not supported yet"}
}

// ---- buffers (module-scope variables) ----

// layoutOf is the size and alignment of a scalar or vector in a buffer.
func layoutOf(t wgsl.Type) (size, align int64, ok bool) {
	switch t := t.(type) {
	case *wgsl.ScalarType:
		return 4, 4, t.Kind != wgsl.KBool
	case *wgsl.VecType:
		switch t.N {
		case 2:
			return 8, 8, t.Elem.Kind != wgsl.KBool
		case 3:
			return 12, 16, t.Elem.Kind != wgsl.KBool
		}
		return 16, 16, t.Elem.Kind != wgsl.KBool
	}
	return 0, 0, false
}

// buffer declares module-scope variable v: a Block-decorated structure holding
// what v holds, in the StorageBuffer or Uniform class, bound at its group and
// binding. A bare runtime-sized array or scalar is wrapped in a structure of
// one member, as the specification's Block rule requires.
func (e *emitter) buffer(v *wgsl.VarObj) (*global, error) {
	if g, ok := e.glob[v]; ok {
		return g, nil
	}
	off := v.Decl.Start
	if v.AS != wgsl.ASStorage && v.AS != wgsl.ASUniform {
		return nil, &Error{off, "the " + v.AS.String() + " variable " + v.Name + " is not supported yet (only storage and uniform buffers)"}
	}
	if v.Group == nil || v.Binding == nil {
		return nil, &Error{off, "the buffer " + v.Name + " has no @group and @binding"}
	}
	sc := uint32(StorageClassStorageBuffer)
	if v.AS == wgsl.ASUniform {
		sc = StorageClassUniform
	}
	// What the wrapper's member 0 is, and its layout decorations.
	var inner uint32
	var memberDecor [][]uint32 // decorations on member 0 of the wrapper, or per member of a struct
	wrapper := e.id()
	var members []uint32
	switch st := v.Store.(type) {
	case *wgsl.ArrayType:
		if !st.Runtime || v.AS == wgsl.ASUniform {
			return nil, &Error{off, "the array " + v.Name + " is not supported yet (only runtime-sized storage arrays)"}
		}
		size, align, ok := layoutOf(st.Elem)
		if !ok {
			return nil, &Error{off, "the elements of " + v.Name + " are not scalars or vectors (not supported yet)"}
		}
		el, err := e.typeID(st.Elem, off)
		if err != nil {
			return nil, err
		}
		inner = e.id()
		inst(&e.decls, OpTypeRuntimeArray, inner, el)
		inst(&e.annot, OpDecorate, inner, DecorationArrayStride, uint32((size+align-1)/align*align))
		members = []uint32{inner}
		memberDecor = [][]uint32{{0}}
	case *wgsl.StructType:
		if st.Runtime {
			return nil, &Error{off, "the structure " + st.Name + " has a runtime-sized array (not supported yet)"}
		}
		for _, m := range st.Members {
			if _, _, ok := layoutOf(m.Type); !ok {
				return nil, &Error{m.Decl.Start, "member " + m.Name + " of " + st.Name + " is not a scalar or vector (not supported yet)"}
			}
			mt, err := e.typeID(m.Type, m.Decl.Start)
			if err != nil {
				return nil, err
			}
			members = append(members, mt)
			memberDecor = append(memberDecor, []uint32{uint32(m.Offset)})
		}
		inner = wrapper // the structure itself is the Block
	default:
		if _, _, ok := layoutOf(v.Store); !ok {
			return nil, &Error{off, "the buffer " + v.Name + " holds " + v.Store.String() + " (not supported yet)"}
		}
		mt, err := e.typeID(v.Store, off)
		if err != nil {
			return nil, err
		}
		inner = mt
		members = []uint32{mt}
		memberDecor = [][]uint32{{0}}
	}
	inst(&e.decls, OpTypeStruct, append([]uint32{wrapper}, members...)...)
	inst(&e.annot, OpDecorate, wrapper, DecorationBlock)
	for i, d := range memberDecor {
		inst(&e.annot, OpMemberDecorate, wrapper, uint32(i), DecorationOffset, d[0])
		if v.AS == wgsl.ASStorage && v.Access == wgsl.AccessRead {
			inst(&e.annot, OpMemberDecorate, wrapper, uint32(i), DecorationNonWritable)
		}
	}
	vid := e.id()
	inst(&e.decls, OpVariable, e.ptrID(sc, wrapper), vid, sc)
	inst(&e.annot, OpDecorate, vid, DecorationDescriptorSet, uint32(*v.Group))
	inst(&e.annot, OpDecorate, vid, DecorationBinding, uint32(*v.Binding))
	inst(&e.names, OpName, append([]uint32{vid}, str(v.Name)...)...)
	g := &global{id: vid, inner: inner, sc: sc, zero: e.u32Const(0), wrapped: true}
	if _, isStruct := v.Store.(*wgsl.StructType); isStruct {
		g.inner, g.wrapped = wrapper, false
	}
	e.glob[v] = g
	return g, nil
}

// builtinInputs maps an entry point's @builtin input to its decoration.
var builtinInputs = map[string]uint32{
	"global_invocation_id": BuiltInGlobalInvocationId,
	"local_invocation_id":  BuiltInLocalInvocationId,
	"workgroup_id":         BuiltInWorkgroupId,
}

// ---- helpers shared by the function emitter ----

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

func hasCall(x wgsl.Expr) bool {
	found := false
	walkExpr(x, func(y wgsl.Expr) {
		if _, ok := y.(*wgsl.Call); ok {
			found = true
		}
	})
	return found
}

// describe names a type for a message.
func describe(t wgsl.Type) string { return strings.TrimSpace(t.String()) }
