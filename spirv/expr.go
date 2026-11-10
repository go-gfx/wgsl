package spirv

import (
	"fmt"
	"strings"

	"github.com/go-gfx/wgsl"
)

// place is a memory location: a pointer id, what it points at in SPIR-V and in
// WGSL.
type place struct {
	id   uint32
	elem uint32 // type id of the pointee
	sc   uint32
	t    wgsl.Type
}

// concrete gives an abstract numeric type the concrete type WGSL defaults it
// to, so that a constant the checker left abstract still has a SPIR-V type.
func concrete(t wgsl.Type) wgsl.Type {
	switch t := t.(type) {
	case *wgsl.ScalarType:
		switch t.Kind {
		case wgsl.KAbstractInt:
			return &wgsl.ScalarType{Kind: wgsl.KI32}
		case wgsl.KAbstractFloat:
			return &wgsl.ScalarType{Kind: wgsl.KF32}
		}
	case *wgsl.VecType:
		if c := concrete(t.Elem).(*wgsl.ScalarType); c != t.Elem {
			return &wgsl.VecType{N: t.N, Elem: c}
		}
	}
	return t
}

// ref evaluates an expression that denotes memory.
func (e *emitter) ref(x wgsl.Expr) (*place, error) {
	switch x := x.(type) {
	case *wgsl.Paren:
		return e.ref(x.X)
	case *wgsl.Ident:
		switch o := e.info.Objects[x].(type) {
		case *wgsl.VarObj:
			if !o.Global {
				l := e.locals[o]
				ty, err := e.typeID(l.typeT, x.Start)
				if err != nil {
					return nil, err
				}
				return &place{id: l.id, elem: ty, sc: StorageClassFunction, t: l.typeT}, nil
			}
			g, err := e.buffer(o)
			if err != nil {
				return nil, err
			}
			e.fn.uses[g.id] = true
			if !g.wrapped {
				return &place{id: g.id, elem: g.inner, sc: g.sc, t: o.Store}, nil
			}
			r := e.id()
			inst(&e.fn.body, OpAccessChain, e.ptrID(g.sc, g.inner), r, g.id, g.zero)
			return &place{id: r, elem: g.inner, sc: g.sc, t: o.Store}, nil
		}
		return nil, &Error{x.Start, x.Name + " is not a variable"}
	case *wgsl.Member:
		base, err := e.ref(x.X)
		if err != nil {
			return nil, err
		}
		switch bt := base.t.(type) {
		case *wgsl.StructType:
			for i, m := range bt.Members {
				if m.Name != x.Name {
					continue
				}
				mt, err := e.typeID(m.Type, x.Start)
				if err != nil {
					return nil, err
				}
				return e.chain(base, mt, m.Type, e.u32Const(uint32(i))), nil
			}
			return nil, &Error{x.Start, "no member " + x.Name}
		case *wgsl.VecType:
			i := strings.Index("xyzw", x.Name)
			if i < 0 || len(x.Name) != 1 {
				i = strings.Index("rgba", x.Name)
			}
			if i < 0 || len(x.Name) != 1 {
				return nil, &Error{x.Start, "a swizzle of a reference (" + x.Name + ") is not supported"}
			}
			et, err := e.typeID(bt.Elem, x.Start)
			if err != nil {
				return nil, err
			}
			return e.chain(base, et, bt.Elem, e.u32Const(uint32(i))), nil
		}
		return nil, &Error{x.Start, "member access on " + describe(base.t) + " is not supported yet"}
	case *wgsl.Index:
		base, err := e.ref(x.X)
		if err != nil {
			return nil, err
		}
		var elem wgsl.Type
		switch bt := base.t.(type) {
		case *wgsl.ArrayType:
			elem = bt.Elem
		case *wgsl.VecType:
			elem = bt.Elem
		default:
			return nil, &Error{x.Start, "indexing " + describe(base.t) + " is not supported yet"}
		}
		et, err := e.memTypeID(elem, x.Start) // an element of a buffer array may be a laid-out structure
		if err != nil {
			return nil, err
		}
		i, err := e.rvalue(x.I)
		if err != nil {
			return nil, err
		}
		return e.chain(base, et, elem, i), nil
	}
	return nil, &Error{x.NodeSpan().Start, fmt.Sprintf("expression %T is not an lvalue", x)}
}

func (e *emitter) chain(base *place, elem uint32, t wgsl.Type, idx uint32) *place {
	r := e.id()
	inst(&e.fn.body, OpAccessChain, e.ptrID(base.sc, elem), r, base.id, idx)
	return &place{id: r, elem: elem, sc: base.sc, t: t}
}

// load reads a scalar or vector from p.
func (e *emitter) load(p *place, x wgsl.Expr) (uint32, error) {
	switch pt := p.t.(type) {
	case *wgsl.ScalarType, *wgsl.VecType:
	case *wgsl.StructType:
		// A whole structure of scalars and vectors: each member is loaded through
		// its own access chain and the value is built in the undecorated struct type.
		ty, err := e.typeID(pt, x.NodeSpan().Start)
		if err != nil {
			return 0, err
		}
		vals := make([]uint32, len(pt.Members))
		for i, m := range pt.Members {
			mt, err := e.typeID(m.Type, x.NodeSpan().Start)
			if err != nil {
				return 0, err
			}
			mp := e.chain(p, mt, m.Type, e.u32Const(uint32(i)))
			vals[i] = e.id()
			inst(&e.fn.body, OpLoad, mt, vals[i], mp.id)
		}
		r := e.id()
		inst(&e.fn.body, OpCompositeConstruct, append([]uint32{ty, r}, vals...)...)
		return r, nil
	default:
		return 0, &Error{x.NodeSpan().Start, "loading " + describe(p.t) + " is not supported yet"}
	}
	r := e.id()
	inst(&e.fn.body, OpLoad, p.elem, r, p.id)
	return r, nil
}

// rvalue evaluates an expression to the id of its value.
func (e *emitter) rvalue(x wgsl.Expr) (uint32, error) {
	t := concrete(e.valueType(x))
	if c, ok := e.info.Consts[x]; ok && c != nil {
		if id, err := e.constant(c, t, x.NodeSpan().Start); err == nil {
			return id, nil
		}
	}
	if _, isRef := e.info.Types[x].(*wgsl.RefType); isRef {
		p, err := e.ref(x)
		if err != nil {
			return 0, err
		}
		return e.load(p, x)
	}
	switch x := x.(type) {
	case *wgsl.Paren:
		return e.rvalue(x.X)
	case *wgsl.BoolLit:
		return e.boolConst(x.Value)
	case *wgsl.Ident:
		switch o := e.info.Objects[x].(type) {
		case *wgsl.ParamObj, *wgsl.ValueObj:
			if l, ok := e.locals[o]; ok {
				return l.id, nil
			}
		}
		return 0, &Error{x.Start, "identifier " + x.Name + " is not supported yet"}
	case *wgsl.Member:
		return e.extract(x, t)
	case *wgsl.Unary:
		return e.unary(x, t)
	case *wgsl.Binary:
		return e.binary(x, t)
	case *wgsl.Call:
		return e.call(x)
	}
	return 0, &Error{x.NodeSpan().Start, fmt.Sprintf("expression %T is not supported yet", x)}
}

// extract is v.x or a swizzle of a vector value.
func (e *emitter) extract(x *wgsl.Member, t wgsl.Type) (uint32, error) {
	v, err := e.rvalue(x.X)
	if err != nil {
		return 0, err
	}
	if st, ok := e.valueType(x.X).(*wgsl.StructType); ok {
		for i, m := range st.Members {
			if m.Name == x.Name {
				ty, err := e.typeID(t, x.Start)
				if err != nil {
					return 0, err
				}
				r := e.id()
				inst(&e.fn.body, OpCompositeExtract, ty, r, v, uint32(i))
				return r, nil
			}
		}
		return 0, &Error{x.Start, "no member " + x.Name}
	}
	if _, ok := e.valueType(x.X).(*wgsl.VecType); !ok {
		return 0, &Error{x.Start, "member access on a " + describe(e.valueType(x.X)) + " value is not supported yet"}
	}
	var idx []uint32
	for _, c := range x.Name {
		i := strings.IndexRune("xyzw", c)
		if i < 0 {
			i = strings.IndexRune("rgba", c)
		}
		if i < 0 {
			return 0, &Error{x.Start, "swizzle " + x.Name}
		}
		idx = append(idx, uint32(i))
	}
	ty, err := e.typeID(t, x.Start)
	if err != nil {
		return 0, err
	}
	r := e.id()
	if len(idx) == 1 {
		inst(&e.fn.body, OpCompositeExtract, ty, r, v, idx[0])
	} else {
		inst(&e.fn.body, OpVectorShuffle, append([]uint32{ty, r, v, v}, idx...)...)
	}
	return r, nil
}

func (e *emitter) unary(x *wgsl.Unary, t wgsl.Type) (uint32, error) {
	a, err := e.rvalue(x.X)
	if err != nil {
		return 0, err
	}
	ty, err := e.typeID(t, x.Start)
	if err != nil {
		return 0, err
	}
	k, _ := elemKind(t)
	r := e.id()
	switch {
	case x.Op == "-" && k == wgsl.KF32:
		inst(&e.fn.body, OpFNegate, ty, r, a)
		inst(&e.annot, OpDecorate, r, DecorationNoContraction)
	case x.Op == "-" && k == wgsl.KI32:
		inst(&e.fn.body, OpSNegate, ty, r, a)
	case x.Op == "!" && k == wgsl.KBool:
		inst(&e.fn.body, OpLogicalNot, ty, r, a)
	case x.Op == "~" && (k == wgsl.KI32 || k == wgsl.KU32):
		inst(&e.fn.body, OpNot, ty, r, a)
	default:
		return 0, &Error{x.Start, "unary " + x.Op + " on " + describe(t) + " is not supported yet"}
	}
	return r, nil
}

func (e *emitter) binary(x *wgsl.Binary, t wgsl.Type) (uint32, error) {
	a, err := e.rvalue(x.X)
	if err != nil {
		return 0, err
	}
	b, err := e.rvalue(x.Y)
	if err != nil {
		return 0, err
	}
	return e.binop(x.Op, concrete(e.valueType(x.X)), concrete(e.valueType(x.Y)), a, b, t, x.Y, x.Start)
}

var intCompare = map[string][2]uint32{ // signed, unsigned
	"==": {OpIEqual, OpIEqual}, "!=": {OpINotEqual, OpINotEqual},
	"<": {OpSLessThan, OpULessThan}, "<=": {OpSLessThanEqual, OpULessThanEqual},
	">": {OpSGreaterThan, OpUGreaterThan}, ">=": {OpSGreaterThanEqual, OpUGreaterThanEqual},
}

// Ordered comparisons are false for a NaN; != is true for one (WGSL follows
// IEEE 754), which is OpFUnordNotEqual and not OpFOrdNotEqual.
var floatCompare = map[string]uint32{
	"==": OpFOrdEqual, "!=": OpFUnordNotEqual, "<": OpFOrdLessThan, "<=": OpFOrdLessThanEqual,
	">": OpFOrdGreaterThan, ">=": OpFOrdGreaterThanEqual,
}

// binop applies a WGSL binary operator to two evaluated operands of types ta
// and tb, giving a value of type rt.
func (e *emitter) binop(op string, ta, tb wgsl.Type, a, b uint32, rt wgsl.Type, rhs wgsl.Expr, off int) (uint32, error) {
	rt = concrete(rt)
	ty, err := e.typeID(rt, off)
	if err != nil {
		return 0, err
	}
	ka, scalarA := elemKind(ta)
	_, scalarB := elemKind(tb)
	bodyOp := func(code uint32, x, y uint32, rty uint32) uint32 {
		r := e.id()
		inst(&e.fn.body, code, rty, r, x, y)
		return r
	}
	// vector OP scalar (and scalar OP vector): the scalar is repeated.
	if scalarA != scalarB && isArith(op) {
		if scalarA {
			a = e.splat(rt, a)
		} else {
			b = e.splat(rt, b)
		}
		scalarA = false
	}
	switch op {
	case "+", "-", "*":
		switch ka {
		case wgsl.KI32, wgsl.KU32:
			return bodyOp(map[string]uint32{"+": OpIAdd, "-": OpISub, "*": OpIMul}[op], a, b, ty), nil
		case wgsl.KF32:
			r := bodyOp(map[string]uint32{"+": OpFAdd, "-": OpFSub, "*": OpFMul}[op], a, b, ty)
			inst(&e.annot, OpDecorate, r, DecorationNoContraction)
			return r, nil
		}
	case "/":
		switch {
		case ka == wgsl.KF32:
			r := bodyOp(OpFDiv, a, b, ty)
			inst(&e.annot, OpDecorate, r, DecorationNoContraction)
			return r, nil
		case ka == wgsl.KU32 && scalarA:
			return e.udivrem(op, a, b, ty), nil
		case ka == wgsl.KI32 && scalarA:
			return e.sdivrem(op, a, b, ty), nil
		}
		return 0, &Error{off, "division of " + describe(ta) + " is not supported yet"}
	case "%":
		if ka == wgsl.KU32 && scalarA {
			return e.udivrem(op, a, b, ty), nil
		}
		if ka == wgsl.KI32 && scalarA {
			return e.sdivrem(op, a, b, ty), nil
		}
		return 0, &Error{off, "remainder of " + describe(ta) + " is not supported yet"}
	case "&", "|", "^":
		switch {
		case ka == wgsl.KBool && op != "^":
			return bodyOp(map[string]uint32{"&": OpLogicalAnd, "|": OpLogicalOr}[op], a, b, ty), nil
		case ka == wgsl.KI32 || ka == wgsl.KU32:
			return bodyOp(map[string]uint32{"&": OpBitwiseAnd, "|": OpBitwiseOr, "^": OpBitwiseXor}[op], a, b, ty), nil
		}
	case "&&", "||":
		// WGSL short-circuits; OpLogicalAnd and OpLogicalOr do not, which is the
		// same only when the right operand has no side effect to skip.
		if e.hasUserCall(rhs) {
			return 0, &Error{off, "a call on the right of " + op + " is not supported yet"}
		}
		return bodyOp(map[string]uint32{"&&": OpLogicalAnd, "||": OpLogicalOr}[op], a, b, ty), nil
	case "<<", ">>":
		if !scalarA || (ka != wgsl.KI32 && ka != wgsl.KU32) {
			return 0, &Error{off, "shift of " + describe(ta) + " is not supported yet"}
		}
		// The count is taken modulo 32 (the bit width), as WGSL defines.
		u32 := e.u32Const(31)
		cty, _ := e.typeID(&wgsl.ScalarType{Kind: wgsl.KU32}, off)
		count := bodyOp(OpBitwiseAnd, b, u32, cty)
		code := uint32(OpShiftLeftLogical)
		if op == ">>" {
			code = OpShiftRightLogical
			if ka == wgsl.KI32 {
				code = OpShiftRightArithmetic
			}
		}
		return bodyOp(code, a, count, ty), nil
	case "==", "!=", "<", "<=", ">", ">=":
		if !scalarA {
			return 0, &Error{off, "comparison of vectors is not supported yet"}
		}
		switch ka {
		case wgsl.KBool:
			if op == "==" {
				return bodyOp(OpLogicalEqual, a, b, ty), nil
			}
			if op == "!=" {
				return bodyOp(OpLogicalNotEqual, a, b, ty), nil
			}
		case wgsl.KI32:
			return bodyOp(intCompare[op][0], a, b, ty), nil
		case wgsl.KU32:
			return bodyOp(intCompare[op][1], a, b, ty), nil
		case wgsl.KF32:
			return bodyOp(floatCompare[op], a, b, ty), nil
		}
	}
	return 0, &Error{off, "operator " + op + " on " + describe(ta) + " is not supported yet"}
}

func isArith(op string) bool { return op == "+" || op == "-" || op == "*" || op == "/" }

// splat repeats scalar s into a vector of type vt.
func (e *emitter) splat(vt wgsl.Type, s uint32) uint32 {
	v := vt.(*wgsl.VecType)
	ty, _ := e.typeID(vt, 0)
	parts := make([]uint32, v.N)
	for i := range parts {
		parts[i] = s
	}
	r := e.id()
	inst(&e.fn.body, OpCompositeConstruct, append([]uint32{ty, r}, parts...)...)
	return r
}

// udivrem is u32 division or remainder with WGSL's definition for a zero
// divisor (the dividend, and 0), which OpUDiv and OpUMod leave undefined: the
// divisor is replaced by 1 when it is 0 and the result selected back.
func (e *emitter) udivrem(op string, a, b, ty uint32) uint32 {
	bty, _ := e.typeID(&wgsl.ScalarType{Kind: wgsl.KBool}, 0)
	zero, one := e.scalarConst(ty, 0), e.scalarConst(ty, 1)
	isZero := e.id()
	inst(&e.fn.body, OpIEqual, bty, isZero, b, zero)
	d := e.id()
	inst(&e.fn.body, OpSelect, ty, d, isZero, one, b)
	q := e.id()
	code, special := uint32(OpUDiv), a
	if op == "%" {
		code, special = OpUMod, zero
	}
	inst(&e.fn.body, code, ty, q, a, d)
	r := e.id()
	inst(&e.fn.body, OpSelect, ty, r, isZero, special, q)
	return r
}

// sdivrem is i32 division or remainder with WGSL's definition of the two cases
// OpSDiv and OpSRem leave undefined (wgsl/index.bs, "Arithmetic Expressions"):
// a zero divisor, and the most negative value divided by -1. In both the
// divisor is replaced by 1, which gives e1 for / (the result WGSL defines) and
// 0 for % (likewise). Otherwise OpSDiv truncates toward zero and OpSRem takes
// the sign of the dividend, as WGSL does.
func (e *emitter) sdivrem(op string, a, b, ty uint32) uint32 {
	bty, _ := e.typeID(&wgsl.ScalarType{Kind: wgsl.KBool}, 0)
	zero, one := e.scalarConst(ty, 0), e.scalarConst(ty, 1)
	minInt, negOne := e.scalarConst(ty, 0x80000000), e.scalarConst(ty, 0xFFFFFFFF)
	bz, amin, bneg := e.id(), e.id(), e.id()
	inst(&e.fn.body, OpIEqual, bty, bz, b, zero)
	inst(&e.fn.body, OpIEqual, bty, amin, a, minInt)
	inst(&e.fn.body, OpIEqual, bty, bneg, b, negOne)
	ov, bad := e.id(), e.id()
	inst(&e.fn.body, OpLogicalAnd, bty, ov, amin, bneg)
	inst(&e.fn.body, OpLogicalOr, bty, bad, bz, ov)
	d := e.id()
	inst(&e.fn.body, OpSelect, ty, d, bad, one, b)
	code := uint32(OpSDiv)
	if op == "%" {
		code = OpSRem
	}
	r := e.id()
	inst(&e.fn.body, code, ty, r, a, d)
	return r
}

// call is a call of a function, a constructor, a conversion or a built-in.
func (e *emitter) call(x *wgsl.Call) (uint32, error) {
	if ct, ok := e.info.TypeSpecs[x.Fn]; ok {
		return e.construct(x, concrete(ct))
	}
	switch o := e.info.Objects[x.Fn].(type) {
	case *wgsl.FuncObj:
		if o.Entry != nil {
			return 0, &Error{x.Start, "an entry point cannot be called"}
		}
		args := make([]uint32, len(x.Args))
		for i, a := range x.Args {
			v, err := e.rvalue(a)
			if err != nil {
				return 0, err
			}
			args[i] = v
		}
		var rty uint32
		if _, void := o.Ret.(*wgsl.VoidType); void {
			rty = e.voidID()
		} else {
			var err error
			if rty, err = e.typeID(o.Ret, x.Start); err != nil {
				return 0, err
			}
		}
		e.fn.calls = append(e.fn.calls, o)
		r := e.id()
		inst(&e.fn.body, OpFunctionCall, append([]uint32{rty, r, e.fnID[o]}, args...)...)
		return r, nil
	case *wgsl.BuiltinObj:
		if o.Name == "select" && len(x.Args) == 3 {
			f, err := e.rvalue(x.Args[0])
			if err != nil {
				return 0, err
			}
			t, err := e.rvalue(x.Args[1])
			if err != nil {
				return 0, err
			}
			c, err := e.rvalue(x.Args[2])
			if err != nil {
				return 0, err
			}
			ty, err := e.typeID(concrete(e.valueType(x.Args[0])), x.Start)
			if err != nil {
				return 0, err
			}
			r := e.id()
			inst(&e.fn.body, OpSelect, ty, r, c, t, f) // condition, then the true value, then the false one
			return r, nil
		}
		return e.builtin(x, o.Name)
	}
	return 0, &Error{x.Start, "call of " + x.Fn.Name + " is not supported yet"}
}

func (e *emitter) construct(x *wgsl.Call, ct wgsl.Type) (uint32, error) {
	ty, err := e.typeID(ct, x.Start)
	if err != nil {
		return 0, err
	}
	if len(x.Args) == 0 {
		return e.zero(ct, x.Start)
	}
	args := make([]uint32, len(x.Args))
	for i, a := range x.Args {
		if args[i], err = e.rvalue(a); err != nil {
			return 0, err
		}
	}
	if x.Fn.Name == "bitcast" {
		r := e.id()
		inst(&e.fn.body, OpBitcast, ty, r, args[0])
		return r, nil
	}
	switch ct := ct.(type) {
	case *wgsl.ScalarType:
		from, _ := elemKind(concrete(e.valueType(x.Args[0])))
		switch {
		case from == ct.Kind:
			return args[0], nil
		case (ct.Kind == wgsl.KI32 && from == wgsl.KU32) || (ct.Kind == wgsl.KU32 && from == wgsl.KI32):
			r := e.id()
			inst(&e.fn.body, OpBitcast, ty, r, args[0]) // WGSL reinterprets the bits
			return r, nil
		case ct.Kind == wgsl.KF32 && (from == wgsl.KU32 || from == wgsl.KI32):
			code := uint32(OpConvertUToF)
			if from == wgsl.KI32 {
				code = OpConvertSToF
			}
			r := e.id()
			inst(&e.fn.body, code, ty, r, args[0])
			return r, nil
		case (ct.Kind == wgsl.KU32 || ct.Kind == wgsl.KI32) && from == wgsl.KF32:
			if _, scalar := elemKind(concrete(e.valueType(x.Args[0]))); scalar {
				return e.f2int(ct.Kind, ty, args[0]), nil
			}
		}
		return 0, &Error{x.Start, "conversion to " + describe(ct) + " is not supported yet"}
	case *wgsl.VecType:
		for _, a := range x.Args {
			at := concrete(e.valueType(a))
			if k, scalar := elemKind(at); !scalar || k != ct.Elem.Kind {
				if at == wgsl.Type(ct) || describe(at) == describe(ct) {
					return args[0], nil
				}
				return 0, &Error{x.Start, "constructing " + describe(ct) + " from " + describe(at) + " is not supported yet"}
			}
		}
		if len(args) == 1 {
			return e.splat(ct, args[0]), nil
		}
		if len(args) != ct.N {
			return 0, &Error{x.Start, "constructing " + describe(ct) + " from " + fmt.Sprint(len(args)) + " values is not supported yet"}
		}
		r := e.id()
		inst(&e.fn.body, OpCompositeConstruct, append([]uint32{ty, r}, args...)...)
		return r, nil
	}
	return 0, &Error{x.Start, "constructing " + describe(ct) + " is not supported yet"}
}
