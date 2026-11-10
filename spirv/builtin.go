package spirv

import (
	"math"

	"github.com/go-gfx/wgsl"
)

// f2int is u32(f32) or i32(f32) of a scalar. WGSL (wgsl/index.bs, "Floating
// Point Conversion"): a NaN gives an indeterminate value; otherwise the result
// is the value of the integer type closest to truncate(x) that is also exactly
// representable in f32, so u32 saturates at 4294967040 and i32 at 2147483520
// and -2147483648. OpConvertFToU and OpConvertFToS are undefined outside the
// range of the destination, so x is clamped first with comparisons and selects
// (NaN-safe: a NaN fails both comparisons) and a NaN is given 0.
func (e *emitter) f2int(kind wgsl.ScalarKind, ty, x uint32) uint32 {
	fty, _ := e.typeID(&wgsl.ScalarType{Kind: wgsl.KF32}, 0)
	bty, _ := e.typeID(&wgsl.ScalarType{Kind: wgsl.KBool}, 0)
	lo, hi, conv := float32(0), float32(4294967040), uint32(OpConvertFToU)
	if kind == wgsl.KI32 {
		lo, hi, conv = -2147483648, 2147483520, OpConvertFToS
	}
	loC := e.scalarConst(fty, math.Float32bits(lo))
	hiC := e.scalarConst(fty, math.Float32bits(hi))
	zeroC := e.scalarConst(fty, 0)
	body := &e.fn.body
	isNaN := e.id()
	inst(body, OpIsNan, bty, isNaN, x)
	below, above := e.id(), e.id()
	inst(body, OpFOrdLessThan, bty, below, x, loC)
	inst(body, OpFOrdGreaterThan, bty, above, x, hiC)
	t1, t2, t3 := e.id(), e.id(), e.id()
	inst(body, OpSelect, fty, t1, below, loC, x)
	inst(body, OpSelect, fty, t2, above, hiC, t1)
	inst(body, OpSelect, fty, t3, isNaN, zeroC, t2) // never convert a NaN
	r := e.id()
	inst(body, conv, ty, r, t3)
	return r
}

// builtin emits a built-in function other than select. The float functions
// are the GLSL.std.450 ones; WGSL defines clamp as min(max(e, low), high), which
// is how it is emitted.
func (e *emitter) builtin(x *wgsl.Call, name string) (uint32, error) {
	rt := concrete(e.valueType(x))
	ty, err := e.typeID(rt, x.Start)
	if err != nil {
		return 0, err
	}
	args := make([]uint32, len(x.Args))
	for i, a := range x.Args {
		if args[i], err = e.rvalue(a); err != nil {
			return 0, err
		}
	}
	k, scalar := elemKind(concrete(e.valueType(x.Args[0])))
	ext := func(op uint32, operands ...uint32) uint32 {
		r := e.id()
		inst(&e.fn.body, OpExtInst, append([]uint32{ty, r, e.glslID(), op}, operands...)...)
		return r
	}
	pick := func(f, s, u uint32) (uint32, bool) {
		switch k {
		case wgsl.KF32:
			return f, true
		case wgsl.KI32:
			return s, true
		case wgsl.KU32:
			return u, true
		}
		return 0, false
	}
	unsupported := &Error{x.Start, "builtin " + name + " of " + describe(rt) + " is not supported yet"}
	switch name {
	case "abs":
		switch {
		case k == wgsl.KF32:
			return ext(glslFAbs, args[0]), nil
		case k == wgsl.KU32:
			return args[0], nil
		case k == wgsl.KI32 && scalar:
			// abs(i32::MIN) is i32::MIN in WGSL; SAbs leaves it undefined, so
			// select(e, 0 - e, e < 0) with a wrapping subtraction.
			bty, _ := e.typeID(&wgsl.ScalarType{Kind: wgsl.KBool}, 0)
			zero := e.scalarConst(ty, 0)
			neg, lt, r := e.id(), e.id(), e.id()
			inst(&e.fn.body, OpISub, ty, neg, zero, args[0])
			inst(&e.fn.body, OpSLessThan, bty, lt, args[0], zero)
			inst(&e.fn.body, OpSelect, ty, r, lt, neg, args[0])
			return r, nil
		}
	case "min", "max":
		f, s, u := uint32(glslFMin), uint32(glslSMin), uint32(glslUMin)
		if name == "max" {
			f, s, u = glslFMax, glslSMax, glslUMax
		}
		if op, ok := pick(f, s, u); ok {
			return ext(op, args[0], args[1]), nil
		}
	case "clamp":
		mx, _ := pick(glslFMax, glslSMax, glslUMax)
		mn, _ := pick(glslFMin, glslSMin, glslUMin)
		if k == wgsl.KF32 || k == wgsl.KI32 || k == wgsl.KU32 {
			lowered := ext(mx, args[0], args[1])
			return ext(mn, lowered, args[2]), nil
		}
	case "sqrt", "floor":
		if k == wgsl.KF32 {
			op := uint32(glslSqrt)
			if name == "floor" {
				op = glslFloor
			}
			return ext(op, args[0]), nil
		}
	case "length":
		vt, ok := concrete(e.valueType(x.Args[0])).(*wgsl.VecType)
		if !ok || vt.Elem.Kind != wgsl.KF32 {
			break
		}
		// The sum of squares in the order of WGSL's definition, each step
		// rounded to f32 and never fused, then the square root.
		ety, err := e.typeID(vt.Elem, x.Start)
		if err != nil {
			return 0, err
		}
		var sum uint32
		for i := 0; i < vt.N; i++ {
			c := e.id()
			inst(&e.fn.body, OpCompositeExtract, ety, c, args[0], uint32(i))
			sq := e.id()
			inst(&e.fn.body, OpFMul, ety, sq, c, c)
			inst(&e.annot, OpDecorate, sq, DecorationNoContraction)
			if i == 0 {
				sum = sq
				continue
			}
			s := e.id()
			inst(&e.fn.body, OpFAdd, ety, s, sum, sq)
			inst(&e.annot, OpDecorate, s, DecorationNoContraction)
			sum = s
		}
		r := e.id()
		inst(&e.fn.body, OpExtInst, ty, r, e.glslID(), glslSqrt, sum)
		return r, nil
	default:
		return 0, &Error{x.Start, "builtin " + name + " is not supported yet"}
	}
	return 0, unsupported
}
