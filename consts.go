package wgsl

import (
	"fmt"
	"math"
	"math/bits"
	"strconv"
	"strings"
)

// Const is the value of a const-expression (spec "const Expressions"). The
// representation is exact for the abstract types: AbstractInt is a 64-bit
// two's complement integer, AbstractFloat an IEEE binary64 value.
//
// Composite values (vectors, matrices, arrays, structures) hold their
// components in Elems, except that a zero-valued composite is kept lazily
// (Zero) so that a const-expression such as array<i32, 1000000000>() costs
// no memory.
type Const struct {
	Type  Type
	Bool  bool
	Int   int64   // AbstractInt, i32 (sign-extended) and u32 (0..2^32-1)
	Float float64 // AbstractFloat, f32 and f16 (always exactly representable)
	Elems []*Const
	Zero  bool
}

func (c *Const) String() string {
	if c == nil {
		return "<unknown>"
	}
	switch t := c.Type.(type) {
	case *ScalarType:
		switch t.Kind {
		case KBool:
			return strconv.FormatBool(c.Bool)
		case KAbstractInt, KI32:
			return strconv.FormatInt(c.Int, 10)
		case KU32:
			return strconv.FormatInt(c.Int, 10) + "u"
		}
		return strconv.FormatFloat(c.Float, 'g', -1, 64)
	}
	if c.Zero {
		return c.Type.String() + "()"
	}
	parts := make([]string, len(c.Elems))
	for i, e := range c.Elems {
		parts[i] = e.String()
	}
	return c.Type.String() + "(" + strings.Join(parts, ", ") + ")"
}

// evalError is a failed const-evaluation (overflow, division by zero, ...).
type evalError struct{ Rule, Msg string }

func evalErrf(rule, f string, a ...any) *evalError {
	return &evalError{rule, fmt.Sprintf(f, a...)}
}

func (e *evalError) Error() string { return e.Msg }

// numElems returns the number of top-level components of a composite type
// (0 for scalars). It is the value of N for vecN, C for matCxR, the element
// count for fixed arrays and the member count for structures.
func numElems(t Type) int64 {
	switch t := t.(type) {
	case *VecType:
		return int64(t.N)
	case *MatType:
		return int64(t.C)
	case *ArrayType:
		return t.Count
	case *StructType:
		return int64(len(t.Members))
	}
	return 0
}

// elemType returns the type of component i of a composite type.
func elemType(t Type, i int64) Type {
	switch t := t.(type) {
	case *VecType:
		return t.Elem
	case *MatType:
		return vec(t.R, t.Elem)
	case *ArrayType:
		return t.Elem
	case *StructType:
		return t.Members[i].Type
	}
	return Invalid
}

// Elem returns component i of a composite constant.
func (c *Const) Elem(i int64) *Const {
	if c.Zero {
		return zeroConst(elemType(c.Type, i))
	}
	return c.Elems[i]
}

// zeroConst returns the zero value of t (spec "Zero Value Built-in
// Functions").
func zeroConst(t Type) *Const {
	if s, ok := t.(*ScalarType); ok {
		return &Const{Type: s}
	}
	return &Const{Type: t, Zero: true}
}

func boolConst(b bool) *Const { return &Const{Type: Bool, Bool: b} }

func intConst(t *ScalarType, v int64) *Const { return &Const{Type: t, Int: v} }

func floatConst(t *ScalarType, v float64) *Const { return &Const{Type: t, Float: v} }

// ---- float rounding ----

// roundF16 rounds x to the nearest binary16 value (ties to even). ok is
// false when the result would be infinite or x is not finite.
func roundF16(x float64) (float64, bool) {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return x, false
	}
	ax := math.Abs(x)
	if ax == 0 {
		return x, true
	}
	if ax >= 65520 {
		return x, false
	}
	e := math.Ilogb(ax)
	if e < -14 {
		e = -14
	}
	q := math.Ldexp(1, e-10)
	r := math.RoundToEven(ax/q) * q
	return math.Copysign(r, x), true
}

// roundTo rounds x to the floating point type t, reporting false on
// overflow.
func roundTo(t *ScalarType, x float64) (float64, bool) {
	switch t.Kind {
	case KF32:
		f := float64(float32(x))
		if math.IsInf(f, 0) || math.IsNaN(f) {
			return f, false
		}
		return f, true
	case KF16:
		return roundF16(x)
	}
	if math.IsInf(x, 0) || math.IsNaN(x) {
		return x, false
	}
	return x, true
}

// exactIn reports whether x is exactly representable in t.
func exactIn(t *ScalarType, x float64) bool {
	r, ok := roundTo(t, x)
	return ok && r == x
}

// ---- literals ----

type literal struct {
	c      *Const
	suffix byte // 'i', 'u', 'f', 'h' or 0
}

// parseIntLiteral evaluates an integer literal token (spec "Numeric
// Literals").
func parseIntLiteral(text string) (*Const, *evalError) {
	var suffix byte
	if n := len(text); n > 1 && (text[n-1] == 'i' || text[n-1] == 'u') {
		suffix = text[n-1]
		text = text[:n-1]
	}
	base := 10
	digits := text
	if len(text) > 2 && (text[1] == 'x' || text[1] == 'X') && text[0] == '0' {
		base, digits = 16, text[2:]
	}
	v, err := strconv.ParseUint(digits, base, 64)
	if err != nil || v > math.MaxInt64 {
		if suffix == 0 {
			return nil, evalErrf("literal-range", "integer literal %s is too large for AbstractInt", text)
		}
	}
	switch suffix {
	case 'i':
		if err != nil || v > math.MaxInt32 {
			return nil, evalErrf("literal-range", "integer literal %si is out of range for i32", text)
		}
		return intConst(I32, int64(v)), nil
	case 'u':
		if err != nil || v > math.MaxUint32 {
			return nil, evalErrf("literal-range", "integer literal %su is out of range for u32", text)
		}
		return intConst(U32, int64(v)), nil
	}
	return intConst(AbstractInt, int64(v)), nil
}

// parseFloatLiteral evaluates a floating point literal token.
func parseFloatLiteral(text string) (*Const, *evalError) {
	hex := len(text) > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')
	body := text
	var suffix byte
	if n := len(text); n > 1 && (text[n-1] == 'f' || text[n-1] == 'h') {
		if !hex || strings.ContainsAny(text, "pP") {
			suffix = text[n-1]
			body = text[:n-1]
		}
	}
	if hex && !strings.ContainsAny(body, "pP") {
		body += "p0"
	}
	if !hex && strings.HasPrefix(body, ".") {
		body = "0" + body
	}
	v, err := strconv.ParseFloat(body, 64)
	if err != nil && !math.IsInf(v, 0) {
		return nil, evalErrf("literal-range", "malformed floating point literal %s", text)
	}
	t := AbstractFloat
	switch suffix {
	case 'f':
		t = F32
	case 'h':
		t = F16
	}
	if math.IsInf(v, 0) {
		return nil, evalErrf("literal-range", "floating point literal %s overflows %s", text, t)
	}
	if suffix != 0 {
		r, ok := roundTo(t, v)
		if !ok {
			return nil, evalErrf("literal-range", "floating point literal %s overflows %s", text, t)
		}
		if hex && r != v {
			return nil, evalErrf("literal-range", "hexadecimal floating point literal %s is not exactly representable in %s", text, t)
		}
		v = r
	}
	return floatConst(t, v), nil
}

// ---- conversions ----

// convertConst applies a feasible automatic conversion (spec "Conversion
// Rank") or an explicit conversion of an abstract/concrete numeric value to
// type to.
func convertConst(c *Const, to Type) (*Const, *evalError) {
	if c == nil {
		return nil, nil
	}
	if Same(c.Type, to) {
		return c, nil
	}
	switch to := to.(type) {
	case *ScalarType:
		return convertScalar(c, to)
	case *VecType, *MatType, *ArrayType:
		n := numElems(to)
		if c.Zero {
			return zeroConst(to), nil
		}
		out := &Const{Type: to, Elems: make([]*Const, n)}
		for i := int64(0); i < n; i++ {
			e, err := convertConst(c.Elems[i], elemType(to, i))
			if err != nil {
				return nil, err
			}
			out.Elems[i] = e
		}
		return out, nil
	case *StructType:
		if c.Zero {
			return zeroConst(to), nil
		}
		src, ok := c.Type.(*StructType)
		if !ok || len(src.Members) != len(to.Members) {
			return nil, nil
		}
		out := &Const{Type: to, Elems: make([]*Const, len(to.Members))}
		for i := range to.Members {
			e, err := convertConst(c.Elems[i], to.Members[i].Type)
			if err != nil {
				return nil, err
			}
			out.Elems[i] = e
		}
		return out, nil
	}
	return c, nil
}

func convertScalar(c *Const, to *ScalarType) (*Const, *evalError) {
	from, ok := c.Type.(*ScalarType)
	if !ok {
		return nil, nil
	}
	switch to.Kind {
	case KBool:
		switch from.Kind {
		case KBool:
			return c, nil
		case KAbstractInt, KI32, KU32:
			return boolConst(c.Int != 0), nil
		}
		return boolConst(c.Float != 0), nil
	case KI32, KU32, KAbstractInt:
		switch from.Kind {
		case KBool:
			if c.Bool {
				return intConst(to, 1), nil
			}
			return intConst(to, 0), nil
		case KAbstractInt, KI32, KU32:
			return convertInt(c.Int, from, to)
		}
		// floating point to integral: clamp, then truncate (spec "Floating
		// Point Conversion").
		x := c.Float
		if math.IsNaN(x) {
			return intConst(to, 0), nil
		}
		x = math.Trunc(x)
		// The result is the value of the target type closest to the truncated
		// source "and also exactly representable in the original floating point
		// type" (wgsl/index.bs, "Floating Point Conversion", line 14756). For an
		// f32 source the largest such i32 is 2147483520 and the largest such u32
		// 4294967040: 2^31-1 and 2^32-1 are not f32 values. An AbstractFloat is
		// an f64, which holds both; an f16 never reaches either bound.
		maxI32, maxU32 := float64(math.MaxInt32), float64(math.MaxUint32)
		if from.Kind == KF32 {
			maxI32, maxU32 = float64(math.Nextafter32(1<<31, 0)), float64(math.Nextafter32(1<<32, 0))
		}
		switch to.Kind {
		case KI32:
			if x < math.MinInt32 {
				x = math.MinInt32
			}
			if x > maxI32 {
				x = maxI32
			}
		case KU32:
			if x < 0 {
				x = 0
			}
			if x > maxU32 {
				x = maxU32
			}
		default:
			if x < math.MinInt64 {
				x = math.MinInt64
			}
			if x >= math.MaxInt64 {
				return intConst(to, math.MaxInt64), nil
			}
		}
		return intConst(to, int64(x)), nil
	}
	// floating point destination
	var x float64
	switch from.Kind {
	case KBool:
		if c.Bool {
			x = 1
		}
	case KAbstractInt, KI32:
		x = float64(c.Int)
	case KU32:
		x = float64(uint32(c.Int))
	default:
		x = c.Float
	}
	r, ok := roundTo(to, x)
	if !ok {
		return nil, evalErrf("const-overflow", "value %v cannot be represented as %s", x, to)
	}
	return floatConst(to, r), nil
}

func convertInt(v int64, from, to *ScalarType) (*Const, *evalError) {
	switch to.Kind {
	case KAbstractInt:
		return intConst(to, v), nil
	case KI32:
		if from.Kind == KU32 {
			return intConst(to, int64(int32(uint32(v)))), nil // bit reinterpretation
		}
		if from.Kind == KAbstractInt && (v < math.MinInt32 || v > math.MaxInt32) {
			return nil, evalErrf("const-overflow", "value %d cannot be represented as i32", v)
		}
		return intConst(to, v), nil
	}
	if from.Kind == KI32 {
		return intConst(to, int64(uint32(int32(v)))), nil
	}
	if from.Kind == KAbstractInt && (v < 0 || v > math.MaxUint32) {
		return nil, evalErrf("const-overflow", "value %d cannot be represented as u32", v)
	}
	return intConst(to, v), nil
}

// ---- operators ----

func wrapInt(t *ScalarType, v int64) int64 {
	switch t.Kind {
	case KI32:
		return int64(int32(v))
	case KU32:
		return int64(uint32(v))
	}
	return v
}

func isSignedKind(k ScalarKind) bool { return k == KAbstractInt || k == KI32 }

// arith evaluates a scalar binary arithmetic operator.
func arith(op string, t *ScalarType, a, b *Const) (*Const, *evalError) {
	switch t.Kind {
	case KAbstractInt:
		x, y := a.Int, b.Int
		switch op {
		case "+":
			s := x + y
			if (x >= 0) == (y >= 0) && (s >= 0) != (x >= 0) {
				return nil, evalErrf("const-overflow", "AbstractInt addition overflows")
			}
			return intConst(t, s), nil
		case "-":
			s := x - y
			if (x >= 0) != (y >= 0) && (s >= 0) != (x >= 0) {
				return nil, evalErrf("const-overflow", "AbstractInt subtraction overflows")
			}
			return intConst(t, s), nil
		case "*":
			hi, lo := bits.Mul64(uint64(absInt(x)), uint64(absInt(y)))
			neg := (x < 0) != (y < 0)
			if hi != 0 || (!neg && lo > math.MaxInt64) || (neg && lo > 1<<63) || (x == math.MinInt64 && y != 0 && y != 1) || (y == math.MinInt64 && x != 0 && x != 1) {
				return nil, evalErrf("const-overflow", "AbstractInt multiplication overflows")
			}
			return intConst(t, x*y), nil
		case "/", "%":
			if y == 0 {
				return nil, evalErrf("const-div-zero", "division by zero in a const-expression")
			}
			if x == math.MinInt64 && y == -1 {
				return nil, evalErrf("const-overflow", "AbstractInt division overflows")
			}
			if op == "/" {
				return intConst(t, x/y), nil
			}
			return intConst(t, x%y), nil
		}
	case KI32:
		x, y := a.Int, b.Int
		switch op {
		case "+":
			return intConst(t, wrapInt(t, x+y)), nil
		case "-":
			return intConst(t, wrapInt(t, x-y)), nil
		case "*":
			return intConst(t, wrapInt(t, x*y)), nil
		case "/", "%":
			if y == 0 {
				return nil, evalErrf("const-div-zero", "division by zero in a const-expression")
			}
			if x == math.MinInt32 && y == -1 {
				return nil, evalErrf("const-overflow", "i32 division overflows")
			}
			if op == "/" {
				return intConst(t, x/y), nil
			}
			return intConst(t, x%y), nil
		}
	case KU32:
		x, y := uint32(a.Int), uint32(b.Int)
		switch op {
		case "+":
			return intConst(t, int64(x+y)), nil
		case "-":
			return intConst(t, int64(x-y)), nil
		case "*":
			return intConst(t, int64(x*y)), nil
		case "/", "%":
			if y == 0 {
				return nil, evalErrf("const-div-zero", "division by zero in a const-expression")
			}
			if op == "/" {
				return intConst(t, int64(x/y)), nil
			}
			return intConst(t, int64(x%y)), nil
		}
	case KAbstractFloat, KF32, KF16:
		x, y := a.Float, b.Float
		var r float64
		switch op {
		case "+":
			r = x + y
		case "-":
			r = x - y
		case "*":
			r = x * y
		case "/":
			r = x / y
		case "%":
			r = x - y*math.Trunc(x/y)
		}
		v, ok := roundTo(t, r)
		if !ok {
			return nil, evalErrf("const-overflow", "floating point %s overflows or is not finite in a const-expression", op)
		}
		return floatConst(t, v), nil
	}
	return nil, nil
}

func absInt(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}

// negate evaluates unary minus on a scalar.
func negate(t *ScalarType, a *Const) (*Const, *evalError) {
	switch t.Kind {
	case KAbstractInt:
		if a.Int == math.MinInt64 {
			return nil, evalErrf("const-overflow", "AbstractInt negation overflows")
		}
		return intConst(t, -a.Int), nil
	case KI32:
		return intConst(t, wrapInt(t, -a.Int)), nil
	}
	return floatConst(t, -a.Float), nil
}

// compare evaluates a scalar comparison.
func compare(op string, t *ScalarType, a, b *Const) bool {
	var c int
	switch t.Kind {
	case KBool:
		if a.Bool == b.Bool {
			c = 0
		} else if !a.Bool {
			c = -1
		} else {
			c = 1
		}
	case KAbstractInt, KI32, KU32:
		switch {
		case a.Int < b.Int:
			c = -1
		case a.Int > b.Int:
			c = 1
		}
	default:
		x, y := a.Float, b.Float
		switch {
		case x < y:
			c = -1
		case x > y:
			c = 1
		}
	}
	switch op {
	case "==":
		return c == 0
	case "!=":
		return c != 0
	case "<":
		return c < 0
	case "<=":
		return c <= 0
	case ">":
		return c > 0
	}
	return c >= 0
}

// bitop evaluates & | ^ on integers (and booleans for & |).
func bitop(op string, t *ScalarType, a, b *Const) *Const {
	if t.Kind == KBool {
		switch op {
		case "&":
			return boolConst(a.Bool && b.Bool)
		case "|":
			return boolConst(a.Bool || b.Bool)
		}
	}
	var v int64
	switch op {
	case "&":
		v = a.Int & b.Int
	case "|":
		v = a.Int | b.Int
	default:
		v = a.Int ^ b.Int
	}
	return intConst(t, wrapInt(t, v))
}

// shift evaluates << and >> (spec "Bit Expressions"). amount is the value of
// the shift count.
func shift(op string, t *ScalarType, a *Const, amount uint32) (*Const, *evalError) {
	switch t.Kind {
	case KAbstractInt:
		x := a.Int
		if op == "<<" {
			if amount >= 64 {
				if x != 0 {
					return nil, evalErrf("const-overflow", "AbstractInt shift left overflows")
				}
				return a, nil
			}
			r := x << amount
			if r>>amount != x {
				return nil, evalErrf("const-overflow", "AbstractInt shift left overflows")
			}
			return intConst(t, r), nil
		}
		if amount >= 64 {
			if x < 0 {
				return intConst(t, -1), nil
			}
			return intConst(t, 0), nil
		}
		return intConst(t, x>>amount), nil
	case KI32:
		if amount >= 32 {
			return nil, evalErrf("const-shift", "shift count %d is not less than the bit width 32", amount)
		}
		x := int32(a.Int)
		if op == "<<" {
			r := x << amount
			if r>>amount != x {
				return nil, evalErrf("const-overflow", "i32 shift left overflows")
			}
			return intConst(t, int64(r)), nil
		}
		return intConst(t, int64(x>>amount)), nil
	}
	if amount >= 32 {
		return nil, evalErrf("const-shift", "shift count %d is not less than the bit width 32", amount)
	}
	x := uint32(a.Int)
	if op == "<<" {
		r := x << amount
		if r>>amount != x {
			return nil, evalErrf("const-overflow", "u32 shift left overflows")
		}
		return intConst(t, int64(r)), nil
	}
	return intConst(t, int64(x>>amount)), nil
}

// mapScalars applies f to each scalar component of the (scalar or vector)
// constants, producing a constant of type res.
func mapScalars(res Type, args []*Const, f func(xs []*Const) (*Const, *evalError)) (*Const, *evalError) {
	if _, ok := res.(*ScalarType); ok {
		return f(args)
	}
	v, ok := res.(*VecType)
	if !ok {
		return nil, nil
	}
	out := &Const{Type: res, Elems: make([]*Const, v.N)}
	xs := make([]*Const, len(args))
	for i := 0; i < v.N; i++ {
		for j, a := range args {
			if _, isVec := a.Type.(*VecType); isVec {
				xs[j] = a.Elem(int64(i))
			} else {
				xs[j] = a
			}
		}
		e, err := f(xs)
		if err != nil || e == nil {
			return nil, err
		}
		out.Elems[i] = e
	}
	return out, nil
}

// constEqual reports whether two scalar constants of the same type are
// equal.
func constEqual(a, b *Const) bool {
	if s, ok := a.Type.(*ScalarType); ok {
		return compare("==", s, a, b)
	}
	return false
}

// asInt returns the integer value of an integer scalar constant.
func (c *Const) asInt() (int64, bool) {
	if c == nil {
		return 0, false
	}
	s, ok := c.Type.(*ScalarType)
	if !ok || !isIntegerScalar(s) {
		return 0, false
	}
	return c.Int, true
}
