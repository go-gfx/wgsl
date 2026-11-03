package wgsl

import (
	"math"
	"math/bits"
)

func (c *checker) loadArgs(e *Call) []*val {
	args := make([]*val, len(e.Args))
	for i, a := range e.Args {
		args[i] = c.load(c.expr(a))
	}
	return args
}

func anyInvalid(args []*val) bool {
	for _, a := range args {
		if isInvalid(a.t) {
			return true
		}
	}
	return false
}

func allConstVals(args []*val) bool {
	for _, a := range args {
		if a.k != kConst || a.c == nil {
			return false
		}
	}
	return true
}

// ctorCall checks a value constructor call (spec "Constructor Built-in
// Functions"). When infer is true the type is deduced from the arguments
// (vecN(...), matCxR(...), array(...)).
func (c *checker) ctorCall(e *Call, t Type, infer bool) *val {
	args := c.loadArgs(e)
	if anyInvalid(args) {
		return invalidVal()
	}
	name := e.Fn.Name
	kind, dims, _, _ := vecMatName(name)
	if infer {
		switch {
		case kind == 'v':
			if len(args) == 0 {
				return &val{t: vec(dims[0], AbstractInt), k: kConst, c: zeroConst(vec(dims[0], AbstractInt))}
			}
			return c.vecCtor(e, dims[0], nil, args)
		case kind == 'm':
			return c.matCtor(e, dims[0], dims[1], nil, args)
		}
		return c.arrayCtor(e, nil, args)
	}
	if t == nil || isInvalid(t) {
		return invalidVal()
	}
	if len(args) == 0 {
		if !constructible(t) || IsAbstract(t) {
			c.err(e, "ctor", "type %s has no zero-value constructor", t)
			return invalidVal()
		}
		return &val{t: t, k: kConst, c: zeroConst(t)}
	}
	switch t := t.(type) {
	case *ScalarType:
		return c.scalarCtor(e, t, args)
	case *VecType:
		return c.vecCtor(e, t.N, t.Elem, args)
	case *MatType:
		return c.matCtor(e, t.C, t.R, t.Elem, args)
	case *ArrayType:
		return c.arrayCtor(e, t, args)
	case *StructType:
		if !constructible(t) {
			c.err(e, "ctor", "structure %s is not constructible", t.Name)
			return invalidVal()
		}
		if len(args) != len(t.Members) {
			c.err(e, "arg-count", "%s expects %d members, found %d", t.Name, len(t.Members), len(args))
			return invalidVal()
		}
		out := &Const{Type: t, Elems: make([]*Const, len(args))}
		allC := true
		k := kConst
		for i, a := range args {
			a = c.convertTo(a, t.Members[i].Type)
			if isInvalid(a.t) {
				return a
			}
			if a.k < k {
				k = a.k
			}
			if a.k != kConst || a.c == nil {
				allC = false
			} else {
				out.Elems[i] = a.c
			}
		}
		r := &val{t: t, k: k}
		if allC {
			r.c = out
		}
		return r
	}
	c.err(e, "ctor", "type %s cannot be constructed", t)
	return invalidVal()
}

func (c *checker) scalarCtor(e *Call, t *ScalarType, args []*val) *val {
	if len(args) != 1 {
		c.err(e, "arg-count", "%s() takes one argument", t)
		return invalidVal()
	}
	var cands []sig
	for _, s := range append([]*ScalarType{Bool}, scalarsAll...) {
		if t == F32 && IsAbstract(s) {
			continue
		}
		cands = append(cands, sig{params: []Type{s}, ret: t})
	}
	s := c.pick(e, t.String(), args, cands)
	if s == nil {
		return invalidVal()
	}
	a := c.convertTo(args[0], s.params[0])
	r := &val{t: t, k: a.k}
	if a.k == kConst && a.c != nil {
		cv, err := convertConst(a.c, t)
		if err != nil {
			c.err(e, err.Rule, "%s", err.Msg)
		} else {
			r.c = cv
		}
	}
	return r
}

func argShape(t Type) (int, Type, bool) {
	switch t := t.(type) {
	case *ScalarType:
		return 1, t, true
	case *VecType:
		return t.N, t.Elem, true
	}
	return 0, nil, false
}

func flatten(c *Const, out []*Const) []*Const {
	if v, ok := c.Type.(*VecType); ok {
		for i := 0; i < v.N; i++ {
			out = append(out, c.Elem(int64(i)))
		}
		return out
	}
	return append(out, c)
}

func (c *checker) vecCtor(e *Call, n int, elem *ScalarType, args []*val) *val {
	total := 0
	valid := true
	for _, a := range args {
		sz, _, ok := argShape(a.t)
		if !ok {
			valid = false
		}
		total += sz
	}
	if !valid {
		c.err(e, "ctor", "invalid argument types for vec%d: %s", n, describeArgs(args))
		return invalidVal()
	}
	elems := []*ScalarType{elem}
	if elem == nil {
		elems = append([]*ScalarType{Bool}, scalarsAll...)
	}
	var cands []sig
	for _, t := range elems {
		vt := &VecType{n, t}
		if len(args) == 1 {
			if sz, el, _ := argShape(args[0].t); sz == 1 {
				cands = append(cands, sig{params: []Type{t}, ret: vt})
			} else if sz == n {
				if elem != nil {
					cands = append(cands, sig{params: []Type{args[0].t}, ret: vt, tag: "conv"})
				} else if t == Bool {
					cands = append(cands, sig{params: []Type{args[0].t}, ret: &VecType{n, el.(*ScalarType)}, tag: "conv"})
				}
			}
			continue
		}
		if total == n {
			var ps []Type
			for _, a := range args {
				sz, _, _ := argShape(a.t)
				if sz == 1 {
					ps = append(ps, t)
				} else {
					ps = append(ps, &VecType{sz, t})
				}
			}
			cands = append(cands, sig{params: ps, ret: vt})
		}
	}
	s := c.pick(e, e.Fn.Name, args, cands)
	if s == nil {
		return invalidVal()
	}
	conv := make([]*val, len(args))
	for i, a := range args {
		if s.tag == "conv" {
			conv[i] = a
		} else {
			conv[i] = c.convertTo(a, s.params[i])
		}
		if isInvalid(conv[i].t) {
			return invalidVal()
		}
	}
	r := &val{t: s.ret, k: minKind(conv...)}
	if len(args) == 1 {
		r.k = conv[0].k
	}
	if allConstVals(conv) {
		vt := s.ret.(*VecType)
		var flat []*Const
		for _, a := range conv {
			flat = flatten(a.c, flat)
		}
		out := &Const{Type: vt, Elems: make([]*Const, n)}
		for i := 0; i < n; i++ {
			src := flat[0]
			if len(flat) > 1 {
				src = flat[i]
			}
			cv, err := convertConst(src, vt.Elem)
			if err != nil {
				c.err(e, err.Rule, "%s", err.Msg)
				return r
			}
			out.Elems[i] = cv
		}
		r.c = out
	}
	return r
}

func (c *checker) matCtor(e *Call, cc, rr int, elem *ScalarType, args []*val) *val {
	elems := []*ScalarType{elem}
	if elem == nil {
		elems = floatsAll
	}
	var cands []sig
	for _, t := range elems {
		mt := &MatType{cc, rr, t}
		if len(args) == 1 {
			if m, ok := args[0].t.(*MatType); ok && m.C == cc && m.R == rr {
				if elem != nil {
					cands = append(cands, sig{params: []Type{m}, ret: mt, tag: "conv"})
				} else if t == floatsAll[0] {
					cands = append(cands, sig{params: []Type{m}, ret: m, tag: "conv"})
				}
			}
			continue
		}
		if len(args) == cc {
			ps := make([]Type, cc)
			for i := range ps {
				ps[i] = &VecType{rr, t}
			}
			cands = append(cands, sig{params: ps, ret: mt})
		}
		if len(args) == cc*rr {
			ps := make([]Type, len(args))
			for i := range ps {
				ps[i] = t
			}
			cands = append(cands, sig{params: ps, ret: mt})
		}
	}
	s := c.pick(e, e.Fn.Name, args, cands)
	if s == nil {
		return invalidVal()
	}
	conv := make([]*val, len(args))
	for i, a := range args {
		if s.tag == "conv" {
			conv[i] = a
		} else {
			conv[i] = c.convertTo(a, s.params[i])
		}
		if isInvalid(conv[i].t) {
			return invalidVal()
		}
	}
	mt := s.ret.(*MatType)
	r := &val{t: mt, k: minKind(conv...)}
	if allConstVals(conv) {
		out := &Const{Type: mt, Elems: make([]*Const, cc)}
		var flat []*Const
		for _, a := range conv {
			if m, ok := a.t.(*MatType); ok {
				for i := 0; i < m.C; i++ {
					flat = flatten(a.c.Elem(int64(i)), flat)
				}
				continue
			}
			flat = flatten(a.c, flat)
		}
		for i := 0; i < cc; i++ {
			col := &Const{Type: vec(rr, mt.Elem), Elems: make([]*Const, rr)}
			for j := 0; j < rr; j++ {
				cv, err := convertConst(flat[i*rr+j], mt.Elem)
				if err != nil {
					c.err(e, err.Rule, "%s", err.Msg)
					return r
				}
				col.Elems[j] = cv
			}
			out.Elems[i] = col
		}
		r.c = out
	}
	return r
}

func (c *checker) arrayCtor(e *Call, t *ArrayType, args []*val) *val {
	if len(args) > 65536 {
		c.err(e, "limit-array", "too many array constructor arguments")
		return invalidVal()
	}
	var elem Type
	if t != nil {
		if t.Runtime || t.Override != nil || !constructible(t) {
			c.err(e, "ctor", "array type %s is not constructible", t)
			return invalidVal()
		}
		if int64(len(args)) != t.Count {
			c.err(e, "arg-count", "array<%s, %d> constructor takes %d arguments, found %d", t.Elem, t.Count, t.Count, len(args))
			return invalidVal()
		}
		elem = t.Elem
	} else {
		var cands []sig
		for _, cand := range targets(args[0].t) {
			if !constructible(cand) {
				continue
			}
			ps := make([]Type, len(args))
			for i := range ps {
				ps[i] = cand
			}
			cands = append(cands, sig{params: ps, ret: cand})
		}
		s := c.pick(e, "array", args, cands)
		if s == nil {
			return invalidVal()
		}
		elem = s.ret
		t = &ArrayType{Elem: elem, Count: int64(len(args))}
	}
	conv := make([]*val, len(args))
	for i, a := range args {
		conv[i] = c.convertTo(a, elem)
		if isInvalid(conv[i].t) {
			return invalidVal()
		}
	}
	r := &val{t: t, k: minKind(conv...)}
	if allConstVals(conv) {
		out := &Const{Type: t, Elems: make([]*Const, len(args))}
		for i, a := range conv {
			out.Elems[i] = a.c
		}
		r.c = out
	}
	return r
}

// ---- const evaluation of built-ins ----

type fl1 func(float64) float64

var floatFns = map[string]fl1{
	"acos": math.Acos, "acosh": math.Acosh, "asin": math.Asin, "asinh": math.Asinh, "atan": math.Atan,
	"atanh": math.Atanh, "ceil": math.Ceil, "cos": math.Cos, "cosh": math.Cosh, "exp": math.Exp,
	"exp2": math.Exp2, "floor": math.Floor, "log": math.Log, "log2": math.Log2, "sin": math.Sin,
	"sinh": math.Sinh, "sqrt": math.Sqrt, "tan": math.Tan, "tanh": math.Tanh, "trunc": math.Trunc,
	"round":       math.RoundToEven,
	"fract":       func(x float64) float64 { return x - math.Floor(x) },
	"inverseSqrt": func(x float64) float64 { return 1 / math.Sqrt(x) },
	"degrees":     func(x float64) float64 { return x * 180 / math.Pi },
	"radians":     func(x float64) float64 { return x * math.Pi / 180 },
	"saturate":    func(x float64) float64 { return math.Min(math.Max(x, 0), 1) },
}

func finite(t *ScalarType, x float64) (*Const, *evalError) {
	r, ok := roundTo(t, x)
	if !ok {
		return nil, evalErrf("const-overflow", "result is not finite in a const-expression")
	}
	return floatConst(t, r), nil
}

func evalGeneric(name string) func(*sig, []*Const) (*Const, *evalError) {
	return func(s *sig, args []*Const) (*Const, *evalError) {
		sc := scalarOf(s.params[0])
		if sc == nil {
			if m, ok := s.params[0].(*MatType); ok {
				sc = m.Elem
			}
		}
		if f, ok := floatFns[name]; ok {
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				return finite(sc, f(xs[0].Float))
			})
		}
		isF := sc != nil && isFloatScalar(sc)
		switch name {
		case "abs":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				if isF {
					return floatConst(sc, math.Abs(xs[0].Float)), nil
				}
				if sc.Kind == KU32 {
					return xs[0], nil
				}
				if sc.Kind == KAbstractInt && xs[0].Int == math.MinInt64 {
					return nil, evalErrf("const-overflow", "abs overflows")
				}
				return intConst(sc, wrapInt(sc, absInt(xs[0].Int))), nil
			})
		case "sign":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				var v float64
				if isF {
					v = xs[0].Float
				} else {
					v = float64(xs[0].Int)
				}
				r := 0.0
				if v > 0 {
					r = 1
				} else if v < 0 {
					r = -1
				}
				if isF {
					return floatConst(sc, r), nil
				}
				return intConst(sc, int64(r)), nil
			})
		case "min", "max":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				less := compare("<", sc, xs[0], xs[1])
				if (name == "min") == less {
					return xs[0], nil
				}
				return xs[1], nil
			})
		case "clamp":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				r := xs[0]
				if compare("<", sc, r, xs[1]) {
					r = xs[1]
				}
				if compare(">", sc, r, xs[2]) {
					r = xs[2]
				}
				return r, nil
			})
		case "atan2":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) { return finite(sc, math.Atan2(xs[0].Float, xs[1].Float)) })
		case "pow":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) { return finite(sc, math.Pow(xs[0].Float, xs[1].Float)) })
		case "step":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				if xs[1].Float < xs[0].Float {
					return floatConst(sc, 0), nil
				}
				return floatConst(sc, 1), nil
			})
		case "fma":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				return finite(sc, xs[0].Float*xs[1].Float+xs[2].Float)
			})
		case "smoothstep":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				t := math.Min(math.Max((xs[2].Float-xs[0].Float)/(xs[1].Float-xs[0].Float), 0), 1)
				return finite(sc, t*t*(3-2*t))
			})
		case "mix":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				return finite(sc, xs[0].Float*(1-xs[2].Float)+xs[1].Float*xs[2].Float)
			})
		case "quantizeToF16":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				r, ok := roundF16(xs[0].Float)
				if !ok {
					return nil, evalErrf("const-overflow", "value is not representable in f16")
				}
				return floatConst(F32, r), nil
			})
		case "countOneBits", "countLeadingZeros", "countTrailingZeros", "reverseBits", "firstLeadingBit", "firstTrailingBit":
			return mapScalars(s.ret, args, func(xs []*Const) (*Const, *evalError) {
				u := uint32(xs[0].Int)
				var r uint32
				switch name {
				case "countOneBits":
					r = uint32(bits.OnesCount32(u))
				case "countLeadingZeros":
					r = uint32(bits.LeadingZeros32(u))
				case "countTrailingZeros":
					r = uint32(bits.TrailingZeros32(u))
				case "reverseBits":
					r = bits.Reverse32(u)
				case "firstTrailingBit":
					if u == 0 {
						r = math.MaxUint32
					} else {
						r = uint32(bits.TrailingZeros32(u))
					}
				default:
					if sc.Kind == KI32 && int32(u) < 0 {
						u = ^u
					}
					if u == 0 {
						r = math.MaxUint32
					} else {
						r = uint32(31 - bits.LeadingZeros32(u))
					}
				}
				return intConst(sc, wrapInt(sc, int64(r))), nil
			})
		case "extractBits":
			return mapScalars(s.ret, []*Const{args[0]}, func(xs []*Const) (*Const, *evalError) {
				off, cnt := uint32(args[1].Int), uint32(args[2].Int)
				if off+cnt > 32 {
					return nil, evalErrf("const-bits", "offset + count exceeds 32")
				}
				u := uint32(xs[0].Int)
				if cnt == 0 {
					return intConst(sc, 0), nil
				}
				if sc.Kind == KI32 {
					return intConst(sc, int64(int32(u<<(32-off-cnt))>>(32-cnt))), nil
				}
				return intConst(sc, int64((u<<(32-off-cnt))>>(32-cnt))), nil
			})
		case "insertBits":
			return mapScalars(s.ret, []*Const{args[0], args[1]}, func(xs []*Const) (*Const, *evalError) {
				off, cnt := uint32(args[2].Int), uint32(args[3].Int)
				if off+cnt > 32 {
					return nil, evalErrf("const-bits", "offset + count exceeds 32")
				}
				var mask uint32
				if cnt >= 32 {
					mask = math.MaxUint32
				} else {
					mask = ((1 << cnt) - 1) << off
				}
				r := (uint32(xs[0].Int) &^ mask) | ((uint32(xs[1].Int) << off) & mask)
				return intConst(sc, wrapInt(sc, int64(r))), nil
			})
		case "select":
			var f func(i int) *Const
			cond := args[2]
			if v, ok := s.ret.(*VecType); ok {
				f = func(i int) *Const {
					b := cond
					if _, isV := cond.Type.(*VecType); isV {
						b = cond.Elem(int64(i))
					}
					if b.Bool {
						return args[1].Elem(int64(i))
					}
					return args[0].Elem(int64(i))
				}
				out := &Const{Type: s.ret, Elems: make([]*Const, v.N)}
				for i := range out.Elems {
					out.Elems[i] = f(i)
				}
				return out, nil
			}
			if cond.Bool {
				return args[1], nil
			}
			return args[0], nil
		case "all", "any":
			r := name == "all"
			var elems []*Const
			elems = flatten(args[0], elems)
			for _, x := range elems {
				if name == "all" {
					r = r && x.Bool
				} else {
					r = r || x.Bool
				}
			}
			return boolConst(r), nil
		case "dot", "length", "distance", "normalize", "cross":
			return evalVecMath(name, s, args, sc)
		case "transpose":
			m := s.params[0].(*MatType)
			out := &Const{Type: s.ret, Elems: make([]*Const, m.R)}
			for j := 0; j < m.R; j++ {
				col := &Const{Type: vec(m.C, m.Elem), Elems: make([]*Const, m.C)}
				for i := 0; i < m.C; i++ {
					col.Elems[i] = args[0].Elem(int64(i)).Elem(int64(j))
				}
				out.Elems[j] = col
			}
			return out, nil
		}
		return nil, nil
	}
}

func evalVecMath(name string, s *sig, args []*Const, sc *ScalarType) (*Const, *evalError) {
	comps := func(c *Const) []*Const { return flatten(c, nil) }
	switch name {
	case "dot":
		a, b := comps(args[0]), comps(args[1])
		if isFloatScalar(sc) {
			sum := 0.0
			for i := range a {
				sum += a[i].Float * b[i].Float
			}
			return finite(sc, sum)
		}
		var acc *Const = intConst(sc, 0)
		for i := range a {
			p, err := arith("*", sc, a[i], b[i])
			if err != nil {
				return nil, err
			}
			acc, err = arith("+", sc, acc, p)
			if err != nil {
				return nil, err
			}
		}
		return acc, nil
	case "length", "distance":
		a := comps(args[0])
		var b []*Const
		if name == "distance" {
			b = comps(args[1])
		}
		sum := 0.0
		for i := range a {
			d := a[i].Float
			if b != nil {
				d -= b[i].Float
			}
			sum += d * d
		}
		return finite(sc, math.Sqrt(sum))
	case "normalize":
		a := comps(args[0])
		sum := 0.0
		for _, x := range a {
			sum += x.Float * x.Float
		}
		l := math.Sqrt(sum)
		if l == 0 {
			return nil, evalErrf("const-overflow", "normalize of a zero vector")
		}
		out := &Const{Type: s.ret, Elems: make([]*Const, len(a))}
		for i, x := range a {
			k, err := finite(sc, x.Float/l)
			if err != nil {
				return nil, err
			}
			out.Elems[i] = k
		}
		return out, nil
	case "cross":
		a, b := comps(args[0]), comps(args[1])
		out := &Const{Type: s.ret, Elems: make([]*Const, 3)}
		f := func(i, j int) float64 { return a[i].Float*b[j].Float - a[j].Float*b[i].Float }
		vals := [3]float64{f(1, 2), f(2, 0), f(0, 1)}
		for i, v := range vals {
			k, err := finite(sc, v)
			if err != nil {
				return nil, err
			}
			out.Elems[i] = k
		}
		return out, nil
	}
	return nil, nil
}
