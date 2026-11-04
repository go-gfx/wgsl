package wgsl

import "math"

// fcalc performs floating point operations in the type sc and records the
// first operation whose result is not finite (spec "Floating Point
// Evaluation": a const-expression evaluation that overflows is a
// shader-creation error). Every operation is rounded, so intermediate
// overflow is detected as the reference implementations do.
type fcalc struct {
	sc  *ScalarType
	err *evalError
}

func (f *fcalc) r(x float64) float64 {
	if f.err != nil {
		return 0
	}
	v, ok := roundTo(f.sc, x)
	if !ok {
		f.err = evalErrf("const-overflow", "floating point overflow in a const-expression")
		return 0
	}
	return v
}

func (f *fcalc) mul(a, b float64) float64 { return f.r(a * b) }
func (f *fcalc) add(a, b float64) float64 { return f.r(a + b) }
func (f *fcalc) sub(a, b float64) float64 { return f.r(a - b) }

func (f *fcalc) dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s = f.add(s, f.mul(a[i], b[i]))
	}
	return s
}

func floats(c *Const) []float64 {
	var out []float64
	for _, k := range flatten(c, nil) {
		out = append(out, k.Float)
	}
	return out
}

func (f *fcalc) result(ret Type, vals []float64) (*Const, *evalError) {
	if f.err != nil {
		return nil, f.err
	}
	if _, ok := ret.(*VecType); !ok {
		return floatConst(f.sc, vals[0]), nil
	}
	out := &Const{Type: ret, Elems: make([]*Const, len(vals))}
	for i, v := range vals {
		out.Elems[i] = floatConst(f.sc, v)
	}
	return out, nil
}

// evalFloatVec evaluates the vector math built-ins on float operands with
// per-operation overflow checking. ok is false for functions it does not
// handle.
func evalFloatVec(name string, s *sig, args []*Const, sc *ScalarType) (k *Const, err *evalError, ok bool) {
	f := &fcalc{sc: sc}
	switch name {
	case "dot":
		d := f.dot(floats(args[0]), floats(args[1]))
		k, err = f.result(s.ret, []float64{d})
		return k, err, true
	case "length", "distance":
		a := floats(args[0])
		if name == "distance" {
			b := floats(args[1])
			for i := range a {
				a[i] = f.sub(a[i], b[i])
			}
		}
		if len(a) == 1 { // scalar: |e|, no squaring
			k, err = f.result(s.ret, []float64{math.Abs(a[0])})
			return k, err, true
		}
		l := f.r(math.Sqrt(f.dot(a, a)))
		k, err = f.result(s.ret, []float64{l})
		return k, err, true
	case "normalize":
		a := floats(args[0])
		l := f.r(math.Sqrt(f.dot(a, a)))
		if f.err == nil && l == 0 {
			return nil, evalErrf("const-overflow", "normalize of a zero vector"), true
		}
		out := make([]float64, len(a))
		for i := range a {
			out[i] = f.r(a[i] / l)
		}
		k, err = f.result(s.ret, out)
		return k, err, true
	case "cross":
		a, b := floats(args[0]), floats(args[1])
		c := func(i, j int) float64 { return f.sub(f.mul(a[i], b[j]), f.mul(a[j], b[i])) }
		k, err = f.result(s.ret, []float64{c(1, 2), c(2, 0), c(0, 1)})
		return k, err, true
	case "reflect": // e1 - 2 * dot(e2, e1) * e2
		e1, e2 := floats(args[0]), floats(args[1])
		d := f.mul(2, f.dot(e2, e1))
		out := make([]float64, len(e1))
		for i := range e1 {
			out[i] = f.sub(e1[i], f.mul(d, e2[i]))
		}
		k, err = f.result(s.ret, out)
		return k, err, true
	case "faceForward": // dot(e2, e3) < 0 ? e1 : -e1
		e1, e2, e3 := floats(args[0]), floats(args[1]), floats(args[2])
		d := f.dot(e2, e3)
		out := make([]float64, len(e1))
		for i := range e1 {
			if d < 0 {
				out[i] = e1[i]
			} else {
				out[i] = -e1[i]
			}
		}
		k, err = f.result(s.ret, out)
		return k, err, true
	case "refract": // k = 1 - e3*e3*(1 - dot(e2,e1)^2); k < 0 ? 0 : e3*e1 - (e3*dot(e2,e1) + sqrt(k))*e2
		e1, e2 := floats(args[0]), floats(args[1])
		e3 := floats(args[2])[0]
		d := f.dot(e2, e1)
		kk := f.sub(1, f.mul(f.mul(e3, e3), f.sub(1, f.mul(d, d))))
		out := make([]float64, len(e1))
		if f.err == nil && kk >= 0 {
			t := f.add(f.mul(e3, d), f.r(math.Sqrt(kk)))
			for i := range e1 {
				out[i] = f.sub(f.mul(e3, e1[i]), f.mul(t, e2[i]))
			}
		}
		k, err = f.result(s.ret, out)
		return k, err, true
	}
	return nil, nil, false
}
