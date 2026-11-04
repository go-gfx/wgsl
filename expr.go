package wgsl

import (
	"strings"
)

var invalidVal = func() *val { return &val{t: Invalid} }

// ---- conversion ranks (spec "Conversion Rank") ----

func convRank(src, dst Type) (int, bool) {
	if Same(src, dst) {
		return 0, true
	}
	switch s := src.(type) {
	case *RefType:
		if s.Access.Readable() && Same(s.Elem, dst) {
			return 0, true
		}
	case *ScalarType:
		d, ok := dst.(*ScalarType)
		if !ok {
			return 0, false
		}
		switch s.Kind {
		case KAbstractFloat:
			switch d.Kind {
			case KF32:
				return 1, true
			case KF16:
				return 2, true
			}
		case KAbstractInt:
			switch d.Kind {
			case KI32:
				return 3, true
			case KU32:
				return 4, true
			case KAbstractFloat:
				return 5, true
			case KF32:
				return 6, true
			case KF16:
				return 7, true
			}
		}
	case *VecType:
		if d, ok := dst.(*VecType); ok && d.N == s.N {
			return convRank(s.Elem, d.Elem)
		}
	case *MatType:
		if d, ok := dst.(*MatType); ok && d.C == s.C && d.R == s.R {
			return convRank(s.Elem, d.Elem)
		}
	case *ArrayType:
		if d, ok := dst.(*ArrayType); ok && !s.Runtime && !d.Runtime && s.Override == nil && d.Override == nil && s.Count == d.Count {
			return convRank(s.Elem, d.Elem)
		}
	case *StructType:
		if d, ok := dst.(*StructType); ok && s.Predeclared && d.Predeclared && len(s.Members) == len(d.Members) &&
			strings.HasPrefix(s.Name, "__frexp") == strings.HasPrefix(d.Name, "__frexp") &&
			strings.HasPrefix(s.Name, "__modf") == strings.HasPrefix(d.Name, "__modf") {
			r := 0
			for i := range s.Members {
				x, ok := convRank(s.Members[i].Type, d.Members[i].Type)
				if !ok {
					return 0, false
				}
				if x > r {
					r = x
				}
			}
			return r, true
		}
	}
	return 0, false
}

// concretize returns the concretization of t (spec "Conversion Rank").
func concretize(t Type) Type {
	switch t := t.(type) {
	case *ScalarType:
		switch t.Kind {
		case KAbstractInt:
			return I32
		case KAbstractFloat:
			return F32
		}
	case *VecType:
		return &VecType{t.N, concretize(t.Elem).(*ScalarType)}
	case *MatType:
		return &MatType{t.C, t.R, concretize(t.Elem).(*ScalarType)}
	case *ArrayType:
		if IsAbstract(t) {
			a := *t
			a.Elem = concretize(t.Elem)
			return &a
		}
	case *RefType:
		return t
	case *StructType:
		// the abstract frexp/modf result structures concretize to the f32
		// versions (spec "frexp", "modf": return type tables)
		if IsAbstract(t) && len(t.Members) > 0 {
			fn := ""
			switch {
			case strings.HasPrefix(t.Name, "__frexp"):
				fn = "frexp"
			case strings.HasPrefix(t.Name, "__modf"):
				fn = "modf"
			}
			if fn != "" {
				return resultStruct(fn, concretize(t.Members[0].Type))
			}
		}
	}
	return t
}

// targets lists every type reachable from t by a feasible automatic
// conversion (itself first).
func targets(t Type) []Type {
	switch t := t.(type) {
	case *ScalarType:
		switch t.Kind {
		case KAbstractInt:
			return []Type{AbstractInt, I32, U32, AbstractFloat, F32, F16}
		case KAbstractFloat:
			return []Type{AbstractFloat, F32, F16}
		}
	case *VecType:
		var out []Type
		for _, e := range targets(t.Elem) {
			out = append(out, &VecType{t.N, e.(*ScalarType)})
		}
		return out
	case *MatType:
		var out []Type
		for _, e := range targets(t.Elem) {
			out = append(out, &MatType{t.C, t.R, e.(*ScalarType)})
		}
		return out
	case *ArrayType:
		if IsAbstract(t) {
			var out []Type
			for _, e := range targets(t.Elem) {
				a := *t
				a.Elem = e
				out = append(out, &a)
			}
			return out
		}
	}
	return []Type{t}
}

type sig struct {
	params []Type
	ret    Type
	tag    string
}

func dominates(a, b []int) bool {
	strict := false
	for i := range a {
		if a[i] > b[i] {
			return false
		}
		if a[i] < b[i] {
			strict = true
		}
	}
	return strict
}

func describeArgs(args []*val) string {
	s := make([]string, len(args))
	for i, a := range args {
		s[i] = a.t.String()
	}
	return strings.Join(s, ", ")
}

// pick performs overload resolution (spec "Overload Resolution").
func (c *checker) pick(n Node, name string, args []*val, cands []sig) *sig {
	for _, a := range args {
		if isInvalid(a.t) {
			return nil
		}
	}
	anyNonConst := false
	for _, a := range args {
		if a.k != kConst {
			anyNonConst = true
		}
	}
	type scored struct {
		s     *sig
		ranks []int
	}
	var ok []scored
	for i := range cands {
		s := &cands[i]
		if len(s.params) != len(args) {
			continue
		}
		ranks := make([]int, len(args))
		feasible := true
		for j, a := range args {
			r, f := convRank(a.t, s.params[j])
			if !f {
				feasible = false
				break
			}
			ranks[j] = r
			if anyNonConst && IsAbstract(s.params[j]) {
				feasible = false
				break
			}
		}
		if feasible {
			ok = append(ok, scored{s, ranks})
		}
	}
	if len(ok) == 0 {
		c.err(n, "no-matching-overload", "no matching overload for %s(%s)", name, describeArgs(args))
		return nil
	}
	for i, a := range ok {
		best := true
		for j, b := range ok {
			if i != j && !dominates(a.ranks, b.ranks) {
				best = false
				break
			}
		}
		if best {
			return a.s
		}
	}
	if len(ok) == 1 {
		return ok[0].s
	}
	c.err(n, "ambiguous-overload", "ambiguous call to %s(%s)", name, describeArgs(args))
	return nil
}

// ---- values and conversions ----

func minKind(vs ...*val) exprKind {
	k := kConst
	for _, v := range vs {
		if v.k < k {
			k = v.k
		}
	}
	return k
}

// load applies the load rule.
func (c *checker) load(v *val) *val {
	if r, ok := v.t.(*RefType); ok {
		if !r.Access.Readable() {
			c.err(v.expr, "access-mode", "cannot read from a %s reference", r.Access)
			return invalidVal()
		}
		return &val{t: r.Elem, k: kRuntime, expr: v.expr}
	}
	return v
}

// convertTo applies the feasible automatic conversion of v to type to.
func (c *checker) convertTo(v *val, to Type) *val {
	if isInvalid(v.t) || isInvalid(to) {
		return invalidVal()
	}
	v = c.load(v)
	if isInvalid(v.t) {
		return v
	}
	if Same(v.t, to) {
		return v
	}
	if _, ok := convRank(v.t, to); !ok {
		c.err(v.expr, "type-mismatch", "cannot convert %s to %s", v.t, to)
		return invalidVal()
	}
	out := &val{t: to, k: v.k, expr: v.expr}
	if v.expr != nil {
		c.info.Conversions[v.expr] = to
	}
	if v.c != nil {
		cc, err := convertConst(v.c, to)
		if err != nil {
			c.err(v.expr, err.Rule, "%s", err.Msg)
		} else {
			out.c = cc
		}
	}
	return out
}

// materialize concretizes v (for let, var, override and runtime contexts).
func (c *checker) materialize(v *val) *val {
	v = c.load(v)
	if isInvalid(v.t) || !IsAbstract(v.t) {
		return v
	}
	return c.convertTo(v, concretize(v.t))
}

func (c *checker) expr(e Expr) *val {
	c.depth++
	defer func() { c.depth-- }()
	if c.depth > maxDepth {
		c.err(e, "limit-depth", "expression nesting is too deep")
		return invalidVal()
	}
	c.steps++
	if c.steps > c.maxSteps {
		c.err(e, "const-limit", "const-expression evaluation budget exceeded")
		panic(checkLimit{})
	}
	v := c.expr0(e)
	v.expr = e
	if !isInvalid(v.t) {
		c.info.Types[e] = v.t
		if v.k == kConst && v.c != nil {
			c.info.Consts[e] = v.c
		}
	}
	return v
}

func (c *checker) expr0(e Expr) *val {
	switch e := e.(type) {
	case *IntLit:
		k, err := parseIntLiteral(e.Text)
		if err != nil {
			c.err(e, err.Rule, "%s", err.Msg)
			return invalidVal()
		}
		return &val{t: k.Type, k: kConst, c: k}
	case *FloatLit:
		k, err := parseFloatLiteral(e.Text)
		if err != nil {
			c.err(e, err.Rule, "%s", err.Msg)
			return invalidVal()
		}
		if k.Type == F16 && !c.enabled["f16"] {
			c.err(e, "extension-required", "f16 literal requires `enable f16;`")
		}
		return &val{t: k.Type, k: kConst, c: k}
	case *BoolLit:
		return &val{t: Bool, k: kConst, c: boolConst(e.Value)}
	case *Paren:
		v := c.expr(e.X)
		return &val{t: v.t, k: v.k, c: v.c, comp: v.comp}
	case *Ident:
		return c.identExpr(e)
	case *Call:
		return c.callExpr(e)
	case *Unary:
		return c.unaryExpr(e)
	case *Binary:
		return c.binaryExpr(e)
	case *Index:
		return c.indexExpr(e)
	case *Member:
		return c.memberExpr(e)
	}
	return invalidVal()
}

func (c *checker) identExpr(id *Ident) *val {
	o := c.lookup(id.Name)
	if o == nil {
		c.err(id, "undeclared-ident", "undeclared identifier %q", id.Name)
		return invalidVal()
	}
	c.info.Objects[id] = o
	if len(id.Targs) > 0 {
		c.err(id, "type-as-value", "type %q cannot be used as a value", id.Name)
		return invalidVal()
	}
	if !c.use(o, id) {
		return invalidVal()
	}
	switch o := o.(type) {
	case *VarObj:
		if o.Global && c.cur != nil {
			c.cur.uses[o] = true
		}
		if o.Store == nil || isInvalid(o.Store) {
			return invalidVal()
		}
		return &val{t: &RefType{AS: o.AS, Elem: o.Store, Access: o.Access}, k: kRuntime}
	case *ValueObj:
		if o.Global() && c.cur != nil {
			c.cur.uses[o] = true
		}
		if o.Type == nil || isInvalid(o.Type) {
			return invalidVal()
		}
		switch o.Kind {
		case "const":
			return &val{t: o.Type, k: kConst, c: o.Val}
		case "override":
			return &val{t: o.Type, k: kOverride}
		}
		return &val{t: o.Type, k: kRuntime}
	case *ParamObj:
		return &val{t: o.Type, k: kRuntime}
	case *FuncObj:
		c.err(id, "function-as-value", "function %q used as a value", id.Name)
	case *TypeObj:
		c.err(id, "type-as-value", "type %q cannot be used as a value", id.Name)
	default:
		c.err(id, "type-as-value", "%q cannot be used as a value", id.Name)
	}
	return invalidVal()
}

// Global reports whether the value is declared at module scope.
func (o *ValueObj) Global() bool { return !o.Local }

// ---- operators ----

var (
	scalarsAll = []*ScalarType{AbstractInt, AbstractFloat, I32, U32, F32, F16}
	floatsAll  = []*ScalarType{AbstractFloat, F32, F16}
	intsAll    = []*ScalarType{AbstractInt, I32, U32}
)

// shapes expands scalars into scalar and vec2..4 types.
func shapes(ss []*ScalarType) []Type {
	var out []Type
	for _, s := range ss {
		out = append(out, s)
		for n := 2; n <= 4; n++ {
			out = append(out, &VecType{n, s})
		}
	}
	return out
}

func boolOf(t Type) Type {
	if v, ok := t.(*VecType); ok {
		return &VecType{v.N, Bool}
	}
	return Bool
}

var opCands = map[string][]sig{}

func init() {
	arith := func(op string, ss []*ScalarType) {
		var c []sig
		for _, t := range shapes(ss) {
			c = append(c, sig{params: []Type{t, t}, ret: t})
		}
		for _, s := range ss {
			for n := 2; n <= 4; n++ {
				v := &VecType{n, s}
				c = append(c, sig{params: []Type{v, s}, ret: v}, sig{params: []Type{s, v}, ret: v})
			}
		}
		opCands[op] = c
	}
	for _, op := range []string{"+", "-", "*", "/", "%"} {
		arith(op, scalarsAll)
	}
	for _, f := range floatsAll {
		for cc := 2; cc <= 4; cc++ {
			for r := 2; r <= 4; r++ {
				m := &MatType{cc, r, f}
				opCands["+"] = append(opCands["+"], sig{params: []Type{m, m}, ret: m})
				opCands["-"] = append(opCands["-"], sig{params: []Type{m, m}, ret: m})
				opCands["*"] = append(opCands["*"], sig{params: []Type{m, f}, ret: m}, sig{params: []Type{f, m}, ret: m},
					sig{params: []Type{m, &VecType{cc, f}}, ret: &VecType{r, f}},
					sig{params: []Type{&VecType{r, f}, m}, ret: &VecType{cc, f}})
				for k := 2; k <= 4; k++ {
					// e1: mat KxR, e2: mat CxK -> mat CxR
					opCands["*"] = append(opCands["*"], sig{params: []Type{&MatType{k, r, f}, &MatType{cc, k, f}}, ret: m})
				}
			}
		}
	}
	for _, op := range []string{"==", "!="} {
		var c []sig
		for _, t := range shapes(append([]*ScalarType{Bool}, scalarsAll...)) {
			c = append(c, sig{params: []Type{t, t}, ret: boolOf(t)})
		}
		opCands[op] = c
	}
	for _, op := range []string{"<", "<=", ">", ">="} {
		var c []sig
		for _, t := range shapes(scalarsAll) {
			c = append(c, sig{params: []Type{t, t}, ret: boolOf(t)})
		}
		opCands[op] = c
	}
	var bit []sig
	for _, t := range shapes(intsAll) {
		bit = append(bit, sig{params: []Type{t, t}, ret: t})
	}
	var boolOps []sig
	for _, t := range shapes([]*ScalarType{Bool}) {
		boolOps = append(boolOps, sig{params: []Type{t, t}, ret: t})
	}
	opCands["&"] = append(append([]sig{}, bit...), boolOps...)
	opCands["|"] = append(append([]sig{}, bit...), boolOps...)
	opCands["^"] = bit
	var sh []sig
	for _, s := range intsAll {
		sh = append(sh, sig{params: []Type{s, U32}, ret: s})
		for n := 2; n <= 4; n++ {
			sh = append(sh, sig{params: []Type{&VecType{n, s}, &VecType{n, U32}}, ret: &VecType{n, s}})
		}
	}
	opCands["<<"] = sh
	opCands[">>"] = sh
	var neg []sig
	for _, t := range shapes([]*ScalarType{AbstractInt, AbstractFloat, I32, F32, F16}) {
		neg = append(neg, sig{params: []Type{t}, ret: t})
	}
	opCands["u-"] = neg
	var not []sig
	for _, t := range shapes([]*ScalarType{Bool}) {
		not = append(not, sig{params: []Type{t}, ret: t})
	}
	opCands["u!"] = not
	var cmpl []sig
	for _, t := range shapes(intsAll) {
		cmpl = append(cmpl, sig{params: []Type{t}, ret: t})
	}
	opCands["u~"] = cmpl
}

func (c *checker) binaryExpr(e *Binary) *val {
	if e.Op == "&&" || e.Op == "||" {
		x := c.convertTo(c.expr(e.X), Bool)
		short := x.c != nil && x.k == kConst && x.c.Bool == (e.Op == "||")
		if short {
			c.suppress++
		}
		y := c.convertTo(c.expr(e.Y), Bool)
		if short {
			c.suppress--
		}
		if isInvalid(x.t) || isInvalid(y.t) {
			return invalidVal()
		}
		r := &val{t: Bool, k: minKind(x, y)}
		if x.c != nil {
			switch {
			case short:
				r.c = x.c
			case y.c != nil:
				if e.Op == "&&" {
					r.c = boolConst(x.c.Bool && y.c.Bool)
				} else {
					r.c = boolConst(x.c.Bool || y.c.Bool)
				}
			}
		}
		return r
	}
	x := c.load(c.expr(e.X))
	y := c.load(c.expr(e.Y))
	s := c.pick(e, e.Op, []*val{x, y}, opCands[e.Op])
	if s == nil {
		return invalidVal()
	}
	x, y = c.convertTo(x, s.params[0]), c.convertTo(y, s.params[1])
	if isInvalid(x.t) || isInvalid(y.t) {
		return invalidVal()
	}
	r := &val{t: s.ret, k: minKind(x, y)}
	if x.k == kConst && y.k == kConst && x.c != nil && y.c != nil {
		cv, err := evalBinary(e.Op, s, x.c, y.c)
		if err != nil {
			c.err(e, err.Rule, "%s", err.Msg)
		} else {
			r.c = cv
		}
	} else if e.Op == "/" || e.Op == "%" {
		c.divisorCheck(e, s, x, y)
	} else if e.Op == "<<" || e.Op == ">>" {
		c.shiftCheck(e, s, y)
	}
	return r
}

// shiftCheck reports a const shift count that is not less than the bit width
// of the shifted type even when the shifted operand is not a
// const-expression (spec "Bit Expressions").
func (c *checker) shiftCheck(e Node, s *sig, y *val) {
	if y.k != kConst || y.c == nil {
		return
	}
	bad := false
	forEachScalar(y.c, func(k *Const) {
		if k.Int >= 32 {
			bad = true
		}
	})
	if bad {
		c.err(e, "const-shift", "shift count is not less than the bit width 32")
	}
}

// divisorCheck reports division by a const zero when the dividend is not a
// const-expression (spec "Arithmetic Expressions").
func (c *checker) divisorCheck(e Node, s *sig, x, y *val) {
	sc := scalarOf(s.params[0])
	if sc == nil || !isIntegerScalar(sc) || y.k != kConst || y.c == nil {
		return
	}
	zero := func(k *Const) bool {
		if k.Zero {
			return true
		}
		if v, ok := k.Type.(*VecType); ok {
			for i := 0; i < v.N; i++ {
				if k.Elem(int64(i)).Int == 0 {
					return true
				}
			}
			return false
		}
		return k.Int == 0
	}
	if zero(y.c) {
		c.err(e, "const-div-zero", "division by zero")
	}
}

func evalBinary(op string, s *sig, a, b *Const) (*Const, *evalError) {
	if _, ok := s.params[0].(*MatType); ok {
		return evalMat(op, s, a, b)
	}
	if _, ok := s.params[1].(*MatType); ok {
		return evalMat(op, s, a, b)
	}
	at, bt := s.params[0], s.params[1]
	_, av := at.(*VecType)
	_, bv := bt.(*VecType)
	res := s.ret
	elemT := scalarOf(at)
	if av && !bv {
		elemT = scalarOf(at)
	}
	if !av && bv {
		elemT = scalarOf(bt)
	}
	return mapScalars(res, []*Const{a, b}, func(xs []*Const) (*Const, *evalError) {
		switch op {
		case "+", "-", "*", "/", "%":
			return arith(op, elemT, xs[0], xs[1])
		case "==", "!=", "<", "<=", ">", ">=":
			return boolConst(compare(op, elemT, xs[0], xs[1])), nil
		case "&", "|", "^":
			return bitop(op, elemT, xs[0], xs[1]), nil
		case "<<", ">>":
			return shift(op, elemT, xs[0], uint32(xs[1].Int))
		}
		return nil, nil
	})
}

func evalMat(op string, s *sig, a, b *Const) (*Const, *evalError) {
	rt := s.ret
	get := func(c *Const, t Type, i, j int) *Const { // column i, row j
		if m, ok := t.(*MatType); ok {
			_ = m
			return c.Elem(int64(i)).Elem(int64(j))
		}
		return nil
	}
	_ = get
	elem := scalarOf(rt)
	if m, ok := rt.(*MatType); ok {
		elem = m.Elem
	}
	mat := func(c *Const) *Const { return c }
	_ = mat
	at, bt := s.params[0], s.params[1]
	col := func(c *Const, t Type, i int) []float64 {
		if m, ok := t.(*MatType); ok {
			out := make([]float64, m.R)
			for j := range out {
				out[j] = c.Elem(int64(i)).Elem(int64(j)).Float
			}
			return out
		}
		return nil
	}
	_ = col
	fl := func(v float64) (float64, *evalError) {
		r, ok := roundTo(elem, v)
		if !ok {
			return 0, evalErrf("const-overflow", "floating point overflow in a const-expression")
		}
		return r, nil
	}
	build := func(cols, rows int, f func(i, j int) (float64, *evalError)) (*Const, *evalError) {
		out := &Const{Type: rt, Elems: make([]*Const, cols)}
		if _, ok := rt.(*VecType); ok {
			out.Elems = make([]*Const, rows)
			for j := 0; j < rows; j++ {
				v, err := f(0, j)
				if err != nil {
					return nil, err
				}
				out.Elems[j] = floatConst(elem, v)
			}
			return out, nil
		}
		for i := 0; i < cols; i++ {
			cv := &Const{Type: vec(rows, elem), Elems: make([]*Const, rows)}
			for j := 0; j < rows; j++ {
				v, err := f(i, j)
				if err != nil {
					return nil, err
				}
				cv.Elems[j] = floatConst(elem, v)
			}
			out.Elems[i] = cv
		}
		return out, nil
	}
	_ = at
	_ = bt
	am, aIsMat := at.(*MatType)
	bm, bIsMat := bt.(*MatType)
	at0 := func(i, j int) float64 { return a.Elem(int64(i)).Elem(int64(j)).Float }
	bt0 := func(i, j int) float64 { return b.Elem(int64(i)).Elem(int64(j)).Float }
	switch {
	case op == "+" && aIsMat && bIsMat:
		return build(am.C, am.R, func(i, j int) (float64, *evalError) { return fl(at0(i, j) + bt0(i, j)) })
	case op == "-" && aIsMat && bIsMat:
		return build(am.C, am.R, func(i, j int) (float64, *evalError) { return fl(at0(i, j) - bt0(i, j)) })
	case op == "*" && aIsMat && !bIsMat && isScalar(bt):
		return build(am.C, am.R, func(i, j int) (float64, *evalError) { return fl(at0(i, j) * b.Float) })
	case op == "*" && !aIsMat && isScalar(at) && bIsMat:
		return build(bm.C, bm.R, func(i, j int) (float64, *evalError) { return fl(a.Float * bt0(i, j)) })
	case op == "*" && aIsMat && !bIsMat:
		return build(1, am.R, func(_, j int) (float64, *evalError) {
			s := 0.0
			for k := 0; k < am.C; k++ {
				s += at0(k, j) * b.Elem(int64(k)).Float
			}
			return fl(s)
		})
	case op == "*" && !aIsMat && bIsMat:
		return build(1, bm.C, func(_, j int) (float64, *evalError) {
			s := 0.0
			for k := 0; k < bm.R; k++ {
				s += a.Elem(int64(k)).Float * bt0(j, k)
			}
			return fl(s)
		})
	case op == "*" && aIsMat && bIsMat:
		return build(bm.C, am.R, func(i, j int) (float64, *evalError) {
			s := 0.0
			for k := 0; k < am.C; k++ {
				s += at0(k, j) * bt0(i, k)
			}
			return fl(s)
		})
	}
	return nil, nil
}

func (c *checker) unaryExpr(e *Unary) *val {
	switch e.Op {
	case "&":
		x := c.expr(e.X)
		if isInvalid(x.t) {
			return x
		}
		r, ok := x.t.(*RefType)
		if !ok {
			c.err(e, "address-of", "operand of & must be a reference, found %s", x.t)
			return invalidVal()
		}
		if r.AS == ASHandle {
			c.err(e, "address-of", "cannot take the address of a handle-space variable")
			return invalidVal()
		}
		if x.comp {
			c.err(e, "address-of", "cannot take the address of a vector component")
			return invalidVal()
		}
		return &val{t: &PtrType{AS: r.AS, Elem: r.Elem, Access: r.Access}, k: kRuntime}
	case "*":
		x := c.expr(e.X)
		if isInvalid(x.t) {
			return x
		}
		p, ok := x.t.(*PtrType)
		if !ok {
			c.err(e, "indirection", "operand of * must be a pointer, found %s", x.t)
			return invalidVal()
		}
		return &val{t: &RefType{AS: p.AS, Elem: p.Elem, Access: p.Access}, k: kRuntime}
	}
	x := c.load(c.expr(e.X))
	if isInvalid(x.t) {
		return x
	}
	key := "u" + e.Op
	s := c.pick(e, e.Op, []*val{x}, opCands[key])
	if s == nil {
		return invalidVal()
	}
	x = c.convertTo(x, s.params[0])
	r := &val{t: s.ret, k: x.k}
	if x.k == kConst && x.c != nil {
		cv, err := mapScalars(s.ret, []*Const{x.c}, func(xs []*Const) (*Const, *evalError) {
			sc := scalarOf(s.ret)
			switch e.Op {
			case "-":
				return negate(sc, xs[0])
			case "!":
				return boolConst(!xs[0].Bool), nil
			}
			return intConst(sc, wrapInt(sc, ^xs[0].Int)), nil
		})
		if err != nil {
			c.err(e, err.Rule, "%s", err.Msg)
		} else {
			r.c = cv
		}
	}
	return r
}

// ---- composite access ----

func (c *checker) indexValue(i Expr) *val {
	idx := c.load(c.expr(i))
	if isInvalid(idx.t) {
		return idx
	}
	if !isIntegerScalar(idx.t) {
		c.err(i, "index-type", "index must be an integer scalar, found %s", idx.t)
		return invalidVal()
	}
	if idx.t == AbstractInt {
		idx = c.convertTo(idx, I32)
	}
	return idx
}

func (c *checker) indexExpr(e *Index) *val {
	base := c.expr(e.X)
	idx := c.indexValue(e.I)
	if isInvalid(base.t) || isInvalid(idx.t) {
		return invalidVal()
	}
	var n int64 = -1
	check := func(count int64) {
		n = count
		if idx.k == kConst && idx.c != nil && count >= 0 {
			if v, _ := idx.c.asInt(); v < 0 || v >= count {
				c.err(e.I, "const-index-range", "index %d is out of range [0, %d)", v, count)
			}
		} else if idx.k == kConst && idx.c != nil && count < 0 {
			if v, _ := idx.c.asInt(); v < 0 {
				c.err(e.I, "const-index-range", "negative index %d", v)
			}
		}
	}
	refOf := func(r Type, as AddressSpace, acc Access, elem Type, comp bool) *val {
		return &val{t: &RefType{AS: as, Elem: elem, Access: acc}, k: kRuntime, comp: comp}
	}
	var mem Type
	var as AddressSpace
	var acc Access
	isMem := false
	switch b := base.t.(type) {
	case *RefType:
		mem, as, acc, isMem = b.Elem, b.AS, b.Access, true
	case *PtrType:
		if !c.lang["pointer_composite_access"] {
			c.err(e, "pointer-access", "indexing a pointer requires pointer_composite_access")
			return invalidVal()
		}
		mem, as, acc, isMem = b.Elem, b.AS, b.Access, true
	}
	if isMem {
		switch t := mem.(type) {
		case *VecType:
			check(int64(t.N))
			return refOf(nil, as, acc, t.Elem, true)
		case *MatType:
			check(int64(t.C))
			return refOf(nil, as, acc, vec(t.R, t.Elem), false)
		case *ArrayType:
			if t.Runtime {
				check(-1)
			} else if t.Override == nil {
				check(t.Count)
			}
			return refOf(nil, as, acc, t.Elem, false)
		}
		c.err(e, "index-base", "cannot index a value of type %s", mem)
		return invalidVal()
	}
	var res Type
	switch t := base.t.(type) {
	case *VecType:
		check(int64(t.N))
		res = t.Elem
	case *MatType:
		check(int64(t.C))
		res = vec(t.R, t.Elem)
	case *ArrayType:
		if t.Runtime {
			c.err(e, "index-base", "cannot index a runtime-sized array value")
			return invalidVal()
		}
		if t.Override == nil {
			check(t.Count)
		}
		res = t.Elem
	default:
		c.err(e, "index-base", "cannot index a value of type %s", base.t)
		return invalidVal()
	}
	if IsAbstract(base.t) && idx.k != kConst {
		base = c.materialize(base)
		switch t := base.t.(type) {
		case *VecType:
			res = t.Elem
		case *MatType:
			res = vec(t.R, t.Elem)
		case *ArrayType:
			res = t.Elem
		}
	}
	out := &val{t: res, k: minKind(base, idx)}
	if base.c != nil && idx.c != nil && base.k == kConst && idx.k == kConst && n >= 0 {
		if v, _ := idx.c.asInt(); v >= 0 && v < n {
			out.c = base.c.Elem(v)
		}
	}
	return out
}

func swizzleIndices(name string, n int) ([]int, string) {
	if len(name) < 1 || len(name) > 4 {
		return nil, "a swizzle must have 1 to 4 letters"
	}
	set := ""
	var idx []int
	for _, r := range name {
		var s string
		var i int
		switch r {
		case 'x', 'y', 'z', 'w':
			s, i = "xyzw", strings.IndexRune("xyzw", r)
		case 'r', 'g', 'b', 'a':
			s, i = "rgba", strings.IndexRune("rgba", r)
		default:
			return nil, "invalid swizzle letter"
		}
		if set == "" {
			set = s
		} else if set != s {
			return nil, "swizzle letters must not be mixed"
		}
		if i >= n {
			return nil, "swizzle letter is past the end of the vector"
		}
		idx = append(idx, i)
	}
	return idx, ""
}

func (c *checker) memberExpr(e *Member) *val {
	base := c.expr(e.X)
	if isInvalid(base.t) {
		return base
	}
	var mem Type
	var as AddressSpace
	var acc Access
	isMem := false
	switch b := base.t.(type) {
	case *RefType:
		mem, as, acc, isMem = b.Elem, b.AS, b.Access, true
	case *PtrType:
		if !c.lang["pointer_composite_access"] {
			c.err(e, "pointer-access", "member access through a pointer requires pointer_composite_access")
			return invalidVal()
		}
		mem, as, acc, isMem = b.Elem, b.AS, b.Access, true
	default:
		mem = base.t
	}
	switch t := mem.(type) {
	case *StructType:
		m := t.Member(e.Name)
		if m == nil {
			c.err(e, "no-member", "structure %s has no member %q", t.Name, e.Name)
			return invalidVal()
		}
		out := &val{t: m.Type, k: base.k}
		if isMem {
			out = &val{t: &RefType{AS: as, Elem: m.Type, Access: acc}, k: kRuntime}
		} else if base.c != nil && base.k == kConst {
			for i, mm := range t.Members {
				if mm == m {
					out.c = base.c.Elem(int64(i))
				}
			}
		}
		return out
	case *VecType:
		idx, why := swizzleIndices(e.Name, t.N)
		if why != "" {
			c.err(e, "swizzle", "%s: %q", why, e.Name)
			return invalidVal()
		}
		if isMem && len(idx) == 1 {
			return &val{t: &RefType{AS: as, Elem: t.Elem, Access: acc}, k: kRuntime, comp: true}
		}
		if isMem {
			if !acc.Readable() {
				c.err(e, "access-mode", "cannot read from a %s reference", acc)
				return invalidVal()
			}
		}
		var rt Type = t.Elem
		if len(idx) > 1 {
			rt = vec(len(idx), t.Elem)
		}
		out := &val{t: rt, k: base.k}
		if isMem {
			out.k = kRuntime
		}
		if !isMem && base.c != nil && base.k == kConst {
			if len(idx) == 1 {
				out.c = base.c.Elem(int64(idx[0]))
			} else {
				cv := &Const{Type: rt, Elems: make([]*Const, len(idx))}
				for i, ix := range idx {
					cv.Elems[i] = base.c.Elem(int64(ix))
				}
				out.c = cv
			}
		}
		return out
	}
	c.err(e, "no-member", "type %s has no member %q", mem, e.Name)
	return invalidVal()
}
