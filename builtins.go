package wgsl

import (
	"fmt"
	"strings"
)

// builtinFn describes a built-in function (spec "Built-in Functions"). The
// overload tables are generated from the type-variable families below and
// cross-checked against the specification text by TestBuiltinsMatchSpec
// (internal/specbuiltins prints the spec's overload tables).
type builtinFn struct {
	name    string
	mustUse bool
	isConst bool
	// cands yields the candidate signatures for already checked arguments.
	cands func(c *checker, args []*val) []sig
	// custom replaces the generic overload resolution.
	custom func(c *checker, e *Call, args []*val) *val
	eval   func(s *sig, args []*Const) (*Const, *evalError)
	// stage restricts where the function may be used: "fragment",
	// "compute", "fragment|compute", "notvertex".
	stage string
	ext   string // required enable-extension
	lang  string // required language extension
	// ptrArgs lists argument positions that are pointers (not loaded).
	ptrArgs bool
}

var builtinFuncs = map[string]*builtinFn{}

func reg(f *builtinFn) { builtinFuncs[f.name] = f }

func fixed(sigs ...sig) func(*checker, []*val) []sig {
	return func(*checker, []*val) []sig { return sigs }
}

func unary(ts []Type) []sig {
	var out []sig
	for _, t := range ts {
		out = append(out, sig{params: []Type{t}, ret: t})
	}
	return out
}

func binary(ts []Type) []sig {
	var out []sig
	for _, t := range ts {
		out = append(out, sig{params: []Type{t, t}, ret: t})
	}
	return out
}

func ternary(ts []Type) []sig {
	var out []sig
	for _, t := range ts {
		out = append(out, sig{params: []Type{t, t, t}, ret: t})
	}
	return out
}

func vecsOf(ss ...*ScalarType) []Type {
	var out []Type
	for _, s := range ss {
		for n := 2; n <= 4; n++ {
			out = append(out, &VecType{n, s})
		}
	}
	return out
}

func init() {
	floatT := shapes(floatsAll)
	numT := shapes(scalarsAll)
	signedNumT := shapes([]*ScalarType{AbstractInt, AbstractFloat, I32, F32, F16})
	intT := shapes([]*ScalarType{I32, U32})
	sintT := shapes([]*ScalarType{I32})
	uintT := shapes([]*ScalarType{U32})

	cf := func(name string, ts []Type, arity int) {
		var s []sig
		switch arity {
		case 1:
			s = unary(ts)
		case 2:
			s = binary(ts)
		default:
			s = ternary(ts)
		}
		reg(&builtinFn{name: name, mustUse: true, isConst: true, cands: fixed(s...), eval: evalGeneric(name)})
	}
	for _, n := range []string{"acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil", "cos", "cosh", "degrees",
		"exp", "exp2", "floor", "fract", "inverseSqrt", "log", "log2", "radians", "round", "saturate", "sin", "sinh",
		"sqrt", "tan", "tanh", "trunc"} {
		cf(n, floatT, 1)
	}
	cf("abs", numT, 1)
	cf("sign", signedNumT, 1)
	cf("atan2", floatT, 2)
	cf("pow", floatT, 2)
	cf("step", floatT, 2)
	cf("max", numT, 2)
	cf("min", numT, 2)
	cf("clamp", numT, 3)
	cf("fma", floatT, 3)
	cf("smoothstep", floatT, 3)
	for _, n := range []string{"countLeadingZeros", "countOneBits", "countTrailingZeros", "firstLeadingBit",
		"firstTrailingBit", "reverseBits"} {
		cf(n, intT, 1)
	}
	_ = sintT
	_ = uintT
	cf("insertBits", intT, 3)
	for _, t := range intT {
		_ = t
	}
	var ins []sig
	for _, t := range intT {
		ins = append(ins, sig{params: []Type{t, t, U32, U32}, ret: t})
	}
	reg(&builtinFn{name: "insertBits", mustUse: true, isConst: true, cands: fixed(ins...), eval: evalGeneric("insertBits")})
	var ext []sig
	for _, t := range intT {
		ext = append(ext, sig{params: []Type{t, U32, U32}, ret: t})
	}
	reg(&builtinFn{name: "extractBits", mustUse: true, isConst: true, cands: fixed(ext...), eval: evalGeneric("extractBits")})

	// length, distance, dot, cross, normalize, faceForward, reflect, refract
	var ln, ds []sig
	for _, t := range floatT {
		ln = append(ln, sig{params: []Type{t}, ret: scalarOf(t)})
		ds = append(ds, sig{params: []Type{t, t}, ret: scalarOf(t)})
	}
	reg(&builtinFn{name: "length", mustUse: true, isConst: true, cands: fixed(ln...), eval: evalGeneric("length")})
	reg(&builtinFn{name: "distance", mustUse: true, isConst: true, cands: fixed(ds...), eval: evalGeneric("distance")})
	var dot, cross, norm, ff, refl, refr []sig
	for _, s := range scalarsAll {
		for n := 2; n <= 4; n++ {
			v := &VecType{n, s}
			dot = append(dot, sig{params: []Type{v, v}, ret: s})
		}
	}
	for _, s := range floatsAll {
		v3 := &VecType{3, s}
		cross = append(cross, sig{params: []Type{v3, v3}, ret: v3})
		for n := 2; n <= 4; n++ {
			v := &VecType{n, s}
			norm = append(norm, sig{params: []Type{v}, ret: v})
			ff = append(ff, sig{params: []Type{v, v, v}, ret: v})
			refl = append(refl, sig{params: []Type{v, v}, ret: v})
			refr = append(refr, sig{params: []Type{v, v, s}, ret: v})
		}
	}
	reg(&builtinFn{name: "dot", mustUse: true, isConst: true, cands: fixed(dot...), eval: evalGeneric("dot")})
	reg(&builtinFn{name: "cross", mustUse: true, isConst: true, cands: fixed(cross...), eval: evalGeneric("cross")})
	reg(&builtinFn{name: "normalize", mustUse: true, isConst: true, cands: fixed(norm...), eval: evalGeneric("normalize")})
	reg(&builtinFn{name: "faceForward", mustUse: true, isConst: true, cands: fixed(ff...)})
	reg(&builtinFn{name: "reflect", mustUse: true, isConst: true, cands: fixed(refl...)})
	reg(&builtinFn{name: "refract", mustUse: true, isConst: true, cands: fixed(refr...)})
	// mix
	mix := ternary(floatT)
	for _, s := range floatsAll {
		for n := 2; n <= 4; n++ {
			v := &VecType{n, s}
			mix = append(mix, sig{params: []Type{v, v, s}, ret: v})
		}
	}
	reg(&builtinFn{name: "mix", mustUse: true, isConst: true, cands: fixed(mix...), eval: evalGeneric("mix")})
	// ldexp
	var ld []sig
	for _, p := range [][2]*ScalarType{{AbstractFloat, AbstractInt}, {F32, I32}, {F16, I32}} {
		ld = append(ld, sig{params: []Type{p[0], p[1]}, ret: p[0]})
		for n := 2; n <= 4; n++ {
			ld = append(ld, sig{params: []Type{&VecType{n, p[0]}, &VecType{n, p[1]}}, ret: &VecType{n, p[0]}})
		}
	}
	reg(&builtinFn{name: "ldexp", mustUse: true, isConst: true, cands: fixed(ld...)})
	// matrices
	var det, tr []sig
	for _, s := range floatsAll {
		for n := 2; n <= 4; n++ {
			det = append(det, sig{params: []Type{&MatType{n, n, s}}, ret: s})
		}
		for cc := 2; cc <= 4; cc++ {
			for r := 2; r <= 4; r++ {
				tr = append(tr, sig{params: []Type{&MatType{cc, r, s}}, ret: &MatType{r, cc, s}})
			}
		}
	}
	reg(&builtinFn{name: "determinant", mustUse: true, isConst: true, cands: fixed(det...)})
	reg(&builtinFn{name: "transpose", mustUse: true, isConst: true, cands: fixed(tr...), eval: evalGeneric("transpose")})
	// frexp, modf
	for _, name := range []string{"frexp", "modf"} {
		name := name
		var s []sig
		for _, t := range floatT {
			s = append(s, sig{params: []Type{t}, ret: resultStruct(name, t)})
		}
		reg(&builtinFn{name: name, mustUse: true, isConst: true, cands: fixed(s...)})
	}
	reg(&builtinFn{name: "quantizeToF16", mustUse: true, isConst: true,
		cands: fixed(unary(append([]Type{F32}, vecsOf(F32)...))...), eval: evalGeneric("quantizeToF16")})
	// logical
	var all []sig
	all = append(all, sig{params: []Type{Bool}, ret: Bool})
	for _, v := range vecsOf(Bool) {
		all = append(all, sig{params: []Type{v}, ret: Bool})
	}
	reg(&builtinFn{name: "all", mustUse: true, isConst: true, cands: fixed(all...), eval: evalGeneric("all")})
	reg(&builtinFn{name: "any", mustUse: true, isConst: true, cands: fixed(all...), eval: evalGeneric("any")})
	var sel []sig
	for _, s := range append([]*ScalarType{Bool}, scalarsAll...) {
		sel = append(sel, sig{params: []Type{s, s, Bool}, ret: s})
		for n := 2; n <= 4; n++ {
			v := &VecType{n, s}
			sel = append(sel, sig{params: []Type{v, v, Bool}, ret: v}, sig{params: []Type{v, v, &VecType{n, Bool}}, ret: v})
		}
	}
	reg(&builtinFn{name: "select", mustUse: true, isConst: true, cands: fixed(sel...), eval: evalGeneric("select")})
	// derivatives
	for _, n := range []string{"dpdx", "dpdxCoarse", "dpdxFine", "dpdy", "dpdyCoarse", "dpdyFine", "fwidth",
		"fwidthCoarse", "fwidthFine"} {
		reg(&builtinFn{name: n, mustUse: true, cands: fixed(unary(append([]Type{F32}, vecsOf(F32)...))...), stage: "fragment"})
	}
	// packing
	pk := func(name string, p, r Type, lang string) {
		reg(&builtinFn{name: name, mustUse: true, isConst: true, cands: fixed(sig{params: []Type{p}, ret: r}), lang: lang})
	}
	pk("pack4x8snorm", vec(4, F32), U32, "")
	pk("pack4x8unorm", vec(4, F32), U32, "")
	pk("pack2x16snorm", vec(2, F32), U32, "")
	pk("pack2x16unorm", vec(2, F32), U32, "")
	pk("pack2x16float", vec(2, F32), U32, "")
	pk("pack4xI8", vec(4, I32), U32, "packed_4x8_integer_dot_product")
	pk("pack4xU8", vec(4, U32), U32, "packed_4x8_integer_dot_product")
	pk("pack4xI8Clamp", vec(4, I32), U32, "packed_4x8_integer_dot_product")
	pk("pack4xU8Clamp", vec(4, U32), U32, "packed_4x8_integer_dot_product")
	pk("unpack4x8snorm", U32, vec(4, F32), "")
	pk("unpack4x8unorm", U32, vec(4, F32), "")
	pk("unpack2x16snorm", U32, vec(2, F32), "")
	pk("unpack2x16unorm", U32, vec(2, F32), "")
	pk("unpack2x16float", U32, vec(2, F32), "")
	pk("unpack4xI8", U32, vec(4, I32), "packed_4x8_integer_dot_product")
	pk("unpack4xU8", U32, vec(4, U32), "packed_4x8_integer_dot_product")
	reg(&builtinFn{name: "dot4U8Packed", mustUse: true, isConst: true, cands: fixed(sig{params: []Type{U32, U32}, ret: U32}), lang: "packed_4x8_integer_dot_product"})
	reg(&builtinFn{name: "dot4I8Packed", mustUse: true, isConst: true, cands: fixed(sig{params: []Type{U32, U32}, ret: I32}), lang: "packed_4x8_integer_dot_product"})
	// synchronization
	reg(&builtinFn{name: "workgroupBarrier", stage: "compute", cands: fixed(sig{ret: Void})})
	reg(&builtinFn{name: "storageBarrier", stage: "compute", cands: fixed(sig{ret: Void})})
	reg(&builtinFn{name: "textureBarrier", stage: "compute", cands: fixed(sig{ret: Void}), lang: "readonly_and_readwrite_storage_textures"})
	// subgroup and quad
	numScalarVec := shapes([]*ScalarType{I32, U32, F32, F16})
	sg := func(name string, ts []Type, extra ...Type) {
		var s []sig
		for _, t := range ts {
			s = append(s, sig{params: append([]Type{t}, extra...), ret: t})
		}
		reg(&builtinFn{name: name, mustUse: true, cands: fixed(s...), stage: "fragment|compute", ext: "subgroups"})
	}
	for _, n := range []string{"subgroupAdd", "subgroupExclusiveAdd", "subgroupInclusiveAdd", "subgroupBroadcastFirst",
		"subgroupMax", "subgroupMin", "subgroupMul", "subgroupExclusiveMul", "subgroupInclusiveMul", "quadSwapDiagonal", "quadSwapX", "quadSwapY"} {
		sg(n, numScalarVec)
	}
	for _, n := range []string{"subgroupAnd", "subgroupOr", "subgroupXor"} {
		sg(n, intT)
	}
	for _, n := range []string{"subgroupShuffleDown", "subgroupShuffleUp", "subgroupShuffleXor"} {
		sg(n, numScalarVec, U32)
	}
	for _, n := range []string{"subgroupBroadcast", "subgroupShuffle", "quadBroadcast"} {
		var s []sig
		for _, t := range numScalarVec {
			s = append(s, sig{params: []Type{t, I32}, ret: t}, sig{params: []Type{t, U32}, ret: t})
		}
		reg(&builtinFn{name: n, mustUse: true, cands: fixed(s...), stage: "fragment|compute", ext: "subgroups"})
	}
	for _, n := range []string{"subgroupAll", "subgroupAny"} {
		reg(&builtinFn{name: n, mustUse: true, cands: fixed(sig{params: []Type{Bool}, ret: Bool}), stage: "fragment|compute", ext: "subgroups"})
	}
	reg(&builtinFn{name: "subgroupBallot", mustUse: true, cands: fixed(sig{params: []Type{Bool}, ret: vec(4, U32)}), stage: "fragment|compute", ext: "subgroups"})
	reg(&builtinFn{name: "subgroupElect", mustUse: true, cands: fixed(sig{ret: Bool}), stage: "fragment|compute", ext: "subgroups"})

	registerCustom()
}

var resultCache = map[string]*StructType{}

// resultStruct returns the predeclared result structure of frexp/modf for
// operand type t.
func resultStruct(fn string, t Type) *StructType {
	sc := scalarOf(t)
	suffix := sc.String()
	switch sc.Kind {
	case KAbstractFloat:
		suffix = "abstract"
	}
	name := "__" + fn + "_result_"
	if v, ok := t.(*VecType); ok {
		name += fmt.Sprintf("vec%d_", v.N)
	}
	name += suffix
	if s, ok := resultCache[name]; ok {
		return s
	}
	st := &StructType{Name: name, Predeclared: true}
	var second Type = t
	if fn == "frexp" {
		ei := Type(I32)
		if sc.Kind == KAbstractFloat {
			ei = AbstractInt
		}
		if v, ok := t.(*VecType); ok {
			ei = &VecType{v.N, ei.(*ScalarType)}
		}
		second = ei
		st.Members = []*FieldInfo{{Name: "fract", Type: t}, {Name: "exp", Type: second}}
	} else {
		st.Members = []*FieldInfo{{Name: "fract", Type: t}, {Name: "whole", Type: t}}
	}
	resultCache[name] = st
	return st
}

// ---- calls ----

func (c *checker) callExpr(e *Call) *val {
	fn := e.Fn
	o := c.lookup(fn.Name)
	if o == nil {
		c.err(fn, "undeclared-ident", "undeclared function or type %q", fn.Name)
		c.argsOnly(e)
		return invalidVal()
	}
	c.info.Objects[fn] = o
	switch o := o.(type) {
	case *FuncObj:
		return c.userCall(e, o)
	case *TypeObj:
		if len(fn.Targs) > 0 {
			c.err(fn, "type-args", "%q does not take template arguments", fn.Name)
		}
		if !c.use(o, fn) || o.Type == nil {
			c.argsOnly(e)
			return invalidVal()
		}
		c.info.TypeSpecs[fn] = o.Type
		return c.ctorCall(e, o.Type, false)
	case *BuiltinObj:
		if f := builtinFuncs[fn.Name]; f != nil {
			if len(fn.Targs) > 0 && fn.Name != "bitcast" {
				c.err(fn, "type-args", "%q does not take template arguments", fn.Name)
			}
			return c.builtinCall(e, f)
		}
		name := fn.Name
		_, _, el, isVM := vecMatName(name)
		if name == "bitcast" {
			return c.bitcast(e)
		}
		if (isVM && el == nil && len(fn.Targs) == 0) || (name == "array" && len(fn.Targs) == 0) {
			return c.ctorCall(e, nil, true)
		}
		if _, ok := scalarByName[name]; ok || isVM || name == "array" {
			t := c.typeSpec(fn)
			if isInvalid(t) {
				c.argsOnly(e)
				return invalidVal()
			}
			return c.ctorCall(e, t, false)
		}
		c.err(fn, "not-callable", "%q cannot be called", name)
		c.argsOnly(e)
		return invalidVal()
	}
	c.err(fn, "not-callable", "%q is not a function", fn.Name)
	c.argsOnly(e)
	return invalidVal()
}

func (c *checker) argsOnly(e *Call) {
	for _, a := range e.Args {
		c.expr(a)
	}
}

func (c *checker) userCall(e *Call, f *FuncObj) *val {
	if len(e.Fn.Targs) > 0 {
		c.err(e.Fn, "type-args", "function %q does not take template arguments", f.Name)
	}
	c.use(f, e.Fn)
	if c.cur != nil {
		c.cur.uses[f] = true
	}
	args := make([]*val, len(e.Args))
	for i, a := range e.Args {
		args[i] = c.expr(a)
	}
	if f.Entry != nil || (f.Decl != nil && hasStageAttr(f.Decl)) {
		c.err(e, "entry-point-call", "entry point %q cannot be called", f.Name)
	}
	if len(args) != len(f.Params) {
		c.err(e, "arg-count", "%s expects %d arguments, found %d", f.Name, len(f.Params), len(args))
		return c.retVal(f)
	}
	for i, a := range args {
		p := f.Params[i]
		if isInvalid(a.t) || p.Type == nil || isInvalid(p.Type) {
			continue
		}
		if _, isPtr := p.Type.(*PtrType); isPtr {
			if !Same(a.t, p.Type) {
				c.err(e.Args[i], "type-mismatch", "argument %d: expected %s, found %s", i+1, p.Type, a.t)
			}
			continue
		}
		c.convertTo(a, p.Type)
	}
	return c.retVal(f)
}

func (c *checker) retVal(f *FuncObj) *val {
	if f.Ret == nil || isInvalid(f.Ret) {
		if f.Ret == nil {
			return &val{t: Invalid}
		}
		return invalidVal()
	}
	return &val{t: f.Ret, k: kRuntime}
}

func hasStageAttr(f *Func) bool {
	for _, a := range f.Attrs {
		switch a.Name {
		case "vertex", "fragment", "compute":
			return true
		}
	}
	return false
}

func (c *checker) noteFeature(e Node, key string) {
	if c.cur == nil {
		return
	}
	if c.cur.feats == nil {
		c.cur.feats = map[string]*Node0{}
	}
	if _, ok := c.cur.feats[key]; !ok {
		c.cur.feats[key] = &Node0{Off: e.NodeSpan().Start, What: key}
	}
}

func (c *checker) builtinCall(e *Call, f *builtinFn) *val {
	if f.ext != "" && !c.enabled[f.ext] {
		c.err(e.Fn, "extension-required", "%s requires `enable %s;`", f.name, f.ext)
	}
	if f.lang != "" && !c.lang[f.lang] {
		c.err(e.Fn, "extension-required", "%s requires the %s language extension", f.name, f.lang)
	}
	if f.stage != "" {
		c.noteFeature(e.Fn, "stage:"+f.name+":"+f.stage)
	}
	args := make([]*val, len(e.Args))
	for i, a := range e.Args {
		args[i] = c.expr(a)
	}
	if f.custom != nil {
		return f.custom(c, e, args)
	}
	for i := range args {
		if _, isPtr := args[i].t.(*PtrType); !isPtr {
			args[i] = c.load(args[i])
		}
	}
	cands := f.cands(c, args)
	s := c.pick(e, f.name, args, cands)
	if s == nil {
		return invalidVal()
	}
	allConst := true
	for i := range args {
		args[i] = c.convertTo(args[i], s.params[i])
		if isInvalid(args[i].t) {
			return invalidVal()
		}
		if args[i].k != kConst || args[i].c == nil {
			allConst = false
		}
	}
	r := &val{t: s.ret, k: kRuntime}
	if f.isConst {
		r.k = minKind(append([]*val{{k: kConst}}, args...)...)
		if allConst && f.eval != nil {
			cs := make([]*Const, len(args))
			for i, a := range args {
				cs[i] = a.c
			}
			cv, err := f.eval(s, cs)
			if err != nil {
				c.err(e, err.Rule, "%s", err.Msg)
			} else {
				r.c = cv
			}
		}
	}
	c.builtinChecks(e, f, s, args)
	if len(args) == 0 {
		r.k = kRuntime
	}
	return r
}

// builtinChecks applies the per-function constraints on argument values.
func (c *checker) builtinChecks(e *Call, f *builtinFn, s *sig, args []*val) {
	switch f.name {
	case "clamp":
		if args[1].c != nil && args[2].c != nil && args[1].k == kConst && args[2].k == kConst && s.params[1] != nil {
			if compareVec("<", args[2].c, args[1].c) {
				c.err(e, "const-clamp", "clamp: low must not be greater than high")
			}
		}
	case "extractBits", "insertBits":
		n := len(args)
		if args[n-1].k == kConst && args[n-2].k == kConst && args[n-1].c != nil && args[n-2].c != nil {
			if args[n-1].c.Int+args[n-2].c.Int > 32 {
				c.err(e, "const-bits", "offset + count must not exceed 32")
			}
		}
	case "smoothstep":
		if args[0].k == kConst && args[1].k == kConst && args[0].c != nil && args[1].c != nil && compareVec("==", args[0].c, args[1].c) {
			c.err(e, "const-smoothstep", "smoothstep: edge0 must not equal edge1")
		}
	case "ldexp":
		if args[1].k == kConst && args[1].c != nil {
			max := int64(127)
			if scalarOf(s.params[0]).Kind == KF16 {
				max = 15
			}
			if scalarOf(s.params[0]).Kind == KAbstractFloat {
				max = 1023
			}
			forEachScalar(args[1].c, func(k *Const) {
				if k.Int > max+1 {
					c.err(e, "const-ldexp", "ldexp exponent %d is too large", k.Int)
				}
			})
		}
	case "quantizeToF16":
		_ = s
	}
}

func forEachScalar(k *Const, f func(*Const)) {
	if v, ok := k.Type.(*VecType); ok {
		for i := 0; i < v.N; i++ {
			f(k.Elem(int64(i)))
		}
		return
	}
	f(k)
}

func compareVec(op string, a, b *Const) bool {
	r := false
	forEachScalar(a, func(x *Const) {
		forEachScalar(b, func(y *Const) {
			if s, ok := x.Type.(*ScalarType); ok && compare(op, s, x, y) {
				r = true
			}
		})
	})
	return r
}

func (c *checker) bitcast(e *Call) *val {
	args := make([]*val, len(e.Args))
	for i, a := range e.Args {
		args[i] = c.load(c.expr(a))
	}
	if len(e.Fn.Targs) != 1 || len(args) != 1 {
		c.err(e, "type-args", "bitcast<T>(e) takes one template argument and one argument")
		return invalidVal()
	}
	to := c.typeArg(e.Fn.Targs[0])
	c.info.TypeSpecs[e.Fn] = to
	a := args[0]
	if isInvalid(to) || isInvalid(a.t) {
		return invalidVal()
	}
	okScalar := func(t Type) bool {
		s, ok := t.(*ScalarType)
		return ok && (s == I32 || s == U32 || s == F32)
	}
	var ok bool
	switch t := to.(type) {
	case *ScalarType:
		ok = (okScalar(t) && (okScalar(a.t) || a.t == AbstractInt || a.t == AbstractFloat && t == F32)) ||
			(t == F16 && (a.t == F16 || Same(a.t, vec(2, F16)) == false && false))
		if t == F16 && a.t == F16 {
			ok = true
		}
		if t == F16 {
			ok = ok || false
		}
	case *VecType:
		if av, isV := a.t.(*VecType); isV && av.N == t.N {
			ok = okScalar(t.Elem) && (okScalar(av.Elem) || av.Elem == AbstractInt) || t.Elem == F16 && av.Elem == F16
		}
		if t.N == 2 && t.Elem == F16 && (a.t == U32 || a.t == I32 || a.t == F32 || a.t == AbstractInt) {
			ok = true
		}
		if t.N == 4 && t.Elem == F16 && Same(a.t, vec(2, U32)) {
			ok = true
		}
		if t.N == 2 && okScalar(t.Elem) && Same(a.t, vec(4, F16)) {
			ok = true
		}
		if t.N == 2 && okScalar(t.Elem) && a.t == F16 {
			ok = false
		}
	}
	// reinterpretations between f16 vectors and 32-bit values
	if s, isS := to.(*ScalarType); isS && okScalar(s) {
		if Same(a.t, vec(2, F16)) {
			ok = true
		}
	}
	if !ok {
		c.err(e, "bitcast", "cannot bitcast %s to %s", a.t, to)
		return invalidVal()
	}
	if (scalarOf(to) == F16 || scalarOf(a.t) == F16) && !c.enabled["f16"] {
		c.err(e, "extension-required", "f16 requires `enable f16;`")
	}
	if IsAbstract(a.t) {
		a = c.convertTo(a, concreteFor(a.t, to))
	}
	return &val{t: to, k: kRuntime}
}

func concreteFor(src, to Type) Type {
	if scalarOf(src) == AbstractFloat {
		return concretize(src)
	}
	if v, ok := src.(*VecType); ok {
		return &VecType{v.N, U32}
	}
	if scalarOf(to) == I32 {
		return I32
	}
	return U32
}

func typeKey(t Type) string { return strings.ReplaceAll(t.String(), " ", "") }
