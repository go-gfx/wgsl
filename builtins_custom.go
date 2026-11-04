package wgsl

import "strings"

func registerCustom() {
	reg(&builtinFn{name: "arrayLength", mustUse: true, ptrArgs: true, custom: arrayLength})
	reg(&builtinFn{name: "workgroupUniformLoad", mustUse: true, stage: "compute", custom: workgroupUniformLoad})
	for _, n := range []string{"atomicLoad", "atomicStore", "atomicAdd", "atomicSub", "atomicMax", "atomicMin",
		"atomicAnd", "atomicOr", "atomicXor", "atomicExchange", "atomicCompareExchangeWeak"} {
		n := n
		reg(&builtinFn{name: n, stage: "notvertex", custom: func(c *checker, e *Call, a []*val) *val { return c.atomicCall(n, e, a) }})
	}
	for _, n := range []string{"atomicStoreMin", "atomicStoreMax"} {
		n := n
		reg(&builtinFn{name: n, stage: "notvertex", custom: func(c *checker, e *Call, a []*val) *val { return c.atomicCall(n, e, a) }})
	}
	for _, n := range []string{"textureDimensions", "textureGather", "textureGatherCompare", "textureLoad",
		"textureNumLayers", "textureNumLevels", "textureNumSamples", "textureSample", "textureSampleBias",
		"textureSampleCompare", "textureSampleCompareLevel", "textureSampleGrad", "textureSampleLevel",
		"textureSampleBaseClampToEdge", "textureStore"} {
		n := n
		f := &builtinFn{name: n, mustUse: n != "textureStore", custom: func(c *checker, e *Call, a []*val) *val { return c.texCall(n, e, a) }}
		switch n {
		case "textureSample", "textureSampleBias", "textureSampleCompare":
			f.stage = "fragment"
		}
		reg(f)
	}
}

func arrayLength(c *checker, e *Call, args []*val) *val {
	if len(args) != 1 {
		c.err(e, "arg-count", "arrayLength takes one argument")
		return invalidVal()
	}
	a := args[0]
	if isInvalid(a.t) {
		return a
	}
	p, ok := a.t.(*PtrType)
	if !ok {
		c.err(e, "type-mismatch", "arrayLength requires a pointer to a runtime-sized array, found %s", a.t)
		return invalidVal()
	}
	if arr, ok := p.Elem.(*ArrayType); !ok || !arr.Runtime || p.AS != ASStorage {
		c.err(e, "type-mismatch", "arrayLength requires a pointer to a runtime-sized array in the storage address space, found %s", a.t)
		return invalidVal()
	}
	return &val{t: U32, k: kRuntime}
}

func workgroupUniformLoad(c *checker, e *Call, args []*val) *val {
	if len(args) != 1 || isInvalid(args[0].t) {
		if len(args) != 1 {
			c.err(e, "arg-count", "workgroupUniformLoad takes one argument")
		}
		return invalidVal()
	}
	p, ok := args[0].t.(*PtrType)
	if !ok || p.AS != ASWorkgroup || p.Access != AccessReadWrite || !constructible(p.Elem) && !containsAtomic(p.Elem) {
		c.err(e, "type-mismatch", "workgroupUniformLoad requires ptr<workgroup, T, read_write> with constructible T, found %s", args[0].t)
		return invalidVal()
	}
	if containsAtomic(p.Elem) || !constructible(p.Elem) {
		c.err(e, "type-mismatch", "workgroupUniformLoad requires a constructible type")
		return invalidVal()
	}
	return &val{t: p.Elem, k: kRuntime}
}

var atomicCmpCache = map[Type]*StructType{}

func (c *checker) atomicCall(name string, e *Call, args []*val) *val {
	want := 2
	switch name {
	case "atomicLoad":
		want = 1
	case "atomicCompareExchangeWeak":
		want = 3
	}
	if len(args) != want {
		c.err(e, "arg-count", "%s takes %d arguments, found %d", name, want, len(args))
		return invalidVal()
	}
	if anyInvalid(args) {
		return invalidVal()
	}
	p, ok := args[0].t.(*PtrType)
	var at *AtomicType
	if ok {
		at, _ = p.Elem.(*AtomicType)
	}
	if !ok || at == nil || p.Access != AccessReadWrite || (p.AS != ASStorage && p.AS != ASWorkgroup) {
		c.err(e, "type-mismatch", "%s requires ptr<storage|workgroup, atomic<T>, read_write>, found %s", name, args[0].t)
		return invalidVal()
	}
	_, vec2 := at.Elem.(*VecType)
	isMM := name == "atomicStoreMin" || name == "atomicStoreMax"
	if vec2 != isMM {
		c.err(e, "type-mismatch", "%s cannot operate on %s", name, at)
		return invalidVal()
	}
	if vec2 && p.AS != ASStorage {
		c.err(e, "type-mismatch", "atomic<vec2<u32>> requires the storage address space")
		return invalidVal()
	}
	for i := 1; i < len(args); i++ {
		if r := c.convertTo(args[i], at.Elem); isInvalid(r.t) {
			return invalidVal()
		}
	}
	switch name {
	case "atomicStore", "atomicStoreMin", "atomicStoreMax":
		return &val{t: Void, k: kRuntime}
	case "atomicCompareExchangeWeak":
		st := atomicCmpCache[at.Elem]
		if st == nil {
			st = &StructType{Name: "__atomic_compare_exchange_result_" + at.Elem.String(), Predeclared: true,
				Members: []*FieldInfo{{Name: "old_value", Type: at.Elem}, {Name: "exchanged", Type: Bool}}}
			atomicCmpCache[at.Elem] = st
		}
		return &val{t: st, k: kRuntime}
	}
	return &val{t: at.Elem, k: kRuntime}
}

// ---- textures ----

var intIdx = []*ScalarType{I32, U32}

func coordF(d TextureDim) Type {
	switch d {
	case Dim1D:
		return F32
	case Dim2D:
		return vec(2, F32)
	}
	return vec(3, F32)
}

func coordI(d TextureDim, s *ScalarType) Type {
	switch d {
	case Dim1D:
		return s
	case Dim2D:
		return vec(2, s)
	}
	return vec(3, s)
}

func dimsRet(t *TextureType) Type {
	switch {
	case t.Dim == Dim1D:
		return U32
	case t.Dim == Dim3D:
		return vec(3, U32)
	}
	return vec(2, U32)
}

func storageChannel(format string) *ScalarType {
	switch {
	case strings.HasSuffix(format, "uint"):
		return U32
	case strings.HasSuffix(format, "sint"):
		return I32
	}
	return F32
}

type tsig struct {
	params []Type
	roles  []string
	ret    Type
}

func (c *checker) texCall(name string, e *Call, args []*val) *val {
	for i := range args {
		args[i] = c.load(args[i])
	}
	if anyInvalid(args) {
		return invalidVal()
	}
	ti := 0
	if name == "textureGather" && len(args) > 0 && isScalar(args[0].t) {
		ti = 1
	}
	if len(args) <= ti {
		c.err(e, "arg-count", "%s: missing arguments", name)
		return invalidVal()
	}
	tex, ok := args[ti].t.(*TextureType)
	if !ok {
		c.err(e, "type-mismatch", "%s: expected a texture argument, found %s", name, args[ti].t)
		return invalidVal()
	}
	ts := texSigs(name, tex, ti == 1)
	var cands []sig
	for _, t := range ts {
		cands = append(cands, sig{params: t.params, ret: t.ret, tag: strings.Join(t.roles, ",")})
	}
	s := c.pick(e, name, args, cands)
	if s == nil {
		return invalidVal()
	}
	roles := strings.Split(s.tag, ",")
	for i := range args {
		args[i] = c.convertTo(args[i], s.params[i])
	}
	for i, r := range roles {
		if i >= len(args) {
			break
		}
		switch r {
		case "offset":
			if args[i].k != kConst {
				c.err(e.Args[i], "const-arg", "texture offset must be a const-expression")
			} else if args[i].c != nil {
				forEachScalar(args[i].c, func(k *Const) {
					if k.Int < -8 || k.Int > 7 {
						c.err(e.Args[i], "const-arg", "texture offset component %d is outside [-8, 7]", k.Int)
					}
				})
			}
		case "component":
			if args[i].k != kConst {
				c.err(e.Args[i], "const-arg", "textureGather component must be a const-expression")
			} else if args[i].c != nil && (args[i].c.Int < 0 || args[i].c.Int > 3) {
				c.err(e.Args[i], "const-arg", "textureGather component must be in [0, 3]")
			}
		}
	}
	return &val{t: s.ret, k: kRuntime}
}

func texSigs(name string, t *TextureType, hasComp bool) []tsig {
	var out []tsig
	add := func(ret Type, ps []Type, roles []string) {
		out = append(out, tsig{params: ps, roles: roles, ret: ret})
	}
	samp := &SamplerType{}
	cmp := &SamplerType{Comparison: true}
	cube := t.Dim == DimCube
	offOK := (t.Dim == Dim2D || t.Dim == Dim3D) && !cube
	var offT Type
	if t.Dim == Dim2D {
		offT = vec(2, I32)
	} else if t.Dim == Dim3D {
		offT = vec(3, I32)
	}
	depth := t.Kind == TexDepth
	sampled := t.Kind == TexSampled
	sampleRet := Type(vec(4, F32))
	if depth {
		sampleRet = F32
	}
	// withExtras builds [t, s, coords, (array_index), extras..., (offset)].
	build := func(s Type, extra []Type, extraRoles []string, offsetOK bool, ret Type) {
		for _, ai := range arrayIdx(t.Arrayed) {
			ps := []Type{t, s, coordF(t.Dim)}
			roles := []string{"t", "s", "coords"}
			if ai != nil {
				ps = append(ps, ai)
				roles = append(roles, "array_index")
			}
			ps = append(ps, extra...)
			roles = append(roles, extraRoles...)
			add(ret, ps, roles)
			if offsetOK && offT != nil {
				add(ret, append(append([]Type{}, ps...), offT), append(append([]string{}, roles...), "offset"))
			}
		}
	}
	f32sampled := (sampled && t.Sampled == F32) || depth
	switch name {
	case "textureSample":
		if f32sampled {
			build(samp, nil, nil, offOK, sampleRet)
		}
	case "textureSampleBias":
		if sampled && t.Sampled == F32 && t.Dim != Dim1D {
			build(samp, []Type{F32}, []string{"bias"}, offOK, sampleRet)
		}
	case "textureSampleLevel":
		if sampled && t.Sampled == F32 {
			build(samp, []Type{F32}, []string{"level"}, offOK, sampleRet)
		} else if depth {
			for _, l := range intIdx {
				build(samp, []Type{l}, []string{"level"}, offOK, F32)
			}
		}
	case "textureSampleGrad":
		if sampled && t.Sampled == F32 && t.Dim != Dim1D {
			g := coordF(t.Dim)
			build(samp, []Type{g, g}, []string{"ddx", "ddy"}, offOK, sampleRet)
		}
	case "textureSampleCompare", "textureSampleCompareLevel":
		if depth {
			build(cmp, []Type{F32}, []string{"depth_ref"}, offOK, F32)
		}
	case "textureGatherCompare":
		if depth && t.Dim != Dim1D {
			build(cmp, []Type{F32}, []string{"depth_ref"}, offOK, vec(4, F32))
		}
	case "textureGather":
		if t.Dim == Dim1D || t.Dim == Dim3D {
			break
		}
		if depth && !hasComp {
			build(samp, nil, nil, offOK, vec(4, F32))
		} else if sampled && hasComp {
			for _, cs := range intIdx {
				for _, ai := range arrayIdx(t.Arrayed) {
					ps := []Type{cs, t, samp, coordF(t.Dim)}
					roles := []string{"component", "t", "s", "coords"}
					if ai != nil {
						ps = append(ps, ai)
						roles = append(roles, "array_index")
					}
					add(vec(4, t.Sampled), ps, roles)
					if offOK {
						add(vec(4, t.Sampled), append(append([]Type{}, ps...), offT), append(append([]string{}, roles...), "offset"))
					}
				}
			}
		}
	case "textureSampleBaseClampToEdge":
		if (sampled && t.Dim == Dim2D && !t.Arrayed && t.Sampled == F32) || t.Kind == TexExternal {
			add(vec(4, F32), []Type{t, samp, vec(2, F32)}, []string{"t", "s", "coords"})
		}
	case "textureDimensions":
		ret := dimsRet(t)
		add(ret, []Type{t}, []string{"t"})
		if (sampled || depth) && t.Kind != TexMultisampled {
			for _, l := range intIdx {
				add(ret, []Type{t, l}, []string{"t", "level"})
			}
		}
	case "textureNumLayers":
		if t.Arrayed {
			add(U32, []Type{t}, []string{"t"})
		}
	case "textureNumLevels":
		if sampled || depth {
			add(U32, []Type{t}, []string{"t"})
		}
	case "textureNumSamples":
		if t.Kind == TexMultisampled || t.Kind == TexDepthMultisampled {
			add(U32, []Type{t}, []string{"t"})
		}
	case "textureLoad":
		for _, cs := range intIdx {
			cc := coordI(t.Dim, cs)
			switch t.Kind {
			case TexSampled, TexDepth:
				if t.Dim == DimCube {
					break
				}
				ret := Type(vec(4, t.Sampled))
				if depth {
					ret = F32
				}
				for _, ai := range arrayIdx(t.Arrayed) {
					for _, l := range intIdx {
						ps := []Type{t, cc}
						roles := []string{"t", "coords"}
						if ai != nil {
							ps = append(ps, ai)
							roles = append(roles, "array_index")
						}
						add(ret, append(ps, l), append(roles, "level"))
					}
				}
			case TexMultisampled, TexDepthMultisampled:
				ret := Type(vec(4, t.Sampled))
				if t.Kind == TexDepthMultisampled {
					ret = F32
				}
				for _, sm := range intIdx {
					add(ret, []Type{t, cc, sm}, []string{"t", "coords", "sample_index"})
				}
			case TexExternal:
				add(vec(4, F32), []Type{t, cc}, []string{"t", "coords"})
			case TexStorage:
				if t.Access == AccessWrite {
					break
				}
				for _, ai := range arrayIdx(t.Arrayed) {
					ps := []Type{t, cc}
					roles := []string{"t", "coords"}
					if ai != nil {
						ps = append(ps, ai)
						roles = append(roles, "array_index")
					}
					add(vec(4, storageChannel(t.Format)), ps, roles)
				}
			}
		}
	case "textureStore":
		if t.Kind == TexStorage && t.Access != AccessRead {
			for _, cs := range intIdx {
				for _, ai := range arrayIdx(t.Arrayed) {
					ps := []Type{t, coordI(t.Dim, cs)}
					roles := []string{"t", "coords"}
					if ai != nil {
						ps = append(ps, ai)
						roles = append(roles, "array_index")
					}
					add(Void, append(ps, vec(4, storageChannel(t.Format))), append(roles, "value"))
				}
			}
		}
	}
	return out
}

// arrayIdx lists the array index types (a single nil for non-arrayed).
func arrayIdx(arrayed bool) []Type {
	if !arrayed {
		return []Type{nil}
	}
	return []Type{I32, U32}
}
