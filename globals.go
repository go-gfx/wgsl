package wgsl

// ---- attributes ----

var attrPlaces = map[string]map[string]bool{
	"align":          {"member": true},
	"size":           {"member": true},
	"binding":        {"var": true},
	"group":          {"var": true},
	"id":             {"override": true},
	"builtin":        {"param": true, "ret": true, "member": true},
	"location":       {"param": true, "ret": true, "member": true},
	"interpolate":    {"param": true, "ret": true, "member": true},
	"invariant":      {"param": true, "ret": true, "member": true},
	"blend_src":      {"param": true, "ret": true, "member": true},
	"compute":        {"func": true},
	"vertex":         {"func": true},
	"fragment":       {"func": true},
	"workgroup_size": {"func": true},
	"subgroup_size":  {"func": true},
	"must_use":       {"func": true},
	"const":          {"func": true},
	"diagnostic":     {"func": true, "block": true},
}

// checkAttrs validates attribute placement and duplicates. The values of
// the layout and binding attributes are checked where they are used.
func (c *checker) checkAttrs(attrs []*Attr, where string, n Node) {
	seen := map[string]bool{}
	diag := map[string]bool{}
	for _, a := range attrs {
		places, known := attrPlaces[a.Name]
		if !known {
			c.err(a, "unknown-attribute", "unknown attribute @%s", a.Name)
			continue
		}
		if !places[where] {
			c.err(a, "attr-placement", "@%s cannot be applied here", a.Name)
			continue
		}
		if a.Name == "diagnostic" && len(a.Args) == 2 {
			r := a.Args[1].(*Ident).Name
			if diag[r] {
				c.err(a, "diagnostic-conflict", "duplicate diagnostic filter for %q", r)
			}
			diag[r] = true
			continue
		}
		if seen[a.Name] {
			c.err(a, "attr-duplicate", "@%s specified more than once", a.Name)
		}
		seen[a.Name] = true
		if a.Name == "const" {
			c.err(a, "attr-placement", "@const cannot be applied to a user-defined function")
		}
		if a.Name == "blend_src" && !c.enabled["dual_source_blending"] {
			c.err(a, "extension-required", "@blend_src requires `enable dual_source_blending;`")
		}
		if a.Name == "subgroup_size" && !c.enabled["subgroup_size_control"] {
			c.err(a, "extension-required", "@subgroup_size requires `enable subgroup_size_control;`")
		}
	}
}

// attrArg evaluates attribute argument idx as a const-expression integer.
func (c *checker) attrInt(a *Attr, idx int) (int64, bool) {
	if idx >= len(a.Args) {
		return 0, false
	}
	v := c.load(c.expr(a.Args[idx]))
	if isInvalid(v.t) {
		return 0, false
	}
	if v.k != kConst {
		c.err(a.Args[idx], "attr-value", "@%s requires a const-expression", a.Name)
		return 0, false
	}
	if !isIntegerScalar(v.t) {
		c.err(a.Args[idx], "attr-value", "@%s requires an i32 or u32, found %s", a.Name, v.t)
		return 0, false
	}
	if v.t == AbstractInt {
		v = c.convertTo(v, I32)
	}
	if v.c == nil {
		return 0, false
	}
	return v.c.Int, true
}

func (c *checker) findAttr(attrs []*Attr, name string) *Attr {
	for _, a := range attrs {
		if a.Name == name {
			return a
		}
	}
	return nil
}

// structAttrs records the IO attributes of structure members and validates
// the value attributes.
func (c *checker) structAttrs(st *StructType) {
	for _, m := range st.Members {
		if m.Decl == nil {
			continue
		}
		c.checkAttrs(m.Decl.Attrs, "member", m.Decl)
		c.ioAttrs(m.Decl.Attrs, &m.BuiltinName, &m.Location, &m.BlendSrc, &m.Interpolate, &m.Invariant)
	}
	// location uniqueness per structure (spec "Input-output Locations")
	seen := map[int64]bool{}
	dual := 0
	for _, m := range st.Members {
		if m.BlendSrc != nil {
			dual++
		}
	}
	if dual > 0 {
		n := 0
		var first Type
		okSame := true
		for _, m := range st.Members {
			if m.Location != nil {
				n++
				if first == nil {
					first = m.Type
				} else if !Same(first, m.Type) {
					okSame = false
				}
			}
		}
		if n != 2 || !okSame {
			c.err(st.Decl, "blend-src", "dual-source blending requires exactly two members of the same type")
		}
		return
	}
	for _, m := range st.Members {
		if m.Location != nil {
			if seen[*m.Location] {
				c.err(m.Decl, "location-duplicate", "location %d is used by more than one member", *m.Location)
			}
			seen[*m.Location] = true
		}
	}
}

func (c *checker) ioAttrs(attrs []*Attr, builtin *string, loc, blend **int64, interp *[2]string, inv *bool) {
	for _, a := range attrs {
		switch a.Name {
		case "builtin":
			if len(a.Args) >= 1 {
				*builtin = a.Args[0].(*Ident).Name
				if !builtinValues[*builtin] {
					c.err(a, "attr-value", "unknown built-in value %q", *builtin)
				}
				if len(a.Args) == 2 {
					c.err(a, "attr-value", "@builtin takes one argument")
				}
			}
		case "location":
			if v, ok := c.attrInt(a, 0); ok {
				if v < 0 {
					c.err(a, "attr-value", "@location must be non-negative")
				}
				*loc = &v
			}
		case "blend_src":
			if v, ok := c.attrInt(a, 0); ok {
				if v != 0 && v != 1 {
					c.err(a, "attr-value", "@blend_src must be 0 or 1")
				}
				*blend = &v
			}
		case "interpolate":
			t := a.Args[0].(*Ident).Name
			switch t {
			case "perspective", "linear", "flat":
			default:
				c.err(a, "attr-value", "unknown interpolation type %q", t)
			}
			interp[0] = t
			if len(a.Args) > 1 {
				s := a.Args[1].(*Ident).Name
				switch s {
				case "center", "centroid", "sample", "first", "either":
				default:
					c.err(a, "attr-value", "unknown interpolation sampling %q", s)
				}
				interp[1] = s
				if t == "flat" && s != "first" && s != "either" {
					c.err(a, "attr-value", "flat interpolation sampling must be first or either")
				}
				if t != "flat" && (s == "first" || s == "either") {
					c.err(a, "attr-value", "%s interpolation sampling must be center, centroid or sample", t)
				}
			}
		case "invariant":
			*inv = true
		}
	}
	if len(attrs) > 0 {
		if *loc == nil && c.findAttr(attrs, "interpolate") != nil {
			c.err(c.findAttr(attrs, "interpolate"), "attr-placement", "@interpolate requires @location")
		}
		if *loc == nil && *blend != nil {
			c.err(c.findAttr(attrs, "blend_src"), "attr-placement", "@blend_src requires @location")
		}
		if *builtin != "" && *loc != nil {
			c.err(c.findAttr(attrs, "location"), "attr-placement", "@builtin and @location cannot be combined")
		}
		if *inv && *builtin != "position" {
			c.err(c.findAttr(attrs, "invariant"), "attr-placement", "@invariant can only be applied to the position built-in")
		}
	}
}

var builtinValues = map[string]bool{"primitive_id": true, "vertex_index": true, "instance_index": true, "position": true,
	"front_facing": true, "frag_depth": true, "sample_index": true, "sample_mask": true,
	"local_invocation_id": true, "local_invocation_index": true, "global_invocation_id": true,
	"global_invocation_index": true, "workgroup_id": true, "workgroup_index": true, "num_workgroups": true,
	"subgroup_invocation_id": true, "subgroup_size": true, "primitive_index": true, "subgroup_id": true,
	"num_subgroups": true, "clip_distances": true}

// ---- value declarations ----

func (c *checker) valueDecl(o *ValueObj) {
	d := o.Decl
	if d.Kind == "override" {
		c.checkAttrs(d.Attrs, "override", d)
	} else {
		c.checkAttrs(d.Attrs, "none", d)
	}
	var decl Type
	if d.Type != nil {
		decl = c.typeSpec(d.Type)
	}
	var v *val
	if d.Init != nil {
		v = c.load(c.expr(d.Init))
	}
	if v != nil && isInvalid(v.t) {
		o.Type = Invalid
		return
	}
	switch d.Kind {
	case "const":
		if v.k != kConst {
			c.err(d.Init, "const-init", "the initializer of a const declaration must be a const-expression")
		}
		if decl != nil {
			if isInvalid(decl) {
				o.Type = Invalid
				return
			}
			v = c.convertTo(v, decl)
		}
		o.Type, o.Val = v.t, v.c
		if !isInvalid(v.t) && !constructible(stripAbstract(v.t)) {
			c.err(d, "const-type", "const type %s must be constructible", v.t)
		}
	case "override":
		if v != nil && v.k == kRuntime {
			c.err(d.Init, "override-init", "the initializer of an override must be an override-expression")
		}
		switch {
		case decl != nil && !isInvalid(decl):
			if v != nil {
				v = c.convertTo(v, decl)
			}
			o.Type = decl
		case v != nil:
			v = c.materialize(v)
			o.Type = v.t
		case decl != nil:
			o.Type = Invalid
		default:
			c.err(d, "override-type", "an override needs a type or an initializer")
			o.Type = Invalid
		}
		if !isInvalid(o.Type) && !isScalar(o.Type) {
			c.err(d, "override-type", "override type %s must be a concrete scalar", o.Type)
		}
		if a := c.findAttr(d.Attrs, "id"); a != nil {
			if id, ok := c.attrInt(a, 0); ok {
				if id < 0 || id > 65535 {
					c.err(a, "attr-value", "@id must be in [0, 65535]")
				}
				o.ID = id
			}
		}
	}
	c.info.DeclTypes[d] = o.Type
	if d.Kind == "override" {
		for _, other := range c.info.Overrides {
			if other != o && other.ID >= 0 && other.ID == o.ID && other.state == 2 {
				c.err(d, "override-id", "pipeline constant ID %d is used twice", o.ID)
			}
		}
	}
}

// ---- module-scope variables ----

func (c *checker) globalVar(o *VarObj) {
	d := o.Decl
	c.checkAttrs(d.Attrs, "var", d)
	as := ASNone
	acc := AccessNone
	if len(d.Template) > 0 {
		name := c.enumArg(d.Template[0], "address_space")
		as = asOf(name)
		if as == ASFunction {
			c.err(d, "address-space", "variables at module scope cannot be in the function address space")
		}
		if len(d.Template) > 2 {
			c.err(d, "address-space", "too many template arguments")
		} else if len(d.Template) == 2 {
			acc = accessOf(c.enumArg(d.Template[1], "access_mode"))
			if as != ASStorage {
				c.err(d, "address-space", "access mode may only be written for the storage address space")
				acc = AccessNone
			} else if acc == AccessWrite {
				c.err(d, "address-space", "storage variables cannot be write-only")
			}
		}
	}
	var store Type
	var init *val
	if d.Type != nil {
		store = c.typeSpec(d.Type)
	}
	if d.Init != nil {
		init = c.load(c.expr(d.Init))
	}
	switch {
	case store == nil && init != nil:
		init = c.materialize(init)
		store = init.t
	case store == nil:
		c.err(d, "var-type", "a variable needs a type or an initializer")
		store = Invalid
	}
	if isInvalid(store) {
		o.Store = Invalid
		return
	}
	if as == ASNone {
		switch store.(type) {
		case *TextureType, *SamplerType:
			as = ASHandle
		default:
			c.err(d, "address-space", "module-scope variable of type %s needs an address space", store)
			o.Store = Invalid
			return
		}
	} else if _, isH := store.(*TextureType); isH {
		c.err(d, "address-space", "texture variables must not specify an address space")
	} else if _, isS := store.(*SamplerType); isS {
		c.err(d, "address-space", "sampler variables must not specify an address space")
	}
	if acc == AccessNone {
		acc = defaultAccess(as)
	}
	if !storable(store) {
		c.err(d, "var-type", "type %s cannot be stored in a variable", store)
	}
	o.AS, o.Access, o.Store = as, acc, store
	if init != nil {
		if as != ASPrivate {
			c.err(d.Init, "var-init", "only private and function variables may have an initializer")
		} else if init.k == kRuntime {
			c.err(d.Init, "var-init", "the initializer of a private variable must be a const- or override-expression")
		}
		if d.Type != nil {
			c.convertTo(init, store)
		}
	}
	// per address space rules
	switch as {
	case ASUniform, ASStorage:
		if !hostShareable(store) {
			c.err(d, "var-type", "%s variables need a host-shareable type, found %s", as, store)
		} else if as == ASUniform && !constructible(store) {
			c.err(d, "var-type", "uniform variables need a constructible type, found %s", store)
		}
		c.checkLayout(store, as, d)
	case ASImmediate:
		if !hostShareable(store) || !constructible(store) || containsArray(store) {
			c.err(d, "var-type", "immediate variables need a constructible host-shareable type without arrays")
		}
		c.checkLayout(store, as, d)
	case ASWorkgroup:
		if !plainType(store) {
			c.err(d, "var-type", "workgroup variables need a plain type, found %s", store)
		} else if containsRuntimeArray(store) {
			c.err(d, "var-type", "workgroup variables cannot be runtime-sized")
		}
		if hasOverrideCount(store) {
			a, ok := store.(*ArrayType)
			if !ok || !hasOverrideCount(a) || hasOverrideCount(a.Elem) {
				c.err(d, "var-type", "an override-sized array must be the outermost type of a workgroup variable")
			}
		}
	case ASPrivate:
		if !constructible(store) {
			c.err(d, "var-type", "private variables need a constructible type, found %s", store)
		}
	}
	if as != ASWorkgroup && as != ASFunction && hasOverrideCount(store) {
		c.err(d, "var-type", "an override-sized array is only valid in the workgroup address space")
	}
	if containsAtomic(store) && !(as == ASWorkgroup || (as == ASStorage && acc == AccessReadWrite)) {
		c.err(d, "var-type", "atomic types are only valid in workgroup or read_write storage variables")
	}
	if containsAtomicVec2u(store) && as != ASStorage {
		c.err(d, "var-type", "atomic<vec2<u32>> requires the storage address space")
	}
	// resource attributes
	resource := as == ASUniform || as == ASStorage || as == ASHandle
	ga, ba := c.findAttr(d.Attrs, "group"), c.findAttr(d.Attrs, "binding")
	if ga != nil {
		if v, ok := c.attrInt(ga, 0); ok {
			if v < 0 {
				c.err(ga, "attr-value", "@group must be non-negative")
			}
			o.Group = &v
		}
	}
	if ba != nil {
		if v, ok := c.attrInt(ba, 0); ok {
			if v < 0 {
				c.err(ba, "attr-value", "@binding must be non-negative")
			}
			o.Binding = &v
		}
	}
	if resource {
		if ga == nil || ba == nil {
			c.err(d, "resource-binding", "resource variable %q needs @group and @binding", o.Name)
		}
	} else if ga != nil || ba != nil {
		c.err(d, "attr-placement", "@group and @binding can only be applied to resource variables")
	}
	c.info.DeclTypes[d] = store
}

func containsArray(t Type) bool {
	switch t := t.(type) {
	case *ArrayType:
		return true
	case *StructType:
		for _, m := range t.Members {
			if containsArray(m.Type) {
				return true
			}
		}
	}
	return false
}

// requiredAlign is RequiredAlignOf (spec "Address Space Layout
// Constraints").
func (c *checker) requiredAlign(t Type, as AddressSpace) int64 {
	a, _ := alignSize(t)
	if as == ASUniform && !c.lang["uniform_buffer_standard_layout"] {
		switch t.(type) {
		case *ArrayType, *StructType:
			return roundUp(16, a)
		}
	}
	return a
}

// checkLayout applies the address space layout constraints to all
// structures and arrays reachable from t.
func (c *checker) checkLayout(t Type, as AddressSpace, n Node) {
	seen := map[Type]bool{}
	var walk func(t Type)
	walk = func(t Type) {
		switch t := t.(type) {
		case *ArrayType:
			if t.Runtime && as == ASUniform {
				c.err(n, "layout", "runtime-sized arrays are not allowed in the uniform address space")
			}
			a, s := alignSize(t.Elem)
			stride := roundUp(a, s)
			if stride%c.requiredAlign(t.Elem, as) != 0 {
				c.err(n, "layout", "array element stride %d violates the %s alignment %d", stride, as, c.requiredAlign(t.Elem, as))
			}
			if as == ASUniform && !c.lang["uniform_buffer_standard_layout"] && stride%16 != 0 {
				c.err(n, "layout", "array stride %d in the uniform address space must be a multiple of 16", stride)
			}
			walk(t.Elem)
		case *StructType:
			if seen[t] {
				return
			}
			seen[t] = true
			for i, m := range t.Members {
				if r := c.requiredAlign(m.Type, as); r > 0 && m.Offset%r != 0 {
					c.err(n, "layout", "member %q of %s has offset %d, which is not a multiple of %d", m.Name, t.Name, m.Offset, r)
				}
				if m.Decl != nil {
					if a := c.findAttr(m.Decl.Attrs, "align"); a != nil && as != ASUniform {
						if m.Align%c.requiredAlign(m.Type, as) != 0 {
							c.err(a, "layout", "@align(%d) is not a multiple of the required alignment", m.Align)
						}
					}
				}
				if as == ASUniform && !c.lang["uniform_buffer_standard_layout"] {
					if st, ok := m.Type.(*StructType); ok && i+1 < len(t.Members) {
						if t.Members[i+1].Offset-m.Offset < roundUp(16, st.Size) {
							c.err(n, "layout", "member %q of structure type must be followed by at least %d bytes", m.Name, roundUp(16, st.Size))
						}
					}
				}
				walk(m.Type)
			}
		}
	}
	walk(t)
}

// ---- functions ----

func (c *checker) funcSignature(f *FuncObj) {
	d := f.Decl
	c.checkAttrs(d.Attrs, "func", d)
	if len(d.Params) > 255 {
		c.err(d, "limit-params", "functions have at most 255 parameters")
	}
	seen := map[string]bool{}
	for i, p := range d.Params {
		po := &ParamObj{Name: p.Name, Decl: p, Func: f, Index: i}
		c.info.Decls[p] = po
		if seen[p.Name] {
			c.err(p, "redeclaration", "duplicate parameter %q", p.Name)
		}
		seen[p.Name] = true
		c.checkIdentName(p.Name, p)
		c.checkAttrs(p.Attrs, "param", p)
		po.Type = c.typeSpec(p.Type)
		switch t := po.Type.(type) {
		case *PtrType:
			if (t.AS != ASFunction && t.AS != ASPrivate) && !c.lang["unrestricted_pointer_parameters"] {
				c.err(p, "param-type", "pointer parameters must be in the function or private address space")
			}
		case *TextureType, *SamplerType, *InvalidType:
		default:
			if !constructible(po.Type) {
				c.err(p, "param-type", "parameter type %s must be constructible, a pointer, a texture or a sampler", po.Type)
			}
		}
		c.info.DeclTypes[p] = po.Type
		f.Params = append(f.Params, po)
	}
	f.Ret = Void
	if d.Ret != nil {
		f.Ret = c.typeSpec(d.Ret)
		if !isInvalid(f.Ret) && !constructible(f.Ret) {
			c.err(d.Ret, "return-type", "return type %s must be constructible", f.Ret)
		}
		c.checkAttrs(d.RetAttrs, "ret", d.Ret)
	} else if len(d.RetAttrs) > 0 {
		c.err(d, "attr-placement", "return attributes without a return type")
	}
	c.info.DeclTypes[d] = f.Ret
}
