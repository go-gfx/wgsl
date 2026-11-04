package wgsl

import (
	"sort"
	"strings"
)

type builtinSpec struct {
	stage Stage
	in    bool
	typ   func(Type) bool
	ext   string
	lang  string
}

func isT(t Type) func(Type) bool { return func(x Type) bool { return Same(x, t) } }

func builtinSpecs(name string) []builtinSpec {
	u32, v3u, v4f := isT(U32), isT(vec(3, U32)), isT(vec(4, F32))
	switch name {
	case "vertex_index", "instance_index":
		return []builtinSpec{{StageVertex, true, u32, "", ""}}
	case "position":
		return []builtinSpec{{StageVertex, false, v4f, "", ""}, {StageFragment, true, v4f, "", ""}}
	case "front_facing":
		return []builtinSpec{{StageFragment, true, isT(Bool), "", ""}}
	case "frag_depth":
		return []builtinSpec{{StageFragment, false, isT(F32), "", ""}}
	case "sample_index":
		return []builtinSpec{{StageFragment, true, u32, "", ""}}
	case "sample_mask":
		return []builtinSpec{{StageFragment, true, u32, "", ""}, {StageFragment, false, u32, "", ""}}
	case "local_invocation_id", "global_invocation_id", "workgroup_id", "num_workgroups":
		return []builtinSpec{{StageCompute, true, v3u, "", ""}}
	case "local_invocation_index":
		return []builtinSpec{{StageCompute, true, u32, "", ""}}
	case "global_invocation_index", "workgroup_index":
		return []builtinSpec{{StageCompute, true, u32, "", "linear_indexing"}}
	case "subgroup_invocation_id", "subgroup_size":
		return []builtinSpec{{StageCompute, true, u32, "subgroups", ""}, {StageFragment, true, u32, "subgroups", ""}}
	case "subgroup_id", "num_subgroups":
		return []builtinSpec{{StageCompute, true, u32, "subgroups", "subgroup_id"}}
	case "primitive_index":
		return []builtinSpec{{StageFragment, true, u32, "primitive_index", ""}}
	case "clip_distances":
		return []builtinSpec{{StageVertex, false, func(t Type) bool {
			a, ok := t.(*ArrayType)
			return ok && a.Elem == F32 && !a.Runtime && a.Override == nil && a.Count >= 1 && a.Count <= 8
		}, "clip_distances", ""}}
	}
	return nil
}

func (c *checker) entryPoints() {
	ids := map[int64]bool{}
	_ = ids
	for _, f := range c.info.Funcs {
		d := f.Decl
		var stage Stage
		count := 0
		for _, a := range d.Attrs {
			switch a.Name {
			case "vertex":
				stage = StageVertex
				count++
			case "fragment":
				stage = StageFragment
				count++
			case "compute":
				stage = StageCompute
				count++
			}
		}
		if count > 1 {
			c.err(d, "entry-point", "a function can have at most one shader stage attribute")
		}
		if stage == StageNone {
			if a := c.findAttr(d.Attrs, "workgroup_size"); a != nil {
				c.err(a, "attr-placement", "@workgroup_size requires @compute")
			}
			if a := c.findAttr(d.Attrs, "subgroup_size"); a != nil {
				c.err(a, "attr-placement", "@subgroup_size requires @compute")
			}
			c.nonEntryIO(f)
			continue
		}
		ep := &EntryPoint{Func: f, Name: f.Name, Stage: stage, WorkgroupSize: [3]int64{1, 1, 1}}
		f.Entry = ep
		c.info.EntryPoints = append(c.info.EntryPoints, ep)
		c.entryAttrs(ep)
		c.entryIO(ep)
		c.entryAccess(ep)
	}
	// entry points must not be called
	for _, f := range c.info.Funcs {
		for _, callee := range sortedFuncs(f.uses) {
			if callee.Entry != nil {
				c.err(callee.Decl, "entry-point-call", "entry point %q is called by %q", callee.Name, f.Name)
			}
		}
	}
}

func sortedFuncs(m map[Object]bool) []*FuncObj {
	var out []*FuncObj
	for o := range m {
		if f, ok := o.(*FuncObj); ok {
			out = append(out, f)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Decl.Start < out[j].Decl.Start })
	return out
}

func (c *checker) nonEntryIO(f *FuncObj) {
	for _, p := range f.Decl.Params {
		for _, a := range p.Attrs {
			if a.Name != "diagnostic" {
				c.err(a, "attr-placement", "@%s is only valid on entry point parameters", a.Name)
			}
		}
	}
	for _, a := range f.Decl.RetAttrs {
		c.err(a, "attr-placement", "@%s is only valid on entry point return types", a.Name)
	}
}

func (c *checker) entryAttrs(ep *EntryPoint) {
	d := ep.Func.Decl
	wg := c.findAttr(d.Attrs, "workgroup_size")
	if ep.Stage != StageCompute {
		if wg != nil {
			c.err(wg, "attr-placement", "@workgroup_size is only valid on compute entry points")
		}
		if a := c.findAttr(d.Attrs, "subgroup_size"); a != nil {
			c.err(a, "attr-placement", "@subgroup_size is only valid on compute entry points")
		}
		return
	}
	if wg == nil {
		c.err(d, "workgroup-size", "compute entry point %q needs @workgroup_size", ep.Name)
		return
	}
	vals := make([]*val, len(wg.Args))
	for i, a := range wg.Args {
		vals[i] = c.load(c.expr(a))
	}
	if anyInvalid(vals) {
		return
	}
	var T Type
	for _, v := range vals {
		if v.t == I32 || v.t == U32 {
			if T != nil && T != v.t {
				c.err(wg, "workgroup-size", "@workgroup_size arguments must have the same type")
				return
			}
			T = v.t
		}
	}
	if T == nil {
		T = I32
	}
	for i, v := range vals {
		if v.k == kRuntime {
			c.err(wg.Args[i], "workgroup-size", "@workgroup_size requires const- or override-expressions")
			continue
		}
		if !isIntegerScalar(v.t) {
			c.err(wg.Args[i], "workgroup-size", "@workgroup_size requires i32 or u32, found %s", v.t)
			continue
		}
		v = c.convertTo(v, T)
		if isInvalid(v.t) {
			continue
		}
		if v.k == kConst && v.c != nil {
			if v.c.Int < 1 {
				c.err(wg.Args[i], "workgroup-size", "workgroup size must be positive")
			}
			ep.WorkgroupSize[i] = v.c.Int
		} else {
			ep.WorkgroupSize[i] = 0
			ep.WorkgroupSizeExprs[i] = wg.Args[i]
		}
	}
	if ep.WorkgroupSizeExprs[0] == nil && ep.WorkgroupSizeExprs[1] == nil && ep.WorkgroupSizeExprs[2] == nil {
		if p := ep.WorkgroupSize[0] * ep.WorkgroupSize[1] * ep.WorkgroupSize[2]; p > 1<<31 {
			c.err(wg, "workgroup-size", "workgroup size is too large")
		}
	}
	if sa := c.findAttr(d.Attrs, "subgroup_size"); sa != nil {
		v := c.load(c.expr(sa.Args[0]))
		if !isInvalid(v.t) {
			if !isIntegerScalar(v.t) {
				c.err(sa, "attr-value", "@subgroup_size requires i32 or u32")
			} else if v.k == kRuntime {
				c.err(sa, "attr-value", "@subgroup_size requires a const- or override-expression")
			} else if v.k == kConst && v.c != nil && (v.c.Int <= 0 || v.c.Int&(v.c.Int-1) != 0) {
				c.err(sa, "attr-value", "@subgroup_size must be a power of two")
			}
		}
	}
}

// entryIO validates parameters and return value as shader stage inputs and
// outputs (spec "Entry Points").
func (c *checker) entryIO(ep *EntryPoint) {
	d := ep.Func.Decl
	f := ep.Func
	builtins := map[string]bool{}
	locs := map[int64]bool{}
	addIO := func(io IOVar, n Node, input bool, attrNode Node) {
		t := io.Type
		if isInvalid(t) {
			return
		}
		if io.Builtin != "" {
			dir := "output"
			if input {
				dir = "input"
			}
			key := dir + ":" + io.Builtin
			if builtins[key] {
				c.err(n, "builtin-duplicate", "built-in %s used twice as %s", io.Builtin, dir)
			}
			builtins[key] = true
			ok := false
			for _, sp := range builtinSpecs(io.Builtin) {
				if sp.stage == ep.Stage && sp.in == input {
					if sp.ext != "" && !c.enabled[sp.ext] {
						c.err(n, "extension-required", "@builtin(%s) requires `enable %s;`", io.Builtin, sp.ext)
					}
					if sp.lang != "" && !c.lang[sp.lang] {
						c.err(n, "extension-required", "@builtin(%s) requires the %s language extension", io.Builtin, sp.lang)
					}
					if !sp.typ(t) {
						c.err(n, "builtin-type", "@builtin(%s) has type %s, which is invalid", io.Builtin, t)
					}
					ok = true
				}
			}
			if !ok && builtinValues[io.Builtin] {
				c.err(n, "builtin-stage", "@builtin(%s) is not a %s %s of the %s stage", io.Builtin, "valid", dir, ep.Stage)
			}
		} else if io.Location != nil {
			sc := scalarOf(t)
			if sc == nil || sc.Kind == KBool || IsAbstract(sc) {
				c.err(n, "location-type", "@location requires a numeric scalar or vector, found %s", t)
			}
			if ep.Stage == StageCompute {
				c.err(n, "location-stage", "compute shaders must not have user-defined IO")
			}
			if input {
				if locs[*io.Location] {
					c.err(n, "location-duplicate", "input location %d is used twice", *io.Location)
				}
				locs[*io.Location] = true
			}
			if sc != nil && isIntegerScalar(sc) && ((ep.Stage == StageVertex && !input) || (ep.Stage == StageFragment && input)) && io.Interpolate[0] != "flat" {
				c.err(n, "interpolate", "integer user-defined IO must use @interpolate(flat)")
			}
			if io.BlendSrc != nil && (input || ep.Stage != StageFragment) {
				c.err(n, "blend-src", "@blend_src is only valid on fragment outputs")
			}
		} else {
			c.err(n, "io-missing", "entry point IO needs @builtin or @location")
		}
		io.Param = -1
		if input {
			ep.Inputs = append(ep.Inputs, io)
		} else {
			ep.Outputs = append(ep.Outputs, io)
		}
	}
	handle := func(name string, t Type, attrs []*Attr, n Node, input bool, idx int) {
		if isInvalid(t) {
			return
		}
		var io IOVar
		io.Name, io.Type, io.Param = name, t, idx
		c.ioAttrs(attrs, &io.Builtin, &io.Location, &io.BlendSrc, &io.Interpolate, &io.Invariant)
		hasIO := io.Builtin != "" || io.Location != nil
		if st, ok := t.(*StructType); ok && !st.Predeclared {
			if hasIO {
				c.err(n, "io-struct", "IO attributes cannot be applied to a structure type")
				return
			}
			for _, m := range st.Members {
				if _, nested := m.Type.(*StructType); nested {
					c.err(n, "io-struct", "structures used for IO cannot be nested")
					continue
				}
				mio := IOVar{Name: m.Name, Type: m.Type, Builtin: m.BuiltinName, Location: m.Location,
					BlendSrc: m.BlendSrc, Interpolate: m.Interpolate, Invariant: m.Invariant, Member: m.Name}
				addIO(mio, n, input, n)
				if input && len(ep.Inputs) > 0 {
					ep.Inputs[len(ep.Inputs)-1].Param = idx
				}
			}
			return
		}
		addIO(io, n, input, n)
		if input && len(ep.Inputs) > 0 {
			ep.Inputs[len(ep.Inputs)-1].Param = idx
		}
	}
	for i, p := range f.Params {
		handle(p.Name, p.Type, d.Params[i].Attrs, d.Params[i], true, i)
	}
	if d.Ret != nil && !isInvalid(f.Ret) {
		if ep.Stage == StageCompute {
			c.err(d.Ret, "entry-point", "compute entry points cannot return a value")
		} else {
			handle("", f.Ret, d.RetAttrs, d.Ret, false, -1)
		}
	}
	if ep.Stage == StageVertex {
		found := false
		for _, o := range ep.Outputs {
			if o.Builtin == "position" {
				found = true
			}
		}
		if !found {
			c.err(d, "entry-point", "a vertex shader must return @builtin(position)")
		}
	}
}

// entryAccess computes the functions in the stage and the statically
// accessed declarations, and applies the stage rules.
func (c *checker) entryAccess(ep *EntryPoint) {
	seen := map[*FuncObj]bool{}
	var vars []*VarObj
	vseen := map[*VarObj]bool{}
	oseen := map[*ValueObj]bool{}
	var visit func(f *FuncObj)
	visit = func(f *FuncObj) {
		if seen[f] {
			return
		}
		seen[f] = true
		ep.Functions = append(ep.Functions, f)
		var keys []Object
		for o := range f.uses {
			keys = append(keys, o)
		}
		sort.Slice(keys, func(i, j int) bool { return objStart(keys[i]) < objStart(keys[j]) })
		for _, o := range keys {
			switch o := o.(type) {
			case *FuncObj:
				visit(o)
			case *VarObj:
				if !vseen[o] {
					vseen[o] = true
					vars = append(vars, o)
				}
			case *ValueObj:
				if o.Kind == "override" && !oseen[o] {
					oseen[o] = true
					ep.Overrides = append(ep.Overrides, o)
				}
			}
		}
	}
	visit(ep.Func)
	sort.Slice(ep.Functions, func(i, j int) bool { return ep.Functions[i].Decl.Start < ep.Functions[j].Decl.Start })
	immediate := 0
	bind := map[[2]int64]*VarObj{}
	for _, v := range vars {
		switch v.AS {
		case ASUniform, ASStorage, ASHandle:
			ep.Resources = append(ep.Resources, v)
			if v.Group != nil && v.Binding != nil {
				k := [2]int64{*v.Group, *v.Binding}
				if o, dup := bind[k]; dup {
					c.err(v.Decl, "binding-duplicate", "%q and %q share @group(%d) @binding(%d)", o.Name, v.Name, k[0], k[1])
				}
				bind[k] = v
			}
			if ep.Stage == StageVertex {
				if v.AS == ASStorage && v.Access != AccessRead {
					c.err(ep.Func.Decl, "stage-access", "vertex shaders cannot access read_write storage buffer %q", v.Name)
				}
				if tt, ok := v.Store.(*TextureType); ok && tt.Kind == TexStorage && tt.Access != AccessRead {
					c.err(ep.Func.Decl, "stage-access", "vertex shaders cannot access writable storage texture %q", v.Name)
				}
			}
		case ASImmediate:
			immediate++
			ep.Globals = append(ep.Globals, v)
		default:
			ep.Globals = append(ep.Globals, v)
			if v.AS == ASWorkgroup && ep.Stage != StageCompute {
				c.err(ep.Func.Decl, "stage-access", "workgroup variable %q is only accessible from compute shaders", v.Name)
			}
		}
	}
	if immediate > 1 {
		c.err(ep.Func.Decl, "immediate", "an entry point can access at most one immediate variable")
	}
	sort.SliceStable(ep.Resources, func(i, j int) bool {
		a, b := ep.Resources[i], ep.Resources[j]
		if a.Group == nil || b.Group == nil || a.Binding == nil || b.Binding == nil {
			return false
		}
		if *a.Group != *b.Group {
			return *a.Group < *b.Group
		}
		return *a.Binding < *b.Binding
	})
	// stage restricted built-ins
	for _, f := range ep.Functions {
		var keys []string
		for k := range f.feats {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			parts := strings.Split(k, ":")
			what, restr := parts[1], parts[2]
			ok := false
			for _, r := range strings.Split(restr, "|") {
				switch r {
				case "fragment":
					ok = ok || ep.Stage == StageFragment
				case "compute":
					ok = ok || ep.Stage == StageCompute
				case "notvertex":
					ok = ok || ep.Stage != StageVertex
				}
			}
			if !ok {
				c.errAt(f.feats[k].Off, "stage-builtin", "%s cannot be used in a %s shader (reached from entry point %q)", what, ep.Stage, ep.Name)
			}
		}
	}
}

func objStart(o Object) int {
	switch o := o.(type) {
	case *FuncObj:
		return o.Decl.Start
	case *VarObj:
		return o.Decl.Start
	case *ValueObj:
		return o.Decl.Start
	}
	return 0
}
