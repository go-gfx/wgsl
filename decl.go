package wgsl

import (
	"math"
	"sort"
	"strings"
)

type exprKind uint8

const (
	kRuntime exprKind = iota
	kOverride
	kConst
)

// val is the result of checking an expression.
type val struct {
	t    Type
	k    exprKind
	c    *Const
	expr Expr
	// comp marks a reference to a vector component.
	comp bool
}

type checker struct {
	m        *Module
	opt      CheckOptions
	info     *Info
	errs     ErrorList
	maxErrs  int
	suppress int
	module   map[string]Object
	order    []Object
	scopes   []map[string]Object
	cur      *FuncObj
	curDecl  Object
	enabled  map[string]bool
	lang     map[string]bool
	steps    int
	maxSteps int
	depth    int
	uses     map[Object]map[Object]bool
	// statement context
	loopDepth, switchDepth int
	inContinuing           int
	retType                Type
	targets                []byte
	declared               map[Object]bool
}

const maxDepth = 400
const maxNest = 63

func newChecker(m *Module, o CheckOptions) *checker {
	c := &checker{m: m, opt: o, maxErrs: o.MaxErrors, maxSteps: o.MaxConstSteps,
		module: map[string]Object{}, enabled: map[string]bool{}, uses: map[Object]map[Object]bool{},
		declared: map[Object]bool{}}
	if c.maxErrs <= 0 {
		c.maxErrs = 100
	}
	if c.maxSteps <= 0 {
		c.maxSteps = 1 << 20
	}
	c.lang = o.LanguageFeatures
	if c.lang == nil {
		c.lang = DefaultLanguageFeatures()
	}
	c.info = &Info{Types: map[Expr]Type{}, Objects: map[*Ident]Object{}, TypeSpecs: map[*Ident]Type{},
		Conversions: map[Expr]Type{}, Consts: map[Expr]*Const{}, Decls: map[Node]Object{},
		DeclTypes: map[Node]Type{}, Enabled: c.enabled}
	return c
}

func (c *checker) run() {
	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(checkLimit); !ok {
				panic(r)
			}
		}
	}()
	c.directives()
	c.collect()
	for _, o := range c.order {
		c.ensure(o)
	}
	for _, d := range c.m.Decls {
		if a, ok := d.(*ConstAssert); ok {
			c.constAssert(a)
		}
	}
	for _, f := range c.info.Funcs {
		c.checkBody(f)
	}
	c.cycles()
	c.entryPoints()
}

func (c *checker) extEnabled(name string) bool { return c.enabled[name] }

func (c *checker) directives() {
	var seenDiag = map[string]string{}
	for _, d := range c.m.Directives {
		switch d.Kind {
		case "enable":
			for _, n := range d.Names {
				ok := false
				for _, e := range AllEnableExtensions {
					if e == n && (c.opt.Extensions == nil || c.opt.Extensions[n]) {
						ok = true
					}
				}
				if !ok {
					c.err(d, "unknown-extension", "unsupported enable-extension %q", n)
					continue
				}
				c.enabled[n] = true
				if n == "subgroup_size_control" {
					c.enabled["subgroups"] = true
				}
			}
		case "requires":
			for _, n := range d.Names {
				if n == "uniform_buffer_standard_layout" && !c.lang[n] {
					// not on by default; a module that requires it opts in
					l := map[string]bool{}
					for k, v := range c.lang {
						l[k] = v
					}
					l[n] = true
					c.lang = l
				}
				if !c.lang[n] {
					c.err(d, "unknown-extension", "unsupported language extension %q", n)
				}
			}
		case "diagnostic":
			if len(d.Names) >= 2 {
				switch d.Names[0] {
				case "error", "warning", "info", "off":
				default:
					c.err(d, "diagnostic", "unknown diagnostic severity %q", d.Names[0])
				}
				if old, ok := seenDiag[d.Names[1]]; ok && old != d.Names[0] {
					c.err(d, "diagnostic-conflict", "conflicting diagnostic filters for %q", d.Names[1])
				}
				seenDiag[d.Names[1]] = d.Names[0]
			}
		}
	}
}

// ---- scopes ----

func (c *checker) push() { c.scopes = append(c.scopes, map[string]Object{}) }
func (c *checker) pop()  { c.scopes = c.scopes[:len(c.scopes)-1] }

func (c *checker) declareLocal(name string, o Object, n Node) {
	s := c.scopes[len(c.scopes)-1]
	if _, dup := s[name]; dup {
		c.err(n, "redeclaration", "%q is already declared in this scope", name)
		return
	}
	s[name] = o
}

// lookup resolves name: nearest local declaration, then module scope, then
// predeclared objects.
func (c *checker) lookup(name string) Object {
	for i := len(c.scopes) - 1; i >= 0; i-- {
		if o, ok := c.scopes[i][name]; ok {
			return o
		}
	}
	if o, ok := c.module[name]; ok {
		return o
	}
	if predeclared(name) {
		return &BuiltinObj{Name: name}
	}
	return nil
}

func (c *checker) noteUse(o Object) {
	if c.curDecl == nil {
		return
	}
	switch o.(type) {
	case *VarObj, *ValueObj, *FuncObj, *TypeObj:
		if v, ok := o.(*VarObj); ok && !v.Global {
			return
		}
		if v, ok := o.(*ValueObj); ok && v.Local {
			return
		}
		m := c.uses[c.curDecl]
		if m == nil {
			m = map[Object]bool{}
			c.uses[c.curDecl] = m
		}
		m[o] = true
	}
}

// ---- collection ----

func (c *checker) addModule(name string, o Object, n Node) {
	if name == "" {
		return
	}
	if _, dup := c.module[name]; dup {
		c.err(n, "redeclaration", "%q is already declared at module scope", name)
		return
	}
	c.checkIdentName(name, n)
	c.module[name] = o
	c.order = append(c.order, o)
}

func (c *checker) checkIdentName(name string, n Node) {
	if strings.HasPrefix(name, "__") {
		c.err(n, "identifier", "identifier %q must not start with two underscores", name)
	}
}

func (c *checker) collect() {
	for _, d := range c.m.Decls {
		switch d := d.(type) {
		case *Func:
			f := &FuncObj{Name: d.Name, Decl: d, uses: map[Object]bool{}}
			c.info.Decls[d] = f
			c.addModule(d.Name, f, d)
			c.info.Funcs = append(c.info.Funcs, f)
		case *Var:
			v := &VarObj{Name: d.Name, Decl: d, Global: true}
			c.info.Decls[d] = v
			c.addModule(d.Name, v, d)
			c.info.Globals = append(c.info.Globals, v)
		case *Value:
			v := &ValueObj{Name: d.Name, Kind: d.Kind, Decl: d, ID: -1}
			c.info.Decls[d] = v
			c.addModule(d.Name, v, d)
			if d.Kind == "override" {
				c.info.Overrides = append(c.info.Overrides, v)
			}
		case *Alias:
			t := &TypeObj{Name: d.Name, Alias: d}
			c.info.Decls[d] = t
			c.addModule(d.Name, t, d)
		case *Struct:
			t := &TypeObj{Name: d.Name, Struct: d}
			c.info.Decls[d] = t
			c.addModule(d.Name, t, d)
		}
	}
}

// ensure resolves the signature/type/value of a module-scope declaration.
func (c *checker) ensure(o Object) {
	saveDecl, saveScopes, saveCur := c.curDecl, c.scopes, c.cur
	saveLoop, saveSw, saveCont := c.loopDepth, c.switchDepth, c.inContinuing
	defer func() {
		c.curDecl, c.scopes, c.cur = saveDecl, saveScopes, saveCur
		c.loopDepth, c.switchDepth, c.inContinuing = saveLoop, saveSw, saveCont
	}()
	c.scopes = nil
	c.cur = nil
	c.depth++
	defer func() { c.depth-- }()
	if c.depth > maxDepth {
		switch o := o.(type) {
		case *TypeObj:
			if o.Alias != nil {
				c.err(o.Alias, "limit-depth", "declaration nesting is too deep")
			} else {
				c.err(o.Struct, "limit-depth", "declaration nesting is too deep")
			}
			o.Type = Invalid
			o.state = 2
		case *ValueObj:
			c.err(o.Decl, "limit-depth", "declaration nesting is too deep")
			o.Type, o.state = Invalid, 2
		}
		return
	}
	switch o := o.(type) {
	case *TypeObj:
		if o.state != 0 {
			return
		}
		o.state = 1
		c.curDecl = o
		if o.Alias != nil {
			o.Type = c.typeSpec(o.Alias.Type)
			c.info.DeclTypes[o.Alias] = o.Type
		} else {
			c.structDecl(o)
		}
		o.state = 2
	case *ValueObj:
		if o.state != 0 {
			return
		}
		o.state = 1
		c.curDecl = o
		c.valueDecl(o)
		o.state = 2
	case *VarObj:
		if o.state != 0 {
			return
		}
		o.state = 1
		c.curDecl = o
		c.globalVar(o)
		o.state = 2
	case *FuncObj:
		if o.state != 0 {
			return
		}
		o.state = 1
		c.curDecl = o
		c.funcSignature(o)
		o.state = 2
	}
}

// resolved returns whether o is fully resolved, reporting a cycle when it is
// being resolved.
func (c *checker) use(o Object, at Node) bool {
	c.noteUse(o)
	switch o := o.(type) {
	case *TypeObj:
		if o.state == 1 {
			c.err(at, "recursive-declaration", "%q is recursive", o.Name)
			return false
		}
		c.ensure(o)
	case *ValueObj:
		if o.Local {
			return true
		}
		if o.state == 1 {
			c.err(at, "recursive-declaration", "%q is recursive", o.Name)
			return false
		}
		c.ensure(o)
	case *VarObj:
		if !o.Global {
			return true
		}
		if o.state == 1 {
			c.err(at, "recursive-declaration", "%q is recursive", o.Name)
			return false
		}
		c.ensure(o)
	case *FuncObj:
		c.ensure(o)
	}
	return true
}

// cycles reports cycles in the declaration graph (function recursion, and
// any cycle not already caught while resolving).
func (c *checker) cycles() {
	state := map[Object]int{}
	reported := false
	var visit func(o Object)
	visit = func(o Object) {
		if state[o] == 2 || reported {
			return
		}
		if state[o] == 1 {
			reported = true
			var n Node
			switch o := o.(type) {
			case *FuncObj:
				n = o.Decl
			case *VarObj:
				n = o.Decl
			case *ValueObj:
				n = o.Decl
			case *TypeObj:
				if o.Alias != nil {
					n = o.Alias
				} else {
					n = o.Struct
				}
			}
			if n != nil {
				c.err(n, "recursive-declaration", "%q is recursive", o.ObjName())
			}
			return
		}
		state[o] = 1
		for _, d := range sortedObjs(c.uses[o], c) {
			visit(d)
		}
		state[o] = 2
	}
	for _, o := range c.order {
		visit(o)
	}
}

func sortedObjs(m map[Object]bool, c *checker) []Object {
	var out []Object
	for o := range m {
		out = append(out, o)
	}
	idx := map[Object]int{}
	for i, o := range c.order {
		idx[o] = i
	}
	sort.Slice(out, func(i, j int) bool { return idx[out[i]] < idx[out[j]] })
	return out
}

// ---- struct declarations and layout ----

func roundUp(a, n int64) int64 {
	if a <= 0 {
		return n
	}
	r := (n + a - 1) / a * a
	if r < n || r > maxSize {
		return maxSize
	}
	return r
}

const maxSize = int64(1) << 50

func satAdd(a, b int64) int64 {
	if a+b > maxSize || a+b < 0 {
		return maxSize
	}
	return a + b
}

func satMul(a, b int64) int64 {
	if a == 0 || b == 0 {
		return 0
	}
	if a > maxSize/b {
		return maxSize
	}
	return a * b
}

// alignSize returns AlignOf and SizeOf (spec "Alignment and Size").
func alignSize(t Type) (align, size int64) {
	switch t := t.(type) {
	case *ScalarType:
		if t.Kind == KF16 {
			return 2, 2
		}
		return 4, 4
	case *AtomicType:
		return 4, 4
	case *VecType:
		el := int64(4)
		if t.Elem.Kind == KF16 {
			el = 2
		}
		switch t.N {
		case 2:
			return 2 * el, 2 * el
		case 3:
			return 4 * el, 3 * el
		}
		return 4 * el, 4 * el
	case *MatType:
		a, s := alignSize(vec(t.R, t.Elem))
		return a, int64(t.C) * roundUp(a, s)
	case *ArrayType:
		a, s := alignSize(t.Elem)
		if t.Runtime {
			return a, 0
		}
		return a, satMul(t.Count, roundUp(a, s))
	case *StructType:
		return t.Align, t.Size
	}
	return 1, 0
}

func (c *checker) structDecl(o *TypeObj) {
	d := o.Struct
	st := &StructType{Name: d.Name, Decl: d}
	o.Type = st
	c.info.DeclTypes[d] = st
	if len(d.Members) == 0 {
		c.err(d, "struct-empty", "structure %q must have at least one member", d.Name)
	}
	if len(d.Members) > 1023 {
		c.err(d, "limit-members", "structure %q has more than 1023 members", d.Name)
		return
	}
	seen := map[string]bool{}
	for i, md := range d.Members {
		m := &FieldInfo{Name: md.Name, Decl: md}
		if seen[md.Name] {
			c.err(md, "redeclaration", "duplicate member %q", md.Name)
		}
		seen[md.Name] = true
		c.checkIdentName(md.Name, md)
		m.Type = c.typeSpec(md.Type)
		t := m.Type
		switch tt := t.(type) {
		case *ArrayType:
			if tt.Runtime {
				if i != len(d.Members)-1 {
					c.err(md, "struct-member-type", "a runtime-sized array must be the last member")
				} else {
					st.Runtime = true
				}
			} else if !creationFixed(tt) {
				c.err(md, "struct-member-type", "array member must have a creation-fixed footprint")
			}
		case *StructType:
			if !creationFixed(tt) && !isInvalid(t) {
				c.err(md, "struct-member-type", "structure member must have a creation-fixed footprint")
			}
		case *ScalarType, *VecType, *MatType, *AtomicType:
			if IsAbstract(t) {
				c.err(md, "struct-member-type", "abstract type as member")
			}
		default:
			if !isInvalid(t) {
				c.err(md, "struct-member-type", "type %s cannot be a structure member", t)
			}
		}
		if NestDepth(t) > maxNest {
			c.err(md, "limit-depth", "composite type nesting is too deep")
			m.Type = Invalid
		}
		st.Members = append(st.Members, m)
	}
	c.layout(st)
	c.info.Structs = append(c.info.Structs, st)
	c.structAttrs(st)
}

func (c *checker) layout(st *StructType) {
	var off, maxAlign int64 = 0, 1
	for _, m := range st.Members {
		a, s := alignSize(m.Type)
		if isInvalid(m.Type) {
			a, s = 1, 0
		}
		m.Size, m.Align = s, a
		if m.Decl != nil {
			for _, at := range m.Decl.Attrs {
				switch at.Name {
				case "size":
					if v, ok := c.attrInt(at, 0); ok {
						if v < 1 {
							c.err(at, "attr-value", "@size must be positive")
						} else if v < s {
							c.err(at, "attr-value", "@size(%d) is smaller than the size %d of the member type", v, s)
						} else {
							m.Size = v
						}
					}
					if !creationFixed(m.Type) && !isInvalid(m.Type) {
						c.err(at, "attr-placement", "@size requires a type with a creation-fixed footprint")
					}
				case "align":
					if v, ok := c.attrInt(at, 0); ok {
						if v < 1 || v&(v-1) != 0 {
							c.err(at, "attr-value", "@align must be a positive power of two")
						} else {
							m.Align = v
						}
					}
				}
			}
		}
		off = roundUp(m.Align, off)
		m.Offset = off
		off = satAdd(off, m.Size)
		if m.Align > maxAlign {
			maxAlign = m.Align
		}
	}
	st.Align = maxAlign
	st.Size = roundUp(maxAlign, off)
}

// ---- type specifiers ----

var enumKinds = map[string]string{
	"function": "address_space", "private": "address_space", "workgroup": "address_space",
	"uniform": "address_space", "storage": "address_space", "immediate": "address_space",
	"read": "access_mode", "write": "access_mode", "read_write": "access_mode",
}

var texelFormats = []string{"rgba8unorm", "rgba8snorm", "rgba8uint", "rgba8sint", "rgba16unorm", "rgba16snorm",
	"rgba16uint", "rgba16sint", "rgba16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "rg16unorm",
	"rg16snorm", "rg16uint", "rg16sint", "rg16float", "r32uint", "r32sint", "r32float", "rg32uint", "rg32sint",
	"rg32float", "rgba32uint", "rgba32sint", "rgba32float", "bgra8unorm", "r8unorm", "r8snorm", "r8uint",
	"r8sint", "r16unorm", "r16snorm", "r16uint", "r16sint", "r16float", "rgb10a2unorm", "rgb10a2uint",
	"rg11b10ufloat"}

var tier1Formats = map[string]bool{"rgba16unorm": true, "rgba16snorm": true, "rg8unorm": true, "rg8snorm": true,
	"rg8uint": true, "rg8sint": true, "rg16unorm": true, "rg16snorm": true, "rg16uint": true, "rg16sint": true,
	"rg16float": true, "r8unorm": true, "r8snorm": true, "r8uint": true, "r8sint": true, "r16unorm": true,
	"r16snorm": true, "r16uint": true, "r16sint": true, "r16float": true, "rgb10a2unorm": true,
	"rgb10a2uint": true, "rg11b10ufloat": true}

var formatSet = func() map[string]bool {
	m := map[string]bool{}
	for _, f := range texelFormats {
		m[f] = true
	}
	return m
}()

var scalarByName = map[string]*ScalarType{"bool": Bool, "i32": I32, "u32": U32, "f32": F32, "f16": F16}

var texNames = map[string]*TextureType{
	"texture_1d": {Kind: TexSampled, Dim: Dim1D}, "texture_2d": {Kind: TexSampled, Dim: Dim2D},
	"texture_2d_array": {Kind: TexSampled, Dim: Dim2D, Arrayed: true}, "texture_3d": {Kind: TexSampled, Dim: Dim3D},
	"texture_cube": {Kind: TexSampled, Dim: DimCube}, "texture_cube_array": {Kind: TexSampled, Dim: DimCube, Arrayed: true},
	"texture_multisampled_2d":       {Kind: TexMultisampled, Dim: Dim2D},
	"texture_storage_1d":            {Kind: TexStorage, Dim: Dim1D},
	"texture_storage_2d":            {Kind: TexStorage, Dim: Dim2D},
	"texture_storage_2d_array":      {Kind: TexStorage, Dim: Dim2D, Arrayed: true},
	"texture_storage_3d":            {Kind: TexStorage, Dim: Dim3D},
	"texture_depth_2d":              {Kind: TexDepth, Dim: Dim2D},
	"texture_depth_2d_array":        {Kind: TexDepth, Dim: Dim2D, Arrayed: true},
	"texture_depth_cube":            {Kind: TexDepth, Dim: DimCube},
	"texture_depth_cube_array":      {Kind: TexDepth, Dim: DimCube, Arrayed: true},
	"texture_depth_multisampled_2d": {Kind: TexDepthMultisampled, Dim: Dim2D},
	"texture_external":              {Kind: TexExternal, Dim: Dim2D},
}

// predeclared reports whether name is a predeclared object.
func predeclared(name string) bool {
	if _, ok := scalarByName[name]; ok {
		return true
	}
	if _, ok := enumKinds[name]; ok {
		return true
	}
	if formatSet[name] || name == "array" || name == "bitcast" || name == "atomic" || name == "ptr" || name == "sampler" ||
		name == "sampler_comparison" || texNames[name] != nil {
		return true
	}
	if _, _, _, ok := vecMatName(name); ok {
		return true
	}
	return builtinFuncs[name] != nil
}

// vecMatName parses vec2/vec3f/mat2x3h and friends. kind is 'v' or 'm'; elem
// is the alias element type or nil for the generic form.
func vecMatName(name string) (kind byte, dims [2]int, elem *ScalarType, ok bool) {
	suffix := func(s string) (*ScalarType, bool) {
		switch s {
		case "":
			return nil, true
		case "i":
			return I32, true
		case "u":
			return U32, true
		case "f":
			return F32, true
		case "h":
			return F16, true
		}
		return nil, false
	}
	if len(name) >= 4 && name[:3] == "vec" && name[3] >= '2' && name[3] <= '4' {
		e, ok := suffix(name[4:])
		return 'v', [2]int{int(name[3] - '0'), 0}, e, ok
	}
	if len(name) >= 6 && name[:3] == "mat" && name[3] >= '2' && name[3] <= '4' && name[4] == 'x' && name[5] >= '2' && name[5] <= '4' {
		e, ok := suffix(name[6:])
		if e == I32 || e == U32 {
			ok = false
		}
		return 'm', [2]int{int(name[3] - '0'), int(name[5] - '0')}, e, ok
	}
	return 0, dims, nil, false
}

func (c *checker) f16Check(t Type, n Node) Type {
	if s, ok := scalarOf(t), true; ok && s != nil && s.Kind == KF16 && !c.enabled["f16"] {
		c.err(n, "extension-required", "f16 requires `enable f16;`")
	}
	return t
}

// typeSpec resolves a type specifier.
func (c *checker) typeSpec(id *Ident) Type {
	t := c.typeSpec0(id)
	c.info.TypeSpecs[id] = t
	return t
}

func (c *checker) typeSpec0(id *Ident) Type {
	o := c.lookup(id.Name)
	if o == nil {
		c.err(id, "undeclared-ident", "unknown type %q", id.Name)
		return Invalid
	}
	c.info.Objects[id] = o
	if to, ok := o.(*TypeObj); ok {
		if len(id.Targs) > 0 {
			c.err(id, "type-args", "%q does not take template arguments", id.Name)
		}
		if !c.use(to, id) {
			return Invalid
		}
		if to.Type == nil {
			return Invalid
		}
		return to.Type
	}
	if _, ok := o.(*BuiltinObj); !ok {
		c.err(id, "not-a-type", "%q is not a type", id.Name)
		return Invalid
	}
	return c.builtinType(id)
}

// typeArg resolves a template argument that must be a type.
func (c *checker) typeArg(e Expr) Type {
	switch e := e.(type) {
	case *Ident:
		return c.typeSpec(e)
	case *Paren:
		return c.typeArg(e.X)
	}
	c.err(e, "type-args", "expected a type")
	return Invalid
}

func (c *checker) noTargs(id *Ident, n int) bool {
	if len(id.Targs) != n {
		c.err(id, "type-args", "%s takes %d template argument(s), found %d", id.Name, n, len(id.Targs))
		return false
	}
	return true
}

// enumArg resolves a template argument naming an enumerant of the given
// enumeration.
func (c *checker) enumArg(e Expr, kind string) string {
	id, ok := e.(*Ident)
	if !ok || len(id.Targs) > 0 {
		c.err(e, "type-args", "expected a %s", kind)
		return ""
	}
	o := c.lookup(id.Name)
	if _, isB := o.(*BuiltinObj); !isB {
		c.err(e, "type-args", "%q is not a %s", id.Name, kind)
		return ""
	}
	c.info.Objects[id] = o
	k := enumKinds[id.Name]
	if formatSet[id.Name] {
		k = "texel_format"
	}
	if k != kind {
		c.err(e, "type-args", "%q is not a %s", id.Name, kind)
		return ""
	}
	if kind == "texel_format" && tier1Formats[id.Name] && !c.lang["texture_formats_tier1"] {
		c.err(e, "extension-required", "texel format %s requires texture_formats_tier1", id.Name)
	}
	if id.Name == "immediate" && !c.lang["immediate_address_space"] {
		c.err(e, "extension-required", "address space immediate requires immediate_address_space")
	}
	c.info.Types[e] = &EnumType{Enum: kind}
	return id.Name
}

func asOf(name string) AddressSpace {
	switch name {
	case "function":
		return ASFunction
	case "private":
		return ASPrivate
	case "workgroup":
		return ASWorkgroup
	case "uniform":
		return ASUniform
	case "storage":
		return ASStorage
	case "immediate":
		return ASImmediate
	}
	return ASNone
}

func accessOf(name string) Access {
	switch name {
	case "read":
		return AccessRead
	case "write":
		return AccessWrite
	case "read_write":
		return AccessReadWrite
	}
	return AccessNone
}

func defaultAccess(as AddressSpace) Access {
	switch as {
	case ASFunction, ASPrivate, ASWorkgroup:
		return AccessReadWrite
	}
	return AccessRead
}

func (c *checker) builtinType(id *Ident) Type {
	name := id.Name
	if s, ok := scalarByName[name]; ok {
		if len(id.Targs) > 0 {
			c.err(id, "type-args", "%s does not take template arguments", name)
		}
		return c.f16Check(s, id)
	}
	if kind, d, e, ok := vecMatName(name); ok {
		var el *ScalarType
		if e != nil {
			if len(id.Targs) > 0 {
				c.err(id, "type-args", "%s does not take template arguments", name)
			}
			el = e
			c.f16Check(el, id)
		} else {
			if !c.noTargs(id, 1) {
				return Invalid
			}
			t := c.typeArg(id.Targs[0])
			if isInvalid(t) {
				return Invalid
			}
			s, ok := t.(*ScalarType)
			switch {
			case !ok || IsAbstract(s):
				c.err(id, "type-args", "invalid component type %s", t)
				return Invalid
			case kind == 'm' && (s.Kind == KBool || s.Kind == KI32 || s.Kind == KU32):
				c.err(id, "type-args", "matrix component type must be f32 or f16")
				return Invalid
			}
			el = s
		}
		if kind == 'v' {
			return &VecType{d[0], el}
		}
		return &MatType{d[0], d[1], el}
	}
	switch name {
	case "array":
		return c.arrayType(id)
	case "atomic":
		if !c.noTargs(id, 1) {
			return Invalid
		}
		t := c.typeArg(id.Targs[0])
		switch tt := t.(type) {
		case *ScalarType:
			if tt == I32 || tt == U32 {
				return &AtomicType{t}
			}
		case *VecType:
			if tt.N == 2 && tt.Elem == U32 && (c.enabled["atomic_vec2u_min_max"] || c.lang["atomic_vec2u_min_max"]) {
				return &AtomicType{t}
			}
		case *InvalidType:
			return Invalid
		}
		c.err(id, "type-args", "atomic<T> requires T to be i32 or u32")
		return Invalid
	case "ptr":
		if len(id.Targs) < 2 || len(id.Targs) > 3 {
			c.err(id, "type-args", "ptr takes 2 or 3 template arguments")
			return Invalid
		}
		as := asOf(c.enumArg(id.Targs[0], "address_space"))
		t := c.typeArg(id.Targs[1])
		if as == ASNone || isInvalid(t) {
			return Invalid
		}
		acc := defaultAccess(as)
		if len(id.Targs) == 3 {
			a := accessOf(c.enumArg(id.Targs[2], "access_mode"))
			if as != ASStorage {
				c.err(id, "type-args", "access mode may only be written for the storage address space")
			} else if a == AccessWrite || a == AccessNone {
				c.err(id, "type-args", "invalid access mode for storage")
			} else {
				acc = a
			}
		}
		if !storable(t) {
			c.err(id, "type-args", "%s is not a storable type", t)
			return Invalid
		}
		return &PtrType{AS: as, Elem: t, Access: acc}
	case "sampler":
		return &SamplerType{}
	case "sampler_comparison":
		return &SamplerType{Comparison: true}
	}
	if tt, ok := texNames[name]; ok {
		t := *tt
		switch t.Kind {
		case TexSampled, TexMultisampled:
			if !c.noTargs(id, 1) {
				return Invalid
			}
			s := c.typeArg(id.Targs[0])
			if s != I32 && s != U32 && s != F32 {
				if !isInvalid(s) {
					c.err(id, "type-args", "sampled type must be f32, i32 or u32")
				}
				return Invalid
			}
			t.Sampled = s.(*ScalarType)
		case TexStorage:
			if !c.noTargs(id, 2) {
				return Invalid
			}
			t.Format = c.enumArg(id.Targs[0], "texel_format")
			t.Access = accessOf(c.enumArg(id.Targs[1], "access_mode"))
			if t.Format == "" || t.Access == AccessNone {
				return Invalid
			}
			if t.Access != AccessWrite && !c.lang["readonly_and_readwrite_storage_textures"] {
				c.err(id, "extension-required", "read and read_write storage textures require readonly_and_readwrite_storage_textures")
			}
		default:
			if len(id.Targs) > 0 {
				c.err(id, "type-args", "%s does not take template arguments", name)
			}
		}
		return &t
	}
	c.err(id, "not-a-type", "%q is not a type", name)
	return Invalid
}

func (c *checker) arrayType(id *Ident) Type {
	if len(id.Targs) < 1 || len(id.Targs) > 2 {
		c.err(id, "type-args", "array takes 1 or 2 template arguments")
		return Invalid
	}
	elem := c.typeArg(id.Targs[0])
	if isInvalid(elem) {
		return Invalid
	}
	switch e := elem.(type) {
	case *ScalarType, *VecType, *MatType, *AtomicType:
		if IsAbstract(e) {
			c.err(id, "array-element", "abstract array element type")
			return Invalid
		}
	case *ArrayType:
		if !creationFixed(e) {
			c.err(id, "array-element", "array element type must have a creation-fixed footprint")
			return Invalid
		}
	case *StructType:
		if !creationFixed(e) {
			c.err(id, "array-element", "structure element type must have a creation-fixed footprint")
			return Invalid
		}
	default:
		c.err(id, "array-element", "%s cannot be an array element type", elem)
		return Invalid
	}
	a := &ArrayType{Elem: elem}
	if len(id.Targs) == 1 {
		a.Runtime = true
	} else {
		cnt := id.Targs[1]
		v := c.expr(cnt)
		if isInvalid(v.t) {
			return Invalid
		}
		v = c.load(v)
		if !isConcreteIntScalar(v.t) && v.t != AbstractInt {
			c.err(cnt, "array-count", "array element count must be an integer scalar, found %s", v.t)
			return Invalid
		}
		if v.k == kRuntime {
			c.err(cnt, "array-count", "array element count must be an override-expression")
			return Invalid
		}
		c.convertTo(v, scalarOrI32(v.t))
		if v.k == kConst {
			n, ok := v.c.asInt()
			if v.c == nil {
				c.err(cnt, "const-eval-unsupported", "cannot evaluate the array element count")
				return Invalid
			}
			if !ok || n < 1 {
				c.err(cnt, "array-count", "array element count must be positive")
				return Invalid
			}
			a.Count = n
		} else {
			a.Override = cnt
			if o := c.overrideIdent(cnt); o != nil {
				a.OverrideDecl = o
			}
		}
	}
	if NestDepth(a) > maxNest {
		c.err(id, "limit-depth", "composite type nesting is too deep")
		return Invalid
	}
	return a
}

func scalarOrI32(t Type) *ScalarType {
	if t == AbstractInt {
		return I32
	}
	return t.(*ScalarType)
}

func (c *checker) overrideIdent(e Expr) *ValueObj {
	if id, ok := e.(*Ident); ok {
		if o, ok := c.info.Objects[id].(*ValueObj); ok && o.Kind == "override" {
			return o
		}
	}
	return nil
}

var _ int64 = math.MaxInt64
