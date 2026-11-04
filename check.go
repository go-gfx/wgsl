// Name resolution and type checking for WGSL modules.
//
// Check implements the shader-creation rules of the WGSL Candidate
// Recommendation Draft of 26 October 2026 (see SPEC.md for the sections
// read): declaration and scope, types and automatic conversions,
// expression and statement typing, built-in functions, address spaces and
// access modes, attributes and entry point interfaces, const- and
// override-expressions, const_assert, behavior analysis (return, break,
// continue), memory layout, and enable/requires directives.
//
// Not implemented: uniformity analysis (spec "Uniformity"; derivative and
// subgroup uniformity diagnostics are never reported), alias analysis of
// pointer arguments, the buffer_view and swizzle_assignment language
// extensions (rejected as unsupported), and the const evaluation of a few
// rarely used built-ins (their value is then unknown, never wrong).
package wgsl

import (
	"fmt"
	"sort"
)

// Object is what an identifier resolves to.
type Object interface{ ObjName() string }

// VarObj is a variable declaration (module scope or function scope).
type VarObj struct {
	Name    string
	Decl    *Var
	AS      AddressSpace
	Access  Access
	Store   Type
	Global  bool
	Group   *int64
	Binding *int64
	state   int
}

// ValueObj is a const, override or let declaration.
type ValueObj struct {
	Name string
	Kind string // "const", "override", "let"
	Decl *Value
	Type Type
	// Val is the value of a const declaration (nil when unknown).
	Val *Const
	// ID is the pipeline constant ID of an override declaration, or -1.
	ID    int64
	Local bool
	state int
	deps  []Object
}

// ParamObj is a formal parameter.
type ParamObj struct {
	Name  string
	Decl  *Param
	Type  Type
	Func  *FuncObj
	Index int
}

// FuncObj is a user-defined function.
type FuncObj struct {
	Name   string
	Decl   *Func
	Params []*ParamObj
	Ret    Type // Void when there is no return type
	Entry  *EntryPoint
	// Calls and Globals are the direct callees and module-scope variables
	// and overrides referenced by the body.
	Calls   []*FuncObj
	Globals []Object
	state   int
	uses    map[Object]bool
	feats   map[string]*Node0
}

// Node0 records the first use of a stage-restricted feature.
type Node0 struct {
	Off  int
	What string
}

// TypeObj is a type alias or structure declaration.
type TypeObj struct {
	Name   string
	Alias  *Alias
	Struct *Struct
	Type   Type
	state  int
}

// BuiltinObj is a predeclared type, type-generator, function or enumerant.
type BuiltinObj struct{ Name string }

func (o *VarObj) ObjName() string     { return o.Name }
func (o *ValueObj) ObjName() string   { return o.Name }
func (o *ParamObj) ObjName() string   { return o.Name }
func (o *FuncObj) ObjName() string    { return o.Name }
func (o *TypeObj) ObjName() string    { return o.Name }
func (o *BuiltinObj) ObjName() string { return o.Name }

// Stage is a shader stage.
type Stage uint8

const (
	StageNone Stage = iota
	StageVertex
	StageFragment
	StageCompute
)

func (s Stage) String() string {
	return [...]string{"none", "vertex", "fragment", "compute"}[s]
}

// IOVar is one shader stage input or output of an entry point.
type IOVar struct {
	Name        string
	Type        Type
	Builtin     string // built-in value name, or ""
	Location    *int64
	BlendSrc    *int64
	Interpolate [2]string
	Invariant   bool
	// Param is the parameter index for inputs (-1 for outputs); Member is
	// the structure member name when the datum is a member.
	Param  int
	Member string
}

// EntryPoint describes an entry point function for a back end.
type EntryPoint struct {
	Func  *FuncObj
	Name  string
	Stage Stage
	// WorkgroupSize holds the constant sizes (1 when omitted); a dimension
	// given by an override-expression has size 0 and its expression in
	// WorkgroupSizeExprs.
	WorkgroupSize      [3]int64
	WorkgroupSizeExprs [3]Expr
	Inputs, Outputs    []IOVar
	// Functions are the functions in the shader stage (the entry point and
	// all functions it can call), in declaration order.
	Functions []*FuncObj
	// Resources are the resource variables statically accessed, sorted by
	// group then binding. Workgroup, Private are the other module-scope
	// variables statically accessed. Overrides are the statically accessed
	// override declarations.
	Resources []*VarObj
	Globals   []*VarObj
	Overrides []*ValueObj
}

// Info is the result of name resolution and type checking. It is what a
// back end (for example an MSL emitter) consumes next to the AST.
type Info struct {
	// Types maps every checked value expression to its static type. The
	// type is a reference type for variable and memory-view expressions
	// and may be abstract for const-expressions.
	Types map[Expr]Type
	// Objects maps every *Ident use (and declaration-site type name) to the
	// declaration it resolves to.
	Objects map[*Ident]Object
	// TypeSpecs maps every type specifier (an *Ident that denotes a type,
	// including constructor callees and template arguments) to that type.
	TypeSpecs map[*Ident]Type
	// Conversions maps an expression to the type it is automatically
	// converted to by its context (concretization, abstract conversion).
	// The load rule is not recorded: it is implied by a RefType in Types.
	Conversions map[Expr]Type
	// Consts maps every const-expression to its value.
	Consts map[Expr]*Const
	// Decls maps declaration nodes (*Var, *Value, *Param, *Func, *Alias,
	// *Struct) to their objects.
	Decls map[Node]Object
	// DeclTypes maps declaration nodes to their declared (effective) type:
	// store type for variables, value type for values and parameters, return
	// type for functions, denoted type for aliases and structs.
	DeclTypes map[Node]Type
	// Structs lists the user-declared structure types with their layouts.
	Structs []*StructType
	// Funcs lists the user-defined functions in declaration order.
	Funcs []*FuncObj
	// EntryPoints lists the entry points in declaration order.
	EntryPoints []*EntryPoint
	// Globals lists module-scope variables, Overrides override-declarations.
	Globals   []*VarObj
	Overrides []*ValueObj
	// Enabled lists the enable-extensions turned on by directives.
	Enabled map[string]bool
}

// CheckOptions tunes Check.
type CheckOptions struct {
	// Source is used to convert offsets to line and column positions. When
	// nil, positions carry only a byte offset (Line 1, Col offset+1).
	Source *Source
	// Name is the source name used in diagnostics when Source is nil.
	Name string
	// Extensions is the set of enable-extensions the device supports. Nil
	// means all of them.
	Extensions map[string]bool
	// LanguageFeatures is the set of language extensions supported by the
	// implementation. Nil means DefaultLanguageFeatures.
	LanguageFeatures map[string]bool
	// MaxErrors stops checking after this many diagnostics (default 100).
	MaxErrors int
	// MaxConstSteps bounds const-expression evaluation work (default 1<<20).
	MaxConstSteps int
}

// AllEnableExtensions lists the enable-extension names of the spec.
var AllEnableExtensions = []string{"f16", "clip_distances", "dual_source_blending", "subgroups",
	"primitive_index", "subgroup_size_control", "atomic_vec2u_min_max",
	"chromium_experimental_primitive_id"}

// AllLanguageFeatures lists the language extension names of the spec.
var AllLanguageFeatures = []string{"readonly_and_readwrite_storage_textures", "packed_4x8_integer_dot_product",
	"unrestricted_pointer_parameters", "pointer_composite_access", "uniform_buffer_standard_layout",
	"subgroup_id", "subgroup_uniformity", "texture_and_sampler_let", "texture_formats_tier1",
	"linear_indexing", "immediate_address_space", "fragment_depth", "buffer_view", "swizzle_assignment",
	"atomic_vec2u_min_max"}

// DefaultLanguageFeatures is every language extension this checker
// implements: all but buffer_view and swizzle_assignment (not implemented)
// and uniform_buffer_standard_layout (a module opts in with a requires
// directive, so that the strict uniform layout rules are the default).
func DefaultLanguageFeatures() map[string]bool {
	m := map[string]bool{}
	for _, f := range AllLanguageFeatures {
		if f != "buffer_view" && f != "swizzle_assignment" && f != "uniform_buffer_standard_layout" {
			m[f] = true
		}
	}
	return m
}

// Check resolves names and type-checks m.
func Check(m *Module, opts ...CheckOptions) (*Info, ErrorList) {
	var o CheckOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	c := newChecker(m, o)
	c.run()
	sort.SliceStable(c.errs, func(i, j int) bool { return c.errs[i].Pos.Offset < c.errs[j].Pos.Offset })
	return c.info, c.errs
}

// Validate parses and checks src. It is the entry point for "is this a valid
// WGSL module".
func Validate(src string, opts ...Options) error {
	var o Options
	if len(opts) > 0 {
		o = opts[0]
	}
	s := NewSource(o.Name, src)
	m, err := ParseSource(s, o.Limits)
	if err != nil {
		return err
	}
	_, errs := Check(m, CheckOptions{Source: s})
	return errs.Err()
}

// checkLimit is panicked to stop checking after too many errors.
type checkLimit struct{}

func (c *checker) errAt(off int, rule, format string, a ...any) {
	if c.suppress > 0 && len(rule) > 6 && rule[:6] == "const-" {
		return
	}
	var pos Position
	name := c.opt.Name
	if c.opt.Source != nil {
		pos = c.opt.Source.Position(off)
		name = c.opt.Source.Name
	} else {
		pos = Position{Offset: off, Line: 1, Col: off + 1}
	}
	for _, e := range c.errs {
		if e.Pos.Offset == off && e.Rule == rule {
			return
		}
	}
	c.errs = append(c.errs, &Error{Pos: pos, Name: name, Msg: fmt.Sprintf(format, a...), Rule: rule})
	if len(c.errs) >= c.maxErrs {
		panic(checkLimit{})
	}
}

func (c *checker) err(n Node, rule, format string, a ...any) {
	c.errAt(n.NodeSpan().Start, rule, format, a...)
}
