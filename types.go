package wgsl

import (
	"fmt"
	"strings"
)

// Type is a WGSL type (spec sections "Types" and "Memory Views"). The
// concrete implementations are ScalarType, VecType, MatType, ArrayType,
// StructType, AtomicType, PtrType, RefType, SamplerType, TextureType,
// EnumType, VoidType and the internal InvalidType.
//
// Types of the same kind are compared with Same, never with ==, except
// scalars, which are singletons.
type Type interface {
	String() string
	isType()
}

// ScalarKind enumerates the scalar types, including the abstract numeric
// types that cannot be spelled in source.
type ScalarKind uint8

const (
	KBool ScalarKind = iota
	KAbstractInt
	KAbstractFloat
	KI32
	KU32
	KF32
	KF16
)

// ScalarType is a scalar type. There is exactly one value per kind.
type ScalarType struct{ Kind ScalarKind }

// The scalar types. AbstractInt and AbstractFloat only appear as the type of
// const-expressions.
var (
	Bool          = &ScalarType{KBool}
	AbstractInt   = &ScalarType{KAbstractInt}
	AbstractFloat = &ScalarType{KAbstractFloat}
	I32           = &ScalarType{KI32}
	U32           = &ScalarType{KU32}
	F32           = &ScalarType{KF32}
	F16           = &ScalarType{KF16}
)

func (s *ScalarType) isType() {}
func (s *ScalarType) String() string {
	switch s.Kind {
	case KBool:
		return "bool"
	case KAbstractInt:
		return "AbstractInt"
	case KAbstractFloat:
		return "AbstractFloat"
	case KI32:
		return "i32"
	case KU32:
		return "u32"
	case KF32:
		return "f32"
	}
	return "f16"
}

// VecType is vecN<T>.
type VecType struct {
	N    int
	Elem *ScalarType
}

func (*VecType) isType()          {}
func (v *VecType) String() string { return fmt.Sprintf("vec%d<%s>", v.N, v.Elem) }

// MatType is matCxR<T>: C columns of R-component vectors.
type MatType struct {
	C, R int
	Elem *ScalarType
}

func (*MatType) isType()          {}
func (m *MatType) String() string { return fmt.Sprintf("mat%dx%d<%s>", m.C, m.R, m.Elem) }

// ArrayType is array<E, N> (fixed-size), array<E> (runtime-sized) or an
// array whose element count is an override-expression.
type ArrayType struct {
	Elem Type
	// Count is the element count of a fixed-size array whose count is a
	// const-expression. It is 0 for runtime-sized and override-sized arrays.
	Count   int64
	Runtime bool
	// Override is the element count expression when it is an
	// override-expression that is not a const-expression.
	Override Expr
	// OverrideDecl is set when Override is exactly an identifier that
	// resolves to an override-declaration (only then do two array types with
	// the same count compare equal).
	OverrideDecl *ValueObj
}

func (*ArrayType) isType() {}
func (a *ArrayType) String() string {
	switch {
	case a.Runtime:
		return fmt.Sprintf("array<%s>", a.Elem)
	case a.Override != nil:
		if a.OverrideDecl != nil {
			return fmt.Sprintf("array<%s, %s>", a.Elem, a.OverrideDecl.Name)
		}
		return fmt.Sprintf("array<%s, override-expression>", a.Elem)
	}
	return fmt.Sprintf("array<%s, %d>", a.Elem, a.Count)
}

// StructType is a user-declared or predeclared (frexp, modf,
// atomicCompareExchangeWeak result) structure type. Two structure types are
// the same iff they are the same declaration.
type StructType struct {
	Name    string
	Decl    *Struct // nil for predeclared result types
	Members []*FieldInfo
	// Layout, valid after the declaration has been checked. Size excludes the
	// elements of a runtime-sized last member.
	Align, Size int64
	// Runtime reports that the last member is a runtime-sized array.
	Runtime bool
	// Predeclared structures are named with a leading "__" and cannot be
	// spelled in source.
	Predeclared bool
}

// FieldInfo is a structure member with its memory layout (spec "Memory
// Layout"): byte offset, size and alignment, taking @size and @align into
// account.
type FieldInfo struct {
	Name   string
	Type   Type
	Decl   *StructMember
	Offset int64
	Size   int64
	Align  int64
	// IO attributes, meaningful when the structure is used for entry point
	// IO.
	Location    *int64
	BlendSrc    *int64
	BuiltinName string
	Interpolate [2]string
	Invariant   bool
}

func (*StructType) isType()          {}
func (s *StructType) String() string { return s.Name }

// Member returns the named member or nil.
func (s *StructType) Member(name string) *FieldInfo {
	for _, m := range s.Members {
		if m.Name == name {
			return m
		}
	}
	return nil
}

// AtomicType is atomic<T> with T i32, u32 (or vec2<u32> with the
// atomic_vec2u_min_max extension).
type AtomicType struct{ Elem Type }

func (*AtomicType) isType()          {}
func (a *AtomicType) String() string { return fmt.Sprintf("atomic<%s>", a.Elem) }

// AddressSpace is a WGSL address space.
type AddressSpace uint8

const (
	ASNone AddressSpace = iota
	ASFunction
	ASPrivate
	ASWorkgroup
	ASUniform
	ASStorage
	ASImmediate
	ASHandle
)

func (a AddressSpace) String() string {
	return [...]string{"none", "function", "private", "workgroup", "uniform", "storage", "immediate", "handle"}[a]
}

// Access is an access mode.
type Access uint8

const (
	AccessNone Access = iota
	AccessRead
	AccessWrite
	AccessReadWrite
)

func (a Access) String() string {
	return [...]string{"none", "read", "write", "read_write"}[a]
}

// Readable reports whether reads are supported.
func (a Access) Readable() bool { return a == AccessRead || a == AccessReadWrite }

// Writable reports whether writes are supported.
func (a Access) Writable() bool { return a == AccessWrite || a == AccessReadWrite }

// PtrType is ptr<AS, T, AM>.
type PtrType struct {
	AS     AddressSpace
	Elem   Type
	Access Access
}

func (*PtrType) isType() {}
func (p *PtrType) String() string {
	return fmt.Sprintf("ptr<%s, %s, %s>", p.AS, p.Elem, p.Access)
}

// RefType is ref<AS, T, AM>, the type of a memory view. References never
// appear in source; they are the type of variable identifier expressions
// and of member and index accesses into memory.
type RefType struct {
	AS     AddressSpace
	Elem   Type
	Access Access
}

func (*RefType) isType() {}
func (r *RefType) String() string {
	return fmt.Sprintf("ref<%s, %s, %s>", r.AS, r.Elem, r.Access)
}

// SamplerType is sampler or sampler_comparison.
type SamplerType struct{ Comparison bool }

func (*SamplerType) isType() {}
func (s *SamplerType) String() string {
	if s.Comparison {
		return "sampler_comparison"
	}
	return "sampler"
}

// TextureKind classifies the texture types.
type TextureKind uint8

const (
	TexSampled TextureKind = iota
	TexMultisampled
	TexDepth
	TexDepthMultisampled
	TexExternal
	TexStorage
)

// TextureDim is the dimensionality of a texture.
type TextureDim uint8

const (
	Dim1D TextureDim = iota + 1
	Dim2D
	Dim3D
	DimCube
)

// TextureType is any texture type.
type TextureType struct {
	Kind    TextureKind
	Dim     TextureDim
	Arrayed bool
	// Sampled is the sampled type (f32, i32 or u32) for sampled and
	// multisampled textures.
	Sampled *ScalarType
	// Format and Access parameterize storage textures.
	Format string
	Access Access
}

func (*TextureType) isType() {}
func (t *TextureType) String() string {
	dim := map[TextureDim]string{Dim1D: "1d", Dim2D: "2d", Dim3D: "3d", DimCube: "cube"}[t.Dim]
	arr := ""
	if t.Arrayed {
		arr = "_array"
	}
	switch t.Kind {
	case TexSampled:
		return fmt.Sprintf("texture_%s%s<%s>", dim, arr, t.Sampled)
	case TexMultisampled:
		return fmt.Sprintf("texture_multisampled_2d<%s>", t.Sampled)
	case TexDepth:
		return fmt.Sprintf("texture_depth_%s%s", dim, arr)
	case TexDepthMultisampled:
		return "texture_depth_multisampled_2d"
	case TexExternal:
		return "texture_external"
	}
	return fmt.Sprintf("texture_storage_%s%s<%s, %s>", dim, arr, t.Format, t.Access)
}

// EnumType is the type of a predeclared enumerant (access mode, address
// space, texel format) used as a template argument.
type EnumType struct{ Enum string }

func (*EnumType) isType()          {}
func (e *EnumType) String() string { return e.Enum }

// VoidType is the "type" of a call to a function without a return type.
type VoidType struct{}

func (*VoidType) isType()        {}
func (*VoidType) String() string { return "void" }

// Void is the result type of functions that return nothing.
var Void = &VoidType{}

// InvalidType marks an expression whose checking already reported an error;
// it suppresses follow-on diagnostics.
type InvalidType struct{}

func (*InvalidType) isType()        {}
func (*InvalidType) String() string { return "<invalid>" }

// Invalid is the type of erroneous expressions.
var Invalid = &InvalidType{}

// TypeValue is the "type" of an expression that denotes a type (AllTypes).
// It never reaches Info.Types.
type typeValue struct{ T Type }

// ---- classification ----

func isInvalid(t Type) bool { _, ok := t.(*InvalidType); return ok }

// IsAbstract reports whether t is or contains an abstract numeric type.
func IsAbstract(t Type) bool {
	switch t := t.(type) {
	case *ScalarType:
		return t.Kind == KAbstractInt || t.Kind == KAbstractFloat
	case *VecType:
		return IsAbstract(t.Elem)
	case *MatType:
		return IsAbstract(t.Elem)
	case *ArrayType:
		return IsAbstract(t.Elem)
	case *StructType:
		if t.Predeclared {
			for _, m := range t.Members {
				if IsAbstract(m.Type) {
					return true
				}
			}
		}
	}
	return false
}

func isScalar(t Type) bool { _, ok := t.(*ScalarType); return ok }

func isNumericScalar(t Type) bool {
	s, ok := t.(*ScalarType)
	return ok && s.Kind != KBool
}

func isIntegerScalar(t Type) bool {
	s, ok := t.(*ScalarType)
	return ok && (s.Kind == KAbstractInt || s.Kind == KI32 || s.Kind == KU32)
}

func isConcreteIntScalar(t Type) bool {
	s, ok := t.(*ScalarType)
	return ok && (s.Kind == KI32 || s.Kind == KU32)
}

func isFloatScalar(t Type) bool {
	s, ok := t.(*ScalarType)
	return ok && (s.Kind == KAbstractFloat || s.Kind == KF32 || s.Kind == KF16)
}

func isSignedIntScalar(t Type) bool {
	s, ok := t.(*ScalarType)
	return ok && (s.Kind == KAbstractInt || s.Kind == KI32)
}

// scalarOf returns the scalar type of t when t is a scalar, or the component
// type when t is a vector.
func scalarOf(t Type) *ScalarType {
	switch t := t.(type) {
	case *ScalarType:
		return t
	case *VecType:
		return t.Elem
	}
	return nil
}

func vec(n int, e *ScalarType) *VecType { return &VecType{n, e} }

// Same reports whether a and b are the same type.
func Same(a, b Type) bool {
	if a == b {
		return true
	}
	switch a := a.(type) {
	case *ScalarType:
		return false
	case *VecType:
		b, ok := b.(*VecType)
		return ok && a.N == b.N && a.Elem == b.Elem
	case *MatType:
		b, ok := b.(*MatType)
		return ok && a.C == b.C && a.R == b.R && a.Elem == b.Elem
	case *ArrayType:
		b, ok := b.(*ArrayType)
		if !ok || !Same(a.Elem, b.Elem) || a.Runtime != b.Runtime {
			return false
		}
		if a.Runtime {
			return true
		}
		if a.Override != nil || b.Override != nil {
			return a.OverrideDecl != nil && a.OverrideDecl == b.OverrideDecl
		}
		return a.Count == b.Count
	case *StructType:
		return false
	case *AtomicType:
		b, ok := b.(*AtomicType)
		return ok && Same(a.Elem, b.Elem)
	case *PtrType:
		b, ok := b.(*PtrType)
		return ok && a.AS == b.AS && a.Access == b.Access && Same(a.Elem, b.Elem)
	case *RefType:
		b, ok := b.(*RefType)
		return ok && a.AS == b.AS && a.Access == b.Access && Same(a.Elem, b.Elem)
	case *SamplerType:
		b, ok := b.(*SamplerType)
		return ok && a.Comparison == b.Comparison
	case *TextureType:
		b, ok := b.(*TextureType)
		return ok && *a == *b
	case *EnumType:
		b, ok := b.(*EnumType)
		return ok && a.Enum == b.Enum
	}
	return false
}

// NestDepth is the nesting depth of a composite type (spec "Composite
// Types"); non-composite types have depth 0.
func NestDepth(t Type) int {
	switch t := t.(type) {
	case *VecType:
		return 1
	case *MatType:
		return 2
	case *ArrayType:
		return 1 + NestDepth(t.Elem)
	case *StructType:
		d := 0
		for _, m := range t.Members {
			if x := NestDepth(m.Type); x > d {
				d = x
			}
		}
		return 1 + d
	}
	return 0
}

// containsAtomic reports whether t is or contains an atomic type.
func containsAtomic(t Type) bool {
	switch t := t.(type) {
	case *AtomicType:
		return true
	case *ArrayType:
		return containsAtomic(t.Elem)
	case *StructType:
		for _, m := range t.Members {
			if containsAtomic(m.Type) {
				return true
			}
		}
	}
	return false
}

// containsRuntimeArray reports whether t is or contains a runtime-sized array.
func containsRuntimeArray(t Type) bool {
	switch t := t.(type) {
	case *ArrayType:
		return t.Runtime || containsRuntimeArray(t.Elem)
	case *StructType:
		return t.Runtime
	}
	return false
}

// containsAtomicVec2u reports whether t is or contains atomic<vec2<u32>>.
func containsAtomicVec2u(t Type) bool {
	switch t := t.(type) {
	case *AtomicType:
		_, ok := t.Elem.(*VecType)
		return ok
	case *ArrayType:
		return containsAtomicVec2u(t.Elem)
	case *StructType:
		for _, m := range t.Members {
			if containsAtomicVec2u(m.Type) {
				return true
			}
		}
	}
	return false
}

// hasOverrideCount reports whether t is an array (at any depth) whose
// element count is an override-expression.
func hasOverrideCount(t Type) bool {
	switch t := t.(type) {
	case *ArrayType:
		return t.Override != nil || hasOverrideCount(t.Elem)
	case *StructType:
		for _, m := range t.Members {
			if hasOverrideCount(m.Type) {
				return true
			}
		}
	}
	return false
}

// creationFixed reports whether t has a creation-fixed footprint (spec
// "Fixed-Footprint Types").
func creationFixed(t Type) bool {
	switch t := t.(type) {
	case *ScalarType, *VecType, *MatType, *AtomicType:
		return true
	case *ArrayType:
		return !t.Runtime && t.Override == nil && creationFixed(t.Elem)
	case *StructType:
		return !t.Runtime && !hasOverrideCount(t)
	}
	return false
}

// constructible reports whether t is constructible (spec "Constructible
// Types").
func constructible(t Type) bool {
	switch t := t.(type) {
	case *ScalarType, *VecType, *MatType:
		return true
	case *ArrayType:
		return !t.Runtime && t.Override == nil && constructible(t.Elem)
	case *StructType:
		for _, m := range t.Members {
			if !constructible(m.Type) && !(t.Predeclared && IsAbstract(m.Type) == false && isScalarOrVec(m.Type)) {
				return false
			}
		}
		return !(t.Predeclared && IsAbstract(t))
	}
	return false
}

func isScalarOrVec(t Type) bool {
	switch t.(type) {
	case *ScalarType, *VecType:
		return true
	}
	return false
}

// hostShareable reports whether t is host-shareable (spec "Host-shareable
// Types").
func hostShareable(t Type) bool {
	switch t := t.(type) {
	case *ScalarType:
		return t.Kind != KBool && !IsAbstract(t)
	case *VecType:
		return t.Elem.Kind != KBool && !IsAbstract(t)
	case *MatType:
		return !IsAbstract(t)
	case *AtomicType:
		return true
	case *ArrayType:
		if t.Override != nil {
			return false
		}
		return hostShareable(t.Elem)
	case *StructType:
		if t.Predeclared {
			return false
		}
		for _, m := range t.Members {
			if !hostShareable(m.Type) {
				return false
			}
		}
		return true
	}
	return false
}

// storable reports whether a variable can hold a value of type t.
func storable(t Type) bool {
	if IsAbstract(t) {
		return false
	}
	switch t.(type) {
	case *ScalarType, *VecType, *MatType, *AtomicType, *ArrayType, *StructType, *TextureType, *SamplerType:
		return true
	}
	return false
}

// plainType reports whether t is a plain type.
func plainType(t Type) bool {
	switch t.(type) {
	case *ScalarType, *VecType, *MatType, *AtomicType, *ArrayType, *StructType:
		return true
	}
	return false
}

// typeNames lists types for diagnostics.
func typeNames(ts []Type) string {
	s := make([]string, len(ts))
	for i, t := range ts {
		s[i] = t.String()
	}
	return strings.Join(s, ", ")
}
