package wgsl

// Info is the result of name resolution and type checking. It is what a
// back end (for example an MSL emitter) consumes next to the AST.
type Info struct {
	// Types maps every checked expression to its type.
	Types map[Expr]Type
}

// Type is a WGSL type. The concrete types are defined in types.go.
type Type interface{ String() string }

// CheckOptions tunes Check.
type CheckOptions struct{}

// Check resolves names and type-checks m. STUB: the real checker replaces
// this file.
func Check(m *Module, opts ...CheckOptions) (*Info, ErrorList) {
	return &Info{Types: map[Expr]Type{}}, nil
}

// Validate parses and checks src. It is the entry point for "is this a valid
// WGSL module".
func Validate(src string, opts ...Options) error {
	m, err := Parse(src, opts...)
	if err != nil {
		return err
	}
	_, errs := Check(m)
	return errs.Err()
}
