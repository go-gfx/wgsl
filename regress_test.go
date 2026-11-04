package wgsl

import (
	"runtime/debug"
	"testing"
)

// Inputs that once panicked the checker (found by the CTS oracle).
func TestCheckerNeverPanics(t *testing.T) {
	for _, src := range []string{
		"struct S {\n  @location(0) @interpolate(flat, either) value : buffer,\n};\n@fragment\nfn main(in : S)  { }",
		"struct S {\n  @location(0) @interpolate(flat, either) value : buffer<16>,\n};\n@fragment\nfn main(in : S)  { }",
	} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("panic %v on %q\n%s", r, src, debug.Stack())
				}
			}()
			if Validate(src) == nil {
				t.Errorf("accepted %q", src)
			}
		}()
	}
}
