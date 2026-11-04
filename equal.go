package wgsl

import "reflect"

var spanType = reflect.TypeOf(Span{})

// Equal reports whether two nodes are structurally equal, ignoring source
// spans. It is the relation under which Print(Parse(src)) round-trips.
func Equal(a, b Node) bool {
	return eq(reflect.ValueOf(a), reflect.ValueOf(b))
}

func eq(a, b reflect.Value) bool {
	if a.IsValid() != b.IsValid() {
		return false
	}
	if !a.IsValid() {
		return true
	}
	if a.Type() != b.Type() {
		return false
	}
	switch a.Kind() {
	case reflect.Interface, reflect.Pointer:
		if a.IsNil() || b.IsNil() {
			return a.IsNil() == b.IsNil()
		}
		return eq(a.Elem(), b.Elem())
	case reflect.Struct:
		if a.Type() == spanType {
			return true
		}
		for i := 0; i < a.NumField(); i++ {
			if !eq(a.Field(i), b.Field(i)) {
				return false
			}
		}
		return true
	case reflect.Slice:
		if a.Len() != b.Len() {
			return false
		}
		for i := 0; i < a.Len(); i++ {
			if !eq(a.Index(i), b.Index(i)) {
				return false
			}
		}
		return true
	case reflect.String:
		return a.String() == b.String()
	case reflect.Bool:
		return a.Bool() == b.Bool()
	case reflect.Int, reflect.Int64:
		return a.Int() == b.Int()
	}
	return false
}
