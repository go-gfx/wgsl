package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func run(t *testing.T, args ...string) (string, string, error) {
	t.Helper()
	var out, errb bytes.Buffer
	c := RootCmd()
	c.SetOut(&out)
	c.SetErr(&errb)
	c.SetArgs(args)
	err := c.Execute()
	return out.String(), errb.String(), err
}

func write(t *testing.T, src string) string {
	p := filepath.Join(t.TempDir(), "a.wgsl")
	if err := os.WriteFile(p, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCheck(t *testing.T) {
	ok := write(t, "@compute @workgroup_size(1) fn main() {}")
	out, _, err := run(t, "check", ok)
	if err != nil || !strings.Contains(out, "ok") {
		t.Fatalf("%v %q", err, out)
	}
	bad := write(t, "fn f() { let meta = 1; }")
	_, errs, err := run(t, "check", bad)
	if err == nil || !strings.Contains(errs, "reserved word") || !strings.Contains(errs, ":1:14:") {
		t.Fatalf("%v %q", err, errs)
	}
	typeBad := write(t, "fn f() { let x: i32 = 1.5; }")
	if _, _, err := run(t, "check", typeBad); err == nil {
		t.Fatal("type error accepted")
	}
	if _, _, err := run(t, "check", "--parse-only", typeBad); err != nil {
		t.Fatalf("parse-only: %v", err)
	}
}

func TestFmt(t *testing.T) {
	p := write(t, "fn   f(){return;}")
	out, _, err := run(t, "fmt", p)
	if err != nil || out != "fn f() {\n    return;\n}\n" {
		t.Fatalf("%v %q", err, out)
	}
	if out, _, _ := run(t, "fmt", "-l", p); !strings.Contains(out, "a.wgsl") {
		t.Fatalf("list: %q", out)
	}
	if _, _, err := run(t, "fmt", "-w", p); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(p)
	if string(b) != "fn f() {\n    return;\n}\n" {
		t.Fatalf("written %q", b)
	}
	if out, _, _ := run(t, "fmt", "-l", p); out != "" {
		t.Fatalf("formatted file still listed: %q", out)
	}
}

func TestMainExit(t *testing.T) {
	code := 0
	old := osExit
	osExit = func(c int) { code = c }
	defer func() { osExit = old }()
	bad := write(t, "fn")
	os.Args = []string{"wgsl", "check", bad}
	main()
	if code != 1 {
		t.Fatalf("exit code %d", code)
	}
}
