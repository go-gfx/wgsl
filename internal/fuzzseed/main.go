// Command fuzzseed writes the committed fuzz seed corpus
// (testdata/fuzz/Fuzz*/) from the vendored real shaders and a CTS sample.
//
//	go run ./internal/fuzzseed
package main

import (
	"bufio"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

var targets = []string{"FuzzLex", "FuzzParse", "FuzzRoundTrip", "FuzzValidate"}

func write(src string) {
	sum := sha256.Sum256([]byte(src))
	name := hex.EncodeToString(sum[:8])
	body := fmt.Sprintf("go test fuzz v1\nstring(%q)\n", src)
	for _, t := range targets {
		dir := filepath.Join("testdata", "fuzz", t)
		must(os.MkdirAll(dir, 0o755))
		must(os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644))
	}
}

func main() {
	real, err := filepath.Glob("testdata/real/*/*.wgsl")
	must(err)
	for _, f := range real {
		b, err := os.ReadFile(f)
		must(err)
		if len(b) <= 8192 {
			write(string(b))
		}
	}
	for _, s := range edge {
		write(s)
	}
	// every 400th distinct CTS source of the parse pack: syntax edge cases
	fh, err := os.Open("testdata/cts/parse.jsonl.gz")
	must(err)
	defer fh.Close()
	zr, err := gzip.NewReader(fh)
	must(err)
	sc := bufio.NewScanner(zr)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	n := 0
	for sc.Scan() {
		n++
		if n%40 != 0 {
			continue
		}
		var rec struct {
			Source string `json:"source"`
		}
		if json.Unmarshal(sc.Bytes(), &rec) == nil && len(rec.Source) <= 4096 && rec.Source != "" {
			write(rec.Source)
		}
	}
	must(sc.Err())
}

var edge = []string{
	"", "fn", "fn f() {}", "@", "@compute", "var<", "var<private>", "a<b>c", "let x = a<b>(c);",
	"fn f() { x = a < b, c > d; }", "/*", "/* /* */", "//", "0x", "0x1p", "1e", ".e1", "0.f", "1f", "01",
	"fn f() { let x = array<array<array<f32,2>,2>,2>(); }", "struct S { a: i32 }", "const c = 1 << 31;",
	"fn f() { loop { continuing { break if true; } } }", "fn f() { switch 1 { case 1, default: {} } }",
	"enable f16; diagnostic(off, x.y);", "alias A = array<f32, 4>;", "fn f() { _ = 1; ; }",
	"é δ \u0000", "fn f() -> @location(0) vec4<f32> { return vec4<f32>(); }",
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
