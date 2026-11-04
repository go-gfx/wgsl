// Command corpus vendors the WGSL sources of a checkout into testdata/real.
//
// It copies every *.wgsl file and every Go string that looks like a WGSL
// module (contains "fn " and one of @compute/@vertex/@fragment/
// @workgroup_size) into <out>/<name>/, and records provenance (repository,
// commit, file, line, sha256) in <out>/<name>/MANIFEST.json.
//
// Go strings are evaluated the way the Go compiler would: string literals,
// `+` concatenation and references to package-level string constants. A
// function that builds its result with a strings.Builder is simulated: each
// WriteString whose argument is evaluable is appended, any other argument
// (a number formatted in a loop) is replaced by "0", and `if` bodies are
// skipped; such entries have kind "assembled" and are syntactic samples only.
// "{{WG}}" (go-wasmcontainers/llm's workgroup-size placeholder) is replaced
// by 64, as that package's own ReplaceAll does at run time. Strings with fmt
// verbs are templates and are listed without being copied.
//
//	go run ./internal/corpus -name gpu -repo go-wasmcontainers/gpu -root ../wgsl-src-gpu -out testdata/real
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

type entry struct {
	Source string `json:"source"`
	Line   int    `json:"line"`
	File   string `json:"file,omitempty"`
	SHA256 string `json:"sha256,omitempty"`
	Kind   string `json:"kind"` // wgsl-file, go-string, assembled, template
}

type manifest struct {
	Repo    string  `json:"repo"`
	Commit  string  `json:"commit"`
	Licence string  `json:"licence"`
	Entries []entry `json:"entries"`
}

func looksWGSL(s string) bool {
	if !strings.Contains(s, "fn ") || !strings.Contains(s, "{") {
		return false
	}
	for _, k := range []string{"@compute", "@vertex", "@fragment", "@workgroup_size"} {
		if strings.Contains(s, k) {
			return true
		}
	}
	return false
}

func isTemplate(s string) bool {
	for _, v := range []string{"%d", "%s", "%v", "%f", "%g", "%x", "%q", "%[", "%.", "%0", "%t"} {
		if strings.Contains(s, v) {
			return true
		}
	}
	return false
}

type env map[string]string

// eval evaluates a constant string expression.
func (e env) eval(x ast.Expr) (string, bool) {
	switch x := x.(type) {
	case *ast.BasicLit:
		if x.Kind != token.STRING {
			return "", false
		}
		s, err := strconv.Unquote(x.Value)
		return s, err == nil
	case *ast.Ident:
		s, ok := e[x.Name]
		return s, ok
	case *ast.ParenExpr:
		return e.eval(x.X)
	case *ast.BinaryExpr:
		if x.Op != token.ADD {
			return "", false
		}
		a, ok1 := e.eval(x.X)
		b, ok2 := e.eval(x.Y)
		return a + b, ok1 && ok2
	}
	return "", false
}

// loose is eval where an operand that is not a constant becomes "0"; known
// reports whether at least one operand was a constant.
func (e env) loose(x ast.Expr) (string, bool) {
	if b, ok := x.(*ast.BinaryExpr); ok && b.Op == token.ADD {
		l, k1 := e.loose(b.X)
		r, k2 := e.loose(b.Y)
		return l + r, k1 || k2
	}
	if s, ok := e.eval(x); ok {
		return s, true
	}
	return "0", false
}

func isBuilderWrite(c *ast.CallExpr) bool {
	sel, ok := c.Fun.(*ast.SelectorExpr)
	return ok && sel.Sel.Name == "WriteString" && len(c.Args) == 1
}

// simulate runs the WriteString calls of a function body in source order.
func (e env) simulate(body *ast.BlockStmt) (string, bool) {
	var sb strings.Builder
	seen := false
	var walk func(n ast.Node, inIf bool)
	walk = func(n ast.Node, inIf bool) {
		ast.Inspect(n, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.IfStmt:
				return false
			case *ast.CallExpr:
				if isBuilderWrite(n) {
					seen = true
					if s, ok := e.eval(n.Args[0]); ok {
						sb.WriteString(s)
					} else {
						sb.WriteString("0")
					}
					return false
				}
			}
			return true
		})
	}
	walk(body, false)
	return sb.String(), seen
}

func main() {
	name := flag.String("name", "", "corpus name")
	repo := flag.String("repo", "", "org/repo")
	root := flag.String("root", "", "checkout")
	out := flag.String("out", "testdata/real", "output dir")
	flag.Parse()
	if *name == "" || *root == "" || *repo == "" {
		flag.Usage()
		os.Exit(2)
	}
	rev, err := exec.Command("git", "-C", *root, "rev-parse", "HEAD").Output()
	if err != nil {
		fatal(err)
	}
	m := manifest{Repo: *repo, Commit: strings.TrimSpace(string(rev)), Licence: "see LICENSE of " + *repo}
	dst := filepath.Join(*out, *name)
	if err := os.RemoveAll(dst); err != nil {
		fatal(err)
	}
	if err := os.MkdirAll(dst, 0o755); err != nil {
		fatal(err)
	}
	if b, err := os.ReadFile(filepath.Join(*root, "LICENSE")); err == nil {
		if err := os.WriteFile(filepath.Join(dst, "LICENSE"), b, 0o644); err != nil {
			fatal(err)
		}
	}
	save := func(e entry, text string) {
		text = strings.ReplaceAll(text, "{{WG}}", "64")
		sum := sha256.Sum256([]byte(text))
		e.SHA256 = hex.EncodeToString(sum[:])
		e.File = fmt.Sprintf("%s.%d.wgsl", strings.ReplaceAll(e.Source, "/", "__"), e.Line)
		if err := os.WriteFile(filepath.Join(dst, e.File), []byte(text), 0o644); err != nil {
			fatal(err)
		}
		m.Entries = append(m.Entries, e)
	}
	add := func(e entry, s string) {
		if isTemplate(strings.ReplaceAll(s, "{{WG}}", "")) {
			e.Kind = "template"
			m.Entries = append(m.Entries, e)
			return
		}
		save(e, s)
	}

	type goFile struct {
		rel  string
		fset *token.FileSet
		f    *ast.File
	}
	byDir := map[string][]goFile{}
	werr := filepath.WalkDir(*root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" || d.Name() == "node_modules" {
				return fs.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(*root, path)
		switch {
		case strings.HasSuffix(path, ".wgsl"):
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			save(entry{Source: rel, Line: 1, Kind: "wgsl-file"}, string(b))
		case strings.HasSuffix(path, ".go"):
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
			if err != nil {
				return nil
			}
			dir := filepath.Dir(rel)
			byDir[dir] = append(byDir[dir], goFile{rel, fset, f})
		}
		return nil
	})
	if werr != nil {
		fatal(werr)
	}
	for _, files := range byDir {
		ev := env{}
		for changed := true; changed; { // fixpoint over package-level constants
			changed = false
			for _, gf := range files {
				for _, d := range gf.f.Decls {
					gd, ok := d.(*ast.GenDecl)
					if !ok {
						continue
					}
					for _, sp := range gd.Specs {
						vs, ok := sp.(*ast.ValueSpec)
						if !ok || len(vs.Names) != len(vs.Values) {
							continue
						}
						for i, n := range vs.Names {
							if _, done := ev[n.Name]; done {
								continue
							}
							if s, ok := ev.eval(vs.Values[i]); ok {
								ev[n.Name] = s
								changed = true
							}
						}
					}
				}
			}
		}
		for _, gf := range files {
			used := map[*ast.BasicLit]bool{}
			markLits := func(n ast.Node) {
				ast.Inspect(n, func(n ast.Node) bool {
					if l, ok := n.(*ast.BasicLit); ok {
						used[l] = true
					}
					return true
				})
			}
			ast.Inspect(gf.f, func(n ast.Node) bool {
				switch n := n.(type) {
				case *ast.FuncDecl:
					if n.Body == nil {
						return true
					}
					if s, ok := ev.simulate(n.Body); ok && looksWGSL(s) {
						add(entry{Source: gf.rel, Line: gf.fset.Position(n.Pos()).Line, Kind: "assembled"}, s)
						markLits(n.Body)
					}
				case *ast.BinaryExpr:
					if n.Op != token.ADD {
						return true
					}
					// outermost concatenation: operands that are not
					// constants (function parameters) become "0".
					kind := "go-string"
					s, ok := ev.eval(n)
					if !ok {
						kind = "assembled"
						var known bool
						s, known = ev.loose(n)
						ok = known
					}
					if ok && looksWGSL(s) {
						add(entry{Source: gf.rel, Line: gf.fset.Position(n.Pos()).Line, Kind: kind}, s)
					}
					markLits(n)
					return false
				case *ast.BasicLit:
					if n.Kind != token.STRING || used[n] {
						return true
					}
					if s, ok := ev.eval(n); ok && looksWGSL(s) {
						add(entry{Source: gf.rel, Line: gf.fset.Position(n.Pos()).Line, Kind: "go-string"}, s)
					}
				}
				return true
			})
		}
	}
	b, _ := json.MarshalIndent(m, "", "  ")
	if err := os.WriteFile(filepath.Join(dst, "MANIFEST.json"), append(b, '\n'), 0o644); err != nil {
		fatal(err)
	}
	fmt.Printf("%s: %d entries\n", *name, len(m.Entries))
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
