// Command specbuiltins reads the WGSL specification source (wgsl/index.bs)
// and prints, for every built-in function overload table, the overload
// signatures and the parameterization text, one record per table. It is
// used to build and to cross-check the built-in function table of the
// checker (builtins.go).
//
//	go run ./internal/specbuiltins path/to/index.bs > overloads.txt
package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"
)

var (
	tag     = regexp.MustCompile(`<[^>]*>`)
	spaces  = regexp.MustCompile(`\s+`)
	heading = regexp.MustCompile("^#{2,4} (.*?) #{2,4}")
)

func clean(s string) string {
	s = tag.ReplaceAllString(s, " ")
	s = strings.NewReplacer("&lt;", "<", "&gt;", ">", "&amp;", "&", "[=", "", "=]", "", "`", "").Replace(s)
	return strings.TrimSpace(spaces.ReplaceAllString(s, " "))
}

func main() {
	names := len(os.Args) == 3 && os.Args[1] == "-names"
	if len(os.Args) != 2 && !names {
		fmt.Fprintln(os.Stderr, "usage: specbuiltins [-names] index.bs")
		os.Exit(2)
	}
	f, err := os.Open(os.Args[len(os.Args)-1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	var lines []string
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	const first = 14990
	for i := first; i < len(lines); i++ {
		l := lines[i]
		if m := heading.FindStringSubmatch(l); m != nil {
			if names {
				if h := m[1]; strings.HasPrefix(h, "`") {
					n := strings.Trim(strings.Fields(h)[0], "`")
					fmt.Fprintln(out, n)
				}
				continue
			}
			fmt.Fprintf(out, "\n== %s (line %d)\n", m[1], i+1)
			continue
		}
		if names {
			continue
		}
		if !strings.Contains(l, ">Overload") {
			continue
		}
		var sig, par []string
		cur := &sig
		for j := i + 1; j < len(lines); j++ {
			t := lines[j]
			if strings.Contains(t, "Description") || strings.Contains(t, ">Overload") || strings.Contains(t, "</table>") {
				break
			}
			if strings.Contains(t, "Parameterization") {
				cur = &par
			}
			if s := clean(t); s != "" {
				*cur = append(*cur, s)
			}
		}
		fmt.Fprintf(out, "OV %s\n", strings.Join(sig, " ; "))
		if len(par) > 0 {
			fmt.Fprintf(out, "   %s\n", strings.Join(par, " ; "))
		}
	}
}
