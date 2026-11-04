// Command ctsreport summarises a mismatch file written by
// `go test -run TestCTSConformance -ctsreport <file>`: per level, kind and CTS
// test (id up to the parameters) the number of mismatches and a first example.
// With -test <substring> it prints every mismatching source of the tests whose
// name contains the substring (limited by -n).
//
// Usage: go run ./internal/ctsreport -level parse -kind false-accept <file>
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
)

type rec struct {
	Level    string   `json:"level"`
	Kind     string   `json:"kind"`
	ID       string   `json:"id"`
	File     string   `json:"file"`
	CTSLevel string   `json:"level_cts"`
	Features []string `json:"features"`
	Source   string   `json:"source"`
}

func main() {
	level := flag.String("level", "parse", "parse | validate")
	kind := flag.String("kind", "false-accept", "false-accept | false-reject | panic")
	test := flag.String("test", "", "print sources of tests whose name contains this")
	only := flag.String("only", "", "restrict to ids containing this (e.g. webgpu:shader,validation,parse,)")
	n := flag.Int("n", 3, "sources to print per test")
	flag.Parse()
	f, err := os.Open(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReaderSize(f, 1<<20))
	counts := map[string]int{}
	first := map[string]string{}
	printed := map[string]int{}
	for dec.More() {
		var r rec
		if err := dec.Decode(&r); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if r.Level != *level || r.Kind != *kind || !strings.Contains(r.ID, *only) {
			continue
		}
		name := strings.Join(strings.SplitN(r.ID, ":", 4)[:min(3, strings.Count(r.ID, ":")+1)], ":")
		counts[name]++
		if _, ok := first[name]; !ok {
			first[name] = r.ID
		}
		if *test != "" && strings.Contains(name, *test) && printed[name] < *n {
			printed[name]++
			fmt.Printf("---- %s  [%s, %s]\n%s\n", r.ID, r.File, r.CTSLevel, r.Source)
		}
	}
	if *test != "" {
		return
	}
	names := make([]string, 0, len(counts))
	for k := range counts {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		fmt.Printf("%7d %s\n        e.g. %s\n", counts[k], k, first[k])
	}
}
