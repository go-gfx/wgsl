// Command ctserrs groups the false rejects of a -ctsreport file by the rule
// that rejected them: for each group the count, one CTS id and the message.
//
//	go test -run TestCTSConformance -ctsreport rep.jsonl .
//	go run ./internal/ctserrs rep.jsonl
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/go-gfx/wgsl"
)

type rec struct {
	ID     string `json:"id"`
	Kind   string `json:"kind"`
	Level  string `json:"level"`
	Source string `json:"source"`
}

type group struct {
	n       int
	id, msg string
}

func main() {
	kind := "false-reject"
	if len(os.Args) > 2 {
		kind = os.Args[2]
	}
	f, err := os.Open(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	groups := map[string]*group{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		var r rec
		if json.Unmarshal(sc.Bytes(), &r) != nil || r.Kind != kind || r.Level != "validate" {
			continue
		}
		key, msg := "accepted", ""
		if err := wgsl.Validate(r.Source); err != nil {
			if e, ok := err.(wgsl.ErrorList); ok && len(e) > 0 {
				key, msg = e[0].Rule, e[0].Msg
			} else if e, ok := err.(*wgsl.Error); ok {
				key, msg = e.Rule, e.Msg
			}
		}
		g := groups[key]
		if g == nil {
			g = &group{id: r.ID, msg: msg}
			groups[key] = g
		}
		g.n++
	}
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return groups[keys[i]].n > groups[keys[j]].n })
	for _, k := range keys {
		g := groups[k]
		fmt.Printf("%6d %-22s %s\n         e.g. %s\n", g.n, k, g.msg, g.id)
	}
}
