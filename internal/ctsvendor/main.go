// Command ctsvendor vendors the WGSL validation corpus of the W3C gpuweb CTS
// (webgpu:shader,validation,*) into testdata/cts.
//
// It runs the real CTS test modules under Node through recorder.cjs (the one
// JavaScript file of this repository, see its header for why), reads the JSON
// lines it records, dedupes identical (source, verdict) pairs and writes
//
//	<out>/<group>.jsonl.gz  one Case per line (id, verdict, features, sha256, status, source)
//	<out>/MANIFEST.json     cts commit, per-group counts, pack names and sha256
//	<out>/LICENSE-CTS.txt   copy of the CTS LICENSE.txt (BSD-3-Clause)
//
// The corpus is ~384k distinct sources (~157 MB, highly repetitive), so it is
// vendored as one gzip pack per CTS directory instead of 384k loose files;
// -files <dir> additionally writes loose <dir>/<group>/<n>.wgsl files for
// debugging (never committed).
//
// Usage: go run ./internal/ctsvendor -cts <gpuweb-cts checkout> -out testdata/cts
package main

import (
	"bufio"
	"compress/gzip"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

//go:embed recorder.cjs
var recorderJS []byte

// maxSource bounds one vendored source; larger ones are listed as skipped-size.
const maxSource = 1 << 20

// Record is one line written by recorder.cjs.
type Record struct {
	ID       string   `json:"id"`
	Code     string   `json:"code"`
	Fail     bool     `json:"fail"`
	Pipeline bool     `json:"pipeline"`
	Constant bool     `json:"constants"`
	Features []string `json:"features"`
	Error    string   `json:"error"`
	Summary  bool     `json:"summary"`
	Cases    int      `json:"cases"`
	Errors   int      `json:"errors"`
}

// Case is one line of a pack.
type Case struct {
	ID         string   `json:"id"`
	Group      string   `json:"group"`
	Expected   string   `json:"expected"` // "pass" | "fail"
	Level      string   `json:"level"`    // "module" (createShaderModule) | "pipeline" (createXPipeline)
	Features   []string `json:"features,omitempty"`
	SHA256     string   `json:"sha256"` // of Source
	File       string   `json:"file"`   // virtual name <group>/<n>.wgsl
	Status     string   `json:"status"` // "run" | "skipped-uniformity" | "skipped-device" | "skipped-size"
	SkipReason string   `json:"skip_reason,omitempty"`
	Duplicates int      `json:"duplicates,omitempty"` // further CTS cases with the same (source, verdict)
	Source     string   `json:"source,omitempty"`
}

// Pack describes one <group>.jsonl.gz.
type Pack struct {
	Group  string         `json:"group"`
	File   string         `json:"file"`
	SHA256 string         `json:"sha256"` // of the gzip file
	Cases  int            `json:"cases"`
	Status map[string]int `json:"status"`
	Expect map[string]int `json:"expected"` // run cases only
}

// Manifest is MANIFEST.json.
type Manifest struct {
	CTSRepo      string         `json:"cts_repo"`
	CTSCommit    string         `json:"cts_commit"`
	Query        string         `json:"query"`
	Records      int            `json:"records"`       // recorded createShaderModule calls
	CTSCases     int            `json:"cts_cases"`     // test cases executed
	CaseErrors   []string       `json:"case_errors"`   // cases whose execution threw
	Distinct     int            `json:"distinct"`      // cases over all packs
	StatusCounts map[string]int `json:"status_counts"` // by status
	Packs        []Pack         `json:"packs"`
}

func main() {
	cts := flag.String("cts", "", "gpuweb/cts checkout (npm ci already run)")
	out := flag.String("out", "testdata/cts", "output directory")
	query := flag.String("query", "webgpu:shader,validation,*", "CTS query")
	node := flag.String("node", "node", "node binary")
	work := flag.String("work", "", "directory for the raw recording (default: user cache dir)")
	files := flag.String("files", "", "also write loose .wgsl files here (debugging)")
	flag.Parse()
	if *cts == "" {
		fatal("missing -cts")
	}
	if err := run(*cts, *out, *query, *node, *work, *files); err != nil {
		fatal(err.Error())
	}
}

func fatal(msg string) {
	fmt.Fprintln(os.Stderr, "ctsvendor:", msg)
	os.Exit(1)
}

func run(cts, out, query, node, work, loose string) error {
	cts, err := filepath.Abs(cts)
	if err != nil {
		return err
	}
	if work == "" {
		cd, err := os.UserCacheDir()
		if err != nil {
			return err
		}
		work = filepath.Join(cd, "go-gfx-wgsl-ctsvendor")
	}
	if err := os.MkdirAll(work, 0o755); err != nil {
		return err
	}
	rec := filepath.Join(work, "recorder.cjs")
	raw := filepath.Join(work, "records.jsonl")
	if err := os.WriteFile(rec, recorderJS, 0o644); err != nil {
		return err
	}
	commit, err := gitHead(cts)
	if err != nil {
		return err
	}
	cmd := exec.Command(node, rec, cts, raw, query)
	cmd.Dir = cts
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("node recorder: %w", err)
	}
	defer os.Remove(raw)

	man, groups, err := collect(raw)
	if err != nil {
		return err
	}
	man.CTSRepo, man.CTSCommit, man.Query = "https://github.com/gpuweb/cts", commit, query
	if err := writeOut(out, cts, man, groups); err != nil {
		return err
	}
	if loose != "" {
		if err := writeLoose(loose, groups); err != nil {
			return err
		}
	}
	fmt.Printf("cts %s: %d cases executed, %d records, %d distinct, status %v, %d case errors\n",
		commit[:7], man.CTSCases, man.Records, man.Distinct, man.StatusCounts, len(man.CaseErrors))
	for _, p := range man.Packs {
		fmt.Printf("  %-12s %7d cases %v expected %v\n", p.Group, p.Cases, p.Status, p.Expect)
	}
	return nil
}

func gitHead(dir string) (string, error) {
	b, err := exec.Command("git", "-C", dir, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(b)), nil
}

type key struct {
	sum  [32]byte
	fail bool
}

// collect streams the raw records, deduping by (source, verdict).
func collect(path string) (*Manifest, map[string][]*Case, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReaderSize(f, 1<<20))
	man := &Manifest{StatusCounts: map[string]int{}, CaseErrors: []string{}}
	seen := map[key]*Case{}
	groups := map[string][]*Case{}
	for {
		var r Record
		if err := dec.Decode(&r); err == io.EOF {
			break
		} else if err != nil {
			return nil, nil, err
		}
		switch {
		case r.Summary:
			man.CTSCases = r.Cases
			continue
		case r.Error != "":
			man.CaseErrors = append(man.CaseErrors, r.ID+": "+r.Error)
			continue
		}
		man.Records++
		k := key{sha256.Sum256([]byte(r.Code)), r.Fail}
		if c, ok := seen[k]; ok {
			c.Duplicates++
			continue
		}
		group := groupOf(r.ID)
		c := &Case{
			ID: r.ID, Group: group, Expected: "pass", Level: "module",
			SHA256: hex.EncodeToString(k.sum[:]),
			File:   fmt.Sprintf("%s/%05d.wgsl", group, len(groups[group])+1),
			Source: r.Code,
		}
		seen[k] = c
		if r.Fail {
			c.Expected = "fail"
		}
		if r.Pipeline {
			c.Level = "pipeline"
		}
		c.Features = r.Features
		sort.Strings(c.Features)
		c.Status, c.SkipReason = classify(r, group)
		if c.Status == "skipped-size" {
			c.Source = "" // listed, not vendored
		}
		groups[group] = append(groups[group], c)
		man.Distinct++
		man.StatusCounts[c.Status]++
	}
	return man, groups, nil
}

// groupOf returns the CTS directory of a case id
// ("webgpu:shader,validation,<group>,...").
func groupOf(id string) string {
	rest := strings.TrimPrefix(id, "webgpu:shader,validation,")
	g, _, _ := strings.Cut(rest, ",")
	g, _, _ = strings.Cut(g, ":")
	return g
}

// classify decides which cases a v0.1 front end cannot be judged on.
func classify(r Record, group string) (status, reason string) {
	lid := strings.ToLower(r.ID)
	switch {
	case len(r.Code) > maxSource:
		return "skipped-size", "source larger than 1 MiB"
	case group == "uniformity" || strings.Contains(lid, "uniformity"):
		return "skipped-uniformity", "uniformity analysis is not implemented in v0.1"
	case r.Constant:
		return "skipped-device", "verdict depends on pipeline-overridable constant values supplied at pipeline creation"
	case strings.Contains(lid, "limit") || strings.Contains(lid, "subgroup_size"):
		return "skipped-device", "verdict depends on a device limit or subgroup size"
	}
	return "run", ""
}

func writeOut(out, cts string, man *Manifest, groups map[string][]*Case) error {
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	old, err := filepath.Glob(filepath.Join(out, "*.jsonl.gz"))
	if err != nil {
		return err
	}
	for _, p := range old {
		if err := os.Remove(p); err != nil {
			return err
		}
	}
	names := make([]string, 0, len(groups))
	for g := range groups {
		names = append(names, g)
	}
	sort.Strings(names)
	for _, g := range names {
		p := Pack{Group: g, File: g + ".jsonl.gz", Cases: len(groups[g]), Status: map[string]int{}, Expect: map[string]int{}}
		path := filepath.Join(out, p.File)
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		h := sha256.New()
		zw, _ := gzip.NewWriterLevel(io.MultiWriter(f, h), gzip.BestCompression)
		enc := json.NewEncoder(zw)
		for _, c := range groups[g] {
			p.Status[c.Status]++
			if c.Status == "run" {
				p.Expect[c.Expected]++
			}
			if err := enc.Encode(c); err != nil {
				return err
			}
		}
		if err := zw.Close(); err != nil {
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
		p.SHA256 = hex.EncodeToString(h.Sum(nil))
		man.Packs = append(man.Packs, p)
	}
	lic, err := os.ReadFile(filepath.Join(cts, "LICENSE.txt"))
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "LICENSE-CTS.txt"), lic, 0o644); err != nil {
		return err
	}
	b, err := json.MarshalIndent(man, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "MANIFEST.json"), append(b, '\n'), 0o644)
}

func writeLoose(dir string, groups map[string][]*Case) error {
	for _, cs := range groups {
		for _, c := range cs {
			if c.Source == "" {
				continue
			}
			p := filepath.Join(dir, filepath.FromSlash(c.File))
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				return err
			}
			if err := os.WriteFile(p, []byte(c.Source), 0o644); err != nil {
				return err
			}
		}
	}
	return nil
}
