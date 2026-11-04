package wgsl

// Conformance oracle: the WGSL validation corpus of the W3C gpuweb CTS
// (webgpu:shader,validation,*), vendored by internal/ctsvendor into
// testdata/cts. See docs/CONFORMANCE.md.
//
// Every "run" case is judged twice against the verdict the CTS expects:
//
//	parse    : Parse(src)     - lexer+parser only. A syntax-only front end can
//	                            legitimately accept expected-fail cases whose
//	                            error is semantic, so for this level the
//	                            number that matters is false-reject.
//	validate : Validate(src)  - Parse + Check (moves as the checker lands).
//
// Known mismatches are recorded in testdata/cts/BASELINE.json (per level, per
// kind, per CTS directory, as ranges of case numbers). The test fails on any
// mismatch that is not in the baseline; mismatches that disappeared are
// reported and recorded by running `go test -run TestCTSConformance -update`.

import (
	"bufio"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
)

var (
	ctsUpdate = flag.Bool("update", false, "rewrite testdata/cts/BASELINE.json from the current results")
	ctsReport = flag.String("ctsreport", "", "write every mismatch (id, level, kind, source) as JSON lines to this file")
)

const ctsDir = "testdata/cts"

type ctsCase struct {
	ID       string   `json:"id"`
	Group    string   `json:"group"`
	Expected string   `json:"expected"`
	Level    string   `json:"level"`
	Features []string `json:"features"`
	File     string   `json:"file"`
	Status   string   `json:"status"`
	Source   string   `json:"source"`
}

type ctsManifest struct {
	CTSCommit string `json:"cts_commit"`
	Packs     []struct {
		Group  string `json:"group"`
		File   string `json:"file"`
		SHA256 string `json:"sha256"`
	} `json:"packs"`
}

// Mismatch kinds.
const (
	kindFalseReject = "false-reject" // expected pass, rejected
	kindFalseAccept = "false-accept" // expected fail, accepted
	kindPanic       = "panic"        // the front end panicked
)

// ctsBaseline is BASELINE.json. Mismatches[level][kind][group] is a range
// list ("1-5,9") of case numbers (the <n> of file <group>/<n>.wgsl).
type ctsBaseline struct {
	CTSCommit  string                                  `json:"cts_commit"`
	Packs      map[string]string                       `json:"pack_sha256"`
	Mismatches map[string]map[string]map[string]string `json:"mismatches"`
}

type ctsRow struct {
	total, skipped, okPass, okFail, falseReject, falseAccept, panics int
}

func TestCTSConformance(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(ctsDir, "MANIFEST.json"))
	if err != nil {
		t.Skip("no vendored CTS corpus: ", err)
	}
	var man ctsManifest
	if err := json.Unmarshal(raw, &man); err != nil {
		t.Fatal(err)
	}

	levels := []string{"parse", "validate"}
	rows := map[string]map[string]*ctsRow{} // level -> group -> row
	mism := map[string]map[string]map[string]map[int]bool{}
	for _, l := range levels {
		rows[l] = map[string]*ctsRow{}
		mism[l] = map[string]map[string]map[int]bool{}
	}
	var report *bufio.Writer
	if *ctsReport != "" {
		f, err := os.Create(*ctsReport)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		report = bufio.NewWriter(f)
		defer report.Flush()
	}

	packs := map[string]string{}
	var groups []string
	for _, p := range man.Packs {
		packs[p.Group] = p.SHA256
		groups = append(groups, p.Group)
		sum, cases, err := readPack(filepath.Join(ctsDir, p.File), func(c *ctsCase) {
			n := caseNumber(c.File)
			for _, l := range levels {
				r := rows[l][p.Group]
				if r == nil {
					r = &ctsRow{}
					rows[l][p.Group] = r
				}
				r.total++
				if c.Status != "run" {
					r.skipped++
					continue
				}
				kind, panicked := judge(l, c)
				switch {
				case panicked:
					r.panics++
				case kind == "":
					if c.Expected == "pass" {
						r.okPass++
					} else {
						r.okFail++
					}
				case kind == kindFalseReject:
					r.falseReject++
				default:
					r.falseAccept++
				}
				if panicked {
					kind = kindPanic
				}
				if kind != "" {
					byKind := mism[l][kind]
					if byKind == nil {
						byKind = map[string]map[int]bool{}
						mism[l][kind] = byKind
					}
					if byKind[p.Group] == nil {
						byKind[p.Group] = map[int]bool{}
					}
					byKind[p.Group][n] = true
					if report != nil {
						b, _ := json.Marshal(map[string]any{"level": l, "kind": kind, "id": c.ID, "file": c.File, "level_cts": c.Level, "features": c.Features, "source": c.Source})
						report.Write(append(b, '\n'))
					}
				}
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		if sum != p.SHA256 {
			t.Fatalf("%s: pack sha256 %s does not match MANIFEST %s", p.File, sum, p.SHA256)
		}
		_ = cases
	}
	sort.Strings(groups)

	for _, l := range levels {
		t.Log("\n" + table(l, groups, rows[l]))
	}

	cur := ctsBaseline{CTSCommit: man.CTSCommit, Packs: packs, Mismatches: map[string]map[string]map[string]string{}}
	for _, l := range levels {
		cur.Mismatches[l] = map[string]map[string]string{}
		for kind, byGroup := range mism[l] {
			cur.Mismatches[l][kind] = map[string]string{}
			for g, set := range byGroup {
				cur.Mismatches[l][kind][g] = encodeRanges(set)
			}
		}
	}
	basePath := filepath.Join(ctsDir, "BASELINE.json")
	if *ctsUpdate {
		b, err := json.MarshalIndent(cur, "", " ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(basePath, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", basePath)
		return
	}
	braw, err := os.ReadFile(basePath)
	if err != nil {
		t.Fatalf("no baseline (%v); run with -update", err)
	}
	var base ctsBaseline
	if err := json.Unmarshal(braw, &base); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.Packs, packs) {
		t.Fatalf("BASELINE.json was recorded for another corpus (cts %s); rerun with -update", base.CTSCommit)
	}
	for _, l := range levels {
		for _, kind := range []string{kindFalseReject, kindFalseAccept, kindPanic} {
			for _, g := range groups {
				have := decodeRanges(cur.Mismatches[l][kind][g])
				want := decodeRanges(base.Mismatches[l][kind][g])
				var regress, fixed []int
				for n := range have {
					if !want[n] {
						regress = append(regress, n)
					}
				}
				for n := range want {
					if !have[n] {
						fixed = append(fixed, n)
					}
				}
				sort.Ints(regress)
				sort.Ints(fixed)
				if len(fixed) > 0 {
					t.Logf("%s/%s/%s: %d baseline mismatches now match (rerun with -update to record): %s", l, kind, g, len(fixed), encodeRanges(setOf(fixed)))
				}
				if len(regress) > 0 {
					t.Errorf("%s/%s/%s: %d mismatches not in baseline (case numbers %s of %s/<n>.wgsl)", l, kind, g, len(regress), encodeRanges(setOf(regress)), g)
				}
			}
		}
	}
}

// judge runs the front end at the given level and returns the mismatch kind
// ("" when the verdict equals the CTS expectation).
func judge(level string, c *ctsCase) (kind string, panicked bool) {
	defer func() {
		if recover() != nil {
			panicked = true
		}
	}()
	var err error
	if level == "parse" {
		_, err = Parse(c.Source)
	} else {
		err = Validate(c.Source)
	}
	switch {
	case c.Expected == "pass" && err != nil:
		return kindFalseReject, false
	case c.Expected == "fail" && err == nil:
		return kindFalseAccept, false
	}
	return "", false
}

func readPack(path string, fn func(*ctsCase)) (sum string, n int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	zr, err := gzip.NewReader(io.TeeReader(f, h))
	if err != nil {
		return "", 0, err
	}
	dec := json.NewDecoder(bufio.NewReaderSize(zr, 1<<20))
	for {
		var c ctsCase
		if err := dec.Decode(&c); err == io.EOF {
			break
		} else if err != nil {
			return "", n, err
		}
		fn(&c)
		n++
	}
	if _, err := io.Copy(io.Discard, io.TeeReader(f, h)); err != nil { // trailing bytes
		return "", n, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func caseNumber(file string) int {
	base := strings.TrimSuffix(filepath.Base(file), ".wgsl")
	n, _ := strconv.Atoi(base)
	return n
}

func setOf(ns []int) map[int]bool {
	m := make(map[int]bool, len(ns))
	for _, n := range ns {
		m[n] = true
	}
	return m
}

func encodeRanges(set map[int]bool) string {
	ns := make([]int, 0, len(set))
	for n := range set {
		ns = append(ns, n)
	}
	sort.Ints(ns)
	var sb strings.Builder
	for i := 0; i < len(ns); {
		j := i
		for j+1 < len(ns) && ns[j+1] == ns[j]+1 {
			j++
		}
		if sb.Len() > 0 {
			sb.WriteByte(',')
		}
		if j == i {
			sb.WriteString(strconv.Itoa(ns[i]))
		} else {
			fmt.Fprintf(&sb, "%d-%d", ns[i], ns[j])
		}
		i = j + 1
	}
	return sb.String()
}

func decodeRanges(s string) map[int]bool {
	set := map[int]bool{}
	if s == "" {
		return set
	}
	for _, part := range strings.Split(s, ",") {
		lo, hi, isRange := strings.Cut(part, "-")
		a, _ := strconv.Atoi(lo)
		b := a
		if isRange {
			b, _ = strconv.Atoi(hi)
		}
		for n := a; n <= b; n++ {
			set[n] = true
		}
	}
	return set
}

func table(level string, groups []string, rows map[string]*ctsRow) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "CTS conformance, %s level\n", level)
	fmt.Fprintf(&sb, "%-12s %8s %8s %9s %9s %9s %9s %6s\n", "directory", "total", "skipped", "pass-ok", "fail-ok", "f-reject", "f-accept", "panic")
	var sum ctsRow
	for _, g := range groups {
		r := rows[g]
		if r == nil {
			continue
		}
		fmt.Fprintf(&sb, "%-12s %8d %8d %9d %9d %9d %9d %6d\n", g, r.total, r.skipped, r.okPass, r.okFail, r.falseReject, r.falseAccept, r.panics)
		sum.total += r.total
		sum.skipped += r.skipped
		sum.okPass += r.okPass
		sum.okFail += r.okFail
		sum.falseReject += r.falseReject
		sum.falseAccept += r.falseAccept
		sum.panics += r.panics
	}
	fmt.Fprintf(&sb, "%-12s %8d %8d %9d %9d %9d %9d %6d\n", "TOTAL", sum.total, sum.skipped, sum.okPass, sum.okFail, sum.falseReject, sum.falseAccept, sum.panics)
	return sb.String()
}
