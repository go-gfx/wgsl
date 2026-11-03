package wgsl

import (
	"os"
	"path/filepath"
	"testing"
)

// invalid-by-design shaders of the corpus (see corpus_test.go); they must be
// rejected, by the parser or the checker.
var corpusInvalid = map[string]bool{
	"examples__batchcompute__main.go.322.wgsl": true,
	"examples__batchcompute__main.go.323.wgsl": true,
	"gpubatch__module_test.go.619.wgsl":        true,
	"batch__device_js_test.go.247.wgsl":        true,
	"batch__device_js_test.go.249.wgsl":        true,
	// the host test calls a function that does not exist, on purpose
	"batch__device_js_test.go.250.wgsl": true,
}

// corpusExcluded lists samples that only fail to type-check because of
// text the host program splices in at run time.
var corpusExcluded = map[string]string{
	"render__gpuworld__special.go.125.wgsl": "assembled: `array<i32, 256>(0)` stands for a table the Go code generates",
	"render__gpuworld__alias.go.194.wgsl":   "uses the table `anorms`, declared by Go code outside the sample",
}

func TestCheckRealShaders(t *testing.T) {
	files, err := filepath.Glob("testdata/real/*/*.wgsl")
	if err != nil || len(files) < 30 {
		t.Fatalf("corpus not found: %v (%d files)", err, len(files))
	}
	ok := 0
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		base := filepath.Base(f)
		verr := Validate(string(b), Options{Name: base})
		if corpusInvalid[base] {
			if verr == nil {
				t.Errorf("%s: invalid-by-design shader was accepted", base)
			}
			continue
		}
		if why, skip := corpusExcluded[base]; skip {
			t.Logf("%s excluded: %s", base, why)
			continue
		}
		if verr != nil {
			t.Errorf("%s: %v", base, verr)
			continue
		}
		ok++
	}
	t.Logf("%d valid shaders check with zero errors", ok)
}
