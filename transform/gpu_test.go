package transform

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/go-gfx/wgsl"
)

// The GPU test runs transformed shaders on the machine's real GPU through
// Chrome's WebGPU (Dawn on Metal on a Mac). It is off by default:
//
//	WGSL_GPU_TEST=1 go test -run TestGPU -v ./transform
//
// WGSL_GPU_EXPECT_VENDOR=apple additionally asserts the adapter's vendor.
// WGSL_CHROME overrides the browser binary.

type gpuCase struct {
	Name    string `json:"name"`
	Code    string `json:"code"`
	Budget  uint32 `json:"budget"`
	HasBudg bool   `json:"hasBudget"`
	// Expected results, compared in Go.
	WantOut  uint32 `json:"-"`
	WantFlag uint32 `json:"-"`
}

type gpuResult struct {
	Name    string `json:"name"`
	Out     uint32 `json:"out"`
	Flag    uint32 `json:"flag"`
	Error   string `json:"error"`
	Adapter string `json:"adapter"`
	Vendor  string `json:"vendor"`
}

const gpuPage = `<!doctype html><meta charset=utf-8><title>wgsl gpu test</title><script>
async function main() {
  const cases = await (await fetch('/cases')).json();
  const out = [];
  let adapterDesc = '', vendor = '';
  try {
    const adapter = await navigator.gpu.requestAdapter();
    const info = adapter.info;
    adapterDesc = [info.vendor, info.architecture, info.device, info.description].join('|');
    vendor = info.vendor;
    const dev = await adapter.requestDevice();
    for (const c of cases) {
      const r = {name: c.name, out: 0, flag: 0, error: '', adapter: adapterDesc, vendor};
      try {
        dev.pushErrorScope('validation');
        const mod = dev.createShaderModule({code: c.code});
        const info = await mod.getCompilationInfo();
        const errs = info.messages.filter(m => m.type === 'error').map(m => m.lineNum + ':' + m.linePos + ' ' + m.message);
        if (errs.length) throw new Error('compile: ' + errs.join('; '));
        const mk = (size, usage) => dev.createBuffer({size, usage});
        const outB = mk(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        const flagB = mk(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        const budB = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        dev.queue.writeBuffer(outB, 0, new Uint32Array([0, 0, 0, 0]));
        dev.queue.writeBuffer(flagB, 0, new Uint32Array([0, 0, 0, 0]));
        dev.queue.writeBuffer(budB, 0, new Uint32Array([c.budget, 0, 0, 0]));
        const pipe = dev.createComputePipeline({layout: 'auto', compute: {module: mod, entryPoint: 'main'}});
        const entries = [{binding: 0, resource: {buffer: outB}}];
        if (c.hasBudget) {
          entries.push({binding: 1, resource: {buffer: budB}});
          entries.push({binding: 2, resource: {buffer: flagB}});
        }
        const bg = dev.createBindGroup({layout: pipe.getBindGroupLayout(0), entries});
        const enc = dev.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1); pass.end();
        const rb = mk(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const rf = mk(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        enc.copyBufferToBuffer(outB, 0, rb, 0, 16);
        enc.copyBufferToBuffer(flagB, 0, rf, 0, 16);
        dev.queue.submit([enc.finish()]);
        await rb.mapAsync(GPUMapMode.READ); await rf.mapAsync(GPUMapMode.READ);
        r.out = new Uint32Array(rb.getMappedRange())[0];
        r.flag = new Uint32Array(rf.getMappedRange())[0];
        const e = await dev.popErrorScope();
        if (e) r.error = 'validation: ' + e.message;
      } catch (e) { r.error = String(e); }
      out.push(r);
    }
  } catch (e) { out.push({name: 'setup', error: String(e), adapter: adapterDesc, vendor}); }
  await fetch('/results', {method: 'POST', body: JSON.stringify(out)});
}
main();
</script>`

func chromePath() string {
	if p := os.Getenv("WGSL_CHROME"); p != "" {
		return p
	}
	for _, p := range []string{
		"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
		"/usr/bin/google-chrome", "/usr/bin/chromium",
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func runGPU(t *testing.T, cases []gpuCase) []gpuResult {
	t.Helper()
	chrome := chromePath()
	if chrome == "" {
		t.Skip("no Chrome found (set WGSL_CHROME)")
	}
	done := make(chan []gpuResult, 1)
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		io.WriteString(w, gpuPage)
	})
	mux.HandleFunc("/cases", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(cases)
	})
	mux.HandleFunc("/results", func(w http.ResponseWriter, r *http.Request) {
		var res []gpuResult
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&res); err != nil {
			t.Errorf("results: %v", err)
		}
		done <- res
	})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln)
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, chrome, "--headless=new", "--enable-unsafe-webgpu",
		"--no-first-run", "--no-default-browser-check", "--disable-extensions",
		"--user-data-dir="+t.TempDir(), "http://"+ln.Addr().String()+"/")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { cancel(); cmd.Wait() }()
	select {
	case res := <-done:
		return res
	case <-ctx.Done():
		t.Fatal("timed out waiting for the browser")
	}
	return nil
}

const gpuLoopHeader = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
`

func gpuShaders(t *testing.T, budget uint32) []gpuCase {
	bodies := []struct {
		name, body string
		long       bool
	}{
		{"loop", "var n = 0u; loop { if n >= LIMIT { break; } n++; } out[0] = n;", false},
		{"for", "var n = 0u; for (var i = 0u; i < LIMIT; i++) { n++; } out[0] = n;", false},
		{"while", "var n = 0u; while n < LIMIT { n++; } out[0] = n;", false},
		{"nested", "var n = 0u; for (var i = 0u; i < LIMIT; i++) { for (var j = 0u; j < INNER; j++) { n++; } } out[0] = n;", false},
	}
	cfg := Config{Budget: BudgetSource{Group: 0, Binding: 1}, FlagGroup: 0, FlagBinding: 2}
	var cases []gpuCase
	add := func(name, src string, hasBudget bool, wantOut, wantFlag uint32) {
		code := src
		if hasBudget {
			m, err := wgsl.Parse(src)
			if err != nil {
				t.Fatal(err)
			}
			out, rep, err := LoopBudget(m, cfg)
			if err != nil || !rep.Instrumented {
				t.Fatalf("%s: %v %+v", name, err, rep)
			}
			code = wgsl.Print(out)
			if err := wgsl.Validate(code); err != nil {
				t.Fatalf("%s: transformed shader does not validate: %v", name, err)
			}
		}
		cases = append(cases, gpuCase{Name: name, Code: code, Budget: budget, HasBudg: hasBudget, WantOut: wantOut, WantFlag: wantFlag})
	}
	for _, b := range bodies {
		long := gpuLoopHeader + strings.ReplaceAll(strings.ReplaceAll(b.body, "LIMIT", "100000000u"), "INNER", "1000u") + "}"
		short := gpuLoopHeader + strings.ReplaceAll(strings.ReplaceAll(b.body, "LIMIT", "10u"), "INNER", "50u") + "}"
		// The check runs at the top of an iteration: `budget` iterations
		// execute, the next one stops.
		wantLong := budget
		if b.name == "nested" {
			// the budget is shared by the whole nest, the outer iteration
			// counts too: 1 outer + 999 inner iterations run their body
			wantLong = budget - 1
		}
		add(b.name+"/long/transformed", long, true, wantLong, 1)
		wantShort := uint32(10)
		if b.name == "nested" {
			wantShort = 10 * 50 // 10 outer + 500 inner iterations: within the budget
		}
		add(b.name+"/short/transformed", short, true, wantShort, 0)
		add(b.name+"/short/original", short, false, wantShort, 0)
	}
	// A workgroup barrier after a transformed loop: the browser's uniformity
	// analysis must still accept the shader (the counter is a local).
	barrier := `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
var<workgroup> w: array<u32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) {
    var s = 32u;
    loop {
        if s == 0u { break; }
        if li < s { w[li] = w[li] + 1u; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if li == 0u { out[0] = w[0]; }
}`
	add("barrier/transformed", barrier, true, 6, 0)
	return cases
}

func TestGPULoopBudget(t *testing.T) {
	if os.Getenv("WGSL_GPU_TEST") == "" {
		t.Skip("set WGSL_GPU_TEST=1 to run the browser GPU test")
	}
	const budget = 1000
	cases := gpuShaders(t, budget)
	res := runGPU(t, cases)
	if len(res) == 0 {
		t.Fatal("no results")
	}
	if res[0].Name == "setup" {
		t.Fatalf("browser setup failed: %s", res[0].Error)
	}
	t.Logf("adapter: %s", res[0].Adapter)
	if want := os.Getenv("WGSL_GPU_EXPECT_VENDOR"); want != "" && !strings.EqualFold(res[0].Vendor, want) {
		t.Fatalf("adapter vendor %q, want %q", res[0].Vendor, want)
	}
	byName := map[string]gpuResult{}
	for _, r := range res {
		byName[r.Name] = r
	}
	for _, c := range cases {
		r, ok := byName[c.Name]
		switch {
		case !ok:
			t.Errorf("%s: no result", c.Name)
		case r.Error != "":
			t.Errorf("%s: %s", c.Name, r.Error)
		case r.Out != c.WantOut || r.Flag != c.WantFlag:
			t.Errorf("%s: out=%d flag=%d, want out=%d flag=%d", c.Name, r.Out, r.Flag, c.WantOut, c.WantFlag)
		default:
			t.Log(fmt.Sprintf("%-26s out=%-6d flag=%d", c.Name, r.Out, r.Flag))
		}
	}
}
