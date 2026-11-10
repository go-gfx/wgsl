//go:build windows

package hlsltest

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"

	"github.com/go-gfx/wgsl"
	"github.com/go-gfx/wgsl/hlsl"
	"github.com/go-mswin/d3d11"
	"github.com/go-mswin/d3d11/d3dcompiler"
	"golang.org/x/sys/windows"
)

// compileFlags are what a consumer compiles the emitted HLSL with
// (docs/HLSL.md, rule 1).
const compileFlags = d3dcompiler.FlagIEEEStrictness | d3dcompiler.FlagOptimizationLevel3 | d3dcompiler.FlagWarningsAreErrors

// buf is one buffer of a dispatch, bound at its index in the list: the emitted
// HLSL says whether that index is a t# (read), u# (read_write) or b# (uniform)
// register.
type buf struct {
	data    []byte
	uniform bool
	stride  uint32 // element size of a structured buffer
}

func storage(b []byte) buf              { return buf{data: b, stride: 4} }
func structured(b []byte, n uint32) buf { return buf{data: b, stride: n} }
func uniform(b []byte) buf              { return buf{data: b, uniform: true} }

var shared *d3d11.Device

// device opens the WARP device once; the process keeps it. WARP is part of
// every Windows 10 and later, so failing to create it fails the test.
func device(t *testing.T) *d3d11.Device {
	t.Helper()
	if shared != nil {
		return shared
	}
	d, err := d3d11.CreateDevice(d3d11.Options{Software: true})
	if err != nil {
		t.Fatalf("WARP device: %v", err)
	}
	t.Logf("adapter %q, feature level %s", d.Adapter().Description, d.FeatureLevel())
	shared = d
	return d
}

func TestMain(m *testing.M) {
	code := m.Run()
	if shared != nil {
		shared.Close()
	}
	os.Exit(code)
}

// emitHLSL parses, checks and emits src's entry point.
func emitHLSL(t *testing.T, src, entry string) string {
	t.Helper()
	m, err := wgsl.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	out, err := hlsl.Module(m, info, entry)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func compile(t *testing.T, src, entry string) []byte {
	t.Helper()
	dxbc, err := d3dcompiler.Compile([]byte(src), "emitted.hlsl", entry, "cs_5_0", compileFlags)
	if err != nil {
		t.Fatalf("D3DCompile of the emitted HLSL: %v\n%s", err, src)
	}
	return dxbc
}

// runModule emits entry of src to HLSL, compiles it and dispatches it over
// groups thread groups with bufs bound at register 0, 1, ...; it returns every
// buffer's bytes afterwards.
func runModule(t *testing.T, src, entry string, bufs []buf, groups [3]uint32) [][]byte {
	t.Helper()
	return runHLSL(t, emitHLSL(t, src, entry), entry, bufs, groups)
}

func runHLSL(t *testing.T, text, entry string, bufs []buf, groups [3]uint32) [][]byte {
	t.Helper()
	dev := device(t)
	sh, err := dev.CreateComputeShader(compile(t, text, entry))
	if err != nil {
		t.Fatal(err)
	}
	defer sh.Close()
	ctx := dev.Context()
	if err := ctx.SetShader(sh); err != nil {
		t.Fatal(err)
	}
	out := make([][]byte, len(bufs))
	var created []*d3d11.Buffer
	var closers []interface{ Close() }
	defer func() {
		for _, c := range closers {
			c.Close()
		}
		for _, b := range created {
			b.Close()
		}
	}()
	rw := make([]*d3d11.Buffer, len(bufs))
	for i, b := range bufs {
		n := uint64(max(len(b.data), 16))
		if b.uniform {
			n = (n + 15) / 16 * 16
			cb, err := dev.CreateBuffer(d3d11.BufferDesc{Kind: d3d11.KindConstant, Size: n, Initial: pad(b.data, n)})
			if err != nil {
				t.Fatal(err)
			}
			created = append(created, cb)
			if err := ctx.SetConstantBuffers(uint32(i), cb); err != nil {
				t.Fatal(err)
			}
			continue
		}
		stride := uint64(b.stride)
		n = (n + stride - 1) / stride * stride
		isUAV := strings.Contains(text, fmt.Sprintf("register(u%d)", i))
		isSRV := strings.Contains(text, fmt.Sprintf("register(t%d)", i))
		if !isUAV && !isSRV {
			t.Fatalf("buffer %d is bound to no register of the emitted HLSL", i)
		}
		sb, err := dev.CreateBuffer(d3d11.BufferDesc{Kind: d3d11.KindStructured, Size: n, Stride: uint32(stride), SRV: isSRV, UAV: isUAV, Initial: pad(b.data, n)})
		if err != nil {
			t.Fatal(err)
		}
		created = append(created, sb)
		if isUAV {
			v, err := sb.UAV()
			if err != nil {
				t.Fatal(err)
			}
			closers = append(closers, v)
			if err := ctx.SetUAVs(uint32(i), v); err != nil {
				t.Fatal(err)
			}
			rw[i] = sb
		} else {
			v, err := sb.SRV()
			if err != nil {
				t.Fatal(err)
			}
			closers = append(closers, v)
			if err := ctx.SetSRVs(uint32(i), v); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := ctx.Dispatch(groups[0], groups[1], groups[2]); err != nil {
		t.Fatal(err)
	}
	for i, b := range bufs {
		if rw[i] == nil {
			out[i] = append([]byte(nil), b.data...)
			continue
		}
		all, err := ctx.Readback(rw[i], 60*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		out[i] = all[:len(b.data)]
	}
	return out
}

func pad(b []byte, n uint64) []byte {
	out := make([]byte, n)
	copy(out, b)
	return out
}

// disassemble is D3DDisassemble of dxbc, to look for what the compiler made of
// the HLSL (a mad is a fused multiply-add).
func disassemble(t *testing.T, dxbc []byte) string {
	t.Helper()
	proc := windows.NewLazySystemDLL(d3dcompiler.Version).NewProc("D3DDisassemble")
	if err := proc.Find(); err != nil {
		t.Fatal(err)
	}
	var blob unsafe.Pointer
	r, _, _ := proc.Call(uintptr(unsafe.Pointer(&dxbc[0])), uintptr(len(dxbc)), 0, 0, uintptr(unsafe.Pointer(&blob)))
	if int32(uint32(r)) < 0 || blob == nil {
		t.Fatalf("D3DDisassemble: %#x", uint32(r))
	}
	// ID3DBlob: IUnknown QueryInterface(0) AddRef(1) Release(2), GetBufferPointer(3), GetBufferSize(4).
	slot := func(n int) uintptr { return (*(**[8]uintptr)(blob))[n] }
	p, _, _ := syscall.SyscallN(slot(3), uintptr(blob))
	n, _, _ := syscall.SyscallN(slot(4), uintptr(blob))
	text := string(unsafe.Slice(*(**byte)(unsafe.Pointer(&p)), int(n)))
	syscall.SyscallN(slot(2), uintptr(blob))
	return text
}

// groupsFor is the thread-group count for n invocations of a 1-D group of size wg.
func groupsFor(n, wg int) [3]uint32 { return [3]uint32{uint32((n + wg - 1) / wg), 1, 1} }

func u32s(v ...uint32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}

func i32s(v ...int32) []byte {
	u := make([]uint32, len(v))
	for i, x := range v {
		u[i] = uint32(x)
	}
	return u32s(u...)
}

func f32bits(v ...float32) []byte {
	u := make([]uint32, len(v))
	for i, x := range v {
		u[i] = math.Float32bits(x)
	}
	return u32s(u...)
}

func word(b []byte, i int) uint32 { return binary.LittleEndian.Uint32(b[4*i:]) }
