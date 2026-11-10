// Package vulkantest runs SPIR-V emitted by package spirv on a Vulkan
// implementation (lavapipe in CI, RADV on fractal) through go-gfx/vulkan, and
// holds every result to WGSL's definition computed in Go: the integer cases
// SPIR-V leaves undefined (a zero divisor, MIN / -1, shift counts of 32 and
// more), the conversions of f32 to integer at and past the range, and the
// float functions. It is a module of its own so that go-gfx/wgsl does not
// depend on go-gfx/vulkan, as msl/metaltest is for Metal.
//
//	VULKAN_REQUIRE=1     a missing loader or device is a failure, not a skip
//	VULKAN_VALIDATION=1  enable the Khronos validation layer
package vulkantest

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"testing"
	"time"

	"github.com/go-gfx/vulkan"
	"github.com/go-gfx/wgsl"
	"github.com/go-gfx/wgsl/spirv"
)

// buf is one buffer of a dispatch, bound at its index in the list.
type buf struct {
	data    []byte
	uniform bool
}

func storage(b []byte) buf { return buf{data: b} }
func uniform(b []byte) buf { return buf{data: b, uniform: true} }

var (
	devOnce  = map[*testing.T]*vulkan.Device{}
	shared   *vulkan.Device
	sharedIn *vulkan.Instance
)

// TestMain-free sharing: the first test opens the device and the process keeps it.
func device(t *testing.T) *vulkan.Device {
	t.Helper()
	if shared != nil {
		return shared
	}
	require := os.Getenv("VULKAN_REQUIRE") == "1"
	skip := func(err error) {
		t.Helper()
		if require {
			t.Fatal(err)
		}
		t.Skip(err)
	}
	in, err := vulkan.Open(vulkan.Options{ApplicationName: "wgsl spirv tests", Validation: os.Getenv("VULKAN_VALIDATION") == "1"})
	if err != nil {
		skip(err)
	}
	pd, err := in.Pick(vulkan.Filter{})
	if err != nil {
		in.Close()
		skip(err)
	}
	i := pd.Info
	t.Logf("device %q type=%v vendor=%#x api=%v driver=%q %q", i.Name, i.Type, i.VendorID, i.APIVersion, i.DriverName, i.DriverInfo)
	dev, err := pd.Open()
	if err != nil {
		t.Fatal(err)
	}
	shared, sharedIn = dev, in
	return dev
}

// runModule emits entry of src to SPIR-V and dispatches it over groups
// workgroups with bufs bound at binding 0, 1, ...; it returns every buffer's
// bytes afterwards.
func runModule(t *testing.T, src, entry string, bufs []buf, groups [3]uint32) [][]byte {
	t.Helper()
	dev := device(t)
	m, err := wgsl.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	words, err := spirv.Module(m, info, spirv.Options{}, entry)
	if err != nil {
		t.Fatal(err)
	}
	sh, err := dev.NewShader(words)
	if err != nil {
		t.Fatal(err)
	}
	defer sh.Close()
	var bindings []vulkan.Binding
	for i, b := range bufs {
		k := vulkan.StorageBuffer
		if b.uniform {
			k = vulkan.UniformBuffer
		}
		bindings = append(bindings, vulkan.Binding{Binding: uint32(i), Kind: k})
	}
	pipe, err := dev.NewComputePipeline(sh, entry, bindings)
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	vb := make([]*vulkan.Buffer, len(bufs))
	for i, b := range bufs {
		u := vulkan.UsageStorage
		if b.uniform {
			u = vulkan.UsageUniform
		}
		n := max(len(b.data), 16)
		if vb[i], err = dev.NewBuffer(uint64(n), u); err != nil {
			t.Fatal(err)
		}
		defer vb[i].Close()
		copy(vb[i].Bytes(), b.data)
		if err := pipe.Bind(uint32(i), vb[i]); err != nil {
			t.Fatal(err)
		}
	}
	cb, err := dev.NewCommandBuffer()
	if err != nil {
		t.Fatal(err)
	}
	defer cb.Close()
	if err := cb.Begin(); err != nil {
		t.Fatal(err)
	}
	if err := cb.Dispatch(pipe, groups[0], groups[1], groups[2]); err != nil {
		t.Fatal(err)
	}
	if err := cb.End(); err != nil {
		t.Fatal(err)
	}
	fence, err := dev.Submit(cb)
	if err != nil {
		t.Fatal(err)
	}
	defer fence.Close()
	if ok, err := fence.Wait(60 * time.Second); err != nil || !ok {
		t.Fatalf("fence: %v %v", ok, err)
	}
	out := make([][]byte, len(bufs))
	for i, b := range vb {
		out[i] = append([]byte(nil), b.Bytes()[:len(bufs[i].data)]...)
	}
	return out
}

// groupsFor is the workgroup count for n invocations of a 1-D workgroup of size wg.
func groupsFor(n, wg int) [3]uint32 { return [3]uint32{uint32((n + wg - 1) / wg), 1, 1} }

var errSkip = errors.New("skip")

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

func TestMain(m *testing.M) {
	code := m.Run()
	if shared != nil {
		shared.Close()
		sharedIn.Close()
	}
	os.Exit(code)
}
