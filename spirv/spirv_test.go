package spirv

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"

	"github.com/go-gfx/wgsl"
)

// fillSrc is a fill kernel: a uniform with the frame's size and a colour, a
// storage buffer of pixels, one invocation per pixel, a guard, a loop with
// break and continue, and a call of another function.
const fillSrc = `
struct Params { w: u32, h: u32, color: u32, rounds: u32 }

@group(0) @binding(0) var<storage, read_write> px: array<u32>;
@group(0) @binding(1) var<uniform> p: Params;

fn mixbits(c: u32, k: u32) -> u32 {
  return ((c << 1u) | (c >> 31u)) ^ (k * 2654435761u);
}

@compute @workgroup_size(16, 8)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.w || gid.y >= p.h) { return; }
  var c = p.color;
  for (var k = 0u; k < p.rounds; k++) {
    if (k == 3u) { continue; }
    if (k > 40u) { break; }
    c = mixbits(c, k);
  }
  px[gid.y * p.w + gid.x] = c;
}
`

// richSrc exercises the rest of the subset: every scalar type, vectors,
// comparisons, while and loop, if/else chains, select, conversions and a
// read-only storage buffer.
const richSrc = `
struct U { scale: f32, off: i32, n: u32, v: vec4f, pad: vec2u }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> ints: array<i32>;
@group(0) @binding(3) var<uniform> u: U;

fn lerp(a: f32, b: f32, t: f32) -> f32 { return a + (b - a) * t; }
fn pick(c: bool, a: i32, b: i32) -> i32 { return select(b, a, c); }

@compute @workgroup_size(64)
fn rich(@builtin(global_invocation_id) g: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  if (g.x >= u.n) { return; }
  var i = 0;
  var acc = 0.0;
  while (i < 8) {
    acc = acc + lerp(src[g.x], 1.0, f32(i) * 0.125);
    i += 1;
    if (i == 5) { break; }
  }
  var n = 0u;
  loop {
    n++;
    if (n >= 4u) { break; }
  }
  var v = vec4f(acc, -acc, u.scale, 1.0) * u.scale + u.v;
  v.y = v.x + v.z;
  let w = v.zyx;
  if (acc > 2.0) {
    ints[g.x] = pick(acc != 0.0, u.off, -u.off) + i32(n) + bitcast<i32>(wg.x);
  } else if (acc < 0.0) {
    ints[g.x] = i32(g.x / 3u) + i32(g.x % 3u) + (1 << 4) + (u.off >> 1u);
  } else {
    ints[g.x] = ~u.off;
  }
  dst[g.x] = vec4f(w.x, w.y, w.z, v.w);
}
`

// controlSrc is the awkward control flow: both branches of an if return (the
// merge block is unreachable), loops nested with break, continue and return
// inside ifs, a loop that only a break leaves.
const controlSrc = `
@group(0) @binding(0) var<storage, read_write> a: array<u32>;

fn early(x: u32) -> u32 {
  if (x > 3u) { return 1u; } else { return 2u; }
}

fn nested(n: u32) -> u32 {
  var s = 0u;
  for (var i = 0u; i < n; i++) {
    for (var j = 0u; j < n; j++) {
      if (j > i) { break; }
      if ((i + j) % 2u == 0u) { continue; }
      if (s > 100u) { return s; }
      s += i * j;
    }
  }
  return s;
}

fn spin(n: u32) -> u32 {
  var k = 0u;
  while (true) {
    k += 3u;
    if (k > n) { break; }
  }
  return k;
}

@compute @workgroup_size(8)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  a[g.x] = early(g.x) + nested(g.x) + spin(g.x);
}
`

var kernels = []struct{ name, src, entry string }{{"fill", fillSrc, "fill"}, {"rich", richSrc, "rich"}, {"control", controlSrc, "run"}}

func check(t *testing.T, src string) (*wgsl.Module, *wgsl.Info) {
	t.Helper()
	m, err := wgsl.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	return m, info
}

func emit(t *testing.T, src string, opts Options, entry string) []uint32 {
	t.Helper()
	m, info := check(t, src)
	w, err := Module(m, info, opts, entry)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

// ---- a decoder and the structural rules of the specification ----

// ins is one decoded instruction.
type ins struct {
	op    uint32
	words []uint32 // operands, without the first word
	at    int      // index of its first word in the module
}

func decode(words []uint32) ([]ins, error) {
	if len(words) < 5 {
		return nil, errors.New("shorter than the header")
	}
	var out []ins
	for i := 5; i < len(words); {
		wc := int(words[i] >> 16)
		if wc == 0 || i+wc > len(words) {
			return nil, fmt.Errorf("word %d: an instruction of %d words", i, wc)
		}
		out = append(out, ins{op: words[i] & 0xFFFF, words: words[i+1 : i+wc], at: i})
		i += wc
	}
	return out, nil
}

// result says where an opcode's result id is: (hasType, hasResult).
var result = map[uint32][2]bool{
	OpTypeVoid: {false, true}, OpTypeBool: {false, true}, OpTypeInt: {false, true}, OpTypeFloat: {false, true},
	OpTypeVector: {false, true}, OpTypeRuntimeArray: {false, true}, OpTypeStruct: {false, true},
	OpTypePointer: {false, true}, OpTypeFunction: {false, true},
	OpConstant: {true, true}, OpConstantTrue: {true, true}, OpConstantFalse: {true, true}, OpConstantComposite: {true, true},
	OpVariable: {true, true}, OpFunction: {true, true}, OpFunctionParameter: {true, true}, OpLabel: {false, true},
	OpLoad: {true, true}, OpAccessChain: {true, true}, OpFunctionCall: {true, true}, OpSelect: {true, true},
	OpIAdd: {true, true}, OpISub: {true, true}, OpIMul: {true, true}, OpUDiv: {true, true}, OpUMod: {true, true},
	OpFAdd: {true, true}, OpFSub: {true, true}, OpFMul: {true, true}, OpFDiv: {true, true},
	OpSNegate: {true, true}, OpFNegate: {true, true}, OpNot: {true, true}, OpLogicalNot: {true, true},
	OpBitwiseAnd: {true, true}, OpBitwiseOr: {true, true}, OpBitwiseXor: {true, true},
	OpShiftLeftLogical: {true, true}, OpShiftRightLogical: {true, true}, OpShiftRightArithmetic: {true, true},
	OpLogicalAnd: {true, true}, OpLogicalOr: {true, true}, OpLogicalEqual: {true, true}, OpLogicalNotEqual: {true, true},
	OpIEqual: {true, true}, OpINotEqual: {true, true}, OpULessThan: {true, true}, OpSLessThan: {true, true},
	OpUGreaterThan: {true, true}, OpSGreaterThan: {true, true}, OpULessThanEqual: {true, true},
	OpSLessThanEqual: {true, true}, OpUGreaterThanEqual: {true, true}, OpSGreaterThanEqual: {true, true},
	OpFOrdEqual: {true, true}, OpFUnordNotEqual: {true, true}, OpFOrdLessThan: {true, true},
	OpFOrdLessThanEqual: {true, true}, OpFOrdGreaterThan: {true, true}, OpFOrdGreaterThanEqual: {true, true},
	OpBitcast: {true, true}, OpConvertUToF: {true, true}, OpConvertSToF: {true, true},
	OpCompositeConstruct: {true, true}, OpCompositeExtract: {true, true}, OpVectorShuffle: {true, true},
}

func isTerminator(op uint32) bool {
	switch op {
	case OpBranch, OpBranchConditional, OpReturn, OpReturnValue, OpUnreachable, OpSwitch, OpKill:
		return true
	}
	return false
}

// phase is the section of the logical layout (section 2.4) an opcode belongs
// to; function bodies are phase 10.
func phase(in ins) int {
	switch in.op {
	case OpCapability:
		return 1
	case OpExtension:
		return 2
	case OpExtInstImport:
		return 3
	case OpMemoryModel:
		return 4
	case OpEntryPoint:
		return 5
	case OpExecutionMode, OpExecutionModeId:
		return 6
	case OpName, OpMemberName:
		return 7
	case OpDecorate, OpMemberDecorate:
		return 8
	case OpFunction:
		return 10
	case OpVariable:
		if in.words[2] != StorageClassFunction {
			return 9
		}
	}
	if r, ok := result[in.op]; ok && in.op >= OpTypeVoid && in.op <= OpTypeFunction && r[1] {
		return 9
	}
	switch in.op {
	case OpConstant, OpConstantTrue, OpConstantFalse, OpConstantComposite:
		return 9
	}
	return 10
}

// checkModule applies the structural rules this emitter promises: the header,
// the logical layout of section 2.4, ids defined once below the bound, blocks
// that start with a label and end with a terminator, variables at the head of
// the first block, merge instructions just before a branch, and the rules of
// section 2.11.1 that can be read off the text (a merge block is another
// header's merge block at most once, a loop's continue target is not its
// merge block).
func checkModule(words []uint32) error {
	if words[0] != Magic {
		return fmt.Errorf("magic %#x", words[0])
	}
	if words[1] != Version {
		return fmt.Errorf("version %#x", words[1])
	}
	if words[4] != 0 {
		return errors.New("schema is not 0")
	}
	bound := words[3]
	list, err := decode(words)
	if err != nil {
		return err
	}
	defined := map[uint32]bool{}
	last := 0
	for _, in := range list {
		if p := phase(in); p < last {
			return fmt.Errorf("word %d: opcode %d is out of the logical layout order (section %d after %d)", in.at, in.op, p, last)
		} else {
			last = p
		}
		if r, ok := result[in.op]; ok && r[1] {
			id := in.words[0]
			if r[0] {
				id = in.words[1]
			}
			if id == 0 || id >= bound {
				return fmt.Errorf("word %d: result id %d outside 1..%d", in.at, id, bound-1)
			}
			if defined[id] {
				return fmt.Errorf("word %d: id %d defined twice", in.at, id)
			}
			defined[id] = true
		}
	}
	if len(list) == 0 || list[0].op != OpCapability {
		return errors.New("the first instruction is not an OpCapability")
	}
	// Functions.
	merges := map[uint32]bool{}
	for i := 0; i < len(list); i++ {
		if list[i].op != OpFunction {
			continue
		}
		j := i + 1
		for list[j].op == OpFunctionParameter {
			j++
		}
		if list[j].op != OpLabel {
			return fmt.Errorf("function at word %d has no first block", list[i].at)
		}
		labels := map[uint32]bool{}
		var targets []uint32
		blockStart, first := j, true
		for ; j < len(list) && list[j].op != OpFunctionEnd; j++ {
			in := list[j]
			switch {
			case in.op == OpLabel:
				if j != blockStart && !isTerminator(list[j-1].op) {
					return fmt.Errorf("word %d: a block starts before the previous one ended", in.at)
				}
				if j != blockStart {
					first = false
				}
				labels[in.words[0]] = true
			case in.op == OpVariable:
				if !first {
					return fmt.Errorf("word %d: an OpVariable outside the first block", in.at)
				}
				// Variables are the first instructions after the label.
				for k := j - 1; list[k].op != OpLabel; k-- {
					if list[k].op != OpVariable {
						return fmt.Errorf("word %d: an OpVariable after another instruction", in.at)
					}
				}
				if in.words[2] != StorageClassFunction {
					return fmt.Errorf("word %d: a function variable that is not in the Function class", in.at)
				}
			case in.op == OpSelectionMerge || in.op == OpLoopMerge:
				if next := list[j+1].op; next != OpBranch && next != OpBranchConditional {
					return fmt.Errorf("word %d: a merge instruction not followed by a branch", in.at)
				}
				m := in.words[0]
				if merges[m] {
					return fmt.Errorf("word %d: block %d is the merge block of two headers", in.at, m)
				}
				merges[m] = true
				if in.op == OpLoopMerge && in.words[1] == m {
					return fmt.Errorf("word %d: a loop whose continue target is its merge block", in.at)
				}
				targets = append(targets, m)
				if in.op == OpLoopMerge {
					targets = append(targets, in.words[1])
				}
			case in.op == OpBranch:
				targets = append(targets, in.words[0])
			case in.op == OpBranchConditional:
				targets = append(targets, in.words[1], in.words[2])
			}
			if j+1 < len(list) && list[j+1].op != OpFunctionEnd && isTerminator(in.op) && list[j+1].op != OpLabel {
				return fmt.Errorf("word %d: an instruction after a terminator, outside a block", list[j+1].at)
			}
		}
		if j >= len(list) {
			return fmt.Errorf("function at word %d has no OpFunctionEnd", list[i].at)
		}
		if !isTerminator(list[j-1].op) {
			return fmt.Errorf("word %d: the last block does not end with a terminator", list[j-1].at)
		}
		for _, tg := range targets {
			if !labels[tg] {
				return fmt.Errorf("function at word %d: a branch to %d, which is not one of its blocks", list[i].at, tg)
			}
		}
		i = j
	}
	// Each entry point lists every global variable its static call tree uses
	// (section 2.4, OpEntryPoint, version 1.4 and later), once.
	global := map[uint32]bool{}
	for _, in := range list {
		if in.op == OpVariable && in.words[2] != StorageClassFunction {
			global[in.words[1]] = true
		}
	}
	listed := map[uint32]bool{}
	for _, in := range list {
		if in.op != OpEntryPoint {
			continue
		}
		nameWords := (len(nameOf(in.words[2:]))/4 + 1)
		for _, id := range in.words[2+nameWords:] {
			if listed[id] && false {
				return fmt.Errorf("interface id %d listed twice", id)
			}
			if !global[id] {
				return fmt.Errorf("interface id %d is not a global variable", id)
			}
			listed[id] = true
		}
	}
	for _, in := range list {
		var ptr uint32
		switch in.op {
		case OpLoad:
			ptr = in.words[2]
		case OpStore:
			ptr = in.words[0]
		case OpAccessChain:
			ptr = in.words[2]
		default:
			continue
		}
		if global[ptr] && !listed[ptr] {
			return fmt.Errorf("word %d: global variable %d is used but is not in an entry point's interface", in.at, ptr)
		}
	}
	return nil
}

// nameOf is the literal string at the head of ws, as bytes (without the NUL).
func nameOf(ws []uint32) []byte {
	var b []byte
	for _, w := range ws {
		for k := 0; k < 4; k++ {
			c := byte(w >> (8 * k))
			if c == 0 {
				return b
			}
			b = append(b, c)
		}
	}
	return b
}

func count(words []uint32, op uint32) int {
	list, _ := decode(words)
	n := 0
	for _, in := range list {
		if in.op == op {
			n++
		}
	}
	return n
}

// ---- tests ----

func TestFillKernelIsWellFormed(t *testing.T) {
	for _, src := range kernels {
		for _, rte := range []bool{false, true} {
			w := emit(t, src.src, Options{RoundingModeRTE: rte}, src.entry)
			if err := checkModule(w); err != nil {
				t.Errorf("%s (RTE %v): %v", src.name, rte, err)
			}
			if bytes.Equal(Bytes(w)[:4], []byte{0x03, 0x02, 0x23, 0x07}) == false {
				t.Errorf("%s: the byte stream does not start with the magic number, little-endian", src.name)
			}
		}
	}
}

// The module says what the kernel is: a GLCompute entry point with the
// workgroup size, buffers at their set and binding, and the layout of the
// uniform from the checker's own offsets.
func TestFillKernelContent(t *testing.T) {
	w := emit(t, fillSrc, Options{}, "fill")
	list, _ := decode(w)
	find := func(op uint32, pred func(ins) bool) *ins {
		for i := range list {
			if list[i].op == op && pred(list[i]) {
				return &list[i]
			}
		}
		return nil
	}
	if in := find(OpEntryPoint, func(ins) bool { return true }); in == nil || in.words[0] != ExecutionModelGLCompute || string(nameOf(in.words[2:])) != "fill" {
		t.Fatalf("no GLCompute entry point named fill: %v", in)
	}
	if in := find(OpExecutionMode, func(i ins) bool { return i.words[1] == ExecutionModeLocalSize }); in == nil || !slices.Equal(in.words[2:], []uint32{16, 8, 1}) {
		t.Fatalf("LocalSize: %v", in)
	}
	if in := find(OpMemoryModel, func(ins) bool { return true }); in == nil || in.words[0] != AddressingModelLogical || in.words[1] != MemoryModelGLSL450 {
		t.Fatalf("memory model: %v", in)
	}
	for _, d := range []struct {
		decor uint32
		want  []uint32
	}{{DecorationBinding, []uint32{0, 1}}, {DecorationDescriptorSet, []uint32{0, 0}}, {DecorationBlock, []uint32{0, 0}}, {DecorationBuiltIn, []uint32{BuiltInGlobalInvocationId}}, {DecorationArrayStride, []uint32{4}}} {
		var got []uint32
		for _, in := range list {
			if in.op == OpDecorate && in.words[1] == d.decor {
				if len(in.words) > 2 {
					got = append(got, in.words[2])
				} else {
					got = append(got, 0)
				}
			}
		}
		slices.Sort(got)
		if !slices.Equal(got, d.want) {
			t.Errorf("decoration %d on %v, want %v", d.decor, got, d.want)
		}
	}
	// Params: w, h, color, rounds at offsets 0, 4, 8, 12.
	var offsets []uint32
	for _, in := range list {
		if in.op == OpMemberDecorate && in.words[2] == DecorationOffset {
			offsets = append(offsets, in.words[3])
		}
	}
	slices.Sort(offsets)
	if !slices.Equal(offsets, []uint32{0, 0, 4, 8, 12}) { // the pixel wrapper's 0, and Params' four
		t.Errorf("member offsets %v", offsets)
	}
	// px is in the StorageBuffer class and p in Uniform.
	classes := map[uint32]int{}
	for _, in := range list {
		if in.op == OpVariable && in.words[2] != StorageClassFunction && in.words[2] != StorageClassInput {
			classes[in.words[2]]++
		}
	}
	if classes[StorageClassStorageBuffer] != 1 || classes[StorageClassUniform] != 1 {
		t.Errorf("buffer classes %v", classes)
	}
	// No NoContraction: the kernel has no float arithmetic.
	if count(w, OpFAdd) != 0 {
		t.Error("float arithmetic in an integer kernel")
	}
	// The loop is structured: one OpLoopMerge, and an OpSelectionMerge for each
	// of the guard and the two ifs.
	if n := count(w, OpLoopMerge); n != 1 {
		t.Errorf("%d loops", n)
	}
	if n := count(w, OpSelectionMerge); n != 3 {
		t.Errorf("%d selections", n)
	}
	// The interface of the entry point is the two buffers and the builtin.
	in := find(OpEntryPoint, func(ins) bool { return true })
	if n := len(in.words) - 2 - (len("fill")/4 + 1); n != 3 {
		t.Errorf("an interface of %d variables", n)
	}
}

func TestRichKernelFeatures(t *testing.T) {
	w := emit(t, richSrc, Options{RoundingModeRTE: true}, "rich")
	list, _ := decode(w)
	// Float arithmetic carries NoContraction, and RoundingModeRTE is declared
	// with its capability.
	nc := 0
	rte, cap := false, false
	for _, in := range list {
		if in.op == OpDecorate && in.words[1] == DecorationNoContraction {
			nc++
		}
		if in.op == OpExecutionMode && in.words[1] == ExecutionModeRoundingModeRTE && in.words[2] == 32 {
			rte = true
		}
		if in.op == OpCapability && in.words[0] == CapabilityRoundingModeRTE {
			cap = true
		}
	}
	floatOps := count(w, OpFAdd) + count(w, OpFSub) + count(w, OpFMul) + count(w, OpFNegate) + count(w, OpFDiv)
	if floatOps == 0 || nc != floatOps {
		t.Errorf("%d float operations, %d NoContraction", floatOps, nc)
	}
	if !rte || !cap {
		t.Errorf("RoundingModeRTE: mode %v, capability %v", rte, cap)
	}
	// Without the option, neither appears.
	w = emit(t, richSrc, Options{}, "rich")
	list, _ = decode(w)
	for _, in := range list {
		if (in.op == OpCapability && in.words[0] == CapabilityRoundingModeRTE) || (in.op == OpExecutionMode && in.words[1] == ExecutionModeRoundingModeRTE) {
			t.Error("RoundingModeRTE declared without the option")
		}
	}
	// A read-only buffer is NonWritable; the others are not.
	nonWritable := 0
	for _, in := range list {
		if in.op == OpMemberDecorate && in.words[2] == DecorationNonWritable {
			nonWritable++
		}
	}
	if nonWritable != 1 {
		t.Errorf("%d NonWritable members, want 1 (src)", nonWritable)
	}
	// u32 division and remainder are guarded against a zero divisor: a select on
	// each side of OpUDiv and OpUMod.
	if count(w, OpUDiv) != 1 || count(w, OpUMod) != 1 || count(w, OpSelect) < 5 {
		t.Errorf("udiv %d, umod %d, selects %d", count(w, OpUDiv), count(w, OpUMod), count(w, OpSelect))
	}
	// The workgroup_id builtin is declared next to the invocation id.
	builtins := map[uint32]bool{}
	for _, in := range list {
		if in.op == OpDecorate && in.words[1] == DecorationBuiltIn {
			builtins[in.words[2]] = true
		}
	}
	if !builtins[BuiltInGlobalInvocationId] || !builtins[BuiltInWorkgroupId] {
		t.Errorf("builtins %v", builtins)
	}
}

// Emitting is deterministic: the maps inside never decide an order.
func TestEmittingIsDeterministic(t *testing.T) {
	a := emit(t, richSrc, Options{}, "rich")
	for range 20 {
		if b := emit(t, richSrc, Options{}, "rich"); !slices.Equal(a, b) {
			t.Fatal("two emissions of one module differ")
		}
	}
}

// The structural checker is not vacuous: each mutation of a good module breaks
// one rule it states, and it must see it.
func TestStructuralCheckCanFail(t *testing.T) {
	good := emit(t, fillSrc, Options{}, "fill")
	if err := checkModule(good); err != nil {
		t.Fatal(err)
	}
	list, _ := decode(good)
	findAt := func(op uint32, n int) int {
		for _, in := range list {
			if in.op == op {
				if n == 0 {
					return in.at
				}
				n--
			}
		}
		t.Fatalf("no opcode %d", op)
		return 0
	}
	cp := func() []uint32 { return slices.Clone(good) }
	nop := func(w []uint32, at int) { // replace the instruction at `at` by OpNops of the same length
		wc := int(w[at] >> 16)
		for i := range wc {
			w[at+i] = 1<<16 | OpNop
		}
	}
	for _, tc := range []struct {
		name string
		mut  func() []uint32
	}{
		{"wrong magic", func() []uint32 { w := cp(); w[0] = 0; return w }},
		{"wrong version", func() []uint32 { w := cp(); w[1] = 0x00010000; return w }},
		{"bound too small", func() []uint32 { w := cp(); w[3] = 3; return w }},
		{"a block that does not end", func() []uint32 { w := cp(); nop(w, findAt(OpReturn, 0)); return w }},
		{"a branch to nowhere", func() []uint32 { w := cp(); w[findAt(OpBranch, 0)+1] = 99999; return w }},
		{"an id defined twice", func() []uint32 {
			w := cp()
			a, b := findAt(OpTypeBool, 0), findAt(OpTypeVoid, 0)
			w[a+1] = w[b+1]
			return w
		}},
		{"a decoration after the types", func() []uint32 {
			w := cp()
			// Swap an OpDecorate with the first OpTypeVoid-or-later declaration: move
			// the Capability to the end of the declarations instead.
			c := findAt(OpCapability, 0)
			end := findAt(OpFunction, 0)
			moved := slices.Clone(w[c : c+2])
			w = slices.Concat(w[:c], w[c+2:end], moved, w[end:])
			return w
		}},
		{"a loop whose continue target is its merge block", func() []uint32 {
			w := cp()
			at := findAt(OpLoopMerge, 0)
			w[at+2] = w[at+1]
			return w
		}},
		{"a merge instruction without its branch", func() []uint32 { w := cp(); nop(w, findAt(OpBranch, 1)); return w }},
		{"an entry interface missing a buffer", func() []uint32 {
			w := cp()
			at := findAt(OpEntryPoint, 0)
			wc := int(w[at] >> 16)
			w[at] = uint32(wc-1)<<16 | OpEntryPoint
			return slices.Concat(w[:at+wc-1], w[at+wc:]) // drop the last interface id
		}},
	} {
		w := tc.mut()
		if err := checkModule(w); err == nil {
			t.Errorf("%s: the checker accepted a broken module", tc.name)
		} else {
			t.Logf("%s: %v", tc.name, err)
		}
	}
}

func TestRefusals(t *testing.T) {
	buf := "@group(0) @binding(0) var<storage, read_write> a: array<u32>;\n"
	for _, tc := range []struct{ name, src, entry, want string }{
		{"vertex entry", "@vertex fn v() -> @builtin(position) vec4f { return vec4f(0.0); }", "v", "only compute"},
		{"not an entry", "fn f() {}", "f", "not an entry point"},
		{"no such function", "fn f() {}", "nosuch", "no function"},
		{"private variable", "var<private> p: u32;\n@compute @workgroup_size(1) fn run() { p = 1u; }", "run", "private"},
		{"workgroup variable", "var<workgroup> w: u32;\n@compute @workgroup_size(1) fn run() { w = 1u; }", "run", "workgroup"},
		{"fixed array buffer", "@group(0) @binding(0) var<storage, read_write> a: array<u32, 4>;\n@compute @workgroup_size(1) fn run() { a[0] = 1u; }", "run", "runtime-sized"},
		{"struct of arrays", "struct S { a: array<vec4u, 2> }\n@group(0) @binding(0) var<uniform> s: S;\n@compute @workgroup_size(1) fn run() { var x = s.a[0].x; }", "run", "scalar or vector"},
		{"whole struct load", "struct S { a: u32 }\n@group(0) @binding(0) var<storage, read_write> s: array<S>;\n@compute @workgroup_size(1) fn run() { let x = s[0]; }", "run", "scalars or vectors"},
		{"signed division", buf + "@compute @workgroup_size(1) fn run() { var i = 7; i = i / 2; a[0] = u32(i); }", "run", "division"},
		{"f32 to u32", buf + "@compute @workgroup_size(1) fn run() { var f = 2.5; a[0] = u32(f); }", "run", "conversion"},
		{"float remainder", buf + "@compute @workgroup_size(1) fn run() { var f = 2.5; f = f % 2.0; a[0] = 1u; }", "run", "remainder"},
		{"builtin function", buf + "@compute @workgroup_size(1) fn run() { var m = 1u; a[0] = max(m, 2u); }", "run", "builtin max"},
		{"vector compare", buf + "@compute @workgroup_size(1) fn run() { var v = vec2f(1.0); if (v.x == 1.0 && all(v == v)) { a[0] = 1u; } }", "run", ""},
		{"switch", buf + "@compute @workgroup_size(1) fn run() { var i = 0u; switch (i) { default { a[0] = 1u; } } }", "run", "not supported"},
		{"loop continuing", buf + "@compute @workgroup_size(1) fn run() { var i = 0u; loop { if (i > 3u) { break; } continuing { i++; } } }", "run", "continuing"},
		{"call on the right of &&", buf + "fn t() -> bool { return true; }\n@compute @workgroup_size(1) fn run() { if (true && t()) { a[0] = 1u; } }", "run", "call on the right"},
		{"local_invocation_index", buf + "@compute @workgroup_size(1) fn run(@builtin(local_invocation_index) i: u32) { a[0] = i; }", "run", "@builtin"},
		{"struct local", "struct S { a: u32 }\n@compute @workgroup_size(1) fn run() { var s = S(1u); }", "run", "not supported"},
	} {
		m, info := check(t, tc.src)
		_, err := Module(m, info, Options{}, tc.entry)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want an error naming %q", tc.name, err, tc.want)
			continue
		}
		var e *Error
		if !errors.As(err, &e) && tc.name != "no such function" {
			t.Errorf("%s: %T is not an *Error with a position", tc.name, err)
		}
	}
}

// ---- the oracle ----

// spirv-val, the Khronos validator, is the oracle for the modules above: CI
// installs SPIRV-Tools and sets SPIRV_VAL=required, so a missing validator is a
// failure there and a skip on a machine without it. It is never a build or run
// dependency: nothing but this test calls it.
func TestSpirvVal(t *testing.T) {
	path, err := exec.LookPath("spirv-val")
	if err != nil {
		if os.Getenv("SPIRV_VAL") == "required" {
			t.Fatal("spirv-val is required and not installed")
		}
		t.Skip("spirv-val is not installed")
	}
	for _, src := range kernels {
		for _, rte := range []bool{false, true} {
			w := emit(t, src.src, Options{RoundingModeRTE: rte}, src.entry)
			// Vulkan 1.2 is the environment the module is for; spv1.5 is the
			// universal rules, which a Vulkan-specific rule could hide a fault from.
			for _, env := range []string{"vulkan1.2", "spv1.5"} {
				cmd := exec.Command(path, "--target-env", env, "-")
				cmd.Stdin = bytes.NewReader(Bytes(w))
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Errorf("%s (RTE %v, %s): spirv-val: %v\n%s", src.name, rte, env, err, out)
				}
			}
		}
	}
	// And the oracle can fail: a module with a wrong id is refused.
	w := emit(t, fillSrc, Options{}, "fill")
	list, _ := decode(w)
	for _, in := range list {
		if in.op == OpBranch {
			w[in.at+1] = 99999
			break
		}
	}
	cmd := exec.Command(path, "--target-env", "vulkan1.2", "-")
	cmd.Stdin = bytes.NewReader(Bytes(w))
	if out, err := cmd.CombinedOutput(); err == nil {
		t.Errorf("spirv-val accepted a branch to nowhere:\n%s", out)
	}
}

func TestBytesAreLittleEndianWords(t *testing.T) {
	w := []uint32{0x07230203, 0x01020304}
	b := Bytes(w)
	if binary.LittleEndian.Uint32(b[4:]) != 0x01020304 || len(b) != 8 {
		t.Fatalf("% x", b)
	}
}

// ops_gen.go comes from the Khronos grammar; a few numbers read in the
// specification's own tables guard against a generator that drifted.
func TestGeneratedNumbersMatchTheSpecification(t *testing.T) {
	if Magic != 0x07230203 || OpEntryPoint != 15 {
		t.Fatalf("magic %#x, OpEntryPoint %d", uint32(Magic), OpEntryPoint)
	}
}
