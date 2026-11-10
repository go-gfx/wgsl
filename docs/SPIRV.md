# Emitting SPIR-V: the first subset (W1)

Status: implemented (package `spirv`, since v0.7.0), checked structurally on
every platform and validated by `spirv-val` in CI. Not yet run on a GPU: that is
go-gfx/vulkan's milestone V1, and the edge cases of docs/MSL.md are W2.

It is the second emitter of docs/NATIVE.md (go-gfx/gpu2d), decision 1: the
WGSL raster kernel is compiled to Metal by `msl`, to SPIR-V by this package
(for Vulkan on Linux), and later to HLSL.

## What it emits

`spirv.Module(m, info, opts, entries...)` returns the words of a SPIR-V 1.5
module, the version Vulkan 1.2 takes (`spirv.Bytes` gives the little-endian
byte stream). It emits the named compute entry points and what they reach.

**In scope** (anything else is an `*spirv.Error` with its position):
- **Entry points:** `@compute` only, `@workgroup_size` as the `LocalSize`
  execution mode, `@builtin(global_invocation_id | local_invocation_id |
  workgroup_id)` inputs as `Input` variables.
- **Buffers:** group-0 storage (`read`, `read_write`) and uniform variables
  whose type is a runtime-sized array of scalars or vectors (storage), a
  structure of scalars and vectors, or a scalar or vector. Each is wrapped as
  the Block rule requires, with `Offset` from the checker's own layout,
  `ArrayStride`, `NonWritable` on a read-only one, and the group and binding as
  `DescriptorSet` and `Binding`.
- **Types:** `bool`, `i32`, `u32`, `f32` and their vectors.
- **Statements:** `let`, `var`, assignment (compound too), `++` and `--`, `if`
  with `else if` and `else`, `for`, `while`, `loop` without `continuing`,
  `break`, `continue`, `return`, calls of other functions.
- **Expressions:** the arithmetic, bitwise and shift operators, the
  comparisons of scalars, `&&` and `||` (when the right operand has no call),
  unary `-`, `!` and `~`, member access and swizzles of vectors, indexing,
  `bitcast`, `i32`/`u32` conversions in both directions, `f32(i32|u32)`, vector
  constructors, and the `select` built-in.

**Refused for now (W2):** signed `/` and `%`, `f32` to integer conversion,
float `%`, comparisons of vectors, every other built-in function, whole
structures in buffers, structure locals, `switch`, `continuing`, `discard`,
`override`, other stages and builtins.

**Options:** `RoundingModeRTE` declares the capability and the execution mode
for 32-bit floats, so that `+`, `-`, `*` and `/` are correctly rounded. Every
float arithmetic instruction carries `NoContraction` whether or not it is
set. The caller sets it when the device reports
`shaderRoundingModeRTEFloat32`.

## Rules, with their sources

*SPIR-V Specification*, the unified 1.x document published by Khronos
(registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html), opened at these
sections:

1. **The header (section 2.3, Table 1).** Magic number, the version word
   (`0 | major | minor | 0`, so 1.5 is `0x00010500`), a generator number that
   may be 0, the bound (every id is below it) and a 0 schema word. The
   instruction stream follows, each instruction a word count in the high half
   of its first word and the opcode in the low half.
2. **The logical layout (section 2.4).** Capabilities, then the memory model,
   the entry points, the execution modes, the debug names, the decorations,
   then types, constants and global variables, then the functions, each as
   `OpFunction`, its parameters, its blocks, `OpFunctionEnd`. All
   `OpVariable`s of a function are in its first block and are its first
   instructions; a block starts with `OpLabel` and ends with a terminator. The
   emitter builds each section on its own and concatenates them in this order,
   so a type declared while a function is being emitted still lands before it.
3. **Entry-point interface (`OpEntryPoint`).** From version 1.4 on, the
   interface lists every global variable the entry point's static call tree
   references, once each, in every storage class. The emitter records the
   buffers and `Input` variables each function uses and unions them over the
   calls.
4. **Structured control flow (section 2.11).** An `if` is a header ending in
   `OpSelectionMerge` and `OpBranchConditional`; a loop is a header ending in
   `OpLoopMerge` (merge block, continue target) and a branch, a condition
   block, the body, a continue block that holds the update and branches back
   (the one back edge), and the merge block. The merge block of one header is
   never another's and the continue target is not the merge block
   (section 2.11.1). `break` and `continue` branch to the innermost loop's
   merge block and continue target. What follows a `return`, `break` or
   `continue` in a block is not emitted.
5. **`NoContraction`** applies only to an arithmetic instruction and forbids
   fusing it with another and reassociating: it is how WGSL's separate `*` and
   `+` stay separate, which Dekker-style arithmetic and byte parity with the
   CPU reference depend on.
6. **`RoundingModeRTE`** is an execution mode taking the target width, at most
   once per width on an entry point, and exists from version 1.4.

Rules that are the emitter's own choices, each tested:
- **`u32` division and remainder** follow WGSL (a divisor of 0 gives the
  dividend, and 0), which `OpUDiv` and `OpUMod` leave undefined: the divisor is
  replaced by 1 when it is 0 and the result selected back (docs/MSL.md, rule
  5). Shifts take the count modulo 32.
- **`!=` on floats** is `OpFUnordNotEqual` (true for a NaN, as IEEE 754 says);
  the other comparisons are the ordered ones.
- **`&&` and `||`** are not short-circuit in SPIR-V; they are emitted as
  `OpLogicalAnd` and `OpLogicalOr` only when the right operand has no call,
  where skipping it cannot be seen.

## How it is checked

- **Structural tests**, on every platform, decode the emitted words and check
  the header, the layout order of section 2.4, ids defined once below the
  bound, blocks, variables, merge instructions before their branches, the two
  rules of section 2.11.1 above, and the interface. The same checker is run on
  ten mutated modules, each breaking one rule, and must refuse every one.
- **`spirv-val`**, the Khronos validator, is the oracle in CI: the `spirv-val`
  job installs SPIRV-Tools and the tests then run with `SPIRV_VAL=required`,
  validating every module under `vulkan1.2` and the universal `spv1.5` rules
  and requiring that a deliberately wrong module is refused. It is never a
  build or run dependency: only a test calls it, and a machine without it skips.
- **Determinism:** twenty emissions of one module are equal words.
