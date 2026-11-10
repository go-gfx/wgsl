# Emitting SPIR-V: the first subset (W1)

Status: implemented (package `spirv`, since v0.7.0; complete for gpu2d's whole
kernel since W2), checked structurally on every platform, validated by
`spirv-val` in CI, and run on lavapipe through go-gfx/vulkan (`spirv/vulkantest`).
RADV on fractal is pending.

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

**Added by W2** (all of gpu2d's `kernel.wgsl`, which is emitted verbatim and
validated by `spirv-val`, see "W2"): signed `/` and `%`, `f32` to `u32` and
`i32`, the built-ins `abs`, `min`, `max`, `clamp`, `length`, `sqrt` and
`floor`, whole structures of scalars and vectors as the elements of a storage
array, read into a `let`, passed to functions and read member by member, and
`&&`/`||` with built-ins or constructors on the right.

**Still refused:** float `%`, comparisons of vectors, vector `/` and `%` on
`i32`, `f32` vector to integer vector conversions, every other built-in
function, structures written to a buffer, structure `var`s, nested structures,
`switch`, `continuing`, `discard`, `override`, other stages and builtins.

## W2: the rules added, with their sources

- **Signed `/` and `%`** (`wgsl/index.bs`, "Arithmetic Expressions"; the same
  rules as docs/MSL.md 5): a zero divisor gives `e1` for `/` and 0 for `%`; the
  most negative `i32` over -1 gives `e1` and 0. `OpSDiv` and `OpSRem` leave
  those cases undefined, so the divisor is replaced by 1 when
  `e2 == 0 || (e1 == MIN && e2 == -1)`: `e1 / 1` is `e1` and `e1 % 1` is 0.
  `OpSDiv` truncates toward zero and `OpSRem` takes the sign of the dividend,
  as WGSL does.
- **`f32` to integer** (docs/MSL.md rule 6): `OpConvertFToU` and
  `OpConvertFToS` are undefined out of range, so the operand is first clamped
  with comparisons and selects to [0, 4294967040] or [-2147483648, 2147483520]
  (the values `f32` can hold nearest the integer limits), and a NaN, which WGSL
  leaves indeterminate, gives 0 (never converted).
- **Built-ins** use the GLSL.std.450 instruction set (`FAbs`, `FMin`, `SMin`,
  `UMin`, `FMax`, `SMax`, `UMax`, `Sqrt`, `Floor`; numbers from SPIRV-Headers'
  `extinst.glsl.std.450.grammar.json`). `clamp` is `min(max(e, low), high)`,
  WGSL's definition. `abs` of an `i32` is `select(e, 0 - e, e < 0)` so that
  `abs(MIN)` is `MIN`. **`length`** is not the GLSL `Length`: it is the sum of
  squares in order, each step `NoContraction`, then `Sqrt`, as docs/MSL.md rule 2
  has it.
- **Structures in buffers**: the element of a storage array is an `OpTypeStruct`
  with the checker's `Offset`s and the array's `ArrayStride`; it is loaded
  member by member (one access chain and one `OpLoad` each) into a plain
  `OpTypeStruct` of the same members, so the decorated type never leaves memory.
- **`&&` and `||`** refuse only a call of the module's own functions on the
  right; built-ins and constructors have no effect to skip.

How W2 is checked: `testdata/kernel_gpu2d.wgsl` (gpu2d's kernel, verbatim) goes
through the structural tests and `spirv-val` on every change; the nested module
`spirv/vulkantest` runs the edge cases of docs/MSL.md as compute dispatches
through go-gfx/vulkan (integer wrap, zero divisors, `MIN / -1`, shift counts,
NaN, signed zeros, infinities, conversion saturation compared with the checker's
constant evaluator, loops, structures in buffers) and measures the rounding of
`sqrt` and `length` against the correctly rounded value. On lavapipe (CI):
every case equals WGSL's definition, and `sqrt` and `length` match the
correctly rounded value on all inputs measured. **RADV (fractal) is pending**:
the host does not answer.

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
