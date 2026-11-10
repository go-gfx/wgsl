# Emitting Metal Shading Language: design

Status: implemented (package `msl`; the compute modules below since v0.6.0)
and tested on the GPU (`msl/metaltest`). This note fixes the scope and the rules, with
the sources they come from; every rule is taken from an opened
specification. Two findings came from running the code rather than reading:
- **`bitcast` and `i32`/`u32` conversions:** they are emitted as `as_type`,
  since C++17 makes `uint` to `int` implementation-defined past `INT_MAX`.
- **Subnormals:** the GPU flushes some of them to zero, which WGSL allows for
  any input or output of a floating-point operation (`wgsl/index.bs`, "flush
  to zero"). The test accepts that, and nothing else.

## Why

go-gfx/gpu2d's coverage function, the arithmetic that decides how much of a
pixel a rounded rectangle covers, exists three times:
- in Go (`gpu2d.Coverage`, the CPU reference);
- in WGSL (`webgpu/shader.wgsl`);
- in Metal Shading Language (`metal/kernel.metal`).

Each copy is held to the reference by test, but each fix must be made three
times. An emitter that turns the checked WGSL into MSL removes the third copy:
the Metal kernel includes the emitted functions instead of hand-written ones.

## Scope of the first version

The **functions** of a module, not its entry points, limited to what
gpu2d's `cornerCenter` and `coverage` use:
- **Types:** `i32`, `u32`, `f32`, `bool`, `vec2/3/4` of those, and plain
  structs of scalars passed by value.
- **Statements:** `let`, `var`, assignment, `if`/`else`, `return`.
- **Expressions:** arithmetic, comparison, logical, bitwise, and shifts.
- **Constructors:** scalar conversions and vector constructors.
- **Builtins:** `abs`, `clamp`, `min`, `max`, `length`, `sqrt`, `select`,
  `floor`.
- **Out of scope of `Functions`:** loops, arrays, pointers, textures and
  samplers, entry points, bindings. (Loops are accepted since v0.6.0, and
  `Module` takes entry points and buffers: see "Compute modules".) What is
  still out of scope is refused with the construct and its position, never
  emitted approximately.

The emitter reads a module that `Check` accepted, and its `Info`: the checker
has already resolved every type, so the emitter never infers one.

## Rules, with their sources

Apple, *Metal Shading Language Specification*, 2026-06-04 edition, opened at
these pages:

1. **No fast math, anywhere.**
   - **The default** (§1.6.3, p. 15): it is `fast` math, which may assume no
     NaN and no INF, ignore the sign of zero, use reciprocals, reassociate,
     and contract `a*b+c` into a fused multiply-add across statements.
   - **What WGSL guarantees instead:** none of that, and the CPU reference
     does none of it.
   - **The fix:** every emitted file starts with `#pragma METAL fp
     math_mode(safe)` (§1.6.3, p. 16), which disables those transformations,
     and `#pragma METAL fp contract(off)` (p. 17). Safe mode alone sets
     contraction to `on`, which still fuses within a statement.
2. **Precise library functions.**
   - **The default** (§1.6.3, p. 15): `-fmetal-math-fp32-functions` defaults
     to `fast`.
   - **The fix:** the emitter calls the `precise::` variants where the library
     has them (§6.6), as gpu2d's hand-written kernel already does with
     `precise::sqrt` and `precise::fabs`.
   - **`length`:** emitted as `precise::sqrt` of the explicit sum of squares,
     in the order WGSL's definition gives, not as the geometric function
     (§6.9).
3. **`select` keeps its argument order.**
   - **The definition** (Table 6.3, p. 205): MSL's `select(a, b, c)` is
     `c ? b : a`, the same order as WGSL's `select(f, t, cond)`.
   - **The rule:** emitted as is, never rewritten as `?:`, because the
     component-wise vector case is defined only for `select`.
4. **Signed overflow wraps.**
   - **WGSL** (gpuweb `wgsl/index.bs`, line 2800 at the 2026-11-07 fetch):
     "`i32` operations use two's complement arithmetic and have defined
     overflow behavior".
   - **MSL** (§3.1, p. 112): says nothing of `+`, `-` and `*` overflow, so the
     C++ rule applies, under which signed overflow is undefined.
   - **The rule:** signed `+`, `-` and `*` are emitted through `uint` and cast
     back. Unsigned arithmetic wraps by definition, and the cast back is the
     two's complement reinterpretation.
5. **Integer division and remainder: WGSL defines every case, MSL does
   not.**
   - **WGSL division** (`index.bs` 7064–7094): if `e2` is zero, the result is
     `e1`; for signed `T`, if `e1` is the most negative value and `e2` is −1,
     the result is `e1`; otherwise the quotient is truncated.
   - **WGSL remainder** (7109–7119): signed, the same two cases give 0;
     otherwise `e1 − truncate(e1 ÷ e2) × e2`, so a non-zero result has the sign
     of `e1`.
   - **MSL** (§3.1, pp. 112–113):
     - division by zero, and `TYPE_MIN / −1`, give "an unspecified value";
     - `%` with a zero divisor is undefined, and so is `%` with any negative
       operand.
   - **The rule for `/`:** the specification's own form, `e1 / select(e2, 1,
     e2 == 0 || (e1 == MIN && e2 == -1))`, the divisor guard WGSL's comment
     gives.
   - **The rule for `%`:** never the MSL operator on signed values. With the
     same guarded divisor `d`, it is emitted as `e1 - (e1 / d) * d`, then
     forced to 0 in the two special cases. Unsigned `%` uses MSL's `%` with
     `d = select(e2, 1, e2 == 0)` and gives 0 when `e2` is zero.

## Compute modules (`msl.Module`)

`Module` emits compute entry points as `kernel` functions, so that the whole
raster kernel of go-gfx/gpu2d (`kernel.wgsl`, docs/NATIVE.md decision 1) can be
generated rather than hand-written. What it adds to the scope above:

- **Entry points:** `@compute` only, returning nothing, with
  `@builtin(global_invocation_id)` as its only input (emitted as
  `uint3 name [[thread_position_in_grid]]`). Other builtins, struct IO and the
  other stages are refused. MSL has no static workgroup size: the emitted file
  carries `@workgroup_size` as a comment, and the host dispatches threadgroups
  of that size.
- **Module-scope variables:** storage buffers (`read`, `read_write`) and
  uniform buffers of group 0, nothing else (no `private`, `workgroup`,
  textures, samplers, atomics, overrides, or other groups). The binding is the
  buffer index (at most 30). A runtime-sized storage array is a pointer
  (`device const T *`, or `device T *` for `read_write`); any other buffer is
  a reference (`constant T &` for uniform, `device [const] T &` for storage).
  A function that uses a buffer, itself or through a callee, takes it as a
  trailing parameter in binding order, and its callers pass it.
- **Structures in buffers** must have, in MSL, the offsets and size WGSL's
  memory layout gives them. Members are scalars and vectors; a `vec3` member
  is refused whenever it changes a later offset or the size (MSL's `float3` is
  16 bytes, WGSL's `vec3<f32>` is 12). Anything else is an error that names
  both offsets.
- **Statements:** `for`, `while`, `loop` (without a `continuing` block),
  `break`, `continue`, `++` and `--`, and calls as statements. `switch`,
  `discard`, `break if` and `continuing` are refused. On an `i32`, `++` and
  `--` are emitted through `wgsl_add` and `wgsl_sub` so that they wrap; the
  operand is written twice, so one containing a call is refused.
- **Expressions:** indexing (rule 7), and `u32(f32)` and `i32(f32)` (rule 6).
- **Order:** WGSL lets a function call one declared further down and C++ does
  not, so every plain function is declared (a prototype) before any is
  defined (v0.6.1; v0.6.0 emitted declaration order and failed to compile when
  a callee came last).
- **Names:** `main` joins the reserved ones (MSL does not allow a kernel
  called `main`).

Two more rules, each with its source:

6. **`f32` to integer conversion** (gpuweb `wgsl/index.bs`, "Floating Point
   Conversion", line 14751).
   - **WGSL:** a NaN gives an indeterminate value. Otherwise the result is
     the value of the integer type closest to `truncate(X)` *and also exactly
     representable in the original floating-point type*.
   - **Consequence:** `u32` saturates at 4294967040, the largest `f32` below
     2³², not at 4294967295, which `f32` cannot represent; `i32` saturates at
     2147483520 and −2147483648. The constant evaluator of this package
     clamped to 4294967295 and 2147483647 for a constant conversion of an
     `f32` until v0.6.2; it now agrees (an AbstractFloat, an f64, still
     saturates at those two, which it can hold), and `msl/metaltest` compares
     the evaluator with the GPU.
   - **MSL:** a conversion of an out-of-range float is undefined in C++.
   - **The rule:** `wgsl_f2u` and `wgsl_f2i` in the preamble, which give 0 for
     a NaN. Conversions of `f32` vectors to integer vectors are refused.
7. **Indexing is not bounds-checked.**
   - **WGSL** (`index.bs`, "Out-of-Bounds Access", lines 4151–4199): an
     out-of-bounds access is a dynamic error whose outcomes are limited to a
     trap, a load of the zero value or of any location of the bound buffer,
     and a store to any location of the bound buffer or none at all. It never
     reads or writes outside the buffer.
   - **MSL:** the access is undefined and may touch memory outside the
     buffer, and a runtime-sized array's length is not known to the kernel.
   - **The rule:** `a[i]` is emitted as is. The host sizes its buffers, and a
     kernel is only correct if its indices are in range by construction.
     `arrayLength` is refused. This is the one place where `Module` does not
     reproduce what WGSL defines; a kernel that needs the clamp must clamp
     itself.

## How it will be proved

gpu2d's Metal kernel switches to the emitted `cornerCenter` and `coverage`,
and its existing test (`TestMatchesReference`, random scenes against the CPU
reference) must stay byte-identical. The emitter's own tests compile each
emitted file with the system compiler through go-macos/metal and compare
results with the checker's constant evaluator on generated inputs, including
the edges: NaN, ±0, ±INF, `i32` overflow and division by zero.
