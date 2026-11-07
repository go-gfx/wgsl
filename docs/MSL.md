# Emitting Metal Shading Language: design

Status: the first version is implemented (package `msl`, v0.5.0) and tested
on the GPU (`msl/metaltest`). This note fixes the scope and the rules, with
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
- **Out of scope:** loops, arrays, pointers, textures and samplers, entry
  points, bindings. A module that uses them is refused with the construct and
  its position, never emitted approximately.

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

## How it will be proved

gpu2d's Metal kernel switches to the emitted `cornerCenter` and `coverage`,
and its existing test (`TestMatchesReference`, random scenes against the CPU
reference) must stay byte-identical. The emitter's own tests compile each
emitted file with the system compiler through go-macos/metal and compare
results with the checker's constant evaluator on generated inputs, including
the edges: NaN, ±0, ±INF, `i32` overflow and division by zero.
