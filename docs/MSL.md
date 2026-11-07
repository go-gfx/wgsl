# Emitting Metal Shading Language: design

Status: design, nothing implemented. This note fixes the scope and the
rules before code exists, with the sources they come from.

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
4. **Integer overflow wraps.**
   - **The problem:** WGSL defines `i32` arithmetic as wrapping, while MSL
     follows C++, where signed overflow is undefined.
   - **The rule:** signed `+`, `-` and `*` are emitted through `uint` and cast
     back, which wraps by definition. This must be confirmed against the
     specification's operator chapter (§3.1) before implementation; it is not
     yet.
5. **Integer division and remainder by zero.**
   - **The problem:** WGSL defines the result (§ "Arithmetic expressions" of
     the WGSL specification), and MSL does not.
   - **The rule:** the emitter guards the divisor, as WGSL's definition does.
     To open: the WGSL specification's exact rule.

## How it will be proved

gpu2d's Metal kernel switches to the emitted `cornerCenter` and `coverage`,
and its existing test (`TestMatchesReference`, random scenes against the CPU
reference) must stay byte-identical. The emitter's own tests compile each
emitted file with the system compiler through go-macos/metal and compare
results with the checker's constant evaluator on generated inputs, including
the edges: NaN, ±0, ±INF, `i32` overflow and division by zero.
