# Emitting HLSL (Shader Model 5.0): design and rules

Status: implemented (package `hlsl`, v0.9.0) and tested on Direct3D 11 WARP, the
software rasterizer of Windows, on `windows-latest` (`hlsl/hlsltest`). No GPU
has run it. This note fixes the scope and the rules, with the sources they come
from. It mirrors `docs/MSL.md` and `docs/SPIRV.md`.

## Why

go-gfx/gpu2d's raster kernel is one WGSL compute shader (`kernel.wgsl`,
docs/NATIVE.md decision 1) and every native back-end is generated from it: MSL
for Metal, SPIR-V for Vulkan, and HLSL for Direct3D 11. The HLSL is compiled to
DXBC on the Windows CI lane and the bytes are committed, so a user never needs
a compiler.

## Scope

`hlsl.Module(module, info, entries...)` emits compute entry points, as `msl.Module`
does:
- **Entry points:** `@compute` only, returning nothing, with
  `@builtin(global_invocation_id)` as the only input (`uint3 id :
  SV_DispatchThreadID`). `@workgroup_size` becomes `[numthreads(x, y, z)]`,
  refused beyond cs_5_0's 1024 threads and z of 64.
- **Types:** `bool`, `i32`, `u32`, `f32`, their vectors (not `bool` vectors), and
  structures of scalars and vectors.
- **Module-scope variables:** `var<storage>` runtime-sized arrays and
  `var<uniform>` of group 0, nothing else. See "Resources".
- **Statements:** `let`, `var`, assignment (and `op=`), `if`/`else`, `for`,
  `while`, `loop` (no `continuing`), `break`, `continue`, `return`, `++`, `--`
  and calls. `switch`, `discard` and `break if` are refused.
- **Builtins:** `abs` (floats), `sqrt`, `floor`, `min`, `max`, `clamp`, `select`
  (scalar condition), `length`, `bitcast`, and the scalar and vector constructors
  and conversions.
- Anything else is refused with its position as a `*hlsl.Error`, never emitted
  approximately.

## Resources: StructuredBuffer, not ByteAddressBuffer

gpu2d's kernel reads an array of a 64-byte `Inst` of sixteen scalars and three
arrays of `u32`. A `StructuredBuffer<T>` indexes them as WGSL does and needs no
byte arithmetic, and the d3d11 binding creates structured buffers with a stride
(`KindStructured`). So:

| WGSL | HLSL | Register |
|---|---|---|
| `var<storage, read> a: array<T>` | `StructuredBuffer<T> a` | `t<binding>` (0 to 127) |
| `var<storage, read_write> a: array<T>` | `RWStructuredBuffer<T> a` | `u<binding>` (0 to 7) |
| `var<uniform> u: T` | `cbuffer wgsl_cb<binding> { T u; }` | `b<binding>` (0 to 13) |

The binding number is the register index **of its class**; the host binds an SRV
at slot `binding` for a `read` buffer, a UAV for `read_write`, a constant buffer
for a uniform. A binding beyond the class's limit is refused (Direct3D 11's
compute stage has 8 UAV slots and 14 constant buffer slots).

A structure kept in a buffer must have, in HLSL, the offsets WGSL's layout gives
it, or it is refused with both offsets. In a `StructuredBuffer` members are
packed at 4-byte granularity with no register rule, so `{u32, vec3<f32>}` (WGSL
offset 16, HLSL 4) is refused. In a cbuffer, members are packed into 16-byte
registers and a vector does not straddle one.

## Rules, with their sources

1. **Float arithmetic is precise.**
   - **The default:** HLSL may fuse a `mul` and an `add` into a `mad`, reorder
     operations, and use fast math that ignores NaN and INF (HLSL "Variable
     syntax", `precise`). WGSL guarantees none of that, and the CPU reference
     does none of it. The D3D11 specification allows the fusion ("fusing into
     `mad` is permitted", Functional Spec 3.1.3).
   - **What `precise` does** (learn.microsoft.com, "Variable syntax"): applied to
     a variable, it restricts "any calculations used to produce the value
     assigned to that variable": separate operations stay separate (a `mul` and
     an `add` are not fused), the order is kept, and IEEE rules for NaN and INF
     are respected. "Qualifying a variable precise doesn't make operations that
     use the variable precise": it propagates only to the operations that
     contribute to the value assigned to it.
   - **The rule:** every float variable is declared `precise`, and every float
     `+`, `-`, `*` and `/` (scalar or vector) is passed through `wgsl_pf`, which
     assigns its argument to a `precise` variable, so each intermediate result
     contributes to one. `length` is the explicit sum of squares in WGSL's
     order, each step in a `precise` variable.
   - **The flag:** the HLSL is also meant to be compiled with
     `D3DCOMPILE_IEEE_STRICTNESS`. `TestPreciseKeepsMulAndAddApart` shows the
     pins alone are enough, compiling without the flag: `a * b + c` is a `mul` and
     an `add` as emitted, and a `mad` once the pins are removed (a break check, in
     the disassembly).
2. **Signed overflow wraps.** DXBC's `iadd` "is not sensitive to the signedness
   of its operands" and keeps the low 32 bits (Functional Spec, `iadd`), but the
   HLSL language does not say so. Signed `+`, `-`, `*` and `<<` are emitted
   through `asuint`/`asint`, as MSL does.
3. **Integer division and remainder.** WGSL defines a zero divisor (the result
   is `e1`, and 0 for `%`) and `MIN / -1` (`e1`, and 0 for `%`). DXBC's `udiv`
   "returns 0xffffffff for both quotient and remainder" on a zero divisor
   (`udiv`). The divisor is guarded as in docs/MSL.md rule 5 (`select(b, 1, b == 0
   || ...)` as a `?:`), and signed `%` is `e1 - (e1 / d) * d`.
4. **Shifts.** The count is taken modulo 32 (`& 31u`; `ishl` uses "the LSB 5
   bits"), `<<` discards high bits, and `>>` on `i32` is arithmetic, written out
   rather than left to the compiler.
5. **`f32` to integer** (WGSL "Floating Point Conversion"): the value of the
   integer type closest to the truncation and also exactly representable in
   `f32`, so `u32` saturates at 4294967040 and `i32` at 2147483520 and -2147483648.
   DXBC's `ftou` clamps to `[0 .. 4294967295.999f]` and `ftoi` to
   `[-2147483648.999f .. 2147483647.999f]`, both of which round to 2^32 and
   2^31 in `f32`, and NaN gives 0. So `wgsl_f2u` and `wgsl_f2i` clamp first. A NaN
   gives 0 (an indeterminate value in WGSL). Conversions of `f32` vectors to
   integer vectors are refused.
6. **`bitcast` and `i32`/`u32` conversions** are `asint`, `asuint`, `asfloat`:
   the bits, never a value conversion.
7. **Indexing is not bounds-checked.** WGSL allows a trap, a zero or any location
   of the bound buffer. D3D11 defines more (go-mswin/d3d11 README): an
   out-of-range struct index of a structured buffer reads 0 and a write is
   dropped; an out-of-range byte offset inside a struct is undefined, which a
   scalar-indexed array cannot produce. `arrayLength` is refused.
8. **`&&` and `||` do not short-circuit in HLSL.** WGSL's do. The two agree when
   the right operand has no effect, which only a call to a module function can
   have, so a call on the right of `&&` or `||` is refused. `&` and `|` on `bool`
   (which evaluate both operands in WGSL) are emitted as `&&` and `||`.
9. **`select`** is `cond ? t : f`; both operands are evaluated, as in WGSL. A
   vector condition is refused.
10. **A float vector splat** `vec3f(x)` is `((float3)(x))`: HLSL has no
    `float3(x)`. A `var` without an initializer is `(T)0`.
11. **Functions** are emitted callee first (HLSL, like C, needs a declaration
    before a use), and use the buffers as globals, so there are no extra
    parameters.
12. **Names.** HLSL keywords, intrinsics (`mul`, `mad`, `length`, ...), type
    names (`float3`, `uint2x2`, `min16float`), and any name starting with `wgsl_`,
    `__` or `SV_` are refused: the emitted names are WGSL's own.
13. **A loop sits in a block of its own,** because fxc lets a loop variable clash
    with that of an earlier sibling loop (X3078, an error with warnings as
    errors), which WGSL's scopes do not. `[loop]` keeps fxc from unrolling.

## How it is proved

- `hlsl/hlsl_test.go` (every OS): the kernel emits and is deterministic, the
  integer and float rules appear in the code, 24 refusals are exercised.
- `hlsl/hlsltest` (a nested module so the core does not depend on go-mswin/d3d11;
  `windows-latest`): every emitted file is compiled by `D3DCompile` (cs_5_0, IEEE
  strictness, warnings as errors) through go-mswin/d3d11's `d3dcompiler`; the
  edge cases of docs/MSL.md run as dispatches on WARP and are held to WGSL's
  definition computed in Go, including the checker's constant evaluator for
  `u32(f32)` and `i32(f32)`; gpu2d's whole `kernel.wgsl` compiles, as a 16 x 8
  thread group with no `mad`.
- **Break checks:** the `mad` test above, and `TestUnguardedDivisionIsCaught`: the
  same program with the divisor guard removed gives DXBC's 0xFFFFFFFF for `7 / 0`
  where WGSL says 7, and the edge cases see it.
- **What WARP does not prove:** the arithmetic of a hardware driver, and speed.
  The `sqrt` measured on WARP is correctly rounded (0 of 261480 inputs differ);
  the D3D11 specification allows 1 ULP, and the test fails beyond that.
