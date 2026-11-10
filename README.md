# go-gfx/wgsl

A pure-Go WGSL (WebGPU Shading Language) front end: lexer, parser, AST,
printer, name resolution and type checker. No cgo, standard library only.

Pinned to the **WGSL Candidate Recommendation Draft of 26 October 2026**
(see [SPEC.md](SPEC.md)).

```go
m, err := wgsl.Parse(src)          // positioned *wgsl.Error on failure
text := wgsl.Print(m)              // canonical WGSL; re-parses to an equal AST
info, errs := wgsl.Check(m)        // types, resolved names, struct layouts, entry points
err = wgsl.Validate(src)           // Parse + Check
```

```sh
wgsl check shader.wgsl             # validate; exit 1 on error
wgsl check --parse-only shader.wgsl
wgsl fmt -w shader.wgsl            # canonical formatting
```

## Status (v0.1.0)

* Lexer and parser for the whole grammar, template-list discovery exactly as
  specified, reserved words rejected with a line and column. Every resource is
  bounded (`Limits`: source bytes, tokens, nesting depth); deeply nested input
  is refused, not stack-overflowed.
* `Print(Parse(src))` re-parses to an equal AST (`Equal`); fuzzed
  (`FuzzLex`, `FuzzParse`, `FuzzRoundTrip`, `FuzzValidate`, corpus committed).
* Name resolution and type checking, including built-in functions, address
  spaces, entry-point IO and const/override evaluation. `Info` carries what a
  back end needs (types, objects, struct layouts, entry points); an MSL
  emitter is not part of this release.
* **Not yet:** uniformity analysis (CTS `uniformity` cases are counted as
  skipped), alias analysis of pointer arguments, the `buffer_view` and
  `swizzle_assignment` extensions.
* Conformance against the W3C gpuweb CTS (`webgpu:shader,validation,*`,
  384,053 distinct cases executed, 94,312 skipped as uniformity or
  device-dependent): see [docs/CONFORMANCE.md](docs/CONFORMANCE.md).
* All real shaders of go-wasmcontainers/gpu and llm, go-quake1/engine and
  go-gfx/webgpu parse and round-trip (`testdata/real`, vendored by
  `internal/corpus`).

## Not a security boundary

**The browser remains the validation authority.** `Validate` and `wgsl check`
are not yet a security boundary: against the CTS they still wrongly accept
88 invalid cases (uniformity analysis is missing, so its 91,497 cases are
skipped) and wrongly reject none of the 130,057 valid ones (see
[docs/CONFORMANCE.md](docs/CONFORMANCE.md)). Use them for early, positioned
diagnostics (for example the reserved word `meta`), never to decide that a
guest shader is safe, and never to refuse a shader on their word alone.

## Loop budgets (package `transform`, v0.2.0)

`transform.LoopBudget` rewrites every `loop`, `for` and `while` so that the
work of one invocation is bounded by construction. Each outermost loop gets a
function-local counter; every iteration of the loop and of the loops nested in
it increments it and, past the budget, runs `atomicStore(&flag, 1u); break;`.
The budget is an `override` or a `var<uniform>` `u32`; the flag is an
`atomic<u32>` storage variable the host reads back, so a truncated run is
detectable. The rewrite works on the parse tree only and never calls the
checker, so a checker defect cannot refuse a legitimate shader.

```
wgsl transform --budget-group 1 --budget-binding 0 --flag-group 1 --flag-binding 1 kernel.wgsl
wgsl transform --budget-override 1000000 kernel.wgsl
```

Counters are locals, not `var<private>`, so the browser's uniformity analysis
still accepts a `workgroupBarrier()` after a checked loop. The bound is per
function activation (WGSL has no recursion, so the total is finite). Tested on
an Apple GPU through Chrome's WebGPU (Metal): a loop of 10^8 iterations is cut
at the budget with the flag set, a short loop is unchanged and leaves the flag
clear (`WGSL_GPU_TEST=1 go test -run TestGPU ./transform`).

## Metal Shading Language (package `msl`)

`msl.Functions(module, info, names...)` emits MSL for the named functions of a
checked module and every function they call, with the structures they use.
`msl.Module(module, info, entries...)` also emits compute entry points as
`kernel` functions, with the storage and uniform buffers they use, loops,
runtime-array indexing and `f32`-to-integer conversion (v0.6.0,
[docs/MSL.md](docs/MSL.md), "Compute modules"):

```go
m, _ := wgsl.Parse(src)
info, _ := wgsl.Check(m)
out, err := msl.Functions(m, info, "coverage")
```

- **Scope:** functions over `bool`, `i32`, `u32`, `f32`, their vectors, and
  structures of those; `let`, `var`, assignment, `if`, `for`, `while`, `loop`
  and `return`, plus the builtins listed in [docs/MSL.md](docs/MSL.md).
  Anything else is refused as an `*msl.Error` with its position, never emitted
  approximately: other stages, module-scope variables other than group-0
  buffers, fixed-size arrays, textures, `switch`.
- **WGSL's semantics, not Metal's defaults:** each rule comes from the opened
  specifications, with pages in [docs/MSL.md](docs/MSL.md).
  - Fast math is off for the file, and library calls are the `precise::` ones.
  - Signed `+`, `-`, `*` and `<<` wrap.
  - `/` and `%` give WGSL's results for a zero divisor and `MIN / -1`.
  - Shift counts are taken modulo 32, and `>>` on `i32` is arithmetic.
  - `bitcast` and `i32`↔`u32` conversions keep the bits.
- **Tested on the GPU:** `msl/metaltest` is its own module, so this one does
  not depend on go-macos/metal. It runs the emitted code on a Mac's GPU and
  holds every result to WGSL's definition computed in Go:
  - integers: 13 operations over 361 edge pairs, including `MIN`, −1, 0 and
    shift counts 31–64;
  - floats: `clamp` with low > high, `length`, `select`, `abs`, `min` and
    `max`, with the flush-to-zero WGSL allows on subnormals.

  With Metal's plain `a / b`, `1 / 0` comes back −1, where WGSL defines 1. The
  test catches it.
- **In use:** go-gfx/gpu2d's Metal raster kernel is generated from its WGSL
  kernel through this package, and stays byte-identical to its CPU reference.

## SPIR-V (package `spirv`)

`spirv.Module(module, info, spirv.Options{}, entries...)` emits a SPIR-V 1.5
module (the Vulkan 1.2 environment) for compute entry points: storage and
uniform buffers of group 0, scalar and vector arithmetic, and `if`, `for`,
`while` and `loop` with `break` and `continue`. It is the subset a fill kernel
needs; the rest is refused as a `*spirv.Error` with its position
([docs/SPIRV.md](docs/SPIRV.md) lists both, and the specification sections used).

- **Numbers from the grammar:** `spirv/ops_gen.go` is generated by
  `internal/spirvgen` from Khronos's `spirv.core.grammar.json`.
- **Checked on every platform** by a decoder that applies the logical layout
  of section 2.4, the structured control-flow rules of section 2.11 and the
  entry-point interface rule, and that must refuse ten mutated modules.
- **Validated by `spirv-val` in CI** (SPIRV-Tools from the distribution, used
  only by the `spirv-val` job and skipped where it is not installed). Not yet
  run on a GPU.

## HLSL (package `hlsl`)

`hlsl.Module(module, info, entries...)` emits High-Level Shading Language for
Shader Model 5.0 (compile it for `cs_5_0` with `D3DCompile`, with
`D3DCOMPILE_IEEE_STRICTNESS`): compute entry points with `read` storage arrays
as `StructuredBuffer` at `t<binding>`, `read_write` ones as `RWStructuredBuffer`
at `u<binding>`, a uniform as a `cbuffer` at `b<binding>`, the integer and
float rules WGSL defines, and every float operation kept `precise` (no fused
`mad`). It takes the whole of gpu2d's `kernel.wgsl`; the rest is refused as a
`*hlsl.Error` with its position ([docs/HLSL.md](docs/HLSL.md): scope, rules and
their sources).

- **Checked on every platform:** the emitted kernel, and 24 refusals.
- **Run on Direct3D 11 WARP** (the software rasterizer of Windows, in the
  nested module `hlsl/hlsltest` on `windows-latest`, through go-mswin/d3d11):
  every emitted file is compiled by `D3DCompile`, the integer, float and
  conversion edge cases are held to WGSL's definition, and two break checks
  show the tests can fail. No GPU has run it.

Licence: see [LICENSE](LICENSE). The CTS data under `testdata/cts` is BSD-3
(`testdata/cts/LICENSE-CTS.txt`).
