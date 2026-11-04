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
7,432 invalid cases and wrongly reject 866 valid ones (see
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

Licence: see [LICENSE](LICENSE). The CTS data under `testdata/cts` is BSD-3
(`testdata/cts/LICENSE-CTS.txt`).
