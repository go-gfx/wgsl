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

```
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

Next: `wgsl transform` and package `transform` (loop iteration budgets for
host-bounded GPU work).

Licence: see [LICENSE](LICENSE). The CTS data under `testdata/cts` is BSD-3
(`testdata/cts/LICENSE-CTS.txt`).
