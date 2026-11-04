# Pinned specification

This module implements **WebGPU Shading Language, W3C Candidate Recommendation
Draft of 26 October 2026** (https://www.w3.org/TR/WGSL/, "This version"
`CRD-WGSL-20261026`).

The normative text was read from its source, gpuweb/gpuweb at commit
`4e829b310ba7381034b93b974eb1a6aabfb464f7` (2026-10-31, `wgsl/index.bs`):

| Topic | Source opened |
|---|---|
| Recursive-descent grammar | `wgsl/wgsl.recursive.bs.include` (the grammar the parser follows, production by production) |
| Blankspace, comments, literals, identifiers | `wgsl/index.bs` sections 3.1-3.6 and `wgsl/syntax/*_literal.syntax.bs.include`, `ident_pattern_token.syntax.bs.include` |
| Template list discovery | `wgsl/index.bs`, "Template list discovery" algorithm (implemented verbatim in `lexer.go`, `discoverTemplates`) |
| Keywords | `wgsl/index.bs`, "Keyword Summary" |
| Reserved words | `wgsl/wgsl.reserved.plain`, turned into `reserved_gen.go` by `internal/genreserved` |

The checker (`check*.go`, `decl.go`, `expr.go`, `stmt.go`, `builtins*.go`,
`entry.go`, `globals.go`) was written from these sections of the same
`wgsl/index.bs`: Errors, Diagnostics, Limits, Extensions (enable and
language), Declaration and Scope, Types (type checking, conversion rank,
overload resolution, plain types, memory views, textures, aliases,
predeclared types), Variable and Value Declarations, Early Evaluation
Expressions, Expressions, Statements (assignment, control flow, behavior
analysis, const assertions), Functions, Attributes, Entry Points, Memory
(address spaces, memory layout, layout constraints), Floating Point
Evaluation (overflow, conversion), and the built-in function overload tables
(`go run ./internal/specbuiltins` prints them).

Checker limits: uniformity analysis, alias analysis of pointer arguments, and
the `buffer_view` and `swizzle_assignment` language extensions are not
implemented; `uniform_buffer_standard_layout` is enabled by a `requires`
directive only. See the package comment in `check.go`.

Known deviations, all documented where they occur:

* Identifier classes use Go's `unicode` tables (ID_Start/ID_Continue) as an
  approximation of UAX #31 XID_Start/XID_Continue.
* Uniformity analysis (spec section "Uniformity") is not implemented in v0.1.
