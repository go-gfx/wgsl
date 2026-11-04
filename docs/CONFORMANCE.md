# Conformance against the W3C gpuweb CTS

The oracle is the WebGPU conformance test suite, `webgpu:shader,validation,*`
(gpuweb/cts commit `d116bcc`, 2026-10-23, BSD-3-Clause, licence copied to
`testdata/cts/LICENSE-CTS.txt`).

## How the corpus is made

The CTS keeps its WGSL inside TypeScript tests as parametrised template
literals, so it cannot be extracted statically. `internal/ctsvendor` runs the
real CTS modules under Node (the CTS's own `ts-node` setup, `npm ci` done in
the checkout) with the GPU stubbed: `recorder.cjs` (the only JavaScript file,
justified in its header) replaces the GPU acquisition by a fake device and
records every `createShaderModule(code)` together with the verdict the test
expects (the `shouldError` of the enclosing `expectGPUError` /
`expectValidationError`, which `expectCompileResult`, `expectPipelineResult`
and `expectCompileWarning` all go through). 47 576 CTS test cases are
executed, 0 throw.

    cd gpuweb-cts && npm ci --ignore-scripts
    go run ./internal/ctsvendor -cts <gpuweb-cts> -out testdata/cts

629 337 recorded modules dedupe on (source, verdict) to **384 053** distinct
cases. They are stored as one gzip pack per CTS directory
(`testdata/cts/<dir>.jsonl.gz`, 18 MB in total instead of 384 k loose files;
`-files <dir>` writes loose `<dir>/<n>.wgsl` files for debugging) and indexed
by `testdata/cts/MANIFEST.json` (CTS commit, per-pack sha256, counts per status
and verdict). Per case: id, expected verdict, level (`module` =
`createShaderModule`, `pipeline` = the verdict is given at
`createXPipeline`), required features, sha256 of the source, status.

Statuses: `run`; `skipped-uniformity` (directory `uniformity` or id mentioning
uniformity: v0.1 has no uniformity analysis); `skipped-device` (pipeline
overridable constants supplied at pipeline creation, device limits, subgroup
size); `skipped-size` (source above 1 MiB, none today). Skipped cases are never
counted as passes.

## Running

    go test -run TestCTSConformance -v .            # compare with BASELINE.json
    go test -run TestCTSConformance -update .        # record improvements
    go test -run TestCTSConformance -ctsreport f.jsonl .   # every mismatch + source
    go run ./internal/ctsreport -only validation,statement, -level parse f.jsonl

`testdata/cts/BASELINE.json` lists the known mismatches per level, kind and CTS
directory as ranges of case numbers. The test fails on a mismatch that is not
in the baseline (regression or new case); mismatches that now match are logged
and recorded with `-update`, never silently.

Two levels are judged: `parse` (`Parse(src)`) and `validate` (`Validate(src)`).
A syntax-only front end legitimately accepts expected-fail cases whose error is
semantic, so at parse level the decisive column is **false-reject** (expected
pass, rejected) and false-accept is only a defect when the spec classifies the
error as syntactic.

## Results (checker = v0.1 stub, so validate = parse)

| directory | total | skipped | pass accepted | fail rejected | false reject | false accept |
|---|---:|---:|---:|---:|---:|---:|
| const_assert | 2670 | 0 | 1334 | 0 | 0 | 1336 |
| decl | 2817 | 3 | 779 | 48 | 0 | 1987 |
| expression | 264241 | 2588 | 122994 | 3336 | 0 | 135323 |
| extension | 532 | 43 | 205 | 10 | 0 | 274 |
| functions | 10233 | 0 | 607 | 12 | 0 | 9614 |
| parse | 4448 | 169 | 1658 | 1931 | 0 | 690 |
| shader_io | 4930 | 180 | 1494 | 151 | 0 | 3105 |
| statement | 1062 | 0 | 347 | 211 | 0 | 504 |
| types | 1793 | 2 | 639 | 399 | 0 | 753 |
| uniformity | 91327 | 91327 | 0 | 0 | 0 | 0 |
| **total** | 384053 | 94312 | 130057 | 6098 | **0** | 153586 |

The numbers are those of the committed baseline; they are expected to improve
(false accepts to fall) as the checker lands.

## Results with the v0.1.0 checker (validate = Parse + Check)

| directory | total | skipped | pass accepted | fail rejected | false reject | false accept |
|---|---:|---:|---:|---:|---:|---:|
| const_assert | 2670 | 0 | 1334 | 1336 | 0 | 0 |
| decl | 2817 | 3 | 779 | 2034 | 0 | 1 |
| expression | 264241 | 2588 | 122140 | 131266 | 854 | 7393 |
| extension | 532 | 43 | 201 | 278 | 4 | 6 |
| functions | 10233 | 0 | 607 | 9624 | 0 | 2 |
| parse | 4448 | 169 | 1654 | 2609 | 4 | 12 |
| shader_io | 4930 | 180 | 1490 | 3253 | 4 | 3 |
| statement | 1062 | 0 | 347 | 713 | 0 | 2 |
| types | 1793 | 2 | 639 | 1139 | 0 | 13 |
| uniformity | 91327 | 91327 | 0 | 0 | 0 | 0 |
| **total** | 384053 | 94312 | 129191 | 152252 | 866 | 7432 |

Of the 289,741 runnable cases, 281,443 (97.1 %) have the expected verdict.
P1 below is fixed in v0.1.0 (parse-level fail rejected: 6,099, false accept
153,585 before the checker). The remaining false rejects are concentrated in
`expression` (854 of 866) and are open defects for v0.1.x.

## Parser defects found

Every parse-level false accept was read by test group (all groups: the names
of the 153 586 mismatches were listed per CTS test and the syntactic-looking
ones opened). No parse-level false reject exists (0 of 130 057 expected-pass
cases). The false accepts are semantic (type errors, attribute placement,
shadowing, recursion, behaviour analysis, ...) except one.

### P1. NUL code point in a comment is accepted (false accept)

* case: `webgpu:shader,validation,parse,blankspace:null_characters:contains_null=true;placement="comment"`
  (pack `parse`, `parse/00043.wgsl`)
* source: `// Here is a <U+0000> character\n`
* spec: WGSL section "Parsing" (gpuweb/gpuweb `wgsl/index.bs` line 976): *"The
  program text must not include a null code point (U+0000)"*, a
  shader-creation error. It applies to the whole program text including
  comments. `Parse` rejects NUL in the other two CTS placements (between
  tokens, at end of line) but accepts it inside a `//` comment, so the check
  belongs in the source-text pass, before comments are skipped.

### Not parser defects (decided by the spec as semantic)

These show up as parse-level false accepts and are for the checker:

* `diagnostic(none, ...)`, `@diagnostic(warn, ...)`: `severity_control_name`
  is an `ident_pattern_token` in the grammar
  (`syntax/severity_control_name.syntax.bs.include`); the valid names are a
  semantic list (section "Diagnostic Severity Control Names").
* `@mcompute`, `@malign(4)`, `@id(-1)`, duplicate `@invariant`/`@align`,
  `var<read, storage>`, `alias T = atomic;`, `alias T = ptr<function>;`,
  `vec5<u32>`: attributes and template lists are generic in the grammar; the
  names, arities and values are checked later.
* Literal range and type (`var t: i32 = 2147483648;`, `4294967296u`,
  `1.0e+999999999999f`, `-1u`): value representability and typing are decided
  at type check / const-evaluation (section "Literal Value Expressions").
* `for (;;) {}`, `loop { continue; }`, `return;` in a value function,
  `switch` without or with two `default`, `v++` on a vector: behaviour
  analysis and statement typing, not syntax (the CTS itself notes
  "fails due to behavior-analysis").
