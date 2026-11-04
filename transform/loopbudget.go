// Package transform rewrites WGSL modules. It works on the parse-level AST
// only: it never needs name resolution or type checking, so a legitimate
// shader is never refused because of a checker defect.
package transform

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/go-gfx/wgsl"
)

// BudgetSource says where the iteration budget comes from.
type BudgetSource struct {
	// Override, when true, declares `override <name>: u32 = <Default>;`
	// (pipeline-overridable constant). Otherwise a uniform buffer variable
	// `@group(Group) @binding(Binding) var<uniform> <name>: u32;` is declared.
	Override bool
	Default  uint32 // value of the override when the pipeline does not set it
	Group    int
	Binding  int
}

// Config configures LoopBudget.
type Config struct {
	Budget BudgetSource
	// Flag is the storage binding of the truncation flag, declared as
	// `@group(G) @binding(B) var<storage, read_write> <name>: atomic<u32>;`.
	// The host reads it back: 0 means no loop was cut, 1 means at least one
	// invocation was cut at the budget and its result is partial.
	FlagGroup, FlagBinding int
	// Prefix of the names the transform introduces ("wgsl_" by default). The
	// names are made unique against every identifier of the module.
	Prefix string
	// SmallLoop, if nonzero, is the largest constant trip count of a leaf loop
	// that is not checked per iteration: a `for (var i = a; i < b; i++)` whose
	// bounds and step are literals, whose body has no loop and does not assign i,
	// runs at most SmallLoop iterations whatever the invocation does, so the
	// enclosing loop's counter is advanced by its trip count in one addition
	// before it (the bound stays exact) and the loop itself is left as written.
	// A leaf loop outside every loop is left as written. 0: every loop is
	// checked.
	SmallLoop uint32
}

// Report describes what LoopBudget did.
type Report struct {
	Loops        int    // loop, for and while statements instrumented
	Skipped      int    // leaf loops with a small constant trip count, left as written (Config.SmallLoop)
	Counters     int    // local counters declared (one per nesting depth of each outermost loop)
	BudgetName   string // the override or uniform variable
	FlagName     string // the atomic<u32> storage variable
	Instrumented bool   // false when the module has no loops: output equals input
}

// LoopBudget returns a copy of m in which every `loop`, `for` and `while`
// counts its iterations and leaves the loop, setting the truncation flag,
// when the count exceeds the budget.
//
// Semantics. Each outermost loop gets function-local counters, one per
// nesting depth of its nest, declared just before it; the counter of depth d
// is incremented at the top of every iteration of every loop of depth d in the
// nest, so the iterations of the loops of one depth are bounded by the budget
// in total (a nest runs at most depth x budget iterations). When
// `count > budget`, the loop executes `atomicStore(&flag, 1u); break;` and the
// loops of its depth stop at their next iteration, because the counter stays
// above the budget. A loop that runs at most `budget` iterations is unchanged
// in behaviour.
//
// Why one counter per depth and not one per nest: an inner loop whose trip
// count differs between invocations (a strided loop over the thread index)
// would make a shared counter non-uniform, and the outer loop's check with it,
// and the browser then rejects a workgroupBarrier() in the outer loop. With
// one counter per depth the outer counter only counts outer iterations.
//
// The counters are locals, not module-scope `var<private>`, on purpose: the
// uniformity analysis treats a read of a private variable as non-uniform, so
// a `workgroupBarrier()` after the check would be rejected by the browser,
// while a local that is only changed under uniform control flow stays
// uniform. Separate loop nests have separate counters so that a non-uniform
// loop does not make a later loop's check non-uniform.
//
// The bound is per function activation: a function with loops called from
// inside a loop does its own budget. WGSL forbids recursion, so the total is
// finite, but it is the budget times the number of activations.
//
// The flag is written with atomicStore into a read_write storage buffer, so
// the transform is meaningful for compute shaders (vertex and fragment stages
// cannot write storage buffers).
func LoopBudget(m *wgsl.Module, cfg Config) (*wgsl.Module, *Report, error) {
	if cfg.Prefix == "" {
		cfg.Prefix = "wgsl_"
	}
	if cfg.Budget.Group < 0 || cfg.Budget.Binding < 0 || cfg.FlagGroup < 0 || cfg.FlagBinding < 0 {
		return nil, nil, fmt.Errorf("transform: negative group or binding")
	}
	// Clone through print and parse: a fresh tree that shares nothing with m.
	c, err := wgsl.Parse(wgsl.Print(m))
	if err != nil {
		return nil, nil, fmt.Errorf("transform: input does not round-trip: %w", err)
	}
	used := identSet(c)
	pick := func(base string) string {
		name := cfg.Prefix + base
		for i := 0; used[name]; i++ {
			name = cfg.Prefix + base + "_" + strconv.Itoa(i)
		}
		used[name] = true
		return name
	}
	r := &Report{BudgetName: pick("loop_budget"), FlagName: pick("truncated")}
	t := &rewriter{rep: r, pick: pick, small: cfg.SmallLoop}
	for _, d := range c.Decls {
		if f, ok := d.(*wgsl.Func); ok {
			f.Body.Stmts = t.list(f.Body.Stmts, nil, 0)
		}
	}
	if r.Loops == 0 {
		return c, r, nil
	}
	r.Instrumented = true
	var decls []wgsl.Decl
	if cfg.Budget.Override {
		decls = append(decls, &wgsl.Value{Kind: "override", Name: r.BudgetName,
			Type: ident("u32"), Init: &wgsl.IntLit{Text: strconv.FormatUint(uint64(cfg.Budget.Default), 10) + "u"}})
	} else {
		decls = append(decls, &wgsl.Var{
			Attrs:    bindingAttrs(cfg.Budget.Group, cfg.Budget.Binding),
			Template: []wgsl.Expr{ident("uniform")}, Name: r.BudgetName, Type: ident("u32")})
	}
	decls = append(decls, &wgsl.Var{
		Attrs:    bindingAttrs(cfg.FlagGroup, cfg.FlagBinding),
		Template: []wgsl.Expr{ident("storage"), ident("read_write")}, Name: r.FlagName,
		Type: &wgsl.Ident{Name: "atomic", Targs: []wgsl.Expr{ident("u32")}}})
	// after directives, before the first existing declaration
	c.Decls = append(decls, c.Decls...)
	return c, r, nil
}

func ident(n string) *wgsl.Ident { return &wgsl.Ident{Name: n} }

func intAttr(name string, v int) *wgsl.Attr {
	return &wgsl.Attr{Name: name, Args: []wgsl.Expr{&wgsl.IntLit{Text: strconv.Itoa(v)}}}
}

func bindingAttrs(group, binding int) []*wgsl.Attr {
	return []*wgsl.Attr{intAttr("group", group), intAttr("binding", binding)}
}

type rewriter struct {
	rep   *Report
	pick  func(string) string
	small uint32 // Config.SmallLoop
}

// names are fixed before rewriting starts (see LoopBudget); the checks refer
// to them through the report, which is complete before any statement is built.
func (t *rewriter) names() (budget, flag string) { return t.rep.BudgetName, t.rep.FlagName }

func isLoop(s wgsl.Stmt) bool {
	switch s.(type) {
	case *wgsl.Loop, *wgsl.For, *wgsl.While:
		return true
	}
	return false
}

// nest is the counters of one outermost loop nest, one per nesting depth.
type nest struct{ names []string }

// depthOf is the deepest loop nesting in a statement list (0: no loop).
func (t *rewriter) depthOf(ss []wgsl.Stmt) int {
	d := 0
	for _, s := range ss {
		d = max(d, t.depthOfStmt(s))
	}
	return d
}

func (t *rewriter) depthOfStmt(s wgsl.Stmt) int {
	switch s := s.(type) {
	case *wgsl.Block:
		return t.depthOf(s.Stmts)
	case *wgsl.If:
		d := t.depthOfStmt(s.Then)
		if s.Else != nil {
			d = max(d, t.depthOfStmt(s.Else))
		}
		return d
	case *wgsl.Switch:
		d := 0
		for _, c := range s.Cases {
			d = max(d, t.depthOfStmt(c.Body))
		}
		return d
	case *wgsl.Loop:
		d := t.depthOf(s.Body)
		if s.Cont != nil {
			d = max(d, t.depthOf(s.Cont.Stmts))
		}
		return 1 + d
	case *wgsl.For:
		if _, ok := t.smallLeaf(s); ok {
			return 0
		}
		return 1 + t.depthOfStmt(s.Body)
	case *wgsl.While:
		return 1 + t.depthOfStmt(s.Body)
	}
	return 0
}

// list rewrites a statement list. n is the counters of the enclosing loop
// nest (nil outside any loop) and depth the loop depth of the list in it.
func (t *rewriter) list(ss []wgsl.Stmt, n *nest, depth int) []wgsl.Stmt {
	var out []wgsl.Stmt
	for _, s := range ss {
		if f, ok := s.(*wgsl.For); ok {
			if trip, small := t.smallLeaf(f); small {
				t.rep.Skipped++
				if n != nil && depth > 0 { // advance the enclosing loop's counter, once
					out = append(out, &wgsl.Assign{LHS: ident(n.names[depth-1]), Op: "+=", RHS: &wgsl.IntLit{Text: strconv.FormatInt(trip, 10) + "u"}})
				}
				out = append(out, s)
				continue
			}
		}
		if n == nil && isLoop(s) {
			nn := &nest{}
			for i := t.depthOfStmt(s); i > 0; i-- {
				name := t.pick("loop_count")
				t.rep.Counters++
				nn.names = append(nn.names, name)
				out = append(out, &wgsl.DeclStmt{Decl: &wgsl.Var{Name: name, Type: ident("u32"),
					Init: &wgsl.IntLit{Text: "0u"}}})
			}
			out = append(out, t.stmt(s, nn, 0))
			continue
		}
		out = append(out, t.stmt(s, n, depth))
	}
	return out
}

// check builds: count += 1u; if (count > budget) { atomicStore(&flag, 1u); break; }
func (t *rewriter) check(counter string) []wgsl.Stmt {
	budget, flag := t.names()
	return []wgsl.Stmt{
		&wgsl.Assign{LHS: ident(counter), Op: "+=", RHS: &wgsl.IntLit{Text: "1u"}},
		&wgsl.If{
			Cond: &wgsl.Paren{X: &wgsl.Binary{Op: ">", X: ident(counter), Y: ident(budget)}},
			Then: &wgsl.Block{Stmts: []wgsl.Stmt{
				&wgsl.CallStmt{Call: &wgsl.Call{Fn: ident("atomicStore"), Args: []wgsl.Expr{
					&wgsl.Unary{Op: "&", X: ident(flag)}, &wgsl.IntLit{Text: "1u"}}}},
				&wgsl.Break{},
			}},
		},
	}
}

func (t *rewriter) block(b *wgsl.Block, n *nest, depth int) {
	if b != nil {
		b.Stmts = t.list(b.Stmts, n, depth)
	}
}

func (t *rewriter) stmt(s wgsl.Stmt, n *nest, depth int) wgsl.Stmt {
	switch s := s.(type) {
	case *wgsl.Block:
		t.block(s, n, depth)
	case *wgsl.If:
		t.block(s.Then, n, depth)
		switch e := s.Else.(type) {
		case *wgsl.If:
			t.stmt(e, n, depth)
		case *wgsl.Block:
			t.block(e, n, depth)
		}
	case *wgsl.Switch:
		for _, c := range s.Cases {
			t.block(c.Body, n, depth)
		}
	case *wgsl.Loop:
		t.rep.Loops++
		body := t.list(s.Body, n, depth+1)
		s.Body = append(t.check(n.names[depth]), body...)
		if s.Cont != nil {
			s.Cont.Stmts = t.list(s.Cont.Stmts, n, depth+1)
		}
	case *wgsl.For:
		t.rep.Loops++
		t.block(s.Body, n, depth+1)
		s.Body.Stmts = append(t.check(n.names[depth]), s.Body.Stmts...)
	case *wgsl.While:
		t.rep.Loops++
		t.block(s.Body, n, depth+1)
		s.Body.Stmts = append(t.check(n.names[depth]), s.Body.Stmts...)
	}
	return s
}

// identSet collects every identifier spelled in the module (declarations,
// uses, members, attribute arguments), so introduced names cannot collide.
func identSet(m *wgsl.Module) map[string]bool {
	set := map[string]bool{}
	var expr func(e wgsl.Expr)
	attrs := func(as []*wgsl.Attr) {
		for _, a := range as {
			for _, x := range a.Args {
				expr(x)
			}
		}
	}
	expr = func(e wgsl.Expr) {
		switch e := e.(type) {
		case *wgsl.Ident:
			set[e.Name] = true
			for _, a := range e.Targs {
				expr(a)
			}
		case *wgsl.Call:
			expr(e.Fn)
			for _, a := range e.Args {
				expr(a)
			}
		case *wgsl.Unary:
			expr(e.X)
		case *wgsl.Binary:
			expr(e.X)
			expr(e.Y)
		case *wgsl.Index:
			expr(e.X)
			expr(e.I)
		case *wgsl.Member:
			set[e.Name] = true
			expr(e.X)
		case *wgsl.Paren:
			expr(e.X)
		}
	}
	var stmts func(ss []wgsl.Stmt)
	var decl func(d wgsl.Decl)
	typ := func(t *wgsl.Ident) {
		if t != nil {
			expr(t)
		}
	}
	stmt := func(s wgsl.Stmt) {
		switch s := s.(type) {
		case *wgsl.Block:
			attrs(s.Attrs)
			stmts(s.Stmts)
		case *wgsl.If:
			attrs(s.Attrs)
			expr(s.Cond)
			stmts([]wgsl.Stmt{s.Then})
			if s.Else != nil {
				stmts([]wgsl.Stmt{s.Else})
			}
		case *wgsl.Switch:
			attrs(s.Attrs)
			attrs(s.BodyAttrs)
			expr(s.X)
			for _, c := range s.Cases {
				for _, x := range c.Selectors {
					if x != nil {
						expr(x)
					}
				}
				stmts([]wgsl.Stmt{c.Body})
			}
		case *wgsl.Loop:
			attrs(s.Attrs)
			attrs(s.BodyAttrs)
			stmts(s.Body)
			if s.Cont != nil {
				attrs(s.Cont.Attrs)
				stmts(s.Cont.Stmts)
			}
		case *wgsl.For:
			attrs(s.Attrs)
			if s.Init != nil {
				stmts([]wgsl.Stmt{s.Init})
			}
			if s.Cond != nil {
				expr(s.Cond)
			}
			if s.Update != nil {
				stmts([]wgsl.Stmt{s.Update})
			}
			stmts([]wgsl.Stmt{s.Body})
		case *wgsl.While:
			attrs(s.Attrs)
			expr(s.Cond)
			stmts([]wgsl.Stmt{s.Body})
		case *wgsl.Return:
			if s.X != nil {
				expr(s.X)
			}
		case *wgsl.Assign:
			if s.LHS != nil {
				expr(s.LHS)
			}
			expr(s.RHS)
		case *wgsl.IncDec:
			expr(s.X)
		case *wgsl.CallStmt:
			expr(s.Call)
		case *wgsl.DeclStmt:
			decl(s.Decl)
		case *wgsl.AssertStmt:
			expr(s.Assert.X)
		case *wgsl.BreakIf:
			expr(s.Cond)
		}
	}
	stmts = func(ss []wgsl.Stmt) {
		for _, s := range ss {
			stmt(s)
		}
	}
	decl = func(d wgsl.Decl) {
		switch d := d.(type) {
		case *wgsl.Func:
			set[d.Name] = true
			attrs(d.Attrs)
			attrs(d.RetAttrs)
			for _, p := range d.Params {
				set[p.Name] = true
				attrs(p.Attrs)
				typ(p.Type)
			}
			typ(d.Ret)
			attrs(d.Body.Attrs)
			stmts(d.Body.Stmts)
		case *wgsl.Var:
			set[d.Name] = true
			attrs(d.Attrs)
			for _, x := range d.Template {
				expr(x)
			}
			typ(d.Type)
			if d.Init != nil {
				expr(d.Init)
			}
		case *wgsl.Value:
			set[d.Name] = true
			attrs(d.Attrs)
			typ(d.Type)
			if d.Init != nil {
				expr(d.Init)
			}
		case *wgsl.Alias:
			set[d.Name] = true
			typ(d.Type)
		case *wgsl.Struct:
			set[d.Name] = true
			for _, mem := range d.Members {
				set[mem.Name] = true
				attrs(mem.Attrs)
				typ(mem.Type)
			}
		case *wgsl.ConstAssert:
			expr(d.X)
		}
	}
	for _, d := range m.Decls {
		decl(d)
	}
	return set
}

// smallLeaf reports whether f is a leaf loop with a constant trip count of at
// most the configured SmallLoop, and that count: init a literal, condition
// `i < b` or `i <= b` with a literal b, step `i++`, `i += c` or `i = i + c` with a
// literal c > 0, no loop in the body and no assignment to i or &i in it.
func (t *rewriter) smallLeaf(f *wgsl.For) (int64, bool) {
	if t.small == 0 || f.Init == nil || f.Cond == nil || f.Update == nil || f.Body == nil {
		return 0, false
	}
	ds, ok := f.Init.(*wgsl.DeclStmt)
	if !ok {
		return 0, false
	}
	v, ok := ds.Decl.(*wgsl.Var)
	if !ok || v.Init == nil {
		return 0, false
	}
	lo, ok := intLit(v.Init)
	if !ok || lo < 0 {
		return 0, false
	}
	cond, ok := unparen(f.Cond).(*wgsl.Binary)
	if !ok || (cond.Op != "<" && cond.Op != "<=") || !isIdent(cond.X, v.Name) {
		return 0, false
	}
	hi, ok := intLit(cond.Y)
	if !ok || hi < 0 || hi > 1<<30 {
		return 0, false
	}
	if cond.Op == "<=" {
		hi++
	}
	step := int64(0)
	switch u := f.Update.(type) {
	case *wgsl.IncDec:
		if isIdent(u.X, v.Name) && u.Op == "++" {
			step = 1
		}
	case *wgsl.Assign:
		if !isIdent(u.LHS, v.Name) {
			break
		}
		if u.Op == "+=" {
			step, _ = intLit(u.RHS)
		} else if b, ok := unparen(u.RHS).(*wgsl.Binary); ok && u.Op == "=" && b.Op == "+" && isIdent(b.X, v.Name) {
			step, _ = intLit(b.Y)
		}
	}
	if step < 1 {
		return 0, false
	}
	trip := (hi - lo + step - 1) / step
	if hi <= lo {
		trip = 0
	}
	if trip > int64(t.small) || bodyBreaksLeaf(f.Body.Stmts, v.Name) {
		return 0, false
	}
	return trip, true
}

func unparen(e wgsl.Expr) wgsl.Expr {
	for {
		p, ok := e.(*wgsl.Paren)
		if !ok {
			return e
		}
		e = p.X
	}
}

func isIdent(e wgsl.Expr, name string) bool {
	id, ok := unparen(e).(*wgsl.Ident)
	return ok && id.Name == name && len(id.Targs) == 0
}

// intLit is the value of an integer literal expression (i or u suffix allowed).
func intLit(e wgsl.Expr) (int64, bool) {
	l, ok := unparen(e).(*wgsl.IntLit)
	if !ok {
		return 0, false
	}
	n, err := strconv.ParseInt(strings.TrimRight(l.Text, "ui"), 0, 64)
	return n, err == nil
}

// bodyBreaksLeaf reports whether a loop body (that must be a leaf) contains a
// loop, or anything that could change the loop variable: an assignment to it,
// ++ or --, or its address taken.
func bodyBreaksLeaf(ss []wgsl.Stmt, name string) bool {
	bad := false
	var expr func(e wgsl.Expr)
	expr = func(e wgsl.Expr) {
		switch e := e.(type) {
		case *wgsl.Unary:
			if e.Op == "&" && isIdent(e.X, name) {
				bad = true
			}
			expr(e.X)
		case *wgsl.Call:
			for _, a := range e.Args {
				expr(a)
			}
		case *wgsl.Binary:
			expr(e.X)
			expr(e.Y)
		case *wgsl.Index:
			expr(e.X)
			expr(e.I)
		case *wgsl.Member:
			expr(e.X)
		case *wgsl.Paren:
			expr(e.X)
		}
	}
	var stmts func(ss []wgsl.Stmt)
	stmt := func(s wgsl.Stmt) {
		switch s := s.(type) {
		case *wgsl.Loop, *wgsl.For, *wgsl.While:
			bad = true
		case *wgsl.Block:
			stmts(s.Stmts)
		case *wgsl.If:
			expr(s.Cond)
			stmts([]wgsl.Stmt{s.Then})
			if s.Else != nil {
				stmts([]wgsl.Stmt{s.Else})
			}
		case *wgsl.Switch:
			expr(s.X)
			for _, c := range s.Cases {
				stmts([]wgsl.Stmt{c.Body})
			}
		case *wgsl.Assign:
			if s.LHS != nil && lvalueRoot(s.LHS) == name {
				bad = true
			}
			expr(s.RHS)
		case *wgsl.IncDec:
			if lvalueRoot(s.X) == name {
				bad = true
			}
		case *wgsl.CallStmt:
			expr(s.Call)
		case *wgsl.DeclStmt:
			if v, ok := s.Decl.(*wgsl.Var); ok && v.Init != nil {
				expr(v.Init)
			}
			if v, ok := s.Decl.(*wgsl.Value); ok && v.Init != nil {
				expr(v.Init)
			}
		}
	}
	stmts = func(ss []wgsl.Stmt) {
		for _, s := range ss {
			stmt(s)
		}
	}
	stmts(ss)
	return bad
}

// lvalueRoot is the variable at the root of an assignment target (a, a[i],
// a.m, (a)), "" otherwise.
func lvalueRoot(e wgsl.Expr) string {
	for {
		switch x := e.(type) {
		case *wgsl.Ident:
			return x.Name
		case *wgsl.Index:
			e = x.X
		case *wgsl.Member:
			e = x.X
		case *wgsl.Paren:
			e = x.X
		default:
			return ""
		}
	}
}
