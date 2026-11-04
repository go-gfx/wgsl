// Package transform rewrites WGSL modules. It works on the parse-level AST
// only: it never needs name resolution or type checking, so a legitimate
// shader is never refused because of a checker defect.
package transform

import (
	"fmt"
	"strconv"

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
}

// Report describes what LoopBudget did.
type Report struct {
	Loops        int    // loop, for and while statements instrumented
	Counters     int    // local counters declared (one per outermost loop)
	BudgetName   string // the override or uniform variable
	FlagName     string // the atomic<u32> storage variable
	Instrumented bool   // false when the module has no loops: output equals input
}

// LoopBudget returns a copy of m in which every `loop`, `for` and `while`
// counts its iterations and leaves the loop, setting the truncation flag,
// when the count exceeds the budget.
//
// Semantics. Each outermost loop gets a function-local counter, declared just
// before it; the counter is incremented at the top of every iteration of that
// loop and of every loop nested in it, so the iterations of one loop nest are
// bounded by the budget in total (not per level). When `count > budget`, the
// loop executes `atomicStore(&flag, 1u); break;` and the enclosing loops stop
// at their next iteration, because the counter stays above the budget. A loop
// that runs at most `budget` iterations is unchanged in behaviour.
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
	t := &rewriter{rep: r, pick: pick}
	for _, d := range c.Decls {
		if f, ok := d.(*wgsl.Func); ok {
			f.Body.Stmts = t.list(f.Body.Stmts, "")
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
	rep  *Report
	pick func(string) string
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

// list rewrites a statement list. counter is the counter of the enclosing
// loop nest, "" outside any loop.
func (t *rewriter) list(ss []wgsl.Stmt, counter string) []wgsl.Stmt {
	var out []wgsl.Stmt
	for _, s := range ss {
		if counter == "" && isLoop(s) {
			name := t.pick("loop_count")
			t.rep.Counters++
			out = append(out, &wgsl.DeclStmt{Decl: &wgsl.Var{Name: name, Type: ident("u32"),
				Init: &wgsl.IntLit{Text: "0u"}}})
			out = append(out, t.stmt(s, name))
			continue
		}
		out = append(out, t.stmt(s, counter))
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

func (t *rewriter) block(b *wgsl.Block, counter string) {
	if b != nil {
		b.Stmts = t.list(b.Stmts, counter)
	}
}

func (t *rewriter) stmt(s wgsl.Stmt, counter string) wgsl.Stmt {
	switch s := s.(type) {
	case *wgsl.Block:
		t.block(s, counter)
	case *wgsl.If:
		t.block(s.Then, counter)
		switch e := s.Else.(type) {
		case *wgsl.If:
			t.stmt(e, counter)
		case *wgsl.Block:
			t.block(e, counter)
		}
	case *wgsl.Switch:
		for _, c := range s.Cases {
			t.block(c.Body, counter)
		}
	case *wgsl.Loop:
		t.rep.Loops++
		body := t.list(s.Body, counter)
		s.Body = append(t.check(counter), body...)
		if s.Cont != nil {
			s.Cont.Stmts = t.list(s.Cont.Stmts, counter)
		}
	case *wgsl.For:
		t.rep.Loops++
		t.block(s.Body, counter)
		s.Body.Stmts = append(t.check(counter), s.Body.Stmts...)
	case *wgsl.While:
		t.rep.Loops++
		t.block(s.Body, counter)
		s.Body.Stmts = append(t.check(counter), s.Body.Stmts...)
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
