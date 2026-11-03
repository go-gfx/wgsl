package wgsl

import "strings"

// behavior is a set of ways control can leave a statement (spec "Statements
// Behavior Analysis").
type behavior uint8

const (
	bNext behavior = 1 << iota
	bBreak
	bContinue
	bReturn
)

func (b behavior) has(x behavior) bool { return b&x != 0 }

func (c *checker) checkBody(f *FuncObj) {
	if f.Decl == nil || f.Decl.Body == nil {
		return
	}
	c.cur = f
	c.curDecl = f
	c.scopes = nil
	c.push()
	for _, p := range f.Params {
		c.declareLocal(p.Name, p, p.Decl)
	}
	c.loopDepth, c.switchDepth, c.inContinuing = 0, 0, 0
	c.targets = nil
	c.retType = f.Ret
	c.checkAttrs(f.Decl.Body.Attrs, "block", f.Decl.Body)
	b := c.seq(f.Decl.Body.Stmts)
	c.pop()
	if b == 0 {
		c.err(f.Decl, "behavior", "function %q never returns or terminates", f.Name)
	} else if f.Ret != nil && f.Ret != Void && !isInvalid(f.Ret) && b&^(bReturn|bBreak|bContinue) != 0 {
		c.err(f.Decl, "missing-return", "function %q does not return a value on all paths", f.Name)
	}
	c.cur = nil
	c.curDecl = nil
}

func (c *checker) seq(list []Stmt) behavior {
	acc := bNext
	for _, s := range list {
		b := c.stmt(s)
		if acc.has(bNext) {
			acc = (acc &^ bNext) | b
		}
	}
	return acc
}

func (c *checker) block(b *Block) behavior {
	c.checkAttrs(b.Attrs, "block", b)
	c.push()
	r := c.seq(b.Stmts)
	c.pop()
	return r
}

func (c *checker) boolCond(e Expr) {
	v := c.expr(e)
	if isInvalid(v.t) {
		return
	}
	c.convertTo(v, Bool)
}

func (c *checker) pushTarget(k byte) { c.targets = append(c.targets, k) }
func (c *checker) popTarget()        { c.targets = c.targets[:len(c.targets)-1] }

func (c *checker) stmt(s Stmt) behavior {
	c.depth++
	defer func() { c.depth-- }()
	if c.depth > maxDepth {
		c.err(s, "limit-depth", "statement nesting is too deep")
		return bNext
	}
	switch s := s.(type) {
	case *Block:
		return c.block(s)
	case *Empty:
		return bNext
	case *DeclStmt:
		c.localDecl(s.Decl)
		return bNext
	case *AssertStmt:
		c.constAssert(s.Assert)
		return bNext
	case *Assign:
		c.assign(s)
		return bNext
	case *IncDec:
		v := c.expr(s.X)
		if isInvalid(v.t) {
			return bNext
		}
		r, ok := v.t.(*RefType)
		if !ok || r.Access != AccessReadWrite || !isConcreteIntScalar(r.Elem) {
			c.err(s, "type-mismatch", "operand of %s must be a read_write reference to an integer scalar, found %s", s.Op, v.t)
		}
		return bNext
	case *CallStmt:
		v := c.callExpr(s.Call)
		c.info.Types[s.Call] = v.t
		if o, ok := c.info.Objects[s.Call.Fn].(*FuncObj); ok {
			for _, a := range o.Decl.Attrs {
				if a.Name == "must_use" {
					c.err(s, "must-use", "the result of %q must be used", o.Name)
				}
			}
		} else if b := builtinFuncs[s.Call.Fn.Name]; b != nil && b.mustUse {
			if _, isBuiltin := c.info.Objects[s.Call.Fn].(*BuiltinObj); isBuiltin {
				c.err(s, "must-use", "the result of %q must be used", s.Call.Fn.Name)
			}
		} else if !isInvalid(v.t) && v.t != Void {
			if _, isBuiltin := c.info.Objects[s.Call.Fn].(*BuiltinObj); isBuiltin {
				c.err(s, "must-use", "the result of the constructor must be used")
			}
		}
		return bNext
	case *Return:
		return c.returnStmt(s)
	case *Break:
		for i := len(c.targets) - 1; i >= 0; i-- {
			switch c.targets[i] {
			case 'L', 'S':
				return bBreak
			case 'C':
				c.err(s, "break", "break cannot exit a continuing block; use break if")
				return bBreak
			}
		}
		c.err(s, "break", "break must be inside a loop or switch")
		return bBreak
	case *Continue:
		for i := len(c.targets) - 1; i >= 0; i-- {
			switch c.targets[i] {
			case 'L':
				return bContinue
			case 'C':
				c.err(s, "continue", "continue cannot target an enclosing continuing block")
				return bContinue
			}
		}
		c.err(s, "continue", "continue must be inside a loop")
		return bContinue
	case *Discard:
		c.noteFeature(s, "stage:discard:fragment")
		return bNext
	case *BreakIf:
		c.boolCond(s.Cond)
		return bBreak | bNext
	case *If:
		c.checkAttrs(s.Attrs, "block", s)
		c.boolCond(s.Cond)
		b := c.block(s.Then)
		switch e := s.Else.(type) {
		case nil:
			b |= bNext
		case *If:
			b |= c.stmt(e)
		case *Block:
			b |= c.block(e)
		}
		return b
	case *Switch:
		return c.switchStmt(s)
	case *Loop:
		c.checkAttrs(s.Attrs, "block", s)
		c.checkAttrs(s.BodyAttrs, "block", s)
		c.pushTarget('L')
		c.push()
		b1 := c.seq(s.Body)
		var b2 behavior = bNext
		if s.Cont != nil {
			c.checkAttrs(s.Cont.Attrs, "block", s.Cont)
			c.pushTarget('C')
			c.push()
			b2 = c.seq(s.Cont.Stmts)
			c.pop()
			c.popTarget()
			if b2.has(bContinue) || b2.has(bReturn) {
				c.err(s.Cont, "continuing", "continuing block must not contain continue or return")
			}
		}
		c.pop()
		c.popTarget()
		return c.loopBehavior(s, b1, b2)
	case *For:
		c.checkAttrs(s.Attrs, "block", s)
		c.push()
		if s.Init != nil {
			c.stmt(s.Init)
		}
		var b1 behavior = bNext
		if s.Cond != nil {
			c.boolCond(s.Cond)
			b1 = bBreak | bNext
		}
		c.pushTarget('L')
		bb := c.block0(s.Body)
		var b2 behavior = bNext
		if s.Update != nil {
			c.pushTarget('C')
			b2 = c.stmt(s.Update)
			c.popTarget()
		}
		c.popTarget()
		c.pop()
		if b1.has(bBreak) {
			b1 = (b1 &^ bNext) | bb
		} else {
			b1 = bb
		}
		return c.loopBehavior(s, b1, b2)
	case *While:
		c.checkAttrs(s.Attrs, "block", s)
		c.boolCond(s.Cond)
		c.pushTarget('L')
		bb := c.block(s.Body)
		c.popTarget()
		return c.loopBehavior(s, bBreak|bb, bNext)
	}
	return bNext
}

// block0 checks a block whose declarations share the enclosing scope (the
// body of a for loop).
func (c *checker) block0(b *Block) behavior {
	c.checkAttrs(b.Attrs, "block", b)
	return c.seq(b.Stmts)
}

func (c *checker) loopBehavior(n Node, b1, b2 behavior) behavior {
	var r behavior
	switch {
	case b1 == bReturn && !b2.has(bContinue) && !b2.has(bReturn):
		r = bReturn
	case !b1.has(bBreak) && !b2.has(bBreak):
		r = (b1 | b2) &^ (bContinue | bNext)
	default:
		r = (b1 | b2 | bNext) &^ (bBreak | bContinue)
	}
	if r == 0 {
		c.err(n, "behavior", "loop never terminates")
	}
	return r
}

func (c *checker) returnStmt(s *Return) behavior {
	for _, t := range c.targets {
		if t == 'C' {
			c.err(s, "return", "return is not allowed inside a continuing block")
			break
		}
	}
	rt := c.retType
	if s.X == nil {
		if rt != nil && rt != Void && !isInvalid(rt) {
			c.err(s, "return-type", "return requires a value of type %s", rt)
		}
		return bReturn
	}
	v := c.expr(s.X)
	if rt == nil || rt == Void {
		c.err(s, "return-type", "a function without a return type cannot return a value")
		return bReturn
	}
	if !isInvalid(rt) {
		c.convertTo(v, rt)
	}
	return bReturn
}

func (c *checker) switchStmt(s *Switch) behavior {
	c.checkAttrs(s.Attrs, "block", s)
	c.checkAttrs(s.BodyAttrs, "block", s)
	sel := c.load(c.expr(s.X))
	var cases []*val
	defaults := 0
	for _, cl := range s.Cases {
		if cl.Default {
			defaults++
		}
		seenDefault := false
		for _, e := range cl.Selectors {
			if e == nil {
				if seenDefault {
					c.err(cl, "switch", "default appears more than once in a case selector list")
				}
				seenDefault = true
				defaults++
				continue
			}
			v := c.load(c.expr(e))
			if !isInvalid(v.t) && v.k != kConst {
				c.err(e, "switch", "case selector must be a const-expression")
				v = invalidVal()
			}
			cases = append(cases, v)
		}
	}
	if defaults != 1 {
		c.err(s, "switch", "switch must have exactly one default clause, found %d", defaults)
	}
	all := append([]*val{sel}, cases...)
	var T Type
	if !anyInvalid(all) {
		var cands []sig
		for _, t := range []*ScalarType{I32, U32} {
			ps := make([]Type, len(all))
			for i := range ps {
				ps[i] = t
			}
			cands = append(cands, sig{params: ps, ret: t})
		}
		if len(all) == 1 {
			cands = cands[:0]
			for _, t := range []*ScalarType{I32, U32} {
				cands = append(cands, sig{params: []Type{t}, ret: t})
			}
		}
		if sg := c.pick(s, "switch", all, cands); sg != nil {
			T = sg.ret
		}
	}
	if T != nil {
		seen := map[int64]bool{}
		for _, v := range cases {
			cv := c.convertTo(v, T)
			if cv.c != nil {
				if seen[cv.c.Int] {
					c.err(v.expr, "switch", "duplicate case selector %d", cv.c.Int)
				}
				seen[cv.c.Int] = true
			}
		}
		c.convertTo(sel, T)
	}
	c.pushTarget('S')
	var b behavior
	for _, cl := range s.Cases {
		b |= c.block(cl.Body)
	}
	c.popTarget()
	if b.has(bBreak) {
		b = (b | bNext) &^ bBreak
	}
	return b
}

func (c *checker) assign(s *Assign) {
	if s.LHS == nil {
		v := c.expr(s.RHS)
		if isInvalid(v.t) {
			return
		}
		v = c.materialize(v)
		if isInvalid(v.t) {
			return
		}
		switch v.t.(type) {
		case *PtrType, *TextureType, *SamplerType:
		default:
			if !constructible(v.t) {
				c.err(s, "type-mismatch", "cannot assign a value of type %s to _", v.t)
			}
		}
		return
	}
	l := c.expr(s.LHS)
	if isInvalid(l.t) {
		c.expr(s.RHS)
		return
	}
	r, ok := l.t.(*RefType)
	if !ok {
		c.err(s.LHS, "assign-target", "the left-hand side of an assignment must be a reference, found %s", l.t)
		c.expr(s.RHS)
		return
	}
	if !r.Access.Writable() {
		c.err(s.LHS, "access-mode", "cannot assign through a %s reference (address space %s)", r.Access, r.AS)
	}
	if !constructible(r.Elem) {
		c.err(s.LHS, "assign-target", "cannot assign to a value of type %s", r.Elem)
	}
	rhs := c.load(c.expr(s.RHS))
	if isInvalid(rhs.t) {
		return
	}
	if s.Op == "=" {
		c.convertTo(rhs, r.Elem)
		return
	}
	if r.Access != AccessReadWrite {
		c.err(s.LHS, "access-mode", "compound assignment requires a read_write reference")
	}
	op := strings.TrimSuffix(s.Op, "=")
	cur := &val{t: r.Elem, k: kRuntime, expr: s.LHS}
	sg := c.pick(s, op, []*val{cur, rhs}, opCands[op])
	if sg == nil {
		return
	}
	if !Same(sg.ret, r.Elem) {
		c.err(s, "type-mismatch", "%s= would produce %s, which cannot be stored in %s", op, sg.ret, r.Elem)
		return
	}
	c.convertTo(rhs, sg.params[1])
}

// ---- local declarations ----

func (c *checker) localDecl(d Decl) {
	switch d := d.(type) {
	case *Var:
		c.localVar(d)
	case *Value:
		c.localValue(d)
	}
}

func (c *checker) localVar(d *Var) {
	o := &VarObj{Name: d.Name, Decl: d, AS: ASFunction, Access: AccessReadWrite}
	c.info.Decls[d] = o
	c.checkAttrs(d.Attrs, "localvar", d)
	if len(d.Template) > 0 {
		as := c.enumArg(d.Template[0], "address_space")
		if as != "function" {
			if as != "" {
				c.err(d, "address-space", "function-scope variables must be in the function address space")
			}
		}
		if len(d.Template) > 1 {
			c.err(d, "address-space", "access mode must not be written for function variables")
		}
	}
	var store Type
	var init *val
	if d.Type != nil {
		store = c.typeSpec(d.Type)
	}
	if d.Init != nil {
		init = c.expr(d.Init)
	}
	switch {
	case store == nil && init != nil:
		init = c.materialize(init)
		store = init.t
	case store != nil && init != nil:
		c.convertTo(init, store)
	case store == nil:
		c.err(d, "var-type", "a variable needs a type or an initializer")
		store = Invalid
	}
	if !isInvalid(store) && !constructible(store) {
		c.err(d, "var-type", "function-scope variable type %s must be constructible", store)
		store = Invalid
	}
	o.Store = store
	c.info.DeclTypes[d] = store
	c.declareLocal(d.Name, o, d)
}

func (c *checker) localValue(d *Value) {
	o := &ValueObj{Name: d.Name, Kind: d.Kind, Decl: d, Local: true, ID: -1}
	c.info.Decls[d] = o
	if d.Kind == "override" {
		c.err(d, "override-scope", "override declarations are only allowed at module scope")
		return
	}
	var decl Type
	if d.Type != nil {
		decl = c.typeSpec(d.Type)
	}
	v := c.expr(d.Init)
	if isInvalid(v.t) {
		o.Type = Invalid
		c.declareLocal(d.Name, o, d)
		return
	}
	if d.Kind == "const" {
		v = c.load(v)
		if v.k != kConst {
			c.err(d.Init, "const-init", "the initializer of a const declaration must be a const-expression")
		}
		if decl != nil && !isInvalid(decl) {
			v = c.convertTo(v, decl)
		}
		o.Type, o.Val = v.t, v.c
		if !isInvalid(v.t) && !constructible(stripAbstract(v.t)) {
			c.err(d, "const-type", "const type %s must be constructible", v.t)
		}
	} else {
		if decl != nil && !isInvalid(decl) {
			v = c.convertTo(v, decl)
		} else {
			v = c.materialize(v)
		}
		o.Type = v.t
		switch v.t.(type) {
		case *PtrType:
		case *TextureType, *SamplerType:
			if !c.lang["texture_and_sampler_let"] {
				c.err(d, "let-type", "let of texture or sampler type requires texture_and_sampler_let")
			}
		default:
			if !isInvalid(v.t) && !constructible(v.t) {
				c.err(d, "let-type", "let type %s must be constructible or a pointer", v.t)
			}
		}
	}
	c.info.DeclTypes[d] = o.Type
	c.declareLocal(d.Name, o, d)
}

func stripAbstract(t Type) Type {
	if IsAbstract(t) {
		return concretize(t)
	}
	return t
}

func (c *checker) constAssert(a *ConstAssert) {
	v := c.load(c.expr(a.X))
	if isInvalid(v.t) {
		return
	}
	v = c.convertTo(v, Bool)
	if isInvalid(v.t) {
		return
	}
	if v.k != kConst {
		c.err(a, "const-assert", "const_assert requires a const-expression")
		return
	}
	if v.c != nil && !v.c.Bool {
		c.err(a, "const-assert", "const_assert failed")
	}
}
