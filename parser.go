package wgsl

import "fmt"

// Options configures Parse.
type Options struct {
	Name   string // source name used in diagnostics
	Limits Limits // zero values take DefaultLimits
}

// Parse parses a WGSL module. It implements the grammar of the WGSL
// Candidate Recommendation Draft of 26 October 2026 (see SPEC.md). The first
// lexical or syntax error is returned as *Error. Parse does not do name
// resolution or type checking; see Check.
func Parse(src string, opts ...Options) (*Module, error) {
	var o Options
	if len(opts) > 0 {
		o = opts[0]
	}
	s := NewSource(o.Name, src)
	toks, err := Lex(s, o.Limits)
	if err != nil {
		return nil, err
	}
	p := &parser{src: s, toks: toks, max: o.Limits.withDefaults().MaxDepth}
	return p.run()
}

// ParseSource is like Parse but reuses a *Source and returns the module with
// its Source for position lookups.
func ParseSource(s *Source, lim Limits) (*Module, error) {
	toks, err := Lex(s, lim)
	if err != nil {
		return nil, err
	}
	p := &parser{src: s, toks: toks, max: lim.withDefaults().MaxDepth}
	return p.run()
}

type parser struct {
	src   *Source
	toks  []Token
	i     int
	depth int
	max   int
}

func (p *parser) run() (m *Module, err error) {
	defer func() {
		if r := recover(); r != nil {
			e, ok := r.(*Error)
			if !ok {
				panic(r)
			}
			m, err = nil, e
		}
	}()
	return p.module(), nil
}

func (p *parser) fail(off int, rule, f string, a ...any) {
	panic(p.src.errAt(off, rule, f, a...))
}

func (p *parser) enter() {
	p.depth++
	if p.depth > p.max {
		p.fail(p.tok().Off, "limit-depth", "nesting deeper than %d levels", p.max)
	}
}
func (p *parser) leave() { p.depth-- }

func (p *parser) tok() Token { return p.toks[p.i] }
func (p *parser) peek(n int) Token {
	if p.i+n >= len(p.toks) {
		return p.toks[len(p.toks)-1]
	}
	return p.toks[p.i+n]
}
func (p *parser) next() Token {
	t := p.toks[p.i]
	if t.Kind != TEOF {
		p.i++
	}
	return t
}

func describe(t Token) string {
	if t.Kind == TEOF {
		return "end of input"
	}
	return fmt.Sprintf("%q", t.Text)
}

// is reports whether the current token is the punctuation or keyword s.
func (p *parser) is(s string) bool {
	t := p.tok()
	return (t.Kind == TPunct || t.Kind == TKeyword) && t.Text == s
}
func (p *parser) isAt(n int, s string) bool {
	t := p.peek(n)
	return (t.Kind == TPunct || t.Kind == TKeyword) && t.Text == s
}
func (p *parser) accept(s string) bool {
	if p.is(s) {
		p.i++
		return true
	}
	return false
}
func (p *parser) expect(s string) Token {
	if !p.is(s) {
		p.fail(p.tok().Off, "syntax", "expected %q, found %s", s, describe(p.tok()))
	}
	return p.next()
}
func (p *parser) ident() Token {
	t := p.tok()
	if t.Kind != TIdent {
		if t.Kind == TKeyword {
			p.fail(t.Off, "syntax", "keyword %q cannot be used as a name", t.Text)
		}
		p.fail(t.Off, "syntax", "expected identifier, found %s", describe(t))
	}
	return p.next()
}
func (p *parser) last() int { return p.toks[p.i-1].End }

func (p *parser) module() *Module {
	m := &Module{Span: Span{0, len(p.src.Text)}}
	for p.is("enable") || p.is("requires") || p.is("diagnostic") {
		m.Directives = append(m.Directives, p.directive())
	}
	for p.tok().Kind != TEOF {
		if p.accept(";") {
			continue
		}
		if p.is("const_assert") {
			m.Decls = append(m.Decls, p.constAssert())
			continue
		}
		m.Decls = append(m.Decls, p.globalDecl())
	}
	return m
}

func (p *parser) directive() *Directive {
	start := p.tok().Off
	d := &Directive{Kind: p.next().Text}
	switch d.Kind {
	case "diagnostic":
		p.expect("(")
		d.Names = append(d.Names, p.identPattern().Text)
		p.expect(",")
		d.Names = append(d.Names, p.diagRule())
		p.accept(",")
		p.expect(")")
	default:
		for {
			d.Names = append(d.Names, p.identPattern().Text)
			if !p.accept(",") {
				break
			}
			if p.is(";") {
				break
			}
		}
	}
	p.expect(";")
	d.Span = Span{start, p.last()}
	return d
}

// identPattern accepts any ident_pattern_token, keywords included.
func (p *parser) identPattern() Token {
	t := p.tok()
	if t.Kind != TIdent && t.Kind != TKeyword {
		p.fail(t.Off, "syntax", "expected identifier, found %s", describe(t))
	}
	return p.next()
}

func (p *parser) diagRule() string {
	r := p.identPattern().Text
	if p.is(".") {
		p.next()
		r += "." + p.identPattern().Text
	}
	return r
}

func (p *parser) constAssert() *ConstAssert {
	start := p.expect("const_assert").Off
	x := p.expr()
	p.expect(";")
	return &ConstAssert{Span{start, p.last()}, x}
}

func (p *parser) attrs() []*Attr {
	var as []*Attr
	for p.is("@") {
		as = append(as, p.attr())
	}
	return as
}

func (p *parser) attr() *Attr {
	start := p.expect("@").Off
	name := p.identPattern()
	a := &Attr{Name: name.Text}
	exprArgs := func(min, max int) {
		p.expect("(")
		for {
			a.Args = append(a.Args, p.expr())
			if !p.accept(",") || p.is(")") {
				break
			}
		}
		p.expect(")")
		if n := len(a.Args); n < min || n > max {
			p.fail(start, "attribute-args", "@%s takes %d..%d arguments, got %d", a.Name, min, max, n)
		}
	}
	nameArgs := func(min, max int) {
		p.expect("(")
		for {
			t := p.ident()
			a.Args = append(a.Args, &Ident{Span: Span{t.Off, t.End}, Name: t.Text})
			if !p.accept(",") || p.is(")") {
				break
			}
		}
		p.expect(")")
		if n := len(a.Args); n < min || n > max {
			p.fail(start, "attribute-args", "@%s takes %d..%d arguments, got %d", a.Name, min, max, n)
		}
	}
	switch a.Name {
	case "compute", "const", "fragment", "invariant", "must_use", "vertex":
		if p.is("(") {
			p.fail(p.tok().Off, "attribute-args", "@%s takes no arguments", a.Name)
		}
	case "align", "binding", "blend_src", "group", "id", "location", "size", "subgroup_size":
		exprArgs(1, 1)
	case "workgroup_size":
		exprArgs(1, 3)
	case "builtin":
		nameArgs(1, 1)
	case "interpolate":
		nameArgs(1, 2)
	case "diagnostic":
		p.expect("(")
		t := p.identPattern()
		a.Args = append(a.Args, &Ident{Span: Span{t.Off, t.End}, Name: t.Text})
		p.expect(",")
		r := p.tok()
		name := p.diagRule()
		a.Args = append(a.Args, &Ident{Span: Span{r.Off, p.last()}, Name: name})
		p.accept(",")
		p.expect(")")
	default:
		if p.is("(") {
			exprArgs(0, 1<<30)
		}
	}
	a.Span = Span{start, p.last()}
	return a
}

func (p *parser) globalDecl() Decl {
	attrs := p.attrs()
	t := p.tok()
	noAttrs := func() {
		if len(attrs) > 0 {
			p.fail(attrs[0].Start, "syntax", "attributes are not allowed on %q declarations", t.Text)
		}
	}
	switch {
	case p.is("fn"):
		return p.funcDecl(attrs)
	case p.is("var"):
		v := p.varDecl(attrs)
		p.expect(";")
		v.End = p.last()
		return v
	case p.is("override"):
		start := p.next().Off
		v := &Value{Kind: "override", Attrs: attrs}
		p.optTypedIdent(&v.Name, &v.Type)
		if p.accept("=") {
			v.Init = p.expr()
		}
		p.expect(";")
		v.Span = p.spanFrom(attrs, start)
		return v
	case p.is("const"):
		noAttrs()
		start := p.next().Off
		v := &Value{Kind: "const"}
		p.optTypedIdent(&v.Name, &v.Type)
		p.expect("=")
		v.Init = p.expr()
		p.expect(";")
		v.Span = Span{start, p.last()}
		return v
	case p.is("alias"):
		noAttrs()
		start := p.next().Off
		a := &Alias{Name: p.ident().Text}
		p.expect("=")
		a.Type = p.typeSpec()
		p.expect(";")
		a.Span = Span{start, p.last()}
		return a
	case p.is("struct"):
		noAttrs()
		return p.structDecl()
	}
	p.fail(t.Off, "syntax", "expected a global declaration, found %s", describe(t))
	return nil
}

func (p *parser) spanFrom(attrs []*Attr, start int) Span {
	if len(attrs) > 0 {
		start = attrs[0].Start
	}
	return Span{start, p.last()}
}

func (p *parser) optTypedIdent(name *string, typ **Ident) {
	*name = p.ident().Text
	if p.accept(":") {
		*typ = p.typeSpec()
	}
}

func (p *parser) typeSpec() *Ident {
	t := p.ident()
	id := &Ident{Span: Span{t.Off, t.End}, Name: t.Text}
	p.templateArgs(id)
	return id
}

func (p *parser) templateArgs(id *Ident) {
	if p.tok().Kind != TTmplStart {
		return
	}
	p.next()
	p.enter()
	for {
		id.Targs = append(id.Targs, p.expr())
		if !p.accept(",") || p.tok().Kind == TTmplEnd {
			break
		}
	}
	if p.tok().Kind != TTmplEnd {
		p.fail(p.tok().Off, "syntax", "expected '>' to close template list, found %s", describe(p.tok()))
	}
	p.next()
	p.leave()
	id.End = p.last()
}

func (p *parser) varDecl(attrs []*Attr) *Var {
	start := p.expect("var").Off
	v := &Var{Attrs: attrs}
	if p.tok().Kind == TTmplStart {
		p.next()
		p.enter()
		for {
			v.Template = append(v.Template, p.expr())
			if !p.accept(",") || p.tok().Kind == TTmplEnd {
				break
			}
		}
		if p.tok().Kind != TTmplEnd {
			p.fail(p.tok().Off, "syntax", "expected '>' to close template list, found %s", describe(p.tok()))
		}
		p.next()
		p.leave()
	}
	p.optTypedIdent(&v.Name, &v.Type)
	if p.accept("=") {
		v.Init = p.expr()
	}
	v.Span = p.spanFrom(attrs, start)
	return v
}

func (p *parser) structDecl() *Struct {
	start := p.expect("struct").Off
	s := &Struct{Name: p.ident().Text}
	p.expect("{")
	for {
		mstart := p.tok().Off
		attrs := p.attrs()
		m := &StructMember{Attrs: attrs, Name: p.ident().Text}
		p.expect(":")
		m.Type = p.typeSpec()
		m.Span = Span{mstart, p.last()}
		s.Members = append(s.Members, m)
		if !p.accept(",") || p.is("}") {
			break
		}
	}
	p.expect("}")
	s.Span = Span{start, p.last()}
	return s
}

func (p *parser) funcDecl(attrs []*Attr) *Func {
	start := p.expect("fn").Off
	f := &Func{Attrs: attrs, Name: p.ident().Text}
	p.expect("(")
	for !p.is(")") {
		pstart := p.tok().Off
		pa := p.attrs()
		pr := &Param{Attrs: pa, Name: p.ident().Text}
		p.expect(":")
		pr.Type = p.typeSpec()
		pr.Span = Span{pstart, p.last()}
		f.Params = append(f.Params, pr)
		if !p.accept(",") {
			break
		}
	}
	p.expect(")")
	if p.accept("->") {
		f.RetAttrs = p.attrs()
		f.Ret = p.typeSpec()
	}
	f.Body = p.block()
	f.Span = p.spanFrom(attrs, start)
	return f
}

// ---- statements ----

func (p *parser) block() *Block {
	start := p.tok().Off
	b := &Block{Attrs: p.attrs()}
	b.Stmts = p.braced()
	b.Span = Span{start, p.last()}
	return b
}

func (p *parser) braced() []Stmt {
	p.expect("{")
	p.enter()
	var ss []Stmt
	for !p.is("}") {
		if p.tok().Kind == TEOF {
			p.fail(p.tok().Off, "syntax", "unexpected end of input; missing '}'")
		}
		ss = append(ss, p.stmt())
	}
	p.leave()
	p.expect("}")
	return ss
}

func (p *parser) stmt() Stmt {
	p.enter()
	defer p.leave()
	start := p.tok().Off
	attrs := p.attrs()
	t := p.tok()
	if len(attrs) > 0 && !(p.is("for") || p.is("if") || p.is("loop") || p.is("switch") || p.is("while") || p.is("{")) {
		p.fail(attrs[0].Start, "syntax", "attributes are not allowed on this statement")
	}
	switch {
	case p.is("{"):
		b := &Block{Attrs: attrs}
		b.Stmts = p.braced()
		b.Span = Span{start, p.last()}
		return b
	case p.is("if"):
		return p.ifStmt(attrs, start)
	case p.is("for"):
		return p.forStmt(attrs, start)
	case p.is("while"):
		p.next()
		w := &While{Attrs: attrs, Cond: p.expr(), Body: p.block()}
		w.Span = Span{start, p.last()}
		return w
	case p.is("loop"):
		return p.loopStmt(attrs, start)
	case p.is("switch"):
		return p.switchStmt(attrs, start)
	case p.is("return"):
		p.next()
		r := &Return{}
		if !p.is(";") {
			r.X = p.expr()
		}
		p.expect(";")
		r.Span = Span{start, p.last()}
		return r
	case p.is("break"):
		p.next()
		p.expect(";")
		return &Break{Span{start, p.last()}}
	case p.is("continue"):
		p.next()
		p.expect(";")
		return &Continue{Span{start, p.last()}}
	case p.is("discard"):
		p.next()
		p.expect(";")
		return &Discard{Span{start, p.last()}}
	case p.is(";"):
		p.next()
		return &Empty{Span{start, p.last()}}
	case p.is("const_assert"):
		a := p.constAssert()
		return &AssertStmt{a.Span, a}
	}
	_ = t
	s := p.simpleStmt()
	p.expect(";")
	switch x := s.(type) {
	case *Assign:
		x.End = p.last()
	case *IncDec:
		x.End = p.last()
	case *CallStmt:
		x.End = p.last()
	case *DeclStmt:
		x.End = p.last()
	}
	return s
}

// simpleStmt parses a statement without its terminating semicolon: a
// declaration, assignment, increment/decrement or call.
func (p *parser) simpleStmt() Stmt {
	start := p.tok().Off
	switch {
	case p.is("var"):
		v := p.varDecl(nil)
		return &DeclStmt{Span{start, p.last()}, v}
	case p.is("let") || p.is("const"):
		kind := p.next().Text
		v := &Value{Kind: kind}
		p.optTypedIdent(&v.Name, &v.Type)
		p.expect("=")
		v.Init = p.expr()
		v.Span = Span{start, p.last()}
		return &DeclStmt{v.Span, v}
	case p.is("_"):
		p.next()
		p.expect("=")
		a := &Assign{Op: "=", RHS: p.expr()}
		a.Span = Span{start, p.last()}
		return a
	}
	t := p.tok()
	if t.Kind != TIdent && !p.is("(") && !p.is("*") && !p.is("&") {
		p.fail(t.Off, "syntax", "expected a statement, found %s", describe(t))
	}
	e := p.unary()
	switch {
	case p.isAssignOp():
		if !isLHS(e) {
			p.fail(t.Off, "syntax", "invalid assignment target")
		}
		op := p.next().Text
		a := &Assign{LHS: e, Op: op, RHS: p.expr()}
		a.Span = Span{start, p.last()}
		return a
	case p.is("++") || p.is("--"):
		if !isLHS(e) {
			p.fail(t.Off, "syntax", "invalid increment target")
		}
		op := p.next().Text
		return &IncDec{Span{start, p.last()}, e, op}
	}
	if c, ok := e.(*Call); ok {
		return &CallStmt{Span{start, p.last()}, c}
	}
	p.fail(t.Off, "syntax", "expression is not a statement; expected assignment or call, found %s", describe(p.tok()))
	return nil
}

func (p *parser) isAssignOp() bool {
	t := p.tok()
	if t.Kind != TPunct {
		return false
	}
	switch t.Text {
	case "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=":
		return true
	}
	return false
}

func isLHS(e Expr) bool {
	switch x := e.(type) {
	case *Ident:
		return len(x.Targs) == 0
	case *Index:
		return isLHS(x.X)
	case *Member:
		return isLHS(x.X)
	case *Paren:
		return isLHS(x.X)
	case *Unary:
		return (x.Op == "*" || x.Op == "&") && isLHS(x.X)
	}
	return false
}

func (p *parser) ifStmt(attrs []*Attr, start int) *If {
	p.expect("if")
	n := &If{Attrs: attrs, Cond: p.expr(), Then: p.block()}
	if p.accept("else") {
		if p.is("if") {
			p.enter()
			n.Else = p.ifStmt(nil, p.tok().Off)
			p.leave()
		} else {
			n.Else = p.block()
		}
	}
	n.Span = Span{start, p.last()}
	return n
}

func (p *parser) forStmt(attrs []*Attr, start int) *For {
	p.expect("for")
	p.expect("(")
	f := &For{Attrs: attrs}
	if !p.is(";") {
		f.Init = p.simpleStmt()
	}
	p.expect(";")
	if !p.is(";") {
		f.Cond = p.expr()
	}
	p.expect(";")
	if !p.is(")") {
		u := p.simpleStmt()
		if d, ok := u.(*DeclStmt); ok {
			p.fail(d.Start, "syntax", "a declaration is not allowed in a for update")
		}
		f.Update = u
	}
	p.expect(")")
	f.Body = p.block()
	f.Span = Span{start, p.last()}
	return f
}

func (p *parser) loopStmt(attrs []*Attr, start int) *Loop {
	p.expect("loop")
	l := &Loop{Attrs: attrs, BodyAttrs: p.attrs()}
	p.expect("{")
	p.enter()
	for !p.is("}") && !p.is("continuing") {
		if p.tok().Kind == TEOF {
			p.fail(p.tok().Off, "syntax", "unexpected end of input; missing '}'")
		}
		l.Body = append(l.Body, p.stmt())
	}
	if p.is("continuing") {
		cstart := p.next().Off
		c := &Continuing{Attrs: p.attrs()}
		p.expect("{")
		for !p.is("}") {
			if p.tok().Kind == TEOF {
				p.fail(p.tok().Off, "syntax", "unexpected end of input; missing '}'")
			}
			if p.is("break") && p.isAt(1, "if") {
				bstart := p.next().Off
				p.next()
				b := &BreakIf{Cond: p.expr()}
				p.expect(";")
				b.Span = Span{bstart, p.last()}
				c.Stmts = append(c.Stmts, b)
				if !p.is("}") {
					p.fail(p.tok().Off, "syntax", "`break if` must be the last statement of a continuing block")
				}
				break
			}
			c.Stmts = append(c.Stmts, p.stmt())
		}
		p.expect("}")
		c.Span = Span{cstart, p.last()}
		l.Cont = c
	}
	p.leave()
	p.expect("}")
	l.Span = Span{start, p.last()}
	return l
}

func (p *parser) switchStmt(attrs []*Attr, start int) *Switch {
	p.expect("switch")
	s := &Switch{Attrs: attrs, X: p.expr(), BodyAttrs: p.attrs()}
	p.expect("{")
	p.enter()
	for !p.is("}") {
		cstart := p.tok().Off
		c := &Case{}
		switch {
		case p.accept("case"):
			for {
				if p.accept("default") {
					c.Selectors = append(c.Selectors, nil)
				} else {
					c.Selectors = append(c.Selectors, p.expr())
				}
				if !p.accept(",") || p.is(":") || p.is("{") || p.is("@") {
					break
				}
			}
		case p.accept("default"):
			c.Default = true
		default:
			p.fail(p.tok().Off, "syntax", "expected 'case' or 'default', found %s", describe(p.tok()))
		}
		p.accept(":")
		c.Body = p.block()
		c.Span = Span{cstart, p.last()}
		s.Cases = append(s.Cases, c)
	}
	p.leave()
	p.expect("}")
	s.Span = Span{start, p.last()}
	return s
}

// ---- expressions ----

func (p *parser) expr() Expr {
	p.enter()
	defer p.leave()
	x := p.unary()
	return p.exprFrom(x)
}

func bin(op string, x, y Expr) *Binary {
	return &Binary{Span: Span{x.NodeSpan().Start, y.NodeSpan().End}, Op: op, X: x, Y: y}
}

func (p *parser) exprFrom(x Expr) Expr {
	t := p.tok()
	if t.Kind != TPunct {
		return x
	}
	switch t.Text {
	case "&", "^", "|":
		return p.chain(x, func() Expr { return p.unary() }, t.Text)
	}
	x = p.relational(x)
	t = p.tok()
	if t.Kind == TPunct && (t.Text == "&&" || t.Text == "||") {
		return p.chain(x, func() Expr { return p.relational(p.unary()) }, t.Text)
	}
	return x
}

// chain parses a left-associative same-operator chain. Each link counts
// towards the depth limit because it deepens the tree.
func (p *parser) chain(x Expr, operand func() Expr, op string) Expr {
	n := 0
	defer func() { p.depth -= n }()
	for p.is(op) {
		p.next()
		n++
		p.enter()
		x = bin(op, x, operand())
	}
	return x
}

var cmpOps = map[string]bool{"<": true, ">": true, "<=": true, ">=": true, "==": true, "!=": true}

func (p *parser) relational(x Expr) Expr {
	x = p.shiftLevel(x)
	if t := p.tok(); t.Kind == TPunct && cmpOps[t.Text] {
		p.next()
		y := p.shiftLevel(p.unary())
		x = bin(t.Text, x, y)
	}
	return x
}

func (p *parser) shiftLevel(x Expr) Expr {
	if t := p.tok(); t.Kind == TPunct && (t.Text == "<<" || t.Text == ">>") {
		p.next()
		return bin(t.Text, x, p.unary())
	}
	n := 0
	defer func() { p.depth -= n }()
	mul := func(x Expr) Expr {
		for {
			t := p.tok()
			if t.Kind == TPunct && (t.Text == "*" || t.Text == "/" || t.Text == "%") {
				p.next()
				n++
				p.enter()
				x = bin(t.Text, x, p.unary())
				continue
			}
			return x
		}
	}
	x = mul(x)
	for {
		t := p.tok()
		if t.Kind == TPunct && (t.Text == "+" || t.Text == "-") {
			p.next()
			n++
			p.enter()
			x = bin(t.Text, x, mul(p.unary()))
			continue
		}
		return x
	}
}

func (p *parser) unary() Expr {
	p.enter()
	defer p.leave()
	t := p.tok()
	if t.Kind == TPunct {
		switch t.Text {
		case "!", "&", "*", "-", "~":
			p.next()
			x := p.unary()
			return &Unary{Span{t.Off, x.NodeSpan().End}, t.Text, x}
		}
	}
	x := p.primary()
	for {
		switch {
		case p.is("."):
			p.next()
			n := p.ident()
			x = &Member{Span{x.NodeSpan().Start, n.End}, x, n.Text}
		case p.is("["):
			p.next()
			p.enter()
			i := p.expr()
			p.leave()
			p.expect("]")
			x = &Index{Span{x.NodeSpan().Start, p.last()}, x, i}
		default:
			return x
		}
		// a postfix chain makes the tree deeper
		p.depth++
		defer func() { p.depth-- }()
		if p.depth > p.max {
			p.fail(p.tok().Off, "limit-depth", "nesting deeper than %d levels", p.max)
		}
	}
}

func (p *parser) primary() Expr {
	t := p.tok()
	switch t.Kind {
	case TInt:
		p.next()
		return &IntLit{Span{t.Off, t.End}, t.Text}
	case TFloat:
		p.next()
		return &FloatLit{Span{t.Off, t.End}, t.Text}
	case TKeyword:
		if t.Text == "true" || t.Text == "false" {
			p.next()
			return &BoolLit{Span{t.Off, t.End}, t.Text == "true"}
		}
	case TPunct:
		if t.Text == "(" {
			p.next()
			x := p.expr()
			p.expect(")")
			return &Paren{Span{t.Off, p.last()}, x}
		}
	case TIdent:
		p.next()
		id := &Ident{Span: Span{t.Off, t.End}, Name: t.Text}
		p.templateArgs(id)
		if p.is("(") {
			c := &Call{Fn: id}
			p.next()
			p.enter()
			for !p.is(")") {
				c.Args = append(c.Args, p.expr())
				if !p.accept(",") {
					break
				}
			}
			p.leave()
			p.expect(")")
			c.Span = Span{t.Off, p.last()}
			return c
		}
		return id
	}
	p.fail(t.Off, "syntax", "expected an expression, found %s", describe(t))
	return nil
}
