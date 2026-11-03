package wgsl

import (
	"strconv"
	"strings"
)

// Print returns canonical WGSL text for m. Print(Parse(src)) parses back to
// an AST equal (see Equal) to the original. Expressions are parenthesised
// where the WGSL grammar requires it, so trees built by transforms print
// validly even when they carry no Paren nodes.
func Print(m *Module) string {
	p := &printer{}
	p.module(m)
	return p.b.String()
}

// PrintExpr returns WGSL text for a single expression.
func PrintExpr(e Expr) string {
	p := &printer{}
	p.expr(e)
	return p.b.String()
}

// PrintStmt returns WGSL text for a single statement at indent level 0.
func PrintStmt(s Stmt) string {
	p := &printer{}
	p.stmt(s)
	return p.b.String()
}

type printer struct {
	b      strings.Builder
	indent int
}

func (p *printer) w(s string) { p.b.WriteString(s) }
func (p *printer) nl() {
	p.b.WriteByte('\n')
	for i := 0; i < p.indent; i++ {
		p.b.WriteString("    ")
	}
}

func (p *printer) module(m *Module) {
	for _, d := range m.Directives {
		switch d.Kind {
		case "diagnostic":
			p.w("diagnostic(" + d.Names[0] + ", " + d.Names[1] + ");\n")
		default:
			p.w(d.Kind + " " + strings.Join(d.Names, ", ") + ";\n")
		}
	}
	if len(m.Directives) > 0 && len(m.Decls) > 0 {
		p.w("\n")
	}
	for i, d := range m.Decls {
		if i > 0 && !(simpleDecl(m.Decls[i-1]) && simpleDecl(d)) {
			p.w("\n")
		}
		p.decl(d)
		p.w("\n")
	}
}

func simpleDecl(d Decl) bool {
	switch d.(type) {
	case *Var, *Value, *Alias, *ConstAssert:
		return true
	}
	return false
}

func (p *printer) attrs(as []*Attr, sep string) {
	for _, a := range as {
		p.attr(a)
		p.w(sep)
	}
}

func (p *printer) attr(a *Attr) {
	p.w("@" + a.Name)
	if len(a.Args) > 0 {
		p.w("(")
		for i, x := range a.Args {
			if i > 0 {
				p.w(", ")
			}
			p.expr(x)
		}
		p.w(")")
	}
}

func (p *printer) decl(d Decl) {
	switch d := d.(type) {
	case *Func:
		if len(d.Attrs) > 0 {
			for i, a := range d.Attrs {
				if i > 0 {
					p.w(" ")
				}
				p.attr(a)
			}
			p.w("\n")
		}
		p.w("fn " + d.Name + "(")
		for i, pr := range d.Params {
			if i > 0 {
				p.w(", ")
			}
			p.attrs(pr.Attrs, " ")
			p.w(pr.Name + ": ")
			p.expr(pr.Type)
		}
		p.w(")")
		if d.Ret != nil {
			p.w(" -> ")
			p.attrs(d.RetAttrs, " ")
			p.expr(d.Ret)
		}
		p.w(" ")
		p.block(d.Body)
	case *Var:
		p.varDecl(d)
		p.w(";")
	case *Value:
		p.value(d)
		p.w(";")
	case *Alias:
		p.w("alias " + d.Name + " = ")
		p.expr(d.Type)
		p.w(";")
	case *Struct:
		p.w("struct " + d.Name + " {")
		p.indent++
		for _, m := range d.Members {
			p.nl()
			p.attrs(m.Attrs, " ")
			p.w(m.Name + ": ")
			p.expr(m.Type)
			p.w(",")
		}
		p.indent--
		p.nl()
		p.w("}")
	case *ConstAssert:
		p.w("const_assert ")
		p.expr(d.X)
		p.w(";")
	}
}

func (p *printer) varDecl(v *Var) {
	p.attrs(v.Attrs, " ")
	p.w("var")
	if len(v.Template) > 0 {
		p.w("<")
		for i, x := range v.Template {
			if i > 0 {
				p.w(", ")
			}
			p.expr(x)
		}
		p.w(">")
	}
	p.w(" " + v.Name)
	if v.Type != nil {
		p.w(": ")
		p.expr(v.Type)
	}
	if v.Init != nil {
		p.w(" = ")
		p.expr(v.Init)
	}
}

func (p *printer) value(v *Value) {
	p.attrs(v.Attrs, " ")
	p.w(v.Kind + " " + v.Name)
	if v.Type != nil {
		p.w(": ")
		p.expr(v.Type)
	}
	if v.Init != nil {
		p.w(" = ")
		p.expr(v.Init)
	}
}

func (p *printer) block(b *Block) {
	p.attrs(b.Attrs, " ")
	p.stmts(b.Stmts)
}

func (p *printer) stmts(ss []Stmt) {
	p.w("{")
	p.indent++
	for _, s := range ss {
		p.nl()
		p.stmt(s)
	}
	p.indent--
	p.nl()
	p.w("}")
}

func (p *printer) stmt(s Stmt) {
	switch s := s.(type) {
	case *Block:
		p.block(s)
	case *If:
		p.attrs(s.Attrs, " ")
		p.ifChain(s)
	case *Switch:
		p.attrs(s.Attrs, " ")
		p.w("switch ")
		p.expr(s.X)
		p.w(" ")
		p.attrs(s.BodyAttrs, " ")
		p.w("{")
		p.indent++
		for _, c := range s.Cases {
			p.nl()
			if c.Default {
				p.w("default: ")
			} else {
				p.w("case ")
				for i, x := range c.Selectors {
					if i > 0 {
						p.w(", ")
					}
					if x == nil {
						p.w("default")
					} else {
						p.expr(x)
					}
				}
				p.w(": ")
			}
			p.block(c.Body)
		}
		p.indent--
		p.nl()
		p.w("}")
	case *Loop:
		p.attrs(s.Attrs, " ")
		p.w("loop ")
		p.attrs(s.BodyAttrs, " ")
		p.w("{")
		p.indent++
		for _, x := range s.Body {
			p.nl()
			p.stmt(x)
		}
		if s.Cont != nil {
			p.nl()
			p.w("continuing ")
			p.attrs(s.Cont.Attrs, " ")
			p.stmts(s.Cont.Stmts)
		}
		p.indent--
		p.nl()
		p.w("}")
	case *For:
		p.attrs(s.Attrs, " ")
		p.w("for (")
		if s.Init != nil {
			p.simple(s.Init)
		}
		p.w(";")
		if s.Cond != nil {
			p.w(" ")
			p.expr(s.Cond)
		}
		p.w(";")
		if s.Update != nil {
			p.w(" ")
			p.simple(s.Update)
		}
		p.w(") ")
		p.block(s.Body)
	case *While:
		p.attrs(s.Attrs, " ")
		p.w("while ")
		p.expr(s.Cond)
		p.w(" ")
		p.block(s.Body)
	case *Return:
		p.w("return")
		if s.X != nil {
			p.w(" ")
			p.expr(s.X)
		}
		p.w(";")
	case *Break:
		p.w("break;")
	case *Continue:
		p.w("continue;")
	case *Discard:
		p.w("discard;")
	case *Empty:
		p.w(";")
	case *BreakIf:
		p.w("break if ")
		p.expr(s.Cond)
		p.w(";")
	case *AssertStmt:
		p.decl(s.Assert)
	default:
		p.simple(s)
		p.w(";")
	}
}

func (p *printer) ifChain(s *If) {
	p.w("if ")
	p.expr(s.Cond)
	p.w(" ")
	p.block(s.Then)
	switch e := s.Else.(type) {
	case *If:
		p.w(" else ")
		p.ifChain(e)
	case *Block:
		p.w(" else ")
		p.block(e)
	}
}

// simple prints a statement without its terminating semicolon.
func (p *printer) simple(s Stmt) {
	switch s := s.(type) {
	case *Assign:
		if s.LHS == nil {
			p.w("_")
		} else {
			p.expr(s.LHS)
		}
		p.w(" " + s.Op + " ")
		p.expr(s.RHS)
	case *IncDec:
		p.expr(s.X)
		p.w(s.Op)
	case *CallStmt:
		p.expr(s.Call)
	case *DeclStmt:
		switch d := s.Decl.(type) {
		case *Var:
			p.varDecl(d)
		case *Value:
			p.value(d)
		}
	}
}

// ---- expressions ----

type class int

const (
	cPrimary class = iota
	cUnary
	cMul
	cAdd
	cShift
	cCmp
	cBit
	cLogic
)

func opClass(op string) class {
	switch op {
	case "*", "/", "%":
		return cMul
	case "+", "-":
		return cAdd
	case "<<", ">>":
		return cShift
	case "<", ">", "<=", ">=", "==", "!=":
		return cCmp
	case "&", "|", "^":
		return cBit
	}
	return cLogic
}

func exprClass(e Expr) class {
	switch e := e.(type) {
	case *Binary:
		return opClass(e.Op)
	case *Unary:
		return cUnary
	}
	return cPrimary
}

// needParen reports whether child must be parenthesised as an operand of
// the binary expression parent, following the grammar of the spec: only
// multiplicative and additive chains associate freely; shifts, bitwise
// operators and short-circuit operators take restricted operand forms.
func needParen(parent *Binary, child Expr, right bool) bool {
	c, ok := child.(*Binary)
	if !ok {
		return false
	}
	pc, cc := opClass(parent.Op), opClass(c.Op)
	switch pc {
	case cMul:
		return !(cc == cMul && !right)
	case cAdd:
		if right {
			return cc != cMul
		}
		return cc != cMul && cc != cAdd
	case cShift:
		return true
	case cCmp:
		return cc > cShift
	case cBit:
		return right || c.Op != parent.Op
	case cLogic:
		if cc <= cCmp {
			return false
		}
		return right || c.Op != parent.Op || cc != cLogic
	}
	return true
}

func (p *printer) operand(parent *Binary, child Expr, right bool) {
	if needParen(parent, child, right) {
		p.w("(")
		p.expr(child)
		p.w(")")
		return
	}
	p.expr(child)
}

func (p *printer) postfixBase(x Expr) {
	if exprClass(x) != cPrimary {
		p.w("(")
		p.expr(x)
		p.w(")")
		return
	}
	p.expr(x)
}

func (p *printer) args(xs []Expr) {
	for i, x := range xs {
		if i > 0 {
			p.w(", ")
		}
		p.expr(x)
	}
}

func (p *printer) expr(e Expr) {
	switch e := e.(type) {
	case *IntLit:
		p.w(e.Text)
	case *FloatLit:
		p.w(e.Text)
	case *BoolLit:
		p.w(strconv.FormatBool(e.Value))
	case *Ident:
		p.w(e.Name)
		if len(e.Targs) > 0 {
			p.w("<")
			p.args(e.Targs)
			p.w(">")
		}
	case *Call:
		p.expr(e.Fn)
		p.w("(")
		p.args(e.Args)
		p.w(")")
	case *Paren:
		p.w("(")
		p.expr(e.X)
		p.w(")")
	case *Member:
		p.postfixBase(e.X)
		p.w("." + e.Name)
	case *Index:
		p.postfixBase(e.X)
		p.w("[")
		p.expr(e.I)
		p.w("]")
	case *Unary:
		p.w(e.Op)
		var inner printer
		if exprClass(e.X) > cUnary {
			inner.w("(")
			inner.expr(e.X)
			inner.w(")")
		} else {
			inner.expr(e.X)
		}
		s := inner.b.String()
		// avoid forming "--" or "&&" from two adjacent prefix operators
		if (e.Op == "-" || e.Op == "&") && strings.HasPrefix(s, e.Op) {
			p.w(" ")
		}
		p.w(s)
	case *Binary:
		p.operand(e, e.X, false)
		p.w(" " + e.Op + " ")
		p.operand(e, e.Y, true)
	}
}
