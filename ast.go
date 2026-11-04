package wgsl

// Span is the half-open byte range [Start, End) of a node in the source.
// Nodes built by transforms carry the zero Span. Span is ignored by Equal.
type Span struct{ Start, End int }

// Node is implemented by every AST node.
type Node interface{ NodeSpan() Span }

func (s Span) NodeSpan() Span { return s }

// Module is a translation_unit: directives followed by global declarations.
type Module struct {
	Span
	Directives []*Directive
	Decls      []Decl
}

// Directive is an `enable`, `requires` or `diagnostic` directive.
// For "diagnostic", Names holds the severity then the rule name (a dotted
// rule such as "chromium.unreachable_code" is kept as one string).
type Directive struct {
	Span
	Kind  string // "enable", "requires", "diagnostic"
	Names []string
}

// Decl is a global declaration.
type Decl interface {
	Node
	declNode()
}

// Attr is an attribute such as @compute or @workgroup_size(8, 8). Arguments
// that are names (builtin, interpolate, diagnostic) are *Ident expressions.
type Attr struct {
	Span
	Name string
	Args []Expr
}

// Func is a function declaration.
type Func struct {
	Span
	Attrs    []*Attr
	Name     string
	Params   []*Param
	RetAttrs []*Attr
	Ret      *Ident // nil when the function returns nothing
	Body     *Block
}

// Param is a function parameter.
type Param struct {
	Span
	Attrs []*Attr
	Name  string
	Type  *Ident
}

// Var is a `var` declaration, global (Decl) or local (Stmt).
// Template holds the optional <address_space, access> arguments.
type Var struct {
	Span
	Attrs    []*Attr
	Template []Expr
	Name     string
	Type     *Ident // nil when inferred
	Init     Expr   // nil when absent
}

// Value is a `const`, `override` (global or local) or `let` (local) declaration.
type Value struct {
	Span
	Kind  string // "const", "override", "let"
	Attrs []*Attr
	Name  string
	Type  *Ident
	Init  Expr
}

// Alias is `alias name = type;`.
type Alias struct {
	Span
	Name string
	Type *Ident
}

// Struct is a structure declaration.
type Struct struct {
	Span
	Name    string
	Members []*StructMember
}

// StructMember is a structure member.
type StructMember struct {
	Span
	Attrs []*Attr
	Name  string
	Type  *Ident
}

// ConstAssert is `const_assert expr;`, global or local.
type ConstAssert struct {
	Span
	X Expr
}

func (*Func) declNode()        {}
func (*Var) declNode()         {}
func (*Value) declNode()       {}
func (*Alias) declNode()       {}
func (*Struct) declNode()      {}
func (*ConstAssert) declNode() {}

// Stmt is a statement.
type Stmt interface {
	Node
	stmtNode()
}

type (
	// Block is a compound statement.
	Block struct {
		Span
		Attrs []*Attr
		Stmts []Stmt
	}
	// If is an if statement; Else is nil, *If (else if) or *Block.
	If struct {
		Span
		Attrs []*Attr
		Cond  Expr
		Then  *Block
		Else  Stmt
	}
	// Switch is a switch statement.
	Switch struct {
		Span
		Attrs     []*Attr
		X         Expr
		BodyAttrs []*Attr
		Cases     []*Case
	}
	// Loop is a loop statement.
	Loop struct {
		Span
		Attrs     []*Attr
		BodyAttrs []*Attr
		Body      []Stmt
		Cont      *Continuing // nil when absent
	}
	// For is a for statement. Init, Cond and Update may be nil.
	For struct {
		Span
		Attrs  []*Attr
		Init   Stmt
		Cond   Expr
		Update Stmt
		Body   *Block
	}
	// While is a while statement.
	While struct {
		Span
		Attrs []*Attr
		Cond  Expr
		Body  *Block
	}
	// Return is a return statement; X may be nil.
	Return struct {
		Span
		X Expr
	}
	Break    struct{ Span }
	Continue struct{ Span }
	Discard  struct{ Span }
	// Empty is a lone `;`.
	Empty struct{ Span }
	// Assign is an assignment. Op is "=" or a compound operator such as "+=".
	// A phony assignment `_ = e` has LHS == nil.
	Assign struct {
		Span
		LHS Expr
		Op  string
		RHS Expr
	}
	// IncDec is `x++` or `x--`.
	IncDec struct {
		Span
		X  Expr
		Op string
	}
	// CallStmt is a function call used as a statement.
	CallStmt struct {
		Span
		Call *Call
	}
	// DeclStmt wraps a local declaration (*Var, *Value).
	DeclStmt struct {
		Span
		Decl Decl
	}
	// AssertStmt wraps a local const_assert.
	AssertStmt struct {
		Span
		Assert *ConstAssert
	}
	// BreakIf is `break if expr;`, only valid last in a continuing block.
	BreakIf struct {
		Span
		Cond Expr
	}
)

// Continuing is the continuing block of a loop. Its optional `break if`
// is stored as the final statement (*BreakIf) of Stmts.
type Continuing struct {
	Span
	Attrs []*Attr
	Stmts []Stmt
}

// Case is one switch clause. For `default:` Default is true and Selectors is
// empty. In `case 1, default:` a nil Expr in Selectors stands for `default`.
type Case struct {
	Span
	Default   bool
	Selectors []Expr
	Body      *Block
}

func (*Block) stmtNode()      {}
func (*If) stmtNode()         {}
func (*Switch) stmtNode()     {}
func (*Loop) stmtNode()       {}
func (*For) stmtNode()        {}
func (*While) stmtNode()      {}
func (*Return) stmtNode()     {}
func (*Break) stmtNode()      {}
func (*Continue) stmtNode()   {}
func (*Discard) stmtNode()    {}
func (*Empty) stmtNode()      {}
func (*Assign) stmtNode()     {}
func (*IncDec) stmtNode()     {}
func (*CallStmt) stmtNode()   {}
func (*DeclStmt) stmtNode()   {}
func (*AssertStmt) stmtNode() {}
func (*BreakIf) stmtNode()    {}

// Expr is an expression.
type Expr interface {
	Node
	exprNode()
}

type (
	// IntLit is an integer literal; Text is the exact source spelling
	// including any i/u suffix.
	IntLit struct {
		Span
		Text string
	}
	// FloatLit is a floating point literal, exact source spelling.
	FloatLit struct {
		Span
		Text string
	}
	// BoolLit is true or false.
	BoolLit struct {
		Span
		Value bool
	}
	// Ident is a (possibly template-elaborated) identifier. It also serves
	// as the type specifier node: `array<vec4<f32>, 4>`.
	Ident struct {
		Span
		Name  string
		Targs []Expr
	}
	// Call is a call or constructor expression: callee(args).
	Call struct {
		Span
		Fn   *Ident
		Args []Expr
	}
	// Unary is a prefix operator: - ! ~ * &.
	Unary struct {
		Span
		Op string
		X  Expr
	}
	// Binary is an infix operator.
	Binary struct {
		Span
		Op   string
		X, Y Expr
	}
	// Index is x[i].
	Index struct {
		Span
		X, I Expr
	}
	// Member is x.name; swizzles are Members too.
	Member struct {
		Span
		X    Expr
		Name string
	}
	// Paren is a parenthesised expression.
	Paren struct {
		Span
		X Expr
	}
)

func (*IntLit) exprNode()   {}
func (*FloatLit) exprNode() {}
func (*BoolLit) exprNode()  {}
func (*Ident) exprNode()    {}
func (*Call) exprNode()     {}
func (*Unary) exprNode()    {}
func (*Binary) exprNode()   {}
func (*Index) exprNode()    {}
func (*Member) exprNode()   {}
func (*Paren) exprNode()    {}
