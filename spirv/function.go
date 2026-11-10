package spirv

import (
	"fmt"

	"github.com/go-gfx/wgsl"
)

// fnState is the function being emitted: its blocks, the variables that go in
// its first block, and where break and continue lead.
type fnState struct {
	f          *wgsl.FuncObj
	body       []uint32 // every block, from the entry label on
	entryEnd   int      // where in body the entry label ends: variables go there
	vars       []uint32 // OpVariable instructions for the first block (section 2.4)
	terminated bool     // the current block already ended
	breaks     []uint32 // merge block of each enclosing loop
	continues  []uint32
	uses       map[uint32]bool // module-scope variables used
	calls      []*wgsl.FuncObj
	ret        uint32 // return type id (void for none)
}

func (e *emitter) function(f *wgsl.FuncObj) error {
	st := &fnState{f: f, uses: map[uint32]bool{}}
	e.fn = st
	defer func() { e.fn = nil }()
	var err error
	st.ret = e.voidID()
	if _, void := f.Ret.(*wgsl.VoidType); !void {
		if f.Entry != nil {
			return &Error{f.Decl.Start, "entry point " + f.Name + " returns a value"}
		}
		if st.ret, err = e.typeID(f.Ret, f.Decl.Start); err != nil {
			return err
		}
	}
	// Parameters: an entry point has none in SPIR-V; its @builtin inputs are
	// Input-class variables, loaded first. A plain function takes values.
	var ptys []uint32
	var pids []uint32
	if f.Entry == nil {
		for _, p := range f.Params {
			t, err := e.typeID(p.Type, p.Decl.Start)
			if err != nil {
				return err
			}
			ptys = append(ptys, t)
			pid := e.id()
			pids = append(pids, pid)
			e.locals[p] = &local{id: pid, typeT: p.Type}
		}
	}
	fid := e.fnID[f]
	fty := e.fnType(st.ret, ptys)
	inst(&e.funcs, OpFunction, st.ret, fid, 0, fty) // FunctionControl None
	for i, p := range ptys {
		inst(&e.funcs, OpFunctionParameter, p, pids[i])
	}
	inst(&st.body, OpLabel, e.id())
	st.entryEnd = len(st.body)
	if f.Entry != nil {
		for _, in := range f.Entry.Inputs {
			b, ok := builtinInputs[in.Builtin]
			if in.Member != "" || !ok {
				return &Error{f.Decl.Start, "input " + in.Name + " of " + f.Name + " is not a supported @builtin (global_invocation_id, local_invocation_id, workgroup_id)"}
			}
			t, err := e.typeID(in.Type, f.Decl.Start)
			if err != nil {
				return err
			}
			vid := e.id()
			inst(&e.decls, OpVariable, e.ptrID(StorageClassInput, t), vid, StorageClassInput)
			inst(&e.annot, OpDecorate, vid, DecorationBuiltIn, b)
			st.uses[vid] = true
			val := e.id()
			inst(&st.body, OpLoad, t, val, vid)
			p := f.Params[in.Param]
			e.locals[p] = &local{id: val, typeT: p.Type}
		}
	}
	if err := e.block(f.Decl.Body); err != nil {
		return err
	}
	if !st.terminated {
		if st.ret == e.types["void"] {
			inst(&st.body, OpReturn)
		} else {
			inst(&st.body, OpUnreachable)
		}
	}
	// Variables go first in the first block, right after its label.
	out := append([]uint32(nil), st.body[:st.entryEnd]...)
	out = append(out, st.vars...)
	out = append(out, st.body[st.entryEnd:]...)
	e.funcs = append(e.funcs, out...)
	inst(&e.funcs, OpFunctionEnd)
	e.record(f, st)
	return nil
}

// record keeps what an entry point's interface needs to be computed from.
func (e *emitter) record(f *wgsl.FuncObj, st *fnState) {
	if e.uses == nil {
		e.uses = map[*wgsl.FuncObj]map[uint32]bool{}
		e.calls = map[*wgsl.FuncObj][]*wgsl.FuncObj{}
	}
	e.uses[f] = st.uses
	e.calls[f] = st.calls
}

// ---- blocks and statements ----

// label starts a new block with a known id; the previous one must have ended.
func (e *emitter) label(id uint32) {
	inst(&e.fn.body, OpLabel, id)
	e.fn.terminated = false
}

func (e *emitter) branch(target uint32) {
	if !e.fn.terminated {
		inst(&e.fn.body, OpBranch, target)
		e.fn.terminated = true
	}
}

func (e *emitter) block(b *wgsl.Block) error {
	for _, s := range b.Stmts {
		if e.fn.terminated {
			return nil // what follows a return, break or continue is dead
		}
		if err := e.stmt(s); err != nil {
			return err
		}
	}
	return nil
}

func (e *emitter) stmt(s wgsl.Stmt) error {
	st := e.fn
	switch s := s.(type) {
	case *wgsl.Block:
		return e.block(s)
	case *wgsl.Empty:
	case *wgsl.Return:
		if s.X == nil {
			inst(&st.body, OpReturn)
		} else {
			v, err := e.rvalue(s.X)
			if err != nil {
				return err
			}
			inst(&st.body, OpReturnValue, v)
		}
		st.terminated = true
	case *wgsl.Break:
		inst(&st.body, OpBranch, st.breaks[len(st.breaks)-1])
		st.terminated = true
	case *wgsl.Continue:
		inst(&st.body, OpBranch, st.continues[len(st.continues)-1])
		st.terminated = true
	case *wgsl.DeclStmt:
		return e.decl(s)
	case *wgsl.Assign:
		return e.assign(s)
	case *wgsl.IncDec:
		return e.incdec(s)
	case *wgsl.CallStmt:
		_, err := e.call(s.Call)
		return err
	case *wgsl.If:
		return e.ifStmt(s)
	case *wgsl.For:
		return e.loop(s.Start, s.Init, s.Cond, s.Update, s.Body.Stmts)
	case *wgsl.While:
		return e.loop(s.Start, nil, s.Cond, nil, s.Body.Stmts)
	case *wgsl.Loop:
		if s.Cont != nil {
			return &Error{s.Start, "a loop with a continuing block is not supported yet"}
		}
		return e.loop(s.Start, nil, nil, nil, s.Body)
	default:
		return &Error{s.NodeSpan().Start, fmt.Sprintf("statement %T is not supported yet", s)}
	}
	return nil
}

func (e *emitter) ifStmt(s *wgsl.If) error {
	st := e.fn
	c, err := e.rvalue(s.Cond)
	if err != nil {
		return err
	}
	then, merge := e.id(), e.id()
	els := merge
	if s.Else != nil {
		els = e.id()
	}
	inst(&st.body, OpSelectionMerge, merge, SelectionControlNone)
	inst(&st.body, OpBranchConditional, c, then, els)
	st.terminated = true
	e.label(then)
	if err := e.block(s.Then); err != nil {
		return err
	}
	e.branch(merge)
	switch el := s.Else.(type) {
	case nil:
	case *wgsl.If:
		e.label(els)
		if err := e.ifStmt(el); err != nil {
			return err
		}
		e.branch(merge)
	case *wgsl.Block:
		e.label(els)
		if err := e.block(el); err != nil {
			return err
		}
		e.branch(merge)
	}
	e.label(merge)
	return nil
}

// loop emits for, while and loop (section 2.11): a header with OpLoopMerge, a
// condition block, the body, a continue block that holds the update and
// branches back, and the merge block.
func (e *emitter) loop(off int, init wgsl.Stmt, cond wgsl.Expr, update wgsl.Stmt, body []wgsl.Stmt) error {
	st := e.fn
	if init != nil {
		if err := e.stmt(init); err != nil {
			return err
		}
	}
	header, condB, bodyB, cont, merge := e.id(), e.id(), e.id(), e.id(), e.id()
	e.branch(header)
	e.label(header)
	inst(&st.body, OpLoopMerge, merge, cont, LoopControlNone)
	inst(&st.body, OpBranch, condB)
	st.terminated = true
	e.label(condB)
	if cond != nil {
		c, err := e.rvalue(cond)
		if err != nil {
			return err
		}
		inst(&st.body, OpBranchConditional, c, bodyB, merge)
		st.terminated = true
	} else {
		inst(&st.body, OpBranch, bodyB)
		st.terminated = true
	}
	e.label(bodyB)
	st.breaks = append(st.breaks, merge)
	st.continues = append(st.continues, cont)
	err := e.block(&wgsl.Block{Stmts: body})
	st.breaks, st.continues = st.breaks[:len(st.breaks)-1], st.continues[:len(st.continues)-1]
	if err != nil {
		return err
	}
	e.branch(cont)
	e.label(cont)
	if update != nil {
		if err := e.stmt(update); err != nil {
			return err
		}
	}
	e.branch(header)
	e.label(merge)
	return nil
}

func (e *emitter) decl(s *wgsl.DeclStmt) error {
	st := e.fn
	switch d := s.Decl.(type) {
	case *wgsl.Value:
		if d.Kind == "override" {
			return &Error{d.Start, "an override declaration is not supported yet"}
		}
		obj := e.info.Decls[d].(*wgsl.ValueObj)
		v, err := e.rvalue(d.Init)
		if err != nil {
			return err
		}
		e.locals[obj] = &local{id: v, typeT: e.valueType(d.Init)}
	case *wgsl.Var:
		obj := e.info.Decls[d].(*wgsl.VarObj)
		t := e.info.DeclTypes[d]
		ty, err := e.typeID(t, d.Start)
		if err != nil {
			return err
		}
		vid := e.id()
		inst(&st.vars, OpVariable, e.ptrID(StorageClassFunction, ty), vid, StorageClassFunction)
		e.locals[obj] = &local{id: vid, ptr: true, typeT: t}
		var init uint32
		if d.Init != nil {
			if init, err = e.rvalue(d.Init); err != nil {
				return err
			}
		} else if init, err = e.zero(t, d.Start); err != nil { // WGSL zero-initialises a var
			return err
		}
		inst(&st.body, OpStore, vid, init)
	default:
		return &Error{s.Start, fmt.Sprintf("declaration %T is not supported yet", d)}
	}
	return nil
}

func (e *emitter) zero(t wgsl.Type, off int) (uint32, error) {
	return e.constant(&wgsl.Const{Type: t, Zero: true}, t, off)
}

func (e *emitter) assign(s *wgsl.Assign) error {
	if s.LHS == nil {
		return &Error{s.Start, "a phony assignment is not supported yet"}
	}
	p, err := e.ref(s.LHS)
	if err != nil {
		return err
	}
	var v uint32
	if s.Op == "=" {
		if v, err = e.rvalue(s.RHS); err != nil {
			return err
		}
	} else {
		cur, err := e.load(p, s.LHS)
		if err != nil {
			return err
		}
		b, err := e.rvalue(s.RHS)
		if err != nil {
			return err
		}
		if v, err = e.binop(s.Op[:len(s.Op)-1], e.valueType(s.LHS), e.valueType(s.RHS), cur, b, e.valueType(s.LHS), s.RHS, s.Start); err != nil {
			return err
		}
	}
	inst(&e.fn.body, OpStore, p.id, v)
	return nil
}

func (e *emitter) incdec(s *wgsl.IncDec) error {
	p, err := e.ref(s.X)
	if err != nil {
		return err
	}
	cur, err := e.load(p, s.X)
	if err != nil {
		return err
	}
	t := e.valueType(s.X)
	k, scalar := elemKind(t)
	if !scalar || (k != wgsl.KI32 && k != wgsl.KU32) {
		return &Error{s.Start, s.Op + " of " + describe(t) + " is not supported"}
	}
	ty, err := e.typeID(t, s.Start)
	if err != nil {
		return err
	}
	one := e.scalarConst(ty, 1)
	op := uint32(OpIAdd)
	if s.Op == "--" {
		op = OpISub
	}
	r := e.id()
	inst(&e.fn.body, op, ty, r, cur, one)
	inst(&e.fn.body, OpStore, p.id, r)
	return nil
}
