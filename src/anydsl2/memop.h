#ifndef ANYDSL2_MEMOP_H
#define ANYDSL2_MEMOP_H

#include "anydsl2/primop.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

class MemOp : public PrimOp {
protected:

    MemOp(size_t size, int kind, const Type* type, const Def* mem, const std::string& name);

public:

    const Def* mem() const { return op(0); }
};

//------------------------------------------------------------------------------

class Access : public MemOp {
protected:

    Access(size_t size, int kind, const Type* type, const Def* mem, const Def* ptr, const std::string& name)
        : MemOp(size, kind, type, mem, name)
    {
        assert(size >= 2);
        set_op(1, ptr);
    }

public:

    const Def* ptr() const { return op(1); }
};

//------------------------------------------------------------------------------

class Load : public Access {
private:

    Load(const DefTuple2& args, const std::string& name)
        : Access(2, args.get<0>(), args.get<1>(), args.get<2>(), args.get<3>(), name)
        , extract_mem_(0)
        , extract_val_(0)
    {}

    virtual void vdump(Printer &printer) const;

public:

    const Def* ptr() const { return op(1); }
    const Def* extract_mem() const;
    const Def* extract_val() const;

private:

    mutable const Def* extract_mem_;
    mutable const Def* extract_val_;

    friend class World;
};

//------------------------------------------------------------------------------

class Store : public Access {
private:

    Store(const DefTuple3& args, const std::string& name)
        : Access(2, args.get<0>(), args.get<1>(), args.get<2>(), args.get<3>(), name)
    {
        set_op(2, args.get<4>());
    }

    virtual void vdump(Printer &printer) const;

public:

    const Def* val() const { return op(2); }

    friend class World;
};

//------------------------------------------------------------------------------

class Enter : public MemOp {
private:

    Enter(const DefTuple1& args, const std::string& name)
        : MemOp(1, args.get<0>(), args.get<1>(), args.get<2>(), name)
        , extract_mem_(0)
        , extract_frame_(0)
    {}

    virtual void vdump(Printer &printer) const;

public:

    const Def* extract_mem() const;
    const Def* extract_frame() const;

private:

    mutable const Def* extract_mem_;
    mutable const Def* extract_frame_;

    friend class World;
};

//------------------------------------------------------------------------------

class Leave : public MemOp {
private:

    Leave(const DefTuple2& args, const std::string& name)
        : MemOp(2, args.get<0>(), args.get<1>(), args.get<2>(), name)
    {
        set_op(1, args.get<3>());
    }

    virtual void vdump(Printer &printer) const;

public:

    const Def* frame() const { return op(1); }

    friend class World;
};

//------------------------------------------------------------------------------

/**
 * This represents a slot in a stack frame opend via \p Enter.
 * Loads from this address yield \p Bottom if the frame has already been closed via \p Leave.
 * This \p PrimOp is technically \em not a \p MemOp.
 */
class Slot : public PrimOp {
private:

    Slot(const DefTuple2& args, const std::string& name) 
        : PrimOp(2, args.get<0>(), args.get<1>(), name)
    {
        set_op(0, args.get<2>());
        set_op(1, args.get<3>());
    }

    virtual void vdump(Printer &printer) const;

public:

    const Def* frame() const { return op(0); }

    friend class World;
};

//------------------------------------------------------------------------------

typedef boost::tuple<int, const Type*, const std::string&, const Def*, ArrayRef<const Def*>, bool> CCallTuple;

class CCall : public MemOp {
private:

    CCall(const CCallTuple& args, const std::string& name)
        : MemOp(args.get<4>().size() + 1, args.get<0>(), args.get<1>(), args.get<3>(), name)
        , extract_mem_(0)
        , extract_retval_(0)
        , callee_(args.get<2>())
        , vararg_(args.get<5>())
    {
        size_t x = 1;
        for_all (arg, args.get<4>())
            set_op(x++, arg);
    }

    virtual void vdump(Printer &printer) const;

public:

    bool returns_void() const;
    const Def* extract_mem() const;
    const Def* extract_retval() const;
    const std::string& callee() const { return callee_; }
    bool vararg() const { return vararg_; }
    const Type* rettype() const;
    ArrayRef<const Def*> args() const { return ops().slice_back(1); }
    size_t num_args() const { return args().size(); }
    CCallTuple as_tuple() const { 
        return CCallTuple(kind(), type(), callee(), ops().front(), ops().slice_back(1), vararg()); 
    }

private:

    mutable const Def* extract_mem_;
    mutable const Def* extract_retval_;
    ANYDSL2_HASH_EQUAL

    std::string callee_;
    bool vararg_;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif // ANYDSL2_MEMOP_H
