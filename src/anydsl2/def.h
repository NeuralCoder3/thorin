#ifndef ANYDSL2_DEF_H
#define ANYDSL2_DEF_H

#include <cstring>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

#include "anydsl2/enums.h"
#include "anydsl2/node.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/cast.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

class Def;
class Lambda;
class Printer;
class PrimOp;
class Sigma;
class Type;
class World;

//------------------------------------------------------------------------------

class Peek {
public:

    Peek() {}
    Peek(const Def* def, Lambda* from)
        : def_(def)
        , from_(from)
    {}

    const Def* def() const { return def_; }
    Lambda* from() const { return from_; }

private:

    const Def* def_;
    Lambda* from_;
};

typedef Array<Peek> Peeks;

//------------------------------------------------------------------------------

class Use {
public:

    Use() {}
    Use(size_t index, const Def* def)
        : index_(index)
        , def_(def)
    {}

    size_t index() const { return index_; }
    const Def* def() const { return def_; }

    bool operator == (Use use) const { return def() == use.def() && index() == use.index(); }
    bool operator != (Use use) const { return def() != use.def() || index() != use.index(); }

private:

    size_t index_;
    const Def* def_;
};

typedef std::vector<Use> Uses;

//------------------------------------------------------------------------------

class Def : public Node {
private:

    /// Do not copy-assign a \p Def instance.
    Def& operator = (const Def&);

protected:

    /// This variant leaves internal \p ops_ \p Array allocateble via ops_.alloc(size).
    Def(int kind, const Type* type, const std::string& name)
        : Node(kind, name)
        , type_(type)
    {}
    Def(int kind, size_t size, const Type* type, const std::string& name)
        : Node(kind, size, name)
        , type_(type)
    {}

    void set_type(const Type* type) { type_ = type; }
    void unregister_use(size_t i) const;

public:

    virtual ~Def();
    void set_op(size_t i, const Def* def);
    void unset_op(size_t i);

    Lambda* as_lambda() const;
    Lambda* isa_lambda() const;

    bool is_const() const;

    void dump() const;
    void dump(bool fancy) const;

    virtual void vdump(Printer &printer) const = 0;

    const Uses& uses() const { return uses_; }
    size_t num_uses() const { return uses_.size(); }

    /**
     * Copies all use-info into an array.
     * Useful if you want to modfy users while iterating over all users.
     */
    Array<Use> copy_uses() const;
    const Type* type() const { return type_; }
    int order() const;

    /// Updates operand \p i to point to \p def instead.
    void update(size_t i, const Def* def);
    void update(ArrayRef<const Def*> defs);

    World& world() const;
    ArrayRef<const Def*> ops() const { return ops_ref<const Def*>(); }
    ArrayRef<const Def*> ops(size_t begin, size_t end) const { return ops().slice(begin, end); }
    const Def* op(size_t i) const { return ops()[i]; }
    const Def* op_via_lit(const Def* def) const;
    void replace(const Def* with) const;

    /*
     * check for special literals
     */

    bool is_primlit(int val) const;
    bool is_zero() const { return is_primlit(0); }
    bool is_one() const { return is_primlit(1); }
    bool is_allset() const { return is_primlit(-1); }

    // implementation in literal.h
    template<class T> inline T primlit_value() const;

private:

    const Type* type_;
    mutable Uses uses_;
};

template<class T>
inline bool equal_kind_type_size(const T& tuple, size_t size, const Def* node) {
    const Def* def = node->as<Def>();
    return size == def->size() && tuple.template get<0>() == def->kind() && tuple.template get<1>() == def->type();
}

std::ostream& operator << (std::ostream& o, const Def* def);

//------------------------------------------------------------------------------

class Param : public Def {
private:

    Param(const Type* type, Lambda* parent, size_t index, const std::string& name);

public:

    Lambda* lambda() const { return lambda_; }
    size_t index() const { return index_; }
    Peeks peek() const;
    virtual void vdump(Printer& printer) const;

private:

    mutable Lambda* lambda_;
    const size_t index_;

    friend class World;
    friend class Lambda;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
