#include "thorin/transform/mangle.h"

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

class Mangler {
public:
    Mangler(const Scope& scope, Defs args, Defs lift)
        : scope(scope)
        , args(args)
        , lift(lift)
        , oentry(scope.entry())
    {
        assert(!oentry->empty());
        assert(args.size() == oentry->num_params());

        // TODO correctly deal with lambdas here
        std::queue<const Def*> queue;
        auto enqueue = [&](const Def* def) {
            if (!within(def)) {
                defs_.insert(def);
                queue.push(def);
            }
        };

        for (auto def : lift)
            enqueue(def);

        while (!queue.empty()) {
            for (auto use : pop(queue)->uses())
                enqueue(use);
        }
    }

    World& world() const { return scope.world(); }
    Lambda* mangle();
    void mangle_body(Lambda* olambda, Lambda* nlambda);
    Lambda* mangle_head(Lambda* olambda);
    const Def* mangle(const Def* odef);
    bool within(const Def* def) { return scope.contains(def) || defs_.contains(def); }

    const Scope& scope;
    Def2Def def2def;
    Defs args;
    Defs lift;
    Type2Type type2type;
    Lambda* oentry;
    Lambda* nentry;
    DefSet defs_;
};

Lambda* Mangler::mangle() {
    // create nentry - but first collect and specialize all param types
    std::vector<const Type*> param_types;
    for (size_t i = 0, e = oentry->num_params(); i != e; ++i) {
        if (args[i] == nullptr)
            param_types.push_back(oentry->param(i)->type()->specialize(type2type));
    }

    auto fn_type = world().fn_type(param_types);
    nentry = world().lambda(fn_type, oentry->loc(), oentry->name);

    // map value params
    def2def[oentry] = oentry;
    for (size_t i = 0, j = 0, e = oentry->num_params(); i != e; ++i) {
        auto oparam = oentry->param(i);
        if (auto def = args[i])
            def2def[oparam] = def;
        else {
            auto nparam = nentry->param(j++);
            def2def[oparam] = nparam;
            nparam->name = oparam->name;
        }
    }

    for (auto def : lift)
        def2def[def] = nentry->append_param(def->type()->specialize(type2type));

    mangle_body(oentry, nentry);
    return nentry;
}

Lambda* Mangler::mangle_head(Lambda* olambda) {
    assert(!def2def.contains(olambda));
    assert(!olambda->empty());
    Lambda* nlambda = olambda->stub(type2type, olambda->name);
    def2def[olambda] = nlambda;

    for (size_t i = 0, e = olambda->num_params(); i != e; ++i)
        def2def[olambda->param(i)] = nlambda->param(i);

    return nlambda;
}

void Mangler::mangle_body(Lambda* olambda, Lambda* nlambda) {
    assert(!olambda->empty());

    if (olambda->to() == world().branch()) {        // fold branch if possible
        if (auto lit = mangle(olambda->arg(0))->isa<PrimLit>())
            return nlambda->jump(mangle(lit->value().get_bool() ? olambda->arg(1) : olambda->arg(2)), {}, olambda->jump_loc());
    }

    Array<const Def*> nops(olambda->size());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = mangle(olambda->op(i));

    Defs nargs(nops.skip_front()); // new args of nlambda
    auto ntarget = nops.front();   // new target of nlambda

    // check whether we can optimize tail recursion
    if (ntarget == oentry) {
        std::vector<size_t> cut;
        bool substitute = true;
        for (size_t i = 0, e = args.size(); i != e && substitute; ++i) {
            if (auto def = args[i]) {
                substitute &= def == nargs[i];
                cut.push_back(i);
            }
        }

        if (substitute)
            return nlambda->jump(nentry, nargs.cut(cut), olambda->jump_loc());
    }

    nlambda->jump(ntarget, nargs, olambda->jump_loc());
}

const Def* Mangler::mangle(const Def* odef) {
    if (auto ndef = find(def2def, odef))
        return ndef;
    else if (!within(odef))
        return odef;
    else if (auto olambda = odef->isa_lambda()) {
        auto nlambda = mangle_head(olambda);
        mangle_body(olambda, nlambda);
        return nlambda;
    } else if (auto param = odef->isa<Param>()) {
        assert(within(param->lambda()));
        mangle(param->lambda());
        assert(def2def.contains(param));
        return def2def[param];
    } else {
        auto oprimop = odef->as<PrimOp>();
        Array<const Def*> nops(oprimop->size());
        for (size_t i = 0, e = oprimop->size(); i != e; ++i)
            nops[i] = mangle(oprimop->op(i));

        auto type = oprimop->type()->specialize(type2type);
        return def2def[oprimop] = oprimop->rebuild(nops, type);
    }
}

//------------------------------------------------------------------------------

Lambda* mangle(const Scope& scope, Defs args, Defs lift) {
    return Mangler(scope, args, lift).mangle();
}

Lambda* drop(const Call& call) {
    Scope scope(call.to()->as_lambda());
    return drop(scope, call.args());
}

//------------------------------------------------------------------------------

}
