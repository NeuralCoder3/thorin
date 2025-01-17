#include "thorin/check.h"

#include "thorin/world.h"

namespace thorin {

bool Checker::equiv(const Def* d1, const Def* d2) {
    if (d1 == d2 || (d1->is_unset() && d2->is_unset())) return true;

    // normalize: always put smaller gid to the left
    if (d1->gid() > d2->gid()) std::swap(d1, d2);

    // this assumption will either hold true - or we will bail out with false anyway
    auto [_, inserted] = equiv_.emplace(d1, d2);
    if (!inserted) return true;

    if (!equiv(d1->type(), d2->type())) return false;

    if (d1->isa<Top>() || d2->isa<Top>()) return equiv(d1->type(), d2->type());

    if (is_sigma_or_arr(d1)) {
        if (!equiv(d1->arity(), d2->arity())) return false;

        if (auto a = isa_lit(d1->arity())) {
            for (size_t i = 0; i != a; ++i) {
                if (!equiv(d1->proj(*a, i), d2->proj(*a, i))) return false;
            }

            return true;
        }
    } else if (auto p1 = d1->isa<Var>()) {
        // vars are equal if they appeared under the same binder
        for (auto [q1, q2] : vars_) {
            if (p1 == q1) return d2->as<Var>() == q2;
        }
        return true;
    }

    if (auto n1 = d1->isa_nom()) {
        if (auto n2 = d2->isa_nom()) vars_.emplace_back(n1->var(), n2->var());
    }

    if (d1->node() != d2->node() || d1->fields() != d2->fields() || d1->num_ops() != d2->num_ops() ||
        d1->is_set() != d2->is_set())
        return false;

    return std::ranges::equal(d1->ops(), d2->ops(), [this](auto op1, auto op2) { return equiv(op1, op2); });
}

bool Checker::assignable(const Def* type, const Def* val) {
    if (type == val->type()) return true;

    if (auto sigma = type->isa<Sigma>()) {
        if (!equiv(type->arity(), val->type()->arity())) return false;

        auto red = sigma->reduce(val);
        for (size_t i = 0, a = red.size(); i != a; ++i) {
            if (!assignable(red[i], val->proj(a, i))) return false;
        }

        return true;
    } else if (auto arr = type->isa<Arr>()) {
        if (!equiv(type->arity(), val->type()->arity())) return false;

        if (auto a = isa_lit(arr->arity())) {
            ;
            for (size_t i = 0; i != *a; ++i) {
                if (!assignable(arr->proj(*a, i), val->proj(*a, i))) return false;
            }

            return true;
        }
    }

    return equiv(type, val->type());
}

} // namespace thorin
