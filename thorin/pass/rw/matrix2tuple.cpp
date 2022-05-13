#include "thorin/pass/rw/matrix2tuple.h"

namespace thorin {

const Def* Matrix2Tuple::rewrite(const Def* def) {
    if (auto i = old2new_.find(def); i != old2new_.end()) return i->second;

    if (auto mat_type = isa<Tag::Mat>(def)) {
        auto elem_type = mat_type->arg(0);
        return world().sigma({
             world().type_int_width(64),
             world().type_int_width(64),
             world().type_ptr(world().arr(world().top_nat(), elem_type)),
        });
    }else if(auto mat = def->isa<Mat>()){
        return world().tuple({mat->op(0), mat->op(1), mat->op(2)});
    }else{
        auto [app, old_lam] = isa_apped_nom_lam(def);
        if (!isa_workable(old_lam)) return def;

        DefVec new_doms, new_vars, new_args;
        auto old_doms = old_lam->doms();

        bool hasMat = false;
        for (auto dom : old_doms) {
            if(isa<Tag::Mat>(dom)){
                hasMat = true;
                new_doms.emplace_back(rewrite(dom));
            }else{
                new_doms.emplace_back(dom);
            }
        }

        if(!hasMat){
            return def;
        }

        auto new_pi  = world().cn(world().sigma(new_doms));
        auto new_lam = old_lam->stub(world(), new_pi, old_lam->dbg());

        for (size_t arg_i = 0, var_i = 0, n = app->num_args(); arg_i != n; ++arg_i) {
            auto arg = app->arg(arg_i);
            if (old_lam->dom(arg_i)->isa<Pi>()) {
                new_vars.emplace_back(arg);
            } else {
                new_vars.emplace_back(new_lam->var(var_i++));
            }
        }

        new_lam->set(old_lam->reduce(world().tuple(new_vars)));
        return old2new_[def] = world().app(new_lam, app->arg());
    }


    return def;
}

}
