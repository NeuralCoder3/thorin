#include "thorin/pass/rw/matrix2tuple.h"

namespace thorin {

const Def* Matrix2Tuple::rewrite(const Def* def) {
    if (auto i = old2new_.find(def); i != old2new_.end()) return i->second;

    if (auto mat_type = isa<Tag::Mat>(def)) {
        return world().type_mat_tuple(mat_type);
    }else if(auto mat = def->isa<Mat>()){
        return world().tuple(mat->ops());
    }else if(auto slot = isa<Tag::Slot>(def)){
        auto mem = slot->arg(0);
        auto type = slot->arg(1);

        mem->dump();
        type->dump();
        type->dump();

        //return world().op_slot();
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

        for (size_t var_i = 0, n = app->num_args(); var_i != n; ++var_i) {
            new_vars.emplace_back(new_lam->var(var_i));
        }

        auto new_var_tuple = world().tuple(new_vars);
        new_lam->set(old_lam->reduce(new_var_tuple));
        return old2new_[def] = world().app(new_lam, app->arg());
    }


    return def;
}

}
