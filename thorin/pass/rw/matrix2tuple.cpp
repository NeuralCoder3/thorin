#include "thorin/pass/rw/matrix2tuple.h"

namespace thorin {


void Matrix2Tuple::enter() {
    found = false;
    auto lam = curr_nom();
    auto result = rewrite_cached(curr_nom()->body());
    if(found){
        lam->set_body(result);
    }
}

const Def* Matrix2Tuple::rewrite_cached(const Def* def) {
    if(auto mat = def->isa<Mat>()){
        found = true;
    }

    if (auto i = old2new_.find(def); i != old2new_.end()) return i->second;
    return old2new_[def] = rewrite_convert(def);
}

const Def* Matrix2Tuple::rewrite_type_cached(const Def* def) {
    if (auto mat_type = isa<Tag::Mat>(def)) {
        found = true;
    }

    if (auto i = old2new_.find(def); i != old2new_.end()) return i->second;
    return old2new_[def] = rewrite_type(def);
}

const Def* Matrix2Tuple::rewrite_type(const Def* def) {
    if (auto mat_type = isa<Tag::Mat>(def)) {
        return world().type_mat_tuple(mat_type);
    }else if(auto pi = def->isa<Pi>()){
        if(pi->is_cn()){
            return world().cn(rewrite_type_cached(pi->dom()));
        }
    }else if(auto sigma = def->isa<Sigma>()){
        return world().sigma(sigma->ops().map([&](auto elem, auto){return rewrite_type_cached(elem);}));
    }

    return def;
}

const Def* Matrix2Tuple::rewrite_convert(const Def* def) {
    assert(!isa<Tag::MOp>(def) && !isa<Tag::Map>(def));

    if(auto mat = def->isa<Mat>()){
        return world().tuple(mat->ops().map([&](auto elem, auto){return rewrite_cached(elem);}));
    }else if(auto tuple = def->isa<Tuple>()){
        return world().tuple(tuple->ops().map([&](auto elem, auto){return rewrite_cached(elem);}));
    }else if(auto extract = def->isa<Extract>()){
        auto tuple = rewrite_cached(extract->tuple());
        auto idx = rewrite_cached(extract->index());
        return world().extract(tuple, idx);
    }else if(auto lea = isa<Tag::LEA>(def)){
        auto pointee = rewrite_cached(lea->arg(0));
        auto idx = rewrite_cached(lea->arg(1));
        return world().op_lea(pointee, idx);
    }else if(auto store = isa<Tag::Store>(def)){
        auto mem = rewrite_cached(store->arg(0));
        auto ptr = rewrite_cached(store->arg(1));
        auto old_value = store->arg(2);
        auto value = rewrite_cached(old_value);
        return world().op_store(mem, ptr, value);
    }else if(auto load = isa<Tag::Load>(def)){
        auto mem = rewrite_cached(load->arg(0));
        auto ptr = rewrite_cached(load->arg(1));
        return world().op_load(mem, ptr);
    }else if(auto slot = isa<Tag::Slot>(def)){
        auto [type, addr_space] = slot->callee()->isa<App>()->arg()->projs<2>();
        auto mem = rewrite_cached(slot->arg(0));
        return world().op_slot(type, mem);
    }else if(auto old_lam= def->isa_nom<Lam>()) {
        if(isa_workable(old_lam)){
            DefVec new_doms, new_vars, new_args;
            auto old_doms = old_lam->doms();

            for (auto dom : old_doms) {
                new_doms.emplace_back(rewrite_type_cached(dom));
            }

            auto target_sigma = world().sigma(new_doms);

            auto new_pi  = world().cn(target_sigma);
            auto new_lam = world().nom_filter_lam(new_pi, old_lam->dbg());

            for (size_t var_i = 0, n = old_doms.size(); var_i < n; ++var_i) {
                old2new_[old_lam->var(var_i)] = new_lam->var(var_i);
            }
            old2new_[old_lam] = new_lam;

            auto body = rewrite_cached(old_lam->body());
            new_lam->set_body(body);

            return new_lam;
        }else{
            return old_lam;
        }
    }else if (auto app = def->isa<App>()){
        auto old_app_lam = app->callee();

        if(old_app_lam->isa<Lam>()){
            auto new_lam = rewrite_cached(old_app_lam)->isa_nom<Lam>();
            DefVec new_args;

            for (size_t var_i = 0, n = app->num_args(); var_i != n; ++var_i) {
                new_args.emplace_back(rewrite_cached(app->arg(var_i)));
            }

            auto new_args_tuple = world().tuple(new_args);
            return world().app(new_lam, new_args_tuple);
        }else if(auto bitcast = isa<Tag::Bitcast>(def)) {

            auto new_ops = def->ops().map([&](auto elem, auto){return rewrite_cached(elem);});
            auto new_type = rewrite_type_cached(def->type());

            auto [dst_type, src_type] = bitcast->callee()->as<App>()->arg()->projs<2>();
            auto src = bitcast->arg(0);

            return world().op_bitcast(rewrite_type_cached(dst_type), rewrite_cached(src));
        }else if(isa<Tag::Mem>(def->type())) {
            return def;
        }else{
            return def->rebuild(world(), rewrite_type_cached(def->type()), def->ops().map([&](auto elem, auto){return rewrite_cached(elem);}), def->dbg());
        }
    }else if(auto lit = def->isa<Lit>()){
        return def;
    }else if(auto global = def->isa<Global>()){
        return def;
    }else{
        return def->rebuild(world(), rewrite_type_cached(def->type()), def->ops().map([&](auto elem, auto){return rewrite_cached(elem);}), def->dbg());
    }

    thorin::unreachable();
}


}
