#include "thorin/pass/rw/auto_diff_util.h"

#include <algorithm>
#include <string>

#include "thorin/analyses/scope.h"

namespace thorin {

/// @name macros - some useful macro definitions
///@{
#define THORIN_UNREACHABLE        assert(false && "Unreachable")
#define dlog(world, ...)          world.DLOG(__VA_ARGS__)
#define type_dump(world, name, d) world.DLOG("{} {} : {}", name, d, d->type())
///@}
// end macros

/// @name eviction - functions that are general enough to be replaced or moved out to other places
///@{
// TODO: move to world
size_t getDim(const Def* def) {
    // TODO: test def, idef, tuple
    if (auto arr = def->isa<Arr>()) {
        return arr->shape()->as<Lit>()->get<uint8_t>();
    } else if (auto arr = def->type()->isa<Arr>()) {
        return getDim(def->type());
    } else {
        return def->num_projs();
        // ptr -> 1
        // tuple -> size
    }
}

// TODO: replace with correct world method
const Pi* isReturning(const Pi* pi) {
    if (pi->is_cn() && pi->num_doms() > 0) {
        auto ret = pi->dom(pi->num_doms() - 1);
        if (auto ret_pi = ret->isa<Pi>(); ret_pi != nullptr && ret_pi->is_cn()) return ret_pi;
    }

    return nullptr;
}

// TODO: use tuple.cpp flatten
// const Def* flatten(const Def* def);
// size_t flatten(DefVec& ops, const Def* def, bool flatten_sigmas = true);
DefArray flat_tuple(const DefArray& defs, bool preserveFatPtr = false) {
    // or use concat
    std::vector<const Def*> v;
    for (auto def : defs) {
        if (auto tup = def->type()->isa<Sigma>()) {
            auto dim = def->num_projs();
            for (size_t j = 0; j < dim; j++) { v.push_back(def->proj(j)); }
        } else {
            v.push_back(def);
        }
    }
    return {v};
}

// TODO: replace with more general handling
//      or move to mem dialect
DefArray vars_without_mem_cont(const Lam* lam) {
    // ? 1 : 0 is superfluous (see 7.8.4 in C++ 20 standard) but increases readability
    return lam->vars().skip(isa<Tag::Mem>(lam->var(0)->type()) ? 1 : 0, isReturning(lam->type()) != nullptr ? 1 : 0);
}

// TODO: move to for dialect
// special case: count, body =>
// for(int i=0;i<count;i++) body(i)
// repeat n
const Lam* repeatLam(World& world, const Def* count, const Lam* body) {
    // op_for with empty accu
    // args: mem
    //       begin, end, step : lit
    //       init: tuple
    //       body: cn[mem, i32, accu, cn[mem, accu]]
    //                                yield
    //       break: cn[mem, accu]

    // for :: [m: Nat , n: Nat , Ts: «n; *»] → Cn [Mem , Int m, Int m, Int m, «i: n; Is#i», Cn [Mem , i : Int m, «i:
    // n; Is#i», Cn [Mem , «i: n; Is#i»]], Cn [Mem , «i: n; Is#i»]];
    // mod, shape, types
    // cont, body, exit
    auto loop_entry =
        world.nom_filter_lam(world.cn({world.type_mem(), world.cn(world.type_mem())}), world.dbg("loop_entry"));

    auto mem_t = world.type_mem();
    auto i32_t = world.type_int_width(32);
    auto i64_t = world.type_int_width(64);

    auto accumulator_type = world.sigma({i32_t}); // dummy to avoid empty sigma
    auto yield_type       = world.cn({mem_t, accumulator_type});
    auto body_type        = world.cn({mem_t, i32_t, accumulator_type, yield_type});

    auto forbreak = world.nom_lam(world.cn({mem_t, accumulator_type}), world.dbg("break"));
    forbreak->app(false, loop_entry->ret_var(), {forbreak->mem_var(), world.extract(forbreak->var(1), 0_s)});

    auto forbody       = world.nom_lam(body_type, world.dbg("body"));
    auto loop_continue = world.nom_filter_lam(world.cn(world.type_mem()), world.dbg("loop_continue"));

    auto [mem, i, acctpl, yield] =
        forbody->vars<4>({world.dbg("mem"), world.dbg("i"), world.dbg("acctpl"), world.dbg("yield")});
    forbody->app(false, body, {mem, i, loop_continue});
    // use lam application (beta red/specialization) for shorter code
    loop_continue->app(false, yield, {loop_continue->mem_var(), acctpl});

    auto forloop = world.op_for(loop_entry->mem_var(), world.lit_int_width(32, 0), count, world.lit_int_width(32, 1),
                                {world.lit_int(0)}, forbody, forbreak);

    loop_entry->set_body(forloop);

    return loop_entry;
}

// delayed for loop that leaves the body to be filled in by the user
std::pair<const Lam*, Lam*> repeatLam(World& world, const Def* count) {
    Lam* body = world.nom_filter_lam(world.cn({world.type_mem(), world.type_int_width(64), world.cn(world.type_mem())}),
                                     world.dbg("loop_body"));
    const Lam* loop = repeatLam(world, count, body);
    return {loop, body};
}

// TODO: move to array
// TODO: expect array
// TODO: needed? try to avoid and remove it
// copies inputArr to outputArr of size size
const Def* copy(World& world, const Def* inputArr, const Def* outputArr, const Def* size) {
    auto [loop, loop_body] = repeatLam(world, size);

    auto idx = loop_body->var(1);

    auto input_p  = world.op_lea(inputArr, idx, world.dbg("a_p"));
    auto output_p = world.op_lea(outputArr, idx, world.dbg("stencil_p"));

    auto loop_mem = loop_body->mem_var();

    auto [load_mem, loadedValue] = world.op_load(loop_mem, input_p)->projs<2>();
    auto storeMem                = world.op_store(load_mem, output_p, loadedValue);

    loop_body->set_body(world.app(loop_body->ret_var(), storeMem));

    return loop;
}
///@}
// end eviction

/// @name utility - functions that are adjacent to autodiff but not necessarily interlinked
///@{
bool isFatPtrType(World& world_, const Def* type) {
    if (auto sig = type->isa<Sigma>(); sig && sig->num_ops() == 2) {
        // TODO: maybe use original type to detect

        //        isFatPtr = isa_sized_type(sig->op(0));
        //
        //
        if (auto ptr = isa<Tag::Ptr>(sig->op(1)); ptr && isa<Tag::Int>(sig->op(0))) {
            auto [pointee, addr_space] = ptr->arg()->projs<2>();
            if (pointee->isa<Arr>()) return true;
        }
    }
    return false;
}

// multidimensional addition of values
// needed for operation differentiation
// we only need a multidimensional addition

// TODO: Currently: sum takes mem, adds a and b and calls cont
// TODO: possible: sum := \lambda mem a b cont. cont(mem, a+b)
// TODO: revisit and simplify
const Lam* vec_add(World& world, const Def* a, const Def* b, const Def* cont) {
    if (auto arr = a->type()->isa<Arr>(); arr && !(arr->shape()->isa<Lit>())) { THORIN_UNREACHABLE; }

    auto sum_pb = world.nom_filter_lam(world.cn(world.type_mem()), world.dbg("sum_pb"));
    if (auto aptr = isa<Tag::Ptr>(a->type())) {
        auto [ty, addr_space] = aptr->arg()->projs<2>();

        auto mem = sum_pb->mem_var();

        auto [mem2, a_v] = world.op_load(mem, a)->projs<2>();
        auto [mem3, b_v] = world.op_load(mem2, b)->projs<2>();

        auto res_cont_type = world.cn_mem_flat(a_v->type());
        auto res_cont      = world.nom_filter_lam(res_cont_type, world.dbg("ptr_add_cont"));
        auto sum_cont      = vec_add(world, a_v, b_v, res_cont);
        sum_pb->set_body(world.app(sum_cont, mem3));
        auto rmem             = res_cont->mem_var();
        auto s_v              = world.tuple(vars_without_mem_cont(res_cont));
        auto [rmem2, sum_ptr] = world.op_slot(ty, rmem, world.dbg("add_slot"))->projs<2>();
        auto rmem3            = world.op_store(rmem2, sum_ptr, s_v);

        res_cont->set_body(world.app(cont, flat_tuple({rmem3, sum_ptr})));

        return sum_pb;
    }

    if (isFatPtrType(world, a->type())) {
        auto [size_a, arr_a] = a->projs<2>();
        auto [size_b, arr_b] = b->projs<2>();
        // size_b has to be size_a
        auto arr_size_nat             = world.op_bitcast(world.type_nat(), size_a);
        auto [arr_ty, arr_addr_space] = as<Tag::Ptr>(arr_a->type())->arg()->projs<2>();
        auto arr_sized_ty             = world.arr(arr_size_nat, arr_ty->as<Arr>()->body())->as<Arr>();
        auto [mem2, arr_c_def]        = world.op_alloc(arr_sized_ty, sum_pb->mem_var())->projs<2>();
        auto arr_c                    = world.op_bitcast(arr_a->type(), arr_c_def);
        auto [loop, loop_body]        = repeatLam(world, size_a);

        // TODO: replace with for loop
        auto loop_mem     = loop_body->mem_var();
        auto idx          = loop_body->var(1);
        auto loopContinue = loop_body->ret_var();
        auto inc          = world.op(Wrap::add, world.lit_nat(0), world.lit_int_width(64, 1), idx);
        // store into c
        auto a_p = world.op_lea(arr_a, idx, world.dbg("a_p"));
        auto b_p = world.op_lea(arr_b, idx, world.dbg("b_p"));
        auto c_p = world.op_lea(arr_c, idx, world.dbg("c_p"));
        // add pointers using vec_add
        // lea c, store into c

        auto [lmem2, a_v] = world.op_load(loop_mem, a_p)->projs<2>();
        auto [lmem3, b_v] = world.op_load(lmem2, b_p)->projs<2>();
        loop_mem          = lmem3;
        // load values manually to allow for easy (and direct) storage into c
        //        auto elem_res_cont_type = world.cn_mem(a_v->type());
        auto elem_res_cont_type = world.cn_mem_flat(a_v->type());
        auto elem_res_cont      = world.nom_filter_lam(elem_res_cont_type, world.dbg("tuple_add_cont"));
        auto element_sum_pb     = vec_add(world, a_v, b_v, elem_res_cont);
        auto c_v                = world.tuple(vars_without_mem_cont(elem_res_cont));
        auto res_mem            = elem_res_cont->mem_var();
        res_mem                 = world.op_store(res_mem, c_p, c_v);

        //        set loop
        loop_body->set_body(world.app(element_sum_pb, loop_mem));
        elem_res_cont->set_body(world.app(loopContinue, res_mem));
        auto loop_end = world.nom_filter_lam(world.cn(world.type_mem()), world.dbg("add_loop_exit"));
        loop_end->set_body(world.app(cont, flat_tuple({loop_end->mem_var(), world.tuple({size_a, arr_c})})));
        sum_pb->set_body(world.app(loop, {mem2, loop_end}));

        return sum_pb;
    }

    auto dim  = getDim(a);
    auto dimb = getDim(b);
    assert(dim == dimb && "Dimension in add should be equal");

    if (dim == 1) {
        sum_pb->set_body(world.app(cont, flat_tuple({sum_pb->mem_var(), world.op(ROp::add, (nat_t)0, a, b)})));
        return sum_pb;
    }

    DefArray ops{dim};
    auto ret_cont_type = cont->type()->as<Pi>();
    auto current_cont  = sum_pb;

    for (size_t i = 0; i < ops.size(); ++i) {
        // adds component-wise both vectors
        auto ai            = world.extract(a, i); // use op?
        auto bi            = world.extract(b, i);
        auto res_cont_type = world.cn_mem_flat(ai->type());
        auto res_cont      = world.nom_filter_lam(res_cont_type, world.dbg("tuple_add_cont"));
        auto sum_call      = vec_add(world, ai, bi, res_cont);
        ops[i]             = world.tuple(vars_without_mem_cont(res_cont));

        current_cont->set_body(world.app(sum_call, current_cont->mem_var()));

        current_cont = res_cont;
    }

    current_cont->set_body(world.app(cont, flat_tuple({current_cont->mem_var(), world.tuple(ops)})));

    return sum_pb;
}

std::pair<const Def*, const Def*>
lit_of_type(World& world, const Def* mem, const Def* type, const Def* like, r64 lit, const Def* dummy) {
    // TODO: a monad would be easier for memory
    if (like) {}

    auto isFatPtr = isFatPtrType(world, type);
    if (isFatPtr) {
        assert(like != nullptr);
        auto [arr_size, _] = like->projs<2>();

        auto ptr_ty               = as<Tag::Ptr>(type->op(1));
        auto [arr_ty, addr_space] = ptr_ty->arg()->projs<2>();
        auto arr                  = arr_ty->as<Arr>();

        auto arr_size_nat     = world.op_bitcast(world.type_nat(), arr_size);
        auto arr_sized_ty     = world.arr(arr_size_nat, arr_ty->as<Arr>()->body())->as<Arr>();
        auto [mem2, ptr_arr]  = world.op_alloc(arr_sized_ty, mem)->projs<2>();
        auto shape            = arr_size_nat;
        auto body             = arr->body();
        auto [mem3, body_lit] = lit_of_type(world, mem2, body, nullptr, lit, dummy);
        auto init             = world.pack(shape, body_lit);
        auto mem4             = world.op_store(mem3, ptr_arr, init);
        auto fat_ptr_arr      = world.tuple({arr_size, ptr_arr});
        return {mem4, fat_ptr_arr};
    }

    // TODO: not for idef array
    if (auto ptr = isa<Tag::Ptr>(type)) {
        auto [ty, addr_space] = ptr->arg()->projs<2>();

        // ty->isa<Arr> handled already by isFatPtr
        if (auto arr = ty->isa<Arr>()) {
            auto [mem2, ptr_arr]  = world.op_alloc(ty, mem)->projs<2>();
            auto shape            = arr->shape();
            auto body             = arr->body();
            auto [mem3, body_lit] = lit_of_type(world, mem2, body, nullptr, lit, dummy);
            auto init             = world.pack(shape, body_lit);
            auto mem4             = world.op_store(mem3, ptr_arr, init);
            return {mem4, ptr_arr};
        }

        auto [mem2, lit_ptr] = world.op_slot(ty, mem, world.dbg("lit_slot"))->projs<2>();
        auto [mem3, lit_res] = lit_of_type(world, mem2, ty, like, lit, dummy);
        auto mem4            = world.op_store(mem3, lit_ptr, lit_res);

        return {mem4, lit_ptr};
    }
    const Def* litdef;
    if (auto real = isa<Tag::Real>(type))
        litdef = world.lit_real(as_lit(real->arg()), lit);
    else if (auto a = type->isa<Arr>()) {
        auto dim = a->shape()->as<Lit>()->get<uint8_t>();
        DefArray ops{dim, [&](auto) {
                         auto [nmem, op] = lit_of_type(world, mem, a->body(), like, lit, dummy);
                         mem             = nmem;
                         return op;
                     }};
        litdef = world.tuple(ops);
    } else if (auto sig = type->isa<Sigma>()) {
        auto zops = sig->ops().map([&](auto op, auto index) {
            auto [nmem, zop] = lit_of_type(world, mem, op, like->proj(index), lit, dummy);
            mem              = nmem;
            return zop;
        });

        litdef = world.tuple(zops);
    } else
        litdef = dummy;

    return {mem, litdef};
}

// TODO: revisit
std::pair<const Def*, const Def*>
oneHot(World& world_, const Def* mem, u64 idx, const Def* shape, const Def* like, const Def* s) {
    auto [rmem, v] = ZERO(world_, mem, shape, like, s);
    return {rmem, world_.insert_unsafe(v, idx, s)};
}

std::pair<const Def*, const Def*>
oneHot(World& world_, const Def* mem, const Def* idx, const Def* shape, const Def* like, const Def* s) {
    // TODO: extend for different shapes => indef array
    // can one do better for a def array shape? => insert

    // TODO: insert for array; alloc for idef

    if (auto lit = isa_lit(idx)) {
        return oneHot(world_, mem, *lit, shape, like, s);
    } else {
        // TODO: wrong
        // TODO: fix like
        auto dim = getDim(shape);
        // instead of
        // ((1,0,0),(0,1,0),(0,0,1)) # idx
        // we build
        // ((1,0,0)#idx, (0,1,0)#idx, (0,0,1)#idx)
        // which is equivalent
        // but allows flattening (toplevel tupel)
        DefArray ohv{dim};

        for (size_t i = 0; i < dim; ++i) {
            // correct type shape here? => probably not but we know that the tranpose is the same
            auto [nmem, oh] = oneHot(world_, mem, i, shape, like, s);
            mem             = nmem;
            ohv[i]          = world_.extract_unsafe(oh, idx);
        }

        auto oh = world_.tuple(ohv);
        return {mem, oh};
    }
}

const Lam* lam_fat_ptr_wrap(World& world, const Lam* lam) {
    bool changed = false;
    DefArray doms{lam->num_doms()};
    DefArray src_doms = lam->doms();
    size_t i          = 0;
    for (auto dom : src_doms) {
        if (auto ptr = isa<Tag::Ptr>(dom)) {
            changed = true;
            doms[i] = world.sigma({world.type_int_width(64), ptr});
        } else {
            doms[i] = dom;
        }

        doms[i]->dump();

        i++;
    }

    if (changed) {
        auto cn      = world.cn(doms);
        Lam* wrapper = world.nom_filter_lam(cn, world.dbg("wrapper"));

        i = 0;
        DefArray arguments{lam->num_doms()};

        for (auto dom : src_doms) {
            auto var = wrapper->var(i);
            if (auto ptr = isa<Tag::Ptr>(dom)) {
                auto [size, arr] = var->projs<2>();
                arguments[i]     = arr;
            } else {
                arguments[i] = var;
            }

            i++;
        }

        wrapper->set_body(world.app(lam, arguments));

        return wrapper;
    }

    return lam;
}
///@}
// end utility

} // namespace thorin
