#include "thorin/pass/rw/lower_matrix.h"
#include <algorithm>

namespace thorin {

#define buil world().builder()



bool is_scalar(MOp mop){
    switch (mop) {
        case MOp::sadd:
        case MOp::smul:
        case MOp::ssub:
        case MOp::sdiv:{
            return true;
        }
        default:{
            return false;
        }
    }
}

bool is_unary(MOp mop){
    switch (mop) {
        case MOp::transpose:
        case MOp::sum:
        case MOp::max:{
            return true;
        }
        default:{
            return false;
        }
    }
}

bool is_binary(MOp mop){
    switch (mop) {
        case MOp::vec:
        case MOp::add:
        case MOp::sub:
        case MOp::mul:
        case MOp::div:{
            return true;
        }
        default:{
            return false;
        }
    }
}

const Def* zero(World& w, const Def* type){
    if(auto int_type = isa<Tag::Int>(type)){
        return w.lit_int(int_type, (u64)0, {});
    }else if(auto float_type = isa<Tag::Real>(type)){
        return w.lit_real(as_lit(float_type->arg()), 0.0);
    }

    thorin::unreachable();
}

enum Op{
    mul, add, sub, div
};

const Def* op(World& w, Op op, const Def* type, const Def* lhs, const Def* rhs){
    if(auto int_type = isa<Tag::Int>(type)){
        switch (op) {
            case add: return w.op(Wrap::add, WMode::none, lhs, rhs);
            case sub: return w.op(Wrap::sub, WMode::none, lhs, rhs);
            case mul: return w.op(Wrap::mul, WMode::none, lhs, rhs);
            case div: return w.op(Div::sdiv, WMode::none, lhs, rhs);
        }
    }else if(auto float_type = isa<Tag::Real>(type)){
        switch (op) {
            case add: return w.op(ROp::add, RMode::none, lhs, rhs);
            case sub: return w.op(ROp::sub, RMode::none, lhs, rhs);
            case mul: return w.op(ROp::mul, RMode::none, lhs, rhs);
            case div: return w.op(ROp::div, RMode::none, lhs, rhs);
        }
    }

    thorin::unreachable();
}

const Def* mul_add(World& w, const Def* type, const Def* lhs, const Def* rhs, const Def* carry){
    return op(w, Op::add, type, op(w, Op::mul, type, lhs, rhs), carry);
}

void LowerMatrix::construct_mop(MOp mop, const Def* elem_type, ConstructHelper& helper){
    World& w = world();

    auto& input = helper.impl;
    auto body = helper.body;

    switch (mop) {
        case MOp::vec: {
            auto i = helper.indices[0];

            LoopBuilder zeroLoopBuil(world());
            zeroLoopBuil.addLoop(input.right.getCols());
            NestedLoops zeroLoop = zeroLoopBuil.build();

            buil.mem(body).app_body(body, zeroLoop.entry);

            auto zero_j = zeroLoop.indices[0];
            auto zero_lea = helper.result.getLea( i, zero_j );

            auto zero_mem = w.op_store(zeroLoop.body->mem_var(), zero_lea, zero(w, elem_type));
            buil.add(zero_mem).app_body(zeroLoop.body, zeroLoop.body->ret_var());

            LoopBuilder mulLoopBuil(world());
            mulLoopBuil.addLoop(input.left.getCols());
            mulLoopBuil.addLoop(input.right.getCols());
            NestedLoops mulLoop = mulLoopBuil.build();

            buil.mem(zeroLoop.finish).app_body(zeroLoop.finish, mulLoop.entry);

            auto k = mulLoop.indices[0];
            auto j = mulLoop.indices[1];

            auto left_ptr = input.left.getLea(i, k);
            auto right_ptr = input.right.getLea(k, j);
            auto result_lea = helper.result.getLea( i, j );

            auto [result_load_mem, carry] = w.op_load(mulLoop.body->mem_var(), result_lea)->projs<2>();

            auto [left_load_mem, left_value] = w.op_load(result_load_mem, left_ptr)->projs<2>();
            auto [right_load_mem, right_value] = w.op_load(left_load_mem, right_ptr)->projs<2>();

            auto sum = mul_add(w, elem_type, left_value, right_value, carry);

            auto store_mem = w.op_store(right_load_mem, result_lea, sum);

            buil.add(store_mem).app_body(mulLoop.body, mulLoop.body->ret_var());
            buil.mem(mulLoop.finish).app_body(mulLoop.finish, body->ret_var());
            break;
        }
        case MOp::add:
        case MOp::sub:
        case MOp::mul:
        case MOp::div: {
            Op op_ty;
            switch (mop) {
                case MOp::add: op_ty = Op::add; break;
                case MOp::sub: op_ty = Op::sub; break;
                case MOp::mul: op_ty = Op::mul; break;
                case MOp::div: op_ty = Op::div; break;
                default: thorin::unreachable();
            }

            auto left_ptr = input.left.getLea(helper.indices);
            auto right_ptr = input.right.getLea(helper.indices);

            auto [left_load_mem, left_value] = w.op_load(body->mem_var(), left_ptr)->projs<2>();
            auto [right_load_mem, right_value] = w.op_load(left_load_mem, right_ptr)->projs<2>();

            auto result = op(w, op_ty, elem_type, left_value, right_value);
            auto result_lea = helper.result.getLea(helper.indices);
            auto store_mem = w.op_store(right_load_mem, result_lea, result);

            buil.add(store_mem).app_body(body, body->ret_var());
            break;
        }
        case MOp::sadd:
        case MOp::smul:
        case MOp::ssub:
        case MOp::sdiv: {
            Op op_ty;
            switch (mop) {
                case MOp::sadd: op_ty = Op::add; break;
                case MOp::smul: op_ty = Op::mul; break;
                case MOp::ssub: op_ty = Op::sub; break;
                case MOp::sdiv: op_ty = Op::div; break;
                default: thorin::unreachable();
            }

            auto right_ptr = input.right.getLea(helper.indices);
            auto [right_load_mem, right_value] = w.op_load(body->mem_var(), right_ptr)->projs<2>();
            auto result = op(w, op_ty, elem_type, input.arg, right_value);
            auto result_lea = helper.result.getLea(helper.indices);
            auto store_mem = w.op_store(right_load_mem, result_lea, result);

            buil.add(store_mem).app_body(body, body->ret_var());
            break;
        }
        case MOp::transpose: {
            auto src_ptr = input.left.getLea(helper.indices[1], helper.indices[0] );
            auto dst_lea = helper.result.getLea(helper.indices);

            auto [right_load_mem, right_value] = w.op_load(body->mem_var(), src_ptr)->projs<2>();
            auto store_mem = w.op_store(right_load_mem, dst_lea, right_value);

            buil.add(store_mem).app_body(body, body->ret_var());
            break;
        }
        case MOp::sum: {
            auto src_ptr = input.left.getLea(helper.indices);
            auto [right_load_mem, right_value]  = w.op_load(body->mem_var(), src_ptr)->projs<2>();

            auto sum = op(w, Op::add, elem_type, helper.vars[0], right_value);
            buil.add(right_load_mem).add(sum).app_body(body, body->ret_var());
            break;
        }
        case MOp::max: {
            auto index = input.left.getIndex(helper.indices);
            auto src_ptr = input.left.lea(index);

            auto [right_load_mem, right_value]  = w.op_load(body->mem_var(), src_ptr)->projs<2>();

            auto cmp = world().op(RCmp::g, RMode::none, right_value, helper.vars[0]);

            auto left_max = buil.mem().nom_filter_lam("left_max");
            auto right_max = buil.mem().nom_filter_lam("right_max");
            auto max_cmp = buil.mem().nom_lam("max_cmp");
            max_cmp->branch(true, cmp, left_max, right_max, right_load_mem);

            buil.mem(body).app_body(body, max_cmp);

            buil.mem(left_max).add(right_value).add(index).app_body(left_max, body->ret_var());
            buil.mem(right_max).add(helper.vars).app_body(right_max, body->ret_var());
            break;
        }
        /*

        case MOp::max: {
            auto index = input.left.getIndex(helper.indices);
            auto src_ptr = input.left.lea(index);

            auto [right_load_mem, right_value]  = w.op_load(body->mem_var(), src_ptr)->projs<2>();

            auto cmp = world().op(RCmp::g, RMode::none, right_value, helper.vars[0]);

            auto left_max = buil.mem().nom_filter_lam("left_max");
            auto right_max = buil.mem().nom_filter_lam("right_max");
            auto max_cmp = buil.mem().nom_lam("max_cmp");
            max_cmp->branch(true, cmp, left_max, right_max, right_load_mem);

            buil.mem(body).app_body(body, max_cmp);

            buil.mem(left_max).add(right_value).add(index).app_body(left_max, body->ret_var());
            buil.mem(right_max).add(helper.vars).app_body(right_max, body->ret_var());
            break;
        }
         * */

        case MOp::init: {
            auto lea = input.left.getLea(helper.indices);
            auto store_mem  = w.op_store(body->mem_var(), lea, input.arg);
            buil.add(store_mem).app_body(body, body->ret_var());
            break;
        }

        default: {}
    }
}

const Def* LowerMatrix::alloc_stencil(const Def* stencil, const Def* rows, const Def* cols,  const Def*& mem){
    if(auto tuple = stencil->isa<Sigma>()){
        return world().tuple(tuple->ops().map([&](auto elem, auto){ return alloc_stencil(elem, rows, cols, mem); }));
    }else if(auto mat = isa<Tag::Tn>(stencil)){
        auto elem_type = world().elem_ty_of_tn(mat);
        auto [new_mem, result_matrix] = world().op_create_matrix(elem_type, {rows, cols}, mem)->projs<2>();
        mem = new_mem;
        return result_matrix;
    }else{
        thorin::unreachable();
    }
}

void LowerMatrix::construct_mat_loop(const Def* stencil, const Def* rows, const Def* cols, const Def* alloc_rows, const Def* alloc_cols, bool flatten, ConstructHelper& helper){
    auto impl = helper.impl.lam;

    const Def* alloc_mem = impl->mem_var();
    auto result_matrix = alloc_stencil(stencil, alloc_rows, alloc_cols, alloc_mem);

    LoopBuilder loopBuilder(world());

    if(flatten){
        loopBuilder.addLoop(world().op(Wrap::mul, WMode::none, rows, cols));
    }else{
        loopBuilder.addLoop(rows);
        loopBuilder.addLoop(cols);
    }

    NestedLoops result = loopBuilder.build();

    buil
        .add(alloc_mem)
        .app_body(impl, result.entry);

    buil
        .mem(result.finish)
        .add(result_matrix)
        .app_body(result.finish, impl->ret_var());

    if(isa<Tag::Tn>(result_matrix->type())){
        helper.result.setMat(result_matrix);
    }

    helper.raw_result = result_matrix;
    helper.indices = result.indices;
    helper.body = result.body;
}

void assign_name(Lam* lam, MOp mop, const std::string& suffix = ""){
    std::string dbg_name;
    switch (mop) {
        case MOp::add: dbg_name = "add"; break;
        case MOp::sub: dbg_name = "sub"; break;
        case MOp::mul: dbg_name = "mul"; break;
        case MOp::div: dbg_name = "div"; break;

        case MOp::sadd: dbg_name = "sadd"; break;
        case MOp::ssub: dbg_name = "ssub"; break;
        case MOp::smul: dbg_name = "smul"; break;
        case MOp::sdiv: dbg_name = "sdiv"; break;

        case MOp::sum: dbg_name = "sum"; break;
        case MOp::max: dbg_name = "sum"; break;
        case MOp::init: dbg_name = "init"; break;
        case MOp::transpose: dbg_name = "transpose"; break;

        case MOp::vec: dbg_name = "vec"; break;
        default: thorin::unreachable();
    }

    dbg_name = "mop_" + dbg_name + suffix;
    lam->set_dbg(lam->world().dbg(dbg_name));
}

/*
const Lam* LowerMatrix::const_reduction(MOp mop, ROp rop, ConstructHelper& helper){
    auto& input = helper.entry;

    auto mat_const_check = buil.mem().nom_lam("mat_const_check");
    auto right_const_check = buil.mem().nom_lam("left_const");
    auto only_right_const_check = buil.mem().nom_lam("only_right_const_check");
    auto only_left_const_check = buil.mem().nom_lam("only_right_const_check");

    auto both_const = buil.mem().nom_filter_lam("both_const");
    auto only_left_const = buil.mem().nom_filter_lam("only_left_const");
    auto only_right_const = buil.mem().nom_filter_lam("only_right_const");
    auto none_const = buil.mem().nom_filter_lam("none_const");

    mat_const_check->branch(false, input.left.isConst(), right_const_check, only_right_const_check, mat_const_check->mem_var());

    auto [left_scalar_load_mem, left_scalar] = input.left.load_const_value(right_const_check->mem_var())->projs<2>();

    right_const_check->branch(false, input.right.isConst(), both_const, only_left_const, left_scalar_load_mem);
    only_right_const_check->branch(false, input.right.isConst(), only_right_const, none_const, only_right_const_check->mem_var());

    auto only_left_const_reduction = world().op(mop, MMode::none, only_left_const->mem_var(), left_scalar, input.right.getMat());

    auto [only_right_scalar_load_mem, only_right_scalar] = input.right.load_const_value(only_right_const->mem_var())->projs<2>();
    auto only_right_const_reduction = world().op(mop, MMode::none, only_right_scalar_load_mem, only_right_scalar, input.left.getMat());

    auto [right_scalar_load_mem, right_scalar] = input.right.load_const_value(both_const->mem_var())->projs<2>();
    auto right_const_reduction_value = world().op(rop, RMode::none, left_scalar, right_scalar);

    auto [both_result_mem, both_result_mat] = world().op_create_const_matrix(right_scalar->type(), {input.left.getRows(), input.left.getCols()}, right_scalar_load_mem, right_const_reduction_value)->projs<2>();

    auto entry_ret_var = input.lam->ret_var();
    buil.add(both_result_mem).add(both_result_mat).app_body(both_const, entry_ret_var);
    buil.flatten(only_left_const_reduction).app_body(only_left_const, entry_ret_var);
    buil.flatten(only_right_const_reduction).app_body(only_right_const, entry_ret_var);
    buil.mem(none_const).app_body(none_const, helper.impl_entry);

    return mat_const_check;
}*/

bool has_tn_result(MOp mop){
    return mop != MOp::init && mop != MOp::sum && mop != MOp::max;
}

DefArray tn_result_dims(MOp mop, DefArray args){
    MatrixHelper left{args[1]};

    const Def* rows;
    const Def* cols;
    if(mop == MOp::transpose ){
        rows = left.getCols();
        cols = left.getRows();
    }else if(mop == MOp::sum || mop == MOp::max){
        rows = left.getRows();
        cols = left.getCols();
    }else{
        MatrixHelper right{args[2]};

        if(mop == MOp::vec){
            rows = left.getRows();
        }else{
            rows = right.getRows();
        }

        cols = right.getCols();
    }

    return {rows, cols};
}

const Def* tn_result_size(MOp mop, DefArray args){
    World &world = args[0]->world();

    auto dims = tn_result_dims(mop, args);
    auto rows = dims[0];
    auto cols = dims[1];

    return world.op(Wrap::mul, WMode::none, rows, cols);
}

const Pi* LowerMatrix::mop_pi(MOp mop, size_t dims, const Def* elem_type, bool has_result_ptr){
    auto builder = world().builder();
    builder.mem();

    if(is_scalar(mop) || mop == MOp::init){
        builder.add(elem_type);
    }else if(is_binary(mop)){
        builder.type_matrix(dims, elem_type);
    }

    builder.type_matrix(dims, elem_type);

    if(has_result_ptr && has_tn_result(mop)){
        World &w = elem_type->world();
        auto arr_unknown_size_ty = w.arr(w.top_nat(), elem_type)->as<Arr>();
        auto ptr_unknown_size = w.type_ptr(arr_unknown_size_ty);
        builder.add(ptr_unknown_size);
    }

    auto result_builder = world().builder();
    result_builder.mem();

    if(mop == MOp::sum){
        result_builder.add(elem_type);
    }else if(mop == MOp::max){
        result_builder.add(elem_type).add(world().type_int_width(64));
    }else if(mop != MOp::init){
        result_builder.type_matrix(dims, elem_type);
    }

    builder.add(result_builder.cn());

    return builder.cn();
}

Lam* LowerMatrix::mop_lam(MOp mop, size_t dims, const Def* elem_type, bool has_result_ptr, const std::string& name){
    Lam* lam = world().nom_lam(mop_pi(mop, dims, elem_type, has_result_ptr), world().dbg(""));
    assign_name(lam, mop, name);
    return lam;
}

const Def* assign_arguments(Lam* lam, MOp mop, const Def* mmode, InputHelper& helper, bool with_result_ptr){
    const Def* result_mat = nullptr;
    auto mmode_lit = as_lit(mmode);

    /*auto ltrans = (mmode_lit & MMode::ltrans) > 0;
    auto rtrans = (mmode_lit & MMode::rtrans) > 0;

    helper.left.setTranspose(ltrans);
    helper.right.setTranspose(rtrans);
*/
    if(is_scalar(mop)){
        helper.arg = lam->var(1);
        helper.right.setMat(lam->var(2));
    }else if(is_unary(mop)){
        helper.left.setMat(lam->var(1));
    }else if(mop == MOp::init){
        helper.arg = lam->var(1);
        helper.left.setMat(lam->var(2));
    }else{
        helper.left.setMat(lam->var(1));
        helper.right.setMat(lam->var(2));
    }

    if(has_tn_result(mop) && with_result_ptr){
        World &w = lam->world();

        result_mat = w.mat(
                w.type_tn(2, w.type_real(64)),
                w.lit_int_width(64, 0),
                lam->var(lam->num_vars() - 2),
                tn_result_dims(mop, lam->vars())
            );
    }

    helper.lam = lam;
    return result_mat;
}

const Lam* LowerMatrix::create_MOp_impl(const Axiom* mop_axiom, size_t dim_size, const Def* elem_type, const Def* mmode){

    auto signature = world().tuple({mop_axiom, world().lit_int_width(64, dim_size), elem_type, mmode});

    if(mop_variants.contains(signature)){
        return mop_variants[signature];
    }

    ConstructHelper helper{world()};

    MOp mop = MOp(mop_axiom->flags());

    Lam* impl = mop_lam(mop, dim_size, elem_type, true, "_impl");
    impl->set_filter(false);

    mop_variants[signature] = impl;

    auto result_mat = assign_arguments(impl, mop, mmode, helper.impl, true);

    if(result_mat != nullptr){
        helper.result.setMat(result_mat);
        helper.raw_result = result_mat;
    }

    LoopBuilder loopBuilder(world());

    if(mop == MOp::sum){
        loopBuilder.addVar(elem_type, zero(world(), elem_type));
    }else if(mop == MOp::max){
        loopBuilder.addVar(elem_type, zero(world(), elem_type));
        loopBuilder.addVar(world().type_int_width(64), world().lit_int_width(64, 0));
    }

    if(mop != MOp::vec && mop != MOp::transpose){
        MatrixHelper tn{impl->var(2)};
        loopBuilder.addLoop(tn.getSize()); //1d loop
    }else if(mop == MOp::vec){
        auto dims = tn_result_dims(mop, impl->vars());
        auto rows = dims[0];

        loopBuilder.addLoop(rows);
    }else{
        auto dims = tn_result_dims(mop, impl->vars());
        auto rows = dims[0];
        auto cols = dims[1];

        loopBuilder.addLoop(rows); //2d loop
        loopBuilder.addLoop(cols);
    }

    NestedLoops result = loopBuilder.build();

    buil
        .mem(impl)
        .app_body(impl, result.entry);


    helper.vars = result.vars;
    helper.indices = result.indices;
    helper.body = result.body;

    buil.mem(result.finish)
        .add(result.reductions)
        .add_if_non_null(helper.raw_result)
        .app_body(result.finish, impl->ret_var());

    construct_mop( mop, elem_type, helper );
    return impl;
}

const Lam* LowerMatrix::create_MOp_entry(const Axiom* mop_axiom, size_t dims, const Def* elem_type, const Def* mmode){
    World& w = world();

    MOp mop = MOp(mop_axiom->flags());

    bool filter = true;

    Lam* entry = mop_lam(mop, dims, elem_type, false, "_entry");
    auto entry_ret_var = entry->ret_var();
    entry->set_filter(true);

    Lam* impl_entry = buil.mem().filter(filter).nom_filter_lam("impl_entry");

    InputHelper entryHelp(w);

    assign_arguments(entry, mop, mmode, entryHelp, false);
    buil.mem(entry).filter(filter).app_body(entry, impl_entry);

    if(is_binary(mop)){
        auto right_mat_lam = buil.mem().filter(filter).nom_filter_lam("right_mat_lam");
        auto left_mat_lam = buil.mem().filter(filter).nom_filter_lam("left_mat_lam");
        buil.mem(left_mat_lam).add(entryHelp.left.getMat()).app_body(left_mat_lam, entry_ret_var);
        buil.mem(right_mat_lam).add(entryHelp.right.getMat()).app_body(right_mat_lam, entry_ret_var);

        switch (mop) {
            case MOp::add: {
                auto const_check = buil.mem().nom_lam("const_check");
                buil.mem(entry).app_body(entry, const_check);
                auto left_not_zero = buil.mem().nom_lam("left_no_zero");
                const_check->branch(true, entryHelp.left.isZero(), right_mat_lam, left_not_zero, const_check->mem_var());
                left_not_zero->branch(true, entryHelp.right.isZero(), left_mat_lam, impl_entry, left_not_zero->mem_var());
                break;
            }
            case MOp::mul: {
                auto const_check = buil.mem().nom_lam("const_check");
                buil.mem(entry).app_body(entry, const_check);
                auto no_reduce_to_right = buil.mem().nom_lam("no_reduce_to_right");
                const_check->branch(true, world().op(Bit::_or, entryHelp.left.isZero(), entryHelp.right.isOne()), left_mat_lam, no_reduce_to_right, const_check->mem_var());
                no_reduce_to_right->branch(true, world().op(Bit::_or, entryHelp.right.isZero(), entryHelp.left.isOne()), right_mat_lam, impl_entry, no_reduce_to_right->mem_var());
                break;
            }
            default: {
            }
        }
    }

    auto impl = create_MOp_impl(mop_axiom, dims, elem_type, mmode);

    if(false && has_tn_result(mop)){ //reuse or alloc memory for result of operation
        auto arr_unknown_size_ty = w.arr(w.top_nat(), elem_type)->as<Arr>();
        auto ptr_unknown_size = w.type_ptr(arr_unknown_size_ty);

        auto result_size = tn_result_size(mop, entry->vars());

        auto default_size = world().lit_int_width(64, 0);

        auto default_fat = w.tuple({default_size, w.bot(ptr_unknown_size)});
        auto holder = w.global(w.type_ptr(default_fat->type(), (nat_t)0), true);
        holder->set(default_fat);

        auto memory_check = buil.mem().nom_lam("memory_check");
        auto memory_reuse = buil.mem().filter(filter).nom_filter_lam("memory_reuse");
        auto memory_alloc = buil.mem().filter(filter).nom_filter_lam("memory_alloc");
        auto impl_memory_entry = buil.mem().filter(filter).add(ptr_unknown_size).nom_filter_lam("impl_memory_entry");

        auto [load_mem, fat] = w.op_load(impl_entry->mem_var(), holder)->projs<2>();
        auto fat_ptr_size = w.extract(fat, (u64)0);
        auto fat_ptr = w.extract(fat, (u64)1);

        auto size_cmp = w.op(ICmp::uge, fat_ptr_size, result_size);

        memory_check->branch(false, size_cmp, memory_reuse, memory_alloc, memory_check->mem_var());

        auto arr_size_nat = w.op_bitcast(w.type_nat(), result_size);
        auto arr_sized_ty = w.arr(arr_size_nat, elem_type)->as<Arr>();
        auto [alloc_mem, alloc_ptr] = w.op_alloc(arr_sized_ty, memory_alloc->mem_var())->projs<2>();
        alloc_ptr = w.op_bitcast(ptr_unknown_size, alloc_ptr);

        auto new_fat = w.tuple({result_size, alloc_ptr});

        auto store_mem = w.op_store(alloc_mem, holder, new_fat);

        buil
            .add(store_mem)
            .add(alloc_ptr)
            .app_body(memory_alloc, impl_memory_entry);

        buil
            .mem(memory_reuse)
            .add(fat_ptr)
            .app_body(memory_reuse, impl_memory_entry);

        buil
            .add(load_mem)
            .app_body(impl_entry, memory_check);

        buil
            .mem(impl_memory_entry)
            .add(entry->vars().skip(1, 1))
            .add(impl_memory_entry->var(1))
            .add(entry->ret_var())
            .app_body(impl_memory_entry, impl);

    }else if(has_tn_result(mop)){
        auto arr_unknown_size_ty = w.arr(w.top_nat(), elem_type)->as<Arr>();
        auto ptr_unknown_size = w.type_ptr(arr_unknown_size_ty);

        auto result_size = tn_result_size(mop, entry->vars());


        auto arr_size_nat = w.op_bitcast(w.type_nat(), result_size);
        auto arr_sized_ty = w.arr(arr_size_nat, elem_type)->as<Arr>();
        auto [alloc_mem, alloc_ptr] = w.op_alloc(arr_sized_ty, impl_entry->mem_var())->projs<2>();
        alloc_ptr = w.op_bitcast(ptr_unknown_size, alloc_ptr);

        auto new_fat = w.tuple({result_size, alloc_ptr});

        buil
                .add(alloc_mem)
                .add(entry->vars().skip(1, 1))
                .add(alloc_ptr)
                .add(entry->ret_var())
                .app_body(impl_entry, impl);
    }else{
        buil
            .mem(impl_entry)
            .flatten(entry->var())
            .app_body(impl_entry, impl);
    }

    return entry;
}

const Def* LowerMatrix::rewrite_rec(const Def* current, bool convert){
    const Def* result;
    if(old2new.contains(current)){
        result = old2new[current];
    }else{
        assert(convert);
        result = rewrite_rec_convert(current);
        old2new[current] = result;
    }

    assert(!isa<Tag::MOp>(result));
    return result;
}

Lam* LowerMatrix::rewrite_mop(const App* app, const Def* arg_wrap){
    auto mop_app = app->callee()->as<App>();
    auto mop_axiom = mop_app->callee()->as<Axiom>();

    auto rmode = mop_app->arg(0);
    auto dim_count = as_lit(mop_app->arg(1));
    auto elem_type = mop_app->arg(2);

    auto mop_lam = create_MOp_entry(mop_axiom, dim_count, elem_type, rmode);

    auto res_buil = buil.mem();
    if(MOp(mop_axiom->flags()) == MOp::sum){
        res_buil.add(elem_type);
    }else if(MOp(mop_axiom->flags()) == MOp::max){
        res_buil.add(elem_type);
        res_buil.add(world().type_int_width(64));
    }else if(MOp(mop_axiom->flags()) != MOp::init){
        res_buil.type_matrix(2, elem_type);
    }

    auto result_lam = res_buil.nom_filter_lam(mop_lam->name() + "_mop_result");
    buil.flatten(arg_wrap).add(result_lam).app_body(chainHelper.tail, mop_lam);
    return result_lam;
}

void LowerMatrix::store_rec(const Def* value, const Def* mat, const Def* index, const Def*& mem){
    if(auto tuple = value->isa<Tuple>()){
        auto value_size = tuple->num_ops();
        auto mat_size = mat->num_ops();
        assert(value_size == mat_size);

        for(size_t i = 0 ; i < value_size ; i++){
            store_rec(value->op(i), mat->op(i), index, mem);
        }
    }else{
        mem = MatrixHelper(mat).store(mem, index, value);
    }
}


struct FormulaHelper{
    bool unary = false;
    bool scalar = false;
    const Def* elem_type = nullptr;
    const Def* result_type = nullptr;

    std::vector<const Def*> val_input;

    std::vector<size_t> lhs_indices;
    std::vector<size_t> rhs_indices;
    std::vector<size_t> res_indices;

    size_t lhs_dim = 0;
    size_t rhs_dim = 0;
    size_t res_dim = 0;

    std::vector<size_t> order2index;
    std::vector<size_t> index2order;
    std::vector<const Def*> sizes;
};

FormulaHelper constructHelper(const Def* eq, const Def* input){

    auto size = eq->num_ops();

    std::unordered_map<size_t, size_t> count;
    FormulaHelper helper;

    Equation formula;
    thorin::parse_equation(eq, formula);

    size_t val_index = 0;
    size_t in_index = 0;
    size_t dim_index = 0;

    Defs inputs;

    if(input->isa<Tuple>()){
        inputs = input->ops();
    }else{
        inputs = {input};
    }

    for( auto &eq_part : formula.inputs ){
        auto &vars = eq_part.vars;
        dim_index = 0;

        auto current_input = inputs[in_index];

        if(eq_part.variant == EquationInput::Tensor){
            auto dim = vars.size();
            (&helper.lhs_dim)[val_index] = dim;

            if(dim > 0){
                for( auto index_op : vars ){
                    auto index = index_op - 'a';

                    (&helper.lhs_indices)[val_index].push_back(index);

                    if(count.contains(index)){
                        count[index] = count[index] + dim_index;
                    }else{
                        MatrixHelper matrix(current_input);

                        if(helper.elem_type == nullptr){
                            helper.elem_type = matrix.elem_type();
                        }

                        helper.sizes.push_back(matrix.dim(dim_index));
                        count[index] = dim_index;
                    }

                    dim_index++;
                }
            }else if(helper.elem_type == nullptr){
                helper.elem_type = current_input->type();
            }

            helper.val_input.push_back(current_input);
            val_index++;
        }else{
            for( auto index_op : vars ){
                auto index = index_op - 'a';

                if(!count.contains(index)){
                    count[index] = 0;
                    helper.sizes.push_back(current_input);
                }

                dim_index++;
            }
        }

        in_index++;
    }


    dim_index = 0;
    helper.res_dim = formula.output.size();
    helper.scalar = helper.res_dim == 0;
    helper.result_type = helper.scalar ? helper.elem_type : eq->world().type_tn(helper.res_dim, helper.elem_type);
    for( auto index_op : formula.output ){
        auto index = index_op - 'a';
        helper.res_indices.push_back(index);
        assert(count.contains(index));
        count[index] = count[index] + dim_index;
        dim_index++;
    }

    helper.order2index.reserve(count.size());
    helper.index2order.reserve(count.size());
    for(auto const& entries: count){
        helper.order2index.push_back(entries.first);
    }

    sort(helper.order2index.begin(), helper.order2index.end(), [&](size_t i1, size_t i2)
    {
        return (count[i1] < count[i2]);
    });

    for( size_t i = 0 ; i < helper.order2index.size() ; i++){
        helper.index2order[helper.order2index[i]] = i;
    }

    helper.unary = helper.val_input.size() < 2;
    return helper;
}


Lam* LowerMatrix::rewrite_formula(const App* app, const Def* arg_wrap){
    auto map_app = app->callee()->as<App>();
    auto map_axiom = map_app->callee()->as<Axiom>();

    auto mem = arg_wrap->op(0);

    auto eq = arg_wrap->op(1);
    auto input = arg_wrap->op(2);


    FormulaHelper helper = constructHelper(eq, input);

    auto lhs_tn = helper.val_input[0];
    auto rhs_tn = helper.val_input[1];

    LoopBuilder loopBuilder(world());
    for( size_t order_index = 0; order_index < helper.order2index.size() ; order_index++ ){
        auto loop_index = helper.order2index[order_index];
        auto loop_size = helper.sizes[loop_index];

        loopBuilder.addLoop(loop_size);
    }

    if( helper.scalar ){
        loopBuilder.addVar(helper.elem_type, zero(world(), helper.elem_type));
    }

    NestedLoops loops = loopBuilder.build();

    auto loop_size = loops.indices.size();

    auto builder = buil.mem().add(lhs_tn->type());

    if(!helper.unary){
        builder.add(rhs_tn->type());
    }

    Lam *entry;
    const Def* result;
    const Def* after_entry_mem;
    if(helper.scalar){
        entry = builder
                .add( buil.mem().add(helper.elem_type).cn() )
                .filter(false)
                .nom_filter_lam("tn_formula_entry");

        result = loops.reductions[0];
        after_entry_mem = entry->mem_var();
    }else{
        DefVec out_dims;
        for( auto index : helper.res_indices ){
            auto size = helper.sizes[index];
            out_dims.push_back(size);
        }
        auto tn_type = world().type_tn(out_dims.size(), helper.elem_type);

        entry = builder
                .add( buil.mem().add(tn_type).cn() )
                .filter(false)
                .nom_filter_lam("tn_formula_entry");

        auto [result_mem, result_mat] = world().op_create_matrix(helper.elem_type, out_dims, entry->mem_var(), true)->projs<2>();

        after_entry_mem = result_mem;
        result = result_mat;
    }

    buil.add(after_entry_mem).app_body(entry, loops.entry);
    buil.mem(loops.finish).add(result).app_body(loops.finish, entry->ret_var());


    World &w = world();

    auto mapper = [&](std::vector<size_t>& loop_mapping){
        return [&](auto i){
            return loops.indices[helper.index2order[loop_mapping[i]]];
        };
    };


    const Def* result_lea = nullptr;
    const Def* carry;
    const Def* before_left_load_mem = loops.body->mem_var();
    if( helper.scalar ){
        carry = loops.vars[0];
    }else{
        MatrixHelper res_help(result);
        result_lea = res_help.getLea(helper.res_dim, mapper(helper.res_indices));
        auto [result_load_mem, result_load_val] = w.op_load(before_left_load_mem, result_lea)->projs<2>();
        carry = result_load_val;
        before_left_load_mem = result_load_mem;
    }

    const Def* left_value;
    const Def* before_right_load_mem = before_left_load_mem;
    if( helper.lhs_dim == 0){
        left_value = lhs_tn;
    }else{
        MatrixHelper lhs_help(lhs_tn);
        auto left_ptr = lhs_help.getLea(helper.lhs_dim, mapper(helper.lhs_indices));
        auto [left_load_mem, left_value_tmp] = w.op_load(before_left_load_mem, left_ptr)->projs<2>();
        before_right_load_mem = left_load_mem;
        left_value = left_value_tmp;
    }

    const Def* before_store_mem;
    const Def* value;
    if(helper.unary){
        before_store_mem = before_right_load_mem;
        value = left_value;
    }else{
        MatrixHelper rhs_helper(rhs_tn);
        auto right_ptr = rhs_helper.getLea(helper.rhs_dim, mapper(helper.rhs_indices));
        auto [right_load_mem, right_value] = w.op_load(before_right_load_mem, right_ptr)->projs<2>();
        value = op(w, Op::mul, helper.elem_type, left_value, right_value);
        before_store_mem = right_load_mem;
    }

    value = op(w, Op::add, helper.elem_type, value, carry);  //reduce dimension

    if( helper.scalar ){
        buil.add(before_store_mem).add(value).app_body(loops.body, loops.body->ret_var());
    }else{
        auto store_mem = w.op_store(before_store_mem, result_lea, value);
        buil.add(store_mem).app_body(loops.body, loops.body->ret_var());
    }

    auto result_lam = buil.mem()
            .add(helper.result_type)
            .nom_filter_lam("tn_formula_result");

    auto entry_call_buil = buil.add(mem).add(lhs_tn);

    if(!helper.unary){
        entry_call_buil.add(rhs_tn);
    }

    entry_call_buil.add(result_lam).app_body(chainHelper.tail, entry);

    return result_lam;
}

Lam* LowerMatrix::rewrite_map(const App* app, const Def* arg_wrap){
    auto map_app = app->callee()->as<App>();
    auto map_axiom = map_app->callee()->as<Axiom>();

    auto out_type = map_app->arg(1);

    auto mem = arg_wrap->op(0);
    auto mat = arg_wrap->op(1);
    auto map_f = arg_wrap->op(2);

    auto signature = world().tuple({map_axiom, map_f});

    Lam* entry;
    if(mop_variants.contains(signature)){
        entry = mop_variants[signature];
    }else{
        auto elem_type = world().elem_ty_of_tn(mat->type());

        entry = buil.mem()
                .type_matrix(2, elem_type)
                .add(
                        buil.mem().add(out_type).cn()
                )
                .filter(false)
                .nom_filter_lam("matrix_unary_entry");

        ConstructHelper helper{world()};

        auto& impl = helper.impl;

        impl.left.setMat(entry->var(1));

        auto rows = impl.left.getRows();
        auto cols = impl.left.getCols();
        impl.lam = entry;

        construct_mat_loop(out_type, rows, cols, rows, cols, false, helper);

        World &w = world();

        auto body = helper.body;
        auto result_matrix = helper.raw_result;

        auto src_index = impl.left.getIndex(helper.indices);
        auto [right_load_mem, right_value] = impl.left.load(body->mem_var(), src_index)->projs<2>();

        auto f_type = map_f->type()->as<Pi>();
        auto map_result_lam = world().nom_filter_lam(f_type->doms().back()->as<Pi>(), world().lit_false(), world().dbg(""));

        auto store_mem = map_result_lam->mem_var();
        store_rec(world().tuple(map_result_lam->vars().skip_front()), result_matrix, src_index, store_mem);

        map_result_lam->set_body(world().app(body->ret_var(), store_mem));
        buil.add(right_load_mem).add(right_value).add(map_result_lam).app_body(body, map_f);

        mop_variants[signature] = entry;
    }

    auto result_lam = buil.mem()
            .add(out_type)
            .nom_filter_lam("mat_mul_res");
    buil.add(mem).add(mat).add(result_lam).app_body(chainHelper.tail, entry);
    return result_lam;
}

const Def* LowerMatrix::rewrite_app(const App* app){
    auto arg = app->arg();
    auto arg_wrap = rewrite_rec(arg);

    Lam* result_lam;
    if(isa<Tag::MOp>(app)){
        result_lam = rewrite_mop(app, arg_wrap);
    }else if(isa<Tag::Map>(app)){
        result_lam = rewrite_map(app, arg_wrap);
    }else if(isa<Tag::Formula>(app)){
        result_lam = rewrite_formula(app, arg_wrap);
    }else{
        thorin::unreachable();
    }

    chainHelper = {
        .currentMem = result_lam->mem_var(),
        .tail = result_lam,
    };

    return result_lam->var();
}

const Def* LowerMatrix::rewrite_rec_convert(const Def* current){
    if(auto lam = current->isa_nom<Lam>(); isa_workable(lam)){
        ChainHelper oldHelper = chainHelper;

        chainHelper = {
            .currentMem = lam->mem_var(),
            .tail = lam,
        };

        auto result = rewrite_rec(lam->body());

        chainHelper.tail->set_body(result);
        chainHelper = oldHelper;
        return lam;
    }else if (isa<Tag::MOp>(current) || isa<Tag::Map>(current) || isa<Tag::Formula>(current)) {
        return rewrite_app(current->as<App>());
    }else if(auto mat = isa<Tag::Tn>(current->type())){
        return current->rebuild(world(), current->type(), current->ops().map([&](auto elem, auto){return rewrite_rec(elem);}), mat->dbg());
    }else if(auto arr = current->isa<Arr>()){
        auto size = rewrite_rec(arr->op(0));
        auto type = rewrite_rec(arr->op(1));
        return world().arr(size, type);
    }else if(auto alloc = isa<Tag::Alloc>(current)){
        auto mem = rewrite_rec(alloc->arg());
        auto callee = alloc->callee()->as<App>();
        auto arr = rewrite_rec(callee->arg(0));
        return world().op_malloc(arr, mem);
    }else if(auto bitcast = isa<Tag::Bitcast>(current)){
        auto rewritten_arg = rewrite_rec(bitcast->arg());
        auto callee = bitcast->callee()->as<App>();
        auto dst_type = callee->arg(0);
        return world().op_bitcast(dst_type, rewritten_arg);
    }else if(auto app = current->isa<App>()){
        auto arg = app->arg();
        auto args_rewritten = rewrite_rec(arg);
        auto arg_proj = args_rewritten->projs();
        auto callee = app->callee();
        if(auto lam = callee->isa_nom<Lam>()){
            return world().app(rewrite_rec(lam), arg_proj);
        }else{
            return world().app(callee, arg_proj);
        }
    }else if(auto tuple = current->isa<Tuple>()){
        auto wrapped = tuple->projs().map([&](auto elem, auto) { return rewrite_rec(elem); });
        auto resultTuple = world().tuple(wrapped);
        return resultTuple;
    }else if(auto extract = current->isa<Extract>()){
        auto jeidx= rewrite_rec(extract->index());
        auto jtup = rewrite_rec(extract->tuple());

        if(jtup->num_projs() != extract->tuple()->num_projs()){
            rewrite_rec(extract->tuple());
        }

        return world().extract_unsafe(jtup, jeidx,extract->dbg());
    }else if(auto lit = current->isa<Lit>()) {
        return lit;
    }else if(auto var = current->isa<Var>()){
        return var;
    }

    return current;
}

void LowerMatrix::enter() {
    chainHelper = {
        .currentMem = curr_nom()->mem_var(),
        .tail = curr_nom(),
    };

    auto result = rewrite_rec(curr_nom()->body());
    chainHelper.tail->set_body(result);
}


}
