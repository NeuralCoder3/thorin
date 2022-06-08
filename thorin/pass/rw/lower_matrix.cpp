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
        case MOp::sum:{
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

            auto [left_col_loop, left_col_yield] = w.repeat(input.left.getCols(), {elem_type});

            auto left_col_loop_result = buil
                    .mem()
                    .add(elem_type)
                    .nom_filter_lam("left_row_loop_result");

            buil
                    .mem(body)
                    .add(zero(w, elem_type))
                    .add(left_col_loop_result)
                    .app_body(body, left_col_loop);

            buil.mem(left_col_loop_result).app_body(left_col_loop_result, body->ret_var());

            auto index = left_col_yield->var(1);
            auto carry = left_col_yield->var(2);

            auto left_row_index = helper.indices[0];
            auto right_col_index = helper.indices[1];

            auto left_ptr = input.left.getLea(left_row_index, index);
            auto right_ptr = input.right.getLea(index, right_col_index);

            auto [left_load_mem, left_value] = w.op_load(left_col_yield->mem_var(), left_ptr)->projs<2>();
            auto [right_load_mem, right_value] = w.op_load(left_load_mem, right_ptr)->projs<2>();

            auto sum = mul_add(w, elem_type, left_value, right_value, carry);

            buil
                    .add(right_load_mem)
                    .add(sum)
                    .app_body(left_col_yield, left_col_yield->ret_var());

            auto left_lea = helper.result.getLea( left_row_index, right_col_index);

            auto store_mem = w.op_store(left_col_loop_result->mem_var(), left_lea, left_col_loop_result->var(1));
            buil.add(store_mem).app_body(left_col_loop_result, body->ret_var());
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

            auto left_row_index = helper.indices[0];
            auto right_col_index = helper.indices[1];

            auto src_ptr = input.left.getLea(left_row_index, right_col_index);
            auto dst_lea = helper.result.getLea(right_col_index, left_row_index);

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
        case MOp::init: {
            auto lea = input.left.getLea(helper.indices);
            auto store_mem  = w.op_store(body->mem_var(), lea, input.arg);
            buil.add(store_mem).app_body(body, body->ret_var());
            break;
        }

        default: {}
    }
}

void LowerMatrix::construct_scalar_loop(const Def* elem_type, const Def* rows, const Def* cols, ConstructHelper& helper){
    auto impl = helper.impl.lam;

    LoopBuilder loopBuilder(world());

    loopBuilder.addVar(elem_type, zero(world(), elem_type));
    loopBuilder.addLoop(world().op(Wrap::mul, WMode::none, rows, cols));

    LoopBuilderResult result = loopBuilder.build();

    buil
        .mem(result.finish)
        .add(result.reductions)
        .app_body(result.finish, impl->ret_var());

    buil
        .mem(impl)
        .app_body(impl, result.entry);

    helper.vars = result.vars;
    helper.indices = result.indices;
    helper.body = result.body;
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

    LoopBuilderResult result = loopBuilder.build();

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

void LowerMatrix::construct_void_loop(const Def* rows, const Def* cols, ConstructHelper& helper){
    auto impl = helper.impl.lam;

    LoopBuilder loopBuilder(world());

    loopBuilder.addLoop(world().op(Wrap::mul, WMode::none, rows, cols));

    LoopBuilderResult result = loopBuilder.build();

    buil
        .mem(result.finish)
        .app_body(result.finish, impl->ret_var());

    buil
        .mem(impl)
        .app_body(impl, result.entry);

    helper.indices = result.indices;
    helper.body = result.body;
}

void assign_name(Lam* lam, MOp mop, std::string suffix = ""){
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

const Pi* LowerMatrix::mop_pi(MOp mop, const Def* elem_type){

    const Pi* entry;
    if(is_scalar(mop)){
        entry = buil.mem()
                .add(elem_type)
                .type_matrix(2, elem_type)
                .add(
                    buil.mem().type_matrix(2, elem_type).cn()
                )
                .cn();

    }else if(is_unary(mop)){
        auto ret_pi_buil = buil.mem();

        if(mop == MOp::transpose){
            ret_pi_buil.type_matrix(2, elem_type);
        }else{
            ret_pi_buil.add(elem_type);
        }

        entry = buil.mem()
            .type_matrix(2, elem_type)
            .add(
                ret_pi_buil.cn()
            )
            .cn();

    }else if(mop == MOp::init){
        entry = buil.mem()
            .add(elem_type)
            .type_matrix(2, elem_type)
            .add(
                buil.mem().cn()
            )
            .cn();
    }else{
        entry = buil
            .mem()
            .type_matrix(2, elem_type)
            .type_matrix(2, elem_type)
            .add(
                buil.mem().type_matrix(2, elem_type).cn()
            ).cn();
    }

    return entry;
}

Lam* LowerMatrix::mop_lam(MOp mop, const Def* elem_type, const std::string& name){
    Lam* lam = world().nom_lam(mop_pi(mop, elem_type), world().dbg(""));
    assign_name(lam, mop, name);
    return lam;
}

void assign_arguments(Lam* lam, MOp mop, const Def* mmode, InputHelper& helper){
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

    helper.lam = lam;
}

const Lam* LowerMatrix::create_MOp_impl(const Axiom* mop_axiom, const Def* elem_type, const Def* mmode){

    auto signature = world().tuple({mop_axiom, elem_type, mmode});

    if(mop_variants.contains(signature)){
        return mop_variants[signature];
    }

    ConstructHelper helper{world()};

    auto& implHelp = helper.impl;

    MOp mop = MOp(mop_axiom->flags());

    Lam* impl = mop_lam(mop, elem_type, "_impl");
    impl->set_filter(false);

    mop_variants[signature] = impl;

    assign_arguments(impl, mop, mmode, helper.impl);

    const Def *rows, *cols;
    if(is_scalar(mop)){
        rows = implHelp.right.getRows();
        cols = implHelp.right.getCols();

        construct_mat_loop(world().type_tn(2, elem_type), rows, cols, rows, cols, true, helper);
    }else if(is_unary(mop)){
        rows = implHelp.left.getRows();
        cols = implHelp.left.getCols();

        auto const_check = buil.mem().nom_lam("const_check");
        auto const_exit = buil.mem().nom_filter_lam("const_exit");

        if(mop == MOp::transpose){
            construct_mat_loop(world().type_tn(2, elem_type), rows, cols, cols, rows, false, helper);
        }else if(mop == MOp::sum){
            construct_scalar_loop(elem_type, rows, cols, helper);
        }else{
            thorin::unreachable();
        }
    }else if(mop == MOp::init){
        rows = implHelp.left.getRows();
        cols = implHelp.left.getCols();

        construct_void_loop(rows, cols, helper);
    }else{
        rows = implHelp.left.getRows();
        cols = implHelp.right.getCols();

        construct_mat_loop( world().type_tn(2, elem_type), rows, cols, rows, cols, mop != MOp::vec, helper);
    }

    construct_mop( mop, elem_type, helper );
    return impl;
}

const Lam* LowerMatrix::create_MOp_lam(const Axiom* mop_axiom, const Def* elem_type, const Def* mmode){
    World& w = world();

    MOp mop = MOp(mop_axiom->flags());

    Lam* entry = mop_lam(mop, elem_type, "_entry");
    auto entry_ret_var = entry->ret_var();
    entry->set_filter(true);

    Lam* impl_entry = buil.mem().nom_filter_lam("impl_entry");

    InputHelper entryHelp(w);

    assign_arguments(entry, mop, mmode, entryHelp);

    buil.mem(entry).app_body(entry, impl_entry);

    if(is_binary(mop)){
        auto right_mat_lam = buil.mem().nom_filter_lam("right_mat_lam");
        auto left_mat_lam = buil.mem().nom_filter_lam("left_mat_lam");
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

    auto impl = create_MOp_impl(mop_axiom, elem_type, mmode);
    buil.mem(impl_entry).flatten(entry->var()).app_body(impl_entry, impl);
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
    auto dim_count = mop_app->arg(1);
    auto elem_type = mop_app->arg(2);

    auto mop_lam = create_MOp_lam(mop_axiom, elem_type, rmode);

    auto res_buil = buil.mem();
    if(MOp(mop_axiom->flags()) == MOp::sum){
        res_buil.add(elem_type);
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
                .set_filter(false)
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
    }else if (isa<Tag::MOp>(current) || isa<Tag::Map>(current)) {
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
