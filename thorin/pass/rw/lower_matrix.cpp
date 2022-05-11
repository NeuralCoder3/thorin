#include "thorin/pass/rw/lower_matrix.h"
#include <algorithm>

#define dlog(world,...) world.DLOG(__VA_ARGS__)
#define type_dump(world,name,d) world.DLOG("{} {} : {}",name,d,d->type())

namespace thorin {

#define builder world().builder()

bool is_scalar(MOp mop){
    switch (mop) {
        case MOp::sadd:
        case MOp::smul:
        case MOp::ssub:{
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

const Def* multiply(World& w, const Def* type, const Def* lhs, const Def* rhs){
    if(auto int_type = isa<Tag::Int>(type)){
        return w.op(Wrap::mul, (nat_t)0, lhs, rhs);
    }else if(auto float_type = isa<Tag::Real>(type)){
        return w.op(ROp::mul, (nat_t)0, lhs, rhs);
    }

    thorin::unreachable();
}

const Def* multiply_carry(World& w, const Def* type, const Def* lhs, const Def* rhs, const Def* carry){
    auto result = multiply(w, type, lhs, rhs);

    if(auto int_type = isa<Tag::Int>(type)){
        return w.op(Wrap::add, (nat_t)0, result, carry);
    }else if(auto float_type = isa<Tag::Real>(type)){
        return w.op(ROp::add, (nat_t)0, result, carry);
    }

    thorin::unreachable();
}

void LowerMatrix::construct_mop(Lam* entry, const Def* elem_type, MOp mop, const Def* cols, ConstructResult& constructResult){
    World& w = world();

    auto [result_rows, result_cols, result_ptr] = constructResult.result_matrix->projs<3>();
    auto left_row_index = constructResult.left_row_index;
    auto right_col_index = constructResult.right_col_index;
    auto body = constructResult.body;

    switch (mop) {
        case MOp::mul: {

            auto a = entry->var(1);
            auto b = entry->var(2);

            auto [a_rows, a_cols, a_ptr] = a->projs<3>();
            auto [b_rows, b_cols, b_ptr] = b->projs<3>();

            auto [left_col_loop, left_col_yield] = w.repeat(a_cols, {elem_type});

            auto left_col_loop_result = builder
                    .mem()
                    .add(elem_type)
                    .nom_filter_lam("left_row_loop_result");

            builder
                    .mem(body)
                    .add(zero(w, elem_type))
                    .add(left_col_loop_result)
                    .app_body(body, left_col_loop);

            builder.mem(left_col_loop_result).app_body(left_col_loop_result, body->ret_var());

            auto index = left_col_yield->var(1);
            auto carry = left_col_yield->var(2);

            auto left_index = w.row_col_to_index(left_row_index, index, a_cols);
            auto right_index = w.row_col_to_index(index, right_col_index, b_cols);

            auto left_ptr = w.op_lea(a_ptr, left_index);
            auto right_ptr = w.op_lea(b_ptr, right_index);

            auto [left_load_mem, left_value] = w.op_load(left_col_yield->mem_var(), left_ptr)->projs<2>();
            auto [right_load_mem, right_value]  = w.op_load(left_load_mem, right_ptr)->projs<2>();

            auto sum = multiply_carry(w, elem_type, left_value, right_value, carry);

            builder
                    .add(right_load_mem)
                    .add(sum)
                    .app_body(left_col_yield, left_col_yield->ret_var());

            auto result_index = w.row_col_to_index( left_row_index, right_col_index, b_cols);
            auto left_lea = w.op_lea(result_ptr, result_index);

            auto store_mem = w.op_store(left_col_loop_result->mem_var(), left_lea, left_col_loop_result->var(1));
            builder.add(store_mem).app_body(left_col_loop_result, body->ret_var());
            break;
        }
        case MOp::add:
        case MOp::sub: {
            auto rop = mop == MOp::add ? ROp::add : ROp::sub;

            auto a = entry->var(1);
            auto b = entry->var(2);

            auto [a_rows, a_cols, a_ptr] = a->projs<3>();
            auto [b_rows, b_cols, b_ptr] = b->projs<3>();

            auto index = w.row_col_to_index(left_row_index, right_col_index, cols);

            auto left_ptr = w.op_lea(a_ptr, index);
            auto right_ptr = w.op_lea(b_ptr, index);

            auto [left_load_mem, left_value] = w.op_load(body->mem_var(), left_ptr)->projs<2>();
            auto [right_load_mem, right_value]  = w.op_load(left_load_mem, right_ptr)->projs<2>();

            auto result = multiply(w, elem_type, left_value, right_value);
            auto result_lea = w.op_lea(result_ptr, index);
            auto store_mem = w.op_store(right_load_mem, result_lea, result);

            builder.add(store_mem).app_body(body, body->ret_var());
            break;
        }
        case MOp::sadd:
        case MOp::smul:
        case MOp::ssub: {
            ROp rop;
            switch (mop) {
                case MOp::sadd: {
                    rop = ROp::add;
                    break;
                }
                case MOp::smul: {
                    rop = ROp::mul;
                    break;
                }
                case MOp::ssub: {
                    rop = ROp::sub;
                    break;
                }
                default:
                    thorin::unreachable();
            }

            const Def* a = entry->var(1);
            const Def* b = entry->var(2);

            auto [b_rows, b_cols, b_ptr] = b->projs<3>();
            auto index = w.row_col_to_index(left_row_index, right_col_index, cols);
            auto right_ptr = w.op_lea(b_ptr, index);
            auto [right_load_mem, right_value]  = w.op_load(body->mem_var(), right_ptr)->projs<2>();
            auto result = multiply(w, elem_type, a, right_value);
            auto result_lea = w.op_lea(result_ptr, index);
            auto store_mem = w.op_store(right_load_mem, result_lea, result);

            builder.add(store_mem).app_body(body, body->ret_var());
            break;
        }
        default: {}
    }
}

void LowerMatrix::construct_loop(Lam* entry, const Def* elem_type,  const Def* a_rows, const Def* b_cols, ConstructResult& constructResult){
    auto [alloc_mem, result_matrix] = world().op_create_matrix(elem_type, a_rows, b_cols, entry->mem_var())->projs<2>();

    auto [left_row2_loop, left_row2_yield] = world().repeat(a_rows);
    auto left_row2_loop_result = builder.mem().nom_filter_lam("left_row2_loop_result");

    builder
            .mem(left_row2_loop_result)
            .add(result_matrix)
            .app_body(left_row2_loop_result, entry->ret_var());

    builder
            .add(alloc_mem)
            .add(left_row2_loop_result)
            .app_body(entry, left_row2_loop);

    auto [right_col2_loop, right_col2_yield] = world().repeat(b_cols);

    builder
            .mem(left_row2_yield)
            .add(left_row2_yield->ret_var())
            .app_body(left_row2_yield, right_col2_loop);


    auto left_row_index = left_row2_yield->var(1);
    auto right_col_index = right_col2_yield->var(1);

    constructResult = {
        .result_matrix = result_matrix,
        .left_row_index = left_row_index,
        .right_col_index = right_col_index,
        .body = right_col2_yield
    };
}

const Lam* LowerMatrix::create_MOp_lam(const Axiom* mop_axiom, const Def* elem_type){
    World& w = world();
    auto signature = w.tuple({mop_axiom, elem_type});

    if(mop_variants.contains(signature)){
        return mop_variants[signature];
    }

    MOp mop = MOp(mop_axiom->flags());

    ConstructResult constructResult{};
    Lam* entry;
    const Def* rows;
    const Def* cols;
    if(is_scalar(mop)){
        entry = builder
                .mem()
                .add(elem_type)
                .type_matrix(elem_type)
                .add(
                    builder.mem().type_matrix(elem_type).cn()
                ).nom_filter_lam("matrix_mul_entry");

        auto a = entry->var(1);
        auto b = entry->var(2);

        auto [b_rows, b_cols, b_ptr] = b->projs<3>();

        rows = b_rows;
        cols = b_cols;
    }else{
        entry = builder
                .mem()
                .type_matrix(elem_type)
                .type_matrix(elem_type)
                .add(
                    builder.mem().type_matrix(elem_type).cn()
                ).nom_filter_lam("matrix_mul_entry");

        auto a = entry->var(1);
        auto b = entry->var(2);

        auto [a_rows, a_cols, a_ptr] = a->projs<3>();
        auto [b_rows, b_cols, b_ptr] = b->projs<3>();

        rows = a_rows;
        cols = b_cols;
    }

    construct_loop(entry, elem_type, rows, cols, constructResult);
    construct_mop(entry, elem_type, mop, cols, constructResult);

    mop_variants[signature] = entry;
    return entry;
}

const Def* LowerMatrix::rewrite_rec(const Def* current){
    if(old2new.contains(current)){
        return old2new[current];
    }

    auto result = rewrite_rec_convert(current);
    assert(! isa<Tag::MOp>(result));
    old2new[current] = result;
    return result;
}

const Def* LowerMatrix::rewrite_rec_convert(const Def* current){

    if (auto mop = isa<Tag::MOp>(current)) {

        auto enter = builder.mem().nom_filter_lam("enter_mop");
        auto exit = builder.mem().nom_filter_lam("exit_mop");
        if(tail == nullptr){
            this->exit = exit;
        }else{
            builder.mem(exit).app_body(exit, tail);
        }
        tail = enter;
        
        auto arg = mop->arg();
        currentMem = enter->mem_var();
        auto arg_wrap = rewrite_rec(arg);

        auto elem_type = mop->callee()->as<App>()->arg(1);
        auto mop_axiom = mop.axiom();

        auto mop_lam = create_MOp_lam(mop_axiom, elem_type);
        auto result_lam = builder.mem().type_matrix(elem_type).nom_filter_lam("mat_mul_res");
        builder.flatten(arg_wrap).add(result_lam).app_body(enter, mop_lam);
        builder.mem(result_lam).app_body(result_lam, exit);

        if(head == nullptr){
            head = enter;
        }

        currentMem = exit->mem_var();
        auto result = result_lam->var();
        return result;
    }else if(auto app = current->isa<App>()){
        auto arg = app->arg();
        auto args_rewritten = rewrite_rec(arg);
        auto arg_proj = args_rewritten->projs();

        if(is_memop(current)){
            arg_proj.front() = currentMem;
        }

        auto app_wrap = world().app(app->callee(), arg_proj);

        if(is_memop(current)){
            currentMem = app_wrap->proj(0);
        }

        return app_wrap;
    }else if(auto tuple = current->isa<Tuple>()){
        auto wrapped = tuple->projs().map([&](auto elem, auto) { return rewrite_rec(elem); });
        auto resultTuple = world().tuple(wrapped);

        if(tuple->num_projs() != resultTuple->num_projs()){
            tuple->dump();
        }
        return resultTuple;
    }else if(auto extract = current->isa<Extract>()){
        auto jeidx= rewrite_rec(extract->index());
        auto jtup = rewrite_rec(extract->tuple());

        if(jtup->num_projs() != extract->tuple()->num_projs()){
            rewrite_rec(extract->tuple());
            tuple->dump();
        }

        return world().extract_unsafe(jtup, jeidx,extract->dbg());
    }else if(auto lit = current->isa<Lit>()) {
        return lit;
    }else if(auto var = current->isa<Var>()){
        return var;
    }else if(auto lam = current->isa<Lam>()){
        current->dump();
    }

    return current;
}

void LowerMatrix::enter() {
    head = nullptr;
    tail = nullptr;
    exit = curr_nom();
    currentMem = curr_nom()->mem_var();

    auto root = curr_nom()->body();
    auto result = rewrite_rec(root);

    exit->set_body(result);
    if(head != nullptr){
        builder.mem(curr_nom()).app_body(curr_nom(), head);
    }
}


}
