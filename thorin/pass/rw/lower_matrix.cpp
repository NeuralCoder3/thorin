#include "thorin/pass/rw/lower_matrix.h"
#include <algorithm>

#define dlog(world,...) world.DLOG(__VA_ARGS__)
#define type_dump(world,name,d) world.DLOG("{} {} : {}",name,d,d->type())

namespace thorin {

#define builder world().builder()

bool has_scalar(MOp mop){
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

void LowerMatrix::contruct(Lam* entry, const Def* a_rows, const Def* b_cols, ConstructResult& constructResult){
    auto [alloc_mem, result_matrix] = world().op_create_matrix(world().type_real(64), a_rows, b_cols, entry->mem_var())->projs<2>();

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
        .rows = a_rows,
        .cols = b_cols,
        .result_matrix = result_matrix,
        .left_row_index = left_row_index,
        .right_col_index = right_col_index,
        .body = right_col2_yield
    };
}

const Lam* LowerMatrix::create_MOp_lam(MOp mop, const Def* elem_type){
    World& w = world();

    ConstructResult constructResult;
    Lam* entry;
    if(has_scalar(mop)){
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

        contruct(entry, b_rows, b_cols, constructResult);
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

        contruct(entry, a_rows, b_cols, constructResult);
    }

    auto [result_rows, result_cols, result_ptr] = constructResult.result_matrix->projs<3>();

    auto left_row_index = constructResult.left_row_index;
    auto right_col_index = constructResult.right_col_index;
    auto body = constructResult.body;
    auto cols = constructResult.cols;

    switch (mop) {
        case MOp::mul: {

            auto a = entry->var(1);
            auto b = entry->var(2);

            auto [a_rows, a_cols, a_ptr] = a->projs<3>();
            auto [b_rows, b_cols, b_ptr] = b->projs<3>();

            auto [left_col_loop, left_col_yield] = w.repeat(a_cols, {w.type_real(64)});

            auto left_col_loop_result = builder
                    .mem()
                    .type_real(64)
                    .nom_filter_lam("left_row_loop_result");

            builder
                    .mem(body)
                    .add(w.lit_real(0.0))
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

            auto sum = w.op(ROp::add, (nat_t)0, w.op(ROp::mul, (nat_t)0, left_value, right_value), carry);

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

            auto result = w.op(rop, (nat_t)0, left_value, right_value);

            auto result_lea = w.op_lea(result_ptr, index);

            auto store_mem = w.op_store(right_load_mem, result_lea, result);

            builder.add(store_mem).app_body(body, body->ret_var());
            break;
        }
        case MOp::sadd:
        case MOp::smul:
        case MOp::ssub: {
            //auto rop = MOp(flags) == MOp::sadd ? ROp::add : ROp::sub;
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

            auto result = w.op(rop, (nat_t)0, a, right_value);

            auto result_lea = w.op_lea(result_ptr, index);

            auto store_mem = w.op_store(right_load_mem, result_lea, result);

            builder.add(store_mem).app_body(body, body->ret_var());
            break;
        }
        default: {}
    }

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

        auto elem_type = world().elem_ty_of_matrix(arg_wrap->op(2));
        auto mop_lam = create_MOp_lam(MOp(mop.flags()), elem_type);
        auto result_lam = builder.mem().type_matrix(elem_type).nom_filter_lam("mat_mul_res");
        //assert(arg_wrap->proj(0) == currentMem);
        builder.flatten(arg_wrap).add(result_lam).app_body(enter, mop_lam);

        //builder.add(alloc_mem).add(test_matrix).app_body(enter, result_lam);
        builder.mem(result_lam).app_body(result_lam, exit);

        if(head == nullptr){
            head = enter;
        }

        currentMem = exit->mem_var();
        auto result = result_lam->var();
        return result;
    }else if(auto app = current->isa<App>()){
        auto arg = app->arg();
        app->dump();
        app->type()->dump();
        app->callee()->dump();
        app->callee()->type()->dump();
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
