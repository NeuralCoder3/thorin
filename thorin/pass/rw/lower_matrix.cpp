#include "thorin/pass/rw/lower_matrix.h"
#include <algorithm>

namespace thorin {

#define builder world().builder()

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

const Def* zero(World& w, const Def* type){
    if(auto int_type = isa<Tag::Int>(type)){
        return w.lit_int(int_type, (u64)0, {});
    }else if(auto float_type = isa<Tag::Real>(type)){
        return w.lit_real(as_lit(float_type->arg()), 0.0);
    }else if(auto mat_type = isa<Tag::Mat>(type)){
        auto elem_type = mat_type->arg(0);
    }

    thorin::unreachable();
}

enum Op{
    mul, add, sub, div
};

const Def* op(World& w, Op op, const Def* rmode, const Def* type, const Def* lhs, const Def* rhs){
    if(auto int_type = isa<Tag::Int>(type)){
        switch (op) {
            case add: return w.op(Wrap::add, rmode, lhs, rhs);
            case sub: return w.op(Wrap::sub, rmode, lhs, rhs);
            case mul: return w.op(Wrap::mul, rmode, lhs, rhs);
        }
    }else if(auto float_type = isa<Tag::Real>(type)){
        switch (op) {
            case add: return w.op(ROp::add, rmode, lhs, rhs);
            case sub: return w.op(ROp::sub, rmode, lhs, rhs);
            case mul: return w.op(ROp::mul, rmode, lhs, rhs);
            case div: return w.op(ROp::div, rmode, lhs, rhs);
        }
    }

    thorin::unreachable();
}

const Def* mul_add(World& w, const Def* rmode, const Def* type, const Def* lhs, const Def* rhs, const Def* carry){
    return op(w, Op::add, rmode, type, op(w, Op::mul, rmode, type, lhs, rhs), carry);
}

void LowerMatrix::construct_mop(Lam* entry, MOp mop, const Def* rmode, const Def* elem_type, const Def* rows, const Def* cols, ConstructResult& constructResult){
    World& w = world();

    auto left_row_index = constructResult.left_row_index;
    auto right_col_index = constructResult.right_col_index;
    auto body = constructResult.body;

    switch (mop) {
        case MOp::vec: {
            entry->set_dbg(world().dbg("mop_mul"));
            auto [result_ptr, result_rows, result_cols] = constructResult.result_matrix->projs<3>();

            auto a = entry->var(1);
            auto b = entry->var(2);
            auto [b_ptr, b_rows, b_cols] = b->projs<3>();
            auto [a_ptr, a_rows, a_cols] = a->projs<3>();

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

            auto sum = mul_add(w, rmode, elem_type, left_value, right_value, carry);

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
        case MOp::sub:
        case MOp::mul:
        case MOp::div: {
            Op op_ty;
            switch (mop) {
                case MOp::add: entry->set_dbg(world().dbg("mop_add")); op_ty = Op::add; break;
                case MOp::sub: entry->set_dbg(world().dbg("mop_sub")); op_ty = Op::sub; break;
                case MOp::mul: entry->set_dbg(world().dbg("mop_mul")); op_ty = Op::mul; break;
                case MOp::div: entry->set_dbg(world().dbg("mop_div")); op_ty = Op::div; break;
                default: thorin::unreachable();
            }

            auto [result_ptr, result_rows, result_cols] = constructResult.result_matrix->projs<3>();

            auto a = entry->var(1);
            auto b = entry->var(2);
            auto [b_ptr, b_rows, b_cols] = b->projs<3>();

            auto [a_ptr, a_rows, a_cols] = a->projs<3>();

            auto index = w.row_col_to_index(left_row_index, right_col_index, cols);

            auto left_ptr = w.op_lea(a_ptr, index);
            auto right_ptr = w.op_lea(b_ptr, index);

            auto [left_load_mem, left_value] = w.op_load(body->mem_var(), left_ptr)->projs<2>();
            auto [right_load_mem, right_value]  = w.op_load(left_load_mem, right_ptr)->projs<2>();

            auto result = op(w, op_ty, rmode, elem_type, left_value, right_value);
            auto result_lea = w.op_lea(result_ptr, index);
            auto store_mem = w.op_store(right_load_mem, result_lea, result);

            builder.add(store_mem).app_body(body, body->ret_var());
            break;
        }
        case MOp::sadd:
        case MOp::smul:
        case MOp::ssub:
        case MOp::sdiv: {
            Op op_ty;
            switch (mop) {
                case MOp::sadd: entry->set_dbg(world().dbg("mop_sadd")); op_ty = Op::add; break;
                case MOp::smul: entry->set_dbg(world().dbg("mop_smul")); op_ty = Op::mul; break;
                case MOp::ssub: entry->set_dbg(world().dbg("mop_ssub")); op_ty = Op::sub; break;
                case MOp::sdiv: entry->set_dbg(world().dbg("mop_sdiv")); op_ty = Op::div; break;
                default: thorin::unreachable();
            }

            auto [result_ptr, result_rows, result_cols] = constructResult.result_matrix->projs<3>();

            auto a = entry->var(1);
            auto b = entry->var(2);
            auto [b_ptr, b_rows, b_cols] = b->projs<3>();

            auto index = w.row_col_to_index(left_row_index, right_col_index, cols);
            auto right_ptr = w.op_lea(b_ptr, index);
            auto [right_load_mem, right_value]  = w.op_load(body->mem_var(), right_ptr)->projs<2>();
            auto result = op(w, op_ty, rmode, elem_type, a, right_value);
            auto result_lea = w.op_lea(result_ptr, index);
            auto store_mem = w.op_store(right_load_mem, result_lea, result);

            builder.add(store_mem).app_body(body, body->ret_var());
            break;
        }
        case MOp::transpose: {
            entry->set_dbg(world().dbg("mop_transpose"));
            auto [result_ptr, result_rows, result_cols] = constructResult.result_matrix->projs<3>();

            auto b = entry->var(1);
            auto [b_ptr, b_rows, b_cols] = b->projs<3>();

            auto src_index = w.row_col_to_index(left_row_index, right_col_index, cols);
            auto dst_index = w.row_col_to_index(right_col_index, left_row_index, rows);

            auto src_ptr = w.op_lea(b_ptr, src_index);
            auto dst_lea = w.op_lea(result_ptr, dst_index);

            auto [right_load_mem, right_value]  = w.op_load(body->mem_var(), src_ptr)->projs<2>();
            auto store_mem = w.op_store(right_load_mem, dst_lea, right_value);

            builder.add(store_mem).app_body(body, body->ret_var());
            break;
        }
        case MOp::sum: {
            entry->set_dbg(world().dbg("mop_sum"));
            auto result = constructResult.result_matrix;
            auto b = entry->var(1);
            auto [b_ptr, b_rows, b_cols] = b->projs<3>();

            auto src_index = w.row_col_to_index(left_row_index, right_col_index, cols);
            auto src_ptr = w.op_lea(b_ptr, src_index);
            auto [right_load_mem, right_value]  = w.op_load(body->mem_var(), src_ptr)->projs<2>();

            auto sum = op(w, Op::add, rmode, elem_type, result, right_value);
            builder.add(right_load_mem).add(sum).app_body(body, body->ret_var());
            break;
        }
        case MOp::init: {
            entry->set_dbg(world().dbg("mop_init"));
            auto value = entry->var(1);
            auto mat = entry->var(2);
            auto [ptr, _rows, _cols] = mat->projs<3>();

            auto src_index = w.row_col_to_index(left_row_index, right_col_index, cols);
            auto lea = w.op_lea(ptr, src_index);
            auto store_mem  = w.op_store(body->mem_var(), lea, value);
            builder.add(store_mem).app_body(body, body->ret_var());
            break;
        }

        default: {}
    }
}

void LowerMatrix::construct_scalar_loop(Lam* entry, const Def* elem_type, const Def* rows, const Def* cols, ConstructResult& constructResult){
    auto [left_row_loop, left_row_yield] = world().repeat(rows, {elem_type});
    auto left_row_loop_result = builder.mem().add(elem_type).nom_filter_lam("left_row_loop_result");

    builder
            .mem(left_row_loop_result)
            .add(left_row_loop_result->var(1))
            .app_body(left_row_loop_result, entry->ret_var());

    builder
            .add(entry->mem_var())
            .add(zero(world(), elem_type))
            .add(left_row_loop_result)
            .app_body(entry, left_row_loop);

    auto [right_col2_loop, right_col2_yield] = world().repeat(cols, {elem_type});

    builder
            .mem(left_row_yield)
            .add(left_row_yield->var(2))
            .add(left_row_yield->ret_var())
            .app_body(left_row_yield, right_col2_loop);


    constructResult = {
            .result_matrix = right_col2_yield->var(2),
            .left_row_index = left_row_yield->var(1),
            .right_col_index = right_col2_yield->var(1),
            .body = right_col2_yield
    };
}

const Def* LowerMatrix::alloc_stencil(const Def* stencil, const Def* rows, const Def* cols,  const Def*& mem){
    if(auto tuple = stencil->isa<Sigma>()){
        return world().tuple(tuple->ops().map([&](auto elem, auto){ return alloc_stencil(elem, rows, cols, mem); }));
    }else if(auto mat = isa<Tag::Mat>(stencil)){
        auto elem_type = world().elem_ty_of_mat(mat);
        auto [new_mem, result_matrix] = world().op_create_matrix(elem_type, {rows, cols}, mem)->projs<2>();
        mem = new_mem;
        return result_matrix;
    }else{
        thorin::unreachable();
    }
}

void LowerMatrix::construct_mat_loop(Lam* entry, const Def* stencil, const Def* rows, const Def* cols, const Def* alloc_rows, const Def* alloc_cols, ConstructResult& constructResult){
    const Def* alloc_mem = entry->mem_var();
    auto result_matrix = alloc_stencil(stencil, alloc_rows, alloc_cols, alloc_mem);

    auto [left_row_loop, left_row_yield] = world().repeat(rows);
    auto left_row_loop_result = builder.mem().nom_filter_lam("left_row_loop_result");

    builder
            .mem(left_row_loop_result)
            .add(result_matrix)
            .app_body(left_row_loop_result, entry->ret_var());

    builder
            .add(alloc_mem)
            .add(left_row_loop_result)
            .app_body(entry, left_row_loop);

    auto [right_col2_loop, right_col2_yield] = world().repeat(cols);

    builder
            .mem(left_row_yield)
            .add(left_row_yield->ret_var())
            .app_body(left_row_yield, right_col2_loop);


    auto left_row_index = left_row_yield->var(1);
    auto right_col_index = right_col2_yield->var(1);

    constructResult = {
        .result_matrix = result_matrix,
        .left_row_index = left_row_index,
        .right_col_index = right_col_index,
        .body = right_col2_yield
    };
}

void LowerMatrix::construct_void_loop(Lam* entry, const Def* rows, const Def* cols, ConstructResult& constructResult){
    const Def* alloc_mem = entry->mem_var();

    auto [left_row_loop, left_row_yield] = world().repeat(rows);
    auto left_row_loop_result = builder.mem().nom_filter_lam("left_row_loop_result");

    builder
            .mem(left_row_loop_result)
            .app_body(left_row_loop_result, entry->ret_var());

    builder
            .add(alloc_mem)
            .add(left_row_loop_result)
            .app_body(entry, left_row_loop);

    auto [right_col2_loop, right_col2_yield] = world().repeat(cols);

    builder
            .mem(left_row_yield)
            .add(left_row_yield->ret_var())
            .app_body(left_row_yield, right_col2_loop);


    auto left_row_index = left_row_yield->var(1);
    auto right_col_index = right_col2_yield->var(1);

    constructResult = {
            .result_matrix = nullptr,
            .left_row_index = left_row_index,
            .right_col_index = right_col_index,
            .body = right_col2_yield
    };
}

const Lam* LowerMatrix::create_MOp_lam(const Axiom* mop_axiom, const Def* elem_type, const Def* rmode){
    World& w = world();
    auto signature = w.tuple({mop_axiom, elem_type, rmode});

    if(mop_variants.contains(signature)){
        return mop_variants[signature];
    }

    MOp mop = MOp(mop_axiom->flags());

    ConstructResult constructResult{};
    Lam* entry;
    const Def *rows, *cols;
    if(is_scalar(mop)){
        entry = builder.mem()
                .add(elem_type)
                .type_matrix(2, elem_type)
                .add(
                        builder.mem().type_matrix(2, elem_type).cn()
                ).nom_filter_lam("matrix_scalar_entry");

        auto b = entry->var(2);
        auto [b_ptr, b_rows, b_cols] = b->projs<3>();

        rows = b_rows;
        cols = b_cols;

        construct_mat_loop(entry, world().type_mat(2, elem_type), rows, cols, rows, cols, constructResult);
    }else if(is_unary(mop)){
        auto ret_pi_buil = builder.mem();

        if(mop == MOp::transpose){
            ret_pi_buil.type_matrix(2, elem_type);
        }else{
            ret_pi_buil.add(elem_type);
        }

        entry = builder.mem()
                .type_matrix(2, elem_type)
                .add(
                    ret_pi_buil.cn()
                ).nom_filter_lam("matrix_unary_entry");

        auto b = entry->var(1);
        auto [b_ptr, b_rows, b_cols] = b->projs<3>();

        rows = b_rows;
        cols = b_cols;

        if(mop == MOp::transpose){
            construct_mat_loop(entry, world().type_mat(2, elem_type), rows, cols, cols, rows, constructResult);
        }else if(mop == MOp::sum){
            construct_scalar_loop(entry, elem_type, rows, cols, constructResult);
        }else{
            thorin::unreachable();
        }
    }else if(mop == MOp::init){

        entry = builder.mem()
                .add(elem_type)
                .type_matrix(2, elem_type)
                .add(
                    builder.mem().cn()
                ).nom_filter_lam("matrix_unary_entry");

        auto b = entry->var(2);
        auto [b_ptr, b_rows, b_cols] = b->projs<3>();

        rows = b_rows;
        cols = b_cols;

        construct_void_loop(entry, rows, cols, constructResult);
    }else{
        entry = builder
                .mem()
                .type_matrix(2, elem_type)
                .type_matrix(2, elem_type)
                .add(
                    builder.mem().type_matrix(2, elem_type).cn()
                ).nom_filter_lam("matrix_dual_entry");

        auto a = entry->var(1);
        auto b = entry->var(2);

        auto [a_ptr, a_rows, a_cols] = a->projs<3>();
        auto [b_ptr, b_rows, b_cols] = b->projs<3>();

        rows = a_rows;
        cols = b_cols;

        construct_mat_loop(entry, world().type_mat(2, elem_type), rows, cols, rows, cols, constructResult);
    }

    construct_mop(entry, mop, rmode, elem_type, rows, cols, constructResult);

    mop_variants[signature] = entry;
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

    auto res_buil = builder.mem();
    if(MOp(mop_axiom->flags()) == MOp::sum){
        res_buil.add(elem_type);
    }else if(MOp(mop_axiom->flags()) != MOp::init){
        res_buil.type_matrix(2, elem_type);
    }

    auto result_lam = res_buil.nom_filter_lam("mop_result");
    builder.flatten(arg_wrap).add(result_lam).app_body(helper.tail, mop_lam);
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
        auto result_ptr = mat->proj(0);
        auto dst_lea = world().op_lea(result_ptr, index);
        mem = world().op_store(mem, dst_lea, value);
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
        auto elem_type = world().elem_ty_of_mat(mat->type());

        entry = builder.mem()
                .type_matrix(2, elem_type)
                .add(
                        builder.mem().add(out_type).cn()
                ).nom_filter_lam("matrix_unary_entry");

        auto b = entry->var(1);
        auto [b_ptr, b_rows, b_cols] = b->projs<3>();

        auto rows = b_rows;
        auto cols = b_cols;

        ConstructResult constructResult{};
        construct_mat_loop(entry, out_type, rows, cols, rows, cols, constructResult);

        World &w = world();

        auto left_row_index = constructResult.left_row_index;
        auto right_col_index = constructResult.right_col_index;
        auto body = constructResult.body;
        auto result_matrix = constructResult.result_matrix;


        auto src_index = w.row_col_to_index(left_row_index, right_col_index, cols);
        auto src_ptr = w.op_lea(b_ptr, src_index);
        auto [right_load_mem, right_value]  = w.op_load(body->mem_var(), src_ptr)->projs<2>();

        auto f_type = map_f->type()->as<Pi>();
        auto map_result_lam = world().nom_filter_lam(f_type->doms().back()->as<Pi>(), world().lit_bool(false), world().dbg(""));

        auto store_mem = map_result_lam->mem_var();
        store_rec(world().tuple(map_result_lam->vars().skip_front()), result_matrix, src_index, store_mem);


        map_result_lam->set_body(world().app(body->ret_var(), store_mem));
        builder.add(right_load_mem).add(right_value).add(map_result_lam).app_body(body, map_f);

        mop_variants[signature] = entry;
    }

    auto result_lam = builder.mem()
            .add(out_type)
            .nom_filter_lam("mat_mul_res");
    builder.add(mem).add(mat).add(result_lam).app_body(helper.tail, entry);
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

    helper = {
        .currentMem = result_lam->mem_var(),
        .tail = result_lam,
    };

    return result_lam->var();
}

const Def* LowerMatrix::rewrite_rec_convert(const Def* current){
    if(auto lam = current->isa_nom<Lam>(); isa_workable(lam)){
        Helper oldHelper = helper;

        helper = {
            .currentMem = lam->mem_var(),
            .tail = lam,
        };

        auto result = rewrite_rec(lam->body());

        helper.tail->set_body(result);
        helper = oldHelper;
        return lam;
    }else if (isa<Tag::MOp>(current) || isa<Tag::Map>(current)) {
        return rewrite_app(current->as<App>());
    }else if(auto mat = isa<Tag::Mat>(current->type())){
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
    helper = {
        .currentMem = curr_nom()->mem_var(),
        .tail = curr_nom(),
    };

    auto result = rewrite_rec(curr_nom()->body());
    helper.tail->set_body(result);
}


}
