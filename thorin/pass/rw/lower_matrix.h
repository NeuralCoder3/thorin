#ifndef THORIN_PASS_LOWER_MATRIX_H
#define THORIN_PASS_LOWER_MATRIX_H

#include "thorin/pass/pass.h"

namespace thorin {



class LoopBuilderResult{
public:
    DefArray indices;
    DefArray vars;
    DefArray reductions;

    Lam* entry;
    Lam* finish;
    Lam* body;
};

class LoopBuilder{
    World& world;
    DefVec ranges;

    DefVec indices;
    DefArray vars;
    Lam* body;
    Lam* entry;
    Lam* finish;
    DefVec var_ty;
    DefVec var_init;
    bool body_is_yield = false;

#define b world.builder()
public:
    LoopBuilder(World& w) : world(w){
    }

    void addLoop( const Def* count ){
        ranges.push_back(count);
    }

    void addVar(const Def* ty, const Def* init){
        var_ty.push_back(ty);
        var_init.push_back(init);
    }

    LoopBuilderResult build(){
        finish = b.mem().add(var_ty).nom_filter_lam("loop_result");
        entry = b.mem().nom_filter_lam("loop_entry1");
        body = b.mem().add(var_ty).add(b.mem().add(var_ty).cn()).nom_filter_lam("loop_body");

        b.mem(entry).add(var_init).add(finish).app_body(entry, body);
        b.add(body->vars().skip_back(1)).app_body(body, body->ret_var());
        body_is_yield = false;

        for(auto range : ranges){
            auto [loop, yield] = world.repeat(range, var_ty);

            b
                .mem(body)
                .add(body->vars().skip_front(1 + (body_is_yield ? 1 : 0)))
                .app_body(body, loop);

            body_is_yield = true;
            body = yield;
            indices.push_back(yield->var(1));
            vars = yield->vars().skip(2, 1);
        }

        return {
            .indices = indices,
            .vars = vars,
            .reductions = finish->vars().skip_front(1),
            .entry = entry,
            .finish = finish,
            .body = body
        };
    }
};

class MatrixHelper{
    World& world;
    const Def* mat = nullptr;
    bool transpose = false;

public:
    MatrixHelper(World& w) : world(w){

    }

    MatrixHelper(const Def* mat) : world(mat->world()), mat(mat){

    }

    void setMat(const Def* mat){
        this->mat = mat;
    }

    const Def* getMat(){
        return mat;
    }

    void setTranspose(bool transpose){
        this->transpose = transpose;
    }

    const Def* getRows(){
        return mat->proj(2 + (transpose ? 1 : 0));
    }

    const Def* getCols(){
        return mat->proj(2 + (transpose ? 0 : 1));
    }

    const Def* getPointer(){
        return mat->proj(1);
    }

    const Def* getMeta(){
        return mat->proj(0);
    }

    const Def* bitCheck(u32 bit){
        auto meta = getMeta();
        auto mask = world.lit_int_width(32, 1 << bit);
        auto masked = world.op(Bit::_and, meta, mask);
        return world.op(ICmp::e, masked, mask);
    }

    const Def* isZero(){
        return bitCheck(0);
    }

    const Def* isOne(){
        return bitCheck(1);
    }

    const Def* isConst(){
        return bitCheck(2);
    }

    const Def* isTranspose(){
        return bitCheck(3);
    }

    DefArray dims(){
        auto shape_size = as_lit(world.elem_ty_of_tn(mat->type()));

        DefArray result{2};

        for(size_t i = 0 ; i < shape_size ; i++){
            result[i] = world.extract(mat, i + 2);
        }

        return result;
    }

    const Def* getIndex(const DefArray& indices){
        if(indices.size() == 2){
            auto cols = getCols();
            return world.row_col_to_index(indices[0], indices[1], cols);
        }else if(indices.size() == 1){
            return indices[0];
        }else{
            thorin::unreachable();
        }
    }

    const Def* getIndex(const Def* row, const Def* col){
        return getIndex({row, col});
    }

    const Def* getLea(DefArray& indices){
        auto index = getIndex(indices);
        return lea(index);
    }

    const Def* getLea(const Def* row, const Def* col){
        auto index = getIndex(row, col);
        return lea(index);
    }

    const Def* lea(const Def* index){
        auto ptr = getPointer();
        return world.op_lea(ptr, index);
    }

    const Def* load(const Def* mem, const Def* index){
        const Def* ptr = lea(index);
        return world.op_load(mem, ptr);
    }

    const Def* store(const Def* mem, const Def* index, const Def* value){
        const Def* ptr = lea(index);
        return world.op_store(mem, ptr, value);
    }

    const Def* load_const_value(const Def* mem){
        return load(mem, world.lit_int_width(64, 0));
    }
};

class InputHelper{
public:
    Lam* lam = nullptr;
    const Def* arg = nullptr;
    MatrixHelper left;
    MatrixHelper right;

    InputHelper(World& w) : left(w), right(w){

    }
};

class ConstructHelper{
public:
    /*MatrixHelper leftEntry;
    MatrixHelper rightEntry;
    MatrixHelper left;
    MatrixHelper right;
    MatrixHelper result;*/


    InputHelper impl;
    MatrixHelper result;

    const Def* raw_result = nullptr;
    const Def* rows = nullptr;
    const Def* cols = nullptr;
    DefArray indices;
    DefArray vars;
    Lam* body = nullptr;
    //Lam* impl_entry = nullptr;

    ConstructHelper(World& w) : impl(w), result(w){

    }
};

struct ChainHelper{
    const Def* currentMem = nullptr;
    Lam* tail = nullptr;
};

class LowerMatrix : public RWPass<Lam> {
public:
    LowerMatrix(PassMan& man)
        : RWPass(man, "lower_matrix") {}

    void enter() override;
    const Def* rewrite_rec(const Def* current, bool convert = true);
    const Def* rewrite_rec_convert(const Def* current);
    //const Lam* const_reduction(MOp mop, ROp rop,ConstructHelper& helper);
    const Lam* create_MOp_lam(const Axiom* mop_axiom, const Def* elem_type, const Def* rmode);
    const Lam* create_MOp_impl(const Axiom* mop_axiom, const Def* elem_type, const Def* rmode);
    void construct_mat_loop(const Def* elem_type, const Def* a_rows, const Def* b_cols, const Def* alloc_rows, const Def* alloc_cols, bool flatten, ConstructHelper& constructResult);
    void construct_scalar_loop(const Def* elem_type, const Def* a_rows, const Def* b_cols, ConstructHelper& constructResult);
    void construct_void_loop(const Def* rows, const Def* cols, ConstructHelper& constructResult);
    void construct_mop( MOp mop, const Def* elem_type, ConstructHelper& constructResult);
    Lam* rewrite_mop(const App* app, const Def* arg_wrap);
    Lam* rewrite_map(const App* app, const Def* arg_wrap);
    const Def* alloc_stencil(const Def* stencil, const Def* rows, const Def* cols,  const Def*& mem);
    const Def* rewrite_app(const App* app);
    void store_rec(const Def* value, const Def* mat, const Def* index, const Def*& mem);

    const Pi* mop_pi(MOp mop, const Def* elem_type);
    Lam* mop_lam(MOp mop, const Def* elem_type, const std::string& name);

    ChainHelper chainHelper;

    Def2Def old2new;
    DefMap<Lam*> mop_variants;
};

    //Def2Def old2new_;
}

#endif
