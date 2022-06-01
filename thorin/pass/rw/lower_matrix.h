#ifndef THORIN_PASS_LOWER_MATRIX_H
#define THORIN_PASS_LOWER_MATRIX_H

#include "thorin/pass/pass.h"

namespace thorin {

class MatrixHelper{
    World& world;
    const Def* mat = nullptr;
    bool transpose = false;

public:
    MatrixHelper(World& w) : world(w){

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
        return mat->proj(1 + (transpose ? 1 : 0));
    }

    const Def* getCols(){
        return mat->proj(1 + (transpose ? 0 : 1));
    }

    const Def* getPointer(){
        return mat->proj(0);
    }

    const Def* getIndex(const Def* row, const Def* col){
        auto rows = getRows();
        auto cols = getCols();
        if(transpose){
            return world.row_col_to_index(col, row, rows);
        }else{
            return world.row_col_to_index(row, col, cols);
        }
    }

    const Def* getLea(const Def* row, const Def* col){
        auto index = getIndex(row, col);
        auto ptr = getPointer();
        return world.op_lea(ptr, index);
    }
};

class ConstructHelper{
public:
    MatrixHelper left;
    MatrixHelper right;
    MatrixHelper result;

    const Def* scalar_result = nullptr;
    const Def* rows = nullptr;
    const Def* cols = nullptr;
    //const Def* result_matrix;
    const Def* left_row_index = nullptr;
    const Def* right_col_index = nullptr;
    Lam* body = nullptr;

    ConstructHelper(World& w) : left(w), right(w), result(w){

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
    const Lam* create_MOp_lam(const Axiom* mop_axiom, const Def* elem_type, const Def* rmode);
    void construct_mat_loop(Lam* entry, const Def* elem_type, const Def* a_rows, const Def* b_cols, const Def* alloc_rows, const Def* alloc_cols, ConstructHelper& constructResult);
    void construct_scalar_loop(Lam* entry, const Def* elem_type, const Def* a_rows, const Def* b_cols, ConstructHelper& constructResult);
    void construct_void_loop(Lam* entry, const Def* rows, const Def* cols, ConstructHelper& constructResult);
    void construct_mop(Lam* entry, MOp mop, const Def* elem_type, const Def* rows, const Def* cols, ConstructHelper& constructResult);
    Lam* rewrite_mop(const App* app, const Def* arg_wrap);
    Lam* rewrite_map(const App* app, const Def* arg_wrap);
    const Def* alloc_stencil(const Def* stencil, const Def* rows, const Def* cols,  const Def*& mem);
    const Def* rewrite_app(const App* app);
    void store_rec(const Def* value, const Def* mat, const Def* index, const Def*& mem);

    ChainHelper chainHelper;

    Def2Def old2new;
    DefMap<Lam*> mop_variants;
};

    //Def2Def old2new_;
}

#endif
