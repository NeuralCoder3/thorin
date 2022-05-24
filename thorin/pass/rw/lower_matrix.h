#ifndef THORIN_PASS_LOWER_MATRIX_H
#define THORIN_PASS_LOWER_MATRIX_H

#include "thorin/pass/pass.h"

namespace thorin {

struct ConstructResult{
    const Def* rows;
    const Def* cols;
    const Def* result_matrix;
    const Def* left_row_index;
    const Def* right_col_index;
    Lam* body;
};

struct Helper{
    const Def* currentMem;
    //Lam* head = nullptr;
    Lam* tail = nullptr;
    //Lam* exit = nullptr;
};

class LowerMatrix : public RWPass<Lam> {
public:
    LowerMatrix(PassMan& man)
        : RWPass(man, "lower_matrix") {}

    void enter() override;
    const Def* rewrite_rec(const Def* current, bool convert = true);
    const Def* rewrite_rec_convert(const Def* current);
    const Lam* create_MOp_lam(const Axiom* mop_axiom, const Def* elem_type, const Def* rmode);
    void construct_mat_loop(Lam* entry, const Def* elem_type, const Def* a_rows, const Def* b_cols, const Def* alloc_rows, const Def* alloc_cols, ConstructResult& constructResult);
    void construct_scalar_loop(Lam* entry, const Def* elem_type, const Def* a_rows, const Def* b_cols, ConstructResult& constructResult);
    void construct_void_loop(Lam* entry, const Def* rows, const Def* cols, ConstructResult& constructResult);
    void construct_mop(Lam* entry, MOp mop, const Def* rmode, const Def* elem_type, const Def* cols, ConstructResult& constructResult);
    Lam* rewrite_mop(const App* app, const Def* arg_wrap);
    Lam* rewrite_map(const App* app, const Def* arg_wrap);
    const Def* alloc_stencil(const Def* stencil, const Def* rows, const Def* cols,  const Def*& mem);
    const Def* rewrite_app(const App* app);
    void store_rec(const Def* value, const Def* mat, const Def* index, const Def*& mem);

    Helper helper;

    Def2Def old2new;
    DefMap<Lam*> mop_variants;
};

    //Def2Def old2new_;
}

#endif
