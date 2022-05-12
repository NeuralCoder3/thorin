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

class LowerMatrix : public RWPass<Lam> {
public:
    LowerMatrix(PassMan& man)
        : RWPass(man, "lower_matrix") {}

    void enter() override;
    const Def* rewrite_rec(const Def* current);
    const Def* rewrite_rec_convert(const Def* current);
    const Lam* create_MOp_lam(const Axiom* mop_axiom, const Def* elem_type, const Def* rmode);
    void construct_loop(Lam* entry, const Def* elem_type, const Def* a_rows, const Def* b_cols, ConstructResult& constructResult);
    void construct_mop(Lam* entry, const Def* elem_type, MOp mop, const Def* rmode, const Def* cols, ConstructResult& constructResult);

    const Def* currentMem;
    Lam* head = nullptr;
    Lam* tail = nullptr;
    Lam* exit = nullptr;

    Def2Def old2new;
    DefMap<const Lam*> mop_variants;
};

    //Def2Def old2new_;
}

#endif
