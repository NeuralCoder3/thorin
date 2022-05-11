#ifndef THORIN_PASS_LOWER_MATRIX_H
#define THORIN_PASS_LOWER_MATRIX_H

#include "thorin/pass/pass.h"

namespace thorin {

class LowerMatrix : public RWPass<Lam> {
public:
    LowerMatrix(PassMan& man)
        : RWPass(man, "lower_matrix") {}

    void enter() override;
    const Def* rewrite_rec(const Def* current);
    const Def* rewrite_rec_convert(const Def* current);
    const Lam* create_MOp_lam(u32 flags);

    const Def* currentMem;
    Lam* head = nullptr;
    Lam* tail = nullptr;
    Lam* exit = nullptr;

    Def2Def old2new;
};

    //Def2Def old2new_;
}

#endif
