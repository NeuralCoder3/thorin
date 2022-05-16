#ifndef THORIN_PASS_RW_MATRIX2TUPLE_H
#define THORIN_PASS_RW_MATRIX2TUPLE_H

#include "thorin/pass/pass.h"

namespace thorin {

class Matrix2Tuple : public RWPass<Lam> {
public:
    Matrix2Tuple(PassMan& man)
        : RWPass(man, "matrix2tuple") {}

    void enter() override;
    const Def* rewrite_cached(const Def* def);
    const Def* rewrite_type_cached(const Def* def);
    const Def* rewrite_type(const Def* def);
    const Def* rewrite_convert(const Def* def);
    Def2Def old2new_;
    const Def* currentMem;
    bool found = false;
};
}

#endif
