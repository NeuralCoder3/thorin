#ifndef THORIN_PASS_RW_MATRIX2TUPLE_H
#define THORIN_PASS_RW_MATRIX2TUPLE_H

#include "thorin/pass/pass.h"

namespace thorin {

class Matrix2Tuple : public RWPass<Lam> {
public:
    Matrix2Tuple(PassMan& man)
        : RWPass(man, "matrix2tuple") {}

    const Def* rewrite(const Def*) override;
    Def2Def old2new_;
};
}

#endif
