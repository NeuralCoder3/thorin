#ifndef THORIN_AUTO_DIFF_UTIL_H
#define THORIN_AUTO_DIFF_UTIL_H

#include "thorin/pass/pass.h"

namespace thorin {

/// @name eviction - functions that are general enough to be replaced or moved out to other places
///@{
size_t getDim(const Def* def);
const Pi* isReturning(const Pi* pi);
DefArray flat_tuple(const DefArray& defs, bool preserveFatPtr = false);
DefArray vars_without_mem_cont(Lam* lam);
const Lam* repeatLam(World& world, const Def* count, const Lam* body);
std::pair<const Lam*, Lam*> repeatLam(World& world, const Def* count);
const Def* copy(World& world, const Def* inputArr, const Def* outputArr, const Def* size);
///@}
// end eviction


/// @name utility - functions that are adjacent to autodiff but not necessarily interlinked
///@{
bool isFatPtrType(World& world_, const Def* type);
const Lam* vec_add(World& world, const Def* a, const Def* b, const Def* cont);
std::pair<const Def*, const Def*>
lit_of_type(World& world, const Def* mem, const Def* type, const Def* like, r64 lit, const Def* dummy);
std::pair<const Def*, const Def*> ONE(World& world, const Def* mem, const Def* def, const Def* like, const Def* dummy) {
    return lit_of_type(world, mem, def, like, 1, dummy);
}
std::pair<const Def*, const Def*>
ZERO(World& world, const Def* mem, const Def* def, const Def* like, const Def* dummy) {
    return lit_of_type(world, mem, def, like, 0, dummy);
}
std::pair<const Def*, const Def*> ZERO(World& world, const Def* mem, const Def* def, const Def* like) {
    return ZERO(world, mem, def, like, nullptr);
}
std::pair<const Def*, const Def*> ONE(World& world, const Def* mem, const Def* def, const Def* like) {
    return ONE(world, mem, def, like, nullptr);
}
std::pair<const Def*, const Def*> ZERO(World& world, const Def* mem, const Def* def) {
    return ZERO(world, mem, def, nullptr);
}
std::pair<const Def*, const Def*> ONE(World& world, const Def* mem, const Def* def) {
    return ONE(world, mem, def, nullptr);
}
std::pair<const Def*, const Def*>
oneHot(World& world_, const Def* mem, u64 idx, const Def* shape, const Def* like, const Def* s);
std::pair<const Def*, const Def*>
oneHot(World& world_, const Def* mem, const Def* idx, const Def* shape, const Def* like, const Def* s);
const Lam* lam_fat_ptr_wrap(World& world, const Lam* lam);
///@}
// end utility

} // namespace thorin

#endif // THORIN_AUTO_DIFF_UTIL_H
