#ifndef THORIN_PASS_RW_AUTO_DIFF_H
#define THORIN_PASS_RW_AUTO_DIFF_H

#include "thorin/pass/pass.h"

namespace thorin {

/// # Automatic Differentiation in Thorin2
///
/// This pass implements automatic differentiation for the following dialects:
/// * core
/// * arithmetic
/// * memory, pointer
/// * tensor
///
/// The automatic differentiation is based on the ideas of Brunel et al, 2020:
/// Backpropagation in the Simply Typed Lambda-Calculus with Linear Negation
///
/// Each expression is augmented with a pullback function that constructs the derivates of the expression.
/// A function f: A -> B is translated by invokation of the rev_diff axiom to a function f': A -> B * (B -> A)
/// the second part of the result is the pullback function that expects the output tangent and returns the input
/// tangent.
///
/// Example call:
/// ```
/// fn f(a: f32) -> f32 {
///     (a * a)
/// }
/// fn main() -> i32 {
///     let Df: fn(f32) -> (f32, fn(f32) -> f32) = rev_diff(f);
///     let Gf: (f32, fn(f32) -> f32) = (Df(4f));
///     let y: f32 = (Gf(0));             // result
///     let pb: fn(f32) -> f32 = (Gf(1)); // pullback
///     let g: f32 = pb(1f);              // gradient
/// }
/// ```
/// f    = λ a: f32. (a * a)
/// f'   = λ a: f32. (f a, f* a s)
/// f* a = λ s: f32. (2 * a * s)
/// In the example above the names are:
/// Df for f', pb for f*
///
///
/// Features:
/// * modular construction
/// * simple transformation
/// * generation of efficient code
/// * pure functions except for memory operations in the orignal code
/// * support for higher order functions
/// * reverse mode
/// * handles control flow (conditions, recursion)
///
/// [old draft of technical details](https://www.overleaf.com/read/gdpfxvzqpfjf)
///
/// difference to the current main branch of AD:
/// for multiple arguments, AD currently expects always flat elements instead of arrays
/// (`[mem, r64, r64, cn[...]]` instead of `cn[mem, <2;r64>, cn [...]]`)
/// the changes in code are:
///
/// ```
///    if (ops[0]->isa<Pi>() && std::all_of(ops.begin() + 1, ops.end(), [&](auto op) { return ops[0] == op; })) return
///    arr(n, ops[0]);
/// ```
/// in `const Def* World::sigma(Defs ops, const Def* dbg)`
///
/// and remove
/// ```
/// if (std::all_of(ops.begin() + 1, ops.end(), [&](auto op) { return ops[0] == op; })) return pack(n, ops[0]);
/// ```
/// in `const Def* World::tuple(const Def* type, Defs ops, const Def* dbg)`
///
/// as well as
/// ```
/// if(auto arr = type->isa<Arr>()){
///    type=arr->body();
/// }else {
///    type=type->op(0);
/// }
/// ```
/// instead of
/// ```
/// type = type->as<Arr>()->body();
/// ```
class AutoDiff : public RWPass<> {
public:
    AutoDiff(PassMan& man)
        : RWPass(man, "auto_diff") {}
    const Def* rewrite(const Def*) override;
};

/// auxiliary data structure to manage the translation of one unit (one function)
class AutoDiffer {
public:
    AutoDiffer(World& world, const Def2Def& src_to_dst, const Def* A_)
        : world_{world}
        , src_to_dst_{src_to_dst}
        , A_src{A_}
        , A{world.tangent_type(A_, false)} {
        // TODO: is this comment up to date?
        // initializes the differentiation for a function of type A -> B
        // src_to_dst expects the parameters of the source lambda to be mapped
        //  (this property is only used later on)

        // the general principle is that every expression is a function
        //   and has a gradient in respect from its outputs to its inputs
        //   for instance add:R²->R has a pullback R->R²
        //   describing how the result depends on the two inputs
        //      (the derivation of the output w.r. to the inputs)
        //   we mostly directly combine building techniques and chain rule applications
        //   into the basic construction to derive the wanted derivative
        //   w.r. to the function inputs of type A for the rev_diff call we currently are working on
        //   in that sense every expression can be seen as a function from function input to some
        //   intermediate result
        //   Therefore, we need to keep track of A (but B is mostly not important)

        // combination of derivatives is in most parts simply multiplication and application
        // the pullbacks handle this for us as the scalar is applied inside the derivative
        // and scales the derivative
        // Therefore, composition of two pullbacks corresponds to (matrix-)multiplication
        // and represents an application of the chain rule
        // the nested nature emulates the backward adjoint trace used in backpropagation
        // also see "Demystifying Differentiable Programming: Shift/Reset the Penultimate Backpropagator"
        // for a similar approach but with shift and reset primitives
    }

    const Def* reverse_diff(Lam* src); // top level function to compute the reverse differentiation of a function
private:
    const Def* j_wrap(const Def* def); // 'identity' (except for lambdas, functions, and applications) traversal
    // annotating the pullbacks
    const Def* j_wrap_convert(const Def* def);
    const Def*
    j_wrap_rop(ROp op,
               const Def* a,
               const Def* b); // pullback computation for predefined functions, specifically operations like +, -, *, /
    void derive_external(const Lam* fun, Lam* pb, Lam* fw, Lam* res_lam);
    void
    derive_numeric(const Lam* fun, Lam* source, const Def* target, Lam* fw, const Def* fx, const Def* s, r32 delta);

    const Def* zero_pb(const Def* type, const Def* dbg);
    const Def* j_wrap_tuple(DefArray tuple);

    const Def* seen(const Def* src); // lookup in the map

    // chains cn[:mem, A, cn[:mem, B]] and cn[:mem, B, cn[:mem, C]] to a toplevel cn[:mem, A, cn[:mem, C]]
    const Def* chain(const Def* a, const Def* b);
    const Pi* createPbType(const Def* A, const Def* B);
    const Def* extract_pb(const Def* j_extract, const Def* tuple);

    World& world_;
    Def2Def src_to_dst_;           // mapping old def to new def
    DefMap<const Def*> pullbacks_; // <- maps a *copied* src term (a dst term) to its pullback function
    DefMap<const Def*> pointer_map;
    DefMap<const Def*> structure_map;
    const Def *A, *A_src, *zero_grad; // input type

    void initArg(const Def* dst);
    const Def* ptrSlot(const Def* ty, const Def* mem);
    std::pair<const Def*, const Def*>
    reloadPtrPb(const Def* mem, const Def* ptr, const Def* dbg = {}, bool generateLoadPb = false);

    // next mem object to use / most recent memory object
    // no problem as control flow is handled by cps
    // alternative: j_wrap returns mem object
    // only set at memory alternating operations
    //   load, store, slot, alloc, function arg
    const Def* current_mem;
};

} // namespace thorin

#endif
