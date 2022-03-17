#include "thorin/pass/optimize.h"

#include "thorin/pass/fp/beta_red.h"
#include "thorin/pass/fp/copy_prop.h"
#include "thorin/pass/fp/eta_exp.h"
#include "thorin/pass/fp/eta_red.h"
#include "thorin/pass/fp/ssa_constr.h"
#include "thorin/pass/rw/auto_diff.h"
#include "thorin/pass/fp/tail_rec_elim.h"
#include "thorin/pass/rw/alloc2malloc.h"
#include "thorin/pass/fp/unbox_closures.h"
#include "thorin/pass/fp/closure_analysis.h"
#include "thorin/pass/rw/bound_elim.h"
#include "thorin/pass/rw/partial_eval.h"
#include "thorin/pass/rw/remem_elim.h"
#include "thorin/pass/rw/ret_wrap.h"
#include "thorin/pass/rw/scalarize.h"
#include "thorin/pass/rw/peephole.h"
#include "thorin/pass/rw/cconv_prepare.h"
#include "thorin/pass/rw/closure2sjlj.h"

// old stuff
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/closure_conv.h"
#include "thorin/transform/lower_typed_closures.h"

#include "thorin/error.h"

namespace thorin {

static void closure_conv(World& world) {
    PassMan prepare(world);
    auto ee = prepare.add<EtaExp>(nullptr);
    prepare.add<CConvPrepare>(ee);
    prepare.run();

    ClosureConv(world).run();

    PassMan cleanup(world);
    auto er = cleanup.add<EtaRed>(true); // We only want to eta-reduce things in callee position away at this point!
    ee = cleanup.add<EtaExp>(er);
    cleanup.add<Scalerize>(ee);
    cleanup.run();
}

static void lower_closures(World& world) {
    PassMan closure_destruct(world);
    closure_destruct.add<Scalerize>(nullptr);
    closure_destruct.add<UnboxClosure>();
    closure_destruct.add<CopyProp>(nullptr, nullptr, true);
    closure_destruct.add<ClosureAnalysis>();
    closure_destruct.add<Closure2SjLj>();
    closure_destruct.run();

    LowerTypedClosures(world).run();
}

void optimize(World& world) {

    world.set(LogLevel::Debug);
    // world.set(std::make_unique<ErrorHandler>());
//    std::unique_ptr<ErrorHandler> err;
//    ErrorHandler* err;
//    world.set((std::unique_ptr<ErrorHandler>&&) nullptr);

    PassMan optA(world);
    optA.add<AutoDiff>();

//     PassMan optZ(world);
//     optZ.add<ZipEval>();
//     optZ.run();
//     printf("Finished OptiZip\n");

//     return;


//     PassMan opt2(world);
//     auto br = opt2.add<BetaRed>();
//     auto er = opt2.add<EtaRed>();
//     auto ee = opt2.add<EtaExp>(er);
//     opt2.add<SSAConstr>(ee);
//     opt2.add<Scalerize>(ee);
//     // opt2.add<DCE>(br, ee);
//     opt2.add<CopyProp>(br, ee);
//     opt2.add<TailRecElim>(er);
// //    opt2.run();
    printf("Finished Prepare Opti\n");

    PassMan optCCB(world);
    // opt.add<PartialEval>();
    // auto br = opt.add<BetaRed>();
    auto erCCB = optCCB.add<EtaRed>();
    auto eeCCB = optCCB.add<EtaExp>(erCCB);
    // opt.add<SSAConstr>(ee);
    // opt.add<Scalerize>(ee);
    // opt.add<CopyProp>(br, ee);
    // opt.add<TailRecElim>(er);
    optCCB.run();
    printf("Finished Closur Prepare Opti\n");

    closure_conv(world);
    lower_closures(world);
    printf("Finished Closure Opti\n");

    optA.run();
    printf("Finished AutoDiff Opti\n");

    PassMan opt(world);
    opt.add<PartialEval>();
    auto br = opt.add<BetaRed>();
    auto er = opt.add<EtaRed>();
    auto ee = opt.add<EtaExp>(er);
    opt.add<SSAConstr>(ee);
    opt.add<Scalerize>(ee);
    opt.add<CopyProp>(br, ee);
    opt.add<TailRecElim>(er);
    opt.run();

    // PassMan opt3(world);
    // opt3.add<PartialEval>();
    // auto br3 = opt3.add<BetaRed>();
    // auto er3 = opt3.add<EtaRed>();
    // auto ee3 = opt3.add<EtaExp>(er);
    // opt3.add<SSAConstr>(ee3);
    // opt3.add<Scalerize>(ee3);
    // // opt3.add<DCE>(br3, ee3);
    // opt3.add<CopyProp>(br3, ee3);
    // opt3.add<TailRecElim>(er3);
    // opt3.run();
    printf("Finished Simpl Opti\n");


    PassMan optB(world);
    optB.add<Peephole>();
    optB.run();
    printf("Finished Peephole Opti\n");


        cleanup_world(world);
//     partial_evaluation(world, true);
    while (partial_evaluation(world, true)) {} // lower2cff
        cleanup_world(world);

    printf("Finished Cleanup\n");

    PassMan codgen_prepare(world);
    // codgen_prepare.add<BoundElim>();
    codgen_prepare.add<RememElim>();
    codgen_prepare.add<Alloc2Malloc>();
    codgen_prepare.add<RetWrap>();
    codgen_prepare.run();
}

} // namespace thorin
