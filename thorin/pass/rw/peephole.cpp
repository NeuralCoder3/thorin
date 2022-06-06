#include "thorin/pass/rw/peephole.h"

#define dlog(world,...) world.DLOG(__VA_ARGS__)
#define type_dump(world,name,d) world.DLOG("{} {} : {}",name,d,d->type())

namespace thorin {

const Def* Peephole::rewrite(const Def* def) {

    if (auto mop = isa<Tag::MOp>(def)) {
        switch (MOp(mop.flags())) {
            case MOp::mul:
                break;
        }

    }
    return def;
}

}
