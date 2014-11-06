#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class CFG;

//------------------------------------------------------------------------------

class Scope {
public:
    Scope(const Scope&) = delete;
    Scope& operator= (Scope) = delete;

    explicit Scope(Lambda* entry);
    ~Scope();

    /// All lambdas within this scope in reverse post-order.
    ArrayRef<Lambda*> rpo() const { return rpo_; }
    Lambda* rpo(size_t i) const { return rpo_[i]; }
    Lambda* entry() const { return rpo().front(); }
    /// Like \p rpo() but without \p entry()
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(1); }
    const DefSet& in_scope() const { return in_scope_; }
    bool _contains(Def def) const { return in_scope_.contains(def); }
    bool contains(Lambda* lambda) const { return lambda->find_scope(this) != nullptr; }
    bool contains(const Param* param) const { return param->lambda()->find_scope(this) != nullptr; }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return preds_[sid(lambda)]; }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return succs_[sid(lambda)]; }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    size_t sid(Lambda* lambda) const { return lambda->find_scope(this)->sid; }
    uint32_t id() const { return id_; }
    size_t size() const { return rpo_.size(); }
    World& world() const { return world_; }
    void dump() const;
    const CFG* cfg() const;

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

    typedef ArrayRef<Lambda*>::const_reverse_iterator const_reverse_iterator;
    const_reverse_iterator rbegin() const { return rpo().rbegin(); }
    const_reverse_iterator rend() const { return rpo().rend(); }

    template<bool elide_empty = true>
    static void for_each(const World&, std::function<void(const Scope&)>);

private:
    static bool is_candidate(Def def) { return def->candidate_ == candidate_counter_; }
    static void set_candidate(Def def) { def->candidate_ = candidate_counter_; }
    static void unset_candidate(Def def) { assert(is_candidate(def)); --def->candidate_; }

    void identify_scope(Lambda* entry);
    void number();
    size_t number(Lambda* lambda, size_t i);
    void build_cfg();

    void build_in_scope();
    void link(Lambda* src, Lambda* dst) {
        assert(is_candidate(src) && is_candidate(dst));
        succs_[sid(src)].push_back(dst);
        preds_[sid(dst)].push_back(src);
    }

    template<class T> T* lazy(AutoPtr<T>& ptr) const { return ptr ? ptr : ptr = new T(*this); }

    World& world_;
    DefSet in_scope_;
    uint32_t id_;
    std::vector<Lambda*> rpo_;
    std::vector<std::vector<Lambda*>> preds_;
    std::vector<std::vector<Lambda*>> succs_;
    mutable AutoPtr<const CFG> cfg_;

    static uint32_t candidate_counter_;
    static uint32_t id_counter_;

    template<bool> friend class ScopeView;
};

//------------------------------------------------------------------------------

}

#endif
