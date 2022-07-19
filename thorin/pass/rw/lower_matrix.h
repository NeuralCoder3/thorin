#ifndef THORIN_PASS_LOWER_MATRIX_H
#define THORIN_PASS_LOWER_MATRIX_H

#include "thorin/pass/pass.h"

namespace thorin {

#define CODE(T, o) o,
enum class MOpStub : flags_t {
    THORIN_M_OP (CODE)
    maxLast, map, formula
};
#undef CODE

class NestedLoops{
public:
    DefArray indices;
    DefArray vars;
    DefArray reductions;

    Lam* entry;
    Lam* finish;
    Lam* body;
};



class LoopBuilder{
    World& world;
    DefVec starts;
    std::vector<std::pair<const Def*, const Def*>> ranges;
    DefVec var_ty;
    DefVec var_init;

#define b world.builder()
public:
    LoopBuilder(World& w) : world(w){
    }

    void addLoop( const Def* count ){
        ranges.emplace_back(world.lit_int_width(64, 0), count);
    }

    void addLoop( const Def* start, const Def* end ){
        ranges.emplace_back(start, end);
    }

    void addVar(const Def* ty, const Def* init){
        var_ty.push_back(ty);
        var_init.push_back(init);
    }


    NestedLoops build(){
        NestedLoops nestedLoops;
        build(nestedLoops);
        return nestedLoops;
    }

    void build(NestedLoops& nestedLoops){
        Lam* finish = b.mem().add(var_ty).nom_filter_lam("loop_result");
        Lam* entry = b.mem().nom_filter_lam("loop_entry1");
        Lam* body = b.mem().add(var_ty).add(b.mem().add(var_ty).cn()).nom_filter_lam("loop_body");

        b.mem(entry).add(var_init).add(finish).app_body(entry, body);
        b.add(body->vars().skip_back(1)).app_body(body, body->ret_var());
        bool body_is_yield = false;

        DefVec indices;
        for(auto [start, end] : ranges){
            auto [loop, yield] = world.repeat(start, end, var_ty);

            b
                .mem(body)
                .add(body->vars().skip_front(1 + (body_is_yield ? 1 : 0)))
                .app_body(body, loop);

            body_is_yield = true;
            body = yield;
            indices.push_back(yield->var(1));
        }

        nestedLoops = {
            .indices = indices,
            .vars = body->vars().skip(1 + (body_is_yield ? 1 : 0), 1),
            .reductions = finish->vars().skip_front(1),
            .entry = entry,
            .finish = finish,
            .body = body
        };
    }
};

class MatrixHelper{
    World& world;
    const Def* mat = nullptr;
    bool transpose = false;

public:
    MatrixHelper(World& w) : world(w){

    }

    MatrixHelper(const Def* mat) : world(mat->world()), mat(mat){

    }

    void setMat(const Def* mat){
        this->mat = mat;
    }

    const Def* getMat(){
        return mat;
    }

    void setTranspose(bool transpose){
        this->transpose = transpose;
    }

    const Def* getRows(){
        return mat->proj(2 + (transpose ? 1 : 0));
    }

    const Def* getCols(){
        return mat->proj(2 + (transpose ? 0 : 1));
    }

    const Def* getPointer(){
        return mat->proj(1);
    }

    const Def* getMeta(){
        return mat->proj(0);
    }

    const Def* bitCheck(u32 bit){
        auto meta = getMeta();
        auto mask = world.lit_int_width(64, 1 << bit);
        auto masked = world.op(Bit::_and, meta, mask);
        return world.op(ICmp::e, masked, mask);
    }

    const Def* isZero(){
        return bitCheck(0);
    }

    const Def* isOne(){
        return bitCheck(1);
    }

    const Def* isConst(){
        return bitCheck(2);
    }

    const Def* isTranspose(){
        return bitCheck(3);
    }

    const Def* dim(size_t i){
        return world.extract(mat, i + 2);
    }

    size_t shape_size(){
        return as_lit(world.dim_count_of_tn(mat->type()));
    }

    DefArray dims(){
        auto size = shape_size();

        DefArray result{size};

        for(size_t i = 0 ; i < size ; i++){
            result[i] = dim(i);
        }

        return result;
    }

    const Def* elem_type(){
        return world.elem_ty_of_tn(mat->type());
    }

    const Def* getIndex(const DefArray& indices){
        return getIndex(indices.size(), [&](auto i){ return indices[i];});
    }

    const Def* getIndex(const Def* row, const Def* col){
        return getIndex({row, col});
    }

    const Def* getIndex(size_t size, std::function<const Def*(size_t)> f){
        const Def* index = nullptr;
        for( size_t i = 0 ; i < size ; i++ ){
            auto dim_index = f(i);

            if( i == 0 ){
                index = dim_index;
            }else{
                auto dim_size = dim(i);
                index = world.op(Wrap::add, (nat_t)0, dim_index, world.op(Wrap::mul, (nat_t)0, index, dim_size));
            }
        }

        return index;
    }

    const Def* getSize(){
        const Def* size = nullptr;
        for( size_t i = 0 ; i < shape_size() ; i++ ){
            auto dim_size = dim(i);

            if( i == 0 ){
                size = dim_size;
            }else{
                size = world.op(Wrap::mul, (nat_t)0, size, dim_size);
            }
        }

        return size;
    }

    const Def* getLea(DefArray& indices){
        return lea(getIndex(indices.size(), [&](auto i){ return indices[i];} ));
    }

    const Def* getLea(size_t size, std::function<const Def*(size_t)> f){
        const Def* index = nullptr;
        for( size_t i = 0 ; i < size ; i++ ){
            auto dim_index = f(i);

            if( i == 0 ){
                index = dim_index;
            }else{
                auto dim_size = dim(i);
                index = world.op(Wrap::add, (nat_t)0, dim_index, world.op(Wrap::mul, (nat_t)0, index, dim_size));
            }
        }

        return lea(index);
    }

    const Def* getLea(const Def* row, const Def* col){
        auto index = getIndex(row, col);
        return lea(index);
    }

    const Def* lea(const Def* index){
        auto ptr = getPointer();
        return world.op_lea(ptr, index);
    }

    const Def* load(const Def* mem, const Def* index){
        const Def* ptr = lea(index);
        return world.op_load(mem, ptr);
    }

    const Def* store(const Def* mem, const Def* index, const Def* value){
        const Def* ptr = lea(index);
        return world.op_store(mem, ptr, value);
    }

    const Def* load_const_value(const Def* mem){
        return load(mem, world.lit_int_width(64, 0));
    }
};

class InputHelper{
public:
    Lam* lam = nullptr;
    const Def* arg = nullptr;
    MatrixHelper left;
    MatrixHelper right;

    InputHelper(World& w) : left(w), right(w){

    }
};

class ConstructHelper{
public:
    InputHelper impl;
    MatrixHelper result;
    MatrixHelper maxIndices;

    const Def* raw_result = nullptr;
    DefArray indices;
    DefArray vars;
    Lam* body = nullptr;

    ConstructHelper(World& w) : impl(w), result(w), maxIndices(w){

    }
};

struct ChainHelper{
    const Def* currentMem = nullptr;
    Lam* tail = nullptr;
};

class LowerMatrix : public RWPass<Lam> {
public:
    LowerMatrix(PassMan& man)
        : RWPass(man, "lower_matrix") {}

    void enter() override;
    const Def* rewrite_rec(const Def* current, bool convert = true);
    const Def* rewrite_rec_convert(const Def* current);
    //const Lam* const_reduction(MOp mop, ROp rop,ConstructHelper& helper);
    const Lam* create_MOp_entry(const Axiom* mop_axiom, size_t dims, const Def* elem_type, const Def* mmode);
    const Lam* create_MOp_impl(const Axiom* mop_axiom, size_t dims, const Def* elem_type, const Def* rmode);
    void construct_mat_loop(const Def* elem_type, const Def* a_rows, const Def* b_cols, const Def* alloc_rows, const Def* alloc_cols, bool flatten, ConstructHelper& constructResult);
    void construct_mop( MOpStub mop, const Def* elem_type, ConstructHelper& constructResult);
    Lam* rewrite_mop(const App* app, const Def* arg_wrap);
    Lam* rewrite_map(const App* app, const Def* arg_wrap);
    Lam* rewrite_formula(const App* app, const Def* arg_wrap);
    const Def* alloc_stencil(const Def* stencil, MatrixHelper& helper,  const Def*& mem);
    const Def* rewrite_app(const App* app);
    void store_rec(const Def* value, const Def* mat, const Def* index, const Def*& mem);

    const Pi* mop_pi(MOpStub mop, size_t dims, const Def* elem_type);
    Lam* mop_lam(MOpStub mop, size_t dims, const Def* elem_type, const std::string& name);

    ChainHelper chainHelper;

    Def2Def old2new;
    DefMap<Lam*> mop_variants;
};

    //Def2Def old2new_;
}

#endif
