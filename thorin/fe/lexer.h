#ifndef THORIN_FE_LEXER_H
#define THORIN_FE_LEXER_H

#include <absl/container/flat_hash_map.h>

#include "thorin/debug.h"

#include "thorin/fe/tok.h"
#include "thorin/util/utf8.h"

namespace thorin {

class World;

class Lexer {
public:
    Lexer(World&, std::string_view, std::istream&);

    World& world() { return world_; }
    Loc loc() const { return loc_; }
    Tok lex();

private:
    Tok tok(Tok::Tag tag) { return {loc(), tag}; }
    bool eof() const { return peek_.c32 == (char32_t)std::istream::traits_type::eof(); }

    /// @return @c true if @p pred holds.
    /// In this case invoke @p next() and append to @p str_;
    template<class Pred>
    bool accept_if(Pred pred, bool append = true) {
        if (pred(peek_.c32)) {
            if (append) str_ += next();
            return true;
        }
        return false;
    }

    bool accept(char32_t val, bool append = true) {
        return accept_if([val](char32_t p) { return p == val; }, append);
    }

    /// Get next utf8-char in @p stream_ and increase @p loc_ / @p peek_.pos.
    char32_t next();
    Tok lex_lit();
    void eat_comments();

    World& world_;
    Loc loc_; ///< @p Loc%ation of the @p Tok%en we are currently constructing within @p str_,
    struct {
        char32_t c32;
        Pos pos;
    } peek_;
    std::istream& stream_;
    std::string str_;
    absl::flat_hash_map<std::string, Tok::Tag> keywords_;
};

} // namespace thorin

#endif
