# Language Reference {#langref}

[TOC]

## Lexical Structure {#lex}

Thorin files are [UTF-8](https://en.wikipedia.org/wiki/UTF-8) encoded and [lexed](https://en.wikipedia.org/wiki/Lexical_analysis) from left to right.
The [maximal munch](https://en.wikipedia.org/wiki/Maximal_munch) strategy resolves any ambiguities in the lexical rules below.
For Example, `<<<` is lexed as `<<` and `<`.

### Terminals {#terminals}

The actual *[terminals](https://en.wikipedia.org/wiki/Terminal_and_nonterminal_symbols)* are specified in the following tables.

#### Primary Terminals

The [grammatical rules](#productions) will directly reference these *primary [terminals](https://en.wikipedia.org/wiki/Terminal_and_nonterminal_symbols)*.
*Secondary terminals* are [ASCII](https://en.wikipedia.org/wiki/ASCII)-only tokens that represent the **same** lexical element as its corresponding *primary token*.
For example, the lexer doesn't care, if you use `⊥` or `.bot`.
Both tokens are identified as `⊥`.

| Primary Terminals               | Secondary Terminals                             | Comment                   |
|---------------------------------|-------------------------------------------------|---------------------------|
| `(` `)` `[` `]` `{` `}`         |                                                 | delimiters                |
| `‹` `›` `«` `»`                 | `<<` `>>` `<` `>`                               | UTF-8 delimiters          |
| `→` `∷` `⊥` `⊤` `★` `□` `λ` `Π` | `->` `::` `.bot` `.top` `*` `\`  <tt>\|~\|</tt> | further UTF-8 tokens      |
| `=` `,` `;` `.` `#`             |                                                 | further tokens            |
| `<eof>`                         |                                                 | marks the end of the file |



#### Keywords

In addition the following keywords are *terminals*:

| Terminal    | Comment                   |
|-------------|---------------------------|
| `.ax`       | axiom                     |
| `.let`      | let expression            |
| `.Pi`       | nominal Pi                |
| `.lam`      | nominal lam               |
| `.Arr`      | nominal Arr               |
| `.pack`     | nominal pack              |
| `.Sigma`    | nominal Sigma             |
| `.def`      | nominal definition        |
| `.external` | marks nominal as external |
| `.module`   | starts a module           |
| `.Nat`      | thorin::Nat               |
| `.ff`       | alias for `0₂`            |
| `.tt`       | alias for `1₂`            |

All keywords start with a `.` to prevent name clashes with identifiers.

#### Regular Expressions

The following *terminals* comprise more complicated patterns that are specified via [regular expressions](https://en.wikipedia.org/wiki/Regular_expression):

| Terminal | Regular Expression                   | Comment                                                                                           |
|----------|--------------------------------------|---------------------------------------------------------------------------------------------------|
| Sym      | sym                                  | symbol                                                                                            |
| Ax       | `:` sym `.` sym ( `.` sym)?          | Axiom                                                                                             |
| L        | dec+                                 | unsigned decimal literal                                                                          |
| L        | 0b bin+                              | unsigned binary literal                                                                           |
| L        | 0o oct+                              | unsigned octal literal                                                                            |
| L        | 0x hex+                              | unsigned hexadecimal literal                                                                      |
| L        | sign dec+                            | signed decimal literal                                                                            |
| L        | sign 0b bin+                         | signed binary literal                                                                             |
| L        | sign 0o oct+                         | signed octal literal                                                                              |
| L        | sign 0x hex+                         | signed hexadecimal literal                                                                        |
| L        | sign? dec+ eE sign dec+              | floating-point literal                                                                            |
| L        | sign? dec+ `.` dec\* (eE sign dec+)? | floating-point literal                                                                            |
| L        | sign? dec\* `.` dec+ (eE sign dec+)? | floating-point literal                                                                            |
| L        | sign? 0x hex+ pP sign dec+           | [floating-point hexadecimal](https://en.cppreference.com/w/cpp/language/floating_literal) literal |
| L        | sign? 0x hex+ `.` hex\* pP sign dec+ | [floating-point hexadecimal](https://en.cppreference.com/w/cpp/language/floating_literal) literal |
| L        | sign? 0x hex\* `.` hex+ pP sign dec+ | [floating-point hexadecimal](https://en.cppreference.com/w/cpp/language/floating_literal) literal |
| I        | dec+ sub+                            | integer literal of type `:Int mod`                                                                |
| I        | dec+ `_` dec+                        | integer literal of type `:Int mod`                                                                |

The previous table resorts to the following definitions as shorthand:

| Name | Regular Expression                                         | Comment                                         |
|------|------------------------------------------------------------|-------------------------------------------------|
| 0b   | `0` \[ `b``B` \]                                           | prefix for binary literals                      |
| 0o   | `0` \[ `o``O` \]                                           | prefix for octal literals                       |
| 0x   | `0` \[ `x``X` \]                                           | prefix for hexadecimal literals                 |
| bin  | \[ `0``1` \]                                               | binary digit                                    |
| oct  | \[ `0`-`7` \]                                              | octal digit                                     |
| dec  | \[ `0`-`9` \]                                              | decimal digit                                   |
| sub  | \[ `₀`-`₉` \]                                              | subscript digit (always decimal)                |
| hex  | \[ `0`-`9``a`-`f``A`-`F` \]                                | hexadecimal digit                               |
| eE   | \[ `e` `E` \]                                              | exponent in floating point literals             |
| pP   | \[ `p` `P` \]                                              | exponent in floating point hexadecimal literals |
| sign | \[ `+` `-` \]                                              |                                                 |
| sym  | \[ `_``a`-`z``A`-`Z` \]\[ `.``_``0`-`9``a`-`z``A`-`Z` \]\* | symbol                                          |

So, *sym* referes to the shorthand rule while *Sym* refers to the *terminal* that is identical to *sym*.
However, the terminal *Ax* also uses the shorthand rule *sym*.

### Comments

In addition, the following comments are avaiable:
* `/* ... */` multi-line comment
* `//` single-line comment
* `///` single-line comment that is put into the Markdown output (see [Emitters](@ref emitters))

## Grammar

Thorin's grammar is defined as a [context-free grammar](https://en.wikipedia.org/wiki/Context-free_grammar) that consists of the *terminals* defined [above](#terminals) as well as the *nonterminals* and *productions* defined [below](#productions).
The start symbol is "m" (module).

### Productions {#productions}

The following tables comprise all produciton rules:

#### Module

| Nonterminal | Right-Hand Side | Comment | Thorin Class  |
|-------------|-----------------|---------|---------------|
| m           | d ... d         | module  | thorin::World |

#### Declaration

| Nonterminal | Right-Hand Side                                                | Comment                        | Thorin Class  |
|-------------|----------------------------------------------------------------|--------------------------------|---------------|
| d           | `.ax` Sym `:` e<sub>type</sub> `;`                             | axiom                          | thorin::Axiom |
| d           | `.let` Sym `:` e<sub>type</sub> `;`                            | let                            | -             |
| d           | `.Pi` Sym ( `:` e<sub>type</sub> )? `,` e<sub>dom</sub> n      | nominal Pi declaration         | thorin::Pi    |
| d           | `.lam` Sym `,` e<sub>type</sub> n                              | nominal lambda declaration     | thorin::Lam   |
| d           | `.Arr` Sym ( `:` e<sub>type</sub> )? `,` e<sub>shape</sub> n   | nominal array declaration      | thorin::Arr   |
| d           | `.pack` Sym ( `:` e<sub>type</sub> )? `,` e<sub>shape</sub> n  | nominal pack declaration       | thorin::Pack  |
| d           | `.Sigma` Sym ( `:` e<sub>type</sub> )? `,` L<sub>arity</sub> n | nominal sigma declaration      | thorin::Sigma |
| d           | `.def` Sym n                                                   | nominal definition             | nominals      |
| n           | `;` \| o                                                       | nominal definition             | -             |
| o           | `=` e `;`                                                      | operand of nominal definition  | -             |
| o           | `=` `{` e `,` ... `,` e  `}` `;`                               | operands of nominal definition | -             |

#### Expressions

| Nonterminal | Right-Hand Side                    | Comment                             | Thorin Class    |
|-------------|------------------------------------|-------------------------------------|-----------------|
| e           | `*`                                | type                                | thorin::Type    |
| e           | L `∷` e                            | literal                             | thorin::Lit     |
| e           | ( `.bot` \| `.top` ) ( `∷` e )?    | bottom/top                          | thorin::TExt    |
| e           | Sym                                | identifier                          | -               |
| e           | Ax                                 | use of an axiom                     | -               |
| e           | e e                                | application                         | thorin::App     |
| e           | `λ` Sym `:` e `→` e  `.` e         | lambda                              | thorin::Lam     |
| e           | e `→` e                            | function type                       | thorin::Pi      |
| e           | `Π` Sym `:` e `→` e                | dependent function type             | thorin::Pi      |
| e           | e `#` e                            | extract                             | thorin::Extract |
| e           | `.ins` `(` e `,` e `,` e ` )`      | insert                              | thorin::Insert  |
| e           | `(` e `,` ... `,` e` )` ( `:` e )? | tuple with optional type ascription | thorin::Tuple   |
| e           | `[` e `,` ... `,` e `]`            | sigma                               | thorin::Sigma   |
| e           | d ... d e                          | declaration block                   | -               |

An elided type of
* a literal defaults to `.Nat`,
* a bottom/top defaults to `*`,
* a nominals defaults to `*`.

### Precedence

Expressions nesting is disambiguated according to the following precedence table (from strongest to weakest binding):

| Operator      | Description                         | Associativity |
|---------------|-------------------------------------|---------------|
| L `∷` e       | type ascription of a literal        | -             |
| e `#` e       | extract                             | left-to-right |
| e e           | application                         | left-to-right |
| `Π` Sym `:` e | domain of a dependent function type | -             |
| e `→` e       | function type                       | right-to-left |

Note that the domain of a dependent function type binds slightly stronger than `→`.
This has the effect that, e.g., `Π T: * → T -> T` has the exepcted binding like this: (`Π T: *`) `→` (`T → T`).
Otherwise, `→` would be consumed by the domain: `Π T:` (`* →` (`T → T`)) ↯.
