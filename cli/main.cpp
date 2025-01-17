#include <cstdlib>
#include <cstring>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <lyra/lyra.hpp>

#include "thorin/config.h"

#include "cli/dialects.h"
#include "thorin/be/dot/dot.h"
#include "thorin/be/ll/ll.h"
#include "thorin/fe/parser.h"
#include "thorin/pass/pass.h"

#ifdef _WIN32
#    include <windows.h>
#    define popen  _popen
#    define pclose _pclose
#    define WHICH_CLANG "where clang"
#else
#    include <dlfcn.h>
#    define WHICH_CLANG "which clang"
#endif

using namespace thorin;
using namespace std::literals;

static const auto version = "thorin command-line utility version " THORIN_VER "\n";

/// see https://stackoverflow.com/a/478960
static std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) { throw std::runtime_error("error: popen() failed!"); }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) { result += buffer.data(); }
    return result;
}

static std::string get_clang_from_path() {
    std::string clang;
    clang = exec(WHICH_CLANG);
    clang.erase(std::remove(clang.begin(), clang.end(), '\n'), clang.end());
    return clang;
}

int main(int argc, char** argv) {
    try {
        static constexpr const char* Backends = "thorin|h|md|ll|dot";

        std::string input, prefix;
        std::string clang = get_clang_from_path();
        std::vector<std::string> dialects, dialect_paths, emitters;
        std::vector<size_t> breakpoints;

        bool emit_thorin  = false;
        bool emit_h       = false;
        bool emit_md      = false;
        bool emit_ll      = false;
        bool emit_dot     = false;
        bool show_help    = false;
        bool show_version = false;

        int verbose      = 0;
        auto inc_verbose = [&](bool) { ++verbose; };

        // clang-format off
        auto cli = lyra::cli()
            | lyra::help(show_help)
            | lyra::opt(show_version             )["-v"]["--version"     ]("Display version info and exit.")
            | lyra::opt(clang,         "clang"   )["-c"]["--clang"       ]("Path to clang executable (default: '" WHICH_CLANG "').")
            | lyra::opt(dialects,      "dialect" )["-d"]["--dialect"     ]("Dynamically load dialect [WIP].")
            | lyra::opt(dialect_paths, "path"    )["-D"]["--dialect-path"]("Path to search dialects in.")
            | lyra::opt(emitters,      Backends  )["-e"]["--emit"        ]("Select emitter. Multiple emitters can be specified simultaneously.").choices("thorin", "h", "md", "ll", "dot")
            | lyra::opt(inc_verbose              )["-V"]["--verbose"     ]("Verbose mode. Multiple -V options increase the verbosity. The maximum is 4.").cardinality(0, 4)
#ifndef NDEBUG
            | lyra::opt(breakpoints,   "gid"     )["-b"]["--break"       ]("Trigger breakpoint upon construction of node with global id <gid>. Useful when running in a debugger.")
#endif
            | lyra::opt(prefix,        "prefix"  )["-o"]["--output"      ]("Prefix used for various output files.")
            | lyra::arg(input,         "file"    )                        ("Input file.");

        if (auto result = cli.parse({argc, argv}); !result) throw std::invalid_argument(result.message());

        if (show_help) {
            std::cout << cli;
            return EXIT_SUCCESS;
        }

        if (show_version) {
            std::cerr << version;
            std::exit(EXIT_SUCCESS);
        }

        for (const auto& e : emitters) {
            if (false) {}
            else if (e == "thorin") emit_thorin = true;
            else if (e == "h" )     emit_h      = true;
            else if (e == "md")     emit_md     = true;
            else if (e == "ll")     emit_ll     = true;
            else if (e == "dot")    emit_dot    = true;
            else unreachable();
        }
        // clang-format on

        if (!dialects.empty()) {
            for (const auto& dialect : dialects) test_plugin(dialect, dialect_paths);
            return EXIT_SUCCESS;
        }

        if (input.empty()) throw std::invalid_argument("error: no input given");
        if (input[0] == '-' || input.substr(0, 2) == "--")
            throw std::invalid_argument("error: unknown option " + input);

        if (prefix.empty()) {
            auto filename = std::filesystem::path(input).filename();
            if (filename.extension() != ".thorin") throw std::invalid_argument("error: invalid file name '" + input + "'");
            prefix = filename.stem().string();
        }

        World world;
        world.set_log_ostream(&std::cerr);
        world.set_log_level((LogLevel)verbose);
#ifndef NDEBUG
        for (auto b : breakpoints) world.breakpoint(b);
#endif

        std::ifstream ifs(input);
        if (!ifs) {
            errln("error: cannot read file '{}'", input);
            return EXIT_FAILURE;
        }

        std::ofstream md;
        if (emit_md) md.open(prefix + ".md");
        Parser parser(world, input, ifs, emit_md ? &md : nullptr);
        parser.parse_module();

        if (emit_h) {
            std::ofstream h(prefix + ".h");
            parser.bootstrap(h);
        }

        if (emit_thorin) world.dump();

        if (emit_dot) {
            std::ofstream ofs(prefix + ".dot");
            dot::emit(world, ofs);
        }

        if (emit_ll) {
            std::ofstream ofs(prefix + ".ll");
            ll::emit(world, ofs);
        }
    } catch (const std::exception& e) {
        errln("{}", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        errln("error: unknown exception");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
