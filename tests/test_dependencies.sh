#!/bin/sh
# Isolated build; never touch source timestamps. Run from repository root.
set -eu
case $(uname -s) in Darwin) ;; *) echo 'Metal dependency regression requires macOS: skipped'; exit 0;; esac
mkdir -p build
build=$(mktemp -d "build/deps.XXXXXX")
trap 'rm -rf "$build"' EXIT HUP INT TERM
for metal in 0 1 0 1; do
    make METAL=$metal BUILD="$build" all >"$build/build.log" 2>&1
    grep "METAL=$metal" "$build/backend.config" >/dev/null
    if test "$metal" = 0; then
        if "$build/lunelette" --backend metal 2>"$build/backend.err"; then exit 1; fi
        grep 'Metal backend unavailable' "$build/backend.err" >/dev/null
    else
        if "$build/lunelette" --backend metal 2>"$build/backend.err"; then exit 1; fi
        grep 'Require --tokenizer' "$build/backend.err" >/dev/null
    fi
done
make METAL=1 BUILD="$build" "$build/test_metal" "$build/compare_metal" >"$build/tests.log" 2>&1
make -n METAL=1 BUILD="$build" -W include/safetensors.h all "$build/test_metal" "$build/compare_metal" >"$build/dry.log"
# Match compiler commands, not link references to stale objects.
for source in metal_weights.c metal_yarn.c metal_runtime.m metal_encode.m metal_mps.m; do
    grep -- "-c src/$source -o" "$build/dry.log" >/dev/null
done
for source in tests/test_metal.m tests/compare_metal.c; do
    grep -- "$source .*src/safetensors.c -o" "$build/dry.log" >/dev/null
done
make -n METAL=1 BUILD="$build" -W include/model_facts.h all >"$build/facts.log"
grep 'tools/embed_shader.c -o' "$build/facts.log" >/dev/null
grep '/embed_shader src/metal_kernels.metal' "$build/facts.log" >/dev/null
echo 'header compiler scheduling and four same-directory backend switches PASS'
