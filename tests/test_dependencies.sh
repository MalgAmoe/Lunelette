#!/bin/sh
# Isolated build; never touch source timestamps. Run from repository root.
set -eu
case $(uname -s) in Darwin) ;; *) echo 'Metal dependency regression requires macOS: skipped'; exit 0;; esac
mkdir -p build
build=$(mktemp -d "build/deps.XXXXXX")
trap 'rm -rf "$build"' EXIT HUP INT TERM
make BUILD="$build" all >"$build/build.log" 2>&1
make BUILD="$build" "$build/test_metal" "$build/compare_metal" >"$build/tests.log" 2>&1
make -n BUILD="$build" -W include/safetensors.h all "$build/test_metal" "$build/compare_metal" >"$build/dry.log"
# Match compiler commands, not link references to stale objects.
for source in metal_weights.c metal_yarn.c metal_runtime.m metal_encode.m metal_mps.m; do
    grep -- "-c src/$source -o" "$build/dry.log" >/dev/null
done
for source in tests/test_metal.m tests/compare_metal.c; do
    grep -- "$source .*src/safetensors.c -o" "$build/dry.log" >/dev/null
done
make -n BUILD="$build" -W include/model_facts.h all >"$build/facts.log"
grep 'tools/embed_shader.c -o' "$build/facts.log" >/dev/null
grep '/embed_shader src/metal_kernels.metal' "$build/facts.log" >/dev/null
make -n BUILD="$build" -W src/metal_kernels.metal all >"$build/shader.log"
grep '/embed_shader src/metal_kernels.metal' "$build/shader.log" >/dev/null
echo 'header and shader compiler scheduling PASS'
