#!/bin/sh
# Inspect Make's complete schedules, without compiling or executing any target.
set -eu
scratch=$(mktemp -d "${TMPDIR:-/tmp}/lunelette-entrypoints.XXXXXX")
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
build=build/entrypoint-dry-run
make -n -B BUILD="$build" host-test >"$scratch/host"
make -n -B BUILD="$build" python-test >"$scratch/python"
make -n -B BUILD="$build" interactive-test >"$scratch/interactive"
grep "./$build/test_generation" "$scratch/host" >/dev/null
grep "./$build/test_service_session" "$scratch/host" >/dev/null
if grep -E 'src/metal_(runtime|encode|mps)\.m|/test_metal_reference |/test_metal_compact;|/compare_metal --self-test' "$scratch/host"; then
    echo 'host-test unexpectedly schedules Metal compilation/execution' >&2
    exit 1
fi
grep "test_\*workload\*\.py" "$scratch/python" >/dev/null
grep 'test_replay_logits.py' "$scratch/python" >/dev/null
grep 'test_fp8_lookup.py' "$scratch/python" >/dev/null
grep "tests/test_interactive_workload.py ./$build/test_service_server" "$scratch/interactive" >/dev/null
case $(uname -s) in
Darwin)
    make -n -B BUILD="$build" server >"$scratch/server"
    make -n -B BUILD="$build" gpu-test >"$scratch/gpu"
    # Recursive Make is executed by -n, but inherits dry-run for all recipes.
    make -n -B BUILD="$build" dev-check >"$scratch/dev"
    for suite in server test python-test interactive-test service-test; do
        grep -F "BUILD=\"$build\" $suite" "$scratch/dev" >/dev/null
    done
    grep -- "-o $build/lunelette-server" "$scratch/server" >/dev/null
    # Build-only server entrypoint must never launch the server.
    if grep "^\./$build/lunelette-server" "$scratch/server"; then exit 1; fi
    grep "./$build/test_metal_reference tests/frozen/metal-baseline-v1.metal" "$scratch/gpu" >/dev/null
    grep "./$build/test_metal_compact;" "$scratch/gpu" >/dev/null
    ;;
esac
echo 'build/test entrypoint schedules PASS (dry-run only)'
