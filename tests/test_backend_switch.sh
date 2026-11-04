#!/bin/sh
# Model-free macOS regression; one build directory, including the switch back.
# Run from the engine root: sh tests/test_backend_switch.sh [build-directory]
set -eu
build=${1:-build/backend-switch}
for mode in 0 1 0 1; do
    make BUILD="$build" METAL="$mode" all
    links=$(otool -L "$build/lunelette")
    if [ "$mode" = 1 ]; then
        printf '%s\n' "$links" | grep -q '/Metal.framework/'
    elif printf '%s\n' "$links" | grep -q '/Metal.framework/'; then
        echo 'CPU executable still links Metal' >&2
        exit 1
    fi
    echo "PASS METAL=$mode in $build"
done
