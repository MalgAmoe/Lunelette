#!/bin/sh
# Replace only when effective recipe definitions change, preserving mtime otherwise.
set -eu
output=$1
commit=$2
dirty=$3
temporary=$(mktemp "${output}.XXXXXX")
trap 'rm -f "$temporary"' 0
trap 'exit 1' HUP INT TERM
printf 'SERVICE_COMMIT=%s\nSERVICE_DIRTY=%s\n' "$commit" "$dirty" >"$temporary"
if ! cmp -s "$temporary" "$output"; then
    mv "$temporary" "$output"
fi
