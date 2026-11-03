#!/bin/sh
set -eu
cli=$1
reject() {
    if "$cli" "$@" >cli-test.out 2>cli-test.err; then
        echo "unexpected success: $*" >&2
        exit 1
    fi
}
trap 'rm -f cli-test.out cli-test.err' EXIT HUP INT TERM
"$cli" --help >/dev/null
"$cli" --version >/dev/null
reject
reject --invalid
reject --prompt
reject --prompt x
reject --tokenizer missing --prompt x --raw --system s --dump-prompt-ids
reject --tokenizer missing --prompt x --dump-prompt-ids --prompt y
reject --tokenizer missing --prompt x --dump-prompt-ids --unknown
for n in '' -1 +1 257 999999999999999999999999999999999 1x ' 1' 1.0; do
    reject --tokenizer missing --prompt x --dump-prompt-ids --max-new-tokens "$n"
    grep 'max-new-tokens must be decimal 0..256' cli-test.err >/dev/null
done
# Valid budgets must reach tokenizer I/O, not fail numeric parsing.
for n in 0 1 32 256 000256; do
    reject --tokenizer /nonexistent/lunelette-tekken --prompt x --dump-prompt-ids --max-new-tokens "$n"
    if grep 'max-new-tokens must' cli-test.err >/dev/null; then exit 1; fi
done
reject --tokenizer missing --prompt x --weights a
reject --help --invalid
reject --version --invalid
echo 'CLI parser tests passed'
