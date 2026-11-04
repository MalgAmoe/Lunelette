#!/bin/sh
set -eu
cli=$1
reject() {
    if "$cli" "$@" >cli-test.out 2>cli-test.err; then
        echo "unexpected success: $*" >&2
        exit 1
    fi
}
trap 'rm -f cli-test.out cli-test.err test_tokenizer_fixture.tmp' EXIT HUP INT TERM
"$cli" --help >/dev/null
"$cli" --version >/dev/null
reject
reject --invalid
reject --metal-prefill bad
grep -- '--metal-prefill must be' cli-test.err >/dev/null
reject --metal-prefill
reject --metal-prefill mps --metal-prefill custom
for mode in mps custom; do
    reject --metal-prefill "$mode" --tokenizer missing --prompt x --dump-prompt-ids
    if grep 'Unknown or duplicate\|--metal-prefill must be' cli-test.err >/dev/null; then exit 1; fi
done
for option in '--backend cpu' '--backend metal' '--backend' '--uncached'; do
    reject $option
    grep 'were retired' cli-test.err >/dev/null
done
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
# POSIX read-only stdout: no /dev/full dependency, pipe race or SIGPIPE.
if test -c /dev/null; then
    for option in --help --version; do
        if "$cli" "$option" 1< /dev/null 2>cli-test.err; then exit 1; fi
        grep 'Writing .* stdout failed' cli-test.err >/dev/null
    done
    ./test_tokenizer --cli-fixture
    if "$cli" --tokenizer test_tokenizer_fixture.tmp --prompt a --raw \
        --dump-prompt-ids 1< /dev/null 2>cli-test.err; then exit 1; fi
    rm test_tokenizer_fixture.tmp
    grep 'Writing prompt IDs to stdout failed' cli-test.err >/dev/null
fi
