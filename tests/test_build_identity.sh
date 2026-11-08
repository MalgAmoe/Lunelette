#!/bin/sh
# Exercise the real consumer recipes with a recording compiler, not Metal/models.
set -eu
fixture=$(mktemp -d "${TMPDIR:-/tmp}/lunelette-identity.XXXXXX")
trap 'rm -rf "$fixture"' 0
trap 'exit 1' HUP INT TERM
mkdir "$fixture/repo" "$fixture/bin"
cp Makefile "$fixture/repo/"
cp -R src include service "$fixture/repo/"
mkdir "$fixture/repo/tools"
cp tools/update_service_identity.sh "$fixture/repo/tools/"
# Select the native recipe even on portable hosts; no platform compiler runs.
printf '#!/bin/sh\necho Darwin\n' >"$fixture/bin/uname"
cat >"$fixture/bin/record-cc" <<'EOF'
#!/bin/sh
set -eu
output=
identity=
while [ "$#" -gt 0 ]; do
    case $1 in
        -o) shift; output=$1 ;;
        -DSERVICE_COMMIT=*|-DSERVICE_DIRTY=*) identity="$identity$1
" ;;
    esac
    shift
done
[ -n "$output" ]
printf '%s' "$identity" >"$output"
printf '%s\n' "$output" >>build/compiles
EOF
chmod +x "$fixture/bin/record-cc" "$fixture/bin/uname"
PATH="$fixture/bin:$PATH"
export PATH
# Do not inherit caller repository/config/hooks or signing settings.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR
GIT_CONFIG_NOSYSTEM=1
GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM GIT_CONFIG_GLOBAL
cd "$fixture/repo"
git init -q --template=
git config user.name 'Build identity test'
git config user.email 'build-identity@example.invalid'
git config commit.gpgsign false
printf '/build/\n' >.gitignore
printf 'clean\n' >tracked.txt
git add .
git commit -qm initial
build()
{
    # Delay makes missing invalidation visible even on coarse timestamp filesystems.
    sleep 1
    MAKEFLAGS= MFLAGS= MAKEOVERRIDES= make -s BUILD=build CC=record-cc METAL_OBJECTS= METAL_HEADERS= \
        build/test_service_server build/lunelette-server "$@"
}
check()
{
    expected_commit=$1
    expected_dirty=$2
    expected_count=$3
    for target in build/test_service_server build/lunelette-server; do
        grep -Fx -- "-DSERVICE_COMMIT=\"$expected_commit\"" "$target" >/dev/null
        grep -Fx -- "-DSERVICE_DIRTY=\"$expected_dirty\"" "$target" >/dev/null
    done
    [ "$(wc -l <build/compiles | tr -d ' ')" = "$expected_count" ]
}
unchanged()
{
    cp -p build/service-identity "$fixture/stamp"
    cp -p build/test_service_server "$fixture/server"
    cp -p build/lunelette-server "$fixture/native"
    build "$@"
    cmp build/service-identity "$fixture/stamp"
    for pair in 'service-identity stamp' 'test_service_server server' 'lunelette-server native'; do
        set -- $pair
        [ ! "build/$1" -nt "$fixture/$2" ]
        [ ! "$fixture/$2" -nt "build/$1" ]
    done
}
build
check "$(git rev-parse --short HEAD)" false 2
unchanged
check "$(git rev-parse --short HEAD)" false 2
# Empty commit changes no source prerequisite.
git commit -qm commit-only --allow-empty
build
check "$(git rev-parse --short HEAD)" false 4
printf 'dirty\n' >>tracked.txt
build
check "$(git rev-parse --short HEAD)" true 6
unchanged
check "$(git rev-parse --short HEAD)" true 6
git checkout -- tracked.txt
build
check "$(git rev-parse --short HEAD)" false 8
build SERVICE_COMMIT=override SERVICE_DIRTY=true
check override true 10
unchanged SERVICE_COMMIT=override SERVICE_DIRTY=true
check override true 10
# Exported source with no Git metadata retains the existing unknown/false fallback.
mv .git "$fixture/git-metadata"
build
check unknown false 12
unchanged
check unknown false 12
# An identity prerequisite must not manufacture a no-op native server on non-macOS.
printf '#!/bin/sh\necho Linux\n' >"$fixture/bin/uname"
if MAKEFLAGS= MFLAGS= MAKEOVERRIDES= make -s -n BUILD=build/unsupported server >"$fixture/unsupported.log" 2>&1; then
    echo 'unsupported native server unexpectedly succeeded' >&2
    exit 1
fi
grep 'No rule to make target' "$fixture/unsupported.log" >/dev/null
printf 'build identity invalidation PASS\n'
