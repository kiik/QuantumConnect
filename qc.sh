#!/usr/bin/env bash
# Build and run the bot and its tools.
#   ./qc.sh build                 compile everything; package out/QuarkBot.jar (bot class only)
#   ./qc.sh jar <bot class>       package one bot for upload, e.g. ./qc.sh jar ch.qc.meelik.QuarkBot
#   ./qc.sh check <jar>...        pre-upload check: manifest, size, forbidden packages
#   ./qc.sh test                  engine self-test (SelfTest)
#   ./qc.sh arena A B [games]     headless matches, e.g. ./qc.sh arena quark random 20
#   ./qc.sh census [geom] [ply]   state-space census, e.g. ./qc.sh census 6x7x4-nobomb 9
#   ./qc.sh tree [moves] [plies] [evalDepth] [width] [geom]   tree + heatmap, renders out/tree.svg
#   ./qc.sh tune [gens] [depth] [pairs] [openings] [threads]  evolve eval weights -> out/tune-params.txt
#   ./qc.sh league [file] [games] [threads]  round robin of league.txt -> results/league.md
#   ./qc.sh replay [A] [B] [games]           record games -> out/replay.html (browser viewer)
# The bot compiles against the real engine JAR in libs/ when present, otherwise against src/stub.
# Set JAVA_HOME to pick a JDK; otherwise javac on PATH, then a Windows JDK when running under WSL.
set -euo pipefail
cd "$(dirname "$0")"

find_bin() {
    if [ -n "${JAVA_HOME:-}" ]; then
        for f in "$JAVA_HOME/bin/$1" "$JAVA_HOME/bin/$1.exe"; do [ -x "$f" ] && { echo "$f"; return; }; done
    fi
    command -v "$1" 2>/dev/null && return
    for d in /mnt/c/Program\ Files/Java/jdk-* /mnt/c/Program\ Files/Eclipse\ Adoptium/jdk-*; do
        [ -x "$d/bin/$1.exe" ] && last="$d/bin/$1.exe"
    done
    [ -n "${last:-}" ] && echo "$last" && return
    echo "no $1 found; set JAVA_HOME" >&2; exit 1
}
JAVAC=$(find_bin javac); JAVA=$(find_bin java); JAR=$(find_bin jar)
case "$JAVAC" in *.exe) SEP=';' ;; *) SEP=':' ;; esac

engine_cp() {
    local lib; lib=$(ls libs/*.jar 2>/dev/null | head -1 || true)
    if [ -n "$lib" ]; then echo "$lib"; else echo "out/stub"; fi
}

build() {
    rm -rf out/stub out/bot out/tools
    local cp; cp=$(engine_cp)
    if [ "$cp" = out/stub ]; then
        echo "engine: stand-in classes (src/stub); put the engine JAR in libs/ to build against the real one"
        "$JAVAC" --release 17 -d out/stub $(find src/stub -name '*.java')
    else
        echo "engine: $cp"
    fi
    "$JAVAC" --release 17 -Xlint:all -cp "$cp" -d out/bot $(find src/bot -name '*.java')
    "$JAVAC" --release 17 -cp "out/bot$SEP$cp" -d out/tools $(find src/tools -name '*.java')
    package ch.qc.meelik.QuarkBot
}

# One JAR per bot: the class and its nested classes only, with Bot-Class set.
package() {
    local cls=$1 path=${1//.//} name=${1##*.}
    [ -f "out/bot/$path.class" ] || { echo "no compiled bot $cls (run ./qc.sh build)" >&2; exit 1; }
    printf 'Bot-Class: %s\n' "$cls" > "out/$name.manifest"
    local files; files=$(cd out/bot && ls "$path.class" "$path"\$*.class 2>/dev/null || true)
    "$JAR" --create --file "out/$name.jar" --manifest "out/$name.manifest" $(for f in $files; do printf -- '-C out/bot %s ' "$f"; done)
    echo "built out/$name.jar: $(echo $files)"
}

run() { "$JAVA" -cp "out/bot${SEP}out/tools${SEP}$(engine_cp)" "ch.qc.meelik.tools.$@"; }

cmd=${1:-build}; shift || true
case "$cmd" in
    build) build ;;
    jar) package "${1:?bot class}" ;;
    league) run League "$@" ;;
    check) run Check "$@" ;;
    replay)
        run Replay "$@"
        command -v wslpath >/dev/null && echo "open in a browser: $(wslpath -w out/replay.html)"
        ;;
    test) run SelfTest "$@" ;;
    arena) run Arena "$@" ;;
    census) run Census "$@" ;;
    tune) run Tune "$@" ;;
    tree)
        run TreeViz "$@"
        DOT=$(command -v dot || ls /mnt/c/Program\ Files/Graphviz/bin/dot.exe 2>/dev/null || true)
        if [ -n "$DOT" ]; then "$DOT" -Tsvg out/tree.dot -o out/tree.svg && echo "rendered out/tree.svg"; fi
        ;;
    *) sed -n '2,14p' "$0"; exit 1 ;;
esac
