#!/usr/bin/env bash
# Build and run the bot and its tools.
#   ./qc.sh build                 compile everything; package every bot as qc-starter/build/libs/<Name>.jar
#   ./qc.sh jar <bot>             package one bot for upload, e.g. ./qc.sh jar QuarkBot
#   ./qc.sh check <jar>...        pre-upload check: manifest, size, forbidden packages
#   ./qc.sh test                  engine self-test (SelfTest)
#   ./qc.sh arena A B [games] [size]  headless matches, e.g. ./qc.sh arena quark random 20 25x25
#   ./qc.sh census [geom] [ply]   state-space census, e.g. ./qc.sh census 6x7x4-nobomb 9
#   ./qc.sh tree [moves] [plies] [evalDepth] [width] [geom]   tree + heatmap, renders out/tree.svg
#   ./qc.sh tune [gens] [depth] [pairs] [openings] [threads]  evolve eval weights -> out/tune-params.txt
#   ./qc.sh train [recipes]                  train every model in training.txt not yet in params/
#   ./qc.sh league [file] [games] [threads]  round robin of league.txt -> results/league.md, out/league.html
#   ./qc.sh replay [A] [B] [games]           record games -> out/replay.html (browser viewer)
#   ./qc.sh conformance [games] [size]       replay engine games on our Board: same boards, same winners
#   ./qc.sh watch [A] [B]                    one game in the engine's window, as qc-starter runs it
# The bots live in the starter project (qc-starter/src/main/java/ch/qc/meelik), whose build.gradle packages
# them; the tools live in qc-trainer. Both build on the engine library in qc-starter/libs, and every
# arena, league and replay game is played in its Experiment. That library is compiled for Java 25, so
# a JDK 25 or newer is needed: JAVA_HOME, else javac on PATH, else ~/.local/share/jdks/jdk-*, else a
# Windows JDK when running under WSL. Its logging API (slf4j) is downloaded into out/deps once.
set -euo pipefail
cd "$(dirname "$0")"

jdk_major() { "$1" -version 2>&1 | head -1 | grep -oE '[0-9]+' | head -1; }

# The first JDK 25+ among the candidates; prints its bin directory and the classpath separator.
find_jdk() {
    local candidates=()
    [ -n "${JAVA_HOME:-}" ] && candidates+=("$JAVA_HOME/bin")
    command -v javac >/dev/null && candidates+=("$(dirname "$(readlink -f "$(command -v javac)")")")
    for d in "$HOME"/.local/share/jdks/jdk-* /mnt/c/Program\ Files/Java/jdk-* /mnt/c/Program\ Files/Eclipse\ Adoptium/jdk-*; do
        [ -d "$d/bin" ] && candidates+=("$d/bin")
    done
    for bin in "${candidates[@]}"; do
        for ext in "" .exe; do
            [ -x "$bin/javac$ext" ] && [ "$(jdk_major "$bin/java$ext")" -ge 25 ] && { echo "$bin|$ext"; return; }
        done
    done
    echo "no JDK 25 or newer found (the engine library needs Java 25); set JAVA_HOME" >&2; exit 1
}
JDK=$(find_jdk); JDK_BIN=${JDK%|*}; EXT=${JDK#*|}
JAVAC=$JDK_BIN/javac$EXT; JAVA=$JDK_BIN/java$EXT; JAR=$JDK_BIN/jar$EXT
case "$EXT" in .exe) SEP=';' ;; *) SEP=':' ;; esac

ENGINE_JAR=$(ls qc-starter/libs/core-*.jar 2>/dev/null | head -1 || true)
[ -n "$ENGINE_JAR" ] || { echo "no engine library in qc-starter/libs" >&2; exit 1; }

# The engine logs through slf4j; slf4j-simple prints its warnings (timeouts, invalid columns).
SLF4J_VERSION=2.0.17
deps() {
    mkdir -p out/deps
    for artifact in slf4j-api slf4j-simple; do
        local file=out/deps/$artifact-$SLF4J_VERSION.jar
        [ -f "$file" ] || curl -sSfL -o "$file" "https://repo1.maven.org/maven2/org/slf4j/$artifact/$SLF4J_VERSION/$artifact-$SLF4J_VERSION.jar"
    done
}
ENGINE_CP="$ENGINE_JAR${SEP}out/deps/slf4j-api-$SLF4J_VERSION.jar${SEP}out/deps/slf4j-simple-$SLF4J_VERSION.jar"

BOTS=qc-starter/src/main/java/ch/qc/meelik
TRAINER=qc-trainer/src/main/java

build() {
    deps
    rm -rf out/bot out/tools
    "$JAVAC" --release 17 -Xlint:all -cp "$ENGINE_JAR" -d out/bot $(find "$BOTS" -name '*.java')
    "$JAVAC" --release 25 -cp "out/bot${SEP}$ENGINE_JAR" -d out/tools $(find "$TRAINER" -name '*.java')
    package botJars
}

# Upload JARs come from the starter's own build.gradle: qc-starter/build/libs/<Bot>.jar.
package() {
    [ -z "$EXT" ] || { echo "packaging runs Gradle, which needs a Linux JDK 25 under WSL" >&2; exit 1; }
    JAVA_HOME=${JDK_BIN%/bin} qc-starter/gradlew -p qc-starter --quiet --console=plain "$1" 2>&1 | grep -v '^WARNING: \|^$' || true
    for source in "$BOTS"/bots/*Bot.java; do
        local jar=qc-starter/build/libs/$(basename "$source" .java).jar
        [ "$1" = botJars ] || [ "$1" = "$(basename "$source" .java)Jar" ] || continue
        echo "built $jar ($(( $(wc -c < "$jar") / 1024 )) KB)"
    done
}

run() { deps; "$JAVA" -cp "out/bot${SEP}out/tools${SEP}$ENGINE_CP" "ch.qc.meelik.tools.$@"; }

cmd=${1:-build}; shift || true
case "$cmd" in
    build) build ;;
    jar) package "${1:?bot name, e.g. QuarkBot}Jar" ;;
    league)
        run League "$@"
        command -v wslpath >/dev/null && echo "open in a browser: $(wslpath -w out/league.html)"
        ;;
    check) run Check "$@" ;;
    replay)
        run Replay "$@"
        command -v wslpath >/dev/null && echo "open in a browser: $(wslpath -w out/replay.html)"
        ;;
    test) run SelfTest "$@" ;;
    conformance) run Referee conformance "$@" ;;
    watch) run Referee watch "$@" ;;
    arena) run Arena "$@" ;;
    census) run Census "$@" ;;
    tune) run Tune "$@" ;;
    train)
        grep -v '^[[:space:]]*\(#\|$\)' "${1:-training.txt}" | while read -r name gens depth pairs openings seed start; do
            if [ -f "params/$name.txt" ]; then echo "params/$name.txt exists; skipping"; continue; fi
            echo "training $name"
            run Tune "$gens" "$depth" "$pairs" "$openings" "$(nproc)" "$seed" "params/$name.txt" ${start:-defaults} < /dev/null | tee "params/$name.log"
        done
        ;;
    tree)
        run TreeViz "$@"
        DOT=$(command -v dot || ls /mnt/c/Program\ Files/Graphviz/bin/dot.exe 2>/dev/null || true)
        if [ -n "$DOT" ]; then "$DOT" -Tsvg out/tree.dot -o out/tree.svg && echo "rendered out/tree.svg"; fi
        ;;
    *) sed -n '2,20p' "$0"; exit 1 ;;
esac
