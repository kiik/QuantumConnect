#!/usr/bin/env bash
# Build and run the bot and its tools.
#   ./qc.sh build                 compile everything; package every bot as out/<Name>.jar
#   ./qc.sh jar <bot class>       package one bot, e.g. ./qc.sh jar ch.qc.meelik.bots.QuarkBot
#   ./qc.sh check <jar>...        pre-upload check: manifest, size, forbidden packages
#   ./qc.sh test                  engine self-test (SelfTest)
#   ./qc.sh arena A B [games]     headless matches, e.g. ./qc.sh arena quark random 20
#   ./qc.sh census [geom] [ply]   state-space census, e.g. ./qc.sh census 6x7x4-nobomb 9
#   ./qc.sh tree [moves] [plies] [evalDepth] [width] [geom]   tree + heatmap, renders out/tree.svg
#   ./qc.sh tune [gens] [depth] [pairs] [openings] [threads]  evolve eval weights -> out/tune-params.txt
#   ./qc.sh train [recipes]                  train every model in training.txt not yet in params/
#   ./qc.sh league [file] [games] [threads]  round robin of league.txt -> results/league.md, out/league.html
#   ./qc.sh replay [A] [B] [games]           record games -> out/replay.html (browser viewer)
#   ./qc.sh referee [A] [B] [games]          play in the engine's own Experiment (needs Java 25)
#   ./qc.sh watch [A] [B]                    one game in the engine's window, as qc-starter runs it
# The bot compiles against the real engine JAR (qc-starter/libs) when the JDK is 25 or newer, as the
# JAR is; otherwise against the stand-ins in src/stub. The tools always use the stand-ins.
# Set JAVA_HOME to pick a JDK; otherwise javac on PATH, then ~/.local/share/jdks/jdk-*, then a
# Windows JDK when running under WSL.
set -euo pipefail
cd "$(dirname "$0")"

find_bin() {
    if [ -n "${JAVA_HOME:-}" ]; then
        for f in "$JAVA_HOME/bin/$1" "$JAVA_HOME/bin/$1.exe"; do [ -x "$f" ] && { echo "$f"; return; }; done
    fi
    command -v "$1" 2>/dev/null && return
    for d in "$HOME"/.local/share/jdks/jdk-*; do [ -x "$d/bin/$1" ] && last="$d/bin/$1"; done
    [ -n "${last:-}" ] && echo "$last" && return
    for d in /mnt/c/Program\ Files/Java/jdk-* /mnt/c/Program\ Files/Eclipse\ Adoptium/jdk-*; do
        [ -x "$d/bin/$1.exe" ] && last="$d/bin/$1.exe"
    done
    [ -n "${last:-}" ] && echo "$last" && return
    echo "no $1 found; set JAVA_HOME" >&2; exit 1
}
JAVAC=$(find_bin javac); JAVA=$(find_bin java); JAR=$(find_bin jar)
case "$JAVAC" in *.exe) SEP=';' ;; *) SEP=':' ;; esac

ENGINE_JAR=$(ls qc-starter/libs/core-*.jar libs/core-*.jar 2>/dev/null | head -1 || true)
jdk_major() { "$1" -version 2>&1 | head -1 | grep -oE '[0-9]+' | head -1; }

build() {
    rm -rf out/stub out/bot out/tools
    "$JAVAC" --release 17 -d out/stub $(find src/stub -name '*.java')
    local cp=out/stub
    if [ -n "$ENGINE_JAR" ] && [ "$(jdk_major "$JAVAC")" -ge 25 ]; then
        cp=$ENGINE_JAR
        echo "bot compiled against the real engine: $ENGINE_JAR"
    else
        echo "bot compiled against the stand-ins (src/stub): the real engine JAR needs JDK 25+ (this is $(jdk_major "$JAVAC"))"
    fi
    "$JAVAC" --release 17 -Xlint:all -cp "$cp" -d out/bot $(find src/bot -name '*.java')
    "$JAVAC" --release 17 -cp "out/bot${SEP}out/stub" -d out/tools $(find src/tools -name '*.java')
    for bot in $(bot_classes); do package "$bot"; done
}

# The engine's Experiment, for ./qc.sh referee: Java 25, the engine JAR and the logging API it uses.
SLF4J_VERSION=2.0.17
referee() {
    [ -n "$ENGINE_JAR" ] || { echo "no engine JAR in qc-starter/libs" >&2; exit 1; }
    [ "$(jdk_major "$JAVA")" -ge 25 ] || { echo "the engine JAR needs Java 25; set JAVA_HOME to a JDK 25" >&2; exit 1; }
    mkdir -p out/deps
    for artifact in slf4j-api slf4j-simple; do
        local file=out/deps/$artifact-$SLF4J_VERSION.jar
        [ -f "$file" ] || curl -sSfL -o "$file" "https://repo1.maven.org/maven2/org/slf4j/$artifact/$SLF4J_VERSION/$artifact-$SLF4J_VERSION.jar"
    done
    "$JAVA" -cp "out/bot${SEP}out/tools${SEP}$ENGINE_JAR${SEP}out/deps/slf4j-api-$SLF4J_VERSION.jar${SEP}out/deps/slf4j-simple-$SLF4J_VERSION.jar" \
        ch.qc.meelik.tools.Referee "$@"
}

# Every class that extends Engine is a bot, e.g. ch.qc.meelik.bots.QuarkBot.
bot_classes() {
    grep -rl --include='*.java' 'extends Engine' src/bot | sed 's|^src/bot/||; s|\.java$||; s|/|.|g' | sort
}

# One JAR per bot: the engine package plus the bot's own class files, with Bot-Class set.
package() {
    local cls=$1 path=${1//.//} name=${1##*.}
    [ -f "out/bot/$path.class" ] || { echo "no compiled bot $cls (run ./qc.sh build)" >&2; exit 1; }
    printf 'Bot-Class: %s\n' "$cls" > "out/$name.manifest"
    local files; files=$(cd out/bot && ls ch/qc/meelik/engine/*.class "$path.class" "$path"\$*.class 2>/dev/null || true)
    "$JAR" --create --file "out/$name.jar" --manifest "out/$name.manifest" $(for f in $files; do printf -- '-C out/bot %s ' "$f"; done)
    echo "built out/$name.jar: $(echo $files | sed 's|ch/qc/meelik/||g')"
}

run() { "$JAVA" -cp "out/bot${SEP}out/tools${SEP}out/stub" "ch.qc.meelik.tools.$@"; }

cmd=${1:-build}; shift || true
case "$cmd" in
    build) build ;;
    jar) package "${1:?bot class}" ;;
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
    referee) referee "$@" ;;
    watch) referee watch "$@" ;;
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
