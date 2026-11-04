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
#   ./qc.sh tune --islands N --migrate K [--generations g] [--population p] [--openings o] [--size 15x30] [--depth 0]
#                                            island evolution of SerenaBot -> out/serena-islands.txt, results/islands.tsv
#   ./qc.sh train [recipes]                  train every model in training.txt not yet in params/
#   ./qc.sh league [file] [games] [threads]  round robin of league.txt -> results/league.md, out/league.html
#   ./qc.sh stats [file] [games] [threads] [opening plies] [size]
#                                            many games, logged in out/stats/games.tsv -> ratings, out/stats.html
#   ./qc.sh learn <losses file>... [--budget ms] [--window moves] [--min-gap score]
#                                            SerenaBot's loss book from lost games (qcsite archive-losses)
#                                            -> params/serena/book.txt and SerenaBot.java's LOSS_BOOK
#   ./qc.sh snapshot <Bot> [git revision]    freeze a bot as Bots/<Bot>-<revision>.jar, to keep it in the field
#   ./qc.sh tank [learn options]             SerenaBot's revision loop: archive, learn, build, test, check, gate
#                                            vs the newest Bots/Tank-*.jar on 15x30 (GATE_GAMES, default 10),
#                                            -> Bots/Tank-DD-HHMM.jar, params/serena/revisions.tsv; prints the upload
#   ./qc.sh replay [A] [B] [games]           record games -> out/replay.html (browser viewer)
#   ./qc.sh dataset [out dir]               archived matches -> input planes + results under out/dataset/
#   ./qc.sh network [every n] [budget ms]    SerenaBot's network: incremental == rebuilt, nodes/s with and without
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
    for source in "$BOTS"/bots/*Bot.java qc-starter/src/main/java/ch/qc/starter/*Bot.java; do
        local jar=qc-starter/build/libs/$(basename "$source" .java).jar
        [ "$1" = botJars ] || [ "$1" = "$(basename "$source" .java)Jar" ] || continue
        echo "built $jar ($(( $(wc -c < "$jar") / 1024 )) KB)"
    done
}

# A bot as it is now (working tree) or was at a git revision, as a JAR the arena loads on its own:
# Bots/<Bot>-<revision>.jar. It holds our classes only, compiled against today's engine; the arena
# supplies the engine, so it is for local games, not for upload. Old layouts (src/bot/) work too.
snapshot() {
    local bot=${1:?bot class, e.g. QuarkBot} rev=${2:-} work=out/snapshot tag
    rm -rf "$work"; mkdir -p "$work/src" "$work/classes" Bots
    if [ -z "$rev" ]; then
        cp -r "$BOTS" "$work/src/"
        tag=$(git rev-parse --short HEAD)
        [ -z "$(git status --porcelain -- "$BOTS")" ] || tag=$tag-wip$(date +%m%d%H%M)
    else
        local dirs; dirs=$(git ls-tree -d --name-only "$rev" -- qc-starter/src/main/java/ch/qc/meelik src/bot)
        [ -n "$dirs" ] || { echo "no bot sources at $rev" >&2; exit 1; }
        git archive "$rev" -- $dirs | tar -x -C "$work/src"
        tag=$(git rev-parse --short "$rev")
    fi
    "$JAVAC" --release 17 -nowarn -cp "$ENGINE_JAR" -d "$work/classes" $(find "$work/src" -name '*.java' -not -path '*/tools/*')
    local class; class=$(cd "$work/classes" && find . -name "$bot.class" | head -1)
    [ -n "$class" ] || { echo "no class $bot at ${rev:-the working tree}" >&2; exit 1; }
    class=${class#./}; class=${class%.class}
    printf 'Bot-Class: %s\n' "${class//\//.}" > "$work/manifest.txt"
    "$JAR" cfm "Bots/$bot-$tag.jar" "$work/manifest.txt" -C "$work/classes" .
    echo "built Bots/$bot-$tag.jar (Bot-Class ${class//\//.})"
}

# SerenaBot's revision loop. Archives the site's matches (before an upload deletes ours), learns from
# the archived losses of our bots and the top three, rebuilds and checks the bot, plays a short gate
# against the newest Tank on the tournament board and snapshots the upload JAR as Bots/Tank-DD-HHMM.jar.
# Each revision is a row of params/serena/revisions.tsv, each gate one of results/gates.tsv. It prints
# the upload command and never runs it: the user decides what enters the tournament.
tank() {
    local name; name=Tank-$(date +%d-%H%M)
    local site=(uv run --quiet --project ../tools/qc-site qcsite) jar=qc-starter/build/libs/SerenaBot.jar games=${GATE_GAMES:-10}
    mkdir -p out/serena Bots
    "${site[@]}" archive || echo "archive failed: learning from the matches archived so far" >&2
    "${site[@]}" archive-losses --loser 'Tank-*' --loser 'SerenaBot-*' --loser 'QuarkBot-*' --loser top:3 -o out/serena/losses.txt
    run Learn out/serena/losses.txt "$@"
    build
    run SelfTest
    run Check "$jar" | tee out/serena/check.txt
    local size; size=$(grep -o 'size: .*' out/serena/check.txt)
    field() { echo "$size" | sed -n "s/.* $1=\([0-9]*\).*/\1/p"; }

    local previous score=- results=-; previous=$(ls -t Bots/Tank-*.jar 2>/dev/null | head -1 || true)
    if [ -n "$previous" ]; then
        run Arena "jar:$jar" "jar:$previous" "$games" 15x30 | tee out/serena/gate.txt
        results=$(sed -n 's/.* wins \([0-9]*\), draws \([0-9]*\), .* wins \([0-9]*\), avg.*/\1 \2 \3/p' out/serena/gate.txt)
        score=$(echo "$results" | awk -v n="$games" '{ printf "%.3f", ($1 + $2 / 2) / n }')
        [ -f results/gates.tsv ] || printf '%s\n' "time	candidate	opponent	games	wins	draws	losses	score	size" \
            "# ./qc.sh tank: the candidate SerenaBot upload JAR vs the newest Bots/Tank-*.jar, starts alternating, engine Experiment, 15x30" > results/gates.tsv
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$name" "$(basename "$previous" .jar)" "$games" "${results// /	}" "$score	15x30" >> results/gates.tsv
    else
        echo "no Bots/Tank-*.jar yet: no gate"
    fi

    cp "$jar" "Bots/$name.jar"
    local rev; rev=$(git rev-parse --short HEAD)
    [ -z "$(git status --porcelain -- "$BOTS")" ] || rev=$rev-wip
    [ -f params/serena/revisions.tsv ] || printf '%s\n' "name	git_rev	book_entries	gate_games	gate_score	class_kb	jar_kb	heap_mb" \
        "# ./qc.sh tank: one row per SerenaBot revision; gate vs the newest Tank before it on 15x30, sizes from ./qc.sh check" > params/serena/revisions.tsv
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rev" "$(field book)" "$([ -n "$previous" ] && echo "$games" || echo 0)" "$score" \
        "$(field class)" "$(field jar)" "$(field heap)" >> params/serena/revisions.tsv
    echo "$name: gate $score vs ${previous:-nothing}; $size -> Bots/$name.jar, params/serena/revisions.tsv"
    echo "to upload (archive first; replaces our bot), from the repository root:"
    echo "  cd tools/qc-site && uv run qcsite archive && uv run qcsite upload $name ../../arena/Bots/$name.jar --yes"
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
    stats)
        run Stats "$@"
        command -v wslpath >/dev/null && echo "open in a browser: $(wslpath -w out/stats.html)"
        ;;
    snapshot) snapshot "$@" ;;
    tank) tank "$@" ;;
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
    learn) run Learn "$@" ;;
    dataset) run Network dataset "$@" ;;
    network) run Network bench "$@" ;;
    train)
        grep -v '^[[:space:]]*\(#\|$\)' "${1:-training.txt}" | while read -r name gens depth pairs openings seed start bot field; do
            if [ -f "params/$name.txt" ]; then echo "params/$name.txt exists; skipping"; continue; fi
            echo "training $name"
            mkdir -p "$(dirname "params/$name.txt")"
            run Tune "$gens" "$depth" "$pairs" "$openings" "$(nproc)" "$seed" "params/$name.txt" ${start:-defaults} ${bot:-quark} ${field:-none} < /dev/null | tee "params/$name.log"
        done
        ;;
    tree)
        run TreeViz "$@"
        DOT=$(command -v dot || ls /mnt/c/Program\ Files/Graphviz/bin/dot.exe 2>/dev/null || true)
        if [ -n "$DOT" ]; then "$DOT" -Tsvg out/tree.dot -o out/tree.svg && echo "rendered out/tree.svg"; fi
        ;;
    *) sed -n '2,33p' "$0"; exit 1 ;;
esac
