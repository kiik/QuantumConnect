#!/usr/bin/env bash
# Switches SerenaBot to the announced rules and packages it for upload.
#   ./rules.sh N LIMIT_MS [ROWSxCOLUMNS]   e.g. ./rules.sh 5 50 15x30
# Sets LINE_LENGTH = N in SerenaBot and QuarkBot (QuarkBot is the tools' board) and SerenaBot's
# budgetNanos to 80% of the limit. Tests run on an engine patched to N (out/engines/core-N.jar) when
# N is not the stock engine's 6; the upload JAR is always built on the stock engine in qc-starter/libs,
# since the JAR bundles the engine library. Smoke games against random, and for N=6 a 10-game gate
# against the newest Tank JAR. Leaves the JAR as jars/Rules-N<N>-<ms>ms.jar; uploading is a separate step.
set -euo pipefail
cd "$(dirname "$0")"
N=${1:?N}; LIMIT_MS=${2:?limit in ms}; SIZE=${3:-15x30}
BOTS=qc-starter/src/main/java/ch/qc/meelik/bots
BUDGET_MS=$((LIMIT_MS * 8 / 10))

sed -i -E "s/public static final int LINE_LENGTH = [0-9]+;/public static final int LINE_LENGTH = $N;/" $BOTS/SerenaBot.java $BOTS/QuarkBot.java
sed -i -E "s/public long budgetNanos = [0-9_]+L;/public long budgetNanos = ${BUDGET_MS}_000_000L;/" $BOTS/SerenaBot.java
grep -n "LINE_LENGTH = \|public long budgetNanos" $BOTS/SerenaBot.java | head -3

TEST_ENGINE=""
if [ "$N" != 6 ]; then
    if [ -f out/engines/core-$N.jar ]; then TEST_ENGINE=out/engines/core-$N.jar
    else echo "no engine patched to $N: local games play six in a row, so only the build is checked"; fi
fi
if [ -n "$TEST_ENGINE" ] || [ "$N" = 6 ]; then
    ENGINE_JAR=${TEST_ENGINE:-} ./qc.sh build > /dev/null
    ENGINE_JAR=${TEST_ENGINE:-} ./qc.sh test | tail -1
    ENGINE_JAR=${TEST_ENGINE:-} ./qc.sh conformance 50 "$SIZE" | tail -2
    ENGINE_JAR=${TEST_ENGINE:-} ./qc.sh arena serena random 4 "$SIZE" | tail -3
fi

unset ENGINE_JAR
./qc.sh build > /dev/null
./qc.sh check qc-starter/build/libs/SerenaBot.jar | tail -2
OUT=jars/Rules-N$N-${LIMIT_MS}ms.jar
cp qc-starter/build/libs/SerenaBot.jar "$OUT"
if [ "$N" = 6 ]; then
    PREVIOUS=$(ls -t jars/Tank-*.jar | head -1)
    ./qc.sh arena "jar:$OUT" "jar:$PREVIOUS" 10 "$SIZE" | tail -4
fi
echo "-> $OUT; upload: cd ../tools/qc-site && uv run qcsite archive && uv run qcsite upload Rules-N$N-${LIMIT_MS}ms ../../arena/$OUT --yes"
