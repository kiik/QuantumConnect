# CLAUDE.md

## Commits

- Never add a `Co-Authored-By` trailer (or any other AI attribution line) to commit messages.

## Project

QuantumConnect bot (Connect-4 with extra steps: 8×12 board, six in a row, 3×3 bomb). See README.md
for the rules, the tools and the results so far.

- Layout: the root stays clean: `arena/` holds the Java side (`qc-starter/`, `qc-trainer/`, `qc.sh`,
  `params/`, `jars/`, `results/`, `out/`, `league.txt`, `training.txt`); `tools/` holds non-Java
  helpers (`tools/qc-site/`: CLI and MCP server for the arena site; `tools/qc-net/`: trainer of
  SerenaBot's network, from `./qc.sh dataset`); `docs/` and `data/` stay at the
  root. Paths and `./qc.sh` commands below are relative to `arena/`.
- The bots live inside the starter project, in `qc-starter/src/main/java/ch/qc/meelik/bots/`;
  our tools live in `qc-trainer/`. There is no `src/`. Keep
  `qc-starter/build.gradle` the starter's except for the `Bot-Class` line and the labelled per-bot
  block at its end.
- A bot is one self-contained class in `bots/` that `implements Algorithm` itself: no helper
  classes, no superclass of ours. The arena ran games for the starter's one-class samples but none
  for our bot while it came with `Board` and `Engine` classes. Each bot ships as its own upload
  JAR, `qc-starter/build/libs/<Bot>.jar`: that one class plus the bundled engine library (Java 25).
  `QuarkBot` is the bot; a new solution method starts as a copy of it. See docs/roadmap.md.
- Bot code: no nested, anonymous or lambda classes, no method references, only java.lang and
  java.util, and none of the packages the arena forbids (java.io, java.net, reflection, ...).
  Lambdas compile to `java.lang.invoke`, which the sandbox may treat as reflection.
  `./qc.sh check qc-starter/build/libs/<Bot>.jar` scans for this, and for a single class.
- Code style: textbook-like. Classes open with a short explanation of their concepts, code is grouped
  into labelled sections, and names say what things are (`lineQuarks`, `bombCooldown`, not `cnt`, `wait`).
- Build on the qc-starter engine library (`qc-starter/libs/core-*.jar`): its API types, its
  `Experiment` as the referee of every local game. QuarkBot's own board model exists only because a
  search needs a fast model it can undo; its rules are the engine's, not the slides'. The tools use
  QuarkBot as their board. After touching rules in it, run `./qc.sh conformance`, which replays
  real engine games on it.
- Before uploading: `./qc.sh check qc-starter/build/libs/<Bot>.jar`, which also runs the JAR on its own.
- `./qc.sh build` then `./qc.sh test` after any change to bot code. Use `./qc.sh league` or
  `./qc.sh arena` to compare against the previous version before claiming an improvement.
- `out/` is build output and is not committed. Tuned parameter sets go in `params/` with their
  `#` provenance lines.
