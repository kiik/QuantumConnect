# CLAUDE.md

## Commits

- Never add a `Co-Authored-By` trailer (or any other AI attribution line) to commit messages.

## Project

QuantumConnect bot (Connect-4 with extra steps: 8×12 board, six in a row, 3×3 bomb). See README.md
for the rules, the tools and the results so far.

- Layout: the bots live inside the starter project, in `qc-starter/src/main/java/ch/qc/meelik/`
  (`engine/`, `bots/`); our tools live in `qc-trainer/`. There is no root `src/`. Keep
  `qc-starter/build.gradle` the starter's except for the `Bot-Class` line and the labelled per-bot
  block at its end.
- Each solution method is its own bot class in `bots/`, extending `engine/Engine`, and ships as its
  own upload JAR, `qc-starter/build/libs/<Bot>.jar` (engine library bundled, Java 25). See
  docs/roadmap.md.
- Bot code: no nested, anonymous or lambda classes, no method references, and none of the packages
  the arena forbids. Lambdas compile to `java.lang.invoke`, which the sandbox may treat as
  reflection. `./qc.sh check qc-starter/build/libs/<Bot>.jar` scans for this.
- Code style: textbook-like. Classes open with a short explanation of their concepts, code is grouped
  into labelled sections, and names say what things are (`lineQuarks`, `bombCooldown`, not `cnt`, `wait`).
- Build on the qc-starter engine library (`qc-starter/libs/core-*.jar`): its API types, its
  `Experiment` as the referee of every local game. Our own `Board` exists only because a search
  needs a fast model it can undo; its rules are the engine's, not the slides'. After touching rules
  in `Board`, run `./qc.sh conformance`, which replays real engine games on `Board`.
- Before uploading: `./qc.sh check qc-starter/build/libs/<Bot>.jar`, which also runs the JAR on its own.
- `./qc.sh build` then `./qc.sh test` after any change to bot code. Use `./qc.sh league` or
  `./qc.sh arena` to compare against the previous version before claiming an improvement.
- `out/` is build output and is not committed. Tuned parameter sets go in `params/` with their
  `#` provenance lines.
