# CLAUDE.md

## Commits

- Never add a `Co-Authored-By` trailer (or any other AI attribution line) to commit messages.

## Project

QuantumConnect bot (Connect-4 with extra steps: 8×12 board, six in a row, 3×3 bomb). See README.md
for the rules, the tools and the results so far.

- Each solution method is its own bot class in `src/bot/ch/qc/meelik/bots/`, extending
  `engine/Engine`, and ships as its own JAR (engine package plus the bot class). See docs/roadmap.md.
- Bot code (`src/bot`): no nested, anonymous or lambda classes, no method references, and none of the
  packages the arena forbids. Lambdas compile to `java.lang.invoke`, which the sandbox may treat as
  reflection. `./qc.sh check out/<Bot>.jar` scans for this.
- Code style: textbook-like. Classes open with a short explanation of their concepts, code is grouped
  into labelled sections, and names say what things are (`lineQuarks`, `bombCooldown`, not `cnt`, `wait`).
- The rules are the engine's (`qc-starter/libs/core-*.jar`, its `Experiment`), not the slides'. After
  touching rules in `Board`, run `./qc.sh referee conformance` (needs a JDK 25), which replays real
  engine games on `Board`.
- `./qc.sh build` then `./qc.sh test` after any change to bot code. Use `./qc.sh league` or
  `./qc.sh arena` to compare against the previous version before claiming an improvement.
- `out/` is build output and is not committed. Tuned parameter sets go in `params/` with their
  `#` provenance lines.
