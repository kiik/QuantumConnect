# CLAUDE.md

## Commits

- Never add a `Co-Authored-By` trailer (or any other AI attribution line) to commit messages.

## Project

QuantumConnect bot (Connect-4 with extra steps: 8×12 board, six in a row, 3×3 bomb). See README.md
for the rules, the tools and the results so far.

- The submitted bot must stay **one class**: `src/bot/ch/qc/meelik/QuarkBot.java`, no nested or
  anonymous classes, and none of the packages the arena forbids (`./qc.sh check` scans for them).
- `./qc.sh build` then `./qc.sh test` after any change to the bot. Use `./qc.sh league` or
  `./qc.sh arena` to compare against the previous version before claiming an improvement.
- `out/` is build output and is not committed. Tuned parameter sets go in `params/` with their
  `#` provenance lines.
