# Roadmap: solution methods

Each solution method is its own bot class in `qc-starter/src/main/java/ch/qc/meelik/bots/`,
extending the shared `Engine`, and ships as its own upload JAR (`qc-starter/build/libs/<Name>.jar`), packaged by the
starter project's own build.gradle: the bot class, the `engine` package and the bundled engine library.

| Bot | Method | Status | Where |
|---|---|---|---|
| `QuarkBot` | Alpha-beta search, hand-written line weights | Done; the safe upload | `main` |
| `EvolvedBot` | Same search, line weights evolved by self-play | Done; 65% against QuarkBot | `main` |
| models in `params/` | QuarkBot's search with weights from `training.txt` recipes: evolved from the defaults (`es-*`) or refined from a model (`tl-*`) | `./qc.sh train`; compared in the league | `main`, [params/](../params/) |
| `ThreatBot` | Threat-sequence search for forced wins beyond alpha-beta's depth | Skeleton and design | branch `threat-search`, [docs/threat-search.md](threat-search.md) |
| `ParityBot` | Odd/even row rules (who wins the zugzwang endgame) in the evaluation | Skeleton and design | branch `parity-rules`, [docs/parity.md](parity.md) |
| `NetBot` | A small neural network as the evaluation | Planned | README, "Two approaches: simple and complex", step 2b |

## Modules

```
engine/Board    the game model: rules, lines, threats, bombs (no search)
engine/Engine   alpha-beta search and the arena adapter; the base class of every bot
bots/*Bot       one class per method: settings, overridden hooks, extra searches
```

What a bot can change, from least to most invasive:

1. **Weights.** Call `board.setLineWeights(...)` and set `bombReadyBonus` in the constructor
   (EvolvedBot does this).
2. **Evaluation.** Override `extraEvaluation()` to add knowledge, scored for the side to move.
3. **Search.** Override `chooseAction(budget)` to run another search before alpha-beta, or
   instead of it.

The shared building blocks for the threat-based methods are in `Board`. Construct the board with
`tracksThreats = true` and it keeps `threatCount[player * cellCount + cell]` up to date: the number
of lines in which that cell is the only one the player still needs. `completesLine(player, column)`
tells whether a drop in that column wins now. The self-test checks both against actually playing
the drop.

## Adding a method

1. Branch from `main` and add `bots/<Name>Bot.java` with a public no-arg constructor.
2. `./qc.sh build` packages it as `qc-starter/build/libs/<Name>Bot.jar` automatically. `./qc.sh check` scans the JAR
   for forbidden packages and plays one game with the JAR loaded on its own.
3. Add it to `league.txt` by short name (`<name>` resolves to `ch.qc.meelik.bots.<Name>Bot`), then
   run `./qc.sh league` and `./qc.sh arena <name> quark 40`.
4. Merge when `./qc.sh test` passes and the league shows an improvement over QuarkBot.
