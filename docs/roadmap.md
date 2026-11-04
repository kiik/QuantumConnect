# Roadmap: solution methods

Each solution method is its own self-contained bot class in
`arena/qc-starter/src/main/java/ch/qc/meelik/bots/`, and ships as its own upload JAR
(`qc-starter/build/libs/<Name>.jar`), packaged by the starter project's own build.gradle: that one
class and the bundled engine library. No helper classes and no superclass of ours: the arena ran
games for the starter's one-class samples, but none for our bot while it came with `Board` and
`Engine` classes. Paths and `./qc.sh` commands below are relative to `arena/`.

| Bot | Method | Status | Where |
|---|---|---|---|
| `QuarkBot` | Alpha-beta search, line weights evolved by self-play | Done; the upload | `main` |
| models in `params/` | QuarkBot's search with other weights: `training.txt` recipes refined from a model (`tl-*`), QuarkBot's own (`evolved-d4`) and the old hand-written weights (`handwritten`) | `./qc.sh train`; compared in the league | `main`, [params/](../arena/params/) |
| `QuarkV2Bot` | QuarkBot's method in a quarter of the code: alpha-beta over line scores, rescoring only the lines a drop touches; weights trained for its own search, the checkpoint picked against a field of bots | `./qc.sh train` (`quarkV2/*` recipes) | `main`, [params/quarkV2/](../arena/params/quarkV2/) |
| `ScrapBot` | The same code as QuarkV2Bot with `tl-d5-s6`'s weights fixed in: the league winner's weights as raw arithmetic | Done; a baseline | `main` |
| `ThreatBot` | Threat-sequence search for forced wins beyond alpha-beta's depth | Skeleton and design, still on the old `Engine` base class | branch `threat-search`, [docs/threat-search.md](threat-search.md) |
| `ParityBot` | Odd/even row rules (who wins the zugzwang endgame) in the evaluation | Skeleton and design, still on the old `Engine` base class | branch `parity-rules`, [docs/parity.md](parity.md) |
| `NetBot` | A small neural network as the evaluation | Planned | README, "Two approaches: simple and complex", step 2b |

## Inside QuarkBot

One class, in labelled sections:

```
rules, geometry, position    the game model: lines, threats, bombs, saving and restoring
the arena interface          accelerateQuark / useSpecialMove, cooldowns, board size
choosing an action           iterative deepening within the time budget
evaluation                   line values and bomb readiness
alpha-beta search            negamax with a transposition table and history move ordering
```

The tools use QuarkBot as their board model too (`new QuarkBot(rows, columns, ...)`), so the rules
are checked against the engine (`./qc.sh conformance`) on the very code that ships.

What a new method can change, from least to most invasive:

1. **Weights.** A params file, tried as `quark:params/<file>`. To ship it, write the numbers into
   `EVOLVED_LINE_WEIGHTS` and `EVOLVED_BOMB_READY_BONUS`.
2. **Evaluation.** Add terms to `evaluate()`, scored for the side to move.
3. **Search.** Change `chooseAction(budget)` to run another search before alpha-beta, or instead
   of it.

The shared building blocks for the threat-based methods are in the model. Construct it with
`tracksThreats = true` and it keeps `threatCount[player * cellCount + cell]` up to date: the number
of lines in which that cell is the only one the player still needs. `completesLine(player, column)`
tells whether a drop in that column wins now. The self-test checks both against actually playing
the drop.

## Adding a method

1. Branch from `main` and copy `bots/QuarkBot.java` to `bots/<Name>Bot.java`: one class that
   `implements Algorithm`, with a public no-arg constructor. Change it there.
2. `./qc.sh build` packages it as `qc-starter/build/libs/<Name>Bot.jar` automatically. `./qc.sh check`
   checks that the JAR holds a single class of ours, scans it for forbidden packages and plays one
   game with the JAR loaded on its own.
3. Add it to `league.txt` by short name (`<name>` resolves to `ch.qc.meelik.bots.<Name>Bot`), then
   run `./qc.sh stats league.txt 20` (ratings with error bars against everything in the field,
   including teammates' JARs and old snapshots in `Bots/`) or `./qc.sh arena <name> quark 40`.
   Before changing a bot that is in the field, `./qc.sh snapshot <Name>Bot` keeps its current
   version playing as a JAR.
4. Merge when `./qc.sh test` passes and the league shows an improvement over QuarkBot. A method
   that wins can replace QuarkBot's code, so that one bot keeps shipping.
