# QuantumConnect

Meelik's try. Rules sources: the [info-session slides](<Data/QuantumConnect - Particle Collision Challenge v2.pdf>)
and the engine itself (`qc-starter/libs/core-2.0-SNAPSHOT.jar`), whose `Experiment` decides every
detail the slides leave open. Where they differ, this README follows the engine.

## The challenge

Write a bot in Java, upload it as a JAR, and let it fight.

- One shared 8×12 "quantum" field (12 columns, numbered 0–11)
- Connect **6 quarks** to win
- No ML required, no external services: one Java class, one JAR

If you can write a loop and compare numbers, you can compete.

**In short: this is Connect-4 with extra steps.** Same gravity-drop game, but:

| | Connect-4 | QuantumConnect |
|---|---|---|
| Board (rows × columns) | 6 × 7 | 8 × 12 |
| In a row to win | 4 | **6** |
| Extra move | none | **bomb**: clears the 3×3 around your quark, once every 10 of your own moves |

## Tournament format

- Teams of 2–4, **one bot per team**. Re-uploading replaces the bot and deletes its old matches.
- Round robin: every bot plays every other bot **7 games**, alternating who starts.
- Ranking: wins first, then draws.
- Submissions are open in the evenings after lectures. On **Friday** (deadline: noon) they are open all day.
  The leaderboard freezes early Friday morning and may be hidden at other times.

## Rules

### A turn

1. The arena calls `accelerateQuark(field)`. You have **100 ms**. You return a column `0..11`
   that is not full.
2. The arena calls `useSpecialMove(field)` (also 100 ms), on the same field: **nothing has been
   placed yet**. Both methods are called every turn.
3. If you asked for the bomb and may use it, the quark drops and blasts (below). Otherwise it
   falls to the lowest free row in that column.
4. **Win check.** If there are six in a row anywhere on the board, the player who just moved wins
   and the match ends.
5. The other player moves. This repeats until the field is full or 1000 turns have been played.

### The special move (bomb)

- Return `true` from `useSpecialMove` to detonate at the quark you are dropping this turn.
- Everything in the 3×3 area around it is destroyed (clipped at the edges), including that quark
  and your own quarks. The bomb is decided before the drop, so it blasts even a quark that would
  have won.
- **Gravity applies:** quarks above the blast fall down to fill the gaps.
- **Cooldown:** the engine counts your turns since your last bomb, starting from 0 at the start of
  the match, and allows a bomb at 10. So your first bomb can come on your 11th turn, and after that
  on every 10th turn. Every turn counts, also one that was skipped. Asking for a bomb on cooldown
  just gives a plain drop. The engine doesn't tell you the counter, so track it yourself.
- A bomb can erase an opponent's 5-in-a-row. It can also erase your own.
- If quarks falling after a blast form six in a row, even the **opponent's**, the bomber wins: the
  win goes to the player who just moved.

### How a match ends

- **Win:** six connected quarks horizontally, vertically or diagonally appear after your turn ("the
  experiment collapses").
- **Draw** ("uncertain"): the field is full and nobody has six, or 1000 turns (both players
  together) have been played.
- **Loss:** the bot uses a forbidden package, uses reflection, or breaks the sandbox.

### Limits

| Limit | Value |
|---|---|
| Time per move | 100 ms (aim for < 10 ms; the arena machine may be slower than your laptop) |
| Time per match | 30 s |
| Moves per match | 1000 |
| Memory | about 100 MB of RAM (told to us; not in the slides) |
| JAR size | 5 MB |
| Bot name | 5–20 characters, unique this season |

- If you exceed 100 ms or return a full or invalid column, the engine logs an error and **your
  turn is skipped**: nothing is placed and the turn still counts toward your bomb cooldown. (The
  starter README says you *lose* the game for a full column. The local engine only skips the turn,
  but don't rely on it.)
- Every move is logged. stdout is captured, so don't print inside the search.

## Building the bot

```java
public class MyBot implements Algorithm {           // ch.cern.quantumconnect.core.Algorithm
    public MyBot() { }                              // public no-arg: arena uses newInstance()
    @Override public int accelerateQuark(QuantumField field) { ... }  // required
    @Override public boolean useSpecialMove(QuantumField field) { ... } // optional; without it you never bomb
}
```

`QuantumField` API:

| Method | Returns |
|---|---|
| `getNumberOfRows()` / `getNumberOfColumns()` | `int`: 8 / 12 |
| `isColumnFull(col)` | `boolean` |
| `isSpaceEmpty(row, col)` | `boolean` |
| `getField()` | `SpaceOwnership[][]`, a **copy** (`EMPTY`, `FIRST_PLAYER`, `SECOND_PLAYER`) |
| `belongsToMe(row, col)` | `boolean`, relative to you; use it instead of assuming you are first or second |
| `getMoveHistory()` | `List<Move>`, one `Move(isSpecialMove, column, player)` per quark placed (a skipped turn adds none) |

Coordinates: **row 0 is the top**, column 0 is the left.

- **Allowed:** the Java standard library for computation (collections, arrays, math, streams),
  the engine JAR in `libs/`, and your own helper classes. The build uses Gradle with Java 25.
- **Forbidden:** `java.net`, `java.io`, `java.lang.reflect`, `sun.*`, `com.sun.*`, `java.security`,
  `javax.crypto`, `java.lang.Thread*`, `java.util.concurrent.*`, and process or system access.
  That means **search single-threaded, with no background thinking**.
- Every match should create a new instance, but the local `Experiment.restart()` reuses the old
  ones, so reset per-match state when a new game starts.
- Local testing: in the starter project ([qc-starter/](arena/qc-starter/)), edit `EmulatorRunner` to run
  `new Experiment(new MyBot(), new RandomBot())`, then `./gradlew run`. For fast runs without the
  window use `new Experiment(a, b, false, false)`. The target is > 90% wins against `RandomBot`.
  `./qc.sh watch` does this for our bots, and `./qc.sh arena` plays the same `Experiment` headless.
- Packaging: set `'Bot-Class'` in the `jar.manifest` block of `build.gradle`, run `./gradlew build`,
  then upload `build/libs/quantumconnect-starter-1.0-SNAPSHOT.jar`. That JAR bundles the whole
  runtime classpath (the engine, slf4j, logback) next to the bot. Our bots live in the starter
  project itself, but its build.gradle no longer makes that all-in-one JAR: it packages every bot
  on its own, as `build/libs/<Name>.jar`: the one bot class, the same bundled libraries and
  `Bot-Class` set to that bot. A bot is one self-contained class, like the starter's samples: the
  arena ran games for those, but none for our bot while it came with helper classes (`Board`,
  `Engine`). The upload only reads the JAR, so the package name is ours to choose; `ch.qc.meelik`
  keeps it apart from other teams' classes.

## This repo

The Java side, the bots and everything `qc.sh` builds, runs and writes, lives in `arena/`. Paths in
the table and the commands below are relative to it: `cd arena` first, or call `arena/qc.sh`.

| Path | What |
|---|---|
| [arena/qc-starter/](arena/qc-starter/) | **The starter project, with our bots in it.** The organisers' project (engine library `libs/core-*.jar`, Gradle wrapper, `EmulatorRunner`) plus our sources under `src/main/java/ch/qc/meelik/`. Its `build.gradle` is the starter's with one change, a labelled block at the end: instead of the starter's all-in-one JAR it builds one upload JAR per bot, `build/libs/<Name>.jar`, for our bots and for the starter's samples (`RandomBot.jar`, `SpecialMoveBot.jar`). To take a new starter version, copy `src/main/java/ch/qc/meelik/` and that block into it, as the starter's README says. |
| [qc-starter/src/main/java/ch/qc/meelik/bots/QuarkBot.java](arena/qc-starter/src/main/java/ch/qc/meelik/bots/QuarkBot.java) | **The bot, one self-contained class:** the game model (rules, the line counts behind wins, scores and threats, bombs), the alpha-beta search and the arena interface, with the evolved line weights. The tools use it as their board too. It ships as `qc-starter/build/libs/QuarkBot.jar`: that one class and the bundled libraries. |
| [docs/roadmap.md](docs/roadmap.md) | The solution methods, their status and branches, and how to add one. |
| [qc-trainer/](arena/qc-trainer/) | **Our own tools** for training, testing and analysis (arena, league, tuner, replay, conformance, the pre-upload check). They never go in an upload JAR. Its Gradle build compiles them against the starter project, for the IDE; `./qc.sh` builds and runs them with plain `javac`. |
| [qc-trainer/src/main/resources/replay.html](arena/qc-trainer/src/main/resources/replay.html) | The replay viewer template. `./qc.sh replay` and `./qc.sh league` write the recorded games into it. |
| [training.txt](arena/training.txt) | The model recipes. `./qc.sh train` trains every recipe that isn't in `params/` yet. |
| [params/](arena/params/) | The trained models: evaluation weights, one file per model, versioned, with the training log beside each. Each file starts with `#` lines saying how it was made. |
| [league.txt](arena/league.txt) | The entrants of the local league and of `./qc.sh stats`. The line `params/` enters every model, the line `Bots/` every bot JAR in it. |
| [Bots/](arena/Bots/) | Bot JARs from outside the build: teammates' bots (also when mailed as `.jar.zip`) and snapshots of our own older versions (`./qc.sh snapshot`). |
| [results/league.md](arena/results/league.md) | Latest league standings, committed so everyone sees them. `results/stats.md` is the same for `./qc.sh stats`. |
| [qc.sh](arena/qc.sh) | Builds the bots and runs the tools. Needs a JDK 25: `JAVA_HOME`, then `javac` on PATH, then `~/.local/share/jdks/jdk-*`. |

**How the bots work:**
- **Board.** Columns are stacks of quarks. Every possible line of 6 cells (134 of them) keeps a count of each player's quarks. Dropping or removing a quark updates only the lines through that cell. That gives the win test (a count reaching 6), the evaluation, and optionally the threats (cells that would complete a line).
- **Memory.** About 8 MB in total, almost all of it the search's position hash table. The limit is roughly 100 MB.
- **Evaluation.** A line that only one player occupies is worth points by the number of its quarks, per direction (`QuarkBot.lineWeight`). The weights were evolved by self-play (`QuarkBot.EVOLVED_LINE_WEIGHTS`, from `params/evolved-d4.txt`). The hand-written ones they replaced, `16, 64, 256, 1024, 8192` for 1–5 quarks, are `params/handwritten.txt`.
- **Search.** Negamax with alpha-beta pruning, deepening one ply at a time within a 40 ms budget per move. It uses a 512k-entry hash table of positions it has already searched and orders moves by which ones caused cut-offs before.
- **Moves.** Each move is a column plus whether to bomb, so the bomb is planned in `accelerateQuark` and only reported in `useSpecialMove`.
- **Board size.** The engine's `Experiment` takes the size as a parameter (8×12 by default; its own `Main` uses 25×25), so the bot reads it from the field: on a field of another size it rebuilds its board at that size, with the same weights (`QuarkBot.fitBoardTo`). Six in a row wins on every size. The weights were tuned on 8×12: on 25×25, the hand-written weights beat the evolved ones 3–1 in four games.
- **Cooldowns.** The bot counts its own bomb cooldown, and the opponent's from `getMoveHistory()`, which marks every bomb.
- **Why our own board model:** a search tries thousands of moves per turn and takes them back. The
  engine's `QuantumField` can't do that for a bot: its constructor and its drop and blast methods
  aren't public, bots may not use reflection, and it has no undo. So QuarkBot models the same rules
  for the search, and everything else (the API types, the referee, the packaging) is the engine's.
- **Rules,** checked against the engine: `./qc.sh conformance` has the engine's bombing bot
  play itself and replays every game on QuarkBot's model. All final boards and winners must be
  identical. Every arena, league and replay game is checked the same way.
  The engine's rules are in `QuarkBot.FIRST_BOMB_WAIT`, `QuarkBot.play` (a bomb always blasts),
  `QuarkBot.winner()` (any line wins for the last mover) and `QuarkBot.skipTurn()` (counts toward
  the cooldown).

**Commands:**

```sh
./qc.sh build                        # compile; one upload JAR per bot: qc-starter/build/libs/QuarkBot.jar, ...
./qc.sh test                         # engine self-test
./qc.sh arena quark random 20        # headless matches in the engine's Experiment (bot specs below)
./qc.sh arena quark random 4 25x25   # the same on another board size (rows x columns)
./qc.sh replay quark random 2        # record games -> out/replay.html, a viewer with play/pause, speed, scrubbing
./qc.sh train                        # train the models in training.txt -> params/<name>.txt
./qc.sh league league.txt 2          # every bot plays every other twice, swapping sides
                                     #   -> results/league.md and out/league.html (all games, highscores)
./qc.sh stats league.txt 10          # many games: tops every pairing up to 10, logged in out/stats/games.tsv
                                     #   -> ratings with error bars in results/stats.md, charts in out/stats.html
./qc.sh stats league.txt 10 8 4      #   the same with 8 threads and 4 random opening plies (always distinct games)
./qc.sh snapshot QuarkBot            # freeze the working tree's QuarkBot as Bots/QuarkBot-<commit>[-wip<time>].jar
./qc.sh snapshot QuarkBot 239fe55    #   or the one from any commit, to keep old versions in the field
./qc.sh tank                         # SerenaBot revision: archive, learn, build, check, gate -> Bots/Tank-DD-HHMM.jar
./qc.sh tune 150 4                   # evolve the scoring weights once -> out/tune-params.txt
./qc.sh census 6x7x4-nobomb 10       # distinct positions per ply; checked against the known Connect-4 counts
./qc.sh tree 5,6,5 2 6 3             # board, per-column scores, best line, game tree -> out/tree.svg
./qc.sh jar QuarkBot                 # package one bot for upload -> qc-starter/build/libs/QuarkBot.jar
./qc.sh check qc-starter/build/libs/QuarkBot.jar
                                     # pre-upload check: manifest, engine bundled, one class of ours,
                                     #   forbidden packages, and one game with the JAR loaded on its own
./qc.sh arena jar:qc-starter/build/libs/QuarkBot.jar ch.cern.quantumconnect.bots.LinearBot 10
                                     # against the engine's own bots
./qc.sh conformance 300              # replay engine games on our model: same boards, same winners
./qc.sh conformance 40 25x25         #   on another board size
./qc.sh watch quark:params/tl-d5-s6.txt quark:params/tl-d4-s5.txt
                                     # one game in the engine's window, as qc-starter shows it
                                     #   (on WSL: sudo apt install libxi6 libxtst6 fontconfig)
```

**Java and downloads:** the engine library is compiled for Java 25, so everything needs a JDK 25.
The first `./qc.sh build` downloads the logging API the engine uses (slf4j, into `out/deps`), and
the starter's Gradle wrapper downloads Gradle and logback (into `~/.gradle`).

**Bot specs** (used by `arena`, `replay` and `league.txt`):
- `random`: a random legal column, never bombs.
- A bot name: `quark`, ... (the class `ch.qc.meelik.bots.<Name>Bot`), optionally with
  `@<ms>` (time budget per move), `/<depth>` (fixed search depth), or `:<params file>`
  (evaluation weights), e.g. `quark@10`, `quark/2`, `quark:params/evolved-d4.txt`.
- `tuned`: the latest `out/tune-params.txt`.
- A fully qualified class name: any other bot compiled in this repo.
- `jar:<path>`: a packaged bot JAR, loaded in its own class loader, so two versions of `QuarkBot`
  can play each other.
- In `league.txt` only: a directory ending in `/`. Every params file in it joins as a model, and
  every bot JAR in it as `jar:`.

**The replay viewer:** the bots don't run in the browser. Java plays the games (the same code as
`arena`), records every move, and writes them into a self-contained HTML page that only plays them
back. For every engine bot, including JAR versions, each move also shows the search depth, score, nodes
and time. A link like `replay.html#g=1&ply=40` opens that game at that ply. Keys: space plays or
pauses, ← and → step, Home and End jump to the start and end.

The league's page, `out/league.html`, holds every game of the round robin, with three more cards:
- **Highscores:** the bots ranked like the tournament (wins, then draws), with points and score %.
- **By approach:** each solution method's score in its games against *other* methods, and its best
  bot, so you can see which method wins and not only which model.
- **All matches:** a crosstable with one chip per game (W, ½, L for the row bot; first as X, then
  as O). Click a chip to watch that game; click a bot to list only its games.

**Statistics over many games** (`./qc.sh stats`): the league above ranks one round robin; this
tool keeps playing and pools everything it ever played.
- **Every game is logged** in `out/stats/games.tsv`, one line each. A run only tops each pairing up to
  the games asked for, so `./qc.sh stats league.txt 20` after a run at 10 plays 10 more per pairing,
  and a new bot plays only its own pairings. Ctrl-C keeps every finished game.
- **Bots are logged by fingerprint,** a hash of their class, JAR or params file and spec options.
  Change QuarkBot's code and its old games stop counting for it; they stay in the log. To keep a
  version in the field, `./qc.sh snapshot QuarkBot` first. That puts a JAR in `Bots/`, which
  `league.txt` enters. `./qc.sh snapshot QuarkBot <commit>` builds the bot from any commit, also
  from the older `src/bot/` layout, against today's engine. Snapshots are for local games, not for
  upload: they don't bundle the engine.
- **The rating** is an Elo difference to `quark` (QuarkBot as shipped), fitted to all games at once,
  so a bot gets credit for beating strong bots and not only many. Its 95% interval comes from
  refitting on resampled games. While two bots' intervals overlap a lot, play more games.
- **`out/stats.html`** has the ratings with intervals, wins/draws/losses, a head-to-head grid (hover
  a cell for the record, a ± interval and the chance that the row bot is stronger), game lengths,
  and each game's slowest answer against the 100 ms limit, with the late moves counted.
- **Distinct games:** a deterministic bot (fixed depth, no clock) repeats the same game, and copies
  prove nothing. The page counts distinct games per pairing. The fourth argument plays that many
  random opening plies first, the same opening with both colours. Games with and without openings
  are kept apart.

**Results so far** (laptop; Windows JDK 19 and our own referee, before the switch to the engine, unless noted):

- **Self-test:** passes.
  - The incremental board state matches a full rebuild over 26k random actions, including 2.6k blasts.
  - Alpha-beta matches plain minimax on 250 positions where bombs are available.
  - Exact solves agree with and without the hash table.
  - Speed: about 5M nodes/s, reaching depth 7–8 in 40 ms.
- **Census:**
  - Connect-4 (`6x7x4-nobomb`) reproduces the published counts (OEIS A212693) exactly through ply 10.
  - On the real board after 6 plies: 238,492 distinct positions, a branching factor of about 5.6
    per ply against 3.9 for Connect-4. Bombs make no difference this early, because the first
    one can come only at ply 21. (Under the earlier guess that bombs are ready from the first
    move, it was 638,594.)
- **Arena,** 20 games each:

  | Opponent | Wins | Draws | Losses |
  |---|---|---|---|
  | `random` | 20 | 0 | 0 |
  | `quark/2` (searches only 2 plies) | 16 | 4 | 0 |
  | `quark@5` (5 ms per move) | 7 | 12 | 1 |

  Average 37 ms per move, worst 42 ms, no skipped turns. Bombs stretch games to about 130 plies.
- **Evolved weights (`params/evolved-d4.txt`):**
  - **Tuning,** 150 generations at depth 4, about 4 minutes on 16 threads: 62.5% against the
    defaults at depth 4. Single quarks and pairs in horizontal lines count for much more than the
    defaults, four-in-a-line counts for less, and the bomb-ready bonus rose to 20.
  - **Head to head at the real time limit,** 40 games: **14 wins, 24 draws, 2 losses** against the
    defaults, a 65% score.
  - **League,** 16 games per pair: evolved finished first and beat the defaults 11–5. Ranking by
    wins alone hides the difference, because both beat the weaker bots equally. That's why
    `results/league.md` also has a head-to-head table.
  - Tuned before the rules were checked against the engine (bombs ready from the first move).
  - **Re-measured on the engine's rules** (2026-11-03, JDK 25, 200 games, QuarkBot against
    `params/handwritten.txt`): **57 wins, 107 draws, 36 losses**, a 55% score. That is smaller than
    the 65% above, but clear of chance (about 2 standard deviations), so QuarkBot ships these weights.
- **Models from `training.txt`** (engine rules, JDK 25), 30–60 generations each: 55–60% against
  the defaults when tuned from the defaults (`es-*`), 59–65% when started from `evolved-d4` (`tl-*`).
- **League** of 14 bots, 2 games per pair (182 games, 91 s), see `results/league.md`:
  - `tl-d5-s6` first (14 W, 7 D, 5 L), `tl-d4-s4` second, `evolved-d4` third. All ten models
    finished above the hand-written `quark`, which came 12th.
  - Not a strict ladder: `es-d3-s1` lost 0–2 to the winner but beat `tl-d4-s4` 2–0.
  - `tl-d4-s5` lost no game (9 W, 17 D).
  - Pruned 2026-11-04: the `es-*` models and `tl-d4-s4`, which `results/stats.md` (34 games each)
    put within noise of `quark`. They are in git history; `params/` keeps `evolved-d4`, `tl-d4-s5`,
    `tl-d5-s6` and `handwritten`.
- **In the engine's own `Experiment`:** EvolvedBot against QuarkBot 3 W, 3 D,
  0 L; it won every game against the engine's `LinearBot` and `SpecialMoveBot`. No timeouts.
  `./qc.sh conformance`: 300 engine games replayed on our model (28,098 moves, 2,449
  bombs) with identical boards and winners.
- **Timing:** the worst move was 42 ms when games ran one at a time, but up to 66–79 ms when
  several ran in parallel. The 100 ms limit has less margin than it looks if the arena machine is
  busy. Consider lowering `budgetNanos` from 40 ms.

The local arena plays in the engine's own `Experiment`, so local results follow the real rules and time limit.

### Working on several solutions (multi-developer)

- **Git.** Each variant gets its own branch, and `main` always holds a bot that is safe to upload.
  `out/` is ignored and rebuilt by `./qc.sh build`.
- **Each solution method is its own self-contained bot class** in
  `qc-starter/src/main/java/ch/qc/meelik/bots/`, and its own JAR. `./qc.sh build` packages every
  bot. See [docs/roadmap.md](docs/roadmap.md) for the methods and their branches.
- **A model** (a set of tuned weights) is a file in `params/`. To add one, add a recipe line to
  `training.txt` and run `./qc.sh train`. It trains only the recipes that have no file yet and
  writes `params/<name>.txt`, with its `#` provenance lines, and `params/<name>.log`. Commit both.
  Recipes can start from another model, which is transfer learning. Try a model with
  `quark:params/<file>`. To ship it, write the numbers into `QuarkBot.EVOLVED_LINE_WEIGHTS` and
  `EVOLVED_BOMB_READY_BONUS`, because a bot can't read files at runtime.
- **Compare in the league.** `./qc.sh league league.txt 2` plays every pair twice with sides
  swapped, ranked by wins and then draws as the real tournament is. It enters every model in
  `params/` automatically. Commit `results/league.md` along with the change that moved the
  standings, and open `out/league.html` to watch the games.
- **Before merging:** `./qc.sh test` passes, the league shows no regression, and the bot has been
  checked once in the starter project's `Experiment`.

### Two approaches: simple and complex

Both use the same search. They differ only in how a position is scored.

1. **Simple:** hand-written line weights, the same in every direction (`16, 64, 256, 1024,
   8192`). QuarkBot shipped with these until 2026-11-03; they are now `params/handwritten.txt`.
2. **Complex: a learned score, in two steps.**
   - **a. Evolved weights (in `QuarkBot` since 2026-11-03, made with `./qc.sh tune`; see the results above).** There are 16 parameters: line weights for each
     direction (horizontal, vertical, diagonal) and quark count, plus a bonus for having the bomb ready
     while the opponent doesn't. They are evolved by self-play: each generation tries small random
     changes in opposite directions against each other, from random openings and with both colours,
     and the parameters move toward whichever side scored better. Every 10 generations the current
     parameters play the defaults on 100 fixed openings (200 games). The best are saved to
     `out/tune-params.txt`, and `./qc.sh arena tuned quark` tests them at the real time limit.
     The tuning games use a fixed search depth of 4, which is faster than the real 40 ms budget.
   - **b. A small neural network as the score (planned).** Its first layer would be updated as each
     quark is added or removed, the same way the line counts are now, so search speed holds up.
     Trained by the same self-play, plus positions labelled with exact results near the end of the
     game and with deeper offline searches. No solved positions exist for this board, and the solved
     Connect-4 results don't carry over. The bot can't read files, so the weights must be written into
     the class as constants. A small net (a few thousand weights) fits easily in the
     100 MB limit. Too many weights written as an array would exceed Java's 64 KB limit on a method,
     so store them encoded in string constants and decode them at startup.

   The slides say ML is "not required (and discouraged)". Check with the organisers before
   submitting 2b.

## Analysis: state space vs. Connect-4

### Connect-4 baseline

- Gravity means every position is a set of column stacks. A column of height `h` has `2^h`
  colourings, so per column `2^0 + … + 2^6 = 127` → at most `127^7 ≈ 5.3·10^14` boards.
  Requiring the piece counts to differ by 0 or 1 gives `≈ 7.1·10^13`. Excluding positions that
  can't be reached because the game already ended gives `≈ 4.5·10^12` (Tromp). The game is solved:
  the first player wins.
- Standard encoding: a bitboard of 7 columns × (6 + 1 sentinel) = 49 bits in one `long`.
  `position + mask` is a unique key because stacks have no holes. Side to move = piece-count parity.
- 69 winning lines, at most 42 plies, at most 7 moves per ply, left/right mirror symmetry.

### QuantumConnect without the bomb

The same construction, scaled up:

- Per column `2^0 + … + 2^8 = 511` → at most `511^12 ≈ 3.2·10^32` boards, `≈ 2.7·10^31` with balanced
  piece counts. That is roughly 10^18 times Connect-4, so solving it is out of reach. The practical
  option is search plus a heuristic within 100 ms.
- Bitboard: 12 columns × (8 + 1) = 108 bits, which does **not** fit in one `long`. Either use two
  `long`s (emulating 128-bit shifts) or a per-column byte array plus precomputed line masks.
- With column stride 9, the shift directions are: vertical `1`, horizontal `9`, diagonals `8` and `10`.
  A 6-in-a-row test for a direction `d`:
  `m = b & (b >> d); m4 = m & (m >> 2d); win = (m4 & (m >> 4d)) != 0`.
- 134 winning lines (56 horizontal, 36 vertical, 42 diagonal), at most 96 plies, at most 12 moves
  per ply, and the same mirror symmetry.

### What the bomb adds

Gravity applies after a blast, so **the board is still a set of column stacks**. The `511^12` bound,
the stack bitboard and the `position + mask` key all still work. What changes:

- **More state.** You need each player's cooldown (0–10 own turns, so ×121) and the side to move
  stored explicitly. Bombs remove quarks and skipped turns break parity, so piece counts no longer
  tell you whose turn it is.
- **The transition is still cheap.** Say your quark lands at height `h` in column `c`. Column `c`
  loses its top 2 cells: the quark and the one under it. Columns `c±1` lose cells `h-1..h+1`, and
  whatever was above them shifts down. On a bitboard, that is a mask-and-shift on 3 columns.
- **A bomb is a net removal.** Your own quark is always destroyed, so a bomb turn takes 1 to 8
  quarks off the board and places none. Its value is purely what it erases: an opponent threat, or
  a cell that sets up a line.
- **Bigger branching.** A move is (column, bomb?), so up to 24 options per ply. The bomb option is
  only available about 1 turn in 10.
- **Length.** Bombs free up cells, so the game isn't capped at 96 plies and positions can repeat.
  The arena caps a match at 1000 moves and 30 s.
- **Collapse effects.** Quarks falling after a blast can form new lines, or break them, in all 3
  columns. The search has to re-check lines after each collapse. Any line formed wins for the
  bomber, so a blast that drops the opponent's quarks into a line is a winning move too.
- **Not in the opening.** No bomb comes before ply 20 (10 turns each), so the first 20 plies are
  plain Connect-6 on 8×12.

### Rule questions, answered by the engine

The slides left these open. The engine's `Experiment` (core-2.0) answers them, and
`./qc.sh conformance` checks that our model (in `QuarkBot`) agrees:

1. **When is the bomb ready?** After 10 of your own turns, then every 10th turn. Skipped turns
   count. A bomb asked for on cooldown is a plain drop.
2. **Is the win checked after a blast?** Yes: after every turn, anywhere on the board. Any line
   wins for the player who just moved, even if the line is the opponent's.
3. **What happens at 1000 turns?** A draw. The engine has no 30 s match limit; the tournament
   server may have one.
4. **Does `getMoveHistory()` record bombs?** Yes: `Move(isSpecialMove, column, player)`, one per
   quark placed. That is how the engine tracks the opponent's cooldown.
5. **Is the blast clipped at the edges?** Yes.

Still open: whether the server treats a full column as a skipped turn, as the engine does, or as
a loss, as the starter README says, and whether it enforces a 30 s match limit.
