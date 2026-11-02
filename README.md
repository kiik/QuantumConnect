# QuantumConnect

Meelik's try. Rules source: [info-session slides](<Data/QuantumConnect - Particle Collision Challenge v2.pdf>).

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

1. The arena calls `accelerateQuark(field)`. You have **100 ms**.
2. You return a column `0..11`. The column must not be full.
3. The quark falls to the lowest free row in that column.
4. **Win check.** If there are six in a row, the match ends here.
5. The arena calls `useSpecialMove(field)` (also 100 ms). Both methods are called every turn.
6. The other player moves. This repeats until the field is full.

### The special move (bomb)

- Return `true` from `useSpecialMove` to detonate at the quark you just dropped.
- Everything in the 3×3 area around it is destroyed, including that quark and your own quarks.
- **Gravity applies:** quarks above the blast fall down to fill the gaps.
- Cooldown: once every **10 of your own moves**. The slides say "count before you detonate", so plan on tracking it yourself.
- A bomb can erase an opponent's 5-in-a-row. It can also erase your own.

### How a match ends

- **Win:** six connected quarks horizontally, vertically or diagonally ("the experiment collapses").
- **Draw:** the field is full and nobody has six.
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

- If you exceed 100 ms or return a full column, **your turn is skipped**. This doesn't crash the
  match, but an uncaught exception can end it.
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
| `getMoveHistory()` | `List<Move>` |

Coordinates: **row 0 is the top**, column 0 is the left.

- **Allowed:** the Java standard library for computation (collections, arrays, math, streams),
  the engine JAR in `libs/`, and your own helper classes. The build uses Gradle with Java 25.
- **Forbidden:** `java.net`, `java.io`, `java.lang.reflect`, `sun.*`, `com.sun.*`, `java.security`,
  `javax.crypto`, `java.lang.Thread*`, `java.util.concurrent.*`, and process or system access.
  That means **search single-threaded, with no background thinking**.
- Every match creates a new instance, so don't cache state across games.
- Local testing: in the starter project (`qc-starter`), edit `EmulatorRunner` to run
  `new Experiment(new MyBot(), new RandomBot())`, then `./gradlew run`. For fast runs without the
  window use `new Experiment(a, b, false, false)`. The target is > 90% wins against `RandomBot`.
- Packaging: set `'Bot-Class'` in the `jar.manifest` block of `build.gradle`, run `./gradlew build`,
  then upload `build/libs/quantumconnect-starter-1.0-SNAPSHOT.jar`.

## This repo

| Path | What |
|---|---|
| [src/bot/ch/qc/meelik/engine/Board.java](src/bot/ch/qc/meelik/engine/Board.java) | **The game model:** rules, the line counts behind wins, scores and threats, and bombs. No search. |
| [src/bot/ch/qc/meelik/engine/Engine.java](src/bot/ch/qc/meelik/engine/Engine.java) | **The shared brain:** alpha-beta search and the arena adapter. The base class of every bot. |
| [src/bot/ch/qc/meelik/bots/](src/bot/ch/qc/meelik/bots/) | **The bots, one class per solution method:** `QuarkBot` (simple), `EvolvedBot` (evolved weights). Each ships as its own JAR, `out/<Name>.jar`, holding the engine package plus that class. |
| [docs/roadmap.md](docs/roadmap.md) | The solution methods, their status and branches, and how to add one. |
| [src/stub/](src/stub/) | Stand-ins for the engine API (`Algorithm`, `QuantumField`, `SpaceOwnership`, `Move`), written from the slides. They are replaced by the real engine JAR once it is in `libs/`. |
| [src/tools/](src/tools/) | Tools for analysis and testing. They never go in the JAR. |
| [src/tools/replay.html](src/tools/replay.html) | The replay viewer template. `./qc.sh replay` writes the recorded games into it. |
| [params/](params/) | Evaluation parameter sets from `./qc.sh tune`, versioned. Each starts with `#` lines saying how it was made. |
| [league.txt](league.txt) | The entrants of the local league. |
| [results/league.md](results/league.md) | Latest league standings, committed so everyone sees them. |
| [qc.sh](qc.sh) | Builds the bot and runs the tools. Uses `JAVA_HOME`, then `javac` on PATH, then a Windows JDK when running under WSL. |

**How the bots work:**
- **Board.** Columns are stacks of quarks. Every possible line of 6 cells (134 of them) keeps a count of each player's quarks. Dropping or removing a quark updates only the lines through that cell. That gives the win test (a count reaching 6), the evaluation, and optionally the threats (cells that would complete a line).
- **Memory.** About 8 MB in total, almost all of it the search's position hash table. The limit is roughly 100 MB.
- **Evaluation.** A line that only one player occupies is worth `16, 64, 256, 1024, 8192` points for 1–5 quarks in it (`Board.lineWeight`, per direction, so it can be tuned).
- **Search.** Negamax with alpha-beta pruning, deepening one ply at a time within a 40 ms budget per move. It uses a 512k-entry hash table of positions it has already searched and orders moves by which ones caused cut-offs before.
- **Moves.** Each move is a column plus whether to bomb, so the bomb is planned in `accelerateQuark` and only reported in `useSpecialMove`.
- **Cooldowns.** The bot counts its own bomb cooldown. It works out the opponent's by comparing the board with what it expected after its own turn: one new quark means a plain move, a bigger change means a bomb.
- **Assumptions,** each a single constant or method:
  - the bomb is ready from the first move (`Board.FIRST_BOMB_WAIT`);
  - a line formed by quarks falling after a blast wins (`Board.winner()`);
  - if both players have a line, the last mover wins.

**Commands:**

```sh
./qc.sh build                        # compile; one JAR per bot: out/QuarkBot.jar, out/EvolvedBot.jar, ...
./qc.sh test                         # engine self-test
./qc.sh arena quark random 20        # headless matches (bot specs below)
./qc.sh replay quark random 2        # record games -> out/replay.html, a viewer with play/pause, speed, scrubbing
./qc.sh league                       # round robin of league.txt -> results/league.md
./qc.sh tune 150 4                   # evolve the scoring weights -> out/tune-params.txt
./qc.sh census 6x7x4-nobomb 10       # distinct positions per ply; checked against the known Connect-4 counts
./qc.sh tree 5,6,5 2 6 3             # board, per-column scores, best line, game tree -> out/tree.svg
./qc.sh jar ch.qc.meelik.bots.QuarkBot  # package one bot for upload -> out/QuarkBot.jar
./qc.sh check out/QuarkBot.jar       # pre-upload check: manifest, size, forbidden packages
```

**Bot specs** (used by `arena`, `replay` and `league.txt`):
- `random`: a random legal column, never bombs.
- A bot name: `quark`, `evolved`, ... (the class `ch.qc.meelik.bots.<Name>Bot`), optionally with
  `@<ms>` (time budget per move), `/<depth>` (fixed search depth), or `:<params file>`
  (evaluation weights), e.g. `quark@10`, `quark/2`, `quark:params/evolved-d4.txt`.
- `tuned`: the latest `out/tune-params.txt`.
- A fully qualified class name: any other bot compiled in this repo.
- `jar:<path>`: a packaged bot JAR, loaded in its own class loader, so two versions of `QuarkBot`
  can play each other.

**The replay viewer:** the bots don't run in the browser. Java plays the games (the same code as
`arena`), records every move, and writes them into a self-contained HTML page that only plays them
back. For every engine bot, including JAR versions, each move also shows the search depth, score, nodes
and time. A link like `replay.html#g=1&ply=40` opens that game at that ply. Keys: space plays or
pauses, ← and → step, Home and End jump to the start and end.

**Results so far** (laptop, Windows JDK 19, stand-in engine):

- **Self-test:** passes.
  - The incremental board state matches a full rebuild over 26k random actions, including 2.6k blasts.
  - Alpha-beta matches plain minimax on 250 positions where bombs are available.
  - Exact solves agree with and without the hash table.
  - Speed: about 5M nodes/s, reaching depth 7–8 in 40 ms.
- **Census:**
  - Connect-4 (`6x7x4-nobomb`) reproduces the published counts (OEIS A212693) exactly through ply 10.
  - On the real board after 6 plies: 238,492 distinct positions without bombs and 638,594 with them. The branching factor is about 7 per ply with bombs and 5.6 without, against 3.9 for Connect-4.
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
- **Timing:** the worst move was 42 ms when games ran one at a time, but up to 66–79 ms when
  several ran in parallel. The 100 ms limit has less margin than it looks if the arena machine is
  busy. Consider lowering `budgetNanos` from 40 ms.

The local arena uses the bot's own rule model as the referee, so it can't catch a misread rule. Before uploading, check against the starter project's `Experiment`.

### Working on several solutions (multi-developer)

- **Git.** Each variant gets its own branch, and `main` always holds a bot that is safe to upload.
  `out/` is ignored and rebuilt by `./qc.sh build`.
- **Each solution method is its own bot class** in `src/bot/ch/qc/meelik/bots/`, extending the
  shared `Engine`, and its own JAR. `./qc.sh build` packages every bot. See
  [docs/roadmap.md](docs/roadmap.md) for the methods, their branches, and the hooks a bot can use.
- **A set of tuned weights** is a file in `params/`, made by `./qc.sh tune` and copied out of `out/`
  with its `#` provenance lines. Try it with `quark:params/<file>`. To ship it, write the numbers
  into a bot class as `EvolvedBot` does, because a bot can't read files at runtime.
- **Compare in the league.** Add a line to `league.txt`, then run `./qc.sh league`, which plays a
  round robin ranked by wins and then draws, as the real tournament is. Commit
  `results/league.md` along with the change that moved the standings.
- **Before merging:** `./qc.sh test` passes, the league shows no regression, and the bot has been
  checked once in the starter project's `Experiment`.

### Two approaches: simple and complex

Both use the same search. They differ only in how a position is scored.

1. **Simple (`QuarkBot`, shipping):** hand-written line weights, the same in every direction
   (`16, 64, 256, 1024, 8192`). Always keep this version uploaded.
2. **Complex: a learned score, in two steps.**
   - **a. Evolved weights (`EvolvedBot`, made with `./qc.sh tune`; 65% against QuarkBot, see the results above).** There are 16 parameters: line weights for each
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

- **More state.** You need each player's cooldown (0–9 own moves, so ×100) and the side to move
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
  columns. The search has to re-check lines after each collapse.

### Open questions (not answered by the slides)

1. Is the bomb available from your first move, or only after 10 own moves? Do skipped turns count
   toward the cooldown? What happens if you return `true` while it's on cooldown? (Probably ignored.)
2. Is the win checked again after a blast collapses the board? If the collapse forms six in a row for
   one or both players, who wins?
3. What is the result when the 30 s or 1000-move limit is hit?
4. Does `Move` in `getMoveHistory()` record bombs? If it does, you can work out the opponent's cooldown.
5. Is the blast clipped at the board edges? (Assumed yes.)
