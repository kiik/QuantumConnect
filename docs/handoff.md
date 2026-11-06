# Handoff: SerenaBot at the end of the tournament day (2026-11-06)

Where the bot stands, what was measured, what is still open, and how to pick the work up again.
Paths and `./qc.sh` commands are relative to `arena/` unless they start at the repository root.

## The live bot

**WhiteRabbit-06-1150**, uploaded 11:56. Its JAR is byte-identical to Tank-06-1150, uploaded 11:50, which
was the first build with the threat-structure terms. Both JARs are in `jars/`, and `main` builds the same
bot. New uploads that bring measured gains are named `WhiteRabbit-DD-HHMM`. `./qc.sh tank` still writes
`Tank-*` and gates against the newest `jars/Tank-*.jar`, so rename the JAR after its gate.

Uploads were refused around 10:26 ("uploads are closed") and accepted again at 11:50. If an upload is
refused, retry later before concluding that submissions are over.

## SerenaBot's defaults on `main`

| Part | Default |
|---|---|
| Rules | `LINE_LENGTH = 6`, 15x30 board built before the clock, `budgetNanos` 40 ms of the 50 ms limit |
| Search | negamax with iterative deepening and a transposition table; forcing-line quiescence; `refinedSearch` (PVS, late move reductions, killers); logarithmic reductions (`reductionLogBase` 75, `reductionLogDivisor` 100); `bombHorizon` 2 (bomb threats below the root) |
| Evaluation | pattern table (trained for six in a row only); `bombBonusPercent` 150 at six in a row (600 at other N); **threat structure**: `lowestThreatWeight` 1500, `threatParityWeight` 1500, `deadThreatPercent` 50 |
| Loss book | 2044 entries keyed by board size, used only at six in a row |
| Off | `threatSpaceSearch`, `bombHoldPercent`, `varietyNoise`, the network |

## What reached its target

The target was a measured gain over the previous version on 15x30, with no late moves.

| Change | Evidence | Kept |
|---|---|---|
| Refined search + bomb threats below the root (Tank-06-0112) | +316 Elo, 300 games | yes |
| Bomb bonus 150% at six in a row | best of a 0..600 sweep | yes |
| Speed rewrite (Tank-06-0632) | 1.9x nodes/s, +73 Elo, 200 games | yes |
| Logarithmic late move reductions (Tank-06-0734) | +16 Elo, 300 games (+6..+27 over runs) | yes |
| speed2 | +6-7% nodes/s, identical play | yes |
| **Threat structure terms** (Tank-06-1150 = WhiteRabbit-06-1150) | **+64 Elo (95% +38..+87), 900 games**; the lowest-threat term alone +24 | yes |
| Loss book re-learned from the latest losses (Tank-06-1020, not uploaded) | 5-5 gate: neutral | yes |
| Threat-space search, bomb hold value | weaker in ablation | off |
| Bombs only from depth 3 | -82 Elo | not merged |
| Aspiration windows, counter moves and other search2 ideas | noise | not merged |

The evidence for the threat terms is `results/stats.md`, from a 12-thread run. That run had 11-16 late
moves for every bot, the unchanged one included, so the cause is the overloaded machine rather than the
change. Use at most 6 threads for runs whose timing matters.

## Why we still lost to the top bots

At 10:09 Tank-06-0734 was #4 with a 95-17 record. All 17 losses were to 1104-1-1105-4, GitHub CHET45,
Savi24, anysize11054 and coffeeAtMidNight, and against those five we were 18-17.

- **Depth.** A 1.5 s search sees a forced loss 10-14 plies before the end. Our 40 ms search reaches
  depth 7-9 and sees it 4-6 plies before.
- **An overconfident evaluation.** Two to six plies before a forced loss, our search still scored
  +20000..+40000. The evaluation valued every five by its pattern alone, whether or not the gap could
  ever be played. The threat-structure terms address this.
- **Not the cause:** bomb use (we bomb more in our wins than in our losses), timing (no skipped turns
  since 06:32) and repeated games (the top bots vary their moves, so a replay book cannot exploit them).
- **The opening comes from the loss book.** At the empty board the book overrides search, and the move
  changed with each re-learn (column 3, then column 5). The `bookMinQuarks` toggle on `cand/bench`
  would let search choose the opening.

## Open work: untested branches (on origin, not merged)

| Branch | What it holds | State |
|---|---|---|
| `cand/sharp` | threat memory; toggles `bombNearThreats` (below the root, only bombs near threats) and `threatExtension` | committed before its agent was stopped: no identity check or measurement |
| `cand/bench` | `./qc.sh agree`: 40 ms moves against a cached 2 s search on 790 positions from our losses to the top bots (base 63.9% agreement, 7.0% blunders; 100 ms gives 70.4% / 4.7%); toggle `bookMinQuarks` | tool works; the toggle drew 5-5 in a 10-game check |
| `cand/retune` | `cand/sharp` and `cand/bench` merged together | built by the verifier before it was stopped; not verified |
| other `cand/*` | earlier experiments (search, search2, depth, tss, bombhold, hold600, n5, n7, combo, anyrules, speed, speed2) | merged, or measured and rejected; kept for reference |

## Next steps, in order

1. **Verify against hard opponents, not only ourselves.** Self-play gains (+73..+316 Elo) did not carry
   over to the top five, where we stayed near 50%. Gauntlets should include `serena@200`, `serena@500`
   and a bomb-holding variant, plus `./qc.sh agree` as the cheap first screen.
2. **Verify across rules.** The board size, line length and time limit can change; `./rules.sh N LIMIT_MS
   ROWSxCOLUMNS` switches the bot. Test candidates on a matrix (8x12, 12x20, 15x30, 15x100; N 5, 6, 7 on
   the engines patched for that line length; 20, 50 and 100 ms). Prefer features that do not depend on
   the size: lines judged locally, parity measured from the board's bottom row, bomb value by how full
   the board is.
3. **Tune the evaluation on solver-quality labels.** Take positions from the top bots' archived games and
   our losses, weighted to the last 20 plies. Label them by a proof-number search where it can prove a
   result, else by a 2-5 s search score. Fit the pattern table, the threat terms and the bomb bonus to
   those labels; `./qc.sh dataset` and `tools/qc-net` already cover most of the pipeline.
4. **Gain depth in sharp positions.** In sharp positions the bomb-threat scan takes about 36% of CPU and
   blasts about 39%; the forcing line takes 60-85% of nodes. Start from `cand/sharp`, and check every
   change with `./qc.sh speed fixed` (identical play) and `./qc.sh speed timed` (depth).

## Not in the repository

- `tools/qc-site/.env`: the site credentials (`USER_NAME`, `USER_PSK`), needed by `qcsite upload`.
- `out/`: build output, including the engines patched to other line lengths (`out/engines/core-N.jar`)
  and the `./qc.sh agree` reference cache (`out/agree/`). Patch an engine by changing the two `bipush 6`
  instructions in `QuantumField` to N.
- The analysis scripts from this session (loss diagnosis, per-opponent records, bomb statistics). The
  committed tools cover the same ground: `./qc.sh learn`, `./qc.sh agree`, and `qcsite archive` plus the
  match JSON files in `arena/archive/`.
