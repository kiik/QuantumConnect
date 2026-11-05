package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.bots.QuarkBot;
import ch.qc.meelik.bots.SerenaBot;

import java.util.Arrays;
import java.util.Random;

/**
 * Correctness checks for the engine, each against an independent, simpler computation:
 * <ol>
 *   <li>the incrementally updated board (score, wins, threats, hash) equals one rebuilt from
 *       scratch, through random games with bombs;
 *   <li>a threat count says a drop completes a line exactly when it does;
 *   <li>alpha-beta without the transposition table equals plain minimax at fixed depth, bombs
 *       included;
 *   <li>with the transposition table and the forced-move rules (threats tracked), exact solves of
 *       small boards equal plain solves without either;
 *   <li>search speed on the real board (a measurement, not a check);
 *   <li>SerenaBot: a bomb's predicted line ({@code blastFormsLine}) equals the bomb played;
 *   <li>SerenaBot: the threat analysis's wins, losses and ruled-out drops equal the replies played out.
 * </ol>
 *
 * <p>Usage: SelfTest, or SelfTest speed to measure only the speed, in a fresh JVM.
 */
public final class SelfTest {
    static int failures;

    public static void main(String[] args) {
        Random random = new Random(1);
        if (args.length > 0 && args[0].equals("speed")) { // speed alone, in a fresh JVM
            measureSpeed(random);
            return;
        }
        incrementalMatchesRebuild(random);
        threatsMatchDrops(random);
        alphaBetaMatchesMinimax(random, "6x7x4", 4, 150);
        alphaBetaMatchesMinimax(random, "5x6x4", 5, 100);
        tableSolvesMatchPlainSolves(random, "4x5x4-nobomb", 200);
        tableSolvesMatchPlainSolves(random, "5x5x4-nobomb", 60);
        measureSpeed(random);
        serenaBombLinesMatchBlasts(random);
        serenaThreatsMatchReplies(random);
        System.out.println(failures == 0 ? "ALL PASSED" : failures + " FAILURE(S)");
    }

    static void check(boolean ok, String what) {
        if (ok) return;
        failures++;
        System.out.println("  FAIL: " + what);
    }

    // ------------------------------------------------------------------ random positions

    /** A random legal action; bombs, when allowed, half the time. */
    static int randomAction(QuarkBot board, Random random) {
        int action;
        do action = QuarkBot.action(random.nextInt(board.columns), random.nextBoolean()); while (!board.isLegal(action));
        return action;
    }

    static void playRandomly(QuarkBot board, Random random, int maxPlies) {
        for (int i = 0; i < maxPlies && board.winner() < 0 && !board.isFull(); i++) board.play(randomAction(board, random));
    }

    static boolean isOver(QuarkBot board) {
        return board.winner() >= 0 || board.isFull();
    }

    // ------------------------------------------------------------------ 1. incremental state

    static void incrementalMatchesRebuild(Random random) {
        int actions = 0, bombs = 0;
        for (int game = 0; game < 300; game++) {
            QuarkBot board = new QuarkBot(8, 12, 6, true, true), rebuilt = new QuarkBot(8, 12, 6, true, true);
            while (!isOver(board)) {
                int action = randomAction(board, random);
                int[] before = board.saveState();
                long keyBefore = board.positionKey();
                board.play(action);
                actions++;
                if (QuarkBot.isBomb(action)) bombs++;

                rebuilt.load(Tools.field(board), SpaceOwnership.FIRST_PLAYER);
                rebuilt.sideToMove = board.sideToMove;
                rebuilt.bombCooldown[0] = board.bombCooldown[0];
                rebuilt.bombCooldown[1] = board.bombCooldown[1];
                check(rebuilt.staticScore == board.staticScore && rebuilt.positionKey() == board.positionKey()
                                && Arrays.equals(rebuilt.completedLines, board.completedLines) && Arrays.equals(rebuilt.threatCount, board.threatCount),
                        "incremental state after " + Tools.describeAction(action));
                for (int column = 0; column < board.columns; column++)
                    for (int height = board.columnHeight[column]; height < board.rows; height++)
                        check(board.owner[column * board.rows + height] == QuarkBot.EMPTY, "a quark floats above its column");

                board.restoreState(before);
                check(board.positionKey() == keyBefore, "restoring a saved state");
                board.play(action);
            }
        }
        System.out.printf("1. incremental == rebuilt: %d actions, %d of them bombs%n", actions, bombs);
    }

    // ------------------------------------------------------------------ 2. threats

    static void threatsMatchDrops(Random random) {
        int checked = 0, threats = 0;
        for (int game = 0; game < 300; game++) {
            QuarkBot board = new QuarkBot(8, 12, 6, true, true);
            while (!isOver(board)) {
                for (int column = 0; column < board.columns; column++) {
                    if (board.columnHeight[column] == board.rows) continue;
                    for (int player = 0; player < 2; player++) {
                        int[] before = board.saveState();
                        int mover = board.sideToMove;
                        board.sideToMove = player; // try the drop for either player
                        board.drop(column);
                        boolean completes = board.completedLines[player] > 0;
                        board.restoreState(before);
                        board.sideToMove = mover;
                        check(board.completesLine(player, column) == completes, "threat count for a drop in column " + column);
                        checked++;
                        if (completes) threats++;
                    }
                }
                board.play(randomAction(board, random));
            }
        }
        System.out.printf("2. threats == drops: %d drops checked, %d of them complete a line%n", checked, threats);
    }

    // ------------------------------------------------------------------ 3. alpha-beta against minimax

    static int minimax(QuarkBot engine, int depth, int ply) {
        QuarkBot board = engine;
        int winner = board.winner();
        if (winner >= 0) return winner == board.sideToMove ? QuarkBot.WIN - ply : ply - QuarkBot.WIN;
        if (board.isFull()) return 0;
        if (depth == 0) return engine.evaluate();
        int best = -QuarkBot.WIN - 1;
        for (int action = 0; action < board.actionCount(); action++) {
            if (!board.isLegal(action)) continue;
            int[] before = board.saveState();
            board.play(action);
            best = Math.max(best, -minimax(engine, depth - 1, ply + 1));
            board.restoreState(before);
        }
        return best;
    }

    static void alphaBetaMatchesMinimax(Random random, String geometry, int depth, int positions) {
        int withBombReady = 0;
        for (int i = 0; i < positions; i++) {
            QuarkBot engine = Tools.engine(geometry);
            engine.useTranspositionTable = false;
            engine.bombReadyBonus = 7; // exercise every evaluation term
            engine.lineWeight[QuarkBot.VERTICAL][2] = 3;
            playRandomly(engine, random, random.nextInt(engine.cellCount / 2));
            if (isOver(engine)) continue;
            // The first bomb comes after 10 turns, later than these short games reach: shorten the
            // wait to 0-2 turns, so that bombs are tried inside the tree.
            for (int player = 0; player < 2; player++) engine.bombCooldown[player] = random.nextInt(3);
            if (engine.bombCooldown[0] == 0 || engine.bombCooldown[1] == 0) withBombReady++;
            long key = engine.positionKey();
            int expected = minimax(engine, depth, 0), actual = engine.searchValue(depth);
            check(expected == actual && engine.positionKey() == key, geometry + " depth " + depth + ": minimax " + expected + " vs alpha-beta " + actual);
        }
        System.out.printf("3. alpha-beta == minimax on %s at depth %d: %d positions (%d with a bomb ready)%n", geometry, depth, positions, withBombReady);
    }

    // ------------------------------------------------------------------ 4. transposition table

    static void tableSolvesMatchPlainSolves(Random random, String geometry, int positions) {
        int wins = 0, draws = 0, losses = 0;
        for (int i = 0; i < positions; i++) {
            QuarkBot withTable = Tools.board(geometry, true); // tracking threats turns on the forced-move rules
            QuarkBot board = withTable;
            playRandomly(board, random, board.cellCount / 3 + random.nextInt(board.cellCount / 3));
            if (isOver(board)) continue;
            int remaining = board.cellCount - board.quarkCount;
            int solved = withTable.searchValue(remaining);

            QuarkBot plain = Tools.engine(geometry);
            plain.useTranspositionTable = false;
            plain.restoreState(board.saveState());
            int plainSolved = plain.searchValue(remaining);
            check(solved == plainSolved, geometry + " exact solve: with table " + solved + " vs without " + plainSolved);
            if (solved > 0) wins++;
            else if (solved == 0) draws++;
            else losses++;
        }
        System.out.printf("4. table + forced-move solves == plain solves on %s: %d wins, %d draws, %d losses for the side to move%n", geometry, wins, draws, losses);
    }

    // ------------------------------------------------------------------ 6. SerenaBot: bombs that form a line

    /** A random legal action of SerenaBot's side to move, bombs half the time when allowed; NO_ACTION on a full board. */
    static int randomSerenaAction(SerenaBot board, Random random) {
        boolean anyLegal = false;
        for (int column = 0; column < board.columns; column++) anyLegal |= board.isLegal(SerenaBot.action(column, false));
        if (!anyLegal) return SerenaBot.NO_ACTION;
        int action;
        do action = SerenaBot.action(random.nextInt(board.columns), random.nextInt(4) == 0); while (!board.isLegal(action));
        return action;
    }

    /** Random SerenaBot games on both board sizes: each position, every bomb's predicted line against the bomb played. */
    static void serenaBombLinesMatchBlasts(Random random) {
        int checked = 0, lines = 0;
        for (int game = 0; game < 120; game++) {
            SerenaBot board = game % 2 == 0 ? new SerenaBot(8, 12) : new SerenaBot(15, 30);
            int[] saved = new int[board.stateSize];
            for (int ply = 0; ply < 300 && !board.hasLine(); ply++) {
                for (int column = 0; column < board.columns; column++) {
                    if (!board.isLegal(SerenaBot.action(column, false))) continue;
                    boolean predicted = board.blastFormsLine(column);
                    board.saveState(saved);
                    board.play(SerenaBot.action(column, true));
                    boolean formed = board.hasLine();
                    board.restoreState(saved);
                    check(predicted == formed, board.rows + "x" + board.columns + " bomb in column " + column + ": predicted line " + predicted + ", played " + formed);
                    checked++;
                    if (formed) lines++;
                }
                int action = randomSerenaAction(board, random);
                if (action == SerenaBot.NO_ACTION) break;
                board.play(action);
            }
        }
        System.out.printf("6. SerenaBot bomb lines == bombs played: %d bombs checked, %d of them form a line%n", checked, lines);
    }

    // ------------------------------------------------------------------ 7. SerenaBot: the threat analysis

    /** Whether the side to move has an action after which a line stands (it wins at once). */
    static boolean winsAtOnce(SerenaBot board) {
        int[] saved = new int[board.stateSize];
        for (int action = 0; action < 2 * board.columns; action++) {
            if (!board.isLegal(action)) continue;
            board.saveState(saved);
            board.play(action);
            boolean line = board.hasLine();
            board.restoreState(saved);
            if (line) return true;
        }
        return false;
    }

    /** Whether this action of the side to move loses at once: it forms no line and the opponent then wins at once. */
    static boolean losesAtOnce(SerenaBot board, int action) {
        int[] saved = new int[board.stateSize];
        board.saveState(saved);
        board.play(action);
        boolean loses = !board.hasLine() && winsAtOnce(board);
        board.restoreState(saved);
        return loses;
    }

    /**
     * Random SerenaBot games with random bomb cooldowns: each claim of the threat analysis against the
     * replies played out. A win must be there at once; at a loss every action must lose at once; and
     * any drop the analysis rules out must lose at once.
     */
    static void serenaThreatsMatchReplies(Random random) {
        int positions = 0, wins = 0, losses = 0, forced = 0;
        for (int game = 0; game < 160; game++) {
            SerenaBot board = game % 4 == 3 ? new SerenaBot(15, 30) : new SerenaBot(8, 12);
            for (int ply = 0; ply < 300 && !board.hasLine(); ply++) {
                for (int player = 0; player < 2; player++) board.bombCooldown[player] = random.nextInt(3);
                int verdict = board.threatAnalysis(1);
                positions++;
                if (verdict == SerenaBot.WIN - 2) {
                    wins++;
                    check(winsAtOnce(board), "threat analysis: a win that is not there");
                } else if (verdict == 3 - SerenaBot.WIN) {
                    losses++;
                    for (int action = 0; action < 2 * board.columns; action++)
                        if (board.isLegal(action)) check(losesAtOnce(board, action), "threat analysis: a loss, but action " + action + " holds");
                } else if (board.forcedAnswers != SerenaBot.NOT_FORCED) {
                    forced++;
                    for (int column = 0; column < board.columns; column++) {
                        int drop = SerenaBot.action(column, false);
                        if (board.isLegal(drop) && !SerenaBot.allowsDrop(board.forcedAnswers, column))
                            check(losesAtOnce(board, drop), "threat analysis: ruled out the drop in column " + column + ", which holds");
                    }
                }
                int action = randomSerenaAction(board, random);
                if (action == SerenaBot.NO_ACTION) break;
                board.play(action);
            }
        }
        System.out.printf("7. SerenaBot threat analysis == replies: %d positions, %d wins, %d losses, %d with forced answers%n", positions, wins, losses, forced);
    }

    // ------------------------------------------------------------------ 5. speed

    static void measureSpeed(Random random) {
        long nodes = 0, nanos = 0;
        int depths = 0, positions = 0;
        for (int i = 0; i < 30; i++) {
            QuarkBot engine = new QuarkBot();
            playRandomly(engine, random, 4 + random.nextInt(30));
            if (isOver(engine)) continue;
            long started = System.nanoTime();
            engine.chooseAction(engine.budgetNanos);
            nanos += System.nanoTime() - started;
            nodes += engine.nodesSearched;
            depths += engine.lastDepth;
            positions++;
        }
        System.out.printf("5. speed on 8x12x6: %.0fk nodes/s, average depth %.1f in %.0f ms (%d positions)%n",
                nodes / (nanos / 1e6), (double) depths / positions, nanos / 1e6 / positions, positions);
    }
}
