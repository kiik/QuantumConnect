package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.bots.QuarkBot;
import ch.qc.meelik.engine.Board;
import ch.qc.meelik.engine.Engine;

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
 *   <li>with the transposition table, exact solves of small boards equal solves without it;
 *   <li>search speed on the real board (a measurement, not a check).
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
        System.out.println(failures == 0 ? "ALL PASSED" : failures + " FAILURE(S)");
    }

    static void check(boolean ok, String what) {
        if (ok) return;
        failures++;
        System.out.println("  FAIL: " + what);
    }

    // ------------------------------------------------------------------ random positions

    /** A random legal action; bombs, when allowed, half the time. */
    static int randomAction(Board board, Random random) {
        int action;
        do action = Board.action(random.nextInt(board.columns), random.nextBoolean()); while (!board.isLegal(action));
        return action;
    }

    static void playRandomly(Board board, Random random, int maxPlies) {
        for (int i = 0; i < maxPlies && board.winner() < 0 && !board.isFull(); i++) board.play(randomAction(board, random));
    }

    static boolean isOver(Board board) {
        return board.winner() >= 0 || board.isFull();
    }

    // ------------------------------------------------------------------ 1. incremental state

    static void incrementalMatchesRebuild(Random random) {
        int actions = 0, bombs = 0;
        for (int game = 0; game < 300; game++) {
            Board board = new Board(8, 12, 6, true, true), rebuilt = new Board(8, 12, 6, true, true);
            while (!isOver(board)) {
                int action = randomAction(board, random);
                int[] before = board.saveState();
                long keyBefore = board.positionKey();
                board.play(action);
                actions++;
                if (Board.isBomb(action)) bombs++;

                rebuilt.load(Tools.field(board), SpaceOwnership.FIRST_PLAYER);
                rebuilt.sideToMove = board.sideToMove;
                rebuilt.bombCooldown[0] = board.bombCooldown[0];
                rebuilt.bombCooldown[1] = board.bombCooldown[1];
                check(rebuilt.staticScore == board.staticScore && rebuilt.positionKey() == board.positionKey()
                                && Arrays.equals(rebuilt.completedLines, board.completedLines) && Arrays.equals(rebuilt.threatCount, board.threatCount),
                        "incremental state after " + Tools.describeAction(action));
                for (int column = 0; column < board.columns; column++)
                    for (int height = board.columnHeight[column]; height < board.rows; height++)
                        check(board.owner[column * board.rows + height] == Board.EMPTY, "a quark floats above its column");

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
            Board board = new Board(8, 12, 6, true, true);
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

    static int minimax(Engine engine, int depth, int ply) {
        Board board = engine.board;
        int winner = board.winner();
        if (winner >= 0) return winner == board.sideToMove ? Engine.WIN - ply : ply - Engine.WIN;
        if (board.isFull()) return 0;
        if (depth == 0) return engine.evaluate();
        int best = -Engine.WIN - 1;
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
            Engine engine = Tools.engine(geometry);
            engine.useTranspositionTable = false;
            engine.bombReadyBonus = 7; // exercise every evaluation term
            engine.board.lineWeight[Board.VERTICAL][2] = 3;
            playRandomly(engine.board, random, random.nextInt(engine.board.cellCount / 2));
            if (isOver(engine.board)) continue;
            // The first bomb comes after 10 turns, later than these short games reach: shorten the
            // wait to 0-2 turns, so that bombs are tried inside the tree.
            for (int player = 0; player < 2; player++) engine.board.bombCooldown[player] = random.nextInt(3);
            if (engine.board.bombCooldown[0] == 0 || engine.board.bombCooldown[1] == 0) withBombReady++;
            long key = engine.board.positionKey();
            int expected = minimax(engine, depth, 0), actual = engine.searchValue(depth);
            check(expected == actual && engine.board.positionKey() == key, geometry + " depth " + depth + ": minimax " + expected + " vs alpha-beta " + actual);
        }
        System.out.printf("3. alpha-beta == minimax on %s at depth %d: %d positions (%d with a bomb ready)%n", geometry, depth, positions, withBombReady);
    }

    // ------------------------------------------------------------------ 4. transposition table

    static void tableSolvesMatchPlainSolves(Random random, String geometry, int positions) {
        int wins = 0, draws = 0, losses = 0;
        for (int i = 0; i < positions; i++) {
            Engine withTable = Tools.engine(geometry);
            Board board = withTable.board;
            playRandomly(board, random, board.cellCount / 3 + random.nextInt(board.cellCount / 3));
            if (isOver(board)) continue;
            int remaining = board.cellCount - board.quarkCount;
            int solved = withTable.searchValue(remaining);

            Engine plain = Tools.engine(geometry);
            plain.useTranspositionTable = false;
            plain.board.restoreState(board.saveState());
            int plainSolved = plain.searchValue(remaining);
            check(solved == plainSolved, geometry + " exact solve: with table " + solved + " vs without " + plainSolved);
            if (solved > 0) wins++;
            else if (solved == 0) draws++;
            else losses++;
        }
        System.out.printf("4. table solves == plain solves on %s: %d wins, %d draws, %d losses for the side to move%n", geometry, wins, draws, losses);
    }

    // ------------------------------------------------------------------ 5. speed

    static void measureSpeed(Random random) {
        long nodes = 0, nanos = 0;
        int depths = 0, positions = 0;
        for (int i = 0; i < 30; i++) {
            Engine engine = new QuarkBot();
            playRandomly(engine.board, random, 4 + random.nextInt(30));
            if (isOver(engine.board)) continue;
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
