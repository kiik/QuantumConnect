package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.QuarkBot;

import java.util.Random;

/**
 * Correctness checks for the engine:
 * 1. incremental state (score, wins, hash) equals a rebuild from scratch, through random games with bombs;
 * 2. alpha-beta without the transposition table equals plain minimax at fixed depth (bombs included);
 * 3. with the transposition table, exact solves of small boards equal solves without it;
 * 4. search speed on the real board.
 */
public final class SelfTest {
    static int failures;

    public static void main(String[] args) {
        Random rnd = new Random(1);
        incremental(rnd);
        minimax(rnd, "6x7x4", 4, 150);
        minimax(rnd, "5x6x4", 5, 100);
        solves(rnd, "4x5x4-nobomb", 200);
        solves(rnd, "5x5x4-nobomb", 60);
        speed(rnd);
        System.out.println(failures == 0 ? "ALL PASSED" : failures + " FAILURE(S)");
    }

    static void check(boolean ok, String what) {
        if (!ok) { failures++; System.out.println("  FAIL: " + what); }
    }

    /** Plays random actions (bombing whenever allowed half the time) until the game ends. */
    static void randomPlay(QuarkBot e, Random rnd, int maxPlies) {
        for (int i = 0; i < maxPlies && e.winner() < 0 && e.pieces < e.size; i++) e.play(randomAction(e, rnd));
    }

    static int randomAction(QuarkBot e, Random rnd) {
        int a;
        do a = 2 * rnd.nextInt(e.cols) + (rnd.nextBoolean() ? 1 : 0); while (!e.legal(a));
        return a;
    }

    static void incremental(Random rnd) {
        int steps = 0, blasts = 0;
        for (int g = 0; g < 300; g++) {
            QuarkBot e = new QuarkBot(), r = new QuarkBot();
            while (e.winner() < 0 && e.pieces < e.size && steps < 1_000_000) {
                int a = randomAction(e, rnd);
                int[] s = e.snapshot();
                long key = e.key();
                e.play(a);
                blasts += a & 1;
                steps++;
                r.load(Tools.grid(e), SpaceOwnership.FIRST_PLAYER);
                r.turn = e.turn; r.wait[0] = e.wait[0]; r.wait[1] = e.wait[1];
                check(r.score == e.score && r.key() == e.key() && r.full[0] == e.full[0] && r.full[1] == e.full[1],
                        "incremental state after " + Tools.action(a));
                for (int c = 0; c < e.cols; c++)
                    for (int h = e.height[c]; h < e.rows; h++) check(e.cell[c * e.rows + h] < 0, "floating quark");
                e.restore(s);
                check(e.key() == key, "snapshot restore");
                e.play(a);
            }
        }
        System.out.printf("1. incremental vs rebuild: %d actions, %d blasts checked%n", steps, blasts);
    }

    static int minimax(QuarkBot e, int depth, int ply) {
        int w = e.winner();
        if (w >= 0) return w == e.turn ? QuarkBot.WIN - ply : ply - QuarkBot.WIN;
        if (e.pieces == e.size) return 0;
        if (depth == 0) return e.eval();
        int best = -QuarkBot.WIN - 1;
        for (int a = 0; a < 2 * e.cols; a++)
            if (e.legal(a)) {
                int[] s = e.snapshot();
                e.play(a);
                best = Math.max(best, -minimax(e, depth - 1, ply + 1));
                e.restore(s);
            }
        return best;
    }

    static void minimax(Random rnd, String geom, int depth, int positions) {
        int bombPositions = 0;
        for (int i = 0; i < positions; i++) {
            QuarkBot e = Tools.engine(geom);
            e.useTT = false;
            e.bombBonus = 7;
            e.weight[1][2] = 3;
            randomPlay(e, rnd, rnd.nextInt(e.size / 2));
            if (e.winner() >= 0 || e.pieces == e.size) continue;
            if (e.legal(1) || e.legal(3) || e.wait[0] == 0 || e.wait[1] == 0) bombPositions++;
            long key = e.key();
            int expect = minimax(e, depth, 0), got = e.evaluate(depth);
            check(expect == got && e.key() == key, geom + " depth " + depth + ": minimax " + expect + " vs alpha-beta " + got);
        }
        System.out.printf("2. alpha-beta == minimax on %s depth %d: %d positions (%d with a bomb in reach)%n", geom, depth, positions, bombPositions);
    }

    static void solves(Random rnd, String geom, int positions) {
        int[] wdl = new int[3];
        for (int i = 0; i < positions; i++) {
            QuarkBot e = Tools.engine(geom);
            randomPlay(e, rnd, e.size / 3 + rnd.nextInt(e.size / 3));
            if (e.winner() >= 0 || e.pieces == e.size) continue;
            int depth = e.size - e.pieces;
            int withTT = e.evaluate(depth);
            QuarkBot plain = Tools.engine(geom);
            plain.restore(e.snapshot());
            plain.useTT = false;
            int without = plain.evaluate(depth);
            check(withTT == without, geom + " exact solve: TT " + withTT + " vs plain " + without);
            wdl[withTT > 0 ? 0 : withTT == 0 ? 1 : 2]++;
        }
        System.out.printf("3. TT solve == plain solve on %s: %d wins / %d draws / %d losses for the side to move%n", geom, wdl[0], wdl[1], wdl[2]);
    }

    static void speed(Random rnd) {
        long nodes = 0, ns = 0;
        int depthSum = 0, n = 0;
        for (int i = 0; i < 30; i++) {
            QuarkBot e = new QuarkBot();
            randomPlay(e, rnd, 4 + rnd.nextInt(30));
            if (e.winner() >= 0) continue;
            long t0 = System.nanoTime();
            e.think(e.budgetNanos);
            ns += System.nanoTime() - t0;
            nodes += e.nodes;
            depthSum += e.lastDepth;
            n++;
        }
        System.out.printf("4. speed on 8x12x6: %.0f k nodes/s, avg depth %.1f in %.0f ms (%d positions)%n",
                nodes / (ns / 1e6), (double) depthSum / n, ns / 1e6 / n, n);
    }
}
