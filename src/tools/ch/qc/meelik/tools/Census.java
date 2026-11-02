package ch.qc.meelik.tools;

import ch.qc.meelik.QuarkBot;

/**
 * Maps the state space by ply: distinct positions reachable after n plies (transpositions merged,
 * games stop at a win), how many of them are wins, and the effective branching factor.
 *
 * <p>Usage: Census [geom=8x12x6] [maxPly=6]. For 6x7x4-nobomb (plain Connect-4) the counts are
 * checked against the published sequence (OEIS A212693), which validates the method.
 */
public final class Census {
    static final long[] CONNECT4 = {1, 7, 49, 238, 1120, 4263, 16422, 54859, 184275, 558186, 1662623, 4568683};

    long[] distinct, wins;
    LongSet[] seen;
    QuarkBot e;
    int maxPly;

    public static void main(String[] args) {
        String geom = args.length > 0 ? args[0] : "8x12x6";
        int maxPly = args.length > 1 ? Integer.parseInt(args[1]) : 6;
        Census c = new Census();
        c.e = Tools.engine(geom);
        c.maxPly = maxPly;
        c.distinct = new long[maxPly + 1];
        c.wins = new long[maxPly + 1];
        c.seen = new LongSet[maxPly + 1];
        for (int i = 0; i <= maxPly; i++) c.seen[i] = new LongSet();
        long t0 = System.nanoTime();
        c.walk(0);
        boolean c4 = geom.equals("6x7x4-nobomb");
        System.out.printf("%s: distinct positions by ply (%.1f s)%n", geom, (System.nanoTime() - t0) / 1e9);
        System.out.printf("%4s %14s %12s %10s%s%n", "ply", "positions", "wins", "branching", c4 ? "   OEIS A212693" : "");
        for (int p = 0; p <= maxPly; p++) {
            String ref = "";
            if (c4 && p < CONNECT4.length) ref = String.format("   %d %s", CONNECT4[p], CONNECT4[p] == c.distinct[p] ? "ok" : "MISMATCH");
            System.out.printf("%4d %14d %12d %10s%s%n", p, c.distinct[p], c.wins[p],
                    p == 0 ? "" : String.format("%.2f", (double) c.distinct[p] / c.distinct[p - 1]), ref);
        }
    }

    void walk(int ply) {
        if (!seen[ply].add(e.key())) return;
        distinct[ply]++;
        if (e.winner() >= 0) { wins[ply]++; return; }
        if (ply == maxPly || e.pieces == e.size) return;
        int[] s = e.snapshot();
        for (int a = 0; a < 2 * e.cols; a++)
            if (e.legal(a)) {
                e.play(a);
                walk(ply + 1);
                e.restore(s);
            }
    }

    /** Open-addressing set of longs (0 is mapped to 1; a collision there is harmless here). */
    static final class LongSet {
        long[] t = new long[1 << 10];
        int n;

        boolean add(long k) {
            if (k == 0) k = 1;
            if (2 * n >= t.length) grow();
            int i = (int) (k ^ k >>> 29) & t.length - 1;
            while (t[i] != 0) {
                if (t[i] == k) return false;
                i = i + 1 & t.length - 1;
            }
            t[i] = k;
            n++;
            return true;
        }

        void grow() {
            long[] old = t;
            t = new long[old.length * 2];
            n = 0;
            for (long k : old) if (k != 0) add(k);
        }
    }
}
