package ch.qc.meelik.tools;

import ch.qc.meelik.engine.Board;

/**
 * Maps the state space by ply: how many distinct positions can arise after n plies (positions
 * reached by different move orders counted once, games stopping at a win), how many of those are
 * wins, and the effective branching factor.
 *
 * <p>Usage: Census [geometry=8x12x6] [maxPly=6]. For "6x7x4-nobomb", plain Connect-4, the counts
 * are checked against the published sequence (OEIS A212693), which validates the method.
 */
public final class Census {
    static final long[] CONNECT4_POSITIONS = {1, 7, 49, 238, 1120, 4263, 16422, 54859, 184275, 558186, 1662623, 4568683};

    final Board board;
    final int maxPly;
    final long[] positions, wins;
    final LongSet[] seen;

    Census(Board board, int maxPly) {
        this.board = board;
        this.maxPly = maxPly;
        positions = new long[maxPly + 1];
        wins = new long[maxPly + 1];
        seen = new LongSet[maxPly + 1];
        for (int ply = 0; ply <= maxPly; ply++) seen[ply] = new LongSet();
    }

    public static void main(String[] args) {
        String geometry = args.length > 0 ? args[0] : "8x12x6";
        int maxPly = args.length > 1 ? Integer.parseInt(args[1]) : 6;
        Census census = new Census(Tools.board(geometry, false), maxPly);
        long started = System.nanoTime();
        census.visit(0);

        boolean connect4 = geometry.equals("6x7x4-nobomb");
        System.out.printf("%s: distinct positions by ply (%.1f s)%n", geometry, (System.nanoTime() - started) / 1e9);
        System.out.printf("%4s %14s %12s %10s%s%n", "ply", "positions", "wins", "branching", connect4 ? "   OEIS A212693" : "");
        for (int ply = 0; ply <= maxPly; ply++) {
            String branching = ply == 0 ? "" : String.format("%.2f", (double) census.positions[ply] / census.positions[ply - 1]);
            String reference = "";
            if (connect4 && ply < CONNECT4_POSITIONS.length)
                reference = String.format("   %d %s", CONNECT4_POSITIONS[ply], CONNECT4_POSITIONS[ply] == census.positions[ply] ? "ok" : "MISMATCH");
            System.out.printf("%4d %14d %12d %10s%s%n", ply, census.positions[ply], census.wins[ply], branching, reference);
        }
    }

    /** Depth-first walk that counts each position once per ply. */
    void visit(int ply) {
        if (!seen[ply].add(board.positionKey())) return;
        positions[ply]++;
        if (board.winner() >= 0) {
            wins[ply]++;
            return;
        }
        if (ply == maxPly || board.isFull()) return;
        int[] state = board.saveState();
        for (int action = 0; action < board.actionCount(); action++) {
            if (!board.isLegal(action)) continue;
            board.play(action);
            visit(ply + 1);
            board.restoreState(state);
        }
    }

    /** A set of longs by open addressing. Key 0 is stored as 1; a collision there is harmless for counting. */
    static final class LongSet {
        long[] slots = new long[1 << 10];
        int size;

        boolean add(long key) {
            if (key == 0) key = 1;
            if (2 * size >= slots.length) grow();
            int i = (int) (key ^ key >>> 29) & slots.length - 1;
            while (slots[i] != 0) {
                if (slots[i] == key) return false;
                i = i + 1 & slots.length - 1;
            }
            slots[i] = key;
            size++;
            return true;
        }

        void grow() {
            long[] old = slots;
            slots = new long[old.length * 2];
            size = 0;
            for (long key : old) if (key != 0) add(key);
        }
    }
}
