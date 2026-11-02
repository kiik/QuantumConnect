package ch.qc.meelik;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.QuantumField;
import ch.cern.quantumconnect.core.SpaceOwnership;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * QuantumConnect bot: Connect-4 with extra steps (8x12 board, six in a row, 3x3 bomb).
 *
 * <p>Deliberately one class with no nested or anonymous classes, so the JAR holds a single .class
 * file. Board model, search and arena adapter all live here. The (rows, cols, k, bombs) constructor
 * lets the tools run the same engine on other boards, e.g. plain Connect-4, for validation.
 *
 * <p>Board model: cell index = col * rows + h, with h = 0 at the bottom (the arena's row 0 is the
 * top). Every line of k cells (a "window") keeps a count of each player's quarks in it. That gives
 * the win test (a count reaching k) and the evaluation, both updated incrementally per quark.
 *
 * <p>Search: iterative-deepening negamax with alpha-beta, a transposition table and history move
 * ordering. A move is an action {@code col * 2 + bomb}.
 */
public class QuarkBot implements Algorithm {
    /** Own moves from one bomb to the next. */
    public static final int COOLDOWN = 10;
    /** Own moves before the first bomb. Assumption: ready from the first move (open question). */
    public static final int START_WAIT = 0;
    public static final int WIN = 1 << 28;
    static final int MAX_PLY = 64, EXACT = 0, LOWER = 1, UPPER = 2;

    public final int rows, cols, k, size, nWin;
    public final boolean bombs;
    /**
     * Evaluation weight of a window with n quarks of one player and none of the other, by window
     * direction: 0 horizontal, 1 vertical, 2 diagonal (both diagonals, by mirror symmetry).
     */
    public final int[][] weight;
    /** Evaluation bonus for having the bomb ready while the opponent doesn't. */
    public int bombBonus;
    public long budgetNanos = 40_000_000L;
    public int maxDepth = MAX_PLY - 2;
    public boolean useTT = true;

    final int[][] cellWins;
    final int[] wdir;
    final int[] order;
    final long[] zCell, zWait;
    final long zTurn;

    // Position. Player 0 is "me" when playing in the arena, the first mover in the tools.
    public final int[] cell, height, wait = new int[2], full = new int[2];
    final int[] cnt;
    public int turn, pieces, score;
    long hash;

    // Search. The table is 8 MB; the arena allows roughly 100 MB.
    final long[] ttKey = new long[1 << 19], ttVal = new long[1 << 19];
    final int[][] moves = new int[MAX_PLY][], snap = new int[MAX_PLY][], history;
    final int[] tmp;
    final int snapLen;
    long deadline;
    boolean stop;
    public long nodes;
    public int lastValue, lastDepth;

    // Arena adapter.
    SpaceOwnership mine;
    int[] expected;
    int planned = -1, myWait = START_WAIT, oppWait = START_WAIT;
    long usedNanos;

    public QuarkBot() {
        this(8, 12, 6, true);
    }

    public QuarkBot(int rows, int cols, int k, boolean bombs) {
        this.rows = rows;
        this.cols = cols;
        this.k = k;
        this.bombs = bombs;
        size = rows * cols;
        weight = new int[3][k + 1];
        for (int[] wt : weight)
            for (int n = 1; n < k; n++) wt[n] = n < k - 1 ? 16 << 2 * (n - 1) : 16 << 2 * (n - 1) + 1;

        List<int[]> wins = new ArrayList<>();
        List<Integer> dirOf = new ArrayList<>();
        int[][] dirs = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (int c = 0; c < cols; c++)
            for (int h = 0; h < rows; h++)
                for (int d = 0; d < dirs.length; d++) {
                    int dc = dirs[d][0], dh = dirs[d][1], ec = c + (k - 1) * dc, eh = h + (k - 1) * dh;
                    if (ec < 0 || ec >= cols || eh < 0 || eh >= rows) continue;
                    int[] w = new int[k];
                    for (int i = 0; i < k; i++) w[i] = (c + i * dc) * rows + h + i * dh;
                    wins.add(w);
                    dirOf.add(Math.min(d, 2));
                }
        nWin = wins.size();
        wdir = new int[nWin];
        for (int w = 0; w < nWin; w++) wdir[w] = dirOf.get(w);
        int[] per = new int[size];
        for (int[] w : wins) for (int i : w) per[i]++;
        cellWins = new int[size][];
        for (int i = 0; i < size; i++) cellWins[i] = new int[per[i]];
        Arrays.fill(per, 0);
        for (int w = 0; w < nWin; w++) for (int i : wins.get(w)) cellWins[i][per[i]++] = w;

        order = new int[cols];
        for (int i = 0; i < cols; i++) order[i] = i;
        for (int i = 1; i < cols; i++) // insertion sort by distance from the centre
            for (int j = i; j > 0 && Math.abs(2 * order[j] - cols + 1) < Math.abs(2 * order[j - 1] - cols + 1); j--) {
                int t = order[j]; order[j] = order[j - 1]; order[j - 1] = t;
            }

        Random rnd = new Random(20261102L);
        zCell = new long[2 * size];
        for (int i = 0; i < zCell.length; i++) zCell[i] = rnd.nextLong();
        zWait = new long[2 * (COOLDOWN + 1)];
        for (int i = 0; i < zWait.length; i++) zWait[i] = rnd.nextLong();
        zTurn = rnd.nextLong();

        cell = new int[size];
        height = new int[cols];
        cnt = new int[2 * nWin];
        tmp = new int[rows];
        history = new int[2][2 * cols];
        snapLen = size + cols + 2 * nWin + 9;
        for (int p = 0; p < MAX_PLY; p++) moves[p] = new int[2 * cols];
        clear();
    }

    // ---------------------------------------------------------------- arena adapter

    @Override
    public int accelerateQuark(QuantumField field) {
        long t0 = System.nanoTime();
        SpaceOwnership[][] g = field.getField();
        if (mine == null) mine = whoAmI(field, g);
        load(g, mine);
        if (expected == null) {
            if (pieces > 0 && oppWait > 0) oppWait--; // opponent opened with a plain move
        } else {
            int diff = 0; // what the opponent did since our last turn, read from the board
            for (int i = 0; i < size; i++) if (cell[i] != expected[i]) diff++;
            if (diff == 1) oppWait = Math.max(0, oppWait - 1);
            else if (diff > 1) oppWait = COOLDOWN - 1;
        }
        wait[0] = myWait;
        wait[1] = oppWait;
        planned = think(usedNanos > 20_000_000_000L ? budgetNanos / 8 : budgetNanos);
        usedNanos += System.nanoTime() - t0;
        return planned >> 1;
    }

    @Override
    public boolean useSpecialMove(QuantumField field) {
        if (planned < 0) return false;
        play(planned); // the board as we expect the arena to have it after our turn
        myWait = wait[0];
        expected = cell.clone();
        return (planned & 1) != 0;
    }

    static SpaceOwnership whoAmI(QuantumField f, SpaceOwnership[][] g) {
        for (int r = 0; r < g.length; r++)
            for (int c = 0; c < g[r].length; c++)
                if (g[r][c] != SpaceOwnership.EMPTY)
                    return f.belongsToMe(r, c) == (g[r][c] == SpaceOwnership.FIRST_PLAYER)
                            ? SpaceOwnership.FIRST_PLAYER : SpaceOwnership.SECOND_PLAYER;
        return SpaceOwnership.FIRST_PLAYER;
    }

    /** Loads an arena board (row 0 = top). {@code me} becomes player 0; player 0 is to move. */
    public void load(SpaceOwnership[][] g, SpaceOwnership me) {
        clear();
        for (int c = 0; c < cols; c++)
            for (int r = rows - 1; r >= 0 && g[r][c] != SpaceOwnership.EMPTY; r--) put(c, g[r][c] == me ? 0 : 1);
    }

    // ---------------------------------------------------------------- rules

    public void clear() {
        Arrays.fill(cell, -1);
        Arrays.fill(height, 0);
        Arrays.fill(cnt, 0);
        full[0] = full[1] = turn = pieces = score = 0;
        wait[0] = wait[1] = bombs ? START_WAIT : 0;
        hash = 0;
    }

    public boolean legal(int a) {
        return a >= 0 && a < 2 * cols && height[a >> 1] < rows && ((a & 1) == 0 || bombs && wait[turn] == 0);
    }

    /** Drops a quark for the side to move; with the bomb bit set, blasts the 3x3 around it. */
    public void play(int a) {
        int col = a >> 1, p = turn, h = height[col];
        put(col, p);
        if ((a & 1) != 0 && full[p] == 0) { // a winning drop ends the match before the bomb
            for (int c = Math.max(0, col - 1); c <= Math.min(cols - 1, col + 1); c++) collapse(c, h - 1, h + 1);
            wait[p] = COOLDOWN;
        }
        if (wait[p] > 0) wait[p]--;
        turn ^= 1;
    }

    /**
     * Winner, or -1. The last mover's line counts first. Assumption: a line formed by quarks
     * falling after a blast also wins (open question).
     */
    public int winner() {
        return full[turn ^ 1] > 0 ? turn ^ 1 : full[turn] > 0 ? turn : -1;
    }

    /** Removes heights lo..hi of column c; the quarks above fall down. */
    void collapse(int c, int lo, int hi) {
        lo = Math.max(lo, 0);
        hi = Math.min(hi, height[c] - 1);
        if (lo > hi) return;
        int keep = height[c] - 1 - hi;
        for (int j = 0; j < keep; j++) tmp[j] = pop(c);
        for (int j = lo; j <= hi; j++) pop(c);
        for (int j = keep - 1; j >= 0; j--) put(c, tmp[j]);
    }

    void put(int col, int p) {
        int i = col * rows + height[col]++;
        cell[i] = p;
        pieces++;
        hash ^= zCell[2 * i + p];
        for (int w : cellWins[i]) {
            int j = 2 * w;
            score -= f(w, cnt[j], cnt[j + 1]);
            if (++cnt[j + p] == k) full[p]++;
            score += f(w, cnt[j], cnt[j + 1]);
        }
    }

    int pop(int col) {
        int i = col * rows + --height[col], p = cell[i];
        cell[i] = -1;
        pieces--;
        hash ^= zCell[2 * i + p];
        for (int w : cellWins[i]) {
            int j = 2 * w;
            score -= f(w, cnt[j], cnt[j + 1]);
            if (cnt[j + p]-- == k) full[p]--;
            score += f(w, cnt[j], cnt[j + 1]);
        }
        return p;
    }

    /** Window value from player 0's view: only windows still open to one player count. */
    int f(int w, int a, int b) {
        int[] wt = weight[wdir[w]];
        return b == 0 ? wt[a] : a == 0 ? -wt[b] : 0;
    }

    /** Static evaluation for the side to move. */
    public int eval() {
        int v = turn == 0 ? score : -score;
        return bombs ? v + bombBonus * ((wait[turn] == 0 ? 1 : 0) - (wait[turn ^ 1] == 0 ? 1 : 0)) : v;
    }

    /** Forgets everything learned in earlier games and empties the board (weights are kept). */
    public void newGame() {
        clear();
        Arrays.fill(ttKey, 0);
        for (int[] h : history) Arrays.fill(h, 0);
    }

    public long key() {
        long key = turn == 0 ? hash : hash ^ zTurn;
        return bombs ? key ^ zWait[wait[0]] ^ zWait[COOLDOWN + 1 + wait[1]] : key;
    }

    public int[] snapshot() {
        int[] s = new int[snapLen];
        save(s);
        return s;
    }

    void save(int[] s) {
        System.arraycopy(cell, 0, s, 0, size);
        System.arraycopy(height, 0, s, size, cols);
        System.arraycopy(cnt, 0, s, size + cols, 2 * nWin);
        int o = size + cols + 2 * nWin;
        s[o] = full[0]; s[o + 1] = full[1]; s[o + 2] = wait[0]; s[o + 3] = wait[1];
        s[o + 4] = turn; s[o + 5] = pieces; s[o + 6] = score;
        s[o + 7] = (int) hash; s[o + 8] = (int) (hash >>> 32);
    }

    public void restore(int[] s) {
        System.arraycopy(s, 0, cell, 0, size);
        System.arraycopy(s, size, height, 0, cols);
        System.arraycopy(s, size + cols, cnt, 0, 2 * nWin);
        int o = size + cols + 2 * nWin;
        full[0] = s[o]; full[1] = s[o + 1]; wait[0] = s[o + 2]; wait[1] = s[o + 3];
        turn = s[o + 4]; pieces = s[o + 5]; score = s[o + 6];
        hash = s[o + 7] & 0xffffffffL | (long) s[o + 8] << 32;
    }

    // ---------------------------------------------------------------- search

    /** Best action for the side to move, found by iterative deepening within the time budget. */
    public int think(long budget) {
        deadline = System.nanoTime() + budget;
        stop = false;
        nodes = 0;
        for (int[] h : history) for (int i = 0; i < h.length; i++) h[i] >>= 2;
        int best = -1;
        for (int c : order) if (height[c] < rows) { best = 2 * c; break; }
        for (int d = 1; d <= maxDepth; d++) {
            int v = search(d, -WIN - 1, WIN + 1, 0);
            if (stop) break;
            long key = key();
            int slot = (int) key & ttVal.length - 1;
            if (ttKey[slot] == key && ttVal[slot] >>> 42 != 0) best = (int) (ttVal[slot] >>> 42) - 1;
            lastValue = v;
            lastDepth = d;
            if (Math.abs(v) > WIN - 1000) break;
        }
        return best;
    }

    /** Full-width value of the current position for the side to move, without a time limit. */
    public int evaluate(int depth) {
        deadline = Long.MAX_VALUE;
        stop = false;
        return search(depth, -WIN - 1, WIN + 1, 0);
    }

    int search(int depth, int alpha, int beta, int ply) {
        if ((++nodes & 1023) == 0 && System.nanoTime() > deadline) stop = true;
        if (stop) return 0;
        int s = turn;
        if (full[s ^ 1] > 0) return ply - WIN;
        if (full[s] > 0) return WIN - ply;
        if (pieces == size) return 0;
        if (depth == 0 || ply == MAX_PLY - 1) return eval();

        long key = key();
        int slot = (int) key & ttVal.length - 1, ttMove = -1;
        if (useTT && ttKey[slot] == key) {
            long e = ttVal[slot];
            ttMove = (int) (e >>> 42) - 1;
            if ((int) (e >>> 32 & 0xff) >= depth) {
                int v = fromTT((int) e, ply), flag = (int) (e >>> 40 & 3);
                if (flag == EXACT || flag == LOWER && v >= beta || flag == UPPER && v <= alpha) return v;
            }
        }

        int[] m = moves[ply];
        int n = gen(m, ttMove), a0 = alpha, best = -WIN - 1, bestMove = -1;
        for (int i = 0; i < n; i++) {
            int a = m[i], v;
            if ((a & 1) == 0) { // plain drop: cheap make/unmake
                int w = wait[s];
                put(a >> 1, s);
                if (w > 0) wait[s] = w - 1;
                turn ^= 1;
                v = -search(depth - 1, -beta, -alpha, ply + 1);
                turn ^= 1;
                wait[s] = w;
                pop(a >> 1);
            } else { // bomb: snapshot and restore
                if (snap[ply] == null) snap[ply] = new int[snapLen];
                save(snap[ply]);
                play(a);
                v = -search(depth - 1, -beta, -alpha, ply + 1);
                restore(snap[ply]);
            }
            if (stop) return 0;
            if (v > best) { best = v; bestMove = a; }
            if (v > alpha) alpha = v;
            if (alpha >= beta) { history[s][a] += depth * depth; break; }
        }
        if (useTT) {
            int flag = best <= a0 ? UPPER : best >= beta ? LOWER : EXACT;
            ttKey[slot] = key;
            ttVal[slot] = toTT(best, ply) & 0xffffffffL | (long) depth << 32 | (long) flag << 40 | (long) (bestMove + 1) << 42;
        }
        return best;
    }

    /** Legal actions, transposition-table move first, then by history, ties centre-out. */
    int gen(int[] m, int ttMove) {
        int n = 0;
        boolean b = bombs && wait[turn] == 0;
        for (int c : order)
            if (height[c] < rows) {
                m[n++] = 2 * c;
                if (b) m[n++] = 2 * c + 1;
            }
        int[] h = history[turn];
        for (int i = 1; i < n; i++) {
            int x = m[i], kx = x == ttMove ? Integer.MAX_VALUE : h[x], j = i - 1;
            while (j >= 0 && (m[j] == ttMove ? Integer.MAX_VALUE : h[m[j]]) < kx) { m[j + 1] = m[j]; j--; }
            m[j + 1] = x;
        }
        return n;
    }

    // Win scores are stored relative to the node so they stay valid at any ply.
    static int toTT(int v, int ply) {
        return v > WIN - 1000 ? v + ply : v < 1000 - WIN ? v - ply : v;
    }

    static int fromTT(int v, int ply) {
        return v > WIN - 1000 ? v - ply : v < 1000 - WIN ? v + ply : v;
    }
}
