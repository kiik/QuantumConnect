package ch.qc.meelik.bots;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.Move;
import ch.cern.quantumconnect.core.QuantumField;
import ch.cern.quantumconnect.core.SpaceOwnership;

import java.util.Date;

/**
 * ScrapBot: the learned line weights in as little code as possible, with nothing imported but the
 * engine API.
 *
 * <h2>Board</h2>
 * One {@code int} per cell, {@code board[row * columns + column]}, row 0 at the top as the arena
 * numbers it: +1 is our quark, -1 the opponent's, 0 empty. The size is read from the field every
 * turn, so any board size works.
 *
 * <h2>Score</h2>
 * A line is any run of six cells in one of four directions. A line held by one player only is worth
 * {@link #WEIGHT}[direction][quarks in it] to that player. The weights are the league winner's,
 * params/tl-d5-s6.txt, learned by self-play. Having the bomb ready while the opponent doesn't is
 * worth {@link #BOMB_READY_BONUS}. A drop changes only the lines through its cell, so the search
 * rescores just those; a bomb moves quarks in three columns, so it rescores the whole board.
 *
 * <h2>Search</h2>
 * Negamax with alpha-beta pruning, centre columns first, one ply deeper at a time until the time
 * budget runs out. A move is a column and whether to bomb. Any complete line after a move wins for
 * the player who made it, as in the engine, even a line of the opponent's that a blast made.
 */
public final class ScrapBot implements Algorithm {

    // ------------------------------------------------------------------ rules and weights

    static final int LINE_LENGTH = 6;
    /** Own turns before the first bomb, and from one bomb to the next (the bomb turn counts as one). */
    static final int FIRST_BOMB_WAIT = 10, BOMB_COOLDOWN = 10;
    /** Line directions as (row step, column step): horizontal, vertical, two diagonals. */
    static final int[][] DIRECTIONS = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    /** Value of a line by direction and by its number of quarks, 0..5, when the other player has none in it. */
    static final int[][] WEIGHT = {
            {0, 65, 178, 253, 614, 7606},
            {0, 17, 63, 167, 678, 6437},
            {0, 16, 107, 198, 700, 8183},
            {0, 16, 107, 198, 700, 8183}};
    static final int BOMB_READY_BONUS = 22;
    static final int WIN = 1 << 28;
    static final long BUDGET_NANOS = 40_000_000L;

    // ------------------------------------------------------------------ state

    int rows, columns;
    /** Our colour in the current game, from who moved first. */
    SpaceOwnership me;
    /** The move to play this turn, and the best move of the search in progress: column * 2 + (bomb ? 1 : 0). */
    int chosenMove, rootMove;
    /** Set when a scored line turns out to be complete. */
    boolean sawCompleteLine;
    boolean outOfTime;
    long deadline;

    public ScrapBot() {
    }

    // ------------------------------------------------------------------ the arena interface

    @Override
    public int accelerateQuark(QuantumField field) {
        rows = field.getNumberOfRows();
        columns = field.getNumberOfColumns();
        int[] board = new int[rows * columns];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                if (!field.isSpaceEmpty(row, column)) board[row * columns + column] = field.belongsToMe(row, column) ? 1 : -1;

        // Our colour, at a game's first call: first player if nobody has moved yet. Then the bomb
        // cooldowns, from the move history: one entry per quark placed, bombs marked.
        var history = field.getMoveHistory();
        if (history.size() <= 1) me = history.isEmpty() ? SpaceOwnership.FIRST_PLAYER : SpaceOwnership.SECOND_PLAYER;
        int myWait = FIRST_BOMB_WAIT, opponentWait = FIRST_BOMB_WAIT;
        for (Move move : history) {
            if (move.player() == me) myWait = move.isSpecialMove() ? BOMB_COOLDOWN - 1 : myWait - 1;
            else opponentWait = move.isSpecialMove() ? BOMB_COOLDOWN - 1 : opponentWait - 1;
        }

        // Depth 1 always completes: its replies are scored without looking at the clock.
        deadline = nanoTime() + BUDGET_NANOS;
        outOfTime = false;
        int score = scoreBoard(board);
        for (int depth = 1; depth <= rows * columns; depth++) {
            int value = search(board, score, 1, myWait, opponentWait, depth, -WIN - 1, WIN + 1, 0);
            if (outOfTime) break;
            chosenMove = rootMove;
            if (Math.abs(value) > WIN - 1000) break; // a forced result: deeper search can't change it
        }
        return chosenMove / 2;
    }

    @Override
    public boolean useSpecialMove(QuantumField field) {
        return chosenMove % 2 == 1;
    }

    // ------------------------------------------------------------------ the clock

    /**
     * Now, in nanoseconds, from java.util.Date: millisecond resolution, enough for a budget of tens of
     * milliseconds. The arena's sandbox rejects java.lang.System, so System.nanoTime is out.
     */
    static long nanoTime() {
        return new Date().getTime() * 1_000_000L;
    }

    // ------------------------------------------------------------------ search

    /**
     * The value of the board for the side to move ({@code mover}: +1 us, -1 the opponent), searched
     * {@code depth} plies deep. {@code score} is the board's line score from our side, and the board
     * has no complete line. At ply 0 it leaves the best move in rootMove.
     */
    int search(int[] board, int score, int mover, int moverWait, int otherWait, int depth, int alpha, int beta, int ply) {
        if (depth == 0) return mover * score + BOMB_READY_BONUS * ((moverWait <= 0 ? 1 : 0) - (otherWait <= 0 ? 1 : 0));
        if (nanoTime() > deadline) outOfTime = true;
        if (outOfTime) return 0;

        int best = -WIN - 1;
        for (int i = 0; i < columns; i++) {
            int column = columns / 2 + (i % 2 == 0 ? i / 2 : -(i + 1) / 2); // centre out: 6, 5, 7, 4, ...
            if (board[column] != 0) continue;
            int row = rows - 1;
            while (board[row * columns + column] != 0) row--;
            for (int bomb = 0; bomb <= (moverWait <= 0 ? 1 : 0); bomb++) {
                int[] next = board;
                int nextScore;
                sawCompleteLine = false;
                if (bomb == 0) { // a drop: rescore the lines through its cell
                    nextScore = score - scoreAround(board, row, column);
                    board[row * columns + column] = mover;
                    nextScore += scoreAround(board, row, column);
                } else {         // a bomb: blast on a copy and rescore everything
                    next = board.clone();
                    blast(next, row, column);
                    nextScore = scoreBoard(next);
                }
                int value = sawCompleteLine ? WIN - ply - 1
                        : -search(next, nextScore, -mover, otherWait, bomb == 1 ? BOMB_COOLDOWN - 1 : moverWait - 1, depth - 1, -beta, -alpha, ply + 1);
                board[row * columns + column] = 0; // undo the drop (a bomb was played on a copy)
                if (outOfTime) return 0;
                if (value > best) {
                    best = value;
                    if (ply == 0) rootMove = column * 2 + bomb;
                }
                if (value > alpha) alpha = value;
                if (alpha >= beta) return best;
            }
        }
        return best == -WIN - 1 ? 0 : best; // no open column: the board is full, a draw
    }

    // ------------------------------------------------------------------ scoring lines

    /** The score of every line on the board, from our side. */
    int scoreBoard(int[] board) {
        int score = 0;
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                for (int d = 0; d < 4; d++) score += scoreLine(board, row, column, d);
        return score;
    }

    /** The score of the lines through one cell, from our side. */
    int scoreAround(int[] board, int row, int column) {
        int score = 0;
        for (int d = 0; d < 4; d++)
            for (int k = 0; k < LINE_LENGTH; k++) score += scoreLine(board, row - k * DIRECTIONS[d][0], column - k * DIRECTIONS[d][1], d);
        return score;
    }

    /** The value of the line starting at (row, column) in direction d, or 0 if it leaves the board. */
    int scoreLine(int[] board, int row, int column, int d) {
        int lastRow = row + (LINE_LENGTH - 1) * DIRECTIONS[d][0], lastColumn = column + (LINE_LENGTH - 1) * DIRECTIONS[d][1];
        if (row < 0 || column < 0 || column >= columns || lastRow >= rows || lastColumn < 0 || lastColumn >= columns) return 0;
        int ours = 0, theirs = 0;
        for (int i = 0; i < LINE_LENGTH; i++) {
            int quark = board[(row + i * DIRECTIONS[d][0]) * columns + column + i * DIRECTIONS[d][1]];
            if (quark > 0) ours++;
            else if (quark < 0) theirs++;
        }
        if (ours == LINE_LENGTH || theirs == LINE_LENGTH) sawCompleteLine = true;
        else if (theirs == 0) return WEIGHT[d][ours];
        else if (ours == 0) return -WEIGHT[d][theirs];
        return 0;
    }

    // ------------------------------------------------------------------ the bomb

    /** Clears the 3x3 cells around (row, column), where the bomb's quark would land; quarks above fall. */
    void blast(int[] board, int row, int column) {
        for (int c = Math.max(0, column - 1); c <= Math.min(columns - 1, column + 1); c++) {
            for (int r = Math.max(0, row - 1); r <= Math.min(rows - 1, row + 1); r++) board[r * columns + c] = 0;
            int to = rows - 1; // gravity: move the column's quarks down, keeping their order
            for (int from = rows - 1; from >= 0; from--) {
                int quark = board[from * columns + c];
                if (quark == 0) continue;
                board[from * columns + c] = 0;
                board[to-- * columns + c] = quark;
            }
        }
    }
}
