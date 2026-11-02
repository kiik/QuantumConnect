package ch.qc.meelik.engine;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.Move;
import ch.cern.quantumconnect.core.QuantumField;
import ch.cern.quantumconnect.core.SpaceOwnership;

import java.util.Arrays;
import java.util.List;

/**
 * The shared brain of every bot: chooses moves by alpha-beta search and talks to the arena.
 *
 * <h2>Choosing a move</h2>
 * {@link #chooseAction} deepens iteratively: a full search to depth 1, then 2, 3, ... until the
 * time budget runs out, keeping the best action of the deepest completed search. Each search is
 * negamax with alpha-beta pruning ({@link #negamax}):
 * <ul>
 *   <li>a transposition table remembers positions already searched, which the same position
 *       reached by a different move order can reuse;
 *   <li>actions are tried best-guess first: the table's best action, then actions that caused
 *       cut-offs before (the history heuristic), then centre columns first;
 *   <li>a win scores {@code WIN - ply}, so faster wins are preferred and slower losses resisted.
 * </ul>
 *
 * <h2>Talking to the arena</h2>
 * Each turn the arena calls {@link #accelerateQuark} (pick a column) and then {@link #useSpecialMove}
 * (bomb or not), both before anything is placed. The engine decides both at once, as one action,
 * and remembers the bomb decision between the two calls. It counts its own bomb cooldown, and the
 * opponent's from the move history, which marks every bomb. Every turn counts toward a cooldown,
 * also one lost to a timeout. On the engine's board, player 0 is always this bot ({@link #ME}).
 * The arena may use another board size than the standard 8x12: on the first look at a field of
 * another size, the engine replaces its board with one of that size, with the same rules and line
 * weights ({@link #fitBoardTo}). Bots therefore always reach the board through {@link #board}.
 *
 * <h2>Writing a bot</h2>
 * A bot is a subclass with a public no-arg constructor, shipped as a JAR of the engine package
 * plus the subclass. It can change the line weights ({@link Board#setLineWeights}), add knowledge
 * to the static evaluation ({@link #extraEvaluation}), or run another search before or instead of
 * alpha-beta (override {@link #chooseAction}).
 */
public abstract class Engine implements Algorithm {

    public static final int WIN = 1 << 28;
    public static final int ME = 0, OPPONENT = 1;
    public static final int MAX_PLY = 64;
    public static final int NO_ACTION = -1;
    /** After this much thinking in one match, search with an eighth of the budget (the match limit is 30 s). */
    static final long MATCH_TIME_GUARD_NANOS = 20_000_000_000L;

    /** The board searched on; replaced when the arena's field has another size. */
    public Board board;

    // ------------------------------------------------------------------ settings

    /** Static-evaluation bonus for having the bomb ready while the opponent doesn't. */
    public int bombReadyBonus;
    public long budgetNanos = 40_000_000L;
    public int maxDepth = MAX_PLY - 2;
    public boolean useTranspositionTable = true;

    // ------------------------------------------------------------------ results of the last search

    public long nodesSearched;
    public int lastScore, lastDepth;

    // ------------------------------------------------------------------ search memory (about 8 MB; the arena allows ~100 MB)

    static final int TABLE_SIZE = 1 << 19;
    static final int EXACT = 0, LOWER_BOUND = 1, UPPER_BOUND = 2;
    final long[] tableKey = new long[TABLE_SIZE], tableEntry = new long[TABLE_SIZE];
    final int[][] actionLists = new int[MAX_PLY][], undoStates = new int[MAX_PLY][];
    /** How often each action of each player caused a cut-off, weighted by depth. */
    int[][] historyScore;
    long deadlineNanos;
    boolean outOfTime;

    // ------------------------------------------------------------------ match memory

    SpaceOwnership myColour;
    int plannedAction = NO_ACTION;
    int myCooldown = Board.FIRST_BOMB_WAIT, opponentCooldown = Board.FIRST_BOMB_WAIT;
    int myTurns;
    long timeUsedNanos;

    protected Engine(Board board) {
        useBoard(board);
    }

    /** Searches on this board from now on: sizes the search memory to it and forgets what it held. */
    private void useBoard(Board newBoard) {
        board = newBoard;
        for (int ply = 0; ply < MAX_PLY; ply++) {
            actionLists[ply] = new int[board.actionCount()];
            undoStates[ply] = null;
        }
        historyScore = new int[2][board.actionCount()];
        Arrays.fill(tableKey, 0);
    }

    /** Replaces the board if the arena's field has another size: same line length, bomb rule and weights. */
    void fitBoardTo(SpaceOwnership[][] grid) {
        int rows = grid.length, columns = grid[0].length;
        if (rows == board.rows && columns == board.columns) return;
        Board resized = new Board(rows, columns, board.lineLength, board.bombsEnabled, board.tracksThreats);
        resized.setLineWeights(board.lineWeight);
        useBoard(resized);
    }

    /** Starts a new game on an empty board, forgetting the match and all search memory. Weights are kept. */
    public void newGame() {
        board.clear();
        Arrays.fill(tableKey, 0);
        for (int[] scores : historyScore) Arrays.fill(scores, 0);
        myColour = null;
        plannedAction = NO_ACTION;
        myCooldown = opponentCooldown = Board.FIRST_BOMB_WAIT;
        myTurns = 0;
        timeUsedNanos = 0;
    }

    // ------------------------------------------------------------------ the arena interface

    @Override
    public int accelerateQuark(QuantumField field) {
        long started = System.nanoTime();
        SpaceOwnership[][] grid = field.getField();
        if (myColour == null) myColour = whoAmI(field, grid);
        fitBoardTo(grid);
        board.load(grid, myColour);
        updateOpponentCooldown(field.getMoveHistory());
        myTurns++;
        board.bombCooldown[ME] = myCooldown;
        board.bombCooldown[OPPONENT] = opponentCooldown;
        plannedAction = chooseAction(timeUsedNanos > MATCH_TIME_GUARD_NANOS ? budgetNanos / 8 : budgetNanos);
        timeUsedNanos += System.nanoTime() - started;
        return Board.columnOf(plannedAction);
    }

    @Override
    public boolean useSpecialMove(QuantumField field) {
        if (plannedAction == NO_ACTION) return false;
        board.play(plannedAction); // counts the turn toward our cooldown, and resets it after a bomb
        myCooldown = board.bombCooldown[ME];
        return Board.isBomb(plannedAction);
    }

    /** Our colour, from any quark on the field; first player if the field is empty. */
    static SpaceOwnership whoAmI(QuantumField field, SpaceOwnership[][] grid) {
        for (int row = 0; row < grid.length; row++)
            for (int column = 0; column < grid[row].length; column++) {
                SpaceOwnership quark = grid[row][column];
                if (quark == SpaceOwnership.EMPTY) continue;
                boolean firstPlayersQuark = quark == SpaceOwnership.FIRST_PLAYER;
                return field.belongsToMe(row, column) == firstPlayersQuark ? SpaceOwnership.FIRST_PLAYER : SpaceOwnership.SECOND_PLAYER;
            }
        return SpaceOwnership.FIRST_PLAYER;
    }

    /**
     * Counts the opponent's turn since our last one toward its cooldown. The history lists every
     * quark placed, marking bombs; a turn the opponent lost (too slow, full column) adds no entry
     * but counts all the same. (The board can't tell: a bomb on an empty spot leaves it unchanged.)
     */
    void updateOpponentCooldown(List<Move> history) {
        boolean opponentHasMoved = myTurns > 0 || myColour == SpaceOwnership.SECOND_PLAYER;
        if (!opponentHasMoved) return;
        Move last = history.isEmpty() ? null : history.get(history.size() - 1);
        if (last != null && last.player() != myColour && last.isSpecialMove()) opponentCooldown = Board.BOMB_COOLDOWN - 1;
        else if (opponentCooldown > 0) opponentCooldown--;
    }

    // ------------------------------------------------------------------ choosing an action

    /** The best action for the side to move, by iterative deepening within the time budget. */
    public int chooseAction(long budget) {
        deadlineNanos = System.nanoTime() + budget;
        outOfTime = false;
        nodesSearched = 0;
        for (int[] scores : historyScore) for (int a = 0; a < scores.length; a++) scores[a] >>= 2; // age old knowledge

        int best = firstLegalDrop();
        for (int depth = 1; depth <= maxDepth; depth++) {
            int score = negamax(depth, -WIN - 1, WIN + 1, 0);
            if (outOfTime) break;
            int fromTable = tableAction(board.positionKey());
            if (fromTable != NO_ACTION) best = fromTable;
            lastScore = score;
            lastDepth = depth;
            if (Math.abs(score) > WIN - MAX_PLY) break; // a forced result: deeper search can't change it
        }
        return best;
    }

    /** The score of the current position for the side to move, by a full search of the given depth. */
    public int searchValue(int depth) {
        deadlineNanos = Long.MAX_VALUE;
        outOfTime = false;
        return negamax(depth, -WIN - 1, WIN + 1, 0);
    }

    private int firstLegalDrop() {
        for (int column : board.columnsCentreOut) if (board.columnHeight[column] < board.rows) return Board.action(column, false);
        return NO_ACTION;
    }

    // ------------------------------------------------------------------ evaluation

    /** Static evaluation for the side to move: line values, bomb readiness and any extra knowledge. */
    public int evaluate() {
        Board b = board;
        int score = b.sideToMove == 0 ? b.staticScore : -b.staticScore;
        if (b.bombsEnabled) score += bombReadyBonus * (isBombReady(b.sideToMove) - isBombReady(1 - b.sideToMove));
        return score + extraEvaluation();
    }

    private int isBombReady(int player) {
        return board.bombCooldown[player] == 0 ? 1 : 0;
    }

    /** Hook for subclasses: knowledge added to the static evaluation, for the side to move. */
    protected int extraEvaluation() {
        return 0;
    }

    // ------------------------------------------------------------------ alpha-beta search

    int negamax(int depth, int alpha, int beta, int ply) {
        if ((++nodesSearched & 1023) == 0 && System.nanoTime() > deadlineNanos) outOfTime = true;
        if (outOfTime) return 0;

        Board b = board;
        int mover = b.sideToMove;
        if (b.hasLine()) return ply - WIN; // any line wins for the last mover: the opponent
        if (b.isFull()) return 0;
        if (depth == 0 || ply == MAX_PLY - 1) return evaluate();

        long key = b.positionKey();
        int slot = tableSlot(key), tableAction = NO_ACTION;
        if (useTranspositionTable && tableKey[slot] == key) {
            long entry = tableEntry[slot];
            tableAction = entryAction(entry);
            if (entryDepth(entry) >= depth) {
                int score = scoreFromTable(entryScore(entry), ply), bound = entryBound(entry);
                if (bound == EXACT || bound == LOWER_BOUND && score >= beta || bound == UPPER_BOUND && score <= alpha) return score;
            }
        }

        int[] actions = actionLists[ply];
        int actionCount = orderedLegalActions(actions, tableAction);
        int originalAlpha = alpha, bestScore = -WIN - 1, bestAction = NO_ACTION;
        for (int i = 0; i < actionCount; i++) {
            int action = actions[i], score;
            if (Board.isBomb(action)) { // a blast rearranges several columns: save the whole state
                if (undoStates[ply] == null) undoStates[ply] = new int[b.stateSize];
                b.saveState(undoStates[ply]);
                b.play(action);
                score = -negamax(depth - 1, -beta, -alpha, ply + 1);
                b.restoreState(undoStates[ply]);
            } else {
                int column = Board.columnOf(action), previousCooldown = b.drop(column);
                score = -negamax(depth - 1, -beta, -alpha, ply + 1);
                b.undoDrop(column, previousCooldown);
            }
            if (outOfTime) return 0;
            if (score > bestScore) {
                bestScore = score;
                bestAction = action;
            }
            if (score > alpha) alpha = score;
            if (alpha >= beta) {
                historyScore[mover][action] += depth * depth;
                break;
            }
        }

        if (useTranspositionTable) {
            int bound = bestScore <= originalAlpha ? UPPER_BOUND : bestScore >= beta ? LOWER_BOUND : EXACT;
            tableKey[slot] = key;
            tableEntry[slot] = packEntry(scoreToTable(bestScore, ply), depth, bound, bestAction);
        }
        return bestScore;
    }

    /** Fills {@code actions} with the legal actions, best guess first. Returns how many there are. */
    int orderedLegalActions(int[] actions, int preferred) {
        Board b = board;
        boolean mayBomb = b.bombsEnabled && b.bombCooldown[b.sideToMove] == 0;
        int count = 0;
        for (int column : b.columnsCentreOut) {
            if (b.columnHeight[column] == b.rows) continue;
            actions[count++] = Board.action(column, false);
            if (mayBomb) actions[count++] = Board.action(column, true);
        }
        int[] history = historyScore[b.sideToMove];
        for (int i = 1; i < count; i++) { // insertion sort: stable, so ties stay centre-out
            int action = actions[i], priority = action == preferred ? Integer.MAX_VALUE : history[action], j = i - 1;
            while (j >= 0 && (actions[j] == preferred ? Integer.MAX_VALUE : history[actions[j]]) < priority) {
                actions[j + 1] = actions[j];
                j--;
            }
            actions[j + 1] = action;
        }
        return count;
    }

    // ------------------------------------------------------------------ transposition table

    static int tableSlot(long key) {
        return (int) key & TABLE_SIZE - 1;
    }

    /** The best action stored for a position, or NO_ACTION. */
    int tableAction(long key) {
        int slot = tableSlot(key);
        return tableKey[slot] == key ? entryAction(tableEntry[slot]) : NO_ACTION;
    }

    // An entry packs the score (32 bits), depth (8), bound (2) and action + 1 (rest) into one long.
    static long packEntry(int score, int depth, int bound, int action) {
        return score & 0xffffffffL | (long) depth << 32 | (long) bound << 40 | (long) (action + 1) << 42;
    }

    static int entryScore(long entry) {
        return (int) entry;
    }

    static int entryDepth(long entry) {
        return (int) (entry >>> 32 & 0xff);
    }

    static int entryBound(long entry) {
        return (int) (entry >>> 40 & 3);
    }

    static int entryAction(long entry) {
        return (int) (entry >>> 42) - 1;
    }

    // Win scores count plies from the root; the table stores them counted from the position itself.
    static int scoreToTable(int score, int ply) {
        return score > WIN - MAX_PLY ? score + ply : score < MAX_PLY - WIN ? score - ply : score;
    }

    static int scoreFromTable(int score, int ply) {
        return score > WIN - MAX_PLY ? score - ply : score < MAX_PLY - WIN ? score + ply : score;
    }
}
