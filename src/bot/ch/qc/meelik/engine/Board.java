package ch.qc.meelik.engine;

import ch.cern.quantumconnect.core.SpaceOwnership;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * The game model of QuantumConnect: Connect-4 with extra steps.
 *
 * <h2>Rules modelled</h2>
 * <ul>
 *   <li>Quarks drop into columns and stack from the bottom (gravity).
 *   <li>A player wins with {@link #lineLength} of their quarks in a row: horizontally, vertically or
 *       diagonally.
 *   <li>The bomb: instead of a plain drop, a player may drop and blast the 3x3 cells around the
 *       quark, that quark included. Quarks above the blast fall down. A player may bomb once every
 *       {@link #BOMB_COOLDOWN} of their own turns, and not before {@link #FIRST_BOMB_WAIT} turns.
 *   <li>After every turn the arena looks for a line anywhere on the board. If there is one, the
 *       player who just moved wins, whoever's line it is (a blast can make the opponent's).
 * </ul>
 * These are the rules of the engine's {@code Experiment} (core-2.0), read from its bytecode.
 *
 * <h2>Representation</h2>
 * Cells are numbered column by column: {@code cell = column * rows + height}, where height 0 is the
 * bottom row. (The arena numbers rows from the top.) Players are 0 and 1; an empty cell is
 * {@link #EMPTY}. A move, called an action, packs a column and the bomb decision:
 * {@code action = column * 2 + (bomb ? 1 : 0)}.
 *
 * <h2>Lines: the heart of the model</h2>
 * A line is any run of {@link #lineLength} cells in one of the four directions; the 8x12 board has
 * 134. For every line the board keeps how many quarks each player has in it, and updates the counts
 * whenever a single quark is added or removed. Everything else follows from those counts, cheaply:
 * <ul>
 *   <li>the win test: a count reaching lineLength ({@link #completedLines});
 *   <li>the static score: a line open to only one player is worth {@link #lineWeight} to them;
 *   <li>threats, when enabled: cells that would complete a line ({@link #threatCount}).
 * </ul>
 * A bomb is carried out as single-quark removals and re-additions, so it needs no bookkeeping of
 * its own.
 */
public final class Board {

    // ------------------------------------------------------------------ rules

    public static final int EMPTY = -1;
    /** Own turns from one bomb to the next. */
    public static final int BOMB_COOLDOWN = 10;
    /** Own turns before the first bomb: the arena's counter starts at 0 and allows a bomb at 10. */
    public static final int FIRST_BOMB_WAIT = 10;

    /** Line directions as (column step, height step); both diagonals share a weight by symmetry. */
    static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    public static final int HORIZONTAL = 0, VERTICAL = 1, DIAGONAL = 2;

    // ------------------------------------------------------------------ geometry (fixed per board size)

    public final int rows, columns, lineLength, cellCount, lineCount;
    public final boolean bombsEnabled, tracksThreats;
    /** Columns ordered from the centre outwards: the usual best-first guess for move ordering. */
    public final int[] columnsCentreOut;
    final int[][] cellsOfLine, linesThroughCell;
    final int[] lineDirection;

    /** Value of a line holding n quarks of one player and none of the other, per direction class. */
    public final int[][] lineWeight;

    // ------------------------------------------------------------------ position

    /** Owner of each cell: player 0, player 1 or EMPTY. */
    public final int[] owner;
    public final int[] columnHeight;
    /** Own moves each player must still make before they may bomb. */
    public final int[] bombCooldown = new int[2];
    /** Number of complete lines each player has; above zero means that player has won. */
    public final int[] completedLines = new int[2];
    /** Quarks of each player in each line, at {@code 2 * line + player}. */
    final int[] lineQuarks;
    /** Lines in which a cell is the only one a player still needs, at {@code player * cellCount + cell}. */
    public final int[] threatCount;
    public int sideToMove, quarkCount;
    /** Sum of the line values, from player 0's point of view. */
    public int staticScore;

    // ------------------------------------------------------------------ hashing and saving

    final long[] zobristQuark, zobristCooldown;
    final long zobristSideToMove;
    long boardHash;
    final int[] fallingQuarks;
    final int stateSize;

    /** The standard QuantumConnect board: 8 rows, 12 columns, six in a row, bombs on. */
    public Board() {
        this(8, 12, 6, true, false);
    }

    public Board(int rows, int columns, int lineLength, boolean bombsEnabled, boolean tracksThreats) {
        this.rows = rows;
        this.columns = columns;
        this.lineLength = lineLength;
        this.bombsEnabled = bombsEnabled;
        this.tracksThreats = tracksThreats;
        cellCount = rows * columns;

        List<int[]> lines = new ArrayList<>();
        List<Integer> directions = new ArrayList<>();
        for (int column = 0; column < columns; column++)
            for (int height = 0; height < rows; height++)
                for (int d = 0; d < DIRECTIONS.length; d++) {
                    int[] line = lineFrom(column, height, DIRECTIONS[d]);
                    if (line == null) continue;
                    lines.add(line);
                    directions.add(Math.min(d, DIAGONAL));
                }
        lineCount = lines.size();
        cellsOfLine = lines.toArray(new int[0][]);
        lineDirection = new int[lineCount];
        for (int line = 0; line < lineCount; line++) lineDirection[line] = directions.get(line);
        linesThroughCell = invert(cellsOfLine, cellCount);
        columnsCentreOut = centreOut(columns);

        lineWeight = new int[3][];
        for (int d = 0; d < 3; d++) lineWeight[d] = defaultLineWeights(lineLength);

        Random random = new Random(20261102L);
        zobristQuark = randomLongs(random, 2 * cellCount);
        zobristCooldown = randomLongs(random, 2 * (BOMB_COOLDOWN + 1));
        zobristSideToMove = random.nextLong();

        owner = new int[cellCount];
        columnHeight = new int[columns];
        lineQuarks = new int[2 * lineCount];
        threatCount = new int[2 * cellCount];
        fallingQuarks = new int[rows];
        stateSize = owner.length + columnHeight.length + lineQuarks.length + threatCount.length + 9;
        clear();
    }

    /** The cells of the line starting at (column, height) in a direction, or null if it leaves the board. */
    private int[] lineFrom(int column, int height, int[] direction) {
        int lastColumn = column + (lineLength - 1) * direction[0], lastHeight = height + (lineLength - 1) * direction[1];
        if (lastColumn < 0 || lastColumn >= columns || lastHeight < 0 || lastHeight >= rows) return null;
        int[] line = new int[lineLength];
        for (int i = 0; i < lineLength; i++) line[i] = (column + i * direction[0]) * rows + height + i * direction[1];
        return line;
    }

    // Plain loops here and throughout the bot code: lambdas and method references compile to
    // java.lang.invoke calls, which the arena's sandbox may count as forbidden reflection.

    private static int[][] invert(int[][] cellsOfLine, int cellCount) {
        int[] linesPerCell = new int[cellCount];
        for (int[] line : cellsOfLine) for (int cell : line) linesPerCell[cell]++;
        int[][] linesThroughCell = new int[cellCount][];
        for (int cell = 0; cell < cellCount; cell++) linesThroughCell[cell] = new int[linesPerCell[cell]];
        Arrays.fill(linesPerCell, 0);
        for (int line = 0; line < cellsOfLine.length; line++)
            for (int cell : cellsOfLine[line]) linesThroughCell[cell][linesPerCell[cell]++] = line;
        return linesThroughCell;
    }

    private static int[] centreOut(int columns) {
        int[] order = new int[columns];
        for (int column = 0; column < columns; column++) order[column] = column;
        for (int i = 1; i < columns; i++) // insertion sort by distance from the centre
            for (int j = i; j > 0 && distanceFromCentre(order[j], columns) < distanceFromCentre(order[j - 1], columns); j--) {
                int swap = order[j];
                order[j] = order[j - 1];
                order[j - 1] = swap;
            }
        return order;
    }

    private static int distanceFromCentre(int column, int columns) {
        return Math.abs(2 * column - columns + 1);
    }

    /** 16, 64, 256, ... for 1, 2, 3, ... quarks, doubled for a line one quark short of complete. */
    static int[] defaultLineWeights(int lineLength) {
        int[] weights = new int[lineLength + 1];
        for (int n = 1; n < lineLength; n++) weights[n] = 16 << 2 * (n - 1);
        weights[lineLength - 1] *= 2;
        return weights;
    }

    private static long[] randomLongs(Random random, int count) {
        long[] values = new long[count];
        for (int i = 0; i < count; i++) values[i] = random.nextLong();
        return values;
    }

    // ------------------------------------------------------------------ actions

    public static int action(int column, boolean bomb) {
        return 2 * column + (bomb ? 1 : 0);
    }

    public static int columnOf(int action) {
        return action >> 1;
    }

    public static boolean isBomb(int action) {
        return (action & 1) != 0;
    }

    public int actionCount() {
        return 2 * columns;
    }

    // ------------------------------------------------------------------ queries

    public boolean isLegal(int action) {
        if (action < 0 || action >= actionCount() || columnHeight[columnOf(action)] == rows) return false;
        return !isBomb(action) || bombsEnabled && bombCooldown[sideToMove] == 0;
    }

    public boolean isFull() {
        return quarkCount == cellCount;
    }

    /**
     * The winner, or -1. The player who just moved is checked first. Assumption: a line formed by
     * quarks falling after a blast also wins (an open question).
     */
    public int winner() {
        return hasLine() ? 1 - sideToMove : -1;
    }

    /** True when either player has a complete line: the game is over, won by the last mover. */
    public boolean hasLine() {
        return completedLines[0] + completedLines[1] > 0;
    }

    /** True when the player would complete a line by dropping in this column now. Needs tracksThreats. */
    public boolean completesLine(int player, int column) {
        int height = columnHeight[column];
        return height < rows && threatCount[player * cellCount + column * rows + height] > 0;
    }

    /** Hash of everything that affects the game's future: quarks, side to move and bomb cooldowns. */
    public long positionKey() {
        long key = sideToMove == 0 ? boardHash : boardHash ^ zobristSideToMove;
        if (!bombsEnabled) return key;
        return key ^ zobristCooldown[bombCooldown[0]] ^ zobristCooldown[BOMB_COOLDOWN + 1 + bombCooldown[1]];
    }

    // ------------------------------------------------------------------ playing

    /** Plays an action for the side to move: drop, and with the bomb bit set, blast the 3x3 around the quark. */
    public void play(int action) {
        if (!isBomb(action)) {
            drop(columnOf(action));
            return;
        }
        int player = sideToMove, column = columnOf(action), height = columnHeight[column];
        addQuark(column, player); // the bomb is decided before the drop, so it blasts even a winning quark
        for (int c = Math.max(0, column - 1); c <= Math.min(columns - 1, column + 1); c++) blast(c, height - 1, height + 1);
        bombCooldown[player] = BOMB_COOLDOWN;
        endTurn(player);
    }

    /** A plain drop for the side to move. Returns the mover's previous cooldown, which {@link #undoDrop} needs. */
    public int drop(int column) {
        int player = sideToMove, previousCooldown = bombCooldown[player];
        addQuark(column, player);
        endTurn(player);
        return previousCooldown;
    }

    public void undoDrop(int column, int previousCooldown) {
        sideToMove = 1 - sideToMove;
        bombCooldown[sideToMove] = previousCooldown;
        removeTopQuark(column);
    }

    /** Passes the turn without a move, as the arena does when a bot is too slow; it still counts toward the cooldown. */
    public void skipTurn() {
        endTurn(sideToMove);
    }

    private void endTurn(int player) {
        if (bombCooldown[player] > 0) bombCooldown[player]--;
        sideToMove = 1 - player;
    }

    /** Removes the quarks at heights from..to of a column; the quarks above fall down. */
    private void blast(int column, int from, int to) {
        from = Math.max(from, 0);
        to = Math.min(to, columnHeight[column] - 1);
        if (from > to) return;
        int above = columnHeight[column] - 1 - to;
        for (int i = 0; i < above; i++) fallingQuarks[i] = removeTopQuark(column);
        for (int i = from; i <= to; i++) removeTopQuark(column);
        for (int i = above - 1; i >= 0; i--) addQuark(column, fallingQuarks[i]);
    }

    // ------------------------------------------------------------------ the two incremental updates

    void addQuark(int column, int player) {
        int cell = column * rows + columnHeight[column]++;
        owner[cell] = player;
        quarkCount++;
        boardHash ^= zobristQuark[2 * cell + player];
        for (int line : linesThroughCell[cell]) {
            if (tracksThreats) adjustThreat(line, cell, -1); // this cell was the line's gap
            staticScore -= lineValue(line);
            if (++lineQuarks[2 * line + player] == lineLength) completedLines[player]++;
            staticScore += lineValue(line);
            if (tracksThreats) adjustThreat(line, FIND_GAP, +1);
        }
    }

    int removeTopQuark(int column) {
        int cell = column * rows + --columnHeight[column], player = owner[cell];
        quarkCount--;
        boardHash ^= zobristQuark[2 * cell + player];
        for (int line : linesThroughCell[cell]) {
            if (tracksThreats) adjustThreat(line, FIND_GAP, -1);
            staticScore -= lineValue(line);
            if (lineQuarks[2 * line + player]-- == lineLength) completedLines[player]--;
            staticScore += lineValue(line);
            if (tracksThreats) adjustThreat(line, cell, +1); // this cell becomes the line's gap
        }
        owner[cell] = EMPTY;
        return player;
    }

    /** Value of a line from player 0's point of view: only lines open to one player count. */
    int lineValue(int line) {
        int quarks0 = lineQuarks[2 * line], quarks1 = lineQuarks[2 * line + 1];
        int[] weights = lineWeight[lineDirection[line]];
        if (quarks1 == 0) return weights[quarks0];
        if (quarks0 == 0) return -weights[quarks1];
        return 0;
    }

    /** The player for whom this line is one quark short and still open, or -1. */
    private int threatenedBy(int line) {
        int quarks0 = lineQuarks[2 * line], quarks1 = lineQuarks[2 * line + 1];
        if (quarks0 == lineLength - 1 && quarks1 == 0) return 0;
        if (quarks1 == lineLength - 1 && quarks0 == 0) return 1;
        return -1;
    }

    private static final int FIND_GAP = -1;

    /** If the line is a threat, adds delta to its gap's threat count (the gap is looked up if not given). */
    private void adjustThreat(int line, int gap, int delta) {
        int player = threatenedBy(line);
        if (player < 0) return;
        if (gap == FIND_GAP) gap = emptyCellOf(line);
        threatCount[player * cellCount + gap] += delta;
    }

    /** The first empty cell of a line (used when the line has exactly one). */
    private int emptyCellOf(int line) {
        for (int cell : cellsOfLine[line]) if (owner[cell] == EMPTY) return cell;
        return -1;
    }

    // ------------------------------------------------------------------ setting up a position

    public void clear() {
        Arrays.fill(owner, EMPTY);
        Arrays.fill(columnHeight, 0);
        Arrays.fill(lineQuarks, 0);
        Arrays.fill(threatCount, 0);
        completedLines[0] = completedLines[1] = 0;
        bombCooldown[0] = bombCooldown[1] = bombsEnabled ? FIRST_BOMB_WAIT : 0;
        sideToMove = quarkCount = 0;
        boardHash = 0;
        staticScore = 0;
        for (int line = 0; line < lineCount; line++) staticScore += lineValue(line);
    }

    /** Sets up an arena field (row 0 at the top): {@code playerZero}'s quarks become player 0, who is to move. */
    public void load(SpaceOwnership[][] field, SpaceOwnership playerZero) {
        clear();
        for (int column = 0; column < columns; column++)
            for (int row = rows - 1; row >= 0 && field[row][column] != SpaceOwnership.EMPTY; row--)
                addQuark(column, field[row][column] == playerZero ? 0 : 1);
    }

    /** Replaces the line weights (per direction class) and rescores the current position. */
    public void setLineWeights(int[][] weights) {
        for (int d = 0; d < lineWeight.length; d++) System.arraycopy(weights[d], 0, lineWeight[d], 0, lineWeight[d].length);
        staticScore = 0;
        for (int line = 0; line < lineCount; line++) staticScore += lineValue(line);
    }

    // ------------------------------------------------------------------ saving and restoring

    public int[] saveState() {
        int[] state = new int[stateSize];
        saveState(state);
        return state;
    }

    public void saveState(int[] state) {
        int at = copyInto(state, 0, owner);
        at = copyInto(state, at, columnHeight);
        at = copyInto(state, at, lineQuarks);
        at = copyInto(state, at, threatCount);
        state[at] = completedLines[0];
        state[at + 1] = completedLines[1];
        state[at + 2] = bombCooldown[0];
        state[at + 3] = bombCooldown[1];
        state[at + 4] = sideToMove;
        state[at + 5] = quarkCount;
        state[at + 6] = staticScore;
        state[at + 7] = (int) boardHash;
        state[at + 8] = (int) (boardHash >>> 32);
    }

    public void restoreState(int[] state) {
        int at = copyFrom(state, 0, owner);
        at = copyFrom(state, at, columnHeight);
        at = copyFrom(state, at, lineQuarks);
        at = copyFrom(state, at, threatCount);
        completedLines[0] = state[at];
        completedLines[1] = state[at + 1];
        bombCooldown[0] = state[at + 2];
        bombCooldown[1] = state[at + 3];
        sideToMove = state[at + 4];
        quarkCount = state[at + 5];
        staticScore = state[at + 6];
        boardHash = state[at + 7] & 0xffffffffL | (long) state[at + 8] << 32;
    }

    private static int copyInto(int[] state, int at, int[] part) {
        System.arraycopy(part, 0, state, at, part.length);
        return at + part.length;
    }

    private static int copyFrom(int[] state, int at, int[] part) {
        System.arraycopy(state, at, part, 0, part.length);
        return at + part.length;
    }
}
