package ch.qc.meelik.bots;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.Move;
import ch.cern.quantumconnect.core.QuantumField;
import ch.cern.quantumconnect.core.SpaceOwnership;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Random;

/**
 * SerenaBot, the QuantumConnect bot of team WilyQuarks: an alpha-beta search over a fast model of
 * the game, plus a book of lessons learned from its own lost games.
 *
 * <h2>The game</h2>
 * Quarks drop into columns and stack from the bottom. Six in a row, horizontally, vertically or
 * diagonally, wins. Instead of a plain drop a player may bomb: the quark drops and the 3x3 cells
 * around it, itself included, are cleared, and the quarks above fall down. A player may bomb once
 * every {@link #BOMB_COOLDOWN} of their own turns, the first time after {@link #FIRST_BOMB_WAIT}.
 * After every turn, a line anywhere on the board wins for the player who just moved, whoever's line
 * it is.
 *
 * <h2>The model</h2>
 * Cells are numbered column by column, {@code cell = column * rows + height}, with height 0 at the
 * bottom. Player 0 is always this bot. A move, called an <i>action</i>, packs the column and the bomb
 * decision: {@code action = 2 * column + (bomb ? 1 : 0)}.
 *
 * <p>A <i>line</i> is any run of six cells. For every line the board keeps how many quarks each
 * player has in it, and updates the counts quark by quark. Everything else follows from the counts:
 * <ul>
 *   <li>the win test: a count reaching six;
 *   <li>the evaluation: a line open to one player only is worth a weight to that player;
 *   <li>the threats: empty cells that would complete a line.
 * </ul>
 * A bomb is carried out as single-quark removals and additions, so it needs no bookkeeping of its own.
 *
 * <h2>The search</h2>
 * Negamax with alpha-beta pruning, deepened one ply at a time until the time budget runs out. A
 * transposition table reuses positions reached by other move orders, and moves are tried best guess
 * first: the table's move, then moves that caused cut-offs before, then centre columns. The threats
 * force moves: a side that can complete a line wins; one facing two threats without its bomb has
 * lost; one facing a single threat must block it or bomb.
 *
 * <h2>The network (off by default)</h2>
 * A small integer network can take over the evaluation ({@link #useNetwork}). Its inputs are 0/1
 * features seen from one player: own and opponent quarks, playable cells, own and opponent threat
 * cells, both bomb cooldowns in four buckets, and whether that player moved first. The first layer
 * sums the weight rows of the active features into an accumulator per player, kept up to date by
 * the two incremental updates as quarks come and go; a clipped ReLU and one output weight per hidden
 * unit give the score. The weights are trained offline (tools/qc-net) and embedded as base64.
 *
 * <h2>The loss book</h2>
 * After each tournament round we replay every game we lost and search our moves before the loss
 * with far more time than a turn allows. Where that search finds a clearly better move, the position
 * (its hash, {@link #positionKey}) and the better move go into {@link #LOSS_BOOK}. In a game, a
 * position found in the book is answered from it without searching.
 *
 * <p>The bot is this one class and uses only the engine API, java.lang and java.util.
 *
 * @author WilyQuarks
 */
public final class SerenaBot implements Algorithm {

    // ------------------------------------------------------------------ rules

    public static final int EMPTY = -1;
    /** Quarks in a row that win. */
    public static final int LINE_LENGTH = 6;
    /** Own turns from one bomb to the next. */
    public static final int BOMB_COOLDOWN = 10;
    /** Own turns before the first bomb. */
    public static final int FIRST_BOMB_WAIT = 10;
    /** Line directions as (column step, height step). */
    static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    /** Direction classes for the weights; both diagonals share one. */
    static final int HORIZONTAL = 0, VERTICAL = 1, DIAGONAL = 2;

    // ------------------------------------------------------------------ evaluation

    /** Value of a line holding 0..6 quarks of one player and none of the other, per direction class, evolved by self-play. */
    static final int[][] LINE_WEIGHT = {
            {0, 41, 131, 241, 540, 6367, 0},
            {0, 17, 75, 204, 658, 8300, 0},
            {0, 15, 105, 261, 615, 7863, 0}};
    /** Bonus for having the bomb ready while the opponent hasn't. */
    static final int BOMB_READY_BONUS = 20;
    /** This bot's weights and bonus: the constants above, unless a tuning tool sets others. */
    public int[][] lineWeight = LINE_WEIGHT;
    public int bombReadyBonus = BOMB_READY_BONUS;

    // ------------------------------------------------------------------ search constants

    /** Score of a won position; a win in n plies scores WIN - n, so faster wins are preferred. */
    public static final int WIN = 1 << 28;
    public static final int ME = 0, OPPONENT = 1;
    public static final int MAX_PLY = 64;
    public static final int NO_ACTION = -1;
    /** After this much thinking in one match, think an eighth as long per move (a match allows 30 s). */
    static final long MATCH_TIME_GUARD_NANOS = 20_000_000_000L;

    // ------------------------------------------------------------------ board geometry

    public int rows, columns;
    int cellCount, lineCount;
    /** The cells of each line, the lines through each cell, and each line's direction class. */
    int[][] cellsOfLine, linesThroughCell;
    int[] lineDirection;
    /** Columns ordered from the centre outwards. */
    int[] columnsCentreOut;

    // ------------------------------------------------------------------ position

    /** Owner of each cell: player 0, player 1 or EMPTY. */
    public int[] owner;
    int[] columnHeight;
    /** Own turns each player must still wait before they may bomb. */
    public final int[] bombCooldown = new int[2];
    /** Complete lines of each player; above zero means the game is over. */
    final int[] completedLines = new int[2];
    /** Quarks of each player in each line, at {@code 2 * line + player}. */
    int[] lineQuarks;
    /** Lines a cell would complete for a player, at {@code player * cellCount + cell}. */
    int[] threatCount;
    public int sideToMove;
    int quarkCount;
    /** Sum of the line values, from player 0's point of view. */
    int staticScore;

    // ------------------------------------------------------------------ hashing and undo

    long[] zobristQuark, zobristCooldown;
    long zobristSideToMove, boardHash;
    /** Scratch space for the quarks above a blast. */
    int[] fallingQuarks;
    int stateSize;

    // ------------------------------------------------------------------ search memory (about 8 MB at the default table size)

    public static final int TABLE_BITS = 19;
    static final int EXACT = 0, LOWER_BOUND = 1, UPPER_BOUND = 2;
    long[] tableKey = new long[1 << TABLE_BITS], tableEntry = new long[1 << TABLE_BITS];
    final int[][] actionLists = new int[MAX_PLY][], undoStates = new int[MAX_PLY][];
    /** How often each action of each player caused a cut-off, weighted by depth. */
    int[][] historyScore;
    public long budgetNanos = 40_000_000L;
    /** Deepest iteration of the search; a tuning tool may fix it. */
    public int maxDepth = MAX_PLY - 2;
    long deadlineNanos;
    public long nodesSearched;
    boolean outOfTime;
    /** Score of the last completed search, for the side that was to move. */
    public int lastScore;

    // ------------------------------------------------------------------ match memory

    SpaceOwnership myColour;
    int plannedAction = NO_ACTION;
    int myCooldown = FIRST_BOMB_WAIT, opponentCooldown = FIRST_BOMB_WAIT;
    int myTurns;
    long timeUsedNanos;
    /** The player who moved first in this match: player 0 unless we are the second player. */
    public int firstMover = ME;

    // ------------------------------------------------------------------ construction

    /** The standard 8x12 board. The arena calls this one; the bot resizes to the arena's field. */
    public SerenaBot() {
        this(8, 12);
    }

    /** An empty board of the given size. */
    public SerenaBot(int rows, int columns) {
        setUp(rows, columns);
    }

    /** Builds an empty board of this size and sizes the search memory to it. */
    private void setUp(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
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

        // A fixed seed: the loss book stores position hashes, so they must not change between runs.
        Random random = new Random(20261102L);
        zobristQuark = randomLongs(random, 2 * cellCount);
        zobristCooldown = randomLongs(random, 2 * (BOMB_COOLDOWN + 1));
        zobristSideToMove = random.nextLong();

        owner = new int[cellCount];
        columnHeight = new int[columns];
        lineQuarks = new int[2 * lineCount];
        threatCount = new int[2 * cellCount];
        fallingQuarks = new int[rows];
        networkOn = useNetwork && loadNetwork();
        stateSize = owner.length + columnHeight.length + lineQuarks.length + threatCount.length + 9 + (networkOn ? 2 * hidden : 0);
        clear();

        for (int ply = 0; ply < MAX_PLY; ply++) {
            actionLists[ply] = new int[2 * columns];
            undoStates[ply] = null;
        }
        historyScore = new int[2][2 * columns];
        Arrays.fill(tableKey, 0);
    }

    /** The cells of the line starting at (column, height) in a direction, or null if it leaves the board. */
    private int[] lineFrom(int column, int height, int[] direction) {
        int lastColumn = column + (LINE_LENGTH - 1) * direction[0], lastHeight = height + (LINE_LENGTH - 1) * direction[1];
        if (lastColumn < 0 || lastColumn >= columns || lastHeight < 0 || lastHeight >= rows) return null;
        int[] line = new int[LINE_LENGTH];
        for (int i = 0; i < LINE_LENGTH; i++) line[i] = (column + i * direction[0]) * rows + height + i * direction[1];
        return line;
    }

    /** From the cells of each line to the lines through each cell. */
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

    /** The columns sorted by distance from the centre (insertion sort, so ties keep left before right). */
    private static int[] centreOut(int columns) {
        int[] order = new int[columns];
        for (int column = 0; column < columns; column++) order[column] = column;
        for (int i = 1; i < columns; i++)
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

    private static long[] randomLongs(Random random, int count) {
        long[] values = new long[count];
        for (int i = 0; i < count; i++) values[i] = random.nextLong();
        return values;
    }

    // ------------------------------------------------------------------ the arena interface

    /**
     * Picks the column for this turn. The arena asks for the column, then whether to bomb, both
     * before anything is placed; the bot decides both here and remembers the bomb decision.
     */
    @Override
    public int accelerateQuark(QuantumField field) {
        long started = System.nanoTime();
        SpaceOwnership[][] grid = field.getField();
        if (myColour == null) myColour = whoAmI(field, grid);
        firstMover = myColour == SpaceOwnership.FIRST_PLAYER ? ME : OPPONENT;
        if (grid.length != rows || grid[0].length != columns) setUp(grid.length, grid[0].length);
        load(grid, myColour);
        updateOpponentCooldown(field.getMoveHistory());
        myTurns++;
        bombCooldown[ME] = myCooldown;
        bombCooldown[OPPONENT] = opponentCooldown;

        int booked = bookAction(positionKey());
        boolean bookPlayable = booked != NO_ACTION && isLegal(booked);
        plannedAction = bookPlayable ? booked
                : chooseAction(timeUsedNanos > MATCH_TIME_GUARD_NANOS ? budgetNanos / 8 : budgetNanos);
        timeUsedNanos += System.nanoTime() - started;
        return columnOf(plannedAction);
    }

    /** Whether this turn's quark is a bomb, as decided in {@link #accelerateQuark}. */
    @Override
    public boolean useSpecialMove(QuantumField field) {
        if (plannedAction == NO_ACTION) return false;
        play(plannedAction); // counts the turn toward our cooldown, and resets it after a bomb
        myCooldown = bombCooldown[ME];
        return isBomb(plannedAction);
    }

    /** Our colour, from any quark on the field; the first player's if the field is empty. */
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
     * Counts the opponent's last turn toward its cooldown. The move history marks every bomb; a turn
     * the opponent lost to a timeout adds no entry but counts all the same.
     */
    void updateOpponentCooldown(List<Move> history) {
        boolean opponentHasMoved = myTurns > 0 || myColour == SpaceOwnership.SECOND_PLAYER;
        if (!opponentHasMoved) return;
        Move last = history.isEmpty() ? null : history.get(history.size() - 1);
        if (last != null && last.player() != myColour && last.isSpecialMove()) opponentCooldown = BOMB_COOLDOWN - 1;
        else if (opponentCooldown > 0) opponentCooldown--;
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

    // ------------------------------------------------------------------ board queries

    public boolean isLegal(int action) {
        if (action < 0 || action >= 2 * columns || columnHeight[columnOf(action)] == rows) return false;
        return !isBomb(action) || bombCooldown[sideToMove] == 0;
    }

    /** True when either player has a complete line: the game is over, won by the last mover. */
    public boolean hasLine() {
        return completedLines[0] + completedLines[1] > 0;
    }

    /** True when the player would complete a line by dropping in this column now. */
    boolean completesLine(int player, int column) {
        int height = columnHeight[column];
        return height < rows && threatCount[player * cellCount + column * rows + height] > 0;
    }

    /** Zobrist hash of everything that decides the game's future: quarks, side to move, bomb cooldowns. */
    public long positionKey() {
        long key = sideToMove == 0 ? boardHash : boardHash ^ zobristSideToMove;
        return key ^ zobristCooldown[bombCooldown[0]] ^ zobristCooldown[BOMB_COOLDOWN + 1 + bombCooldown[1]];
    }

    // ------------------------------------------------------------------ playing

    /** Plays an action for the side to move: a drop, or with the bomb bit, a drop that blasts the 3x3 around it. */
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

    /** A plain drop for the side to move. Returns the mover's previous cooldown, for {@link #undoDrop}. */
    int drop(int column) {
        int player = sideToMove, previousCooldown = bombCooldown[player];
        addQuark(column, player);
        endTurn(player);
        return previousCooldown;
    }

    void undoDrop(int column, int previousCooldown) {
        sideToMove = 1 - sideToMove;
        bombCooldown[sideToMove] = previousCooldown;
        removeTopQuark(column);
    }

    /** Passes the turn, as the arena does when a bot is too slow; it still counts toward the cooldown. */
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

    /** Puts a quark on top of a column and updates every line through its cell. */
    void addQuark(int column, int player) {
        int cell = column * rows + columnHeight[column]++;
        owner[cell] = player;
        quarkCount++;
        boardHash ^= zobristQuark[2 * cell + player];
        if (networkOn) addRows(quarkRows, 2 * cell * hidden, (2 * cell + 1) * hidden, player);
        for (int line : linesThroughCell[cell]) {
            adjustThreat(line, cell, -1); // this cell was the line's gap
            staticScore -= lineValue(line);
            if (++lineQuarks[2 * line + player] == LINE_LENGTH) completedLines[player]++;
            staticScore += lineValue(line);
            adjustThreat(line, FIND_GAP, +1);
        }
    }

    /** Takes the top quark off a column, updates every line through its cell, and returns its owner. */
    int removeTopQuark(int column) {
        int cell = column * rows + --columnHeight[column], player = owner[cell];
        quarkCount--;
        boardHash ^= zobristQuark[2 * cell + player];
        if (networkOn) subtractRows(quarkRows, 2 * cell * hidden, (2 * cell + 1) * hidden, player);
        for (int line : linesThroughCell[cell]) {
            adjustThreat(line, FIND_GAP, -1);
            staticScore -= lineValue(line);
            if (lineQuarks[2 * line + player]-- == LINE_LENGTH) completedLines[player]--;
            staticScore += lineValue(line);
            adjustThreat(line, cell, +1); // this cell becomes the line's gap
        }
        owner[cell] = EMPTY;
        return player;
    }

    /** Value of a line from player 0's point of view: only a line open to one player counts. */
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
        if (quarks0 == LINE_LENGTH - 1 && quarks1 == 0) return 0;
        if (quarks1 == LINE_LENGTH - 1 && quarks0 == 0) return 1;
        return -1;
    }

    private static final int FIND_GAP = -1;

    /** If the line is a threat, adds delta to its gap's threat count (the gap is looked up if not given). */
    private void adjustThreat(int line, int gap, int delta) {
        int player = threatenedBy(line);
        if (player < 0) return;
        if (gap == FIND_GAP) gap = emptyCellOf(line);
        int count = threatCount[player * cellCount + gap] += delta;
        if (networkOn && count == (delta > 0 ? 1 : 0)) { // the cell became or stopped being a threat
            int ownRow = (OWN_THREAT * cellCount + gap) * hidden, opponentRow = (OPPONENT_THREAT * cellCount + gap) * hidden;
            if (delta > 0) addRows(inputWeights, ownRow, opponentRow, player);
            else subtractRows(inputWeights, ownRow, opponentRow, player);
        }
    }

    /** The first empty cell of a line. */
    private int emptyCellOf(int line) {
        for (int cell : cellsOfLine[line]) if (owner[cell] == EMPTY) return cell;
        return -1;
    }

    // ------------------------------------------------------------------ setting up a position

    /** Empties the board: no quarks, both players before their first bomb, player 0 to move. */
    void clear() {
        Arrays.fill(owner, EMPTY);
        Arrays.fill(columnHeight, 0);
        Arrays.fill(lineQuarks, 0);
        Arrays.fill(threatCount, 0);
        completedLines[0] = completedLines[1] = 0;
        bombCooldown[0] = bombCooldown[1] = FIRST_BOMB_WAIT;
        sideToMove = quarkCount = 0;
        boardHash = 0;
        staticScore = 0;
        for (int line = 0; line < lineCount; line++) staticScore += lineValue(line);
        if (networkOn) for (short[] sums : accumulator) System.arraycopy(emptyBoardRow, 0, sums, 0, hidden);
    }

    /** Sets up an arena field (row 0 at the top): {@code playerZero}'s quarks become player 0, who is to move. */
    public void load(SpaceOwnership[][] field, SpaceOwnership playerZero) {
        clear();
        for (int column = 0; column < columns; column++)
            for (int row = rows - 1; row >= 0 && field[row][column] != SpaceOwnership.EMPTY; row--)
                addQuark(column, field[row][column] == playerZero ? 0 : 1);
    }

    // ------------------------------------------------------------------ saving and restoring (to undo a bomb)

    void saveState(int[] state) {
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
        if (networkOn)
            for (int i = 0; i < hidden; i++) {
                state[at + 9 + i] = accumulator[0][i];
                state[at + 9 + hidden + i] = accumulator[1][i];
            }
    }

    void restoreState(int[] state) {
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
        if (networkOn)
            for (int i = 0; i < hidden; i++) {
                accumulator[0][i] = (short) state[at + 9 + i];
                accumulator[1][i] = (short) state[at + 9 + hidden + i];
            }
    }

    private static int copyInto(int[] state, int at, int[] part) {
        System.arraycopy(part, 0, state, at, part.length);
        return at + part.length;
    }

    private static int copyFrom(int[] state, int at, int[] part) {
        System.arraycopy(state, at, part, 0, part.length);
        return at + part.length;
    }

    // ------------------------------------------------------------------ the loss book

    /**
     * Positions from our lost games, each with the move a deep offline search prefers. One entry per
     * {@code ;}: board size, position key in hex, action (the column, {@code b} for a bomb). Split into
     * chunks because a string constant holds at most 64 KB. Generated; do not edit by hand.
     */
    // LOSS BOOK BEGIN (generated by qc.sh learn)
    static final String[] LOSS_BOOK = {
            "15x30 932b0e9364d6fe0c 18;15x30 92382d2081853b72 12;15x30 3d1c8eea0249f64f 26;15x30 3884c5e144be7c05 11;15x30 4421266e6206a710 12;15x30 e36afd9b38c0a153 14;15x30 7413ca505f75a080 5;15x30 1b239c79eae243af 13;15x30 0c9c0ff82d3b4c69 12;15x30 ce407cc4bc0bb54e 12;15x30 5a524f0c377d069e 11;15x30 24adcc89ea8396c6 2;15x30 73b66f65fd39ca77 2;15x30 e6b989518f1e743e 2;15x30 797a53d710f909fd 11;15x30 ccc5aa438dd1f6a6 11;15x30 f720a2910eecf2d9 11;15x30 4bdf36e477c5c089 15;15x30 34e4b228f1a7f79a 11;15x30 10484206bcd36e4a 8;15x30 78cebae845423dbc 15;15x30 fa5162239353f4e7 10;15x30 368a813bcf491982 1;15x30 805f44e1a039031d 20;15x30 15c078bd63e6fb19 17;15x30 1a26907beafff58b 17;15x30 8b920d506b2933a7 17;15x30 fa328e40b5dcc236 17;15x30 0cc9ee5c0e331e75 17;15x30 3796ccd014f04a99 17;15x30 0e08818d513da5c0 12;15x30 dd8965f7963022b9 11;15x30 5520b702e64c98e8 12;15x30 586c35e777a122bc 12;15x30 f0b11f8769eaa894 12;15x30 4a94ea8a168cb0f9 9b;15x30 786a03bbf68b126f 9b;15x30 2c451f03b44b58d5 9b;15x30 78c3a124fd2793e7 9b;15x30 011041f74c5be783 5;15x30 ce0554a97b050072 22;15x30 b51e2e57441e0577 9b;15x30 5708a12ed49e9267 12b;15x30 1a687bd0ff7a4d6a 12b;15x30 2b6cea86b7c54e68 12b;15x30 dc7b4c52f259b75b 11;15x30 a30e16801d5f2be6 11;15x30 d687323e2dcbc0ea 18;15x30 3397b98bcc4a8189 12;15x30 2482829221816e5d 21b;15x30 7f8c76e3b209f675 29;15x30 a95304f456a10b80 11;15x30 4b9de0477d974315 23;15x30 d3bd696ce15476e2 8;15x30 527c17bed4f1b349 7;15x30 4f07809c2071110c 7;15x30 d18019c2ba79594c 7;15x30 c1b9715e3e3cef71 1;15x30 2cbcc50822cf9f5f 4;15x30 a9adec14bad53fd2 4;15x30 8a5d04177527435a 10;15x30 d2c5538426401b51 10;15x30 d56ed27ded32ff94 18;15x30 3e47bb238001d260 3;15x30 6920751cd12c9972 9;15x30 b9c8e6cb108f0c98 10;15x30 48a1ada210707c63 10;15x30 b2390363237ff708 10;15x30 f7855c4cf43fc461 10;15x30 736fcebf5f64cafc 10;15x30 68d0f8a7c0eea675 4;15x30 9582c92cfc84f49c 3;15x30 74e29a4657058cac 18;15x30 87405f6517898d0c 3;15x30 3361e73adb8d6ff7 12;15x30 1d23ce6425714c39 26;15x30 6cebc2009d310cac 1;15x30 473e9e169f646fdf 2;15x30 980bb0ea879eb4f3 10;15x30 27830eeee972fc74 10;15x30 63049895dbeae80a 10;15x30 b7c56ae70017dc30 12;15x30 a44c5b4e7f560510 16;15x30 e344e11136eb070b 11;15x30 8ef9f92715a8a248 8b;15x30 a6d41f1d66fc6d45 10b;15x30 2f42ab1f70680d13 10b;15x30 a38787b2a4699a35 10b;15x30 a31b310bae1efd05 10b;15x30 4ffbc6ea7204986f 8;15x30 bc7e14be4f98257a 2;15x30 8c44dd088487cdc0 25;15x30 4c4efdb6d5417524 22;15x30 7ff1e4cbe06e2136 13;15x30 e68e57eb0c211583 22;15x30 fbcfdfee7ef4c572 19;15x30 215ddcc755d0dcee 26;15x30 7494c587b681284b 7;15x30 5bed41f61c976f88 29;15x30 ed6c9c98da2d01b6 29;15x30 74bdf351f1a4cc9a 29;15x30 fc419ee31c8a4ab1 25;15x30 7dff2a727cd3036f 25;15x30 0e607aa438a375c9 13;15x30 17cfd63022929fc5 13;15x30 43626b6791a90d44 13;15x30 857e9aba1cd20059 24;15x30 ae3e1494580c286f 13;15x30 400a03055084e09a 3b;15x30 e590a2aef7b19906 3b;15x30 e8efd1d1058299f1 3b;15x30 1d22176943b0cadb 3b;15x30 f10c7e9a9961693e 3b;15x30 3ac0e82399d2d61d 27b;15x30 088fe34b8cf8480b 18;15x30 18ad00213e90b824 18;15x30 fc5d9432ff3b5237 13;15x30 62da4cf187783d34 19;15x30 b56d8a41c09abc60 13;15x30 b0edf2eb29b2840a 19;15x30 c178ec713b87e0a5 13;15x30 a870645ed8f980a5 19;15x30 2837b1a629cebd49 19;15x30 9164d6eab6cb1fa3 19;15x30 775780d3aeb97075 7;15x30 ce80d0a400d38e52 21;15x30 2475474b3716e209 17;15x30 f917272ccfe85e4e 20;15x30 c62a3c95b0a6eda1 13;15x30 bbb9a94190822fb9 22;15x30 8ebc28913c0da43c 7;15x30 35ed60136d90db44 23;15x30 19fc126a936282fd 21;15x30 a2b8663341a762c9 23b;15x30 442992c1bf991813 23b;15x30 6f1a7822f11f3cf6 17;15x30 d43856861b83ba54 17;15x30 d3aa2eba62bdae74 17;15x30 da0062d2131a4bf8 10;15x30 bee1cbf897100357 13;15x30 a23571a471f471e0 23;15x30 f1fb10e728ac5ce0 25;15x30 f00913ca67e265cd 12;15x30 4d21ff2d3822f10a 18;15x30 6eb80342934d5299 8;15x30 677ce1e5e27dcbb4 8;15x30 60876d221d3a0374 18;15x30 86c0e08a81a0206b 6;15x30 1b6f81d8788f6204 26;15x30 1f77a64b13ed6889 6;15x30 e5167f910444869d 6;15x30 012b6008af2ce4f2 18;15x30 685712cede5efbbd 13;15x30 14bc2ab951bfefb2 18;15x30 b966368b018fe6de 15;15x30 eaacf2138e1aad10 19;15x30 f24f741bc61412a2 19;15x30 fca8cac212ef0e76 15;15x30 1d1ab7b822a47682 18;15x30 a9baa12596b0997c 20;15x30 a6d6ee0dfb2d7e56 20;15x30 367aaf9b1693b105 8;15x30 4192e4d657a46dc7 24;15x30 537ec2f0150245cf 2;15x30 b74636b7ba975981 12;15x30 48407d1667116577 12;15x30 778ac39a656a00bb 11;15x30 ea1758d4a2be319c 10;15x30 e098cd14e7f14457 5b;15x30 ef3e5e60dd7984d5 5b;15x30 59aa6fa51fcc702f 5b;15x30 fb94f8675798ba69 5b;15x30 9f2596d5cb024d11 8;15x30 f7963e7312d6d085 8;15x30 d98be0de49621a3f 13;15x30 493b7f120871a15a 8;15x30 680bac500e369c0b 8;15x30 305419263fc02222 15;15x30 adc0d540012f184f 10b;15x30 bfd9979407c5aaaf 11;15x30 f3b9bbb050e7141d 11;15x30 edfd8f5f054ce234 9b;15x30 8efc2c5480363f14 15;15x30 ca3163e3bc4f1d9f 15;",
    };
    // LOSS BOOK END

    /** This board size's entries, sorted by key for binary search; parsed on first use. */
    long[] bookKeys = new long[0];
    int[] bookActions = new int[0];
    int bookRows = -1, bookColumns = -1;

    /** The booked action for a position, or NO_ACTION. */
    public int bookAction(long key) {
        if (bookRows != rows || bookColumns != columns) loadBook();
        int at = Arrays.binarySearch(bookKeys, key);
        return at >= 0 ? bookActions[at] : NO_ACTION;
    }

    /** Parses the entries of LOSS_BOOK for this board size. */
    private void loadBook() {
        // String.concat, not "+": javac compiles "+" on strings to a java.lang.invoke call, which the arena forbids.
        String size = String.valueOf(rows).concat("x").concat(String.valueOf(columns));
        int count = 0;
        long[] keys = new long[64];
        int[] actions = new int[64];
        for (String chunk : LOSS_BOOK)
            for (String entry : chunk.split(";")) {
                String[] parts = entry.trim().split(" ");
                if (parts.length != 3 || !parts[0].equals(size)) continue;
                if (count == keys.length) {
                    keys = Arrays.copyOf(keys, 2 * count);
                    actions = Arrays.copyOf(actions, 2 * count);
                }
                keys[count] = Long.parseUnsignedLong(parts[1], 16);
                actions[count] = parseAction(parts[2]);
                count++;
            }
        sortByKey(keys, actions, count);
        bookKeys = Arrays.copyOf(keys, count);
        bookActions = Arrays.copyOf(actions, count);
        bookRows = rows;
        bookColumns = columns;
    }

    /** An action written as its column, with {@code b} appended for a bomb: "14", "14b". */
    public static int parseAction(String text) {
        boolean bomb = text.endsWith("b");
        return action(Integer.parseInt(bomb ? text.substring(0, text.length() - 1) : text), bomb);
    }

    /** Insertion sort of the first count keys, carrying the actions along. */
    private static void sortByKey(long[] keys, int[] actions, int count) {
        for (int i = 1; i < count; i++) {
            long key = keys[i];
            int action = actions[i], j = i - 1;
            for (; j >= 0 && keys[j] > key; j--) {
                keys[j + 1] = keys[j];
                actions[j + 1] = actions[j];
            }
            keys[j + 1] = key;
            actions[j + 1] = action;
        }
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
            int fromTable = tableAction(positionKey());
            if (fromTable != NO_ACTION) best = fromTable;
            lastScore = score;
            if (Math.abs(score) > WIN - MAX_PLY) break; // a forced result: deeper search can't change it
        }
        return best;
    }

    private int firstLegalDrop() {
        for (int column : columnsCentreOut) if (columnHeight[column] < rows) return action(column, false);
        return NO_ACTION;
    }

    /** Static evaluation for the side to move: line values and bomb readiness. */
    int evaluate() {
        if (networkOn) return networkScore();
        int score = sideToMove == 0 ? staticScore : -staticScore;
        return score + bombReadyBonus * (isBombReady(sideToMove) - isBombReady(1 - sideToMove));
    }

    private int isBombReady(int player) {
        return bombCooldown[player] == 0 ? 1 : 0;
    }

    // ------------------------------------------------------------------ the network

    /** Evaluate with the network where its board size matches; read when a board is set up. Off: the line weights play. */
    public static boolean useNetwork = false;
    /** The cell planes of the features, from one player's point of view, at {@code plane * cellCount + cell}. */
    public static final int OWN_QUARK = 0, OPPONENT_QUARK = 1, PLAYABLE = 2, OWN_THREAT = 3, OPPONENT_THREAT = 4, CELL_PLANES = 5;
    /** Bomb cooldowns in buckets: ready, 1-3, 4-6 and 7 or more turns to wait. */
    public static final int COOLDOWN_BUCKETS = 4;
    /** Fixed point: 1.0 in the first layer (the clipped ReLU's ceiling) and in the output weights. */
    static final int ACTIVATION_ONE = 64, OUTPUT_ONE = 64;
    /** Score of one unit of the network's output, which is trained as the log-odds of winning. */
    static final int SCORE_PER_LOG_ODDS = 1000;

    public boolean networkOn;
    int hidden;
    /** Weight row of each feature (int8 values), output weights (int16), output bias. */
    short[] inputWeights, outputWeights;
    int outputBias;
    /**
     * Rows summed ahead for speed. The empty board: hidden biases plus the bottom row's playable
     * cells. A quark at a cell, for its owner and for the other player, at {@code (2 * cell + 0 or 1)
     * * hidden}, the column's playable cell moving up included. The cooldowns and first mover, at
     * {@code ((own bucket * buckets + opponent bucket) * 2 + 1 if first mover) * hidden}.
     */
    short[] emptyBoardRow, quarkRows, situationRows;
    /** Per player, the sum of the empty-board row and the rows of that player's active cell features (int16). */
    final short[][] accumulator = new short[2][];

    public int featureCount() {
        return CELL_PLANES * cellCount + 2 * COOLDOWN_BUCKETS + 1;
    }

    static int cooldownBucket(int cooldown) {
        return Math.min(COOLDOWN_BUCKETS - 1, (cooldown + 2) / 3);
    }

    /** Lists the position's active features from a player's point of view into {@code active}; returns how many. */
    public int features(int player, int[] active) {
        int count = 0, opponent = 1 - player;
        for (int cell = 0; cell < cellCount; cell++) {
            if (owner[cell] != EMPTY) active[count++] = (owner[cell] == player ? OWN_QUARK : OPPONENT_QUARK) * cellCount + cell;
            if (threatCount[player * cellCount + cell] > 0) active[count++] = OWN_THREAT * cellCount + cell;
            if (threatCount[opponent * cellCount + cell] > 0) active[count++] = OPPONENT_THREAT * cellCount + cell;
        }
        for (int column = 0; column < columns; column++)
            if (columnHeight[column] < rows) active[count++] = PLAYABLE * cellCount + column * rows + columnHeight[column];
        int extras = CELL_PLANES * cellCount;
        active[count++] = extras + cooldownBucket(bombCooldown[player]);
        active[count++] = extras + COOLDOWN_BUCKETS + cooldownBucket(bombCooldown[opponent]);
        if (player == firstMover) active[count++] = extras + 2 * COOLDOWN_BUCKETS;
        return count;
    }

    /** The network's score for the side to move: its accumulator plus the situation row, clipped ReLU, output. */
    public int networkScore() {
        int mover = sideToMove;
        short[] sums = accumulator[mover];
        int situation = ((cooldownBucket(bombCooldown[mover]) * COOLDOWN_BUCKETS + cooldownBucket(bombCooldown[1 - mover])) * 2
                + (mover == firstMover ? 1 : 0)) * hidden;
        int output = outputBias;
        for (int i = 0; i < hidden; i++) output += Math.min(Math.max(sums[i] + situationRows[situation + i], 0), ACTIVATION_ONE) * outputWeights[i];
        return (int) ((long) output * SCORE_PER_LOG_ODDS / (ACTIVATION_ONE * OUTPUT_ONE));
    }

    /** Adds a row to each accumulator: {@code ownRow} to the player's, {@code opponentRow} to the other player's. */
    private void addRows(short[] rows, int ownRow, int opponentRow, int player) {
        short[] own = accumulator[player], other = accumulator[1 - player];
        for (int i = 0; i < hidden; i++) {
            own[i] = (short) (own[i] + rows[ownRow + i]);
            other[i] = (short) (other[i] + rows[opponentRow + i]);
        }
    }

    private void subtractRows(short[] rows, int ownRow, int opponentRow, int player) {
        short[] own = accumulator[player], other = accumulator[1 - player];
        for (int i = 0; i < hidden; i++) {
            own[i] = (short) (own[i] - rows[ownRow + i]);
            other[i] = (short) (other[i] - rows[opponentRow + i]);
        }
    }

    /**
     * Decodes {@link #NETWORK} if it was trained for this board size, and sums the rows ahead. Layout,
     * little-endian: rows, columns, hidden units, version (a byte each); input weights (int8, feature
     * by feature); hidden biases and output weights (int16); output bias (int32).
     */
    private boolean loadNetwork() {
        byte[] blob = Base64.getDecoder().decode(String.join("", NETWORK));
        if (blob.length < 4 || blob[0] != rows || blob[1] != columns) return false;
        hidden = blob[2];
        int weights = featureCount() * hidden, at = 4 + weights;
        if (blob.length != at + 4 * hidden + 4) return false;
        inputWeights = new short[weights];
        for (int i = 0; i < weights; i++) inputWeights[i] = blob[4 + i];
        emptyBoardRow = new short[hidden];
        outputWeights = new short[hidden];
        for (int i = 0; i < hidden; i++) {
            emptyBoardRow[i] = littleEndianShort(blob, at + 2 * i);
            outputWeights[i] = littleEndianShort(blob, at + 2 * (hidden + i));
        }
        at += 4 * hidden;
        outputBias = littleEndianShort(blob, at) & 0xffff | littleEndianShort(blob, at + 2) << 16;

        for (int column = 0; column < columns; column++) addRow(emptyBoardRow, 0, PLAYABLE * cellCount + column * rows, 1);
        quarkRows = new short[2 * cellCount * hidden];
        for (int cell = 0; cell < cellCount; cell++)
            for (int other = 0; other < 2; other++) {
                int row = (2 * cell + other) * hidden;
                addRow(quarkRows, row, (other == 0 ? OWN_QUARK : OPPONENT_QUARK) * cellCount + cell, 1);
                addRow(quarkRows, row, PLAYABLE * cellCount + cell, -1);
                if ((cell + 1) % rows != 0) addRow(quarkRows, row, PLAYABLE * cellCount + cell + 1, 1);
            }
        int extras = CELL_PLANES * cellCount;
        situationRows = new short[COOLDOWN_BUCKETS * COOLDOWN_BUCKETS * 2 * hidden];
        for (int own = 0; own < COOLDOWN_BUCKETS; own++)
            for (int opponent = 0; opponent < COOLDOWN_BUCKETS; opponent++)
                for (int first = 0; first < 2; first++) {
                    int row = ((own * COOLDOWN_BUCKETS + opponent) * 2 + first) * hidden;
                    addRow(situationRows, row, extras + own, 1);
                    addRow(situationRows, row, extras + COOLDOWN_BUCKETS + opponent, 1);
                    addRow(situationRows, row, extras + 2 * COOLDOWN_BUCKETS, first);
                }
        accumulator[0] = new short[hidden];
        accumulator[1] = new short[hidden];
        return true;
    }

    /** Adds sign times a feature's input weights to {@code sums} from {@code at}. */
    private void addRow(short[] sums, int at, int feature, int sign) {
        for (int i = 0; i < hidden; i++) sums[at + i] = (short) (sums[at + i] + sign * inputWeights[feature * hidden + i]);
    }

    private static short littleEndianShort(byte[] bytes, int at) {
        return (short) (bytes[at] & 0xff | bytes[at + 1] << 8);
    }

    /** The network's weights in base64, in chunks (a string constant holds at most 64 KB). Generated; do not edit by hand. */
    // NETWORK BEGIN (generated by tools/qc-net)
    static final String[] NETWORK = {
            "Dx4gAfgC9wcCBwML/PgF//71BwYF+/kDAwf//AEJ+v7/+gUI/AH9Af//AQcBAgABAPkA/wP8/wAD/wD9/wAAAAH+///9AfwCAAICCP/+AgAA+QMCBPz9AQMC//0ABP7/AP0BAgEAAf8AAAAAAAEBAQABAQAAAAAAAAIAAQABAAAAAf7//wH9BAECAAX+/AIA//wEAgH//gEBAv/+AQH9////AwUAAQEBAAMAAP8BAAD//wMBAP/9AQAD/wAAAf7/AAEAA/8AAQEBAQD//wAAAAABAgIAAP8BAAIAAAAB/wAAAP8BAAAAAgABAAUAAAEBAP4B/wAA/wEAAf8BAAD+AAAAAAIAAQADAQIAAP//AgAAAQIBAAD9AgAD/wABA/7//wAAAv8B/gEBAwAA/gAB////AgIAAf0BAAP/AAED////AP8CAgACAAACAP7/AgEAAAECAAAC/QH/A/8AAP/+//8CAAEDAAT+AAH///4CAAAABAAB/gP/Af8CAAEA/gAAAAP/AAABAQIBBQEB/P8C//8ABAQAAfwDAAX//gEB+/7+AAIE/QH/AwIGAQL7/QX+/gEFBAL/+wQBBv7+AgP6/f7+AwUCAAMCAAL///8BAQAAAAACAAH9AgAD/wAAAf3//wMBAfwB/QIBAQEG//8AAAD6AgAC/f4BAgL//wAD/wAA/wAC/v8H/v/9//wCBv0CAQP+/P8CAf7//gEB//wEAQEF/P39AAQAAP//AAED/wEA/gD/AAH/AP8BAAAA/gIAAQT+Af0BAgIBAQAB//8BAP/9AgEB/v4BAAIA/wEC/wAAAP8D/wEB/wH/AAD/AAD//wACAQD/AAEA////AAIA/wAAAQMBAAX+//v/+QIG/gEBBPz7/wID/v/9AQH/+wUBAQX7/P4AAQEAAQD9AAD/AP8C//4A//8A//8AAQAAAgAAAP0AAAD/AgEBAP4BAAAAAP8A//8AAQAA/v8AAAIBAAAA/wD/AP8AAAAA/wAA/wD/AP/+//7/Af//AAAAAAEAAAD/Af8A/wIBAwECAP0BAAD+AQAB/v8BAAH//wAFAQD//gAC/wD/AgEBAAEAAQEA//8A/wH//gEAAQD/AAABAAAA/wQC/wIA//7/AwIC/wEBAf/9/wED///+AAL/AAMBAQP//P4AAAMAAQAB/wH/AAD/AP4A//8A/wAAAQACAAAAAP7+A/8C///8/wEEAf4BAAD+/f4BBP//+wAB//4FAQEC//7+AP8DAQMAAP39Av//AgP/Af/+AQADAP8BA/7//wAAAf8BAAMBAwAC//wBAP/+AgQB/v0BAQL//wEB/gAA/wIDAgAD/wD///8AAf8BAAL///8B/wD//wABAP8CAAEB/wACAAP9APv/+QAB/gEBBv39/gIC//4AAQIA/QQBAQT9/QEAAv8A/f//AQEAAQACAP//AQH/AAAAAQD9AgABAv7+//8C/gD9//4BAf8BAAP+/v8BAP//AAEBAP4DAQEB/f7+Af0BAQEAAv7+Af///gIBAf/+AQAC/wABAv//AP4ABP0AAQAA//8AAAD/AP8AAQIA//8AAAAAAAACAQAAAP0AAAAAAf/9/wECAwABAQL+/f8BAf///gAB//wDAQEC/v/+Af4AAQAABP/9AgAAAAIB///+AQEBAAABBAD///7/AP8A/v8B/gACAAEBAf8AAgAAAQEAAAIAAAAAAgAAAP8CAAABAQD+/wABAwABAAL+/f4CAAAA/QABAP4DAAAC//4B/wEBAAD//gEE/gH/AgD//QIB///9AAH//gQAAQT/AAH/AAD////+AQL/Af8CAP79AwH//v0AAf/9BAABA//+AAD8AAH/AAP//wEAAAADAAEA/wEAAwAAAAEA/wAA/wEAAAEBAQH//v4BAgD+AQEE/wH9AP8D/wAB/////wH/Bf8BAAEAAgABAAEBAAD+/wMC/gEAAQIA/wD9/gAAAAAAAgAC/wD/AP//AgAAAAIA/QEAAAAAAQD/AP3/AAABAAEAAAAAAAAAAf8BAAAA/wAAAP//AQECAAAAAP4AAAEAAf8AAP8AAgAA/v8BAP8B/wAAAAAAAAEBAAAA/gAAAQAAAQAA/gAAAAAAAAAAAAD//wAA/wABAAAAAP7/AQAA/wABAAEAAAAA/wAA/wAAAP7/AAECAAAAAQAAAP8AAQP//wEAAf//Af8AAAH/AQACAP4AAQAAAAABAAD+/wEAAgD+AgAC/wD/AP4AAAAAAAL+/gABAQAAAAEAAAD/AAAE/f8BAAEAAAL/AQAA/wAAAf7+AAECAAD/AAAA//8AAAIAAAEAAAAB/wH+//8B/v8BAAEA/wABAAIA/wEA/wD/AP4BAwAE+//9//0CAv4BAQX7/P4CA///+wEB//sBAgED/vkAAAEBAAIAA///AAD/AP/+AAAAAAAAAP8A/QABAAAC/wD/AAH/AgACAP///wD///8AAAAAAf8A/wD+/wAAAQIBAf8BA///AP8A//8A//3//wAA/QAAAQAAAP3+AQABAP/+AP8CAQMAAv79Av/+/gIBAwD9AAAEAP8BAfwA//4CAfwB/QABAgEB/v8B///9AgABAP4BAQIA/wEA/v///wIDAQAB/wD/AP0AAwAAAAMA/v8BAf8AAQABAP0AAAADAP4CAAEBAAIAAQAAAAAAAAAA/wD/AAAAAAAAAP8AAAEAAAEAAv4AAAD8AAABAAACAP8AAQAA/wAAAQD//wAAAgH/AgADAP8AAP8BAf8BAQMA//8B/wAAAAAB//0AAAACAP8BAAAAAP8A/v8A/wAAAAAC/wH/AP8BAAAA//8AAAABAQAA/gMABAEE//8BAAD/AgAB/vwBAQMAAAAB/gAA/gID/AH9BAEEAQb9/QL//v0EAwL+/AEBBP/+AQL9///9AwT/AAABAAEA////AAAAAQIBAAD+AAABAAEA/f8AAAABAQAAAAIAAQAC/wAAAAAAAgEB//0AAAL/AAD//gAAAAEB/wD+AgABAAH//wAA//4BAwEA/gEAAf//AAL/AAD/AgMBAAP///7//QAD/wAABAABAAEA/wD/AAEA/gABAQD//v8A/wEAAgAB/v8C////AgIBAP4AAAIA/wAD/QAAAAICAgAE/P/8//wBAf8BAAb9Av8CAv///gAC//wCAQEF//0AAAABAAIABP//AgD//wMBAQD8AgED/wEAA/4A/wAAAgAAAP0AAAD+AQP/AAD/AP0AAf//AAABAAAAAAAAAv8A/wH//wABAQP//gH/AP8DAAEA/wAAAQD/AAT/AAAAAQEAAf8AAAEAAAAAAQAA/QADAP8A/wACAAAAAP4AAAAA/wEAAfv//P//AgH+AQEA/f3/AQT///0BAf//AgEBA/79BAAF+//8//0DA/0BAQT+/v8CBf7//AEB//oDAQED/vwDAAH9//8A/gH//wEB//4A/wEBAAD/AQH//wEBAQL9/QIBAf7//gD/AQL/AQEA/wEAAQEAAAAAAP//AQABAv/+AgAA/wAB/wEB/wABAAAAAwABAQAA/wAAAAEAAQH+AAADAf8AAAEAAQAB/wAA/wIBAQH+AAACAAAA/wEAAAD//wMAAQAA//8AAf//AAEC/gL/AAEAAP8AAAAAAQAAAf8AAgEC/wEAAP4AAQEA/wMBBQEAAAAAAQD/AP7/AAAA//8EAAT+//3//AID/gEBBP0C/gID//79AQL//AIBAQH+/AMAAv4A/wABAQL/AQAC/wH/AAEAAP8AAQACAQEAAP8AAwAD/v/+//4BAv4BAQP8Af4CA////gED/wACAQEC/vz/AfkEAQUABv78AgD/+wQBA/77AgEE/wAABf7///wCAwAAAv0A/gD+AQMAAAAAAPsBAAL/AAABAAD/AQEAAv/9A/8G/P/+//wDBP4BAQX9/f4CBP7//QEB//4CAQEG/vwBAAL+//7//gL//gEBAv///gEB////AQH/AQIBAQP//QL/A/3//v//AwT+AQAD///+AgL+//4BAv/9AgEBBP79Av8D/f/9//wBAv0BAQP//f4CAv///gEB//8DAQED/fwAAAL+AP//AAED/gAABAAA/wIC//8AAAEAAAEAAAP+/wAAAf8AAAD/AP7/AP8AAQH/AAD//wEBAAADAAAA////AAD/AP//AAEBAv4BAQABAv4B/wAAAAAA//4BAAH9//8D/wT+//7//AIC/QEBA/4B/QID///8AQL//gMBAQL9/P8B/gEAAAEDAP8AAAD/Av8A//8AAAEA/wAC/wAA/gEBAAD/Af//AP8A/v8AAAIAAv4B/wAA/wAA//8BAAAB//8BAAL/AP4A/wEA/wAAAgEB/wEB/wD/AQEA/wEAAQIA//4B/gEAAAAC//4AAAD+AQIA//8AAAEAAAAC/wAA/AECAP8B/wD9//0BAv4AAAL/AP4CAv///wABAP4BAAAB///+AP4EAQMABf79Af/+AAQBAf79AQEDAP8BA///AAACAwIAAf///wD9AQH/AAAA/wAAAQH///8AAf//AQEBAv///wACAAD+AAEA/gAAAAAA/gAAAAEA/wD/AAEAAAD/AQH+AAABAQEAAQD8AQD//gABAQD/AAACAP8AAv8AAP8BAQAAAgD//gADAQL/AAEC/v3/Af8AAP4AAAAAAgAAAP/+/gAB/wAAAP4B/v8AAAD/AQABAf//AAEAAAABAAAA////AQEBAAIAAgD+AAD//gIBAP//AAACAP8AAv8AAP8AAf8AAAEA/wD/AP4AAAABAAABAQAAAAAAAAAC/wAAAgD/AAAC////AAACAf8AAP/+/f8BAf8A/wEA/wECAQAC///+AAEAAAAAAAH+AAAAAQIB/wD/AAABAAAAAQEAAP///wABAQAAAAAAAf8AAP//AAEAAQAAAAEA/wAAAAAA/v8CAAAB/gD//wMBAv8BAAMAAf8BAgD/AAAA/wABAAABAP4B/wT+//z//QIB/wEAAvz+/gIDAP/+AAEA/wEBAAT//gAAAP8A//8CAQD/AQAA/gD/AgH//wAAAP/+AQAA/wEAAP8B/wECAAAA/wD//gEAAQAAAAD/AQAAAAH/AAAAAQMB/wL+AAD/AAD+/wAAAf7//wEA//8BAAAAAgEAAP//AQIAAf4A/gD+AQMAAAAC/v0AAAMAAP0AAAD/AQAAAf/+AAAD/gD9//wBAgAAAAL/AAEBA/8A/gEAAP8CAQAC//8A/wAB//8A/wEBAAEBAf/+AAD/AAD9AAH/AQIBAQD/AAAAAv7/AP//AgEAAQEC//0AAAIAAP4BAf8BAwEA//8AAAH/AQABAP4AAAEAAAEAAQAAAAAA/wAAAAIAAAAAAP8A/wL/AP4A/gH/AQAAAQH+/wACAAAAAAEAAwIBAP//AQIAAP8AAP/+AQL/AQAA/gAAAQEA//4AAQD/AQEA////Af8B/f/+AP0BAQABAQH//v//AgAA/gAB/wECAQH//gABAAEBAAEAAAD/AQAB/wAAAAACAAAAAAAAAQEAAP4BAQAAAQIAAAD+AP4AAAD+AQMA/wIAAAEAAAAC/wAAAf8BAAEA/wABAP///gEAAP//AQABAwD//wAAAAL/AAD+AQIBAAD9/wAA/gAB/wAAAv4D/wACAAD+AAH/AQAAAf//AAEAAAAAAwD+AAABAAD/AAAB/gEAAAMAAAAAAAAAAgAAAgH//gACAP///gAAAP8BAQAB/wAAAwD+AAL//wD/AQMBAAABAAMA////AQAA/wP/AAD/AAADAAAAAAD/AAH/AwABA/8A/gAAAAIAAAAD/QAAAQP/AP0B/wD/AgAAAQD/AgAB/f/9AP8CA/8BAQP//P8AAv8A/AEC//0DAQEB/fwC/wH//////wEC/wEBAQD+AAAB/wD/AQH//QEBAQP//gT/Bf7//P//AgH+AQED/fr+AQL///0BAv/+AgEBBP/9AQAA/wAAAAAAAAEAAAEA/gD//wAA/wABAAD/AAAA//4B/wH///0A/wEC/wEBAfz7/wAC/wD9AAH/+wEBAQH//gIAAP7//QABAgH/AQEA/gD//wEAAP4AAf/+AQEBAP/9BAACAP/+AP0BA/8AAf/8+/8AAv7//gAB//0CAQEA//0AAP//AAAAAAAAAAAA/wD/AP7/AAD/AAAA/wAAAP0AAAH//QAAAQACAQAAAAD9/wH//gH/AQAAAQAAAAEA/wD/AAD//gAAAP8AAQAAAAH+Af8AAgD//gEBAAEAAAD+AP4CAP8B/wABAAD+/wAB/v//AP4AAAABAAH/AgAAAf0A/wIAAgD/AAABAAD/AQH//gD/AAIAAAEAAf8CAAAB/f//AwAA/gACAQEAAQAAAP8CAQD+/wAAAgABAAEBAAD9/wEBAAD/AAIAAv/+AAD//wICAP//AAACAAAABP8A//8BA/8A/f8A/wABAQEAAAD+AP4A/gAAAf4AAAAAAQAA////AAD+/wD8AP8AAv8AAP/+/AAAAQAA+wAAAP8AAAD/AP8BAP4AAAAAAQH+AAAAAAAAAP8BAAD+AAAAAQAAAP8A/wAAAP4AAAEA//8B//8AAAAB/wAAAAAB/wAC//8A//8AAwAB///+//4BAf8BAQL++wAAA/8A/gAB//0BAQECAP0AAAD+AP4A/v8AAf8AAP4AAAABAAD/AAAA/gAAAAD/AAEA/QEAAAABAP4BAAD9/wEBAAEAAP4A/wD//wAA/gEBAQAA/gD+AAEAAAAAAP79Af//AQAA/gABAP8BAAEA/v4AAP0AAP8A/wD+Af8A+/8AAf8D/wD9AAAA/wAAAP4BAf8AAf8AAAAAAP8AAAD/AAIB//8AAQABAAD+/wAA/wD+AgD//gD8Af0BAQAAAf/+/gD/Av8A/gAB//0BAQAA/v0BAAAAAP4A/QAA/wAA//8CAAAB/wH+AAAA/wAAAAAA/wAAAP8AAQEBAP4BAAD/AQAA/gAAAQAAAAAA/wEA//79AgD+AAD/AAT//wEAAP0AAAD//wEB/gABAAEAAQH//v4CAAH/APwA/wD9AAAA/wABAP8CAAD+AAAA//8AAAD//v8A/v//AAADAQL/AQH+//8BAP//Af8BAAD+AQEA/v/+AQEC/AD9AAEAAP8AAAMA/v8BAv///wAAAP4BAAABAAD/AAL+AP4AAAAAAAAAAv8BAAEDAAD/AP8BAQAAAAABAAEAAfwA/gD+AQAAAAEBAPwA/wH/AAAAAAD/AQEBAf///wAA/gAAAAL/AAAAAAH+/wAAAgAA/wD/AAMBAAD/AQL/AP7/AAABA//+AgAA/wIAAP8AAQEBAAAB//4A//wBA/8AAQEA/wADAAH/AAD//f3/AQD///0AAAD9AQEA/wAA/wD9/gEBAQH+/wL///wCAAH/AAEBAgD/AQD////+AgP/AP7+AAABAAEAAQAAAAL/Af8CAAH/AAEA/wAAAAAAAQD//v4A/QH/AP8AAAD9AP7//wAAAP8AAQD9AQEA/////QD9/wEBAQT//gH///8DAgD+AAEBAQAAAQEAAP//AQIB////AP4AAQEB/wEBAP///gABAAD9AAEA/wIAAAD/AQP/AgD/AP/9AwP/AgIC/f3+/wD/AP4AAv//AgEBAf7/AAD/AwAFAAH+AAIAAAMDAwD//QEBAgAAAAT+AAD9AQIC/wABAP8A/gAC/wEAAf7+/wAB/wEAAAEA/gEAAP//AAEAAv8AAP/8AQL/AAAC/v7/AQEA/wABAAD/AQEABP8AAQAC/QD/AP4AAf8AAAIA/f8BAQAA/wAAAAECAAAAAQEAAP7+AAIA/wEC/wAA/gD/Af/+AAABAQAAAgEAAAIAAAIAAf8AAP/+AQD/AAD+AQD/AQIA/wAAAAACAgEAAP8A/wAC/QAAAAABAv8AAQH/AP8A/gAAAQEAAAECAQAB/wAAAAAAAAH//gEB/wAAAAAA/wH/AP8BAQAAAQEAAAIB/wEA/P4AAQAAAQAAAAD/AP//AQAAAP8AAAD/AQEA/v8B/wH/AQABAAAA/gAAAP0BAP8A/gAAAQD/AAMCAAD/AAL+Af0AAQMAAv/+Av///wEDAv/9AQEDAP8BA/4A//8BAgEAAv4A/wD9AgH/AQAA/wD/AgL///8AAAAAAgEAAP//AgAB/wABAPsBAv8AAAEB//4BAQD/AAAAAP8BAQAC//7/Af4BAQMAAQAAAQD//wEAAAAAAQADAP4AAwAAAAAA/wIAAf8AAAD9AQEAAAAD//4AAgEB//4BAAAAAAAAAQD7/wEAAAAAAQD//gAA/wMBAQAAAAAAAAAAAAIBAAD8//8DAAL/AP///gAC/wAABv/+/gMC//8AAQD/AAEAAAH//wEAAQAAAQD+AAEAAAAB/v8AAQAAAAAAAAAAAAEAAQD/AQABAAAAAAIA/gAAAAEAAgAA/gAA/wAAAAIAAAAAAQICAAMA////AQEC/wEBAf7//wEAAAD+AAEAAAEBAAL//wEAAP4A/wD+AAAAAAEBAP//AAIAAP4AAP//AQAAAv8AAQAAAAAAAAABAAAAAP4AAQEAAAAA/wABAAD/AAAAAAD/AP/+AAABAQD+AAAAAAIBAAD/AAACAAAAAgAAAAEAAQEA/gH/AP//AgH/AQH9/v8AAQAAAP4BAf/8AQEBAv/9AAH9/QEAAQD//gH/AP4DAwD//wABAQD/AAUA/wAAAQP/Af8AAAIBA//9AAD//QIEAf/9AQACAAABA/4AAP4BAgEA+/4AAAD/AP8AAAD/AP8AAQAAAP8AAAAB/wAAAf//AQD+/f//AP8A//8AAf4AAP8BAP8A/gABAP4AAQAB/gABAPkAAP4AAQEA/wAB/f8AAAAAAAH/AAD/AgEAAQH/AAL///3//AD+AgP+AQH//Pv+AQL/APwBAf//AwEBBf7+AQD+/wD9AAAAAQAAAP3//wABAgAA/QD/AAL/AAAAAQEAAP4AAP8AAwAAAQAA/f8BAP8A/wH+AAAAAf8AAAAA/wIABf3//f/9AgH/AAEF/QAAAQX///0BAQD9AgEAAv/9Av8D/f/+//wBA/4AAQP7/v8CBf///QEB//sBAQEC/v0D/wT9//v//AIC/gEBA/3+/wEF///8AQH/+gMBAQD+/QAA/wD/AAD/AQP/AAD//v8AAQL/AP8BAQD9AAEAAf/+AAD+AAD/AAEA/wAAAP8ABAAAAgAAAAAAAAEAAAD9AQH/AP7/AAAAAQEBAAAA/v79AQACAAD+AAEA/wAAAAD//wAA/gAA/wEBAAAAAAD+AAMA/wAAAP8AAAAAAQAA/QD/AAH+/wEAAf///wL//wAAAQEAAwAAAAAAAf7+AP/+AQAAAAH+AP4AAQEB/wAAAf7//wACAAD9AAAA/gIAAP4A/gEA/f4A/gD/AQAAAAAB/gAAAQMAAP4AAQD+AAAA/QAAAAAA///+AAECAf8BAf7+AP8AAP8A/QAB//4CAQH+AP4AAAH///4A/wEC/gABAf38AAEC/wD/AQD//gEBAQP+/gAA/QH//wACAQD/AQH8/wAAAAD/AP8AAQD+/wAA/wH+/wABAQAAAAP//gAAAAACAQH+/wABAQD+AAEAAAD/AQL+Av8DAQMBBv75Af8A/QQEAv37AAIF//4BBf7/APwCAwL/Bfz//QD+AgL/AAEE/f3/AQT///0BAQD9AQEBA//+AgAE/P/9AP0BAf4BAQL9+/8CBP/+/AEBAPwCAQAD/v4D/wP9//0A/gID/gEBBP38/gIF///8AQH//AIBAQH+/gEA/wAAAAD/AAH/AAD+//0AAQH///8AAQD8AAAAAf/+/wD+AAAAAQAA/wD/AP//AAD/AgAAAAD/AAAAAAD/AAEAAP4CAAEA/wAAAAAA/f7/AAAC////AAAA/AEAAAD//v8A/gAAAAECAP8AAAD+/wAA/wAAAAEAAAD/AAAA/wAAAAD/AAEBAf7//wEA////AAEAAgAA/wD/AP7/AAAAAf8BAAD+AP4A/gEA/wAA//79/wED/wD+AAEA/gIAAP4A/wAAAP8AAAH/AP8AAAAA/f//AAP/AP4AAAD9AAAAAAD+AAD+AAD+AAAAAf8AAP79/v8BAAAA/QAAAP0CAAD/AAABAAIA//4A/wEB/wEBAfz+/wAC/wD+AQD//QIBAQP//gAA/AEAAQACAAAAAAD8/gAAAP//AP8AAAD+AAAA/wD//wD/AAACAAMA/wIAAAABAQH//wABAQAAAAL/AAD+AQL+Af4EAAQBB//7Av8A/AQFAv37AAMF//8BBf0AAPwCAwEAAAAAAAD+AAH/AAABAP4AAQD/AP8AAQAAAAEAAwAAAQABAQAAAAAAAAAAAP8BAgAA/wAAAQD/AAEAAAD/AQIBAAIA//4AAAEC/gABAv///wAC/wAAAQAA/gEBAAL//wEAAP7//wAAAAAAAAEBAQD/AAD/AP8AAf8AAQAAAv//AQD+AAAAAP8A/wAAAP8BAAEAAQAAAAAAAP//AAAAAP//Af//AAEAAP8AAQAAAgICAP/+AAEBAAAAAgAAAP4BAP8A/wD//wABAQH/AAD+//8AAQD/AP8AAQD8AAEAAf/+AAH+/wEBAAH//gH/AAADAwH//wABAgD/AAT+///+AgH/Af4BAAMAA//9AQAA/wMDAv/9AAADAP8BAf4AAP4BAgEAAf4AAAD/AAEAAAD/AAD/AAH/AAAAAAABAAAA/wD/AQD//v8AAAAAAAABAQABAf8AAf8AAAABAP8AAQAA//4CAf8A//8AAgAA/wAB/wD//wAAAAAAAAD/AAEBAAH//gP/AP7//P//AgP+AQIA/f3+AQP+//0BAv//AgIBBP39AQD/AgD/AAAB/gD/AP3//gABAv8A/gD/AP/+AAAAAf8BAfwAAAAAAAH/AQAA/v//AQAB/wAAAAAAAv8BAP4AAAEAA/4A/v/+AQL/AAADAP//AgMA//8BAQD+AQAAAwD/AQAD/wD/AP8BA/8AAAEAAf8AAQAAAAEBAP8AAAAAAAEAAP//AAEA/wECAAAAAAECAAD/AAACAAAAAAAAAAEAAAH/AP8A/wAA/wEAAAABAAAAAQD/AAEAAAAAAQAAAf//AAAB/QAAAP8BAv8BAQIBAAACAAAAAAABAAABAQAB//8BAAAAAAH/AAAB/wAAAP8BAAEAAAABAQAAAQAAAAL//wEA/v4A/wD/AQAAAAD+AQABAP//AAAAAQD+/wAA/wD//wD/AAACAAL//gEAAP0CAgEA/wEBAQD/AAQAAAAAAQL+Af8AAQIAAv/+A///AQQCAv/9AAADAP8BAv7///4CAgEAAf0A/gD9AAL/AAACAP8AAQD///8BAAD+AAAAAQAAAgAC/wD/AP0BAgAAAAMAAAABAf//AAAAAP//AAAC//3/AAABAAAA/v8AAQAAAQICAQAAAQACAP8AAP4AAAEAAAIAAP8A/v/7AQEAAAAC/v8AAgL///4BAAD+AAAAAf/8AQAB/wD/AAL//wEAAAIBAv4BAQEAAQAAAAEAAAD9/v8CAAMAAP7//wECAAEBAwAC/wEB///+AAD//wEAAAEAAAAA/gD//gABAgAAAAH7//8B/wD/AP4AAQD/AQEB///+AQAD/wD+AAAB/gAAAAT+/wABAf8A/wAAAP4AAAAA//4AAQD/AAAA/gD9AQAAAAAAAQAD/wD/AAEBAf4AAAAB/wD/Af3//QACAgEAAQAC/gEAAQD/AP8AAQD/AgEAAP///wABAQD/AAEA/gEAAAAA/wH/Av8BAAAAAAL+AAAAAQEAAP4AAAABAv/9AgD//wEBAQD/AQAAAP8BAP8AAP0CAf8A/wEA/QACAgH+AAH+/v8AAQEAAPwAAQD/AQEA/gAA/QD9/gH/AQIA/gIA/wABAgH+AQABAAD/AQH+AP//AAMAAPz+AP8BAQACAQAA/gD/AP4A/wABAAAAAQAAAP//Av8A+/8A/gD/AQH/AAAAAAEAAAAAAP4AAQD+AQEA/v///gH7AQAAAQL//QIA//4BAwH/AAABAgAAAQH/AAD+/wMCAP//AP8AAAICAAAAAQD/AQAA/wD/AAEAAAIBAP8BAQL/AwD//AACAgT+AQED/v7+AQP/APwAAf8BBAEBAf8AAQD+AwAEAf7//wEA//4EAwH+/gEBBAABAAL+/wD8AQMBAAEA//8AAwAC/wAABf7+/wEB/wD+AAD/AAIAAP8AAf8A/QAA/gAAAQH/AAD8AP4A/wH/Af8AAAD9AQEAAP/+AAD9AAD+AP4AAAAAAAD/AAABAwAA/gAAAPoAAAD+AP8BAP4BAAEA/wAAAAAAAAD/AAAA/wEAAAAA+gAAAP8A/gAB/wABAQD+//8BAAAAAv8BAAIAAQEA/wAA/wAA/wAAAQAD/v8AAAABBP8AAQT8/f8BA/4A/gEA//0CAQEC//7/AAD/AP4A//8AAP8AAv/9AAACAAD+AP8AAP8AAAD//wABAAEAAQAEAP4C/wD/AP4AAAEAAAAAAAD//wAA/wEBAAAA/wD9AP4A//8AAAH///8BAgAA/gD/AP8CAAEA/v0AAAD+AQEAAwD+AQD/AP//AQAC/wAAAP8A/v8AAAAAAf8AAf8AAAACAP0A/wAAAQEBAAEAAAEB/wAB/wAAAAD+AQAA/wD+AAAAAf8AAAD//AABAgAA/QD/AP4BAQAA//wBAP7/AP0A/wD///8A/wD+/wECAAD+AP8A/wAAAAH///8B//8AAQD///8AAAABAP0AAAEAAf8B/gABAAAA///8AAAA/wD+Af3//gH/AAABAAD/AAEBAAD+AAD/AAD///8AAAD+AAEAAwD/AAAAAgD//wAA/wD+AP8AAAAAAAD//gAAAv8A/wAAAQAAAAAB/f4AAAT///4BAAD/AQAAAf8BAf8C/v/+AP4CA/8BAQL++/8AAv8A/gEB//wDAQEA/v0AAAIAAAAA/gIC/wAAAf4A/wABAAD/AQAA/gEAAAL//wH/A////v//AgP+AQEC+/3+AQL/AP0AAf/6AgEBAv//AP8AAAAAAP8AAQEAAAEAAAAAAQAB/wAAAP7/AAAA//8BAAD9//0AAAIC/wEBAP3+/wEC/wH9AAH//AIAAQEA/gL//////QD+AgH/AQAB/v8AAAD/AP8AAf/9AQABAf/9Av8B/v/+//4BA/4BAQP8/P4CAv8A/QAA//4CAQEB//4A//7/AAAAAQEBAAAAAQAAAAD/AAH/AAAA/wAAAP4AAAIA/wEA/wD/AQAAAAAAAAAB/wD/AAAAAAD+AAAAAAH//wD+/gD9AP0BAgAAAAL+/v8AA/8A/wAAAAABAAD9//8AAP0CAAEABAH//wEA//8B/wAAAQAAAP8AAQAAAP4A/gAAAAEAAQAEAQH/AQAB/gH/AQAAAAIAAP8AAAAAAAD+AgD8/wADAAIBAf8AAAIAAv///wEBAQAAAAQAAAD//wAAAP79AAIAAP/9AAD/AwECAAD/AAACAP8AAv///wABAwIAAf4A/QH/AAIAAAAC//0AAQL/AP0AAAD/AgAAAQAA//8C/gAA//0AAgAAAAL+/wABAv8A/wEBAP8BAAACAAEB/wEC////AAEA/wEB/v8A/wAB/wD/AAH//gIBAf8A/wD/Av7//wAAAQD/AQEC//7/AQL/AP4BAf8AAQAAAP8A/wAAAgABAP8AAAH/AP8A/wEA//8AAQAAAAD/AAAAAP//////AAL/AAEB/wABAQEA/wIC/wAAAAEAAAEAAAD//wH///8A/gD9AAL/AAD//v3/AQD/AP8AAAD+AQAAAP8BAf/9//8A//4BAv4BAQIAAP4BAf8B/wAC//0CAQH//v0CAP8BAAEA/wAA/wAA/wD+/wAB/wABAAAAAAAAAP8AAAEA/gIA/wAAAAD/AAD9AQH/AAL/AQIAAQAA/wAAAQD+AAABAQEBAAD/AQD//wH//wAAAf8AAgD/AP//////AQECAP3/AP8A/wAB/wAA/wAA/wAC/wECAAIA/AAAAAD//v8AAAEAAP8C/wP/AAD/AQD/AAH/AQIAAQAA/wAAAQABAgABAQAAAAX/AQH/AP4BAf8AAAABAgD/AAT+/wD/AgMC/wEBAAAAAf8CAP8A/wED/wD/AAABAAAAAf8AAAEBAAAAAf8A/gD+AAIAAAABAP//AQH/AP8AAAD+AQAAAgD//wAB/wD/AP8BAAAAAAEA/wD/Af8A/wAAAP4AAAAAAP/8AP8CAQIAAf/+Af///wH/AQD+AAEBAP8BA/8A//8CAAIAAQAA/gD+AQH/AAACAP4AAP8AAP8AAf/+AgAA/wD//wAAAAAAAP4A/wD/AAEB/gD/AAAAAAAAAAEBAAABAAD9AP8CAAAAA//+AAAAAQIBAQD/AAABAAAAAgAAAAABAQAAAAEA/wABAAEAAAACAAD///8AAQAAAAADAAAAAAD+/wD+/wD//wEBAf4BAQH//f8BAP8A/gAA/wEBAQEBAf//AP8AAAAAAf//AAAAAgEAAP8AAAABAAEAAQAAAP8A//8A/wAAAAACAP4AAAD/AQAB/wAAAAIA/wAC/wAA/gECAQAC/wD8AP8AAf8AAAEAAAAAAv8A/wABAAABAAABAP4BAAH+AP3//gED/wABAv/9/wECAAD+AAEA/wAAAAQA/gAAAAAA/f8AAAD/AAAA//3/AAEAAP8AAAD/AAAA/wEA/wEAAQEBAAP//gH//wEBAAAB/gAAAQD/AQL+AP8BAwH/AAH+AAD/BQABAAAABAAAAAEAAAD/AQAAAgAAAP8AAf8AAf4A/v//AAL/AAABAP4AAAH/AAAAAAD/AQEAAv8AAv8F/f/8//wBA/0BAQT+/f4CBP7+/gAB//0DAQEF//0A/wL+/////wEC/gEBA/78/gEB/wAAAAH/AQIBAQP+/AH/Avz//v/6AgP9AQEC/v3+AgL+//4BAv/9AwEBBP79Af8B/f////wBBP4BAQP//f4CAgD//gEC//8DAQEC/fwB/wL/AAAA/AIC/wAAAv///wEC//8AAQEA/wIBAAL///8A//8AAQD/AAD/AAABAf//AQEAAAAAAAAAAQAAAP//Av/+AP8AAP8CAf4BAf////4AAAAAAAAB/wACAQEA//4C/wP+//0A/wID/gEBBP79/gID///+AAH//gMAAQP//AAA/QEAAQABAAAAAAD+AQEA//4BAQEA/wAC/wAA/gH/AQABAQAAAAEAAP8AAAAB/v8AAAEAAAABAP8AAAAA//4BAAL/AP8A/wEB/wAAAv///wACAAD/AAAAAgAAAAL//f8AAAEBAgECAfwA////AgIBAP8BAAEA/wAAAAAA/gH///8AAAAA//0AAf4BAAEAAP8BAQD//QACAP4BAAH///39Af0CAQMBAf79Av///wQFAf/9AgAD//8BAv3///4CAgAAAP8AAf/9AQP/AAH/AAABAAD/AAAAAAD/AAABAP/+/QH9AAECAAAA/wH///0BAQH//wABAQD/AAH/AAD+AAL+AP//AAABBQD/AAAA/gAAAAAAAAEAAP8AA/8AAP8AAP4AAvz//f/+AgP+AQEB/f0AAQP+AP4BAP/+AgEBAv3+Av8G+//8AAACA/0BAQT8+/8CBv7//AEB//wCAQED/v3/AAH9//8AAAIC/wABAP7+/wAA/wD+AQD/AAEBAQL9/gEAAv8A/gEEAgH/AAEA/v8AAAEAAP8AAAABAAEAAf7+AAAAAAD/AQEB/wAAAAABAAAAAP8AAQD/AP8AAAAA/wEBAAAAAAEBBQEB/wAA/wABAAD+AAABAAAAAQAAAP7+AAEAAgAAAAAAAQD/AAABAAD/AQAAAP8A/gD/AAAAAQABAAAD/gEAAf8AAAD//wEBAQEAAAAAAAH/AQD/AP8A//8D/wX9//8A/wMD/gABBf38/gID///8AQH//QMBAQT+/AMAAv4AAAD/AQH/AAEA/wAAAAEAAP8B/wD/AQAB/wAAA/8D//////8BAP4BAAP+/P4CA////QEA//8CAAED//4AAP8CAQUBBP7+Av7//AQEAf38AgEFAP8BBf3/APsCBP4B/QEBAgD+//4C///6AgMC/v4BAQL//wEB/f///gIDAQAB/wAAAPwAAwD/AAAAAAAAAAAAAQAAAAD/AAAB/wABAAEBAAAAAf8B/wABAQEA/wH/AP8AAAAAAQAAAAEAAf4AAv4AAAD9AQIAAAAE/wAAAgD//wAAAAABAQAAAv4AAQAD//8A//8BAf8AAAL/AP8B/wD/AAAA/wABAAAC/wH//wAAAAEA/gEA/wAAAgED/wH+AAABAAEAAP8AAAH/AgAAAAMABAAD//4BAAD+BAMAAPsBAAT//wAD/wAA/wEC/QH8AwEFAQT+/AH///8EBQL+/AEBBP//AQT9///+AgUAAAIAAAIAAQAAAP//AgABAAD+AAABAAAAAP8AAAD/AQAA/wAAAgAE/v8A//8AAwIBAP4BAAIAAAAD//8AAAEC/wAAAwADAAL+/gD///4CBAD//QEAA///AAT+/wD/AgMCAAP/AAEAAv8A/wD/AQABAAEAAAAAAAAAAP8AAAIAAQAAAQIBBAED/v4B//7+AwMB//0BAAMA/wED/f///wIEA/8E+gD+//wBAv4AAAP9/v8BAv///QECAPwBAAAE/f0C/wEBAAIAAP4AAP/+AgICAAH8Av8CAAAA////AAEBA/wAAAAAAQEG/wEA////AAEB/wABAQAA/wAB/v8AAQECAQAC/wACAAAAAgEAAAIA/wAAAAAA/wABAP3+AAAAAAL/AAACAAIABP//AP//AAED/wD+AQEBAP8AAv4AAAAAAf8AAAAAAQAB//4B//8BAAABAAAAAAEA/wAA/wAAAAACAAD//wD/AAAAAAH/AAH/AAD/AAABAAAAAP4AAAAA//8B/wEAAAAAAwD/AAAA//7/AQAAAAAAAAAAAf8AAQEAAAAAAP8AAAABAAAAAAABAQEAAP8AAAAAAAD/AAAAAAD+Af8C/gD/AP8AAv8AAAL9/wABAQD//wAAAAAAAAED/v8BAP/+AAAAAP8AAAAAAP8BAQABAAABAAAAAP8AAAAAAAAAAP4BAQH+/wAB/wABAAEA/wABAAEAAAED/gAAAP7/Av8C+v/6AP0CBP4AAQX7/v4BBP7/+wEB//gDAgED/fwBAAABAAEAAgD/Af8A///9AAD/AAEAAP8AAP8AAAAB/gEAAQAAAQAEAP8BAAD///8B/wAAAQIAAAD+/wAA/wH+AQACAP8AAAQA/gAAAP4AAQAA/gAB/wAAAP7/AAEBAAAAAAABAQMBBf/9Af///QICAv//AQECAP8AAv4AAP0CAf8BAAIBAgEE/v4C///9AgQC/f0BAgIA/gEB/v///gIDAgADAQAAAP4AAP8AAAH//wAAAAAA/gAAAP4BAAAB//8B/wX9AP7/+gED/gEBBf4A/gIB//78AQL//gIBAQP9/gD/A////v//AQL+AAAC/v7/AAEAAP4BAQD+AgABAf/+Af8D/QD///wBBP4BAQL+AP8BAf///wEC//4CAgEC/f7/AP8BAQEBAf/+AP//AAQDAf7/AQEA//8AAAEAAP8BA///AP8AAP/+AAH/AAD/AAH/AAEAAAABAQAAAAEAAv7/Av8D///9AP8CBf0BAQH+/v//Af8A/QAB//4DAQEB///+APwCAQEAAP//Af///gMBAf//AQECAAAAAgEAAP4BAf///wAAAQD+/wEAAAABAQEBAAAAAAAAAAACAgAA/gABAgACAQD9AP4BAv8AAf7+/AAAAAAB/gAA//0BAAABAAAB/wEB/////gED/gAA/wD9AAEC////AAH//QIBAQL/AAH/BP///gD+AgP9AQEA//3/AQL+AP0AAf/+AwEBAv7//wEBAgEDAQD/AAEAAP4CAgH/AAEBAQD/AQD/AP///wIA/wMDAAIA//8B/wD/AwIAAQD+Af8CAAAAAP4A/wD+A/wB/gMAAQED//8B/wD8AQMC/v4BAgH//gECAAD//wEB/v8F/f/8/vkBBf8AAQD+/gABAf///wECAP0DAQEF/f7+AAX/AP///QEC/wEB/gAAAv//AAAAAAEA/wEBAAL+Af0BAQEBAAAC/gAA///9AQIC//4BAAIA/wEB////AAEE/gAAAQECAAP//gAA/wACAQEAAf8AAv8AAP7+/wABAQAB/gX6//z//QMG/AICBPv9/wMD//78AQL//AQCAQX7/v8A//4B/gD9AAACAAABAP4AAP8BAAAAAAD/AAAAAP0C/gD/AAABAAEA//8A////AAIA/wEAAAAAAAD/AAAA/wAAAP//AQAAAAD/AQEAAf8AAP//AQD/AAAA/wEAAAD/Af4A/QABAgAB/vwB////AAEB//4BAAH//wEE/v///wEBAAD+/gEA//8AAQIAAP7/AAH+/wEAAQABAAD/AP8A/wIC/wH+//3//QEBAAEAAv39/wECAP/+AAH/AAEAAAIAAAEAAAAB/v//Af8BAQD//wD/AP8BAQAA/wD9AAAAAf4CAv8B/v/6/wACAP8BAQL9/f8CAv///QAC//0CAQED//3/AAACAQIA/v/8Av/+AP8EAgD+Af8C/wABAf3//wEAAfcC9wcBBgML/foE/v/2BgYF+voCBAb//AEL+/7/+gQF+gH8AP//AgsBAgEAAfsA/wT8AAAD/wD9/wABAAH////9AfwDAAECB//+Av8A+gICA/z9AQMC//0AAv8AAP0BAv8AAv//AAEAAQL/AQEBAAABAAAAAQEA/wACAQEAAf7//gD9BAEAAQX+/QL//vwDAgL//gEBAv//AQT9///+AwQAAAL/AAIAAAABAAAAAAIDAf/+AQECAP8A//8AAAL/AQAAAgAAAQH///8B/wABAgEBAP8BAQIAAAEBAAAAAP8BAQACA///AAMA//8AAf8BAAEA/wAAAQAA/wABAAAAAAAAAAADAQIAAP7/Af///wIBAQD+AgAC/wABAv7//wAAAgAAAAEBAwH//v8B//8AAwQCAP0BAAP//wED/v///v8AAv8CAAABAP//AgD/AAIBAP8B/wH/AQABAAEA/wAC//8E/wP//wEA/P8CAAABAwIB/gIAAf8AAAEA/gAAAAP//gEAAAIBBgEB/P4D/v8BBQQBAPwDAQX//gEB+/7+AAMD/wH/AwIHAQD7/QX+/v8GBgL//AMBBv/+AgL5/v7+AwQE/wMDAAH///8CAP8AAQIAAAH+AQAC//8A/////wIDAgX+B/f+9v74BAb8AgIH+fX8BAj8/fcCBP72BAICB/z5AAD/+AH5AAH//gIA/wD//AAABf8A+gAAAfv+//8AAQAC/wT5APr//QECAAEAA/39/gIF/v/7AQEA/AEAAAMA/gEAAQABAQAB/v8BAP8AAgAAAP8AAP8A/wAA/v//AQIDA/8E+//+//4BBf8BAQT+/v8BAv///wEC//4CAQED/P4C/wEAAP//AAD/AQEAAgEB/wEBAP//AAEAAAAAAAH/AAL/Av//AP8AAAL/AQABAf7/AgAA/wAAAQD/AQAAAgABAQADAAAA//0AAv8AAAMAAv8C/wD/AQABAAACAAAC/gAC/wL+/wD/AAAC/wEBAv4A/gIAAP8AAAL//wEBAAL+/wT/BQD/AP8AAQP/AQEE/gH/Av8A/gEAAv/+AAABA//9AgAB/wAC/wAAAv8AAAIAA/4C/gH/AgACAAIAAAAC/v8BAP4BAQX/AP8AAgAA/gIDAQD7Av8F/wABBP7//wD/AQT+A///Af79AgT+AQID/QD+A/4A/gEAA/8BAgEABfv9Bf4E/v4A/vwDBfwCAwP7APwDAP/+/wEE/v4DAQIG+/sD/wH9AAIA/QAB/wABBP8C/wL+Af4DAAIAAQAAAAL7/wIAAv0A/P//AQL/AAAC/gD/AQT///0AAQD9AQEAAQD//wH+AwEEAQT+/gH///4DBAH//wEBBP//Af/////9AwQAAAIBAAAAAP8BAAAAAQMBAAABAAAAAAAAAAEAAP8BAAEAA/3//P/8AQT9AQEDAf7/AQL///4BAf/9AwABA//+AAAEAP/9//wBBP4BAQIA/QAAAP8AAAEAAP0CAQED/f7+AP4BAQMABP79AP/+/gQCAf79AgAB//8BBP7///4CAwEAAAAA/wAAAQEAAQD/Af7/AQP/AP8AAQABAgABAQD/AP8AAQD/AAEA//8A/wAA/gAA/wD/AAAAAAACAAAAAP8AAAACAP8AAgAAAQAA/wEAAAABAAAAAAAAAgEAAAEA/wH/AQH//v8CAAL+AQAB//3+AQD///8AAf/+AgAAAv//Af8BAAAA/wIA//8BAAAA//4BAgD//wAB/wMDAAAB//3+AP4DAQIAAP7/Af/+/wIBAP79AgAB/wABBAD///4AAwEAAAIAAAACAf8AAf8AAQD/AAABAP//AAAABAAAAf8D+wH7AwECAQH+/QL///4CAQL9/gEBAgD/AQP////8/wID/wL+//4AAQEA/gEAAgAA/gEAAQD+AQD//AMAAAL/AQH/Av3//f//AQL+AAED//z/AQL+//8AAf/9AgEBAv7+/gD/AQABAAH//wAA/wEC/gD//wABAgAAAAD/AAAAAAH9AP4EAQMABv/+Af///gT/Af39AQEB//8ABP7///0CA/4A/gEBAQAC/v0AAP8AAgAB//4BAAEAAAAD/wAA/gEC/wH9AwEBAAT+/gEA//4EAQH//gEAAf8AAAL+/wD+AgL//wP/APz/AAEC/wEAAwD9AAEBAAD/AAD//QIBAAH+/QAA/gEA/wABAAACAAAAAf0AAQEAAP8AAAD+AAAAAAEB/QD/AAEBAAD+/QAA/gADAgD//wEAAQAAAQT////+/wH//wABAP7//wIC/wH/Af/+/wIC///+AAEA/QIAAQL+AP//AAIAAP8BAQD/AAABAAD/AP8AAP4AAQABAQAAAf7//gD8AAEBAAD+/QAA/gAEAwAAAAAAAf8AAAMA/wD//wL+APsCAQAAAv7+AgD//wMBAQD+AQAC/wABBAD/////Av4B/AABAQAB/wABAP//BAIB//8BAQIAAAEEAf///wADAQAB/wD///8AAv8B/wD/AP0BAP///gAB//8DAAAB/wAB/wEDAAD/BAAA/wH/BQABAAL9Af8AAAEAAP7/AAT//wEA//4A/gAB/wAAAAAA//0AAQEAAPwAAAD9AAAAAAABAQAA//8C/wD/AP8AAAEAAv8A/wH/AQAB/wH/AAAA/wECAAL/AAH//gAC/wAAA/7/AAEBAP//AQAA/v8BAAIAAAMAAAD//wAAAAL/AAAAAP8AAP8BAP4AAP8A/wAAAv8BAQEAAQABAP///gEAAAAAAAAAAAAAAAEAAAH+AAACAAABAQD/AAEAAf//AQD/AQEAAQD/AQABAAAA//4AAP8AAAIAAQAAAAD+//4A/wAA/wABAAAAAAAAAAAB/wAAAf8BAQAAAQACAAL/AAAA/wAAAAEA/wAAAQAAAP/+AAD/AQEBAAABAAEB//8AAf8AAAAAAf/+AQACAP8AAP4AAAD/AQEAAQD/A/8CAAH/AQAA//4BAgD/AP8AAAD+/wEAAQH+/QD9BAEDAgP9+wL+/vsCAgP8/gECBP/9AQL8///+AgMAAP//AP8A/gD+AAAAAf8AAQABAAEAAAAAAP4AAAL9AAAAAf4A/wD8/v8A/wAB//8AAAEAAAEA/wAB/wAA//7//gAB/QABAP/+AQH+/wP+AAEAAAAAAQAAAAL+AP/+/wAB/wQA//z//gEC/gAAA/3//gEAAf/+AAH//QEAAQD+/gMAAgD//v/+AgH+AQEB/gP/AgT///wBAf/9AQEBA/79AQEAAgADAAEA/v8AAP8ABAAA/wEAAAAAAAIAAAD+AAIAAQD/AAAA/QAAAAAAAQAEAAH/AAABAQAA/gAAAAAAAQAA/wIAAQACAP//AAAAAAT/AP4AAQEAAAAAAAAA/wAB/wH/AAEBAAD+/wD///8BAwAA/wAAAQD/AQD/AAD/AQIAAAIBAAIAAAABAAAAAwEAAAD/AAEBAAEA//8AAAEAAQIABP0A/f/7AAH/AAAE/wP/AgL///8AAQD9AAAAA///BP8F/P/8//kCA/0BAgT9AP0CAv///QEC//sCAQED/v4BAAH/AAAAAAAC/wAAAP8BAAAAAAAAAAAAAQABAAAAAQEAAf4AAP/+AAD/AAEB/wL/AgAA//8AAQD/AQEAAAABAQAD/v8AAP8BAv8BAQMAAf8BAP//AAACAPwBAQAB//7/AP8CAAIAAf/+AP8A/gIDAAD+AQAC/wAAAf8AAP8BAwEAAv///gD/Af//AQED/gH/AgH///8AAgD9AgEBAf7+/gD+BAAFAQX+/QL///wCAQL+/AIBA///AAP9/wD9AgMB/wMAAAAAAAADAAAABP8BAAL/AP8AAAEA/wAAAAL/AAEBAAIAAgAB//4BAAD/AAYB/wEAAQEA/wAB/wAA/AEAAQEAAP8A//0BAv8BAP/9AgAAAf8A/wAA//0AAQAB//4AAAD9AP8AAQD//wAAAAD/AAABAAH+AP8AAAEAAAAAAP8C/QEBAwEC//4C/wD8AwUC/v4AAQIA/gAB/v///AIC/QH5AQEDAQX++wL///kCBQL+/QEBAwD+AQf9///9AQP/Af8AAQABA/8AAf///gEDAv8AAAEAAP4BAv////4CAwAB//8AAgAB//0BAP//AAIA/wAAAAAA/wAC/wAA/gABAQEA/wD/Af//AAAAAAABAgEA/wABAgD/AAAAAAAC/wABAAH/AAEAAAD/AAAAAv8FAAABAAD/AP8AAQAAAP8AAQEB//8AAgAAAAAAAAD/AgQBAP8AAAIAAAAAAAAA/v//AQH/Af8BAAMAAf8BAf7/AAAA/wAAAAAB/wEBAAH+AP//AvwDAQMBBP7/Av///QMCAv78AQID//4BA/7///4BA/8B/gEAAgD///8BAAD/AAMBAP8AAAEAAAAB/wAA/gAAAAH9AwEDAQL+/gL///0DAgL//QEBAwD/AQP////7AQMD/wX9AP3/+gEE/wABBfz//gIE//7+AQL//wIBAQL9/P8B/QEAAgAC//0AAAD//wYAAAABAQAA/wAC/wAA/QEC/QL7AgEDAQX+/AH///sDBgL+/AECAwD+AQX+///6AgT/Af4CAQMBBf8AAf///gIEAf//AAECAP4AAv4AAP0BAv4B/gMBAgEE/vsB///9AwYB//4BAAP//gAG/v8A+wID/wH+AwEDAQb//wH///4DBgD//gABA///AQT9///8AgMAAf4CAAIAAv/8AAD//AIE/wD9AQACAAAAA///APsBAQAA/gEAAAADAAH/AAABAQL/AQAAAQAAAAD/AAAA/wAAAAABAAEBAAH//QD//wEBAQD/AQAAAAAAAAP/AAABAAD+AfsCAQIBBv7+Av///gQBAf/9AQEE//4BBP7///wCAwEAAf8A///+AQH+AAABAAT/AQH///8AAQAAAgAAAP/9AAAA/gAAAAEAAQAAAP4CAAD/Af8AAQAAAAIAAAD9AAD/AP4CAAAAAf8B/////gD////+AAAA//8AAQAAAPsAAAEAAv8A/QD/AAP/AAACAf//AAH/AP4AAAD9AgAAAf///wEAAwECAAX+/wIAAP8DAAH//gEAAwD/AAH//wD9AQMBAAH+//v//AEE/gEBA//9/gMB//7/AAH/+wIAAQD///4BAQEAAQAC//8AAAABAQAAAP8BAAIA/wAD/wAA/wEB/wAA/wAAAAAB/wAAAAH/AQD/AP8AAAAAAAEAAAAA//8CAAL+//0AAAEB/wABA//+/wEA/wD+AAD/AAIAAAD///8BAAABAQD+APwBAP8AAQMBAP8AAAIA/wAC/wAA/wECAQABAQD/AAIB/wAAAAEAAP8A/wAB/wAAAAIAAAD/AAEAAAD/AP0A/wEB/wABAv4A/wIBAP/+AAAA/wIBAQD//gAAAv8A/wACAQAAAAD//wH/AQD/AP8AAAD/AgAA/v8B/wD/AQEAAAD//gAA/wEBBAEBAAEAAQD/AAD/AAD9AAEAAAIAAAAAAQEB/wAA//0AAAIBAAD/AP8AAAEAAAAAAf7/AQAAAAAAAQD/AAAA/wAAAAAAAP8AAAABAQAAAQD+/gAAAQAAAP4A/QD/APoAAAAA/wAAAAD/AAAAAAD+/wL9AP0BAQMAA//+AAD//AMDAf/+AAACAP4AAv8AAPwAA/8A/wAAAQD/AP8AAP/9AAEAAP8AAAEA/wADAAAA//4CAQAA/wAAAAIB//8BAQD+//8C////AAAA/wECAQH9//8AAP4BAAIA/wAB/wD/AQABAAH9AQAB/wEA//8AAP8BAPwA/gEAAQEDAP4BAAD/AgIB//8AAQEA/wAA/wAA/wEB/wD/AwACAQT//gEAAP4B/gD+/wEBAAAAAAL/AAD9AQH/AQIBAAEAAP//AQAAAAAAAP8BAAACAP8AAP4AAAAA//8B/gIAAAH+//0BAAD9AAEA//8AAAEA/wAB/v8AAQEAAAACAAAAAP4B/gEAAAAA/gD/AAAAAAAAAAEBAQD/AAEAAf4CAAIA////AAAA//8CAQD/AAAAAP8AAP8AAAEA//8AAQEAAAH/AP0CAAAAAv4B/gEAAQIA/wAE/wAAAgACAAEAAgACAAH//QEA//0AAgIAAQAAAQD+AAL+/wABAQH/AAH+AP8A/gD/AAAA//8AAf8B/wAAAAAAAgABAAH//wEAAv4AAAD+AAEBAAEB/v8BAAL/AAAAAAABAgEA/v8AAv8CAP8AAPwBAAABAQAAAQH+Af8BAQAA/wACAQEB/gABAAAAAAEA/QD9AQAA/QIAAf8C/wADAP8AAQAAAAAAAgIAAPwA/wD8AP//AAABAQMAAQIA/wAAAAABAAAA/f8DAf8B/QAC//0BAQABAAIB//8AAf8AAQEB//8AAQAB/gAA/wH7AAL//v8B/gAAAv4A/wIB//8BAf8A/gAAAP8AAf8A/AEAAQEBAP4BAAD9Av8B/v4AAQIAAQAA/wAA/wEA/gH/AgEDAQP+/QL///4AAgH+/wABAgD+AQL9///+AwL/Af8BAAEBAv//Af8AAf8AAP7/AQEBAP8AAv8AAP4BAfwB+wEBAwEC/gAC///+AgIC/f4BAQIA/gEB/f//+wECAAD/AQAAAAEAAQAAAP/+/gD/AQAA/wD/AP4AAAAAAQH+Af0BAQIABP/9AgD//gEBAf3+AQEBAP8BA/7/AP0BAP8AAAIBAwEC/v4C////Af4B//8AAAEA/wEA/gAA/QEB/gD9AQEDAQb//QIA/wADAwH+/gEBAgD/AAL+///+AQICAAEBAAEBAgAAAQABAf8AAAABAAABAP8AAQABAAEA/wEBAgAA/wAA/wABAAAC//4B/wAAAP8A/wD/AAAA/gD/AgABAQABAQH//gEAAP0A/wH9/wABAv8AAP8AAAD+AAEBAP/+AAEAAf8BAgD///8CAP8C/wD/AAAA/wAAAP4AAAIA/QAAAgD//wEC//8AAgIC/gAAAQAAAAAAAAAA/gABAgAAAQAB//4AAAAA///+AQAAAgAA/wAAAAL/AAD9AQAC/wEB//8AAQEB/wEBAf8C/wEAAP8AAAD//wEAAQD+//8B/wEA/wD///8BAAD+/wEB/wIAAf4A/gAA/wAA/wEA/wH/AQACAP4A/QEAAP4BAQH+AQABAAD/AAL/AAD/AP//AQAAAAAA/v8AAP8AAAAAAv//AAEAAf8AAf4AAAAA/wEA/gEAAAD/AQAAAAD///8A/wAAAf4AAAD//wEAAQH+/wH8AQECAQH+/gH///wBAwH+/QEBAQD/AQL+///+AQEAAPwAAAEAAgD+AAAA+wD+Av4BAAH/AAAAAf8AAP8B/gAAAf4A/wD/AAAAAAAA/v8B/gEAAAABAAAAAAAAAP78/wD+AQABAQH//QH/AP0A/wL/AAABAAD/AAD+AAD+AQAAAAH/AAAAAQD/AQAAAf8BAf/+AQAAAAAAAP8AAAD//AIA/gEA/wAAAP4B/wD+//8A/wEAAP4AAAABAAAA/wD//gEAAQECAAT//AL///4BAwL//wEA/wD/AAH9AAD/AQAAAP8AAAEBA//+Av8A/wEBAf4AAAAAAAAAAf8AAAAA/wEAAAAA/QABAAH/AAD9/wEA/wH/AP4AAAD//wAAAAD//wACAAD+AAAAAAAA/wAAAAH/A/8A/wD/AP7+AAD/AP7/AP8AAP4ABAAAAgAA//8AAf4AAAH/AAAA//8AAP8A//8A//4B/wH/AP4B/wAAAP4B/wMAAAAA/wD/AAAAAAEC/wD/AQABAQAA/wAAAPsAAAL//wABAQD/AAMAAAD/AAIAAf//AP8AAQD/AAAA/QH8AP7+AAAAAAAA/wEAAAAAAf8AAAEBAQED/v8A/wD//wAAAQAAAP8A/wEB/wD//wECAAAB/wD+Af8AAAEAAf4B/gH/AAABAAABAP3/AAAB//8AAAH/AP8AAQEC/wAA//z9AAAB/wD+AAAAAAIAAAP///8A/v0A/wEB//8BAAD+Av8B/QIAAQEA/wAB/wAAAQAC/wABAP/+AAMCAf8BAQH8/f8AAf8A/QABAAABAQEC//4AAAIAAAAAA/8A/wAA/v3+//8AAAAAAAAAAAAAAAEAAf8BAQEAAgAEAAABAAAA/wAA/wIAAQAA/wAD/wAAAgEDAQAC/wD+AP8BAgABAf78/f8AA/8A/gABAP8AAAEDAAH/AAABAAIAAf//AgD//v8BAv8AAAECAAABAP4AAAAAAf0A/QEBAQEE/f4D///8AQQBAP8BAQIA/wEB/f///wICAv8A/wD///4BA/8BAP/9//8BAv4AAQACAP4AAAAB////AAABAQMAAv/+Af8AAf8B//8AAAD/AAABAv//AP0AAQAB/QIAAgEB//4AAAD9BAIB//4BAQIAAAADAAAA/gEC/wD7AQACAP4AAAAAAPoBAwD//gAAAQABAAL/AAAB/wABAAABAAAA////AAAA/wIB/wIBAf8AAAAAAQEAAAAAAf8A/f8AAQABAAAAAAD+AQAAAP0AAAEA/wABAAAAAQEBAgD9AQABAP8A/wAAAP0CAQABAQAAAAAAAAIBAAAAAQABAP3+AAABAQAAAAAA/gEB/wAAAAABAAAAAQEAAP7/Af8AAAAAAAD/AP8AAAD/AAEAAAAAAAEB/wACAQAAAQD/AQD//QAAAP4CAgAAAAL/AAABAQD//wAAAP8BAQAB//4CAAL///4A/QIC/gEBAP/+/gIC///+AAD//wQBAQH+/v8A/QEAAgACAP8B/wD/AQAB//0BAQIA/wECAAAA/wAA/wD/AQACAAMAAAAAAP8BAgAA/AAAAQD/AAIAAAD/AAABAAH//wAA/gEB/wAAAAEA/gL/AP/+AQH//gEBAf////4B/wEAAgAEAAAAAAD+AgH///8AAAIA/wACAQEA/P8BAQABAQACAAIBAQAAAAD//wAC/wD/AAD/AAAAAAAAAP3+AAADAQQAAAD+AAD/AAQCAAH9AQACAP8AAf8A//4AAf8AAQEAAQEAAP8AAAAAAgEA//8AAQEAAAABAAAAAAACAAABAAABAPwAAQAAAP8B/gAAAAAAAQAAAAAAAAAC///+AP0AAQEB/v7+Af///QIBAf//AQABAP8AAv//AAACA/8BAAEAAQABAP8AAAD8AAICAP4BAAEA/wEC/gAAAQEB/wD//wAAAAAA/wAAAAAAAAAAAAAAAQD/AAEBAAACAAABAAEB/wD//gEC/wAA/f4AAAAAAAD+AQAA/wABAQIAAP8AAP0BAAEA//4A/wAAAgIA/wAAAAEA/gAF//8AAQEDAQABAP8A//8BAf8BAf/+/gABAf///wABAPwAAQED//0CAP/9//7//AEC/wABAP79/wEC/wD+AQD//wEBAQP+/gABAv8AAQABAAD/AAD+AAIA/wAAAAEAAAABAAAAAQABAAAAAAABAAEA/wAAAP8AAQH+AAAAAQD/AAP/AAABAQH/AAL9AAEAAP//AQAA/wEAAAAAAAAAAf8B//8AAAEBAP4B/gEBAwED/voC///+BAQC//4BAAMA/wED/f///AIC/gD+/QD/AAQA/QAAAQAAAQD+//8CAAEAAAEAAAAA//4BAP/9AP4A/wD+AAD//wD/AAAAAAD/AAAAAQAAAAAAAfwB+AMAAgIE//8BAAD6AQAB//4BAgH//wAC/gAA/AEC/QH7AwABAQT//QEAAPwDAwL+/gEBAgD/AAP+AAD8AQL6AfkDAQMBBf7+Av//+gECAv79AQIDAP8BBf3///4CAv4A/gAAAAEC//4AAAD+AAEB/gEAAQAA/wACAAAA/QEB/gD+AAAAAQAAAQAAAP3+/gEAAQAA/wAAAP4AAAAA//7/AP4BAP8BAv//AAAA/gADAf8AAAEBAP8A/wAAAP0BAP4AAP8AAAABAAAAAAAA/v4BAAIAAAAAAAD//wAA/wAA/gD/AP//AAMBAf4BAf3+AAAA//8A/wD//wACAAH////+Af0BAAEAAgD/AQAA/QAAAv8AAAEBAAAAAP4AAP4AAf0AAQAAAQEDAAAAAAD9AQAB//8AAQEA/wAAAAAA/wD//gEA/wABAQH/AAEA/wEA/wL/AgABAQD/AQD+AAD/AQD/AQAAAAIBBP//AgAA/wIDAv7/AAEAAP8BAf7/APoCAP8ABfwAAAACAAEBAAAEAf8B/wEAAQAA/wABAAAAAAAAAAAC/QD/AAEBAQAAAAH//gEAAQAA/gABAAD/AAAAAP0C/wb7//3//AID/wEBB/z6/gED///8AQH//AIBAAT//f0B+gMAAgEC//0BAAD5AgEC//4BAQEA/wAB/wAA/AEB/QD7AwACAQP//gH/APwCAgL+/gECAgD/AAL/AAD8AQL9AfoDAQMBAv78Av//+AICA/79AQED//4BAv7///0CAf4A/QEAAAEB//4BAAD+/wIA/wEAAQAA/wACAAAA/gECAAD9AQAAAAAAAAAAAfv+/wEAAAAA/wAAAP0AAAAAAP7/AP4AAAABAgD/AQAA/QAAAP8AAAEAAAAAAQAAAP4BAv8A/gAAAQD+/wAAAAD9/v8BAAEAAP8A/wD+AAAA/wEA/gD/AP8AAAMBAP8AAf7+/wD/AAABAAAAAP8CAQH+/wD9Af0CAAMBA//+AQAA/f8BAf4AAAABAP8A//8AAP8BAP4A/QEAAQABAAABAAD8AAAB//8AAQEAAAAAAAAA/gEB/wABAAACAQH//gIAAAAAAQL+AgAAAgD/AQD+AAD/Af/+Af4BAQIBAv7/Av///wIAAf//AQEBAP8BAf0A//wCAQAABv4AAAAA/wABAAAEAP4B/wAAAAAA/wAAAAAAAAEAAAAD/wD/AP0AAP8AAAL+/gAAAP8A/wAAAP0AAAAB//0E/wf8//v/+QEE/wEBB/36/wEC///7AQEA+wIBAQT//AAAAAEAAAAAAAAAAAABAgIA//8AAQEA/wACAAAA/wEBAAD/AAAAAP8AAQAAAAIA/wABAAAAAAEBAAEAAQAD///+AP4AAQIAAf/+AQAA/wMDAAD+AQAAAAABA/4AAAACAv8A/wIAAQAB//8AAP//AAABAAAAAAEA/wEA/gAAAAEB/wAB/wAAAAEAAAEAAAH/AAD//wAAAAAAAAIAAAABAAEBAAEAAP8AAAEA/wAA/gD/AAABAAD/AAEA//8BAAIAAAABAP4BAQEA//4C/wABAgEA/wAAAQEA/wAE//8A/wICAAACAP8AAAABAv8BAAH+/QABAf8A/wEBAP0AAQAC/v8BAAL9AP4A/gEC/wABAf79/gEC/wD9AQAAAAEAAAL//gAAAAAAAQABAP8BAAACAf8AAP8AAAEAAAAA/wAAAgAB/wABAAAAAAIA/wH/AAAA/wEAAAABAAH/AAL+AAAAAQL/AAH9AAEAAAD+AQAAAQEAAQABAAAAAf8BAv4AAP8AAv4B//8BAwED/vwD/v8AAwIC//8AAAMA/gEC/f///AID/gEB+wABAAIA/wEAAAMAAQD/Af8AAQEAAAIBAQAA/gH/AAT6AP8AAQD///8ABP8A/wABAAD/AP8AAQAAAAEAAP8A/AMAAwED/wABAAD9AgMA//0BAQQAAAAA/wAA/gEC/wD7AQABAAP/AQEAAP4DAQAA/wAAAgAAAAD/AAAAAAEBAAAAAAAAAv8BAAAAAAEB/wEAAAABAAAAAAAAAAAAAQAA/gAAAQABAAEAAAD+AgIAAP8BAAEAAAD//wAAAAABAQD9AQAAAAH/AAH/AP8BAgAAAAAAAgAAAP8A/wAAAAEBAP3+AP8AAP8BAQAAAAP/AQH/AAEBAAAA/wAAAP4AAQAA//8AAAABAAAAAAAA/wEAAQEAAAEBAAADAgAAAAABAgD//gD9//0BAgAAAQIA/gABAf8AAAABAPwBAQAA//8E/wH+//wA/AID/gEBAP3+/wIC///+AAH//wIBAQH9/wEB/wEAAAABAP4BAAD+AgEBAP4AAQIAAAACAAAA/wABAAD+AQABAAD//wAAAP0DAAAA+wEAAgD/AAEAAAD/AAID//7+AP8A/wEBAAAA/wH//wH///8BAAEA/wEAAP////8AAAEAAAEF//8CAAD/AwIB//4BAQMAAAACAAAA/AABAAABAgAAAP4BAAAAAAEBAAL//v//AAAAAP4AAAD/AP//AP4AAAEA///+Af//AQQDAAD9AP8DAAAAAP7/AP4BAf4B//8B/wAB//8BAAAC/wAAAAIAAAAA/wAB//8AAAEB/gH8AAABAQEA/wAAAPoBAgH/AAAAAAABAAMAAAAAAQH+AAD/AP4BBQEBAAAB//4BAP///wH/AQAA/wEAAQD/AP0B/gIAAQEBAP0BAAD+AAAB/wEAAP8AAAAB/gAAAAEA/wD+/wD+AAIBAQAAAP//AgAAAAAA/wABAP8BAAAB////AP//AP4AAQIB/wEBAf7/AP8C/wH+AAL/AQEAAQP+AP8A/v4AAAEBAP4CAAACAQEC/gEAAQEAAAAC/wAAAQABAQD/Af/+AAABAf4BAf/+/gABAf8A/gAC/wABAQEB////AQABAP8AAQL//wAAAf8BAAECAAD+AAEAAAEAAAEAAP8AAAEAAAAEAAABAAD///8B/wIAAAEAAAADAAAAAwADAAAAAP/9//8CBP4BAQD+/QAAA/8A/QAB/wABAAACAP/+Af4CAAEAAQABAQAA/v4BAf8CAAEAAAAAAQAAAAD/Af4B+QIBBAH+//8E///9AwID/wAAAQMAAAEA/f///wADAv8B//8B//4BBv4BAQX+//8DAv8AAAAB/wADAAEC//8BAP0CAAIB/wD/AQAA/AECAf8AAAEBAAAAAQAAAPz/Av0AAP8AAAAC//8C/wAA/v8B/wIAAf4A/wAA/gAA/wEC/gABAAACAQYA/wAAAP///gD/AAABAQD/AAL/AAABAAH9AAL+AAABAwABAP8AAf///wABAAAAAf8AAgAAAAH/AP8AAQAAAAADAAL/AAAB/v//AP8AAAAA/wD9AQAAAQD+/wH+AwEBAQH+/AH///4DAAH//gEAAQD+AQH+///+AQIAAP4CAAEABwD/AAAA/f8AAP8BAAABAP8A/QAAAAAA//8AAP8A/wD/AAD/AAAA//8AAAIAAP8B/wAAAAAAAP79/wH/AAECAAYA/gL///0A/wL/AgABAQD/AQD+AP8AAQH/AQABAP8AAAD+/wAA////AAABAQD/Af8AAgAAAAD//QAA/gEAAAABAAAAAAAA/v0AAAEAAP8A/wD/AQAAAP8A/gH/AQEBAAP/+wH///8AAQH+AAABAgD+AAL+/wD/AAH/AQEAAAIAAgAAAQAAAQABAf8BAAECAP8AAf8AAP///wEAAP8A/wACAf4A/wD+AAEAAAEAAP8A/wD/AAAAAf8AAAAB/wAC/wMBAP8A/wD//gEBAwAA/wD/AAEAAAAAAP7/AAABAAAAAgD+Af///QD+Af4CAAH/AP4A//8AAP8A//4B/gAA/wECAP8AAAD+AQEAAP4AAQAA/wABAAAA/QEA/gH/AgEDAQb//AL///4CBAL//gEBAv//AQP9///+AgP/Af8BAAIABf//AQAAAAIAAQD+AAEBAP8AAf8AAP0BAv0B/QEBAwEF/v0D////BAIC/v0CAQMA/wEE/v///AIDAAAAAQAAAAQAAAAAAAD//wAA/wABAQD/AAABAAD/AgL+AP4DAQIBBP/+Av//AAEAAf7/AQEBAP4BAv7///0BAf4BAAEBAgAE//8BAAAAAAAB/wAAAQEA/gEC/////QIC/wH9AgEDAQAA/QP///4CBAL9/gEBAwD/AQL+///9AQEBAQABAAEA/wAAAAAAAf8AAP8BAAEBAAAAAgAAAAEA//4A/wAAAAAGAAEAAAAB/wEAAAEBAQAB/wAC/wAA/QAAAQD/AgACAAcAAAAAAAABAgAA/gEBAgD/AAH/AAD+AQAAAAD/AP8AAQACAQAAA///AQAC/wEAAAAAAAAAAP4AAAEA/gEAAQD/AAEBAAABAQAA/wEAAQAAAAAB/wAA/gEBAQABAQAAAAD/AAAAAAH+AAABA/8AAgAAAP3/AAD+AQADAAH//wH//gEB/wEB/wD+/wIAAAAAAAH//wEAAQD+AfwA/wIAAgACAP4AAAD+AAEB//8AAQEAAAAB/wAA/wAAAAH/AgAAAQUA/gAAAAAB/wEA/wABAAAAAAEAAAD9AP/+AAH/AQIAAP7/Af//BAEAAAD/AAABAAABAf8A/wAAAQAB/gMBAgAB//4B/wD+AAAAAP4AAAL/AAAA/wD/AAEAAf8AAAAA/wIA/wAAAAEA//4BAQAAAAABAAEBAAD/AAIBAAABAP8AAf//Af8A/f//AAD+AAAAAP8AAf8AAAABAf8A/wIAAwAEAP4AAAD/AgIBAAAAAAIAAQAD/wAAAAH/AAAAAgEBAAP+/gH///wA/gEAAAAAAgAAAQP+//8AAgL/AP7/AAAAAgAAAAAA/gEBAAEAAAABAAEAAAAAAP8A/wAAAP8AAf8BAAEAAAD/Af//AQAAAAAAAAAAAQAA/wACAf/+/v///wMAAf8BAfsCAv4BAQAAAAAC/wEBAAAB////AAEAAAEAA/8B//8A/QEB/wAC/wABAP8AAv8AAP8BAwEA//8AAQAA/wEA/wD+AQL/AQL//wMAAAD//wAA/wABAf///QAA//wAA/8AAAEBAAABAP8AAAABAPwAAAAB//0BAAH+AP///AECAAAAAgEB/wEBAAACAAEA//8AAAIAAf8AAAIAAwAB//4AAAAAAQEBAP8AAAIAAAAC/wAA/wEA/gAAAQD/AAH/AAAA/wAAAAAA/wABAAD/AAIAAAAAAAACAAL///0A/gIB/wABBP///wAB/wD/AAD//gIBAQD//v0A/wABAAEA//4B//8BAQAAAP8AAQEA/wED/wAAAQEBAAD+AAD9AAIAAAAAAAAA/gAA/wAA/wAAAAABAAD/AQACAP//AP0A/gECAAAAAf/8/wAAAAD/AAAAAAEBAAAA/v8A/gAA/wAAAP//AAAAAf0BAAAAAP8AAAD/AAAAAAEAAAAAAQH/AQD//wH//wECAAD+/wEAAQAAAQH/////AQABAP8AAP4AAAAB/wAAAP/+AAH//wAAAAAAAQEAAAEBAAEAAP8A/gD+AAL/AAAC//7/AQAAAP8AAQABAQAAAgD9/gD9AAEBAAH/AAAAAP8B/gAA/wEAAQD/AAH/AAD/AQH+AP0BAAAAA//9AP///QMBAP//AAACAP8AAv8AAPwBAgAA/v8AAAABAAAAAAABAQEBAAABAAEAAAAD/wAAAAAAAf8B/wD/AP8AAf8BAQEAAAAAAAAA/gAB//8BAAH+/gABAAEDAAD//wD+AAAA/wECAAD+AAABAAAA/v8AAAAAAAAA/gEAAQEC//4AAAD+AQEA/wAAAAAAAAACAAAA/wEA/gH5AwEEAQX+/gL///sDAwL9/AECA///AQP9AP/7AQL/Af0BAQEBAf/+Af///QMDAf/+AQAB//8AAP4A//4CAv8B/AMBBAEE/v4B///+BAIB//4CAQP//gEE/v///AIDAAH9AgECAQH//QH///0CAwH+/gEBA//+AQP+////AwMAAPwCAAEBA/7/AAAA/wIBAAD9AQEC/wAAAv8AAP4BAQEA/gEAAAAAAQEAAAD/AAAA//8AAAEAAAABAAAAAAAA/gD/AQEBAAD//wH//wIBAQEAAQAAAQD/AQH///8AAQD+AfsCAQMBAf/+AAD//AMDAf7+AQED//8AA/7/APwCAgAAAQAA/wD/AAH/AAACAP//AQP///8AAQD/AQAAAf///wD9AAAAAQAAAAAAAAEBAgAAAf8AAQD/AAIAAAD+AQD/AP4CAAEBAgD+AAAA/wICAP//AQABAAAA/gAAAP0BAQEAAP///gD/AAIAAAAB/v7/AAIAAP4AAQAAAQAAAP/+AAD/AgEBAQIAAAH//wECAAD//gEAAgD/AAEAAAAAAQEC/wP///4A/AID/gABA/78/wEBAP/+AAH//gIBAQP+/QAB/wAAAAEE//8B/wAAAQEAAAAAAQEA/wAAAAAA/wICAQAC/v//AQEBAv8AAQH//v8BAP8A/gAA//8BAQEBAP4AAAD+AP8A/wEBAAAAAAD/Af8B/wD/AP8A/QAAAAAAAAAB/QEBAQIH//wB///8AgEC//8BAQD//gEB/f///QIC/QH5AgEDAgT+/QP///kDAwL9/AECA//9AQT9///8AgMAAf4BAQEBBf/+Af///gIBAv8AAQEBAP4B//0A//4CAv8A/gAAAQEBAP4BAP//AQAA//8AAAEA/gAA/gAA/gECAAABAAAAAAIBAQAAAAAAAAAA/wEAAAD/AAH/AAD/AAABAAL/AAAA/gD/AP8AAgEAAAEBAAAAAP4A/wAAAAEAAQEBAP8AAAEDAAAAAP8AAQEAAP8AAAIAAAABAAAA/v//AgAAAgAAAAMB/wAAAQD/AP8B/wAAAAD+/wEBAAEA/wD/Af0CAQIBA/77Av///AMEA/78AgEDAP0BAv3///sCAwAAAQABAwABAP0B/v8AAQEAAAAAAAEA/wEC/gAAAP8B/wAAAAEDAQL//wL+//0DAgL+/gECBAD+AQH9///7AQIE/wX7//3/+wEC/gEABP/8/gMF//79AQH//AMBAQX+/QEAAv///QABAQP9AQEC/v7+AgT///0BAf//AgEBAv7+/wAAAQABAQIAAP8AAAACAf8B/wAAAQAAAAAAAAD/AAEAAAH/AAAA/gEBAP//AQEAAAH/AAACAAAA/wAAAAD/AAIAAQIAAQACAP7/AAD+AgH/AP4BAAIA/wD/AAAA/gED/wD/AQECAQAA/wD/AAECAwAB/gAAAwD/AQL/AAAAAAIBAAMAAAEAAAABAAAAAQAA/wH+AP8CAP8AAQAAAAEBAQH/A/4A/wD8AgP+AAAF/gD/AgH///4BAQD+AAEABP7/BP8G/f/9//sCBP4BAQT9/f4DAv/+/QEC//wCAQEE/vwCAAH/AAAB/gAA/wAAAAIB/wH/AP8BAAAAAQAAAAIAAQL/BP4AAAD8AQL/AAAC/wD/AgD//wAAAQD/AQEAA/7/Av8D/QAAAP0BA/4AAAP///8CAf///wEBAPwBAQAD/f4BAAIBAAEA/gAAAP8AAAEDAP//AQAAAAEA//8AAAAAAwP/A/3//gD8AAH+AAEE/f/+AQL///0AAv/9AQABA/7+/wAABQADAQP9/AL///8DAgH//AIABP//AAL9/wD+AgQC/wT+AAH/BAAAAAAAAv/9/wEAAf8BAAEA/wAAAAL/AQH/Af4A///9AQH/AAH///4AAQIAAP4AAAD7AAEAAf//AAABAQABAAH///8AAAACAv8B/gAAAgAA/wEAAAACAAACAAL+AAEA/v8BAQAAAv/+AAEBAP8AAAEA/v8AAAIAAAEAAAD/AQACAAH/AAAA/wD/AAABAP8AAQD+AAEAAf8AAQABAQAAAAP//wD/AP7//wAAAQAAAAD/AAH/AAAAAAEAAP4AAAAB/wAAAP//AAABAP8AAAAAAAAA/v8A////AAEAAAEAAAD///8B/wD+/v8AAAAAAAAAAAAB/gAAAQAAAQAAAQABAAL//wH///4BAQD//wEBAgAAAAD/AP/+AQABAAIBAAEA//8AAP8A/v//AP//AQEBAP8AAf8AAAH//wL/AgH/AP8BAAH/AAD///8AAAD/AAAAAAD+AAAAAQH+/gH+AwEEAQb++gP+//oCAgL9/gECBP/9AQf7///8AgD+AAD9Af4B/wD/AP//AP8CAP4CAAABAAABAP8AAAD//wAAAQAA/wADAAD//wAA/gL//wEAAP8A/wABAAAAAP4A/wAC/wEAAAQA/wH//wL+AAH/AQAAAgD/AAD/Af///QABAAT+//3/AQEBAAAAA/7///8AAAABAQH//QEBAAL9/wH/Av7//v/9AQL+AQED/v7+AQP+//4BAv/8AgEBAv/9/AD/AAABAAIAAAH//wADAwD//gEBAQAAAAH/AAAAAQP9AP0EAQMABP/9Af//+wMDAf79AgIE//8BAv////4CBP8A/wEBAgEB//4BAP/9AwQA//4BAQP//wABAAAA/gED/wD/AwABAAL//gH//wADAwD//wIBAf//AAL//wD+AwMA/wL+/////AEEAAAAAP4A/wECAAD/AQIA/wEBAQL//gEAAQAAAAD+AAIA/wABAAEB/wABAP8A/wAAAQAAAAIC/QD9AgEDAAD+/QL//gACAwH//wEBAgAAAQP//wD/AQIB/wT+/wAA/wEE/gAAAv///wAC/wD+AAH//wIAAAL//wAAAgEAAQAAAAL/AP//Af8A/wAAAQEAAAD/AQAAAgEA/gD+AQEEAAD//wAA//8CAgEAAQAAAgAAAAQBAAD//wH+AQADAQMB////Av///wIBAf7/AQEBAAABBP8A//0AAv0B/AMBAwAA/gAC///+AgEC/v8CAQMAAAEE/////gEDAP8CAQD+AAIAA/8A/wEA/f8AAv//AAAB/wABAQAA/v8BAAUDAAMAAP8BAQD/AwAB/wH/AQABAAAAAP8A/wL/AgH/Af4A/f//AQH/AQAB//3/AgT///0AAQD7AQEAAv///wD/AgEBAAP+/QEA//4FAwH//wEBAP//AQD+///+AgT/AAEAAP8AAv8A/wD/AQQB/wECAAAAAAAA/wAAAAAAAAD/A/4A//8AAgP/AQEDAf//AQP///4AAf/7AgEBA/79AP8E/gD9//4BA/8BAQL/AP8B/wD//gEB//4DAQED/QD9Af0DAQQAAv38A//+/gQDAf/9AQEF//8BAf3///4CAgD/AAAA//8BAAD/AQAB/wABAQEAAP8AAAABAAAAAQH/AQD/AAD9AAEA/wIBAAEA/wAA/wAAAAAAAAAAAAD/AAH/AP4BAAD/Af/+AQD//v8AAQD/Af8AAAAAAf7/AAABAAH/Af8A/v/+AQIBAQAA/v//AQAA//4AAv/7AQEAAv//Af8AAQAA/wAA/gAB/wH/AAABAAAA/wAAAAABAAABAAD/AP0AAQIAAv/9AgD//gEDAv7+AQACAAAAAP8AAP//Av8A//4AAP8C//4BAP8B/wEBAP8A/wAAAQACAAAAAP8A/gH7AAICAAP//QL//vwBAwL9/gIBAgD+AQH+AP/8/wQAAP8B//3/AAL/AAEAAvv//wH/Af//AAP/AP8AAAL//gb+B/j/9f75BAX8AgIH+Pn9BAn9/vUBA//2BAIBBvz6AAAA/AH6AAAA/gH//wD+AAEABv8A+wAAAfn9AP//Af8B/wP5APz//gEC/wAABP7+AAIF/v/8AAEA/AABAAMA/QEAAAEBAAAB//8B////AgEBAP8AAAD/AAEA/v//AAIDA/8D/P/+//wCBP0BAQT+//8BAv///wEB//0DAQEE/P0B/wEBAP7/AAD/AAD/AgH/AAEBAAAAAAEAAAAA/wEAAQL/AwAA//8AAAH/AQABAAAAAQH//wAAAf//AAAAAwABAf8D/QAB//3/AgAA/wMBAP8CAAD/AAABAAAAAAAD/wAD/wP+/wD//gEC/wEBAwAA/wIA//4BAAH/AAIAAAL+/QP/BAD/AP/+AAP/AQEC/v7+AgAA/gEAAv/+AAAAAwD/A/8BAAAC//7/Av8AAAABAf8C/gH/AwAAAAD/AP8C/wAAAAACAQQAAP4BAf///wICAQH7Af8F/wABBP7//wEAAQT+A///Af/9AQb9AQEB/v/+A////gEAAv8BAwEABf3+Bv4E/f79/v4DBvsBAgT8/P0EAP79/wED/v8FAQEG/P0B/wD+AAUA/v8CAAABAgIC/wL+Af8DAQIAAgAAAAP8/gAAAAAAAAAAAAAAAAAA/wEAAAEAAAAAAAD/AQAAAAAAAQACAwADAP7//wAAAAEB//8B/QD/AwABAAQAAAACAAH6APr7APkCBQEAAQAA+v35A/sF/wP4AP0A+v8AAP0A/f0B/vwA/AEEAQAAAAD9/f4D/QL/AvwA/gAAAAAA/gD9AAECAAECAP///gH//wEBAAH//gABAQD/AAL+//8AAgAAAP/9APwBAwECAAAB/v//AAAC/wH9Af8A/QAAAAAA/wEA/wAAAAAA//wCAP//AQIB/wAAAQEA/wAC/f8AAAICAAAB/gABAf//AAD//wADAAD/AAEBAQD/AAAA/wABAAH/AAAAAP7/AQIB/wEA/gL/AAAC/wD9AAEA/QEBAQH/Av4A/wIA/wABAP4BAP////4B/wEAAf4AAAAC/wAA/gECAAABAgD/AAEBAP8AAAD///8AAQAAAAAAAAABAAD/AP8BAAEA//8A/wED/wEBAv///wEB///+AAEA/wEAAAIA/wEA/v8BAQAB/wEBAP//BAAAAP8AAAIAAAABAQAA/gEBAAABAAADAAD/AQAAAP8BAQAB/gAAAv//AAH/AP8BAAH/AP8CAAQAAv7+Af8AAQIDAAD+AQABAAABAf3///8CAgAAAAAAAAH/AP8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD9/gD9AQIA//8AAP7//wD/AQAB/gAAAAP/AAD/AQD+AP8AAAEBAgACAAAAAP8AAQAB/wAAAAAA/AAAAAAAAP8AAgIAAgAEAAEAAAD+AQIAAAD/AAIAAAD+AgAAAAH//wAAAQAC/wAAAAAAAAECAv8BAAAAAwABAP8BAAABAAH/AAYAAP3//gABAAAAAgMAAgD+AQABAAAAAP4AAAEAAwABBP0A/f8A/wL/AAAAAQEB/wAAAAAA/gD+AAAAAf7/AAAC/QD///0BBP4BAQL//f8BAv7/AAECAP4DAAIC/gAAAP8BAAIA/QEA/wD/AQH//gEBAP/9AAEAAAEAAQEA/v4A/v8A/wAAAP8AAAD//gAB/wEAAP8A/wD//wAA/wH+/gD+AgD+AAAA/QH/AP///gH+AQAAAQD/AQD+AAD9AAAAAAAAAAAAAAAAAAAB//3/AAH/AAABAP8AAAEAAAEAAAH//vsAAAAB/wQA//8AAP//AQEAAP8AAAAD/v//AQP/AP8DBQAB/wIAAQABAQL/AAH//gH/Af8CAAEC//8D/wL9Af0AAP4AAwMB/wAA+fz+AP0D/wH7/wH/BgUBAfz//gAA/wAAAAEAAAAAAAAA/wAAAAEAAP8AAAAAAAAAAAAAAQEBAAAAAQAB/gAAAAAAAgH/AAABAAD/AP//AAAAAAAAAAEAAAIAAQABAAAAAAAA/wD//wAAAAEA/v8AAAEAAQAAAgIAAAD/AAD/AAADAfwAAP4AAAEAAQD/AAAAAQAAAQAB/gD+/wAB/v8BAAH/AP8AAQAA/gABAAECAQABAP8CAP8CAAEAAP/+AAAAAQACAAD+AAACAAAAAgAAAP8AAAAAAf4AAAAB//4A/wACAP4B/v4BAQIAAAD+/wAA/gAAAAEAAAABAAD+AAD/APwD/wH//wABAQD/AAD///8AAgL/AAIAAAEA/v8AAAAAAf79Af8AAAD/AAAA/gAAAAEB//4BAQAA/wECAfwAAAABAP8AAAP/AP8A/wAA/gEA+//9/wAE/gD7/wMCAQEBAQH+AAEA/wEAAwEBAAEAAQD//AL6Afz9APoBAQP6/f8B+/0AAf0F/gH5Af4A+/8CAfv++f8AAQQAAgD+/wAB/wAB/wABAP4B////AAAB//8AAP//AAD+AgACAP4AAgAA/gIDAP8CAQD//wACAAMCAAAEAQL9Avr6APwBBAIA/wAB9wL6APwH/QP8Af8AAQgBAvkA+wAAAAAAAAH/Af8AAAAAAAAAAAAAAAAAAAABAQAAAAAA/QH+AAD/AQMAAAEAAPz+AAL/A/8B/QD/APv/AAD/AP8CAAH/AAD/AP8AAAAAAAL/AQD/AQACAAAAAAAAAP//AgH/AQAAAQD/AAL/AAAD/gH/AgAA/wEAAAD+AAAAAAABAP8CAAABAAD/AwAAAAEB//8A/wAAAAABAAD/AAACAQD+APz+AP8AAP//AQD//gH+AP7/AQH+AAAAAf3/AAABAf4B/gEAAQACAP4BAAD/AAMA/gAAAQAAAAAB/wAA/gEBAQEB/wD+AP4A/wAAAAAAAAAAAAAAAAAAAAAAAAAB//8DAAL9AP4A/gEA/wAAA/8A/wEB///+AQEA/wAAAQD+/v8BAAMAAwEB//4BAAD+AP4B/wACAQAAAQAC/gAAAv8EBP8CAQAD/wD/BP8BAAMCAv8D/gH/BAACAAIBAAABAAIBAP8AAP8A/wEAAQAA//8AAf8B/wD+AP8AAP8BAP4A/wAB//8AAAADAf4BAAAA/gAB/wAAAAABAAD+/wEAAwD/AQD+/wAAAP8A/f8AAQAB/v8BBAAC/wD9AAACAAABAAEBAf/+AAMB/f3+AP8AAPsAAf8A/wD9AP8AAP0AAAH//gAAAAAAAAEAAAAAAAAAAP8B/wAAAQAAAAABAAAA/wAA/AD/AAD9AQEAAAAAAP3/AQEAAwAB/QAAAP0AAAABAAACAP8BAAIA/wAB/wAAAAAAAAEBAAAAAAAAAQAAAAD/AAEBAP8AAQD/AQAAAAAAAAEAAQEA/wAAAAD/AAAAAf8AAAAAAAAAAAAB/wAAAAD+AAAAAAEAAAAAAAIBAAAAAP4AAP3/AAAAAQAAAAAA/wABAP8AAAAAAAEA/wIAAQEBAP8BAQEAAQAAAQD/AAD/AAD/AQAAAAAA/gAAAQAAAAD/AwAB/wABAP//AAEAAAMAAQAAAAABAQEBAP/+AAACAQAAAAIAAf8A/gAAAAAAAQEAAAAAAAAAAf8A/v8AAAAAAf///gEAAAD9AQD///8A/wAAAQEAAP8AAAAC/wAAAgEBAAH//wABAAD/Af8AAAAA/gD/AAAAAAAAAAD+AAD/AAABAAD/AAH/AwD+/wAA/wD/AAD/AP8BAAAA/QAAAAH/AQEAAgIAAQD9AAT/AAABAQP/AP4AAAEAAAD/AAAA/gEBAAH//wD9AAAAAP8AAAABAf8AAf8AAAABAAEBAAD8AAH/AP0AAAAB/gD4/wAA/v4AAP8D/wH9AP8A/gEBAAAB/v8AAAAA/wEAAAAAAAAB//4AAAAAAP8AAAAAAQAAAAAAAQAA/wABAf8AAQAAAAAAAQH/AP8BAQD/AAAAAAD+AAEAAAEDAAMA/gABAAAAAAAB/wH/Af8BAAAAAP8AAAEAAAIAAQEAAgD/AAH/AAABAAD/Af8AAAIAAQABAAAAAQAA/gD/AAAAAAAAAAEAAP4AAAAAAf8AAAAAAP//AAD/AAD/AP7/AP8AAQH/AAAA/gH///8B/wD/AAAAAAAAAP8A//4AAP4A/gECAAEA/wD+AQAC/wEAAf8A/wD//gAAAAEB/AH8/QD9AQEB/AAAAPz7/QH/BP8B+wD/AAH/AAD+//0AAAD/AP4BAAABAQD//gD/Af8CAAH/AQAAAf8AAAMAAP8A/wEAAgAB/v4AAAAAAv8AAP0BAAMA/wAAAP8AAAEEAwD//QAAAP8B/P8AAAABAwABAQD//wAAAAMAAAEB/wEBAAEAAAEAAQABAAAAAAEDAAAAAAABAP8AAAAAAP4A/QABAQEAAAABAAH/AAABAgEAAP4AAAAA/wD8AAAAAQAAAQEA/gD/Af8AAQAAAAD/AgD+Af8B/wD/AAD/AAD+AAABAv8AAP8AAAH/AAAB/v4FAQACAAD+AP8AAAEAAP//AAAAAAEAAAH/AAAAAAAAAP8AAAAAAAAAAAD/AAAAAAAA/gD+/wD/AQAAAQAAAP/+/wEAAgAA/gAAAP8AAAD/AP8AAAL+AAIA/wABAAAAAAAAAv8BAQEAAAAAAAAAAAAAAf8B//8AAAEAAP4BAAAA/f8C/gH/Af8A/wABAAEAAAD//wACAAABAAP/AAAAAP8CAgH//wAAAQD/AAH/AAABAQEBAAD/AP8AAAEBAQAAAQEBAAAB/wD/AAAAAf8AAAEAAQIAAgIAA////wL/AAAAAgL+Av4B/wIAAQADAAAAAwABAAD//wD/AAMCAAAAAAEBAQAA/gAAAQEAAAAAAAD9////AAEAAAAAAAD+AAAAAP4BAAAB/wABAAAA/wAAAAAA/gEAAQIAAgAC/wEAAAABAQH/Af8AAAMAAAABAAAAAgABAAD/AQABAAH/AQAAAAEABAAAAAAAAQAAAAEAAAD/AP8C///9AAEAAAH6/wEAAAED/wIBAP4AAAAAAwAAAAH+AAEA/wAAAQD/AQH/AAD/AAD/AP8AAAAAAAACAAAAAAD//wH//gD9AQAC//8AAP/+AAD/Av8B/QEAAAAAAAD9APz/AAIBAAAAAf8BAQAA/v8DAAABAAD/AAAAAAAAAP4AAQAA/wAAAAEAAAAAAAAA/wABAAAAAf8AAAD/AAAA/wAAAQAAAQACAP8AAv8AAAACAQAAAAAAAgAAAAEAAAABAAD+AAAAAP8A/wD/AAAA////AP8BAAAAAAAAAf8AAAAAAQABAP8A//8A/wD/AAD/AP4B/wEAAf8A/wD//wAAAAH/AQABAAAAAQEA/wEAAAIA/wEA/gABAgD/AAIAAAD/AAAAAAIAAP8AAAD+AP8AAwAAAAAAAP//AAEAAQH/AAABAgAAAAAAAAAAAAEAAAAAAP8AAAAAAAEAAAD/AQAAAAD//AEA/wD/AAAA/gEAAPwAAP8A/gAB+wAAAAT/AAAA/wACAf7/AP4BAQD+AgAAAv4BAQABAAH+AQEA//0AAAIAAQIBAf///wD/AAL/AAEDAAD/Af8A/wEAAQAAAgAA////AQABAwACAQAB/QAAAAIA/gAA/wAB/wAAAAIDAQD/AP7/AAIBAP8AAQEBAAAAAgIBAP//AAAAAAAAAAIAAPz//v0AAwEAAAD/////AAD/AQEA/wMAAQAA/gD//gAAAwD+/AH9AAD9AAABAgAAAPwAAgABAgAA/wAAAAAAAAAC/wAAAAH/AAEAAgL9AQAA////AAEBAP8AAP8AAQEAAP7/AQAAAAAAAAAAAAAAAAAA/wAAAAAAAAAAAAAAAAAAAAAA/wD+/wAAAQEBAAAAAAD/AQEAAf8A/wD/AP4AAAABAP//AAAAAP8AAf8BAAAAAAH/Af4BAAH+AP8A/wAAAAABAQAAAwAAAAACAAL/AAACAAAA//8AAQAAAAD/AAAAAAAA/gD+AAD9AAEA/wAAAP4A+wD/AQAAAAAAAP8AAAD/AAAAAAICAAEA/wEAAQAAAgD//wH+AAD+AAEAAAEAAAAA/v0A/gEAAf8BAP8BAAACAAP/AP4BAAEBAQABAf8A/QD//gABAQD/AP4CAP8AAAEC+/8A/gD/AQABAAAAAQABAAEBAP4CAAIA/wAA/wEAAQH+/gH/Af//AAIAAQAAAAMAAP8A//8A/wEEAP8C/wAA//0B/wEAAf4A/wAA/gAA/wH+AAEAAQD9AQIAAAEAAP4AAAAB/wABAAAAAAIAAAAAAP//AAMDAAEA/gD/AgAA////Af8AAQD/AAEAAAIAAAH///8AAQAA/QIBAP4DAAAB+wAA/QT/AfsA/wACAAAA/f/+AAAAAP/9APwAAQAAAf7/BAH9Bv8A/QABAAEBAQH/AAEBAAD9AAAB/wH9AAAAAAIBAf8BAAABAAAABAABAAH//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/gH+AQD/Af8AAQEAAP//AQEAAQABAAAAAAABAAAAAAH+AQAAAP8AAQH+Af8A/wAAAP8B/gD+AP4A/wAAAP8B/gEAAAAAAQAAAAIAAAACAQEAAAD/AAEAAAAAAQAAAv8AAAAC/gD+AAEAAP8AAAL/AAH/AAAAAQEBAAABAAD//wEDAAMB/wL//QAD/gEAAgIB/gL/AP4CAAEAAQAAAAT/AfwB/f4A/QAAAf8BAAD//voB/gAAAf0BAAD//wAAAAH/+wAA/wD8/wMA/gAAAP7//wL/AQEB/QABAP79AAD/AP8AAP8CAAEAAQH/AAAAAQD+AAH/AAEBAAAA/QAAAP8A/wEAAQMAAgAAAAEAAAACAf7/Av0AAAMAAAD/AAAAAgAB/gEAAAD+AQIDAv8AAP8B/v///wEB/gD+AP4AAgABAAD/AAMAAAAAAQABAAAAA///AAH//wAAAAAA/wEAAAIA//0B+v4A/QIDAPwCAAD8/PwB/AIAAfwB/wD+AgAA/gD9/wAAAQD/AAH//gIAAAH/AAH+AQAAAAABAP7/AAAAAP8DAP3/AAMB/f8CAQAA/f7+Af4E/wIAAAAAAAAAAPkA/gAAAQAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAQAAAAAA/gH+/wD/AQIAAAIAAP7+AAH/Av8B/wD/AP4AAAAAAAD+AP4AAP0A/gEC/wAA/AD8Af8CAAD8AAAAAAABAP4AAf4A/gAAAAAAAf7/AAAAAQABAAAAAf8A/wABAQEA/v//AAABAAAAAP8AAP8AAAIBAf//AAAA/gD/AAP/AAABAP8EAAQCAAP//gABAAAAAwEA/wD9Af8DAAAAAgAAAAEBAf4B/P4A/wD/AAAAAAD/Af0B/wAAAf8A/wD//gAAAAEAAgAAAAABAP4AAv8AAAL+/wABAQAAAQEBAAABAAABAAEAAP3/AAAABAAC/wAA/P39AAABAAH/AAAA/wEAAP4AAP8A/gAA/gADAP4BAP/9/v4C/wEAAP8AAAD///8AAAEBA/8DAgAD//8AAP8AAAIAAP8C/gD/AwACAAEBAAAAAQH9Afr+APsBAgH5Av8A+/38Av4EAAD6AP4A/P8AAPsA/AEA/wIAAwH8AP8BAAAB/wQBAP8B/wAAAAAAAAAAAwEC/wD+/wABAf4AAAD/AAEAAQD+AAAAAQH/AAABAAACAP8AAAIAAAIAAgAB//8A/QH/AAD/AAEAAQAAAv8BAP7//gAAAQEAAAD/AAAAAAAAAAEAAAAAAAAAAAABAAAAAAAA/QD9/wAAAQEBAQAAAP///gL/AQAB/wD/APwAAAD+AP4BAAD/AAAAAAAA/wAAAAAAAQABAAABAAAAAgEAAAAAAf8A/v4A/QACAP//AAD+AAABAAD+AAAAAAABAQAA//8AAQEDAQAAAAAA/AAA/wIB/wAA/QAAAQD/AAD/AAAAAP8BAgL/AAAA+wAB/wAAAQEBAAABAAD/AP8AAQEAAAEA/wEAAv4A/wABAAAAAAACAP8BAf8AAAEAAAAAAQAAAAABAAEB/QABAAH/Af8AAAAB/QEAAP8A/wD+AP0AAAD/AP/9AP0AAP0ABAIAAQAA+/7/AQAA/wH+AP8A/wAAAP0B/gIAA/8AA//+/wAAAAADAP0AAf0AAQIA/wD//wAA/wAB/gEAAgAAAAAA/gEAAP///QD/AAEAAAD/AAD/AQD+Af/+AAD9AP4BAwD/AAAA/gL+Af8EAAD/AAAAAQEBAAMBAv4AAP4A/AEFAf//AAD//v4B/gMAAP0A/wD9AgAA/wAA/QD//wD9AQAB/gEAAPr9/wH/AQAB/QAAAP8AAAACAAECAAABAAL/Af8EAAAAAQQD/wH9AQAFAQEAAv4AAAIAAf8AAAAA/wEBAQAAAAAA//8B/wH/Af8AAAD/AAAA/wD/AAABAgABAAD//wAAAAECAf8A/wAAAQAAAAEAAAAAAAH/AAAAAAAB/gAAAAAAAP8AAf8BAAEAAQAA//8AAAABAQEA/wEAAgD9AAEAAAABAAEA/wABAAAAAQABAAEAAQAB/wD/AAD/AAAAAAAAAPsC/gH+/wAB/gD/AAAAAAABAAECAP//AAD/AP8AAAAAAf8AAAL/AAABAAEABAIAAAIBAgP/AgAAA/8A/gP+AQD/BQL+Af0C/gUAAQAA/wAABAEAAAD9/QD/AQEA/v8AAP79AAEAAgAA/wAAAAD/AQD9/wD/APn9Af4A/v8AAAAA+gD9AP4DAAH+Af8AAQEAAP8AAAIA/wAAAQD/AAH/AAD/AQH/AQEAAAAAAQACAQAAAgAD/QEA/gD8AQIC/QEAAf/9/gIAA/8A/QEAAQEDAP/8AAICAAEAAAD/AAEB/wAAAAEC/wD8AAACAP8AAf8AAAIAAAH///8AAv/9AAH/AAD7AQL/Af4AAAMAAQD/AAAAAwABAAH6/gAA//8AAwAAAPsC////AQAAAAAAAAABAQABAAH/Af8BAAABAwH/AQAA/v4AAQAAAQH+Af8AAQIAAP0A+/8A/wAA/wEBAAAAAAAA/wAA/wH/Af8AAAD/AAAA/wD/AAABAQACAf8AAQAAAAEAAAAAAAAAAAAAAP8AAAAAAAADAAIAAAMA/wH/AAAAAgH/AAH+AAADAP8AAQAAAP8A/gIAAgIAAAAA/wEAAAABAf8AAP8B/wAAAAACAQAAAAAC/gAAAQD/AP7//wH//wD//wD/AAEA/gAAAAH+AAAAAQH+AAD/AP8BAAD+AAAAAQD/Af8BAAEAAP8A//8AAP8AAAD/BAMAAf8BAAMAAAABAQD/Af4BAAL/AQABAQAAAgED/wD+/wAAAf8B//8AAAH+AAAAAgAA/QD/AP0BAAD/AP/9AP8CAAAA/gAAAAAAAAACAQAAAP8AAAAA/wAAAAEAAf4AAAEAAAEAAAEAAAAAAQABAAD/AP8A/wD/AQAA/wD+AgAA/gACAP4AAv8AAAABAAAB/wD/AgAAAAEBAAAB//8BAQIAAAAA/wEAAAAAAf8AAQD/AAABAAEAAP8BAAH/AgAAAP8AAQD/AP8AAAD9AAAB//8BAQAAAAD//wAA/wAAAgACAQADAP3/AQAAAAACAf8CAAH/AwEBAAAAAAAEAQH+AP3/AP8AAwAA/wABAAAB//8BAAH/AP8AAQAAAAH//wEAAAAAAQD/AAAAAAAAAQAAAP8AAAEAAAABAAAAAQAA/wAAAAAAAAEAAAAAAAD//wAAAQAA/wAAAAAAAAAAAAAAAAH/AP8AAAAAAAAAAQAAAAAAAAD/AAAA//8AAAEAAf8AAgAA/gD+AP4AAAAC//8AAAAAAP8AAAD/AAAAAAAAAf8CAP8BAP8BAf8AAQL//wAAAP8AAQEBAAABAQAB//4BAP8AAAEA/wAAAAAAAQAA/wAC/wD/AAAAAQAAAAEAAP0A/AEAAAECAAAAAAD7/gEAAAIAAf4AAAD+AAAAAAABAAD/AgABAP8A/wEAAP/+AgEAAAAAAQABAAEAAAABAAEAAP8CAAAA//8AAAAA/gAD/wH+AQABAAAAAQAAAAAAAAAA/f8A/gABAAEAAAD+/wEAAAIAAf8AAQD/AQAAAP8AAAD6AQABAP8A/wAAAP3/Af8BAgAAAgAAAAAAAAD//wD+APz9AP4BAAD/AAAA+/z/AQAFAAD8AAAA/AEAAP3/AAAA/QIAAAH9AAL/AAD//wAAAP8AAQAAAAD9AQAA+wD//gD//gAAAAIBAv8AAP8AAQABAQAAAAAAAP4AAAD+///+APz/AP4BAwACAQAA+wABAv4GAAH+AP8AAP8AAP8A/gEBAAEAAQAAAAAAAAABAAIAAP8AAAEAAAABAAAAAAAA/gAAAAAAAAAAAAAAAAAA/wD/AAAA/wAAAAAAAAAAAP//AAL+AP8AAAEAAAAAAQH/AQAAAAH/AAAAAAAAAAAAAAEAA/8A/gD/AP//AAACAf0AAAEAAP4BAAD/AAAA//8AAQAB/wAAAP8AAP8AAAEBAAAA/wAAAAEAAAMAAAAB/wACAP//AAAA/wEB/wAAAP8C/wEB/wD/AQAAAf8AAAIAAP8A+wIA/gABAP8BAAD6/wAC/wEAAP4AAAD+/wAAAAAA/wD+AgAAAAAAAAAAAP4AAAAAAAAAAQAAAP8BAAAAAAEB/wEBAAIAAAAB/wAAAAAA/wH/AP8DAAEAAQEAAAD///8A/f8A/gAAAf//AAD+//4AAAIAAP8AAQD/AQAA/wAB/wD7AgABAf4AAAAAAPsA//8BAgAAAgAAAAEBAAD+/wD8APr+AP8AAQD9AAAA+v3/AQAFAAD+AAAA+wEAAP0AAAAA/AIAAQL/AAAAAAD+/gEAAAEAAAAAAAD+AAAAAAD//wEAAAABAAL/AQAAAP//AAD/AgEB/gAAAP4A/wD/AQH9APsAAAIBAgD/AAAA+/3+AAED/wABAP8A/wAAAP4AAP8AAAAA/wEAAQAAAAD///8B/wH/AP8AAAD/AQAA/wD/AQABAAABAP8AAAAAAAEA/gAA/wAAAAAAAAAAAAABAAADAAEBAAIA/QAAAAAAAgEBAAH+AAADAP8AAgAAAAAA/wAAAQEAAAAAAAP/AAABAQL/Af8AAAAAAQACAQAAAgADAP//AgD/AAAAAP8AAP8AAP8BAAAA/wD/AP//AAABAAD/AP8AAP4BAQH/AAAAAf//AAABAAD+AP8AAQAAAAEA/wD/AgIAAP/+/wEAAAABAgIAAP4B/wIAAgABAAAAAgIB/QD8AAD+AAEA/gAAAP3//wD/AgAA/QAAAAAAAAD/AP/9AAIDAAEAAP8BAQAA/wECAQABAQAB/wAA/f4AAAEBAgAA/P8A/gABAf8AAAD//wABAAD/AP8AAAAAAQAA/wEAAgADAAACAP4AAQAAAAEBAP8B/QAAAgAAAAAAAAABAQD/AAEBAAEBAP8AAAD/AgECAAD+AAECAAAA/wAAAAABAAEAAf8AAAD/AAD/AAD/Af8A/wAAAf8A/wABAAAAAAAAAv///wABAAH/AgAAAP8CAP8B/wH/AwEBAAAAAAAB/wD+Af4CAAEBAgD/AAAA/gABAQAC/wH/AQAA/wAAAAEAAAAAAAAA/wEAAf8AAAAA/v8B/wH/Af4A/wAAAAAA/wD/AAAAAQABAP8AAAAAAAEBAf8B/wAAAQABAP8BAAAAAAD+AAABAAAAAv8BAAAA/wEAAP8BAAH/AAAA//8AAAABAgAAAQEAAQAAAAH/AAABAAL/AP8AAAIAAAAB/wAAAAAC/wD/AQAAAAIAAQAAAPwBAgD/AAABAQAAAP8AAAABAAH/AAEBAAMAAQEA/wAAAQAC/gD/AP8CAAAAAAEBAAEAAAL/AgEAAv8A/gT+AAADBAH+A/4C/wMAAQD//wAAAwH/AAD7/gD/AQIAAf8AAP/8AgAAAv8B/wABAP8BAAH/AP79Afj+AP8BBgD/Af8A+v8AAf8FAAL/AP8A/wEAAP8AAAIA/wAAAf//AAIAAAAB/wAAAQAAAAEAAQAAAAAAAQAC/wH+/QD9AQIB/QIAAP/+/gIABP8A/gEBAQEBAP/8AAIBAAQAAAD//wAB/wAAAQIBAAD9AAABAAAAAgAAAAMAAAEA/f8AAgD9AAL/AAD/Av7/Af0AAAIA/wD//wAABAACAQD6/wAAAAEBAQEAAPwA/wABAgABAAAAAAD/AQACAP//AAIBAP4BAAAAAQAA//79Av8AAAEAAAAAAAIAAPz+/AAAAQAAAQH/AAAAAAABAAAAAAAAAAAAAAAAAAAAAAAA/gD+/wD9AAEBAQAAAP7+/gH/AgAA/gD/AP8AAAD/AP8BAP8BAAAA/wD/AAAA/wEA/wD/AAABAAAAAgAAAAAAAAABAf0AAAAAAAAAAAD//wEA/wL/AAAAAAD/AQAA/wAA/wABAAAAAP8B+wAAAAH//wEA/gAAAAAAAP8AAAAAAP3/AAEBAAAAAgECAAAAAAAC/wIB/wABAAAAAAAAAAEA/gEAAf8AAAABAP0BAAAB/wAAAf4A/wIAAAACAQAAAQD//wAC/QD/AAH//wAAAP8AAQH+AgEB/wAAAQD9AAAAAf/+AP8AAAABAwH8AQD//v7/Av8BAAH8AP8AAP4AAP0B/gIAAv8AAQD/AAAAAAAB////AP///wAAAAAA/wAAAgEB/wAAAAD/AAQA/wAAAAIBAAD/AAAAAAAAAAABAAD/AQD+Afz+APwAAAD+/wAA/wAB/wAA//8AAAAAAv4AAAL/AP4A//4A/QACAf4AAAD++wABAAD/Af4AAQD/AgAA/wAA/AL9AQD5AAgC/gMAAPz+/AL+BAAB+gAAAf//AQAC/wIC/wAAAAUA+gAF/wAAAgQD/wEAAQAGAAEAAQEAAAIB/wEAAQAAAQD/AAAAAAABAAEAAAAAAAEAAAABAQAAAAAA/gH+/gD9AAABAAEAAP0A/wL+AQAB/gD/AAD/AAAAAAD9AP4CAP8BBAD+AQAA/v3/Af8C/wH9AAEA//4AAAEBAf///v4A/gACAv8AAAD/AP8AAAH/AP8AAAAAAQEB/wD//gD//wACAP8AAgAAAAABAQAAAAABAgAAAP8AAAD/AAAE/wUBAAP//AAD/wAABQH//gH8AAACAAEAAAAAAAAAAf8A/gIAAAEEAAIB/wAA/wAA/wEAAQAB/wD7AAAAAAEBAQABAAAAAP4AAv8AAAEA//8A/wAAAAEBAP8BAAAAAAD/AP8AAAEAAQAAAAAA/QL/Af8CAAH/AAAAAAAAAAABAP8BAAAA/wABAP8BAAD//v4BAAEAAP4AAQD/AAAAAAAAAQADAgADAAAA/v8AAAMAAwAB/wD/AgABAAEBAAABAQL9Af3+APwBBwH6AgAA/Pr6A/4C/wH7AP4A/wEAAPz//gAAAAEA/wAA/wH//wACAQH/AQEAAAMAAAD+//8AAgADAQH6AgD+AAIA/wEAAP7//wABAQABAAEAAP//AAAAAP0AAAD+AAEA/P8B/wAAAQABAQECAP8BAQAAAwEAAAH//wAAAAAAAAD/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//wD+AAD9AQABAAAAAP7+/wAAAQAA/gD/AAEBAAD//wD/AAH/AP4AAQH+AAAAAP//Av8B/wD+AP8AAP8BAP4A/wIAAgAAAgAAAQH/AAABAgL/AP8AAAIAAAAAAAEAAP8A/gABAQACAQP//wAA/wAB/wD//wAAAAAAAAD/AAD/AQIAAAICAAP//f8C/wAAAgMC/gL+AQAFAAEAAgAAAAMAA/0A/wAA/gEFAQABAAD+//wBAAIAAf0B/wD7Af8A/gEA/QED/gD/AQj/AAAAAP3/+QL/Af8B/QD/APsBAAABAQL/AP//AAAA/AH9AQAA//4AAP8AAAAAAAAAAQAAAP8AAAIAAQAAAgD6AQEAAAAAAAD/Af0AAAEAAQACAQAAA///AgEBAAD+Af///wIAAP/+AAIA/wAB/QAAAP7+AAADAQEAAAAAAAAA/gAAAQAAAgEBAf/9AAAB/wAAAQAAAP4BAP0B/P4A+gAHAf4BAAD//fwB/wUAAfwA/gABAgAA/P79/QD/AgD/AAIBAgEAAP8AAAAAAAEBAAAAAAAAAAAAAP8AAP4EAAECBgAD/wAAAf0C//4C/wIBAAAAAP8AAPwA/AAAAAAAAAEAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAA/wD//wD/AQABAAEAAP8A/wH/Af8A/gD/AAAAAAABAP//AP//AP4AAgAB/wAAAP/+AQACAAH/AAAAAAEAAAAAAQAAAgAAAAD/AAAAAAACAAEA//8AAP8AAAD+AAAA/wH//wAAAAAAAQIAAAEAAP7/AAD/AwAA/gAAAP4AAAABAf/+AAMDAP8A/wD9AQAAA/8AAQH/AP8AAAAAAQAAAAAAAf0AAv4A/wEA//4B/wEB/gEB//8BAQEA/wAB/wEA/gAB/wADAQACAP8BAAEAAAIBAf8C/gD//wEBAAMBAAACAAIAAP8CAAIAAf//AP8AAf/7AAAAAQH/AAAA//8AAAEAAAAA/f8AAQD/AP4AAAAA/f0BAf//Af4AAAABAQAA/f/9AQD+AgAAAf4A/gEBAAAC/gAA/wAA/wAAAAEBAAAAAP4BAAIBAP//AgH+AAAA////AAL9Af//AQAAAAMAAAEAAAAA+v8A/wEDAPz/AAD7/v4A/wMAAfoA/wABAQAA/P///AD9AQABAAL/Af4AAP7/Av8ABv4AAwABAP3/AAD/AAEBAP4AAAEABP8B/wAA/QH9AAAAAAEBAAAAAP8AAP0AAAAAAAAA/wEAAf8AAAD//wAB/wAAAP8AAAAAAAAA/wAAAgAAAQAD//8AAQAAAAAAAP8CAAD/AgAAAAAAAAABAAH+AAAAAP8BA/8BAAAA/wABAf4CAAEAAP8A//8AAAAA//4AAgEAAAAAAf8AAAAA/wAAAAP/AP8AAAD9/wAA/wD+AQACAAABAP4AAAAAAAEAAf8B//8AAAAAAAABAAAAAP//AAEAAP8AAAD+AAAAAAAAAAH/AAABAAAA/gIAAP4AAgIAAf8A/wD/AAABAAAAAAABAP8AAQAAAAACAgAAAAAB/AAAAgD/AP8B/wAAAAAA/QEB/wAA/wD/AAEBAAABAAD/Af4AAAEBAgECAP8AAAL///0AAgL9AP0AAf8BAAIBAQAA/wEA/wECAAH/AAACAP8AAf4AAAEAAQD+AQAAAAAAAAD/AQACAQAB/QIAAAP+/gEA/wAAAAAAAAUBAAD/AP4CAAIAAP8A/wACAAAAAgD/AAD+AAABAAEAAQMAAQEA//8AAP8A////AAD/AAD+Av8AAAAAAP8A/wD+AAAAAQD//gD7AAD9AAAAAAAAAP4A/gAABAAAAAAAAAEAAAD/AAAAAP//AP8BAP8BAAAAAAH9AP8BAAAAAAAAAAAAAAAB/wAAAAEAAAEAAAAAAAAA/wABAAAAAAAAAAAAAAAA/wAA/wD//wD/AAAAAAAAAP///gAAAQAA/wAAAAAAAAAAAAEBAAEAAP8AAf8BAAAA/wEAAAAB/wAAAAAA//8AAAAAAP0BAP8A/gECAP4BAAD//gAB/wD/Af8A/wD//wAAAAD+/wAB/wAAAAAAAQAAAAACAQAAAAAAAAAAAAD/AAABAAAAAQD/AAAAAAAAAAAAAAD/AAAAAAAAAAAAAQABAAEBAQL/AgEAAf/9AAD/AAAAAgD/Av4A/wIAAQADAQAAAQACAAD+/QACAP8AAf8AAP4A/wL+/wABAQAAAAECAAEAAAD+AP4AAP8A/gAAAAAAAP//AQAAAAAAAP8A/wAAAAIAAAEAAAEAAf//AAH/AAABAQH/Af4A/wEAAAABAQAAAgABAQD+AAAAAP4AAf8AAAH//wABAAAAAAAAAAACAAAB/wABAPwAAAH/AP4D/wAA/wL//wEAAAECAAEAAQH/AP4AAQAAAAEAAAAAAAH/AAABAQH/AAD/AAEA/wACAAAAAQAAAQAB/wD9AQMB/QAAAP/+AAAAAv8A/wD/AAEAAAD/AP7/AP8CAAEAAQEB/wAAAP/+//8CAAH/AAAA/gEAAP0B/gAAAAAA/wEAAP8AAAAB//8AAAAAAP8AAAAAAQAAAAAAAgAAAAABAAAAAgAAAAABAQAAAAAAAwAAAP8AAAAAAQD/AAACAAP//QAB/wAAAAIB/wH+AQABAAEAAf8AAAEAAAAA/wEAAwD/AAEAAAAAAgIAAP0AAAIAAQACAAAAAAEC/wAB/wD+AAAAAAAAAAD//wEAAwAA/QD/AP8BAAAAAP//AP7+AP8BAf//AAAA/v7/AQAC/wH/AAAAAAEAAP8AAf8BAP4A/wAB/wEAAAD/AP4B/wD/Af4A/wD+AAAAAAAA/AD8/gD+AQUA/QAAAPz9/AH+A/8B/AD/AAD/AAD+Af78Af3/AP0A/wIAAQAA/QD/AP4CAQEBAAAAAf4BAAL/Af8AAQEAAgABAAAAAAABAQD/AP8AAAEAAAAB/wAAAAACAAD+AAAAAAEAAv8AAP8BAP8AAAAAAQAAAAAAAAD+AAEBAAEAAP//AgABAAAAAQABAAH/AP8BAAAAAQEAAAD//v8AAAEA/wECAQD/AAABAP8AAP8AAAAA/wD+AAAAAQD/AQEA/wAAAf8C/gAAAP//AgAAAP8A/wD+AAH/AAD+/wD/AP8BAAABAwD/AQAAAP7+AQACAAH+AP4AAP8AAP8AAAAAAAAAAAEAAAAAAAAB//8AAAAAAQAAAAAAAAAA/wD//gH+AQD/AAAA/wAAAPz//wL/AwAA/gAAAAAAAAAAAAAAAAD/AAEA/wAC/wAA/gIC/wABAQH/AAAAAf8AAAAAAAEAAAAA/wAAAAIAAAAAAf8AAAEAAAEAAAD+AAAA/wD/AQAAAAAAAP8AAAAAAP8A/QAAAgAAAAAAAP4BAQD+AP7+AP//AP4AAgEAAAAAAAD/AAEBAAD/AAAAAQIAAP//Af8A/gEAAQD///8BAAAA/wABAf7/AQH//wEBAv///wEE/wAAAAAAAAACAv8AAAEDAf8A/wD/AgD/AAD+AQAAAAH+AAACAAD//wL/AAAAAv3//wEA////AAEA/wABAAEA/wAAAAMAAQAB//4AAAACAQIAAP4A/wAAAAAB//8AAQICAQEA/QAAAAIC/v8BAAAA/wACAP//AAD+AAEAAAAA/QH8APwAAAEA/QACAAAAAAH9AAH/AAACAAAAAgEAAAD/+wIAAwEAAQABAAL/AAD/AQMAAP4AAAEAAQAAAAAAAQACAAAB/gAAAf4AAP//AAIAA/8AAwAA/wAAAAIAAAAAAAD/Af8AAP8A/QL+AP8A/v0CAP8C/wD9AP8AAP4BAP7//wAAAAEAAAD/AP8AAAAAAAAAAAD/AAAAAAABAQAA/wD//AD//QD9AAEBAf8AAP3+/gL/BAAB/QD/AP0AAAAAAAABAAIAAAEAAQABAAAAAQMB/wH/AAABAAAA/wAAAAEAAQEAAgAAAgD//wEAAAACAQAAAf8A/wEAAQAC/wAAAgAA/wABAAACAAH/AQAAAAMBAf8B/wH/AAAAAP4AAAAA/wL9AP4AAAABBQABAP8AAAD/AAACAQH+Af8A/gAAAAAAAv8A/wAA/wABAf4BAAD/AP8A/wEAAf8A/wD/AAAA/gAC/wD+AQAAAQEA/wH///8AAQD//wAAAAAAAAEAAAAAAAD/AAEA//8AAwH//wAAAv8AAAEA/wD/AAH//QEBAQH//wH//AAA/wAC/v8CAP79//8BAP4CAAEAAQAA/v8AAQEDBf8EAQAD//z/BP8BAAEDAv8C/gH+AwADAAIBAAACAAIDAAD+AP8AAgEBAf8A/v8BAQAC/wD8Af8A//4AAP8A/v0A/f8AAQH7/wAA//8AAP0A/wAAAAAA/wABAAAAAAD7AQD+AAD/AAH/AgD/Af7/AQAAA/8BAP//AAAB/wD/AQD+AAL/AP8AAwL+AP8B//sAAAAA/wD8//4A//sBAAD9/QAA/wAAAAEAAAAAAAAA//8AAAEAAAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAQD/AP8AAAAAAP4AAAAAAP4AAAEAAAEAAf8AAAAAAQAAAP8AAQH/AP8A//8AAAEBAf0AAQEAAf/+AQAAAAABAQH/Af0A//8AAQAC/wEAAf8B/wAAAQAAAAIA/gEAAP4BAAAA/wEBAgD/AP8AAAD/AQEBAAEBAAAAAAAAAAAAAgAB/wD+AAACAAAAAQAAAP4A//8AAP4A/wAAAP4BAAABAQT/AAAB/wABAAD+AAAA/wAAAgAC/gD///wBAf8AAf8AA/8BAgAA/wEBAP7/AAED/wD/AAEBAP8AAAAAAAAAAf8BAf8AAQD+/wAAAP4AAP8AAgAA//0A/gADAgEA/wAC/gH/AAMAAP4B/wAAAAEA/v/8Av8B/AACAAL+BAD//wQDA//+/QIBAwEAAAEC/gEDAwH9APv7APkAAwP8//8B+/0AAP8E/QL7AP7//wQBAfwA+QEA/gIBAv/+/v4BAAAAAQEAAf0B/wH/AQD///8AAQH/AQD/AQAB//0AAf8BAf4A/gEAAP8AAAABAAICAAABAgP9Afj8APkBBAIC//8A+P/9AP0H/gL7AQAAAAUCAfwA+wAAAAAAAAEAAAAAAAAA/wAAAAAAAAAAAAAAAAAA/wAAAAD+AQABAQIA/wAAAP/+AAAAAQAB/wD/AAAAAAAAAP//AP//AP4A//8AAQAA/gH/Af8AAAD+AAAA/wAAAAABAf4AAgAA/v/9AP8AAAD/AgABAAEAAAAAAAABAAAA/wAB/wEBAgAAAAIAAQAAAP8DAQH/AAEBAAD/AAEBAAABAAL/AAYBAAD/AP8B/wAAAAIC/wD////+AAAA/gAAAAIBAP//A/8AAQABAAL+AAAAAQL/AQIAAAIBAQD8AQEBAf/9/v8C/AD///wABP4AAAMA/v8DAv///wECAAADAQAE/gAAAAEAAP0BAAAB/wABAwH+AQEA/wEAAP8AAAABAAD/Av0B/gEA/gECAP4A/wD/AAAB/wAAAQAA/wD//wAA/gEA/gH//wECAQEA/gH//wAA/wH//wEBAAD+AQH+/wD9AgABAAH/AAMAAP8CAAAB//8AAAH/AAAAAAAAAf8AAAAA/wAA/QIAAQD8AAEB//8A/wAA/wH/AP4AAAAE/wAA/gP9//8D/wAB///+AQAB/wIA/wAB/QL/Av8BAP8B/v8C/wMCAP75Af7/BQEB/wEB/Qb9//8D/wP8AfwA/gUBAf7/AwAAAAAAAAH/AAAAAAAAAAAAAAAAAAAAAAD/AAAAAAAAAQACAgAA//4AAf8BAAIAAP8A/gAAAQABAAUBAAAB/wH5APr+APoCCAH/Af8A+/v+BPwG/wP5AP0A+f8AAP0B/vwB//wA/AEDAAAC/wD+/f8D/gL/AfwA/wD8/wAA/gD9/wEBAAECAQH//gL//wEBAgL//gEBAgD+AQL+////AgEAAPz+AP4AAQEC/wAA/v4AAAAC/gD+AAAAAgIBAP///v0A/QEA/wECAQAAAP//AgEA/wABAQD//wD//gEAAAIBAAEBAAABAAAA/wIAAAABAQH//wEBAgAAAAD+AAAAAAP/AAP/APz/AgEB/wEAAQD/AAAD/gH9AAAA/QIBAAEAAP4AAAAA/wADAf4AAAAAAf4AAAEAAf8AAAAAAQAA//8B/wACAQABAP///gEAAP8BAgIA/wEAAf8AAAP/AAD+AQAA/wH///0A/wEB/gEBAv3/AAEB///+AAH//wIAAAH+//8AAAAA/wABAQAAAAAAAAAAAAAAAAAAAAD/AQEA///+AQABAAACAAD+/wAAAP8BAgEB/QH/AgAAAAL+/wABAAEAAP8AAAEAAP//AQAA//8BAAD/AP8BAP8AAf7//wABAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAC///+AP4BAv8BAQL+/v8BAf///gABAP4BAQEC//4AAAAEAAEAAQD/AAAAAAEBAAAAAAABAAAAAAAAAP0BAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMA/wD/AAEC/wEBAP79/gAAAAD+AAH/AAMAAQAAAP8A/wAAAQAA/wAAAP8AAQEBAAAAAAEAAAAAAAAA/wAC/gD+AgIDAQD9+wP//gAEBQD//AIABv79AQD9/v7+AAQD/wQA/wD/AAEB/wEAA//+/wIA//z+AQD//gEAAAMA/gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/wAA//4AAAEA/wEBAP///wEB///+AQH/AAEBAQH//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv8A/v////0CA/4BAQH9/f4CAf7//wEC/wADAQED/v8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABf4G+/75/voEBfwCAwb6+/wDBf3++gID/voGAgIG/PoCAP/+APsA/P4A/wABA/79AQEB/wD/AQAA/QABAAT9/gT/AgD+/wAAAgAAAAQC+wD9AAAAAPsAAP4AAAIAAAD7AQACAP8AAAAAAv8CAgIAAP8AAAAAAAABAAACAAACAAD/AQAAAQICAP7+Av///wICAQAAAAADAP8BAf8A/wAAAv0A/QMBBAEC/v0C///9AgIC/v4BAQL//wED/v///gICAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAv7//v/+AQL/AQEC/v7/AQL///4AAf/+AgEBAv/+Av8BAP/8/wABBP4A/wH+/P0DAwAA/gAD/wAEAAEEAP/+AP8AAQMBAP7+Av7/AAMEAQAAAP8BAP8BBP8C/wAAAwEABAD///8AAgP/AQEA/f3+AAL/AP0AAv8AAwEB/wD/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/Av3+/v/+AgT+AQIC//z+AwL+//0BAv/9AwEBAv7+BP8G/v/8/voCBvwCAgb9APwEBP/+/QEE//4FAQEG+/3//gD+//7//gIC/gEBAAH//wIC///+AQL//gIBAQD+/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/gH/AAEBAAAA/wAAAAABAQH//wAAAQD/AQD+AAD+AAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAL/A////v/+AQT9AQIC//z+AgH/AP4AAv/+AgEAAwD+Av8B/f/9//0CA/4BAQL/AP4CA////QEC//0DAQID/v4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT+Bf3+/P/7AwX7AgIE/Pv9AwT+/vsBA/4ABAICAP78Av8C/f/8//wCAP4BAgL//f4BA////QEB//wDAQED/wAAAP8AAP8AAAEA/wAAAP8AAAAB/wD+AAAA/gEAAQAA/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX+Bvv++v77BAb8AgIF+/v8AwX9/voBA/77BQICBvz6BP8E/P/8//sEBP0BAgT8/f0CA/7//QEC/vwEAgIE/PwC/wMB//7//wECAAEBA/39/gIA///+AQH/AwMBAQL+/AP/BP7//f77AgP+AgEF/P3+AwL//v0BA//8BQEBBP79AP8D/QD+AP4BAv8AAAP+/gAAAf8A/gAA//4CAAAB/v//AAMAAAAA/wAC/wAAA/0DAAD+AP/9AAEAAAAAAP7+//sA/gMBAgEC//wB///9/wIC/v4BAAD//gEF/////gAAA/8E///+//0CA/8BAgL8AP8CAv///gAB//0CAAAD//z+AP8DAQQBAP8AAf8AAf8CAQD9AQAB/wABAv7//wABAgIAA/3//f//AQL+AQEC/QD+AgH+//0BAv/9AgEBAP79AAAAAgADAAAAAAAAAAH/AQAB/gH/Af8AAAEAAAABAf4D/gQA",
            "/v///AIF/wEBA/z//wIB////AAP+/wIAAAQA+/4A/wAAAwD//wACAAAA/gEAAP4BAAL/AAAC/f//AP//Av8CAP//AAABA/4BAQL+/v8CAf///wAB//0CAQEDAP4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/A//////+AgP9AQEC/f7+AQH///4BAv/9AwEBA/78/QH+/wABAAUB/AEAAP8AAwD9AQAB/gEAAAX/AAD+Af8E/wQA/wD+/AEE/QECBAAA/QIA//8AAgL//wQBAQP+/wAAAP8AAAABAfz/AP/7/wEBAAL/APwAAAD9/gAA/wH/BQAG/P76//wDB/wCAgT7+f0DAv7+/AEC//kEAQIG/fkCAQP8AgMA/wEBAf///QQCAQAC//8CAP4BA/4A/wIAA/4A/gABAAEA//4A///+AgAA/wAAAQAA/wEBAAIA/gAAAgAA/QD9AAABAwABAQAA/QAAAv4A/gEA/wADAQEAAP/+Af8DAQIAAP4AAf//AAIDAQAAAQAEAAABAwD//wAAAwP/AAD+/v8AAQP/AwEDAP7+AgEAAP8AAP/9AgABAwAAAP8AAv/9/wACAf8AAAT9//8CAP/+AQAC//0BAQAD/wAB/wED//3/AAIB/wABA/4A/wL/AP0C/wIA/QAAAAT//wL/AwMAAv8A/wT/AQEEAP7/A/4A/QEAAAD8AAAABf//Av8DAP///wABA/4BAQD+//4CAf79/wEC/wABAAED//0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/AP/8AP0AAgD+AQAAAAIB/v//AAH+AQAA/AAAAv//+/4AAP8AAQD+AP8AAP8C//wAAAH/AP4AAAD/AQIA/v//AAD+AwAEAAAAAAAAAPz/AAAA/gAABAAAAP8AAAD/Af8C/wL//wH/+wIA/wEBBPv4AAQC///9AAH//gECAQD+/P8AAP0A/QAAAP8AAP8BAP///wH/Af0BAAAAAQABAP/9AQAE/wABAPv/AgAAAAMAAP8CAAD+AQABAAAAAAAA/gEB/wIB/wD//AAC/wEB/gD//wEBAP//AQH//wEAAP8A/wL/AQD//wABAgH+/fwA/v3+AAEAAP8A/f/+AgEBAv8E/gAA/wD/AAEB//8BAP//+gD+Af///gAAAP4CAgD+/v8BAAL/AAAA/wEBAAABAf7+/wEBAP8AAAEA/gAAAAIAAAP/A/7+/f/+AgT+AQED/QD+AwH//v0BAv/9AwEBA//+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/A/7//v/+AgP+AQEC/f7+AgH///4BAv/9AwEBA/79/AH6BAH/AQb9/AP+/vsF/AP++wIC+//+Agf8///8AwT/AAH/AAAA/gEAAQAAAvsBAAAE/wD9AAAAAP8AAAAB/AH///v//v//AAL/AQEAAPz/Af8AAP4AAQD/AQABAAAAAQEA/AAA//8BAgAAAAL7/v8BA/8A/QEBAP8BAQABAPsD/wb+//7++QED/wEBBf///wIEAP/+AgH//QEBAAX//gL/Av3////+AQP/AQED/v/+AgP//wABAf8AAgEBA//9AwAA/P/9APsBAP4CAQD9/v4AA/8D/QH//wIDAQEA/fwC/wH+//8A/wID/gEBAv7+/gAB////AQL//wIBAQD+//0A/wACAgAC/QAD//4ABAMBAPwCAAUAAAID//7+/AIFAgADAAAAAP8AA/8BAgD//v8CAAAA/gAAAP0BAAADAAD/AQAAAQIBAP4AAf//AAABAAD+AgIC//4BAf7//wEAAP8A/wAAAAAA//8B//8AAP8BAAAAAAAAAAAA/wAA/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+Af0AAgQBAP7+Af/+AAQEAv78AgED//4BBP3+//4AAwP/A/3/AP/9AgP+AQEDAP7+AgD//wABAv/+AwEBA/7+Bf4G+/77/vwEBv0CAwb7+/sEBP3/+wED/voGAgIF/foCAAT7//3//AIB/wEBA/wB/gIF/v/8AQH//wIBAQL9/AQAA/7+/gD9AgT8AQED/f39AgH///wBA//+BQECA/z9Bf4F/v78/v0DBv0BAQP7/PwEAv7+/wED//oFAgEG/f4AAAEBAQIA/f8BAf8AAv0CAAICAP//AP8AAv8AAAIBAgIABP7//P/+AwP+AQID/f3+AQP///0BAv8AAwEBAP7+/gH/AwIEAQP9/gT+/v4EBAP+/AEBBv/+AQP8/v//AwMCAAMA//3/AQL//wECAf4A/gIA///+AAL/AAEBAf///gIABAD//v8BAgD/AQIA/P7+/wL/AP0AAf8BAgEB/wD8AP8CAP/8/wMBA/8BAAD9/P4CA/8B/gAC/wQDAAAEAP8CAf/+AP8BAgAA/wAA/wD/AP4B/wEBAAAAAv8BAP7+/wAAAAMAAwAB/gT+AQAAAwH+AP0BAAMAAgADAf8AAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/Av3+/v/+AgT+AgIC/v3+AwL+//0BAv/9AwEBAv7+Bf4F+/76//oECPwCAgf8AP0DBP3+/AED/voEAgIF/foCAAH///sAAgP+/wEBAvv9AAAD///9AAH//gABAQT++wL/BAAAAP/9AgMBAQEEAP0BAAIAAAMAAv8B/wIABfz+Av8D///8//0CBP0BAQT++/0DAv7+/gEC//0EAgEF/v4C/wP/AAL/AQED/gAAAwAA/wECAP8CAQAA/wAAAAH//v8A/gAA/gAAAQD/AP////4B/wEAAP4A/wD+/wAAAP//Av4CAP77/vkDBPwBAgL8/P0CAf7+/gAD//0EAgEB/P4C/wL///z//gMA/QEBAP7//gEC//8BAQL//QQBAQP+/gH/AgD///8CAQL/AQED/v7/Av4A//4AAv/+AQAB///+A/4C/v/9//4DBQABAgP9//4DAf///wAD//0DAQEC/P3/AAAAAPwAAAEBAAAAAP4B/wAC/wACAAAAAAMAAAEA/wL/Av8A/v/+Av7+AQEA/P3/AAIA//8AAQD9AQEAAP79Af8B/v/8//0CA/4BAQL+Af4BA////wEB//4EAQEC/v4A/wMAAAP/AP8DAgD/AwACAAL+AP8C/wEAAAD//wIAAgb+Bvv++v76BQb7AgIG+vr8BAX9/voCBP76BgICBvz6A/8D+//7//0CA/0BAQP9+/4CBP///QEC//sDAQED/f4AAAACAAAA/gEBAAAAAf4BAAAAAAD/AAAAAgIBAP///AP+BPz9+//+BAb9AgIE+vz9AwT+/vsBA/78BgICBf37/gAAAwADAAL/AQAAAAACAAAA/QEAAwAAAP8BAAAAAAIF/gX8/vr++wMG/AICBvz8/AME/v78AgP++wUCAgT8+wX+Bvv9+/75BAb8AgIG+/38BAT9/fsCA/7/BgMCA/77Av8B/wAAAAECAf4BAAABAf4BAf/+/wAB//0CAQEC//4D/wX//vz//AMF/QICA/3+/QMD/v78AQL//gUBAQMA/gAAAAAA/wAAAQEAAAAB/wEAAAAAAP8AAAAAAQAAAAH/Av8BAP79AAACA/8BAQL+/v4BAf8A/gAA//0EAQECAPwC/wIA/v0AAAED/wEBA//+/gIBAP/+AAH//QMBAQMA/gL/+wABA/8A/wMCAgADBP0BAfwCAAQAAAD+Av8BAgIEAv8CAP7/AAABA/8BAQL//v8CAAD+/wAB//0CAAECAP4CAAIA/v8AAAEC/wEBAv/+/wEAAP7/AAH//gIAAQIA/gT/BP7//f/8AwT9AQIE/fz9AgL+//4BAv/8BAIBBP38BP8A/f/7//wDBP0BAgD+/f0CAf7//QEC//4EAQEE/fwBAAL/AP4AAQL+AAAB//z/AAEA//7+Af8A/wQAAP4BAAP/Bf3//v78BAP9AQED/P3+AgL+//wBAv77AwECA/76/wH+/gAAAAAB/gAAAP3+///+/wAB/QAAAAEBAAAAAAACAAH9////+wEC/gEBAv4A/gEC/wD/AAD/AQIBAQD//v4B/wMCBAAB/v4C/v7/BgQB/v0CAQP//gID+wD+AQQEAwAF/f////8BA/4BAQX/A/4BAgD//gEB//0AAQD///8CAP4B//3//AEB/gEB//7//gL////9AAH/AQMBAQL+/f8AAAEAAwAC/wIAAAAAAwEA/wEBAQIA/gAD/v8AAQEC/QH9AAD/AAMB/AAAAP//AP7+Af8B/QAA/wMAAAD+AAAA/wAAAAIAAP8D/wEABAMC/gD+AQABAAL/A/8AAAT//gL/AP7//v/9AgL+AQEA/wD+AwAAAAAAAf/8AwEBA/79AP8DAAAA/wQA/gIBAAIBAv4B/wEBAAEAAAAB/wAD/gP8AAADAgUBAPz+A/7+AAQDAwD8AgAD//8BAvz+/wAABAf+B/v9+v76BQn6AwMG+fn7BQX8/foCBP35BwMDB/r4AAD5AgD7AQEA/wAAAPkC+QD//gEA+gAAAAD/AAAAAAIC//8A//7//gAB/wEA/AP+/wH8////AAH/BAAAAQIAAwX+Bf/+/f/7Awb9AQIC/Pz+Af7///8BA/79BAEBA/39AAEFBAADAP7//gAA/wECAAAB/AH/AwD/AAH//wABAgEG/gUA/v7++gMI/QICBf7+/AP///7+AQT+/gQCAgb8/gEB/gMAAwD+/wAAAAD9BAX/Av0C/wQAAQADAP8AAgEEBf8D///8//4BB/4BAgH8/v0C/gAA/wAD/v4DAQEE/v4D/gX//wL+/AAF/gEBAf/9/QMBAP7/AAP//AABAQX/+AT/Af7//v/7AwP9AQL/AP39AwL///4BAv/9AgIBBPz//wADAQABAP8AAP8AAAEB/wAAAAAAAQD/AAEAAAAAAf8C/wL//////AEE/gEBAAEA/gECAP8AAQH//QEBAQT+/P0AAwIBAwAB/gAB//8AAwMB/wABAQMA/gED/v//AAL/AP8EAv8B//8AA/8A/wAB//8BAgAAAAEA/wABAAADAf0E/gP//v3/+wIE/gEBA/39/QMB//7+AAP/+wEBAQP/AAf+Bfr++v78BQj7AgME+/r8BAT9/voCBP77BgICB/v6AQAC/gD9//8AAgAAAAP//wACAQD//QAAAP8AAAABAAAG/gYA/v7/+wUE/AICB/3//AMB//78AAP+/gYBAgb++wMBBf8A/QD9AAABAQEEAf4A/wMBAP4AAQAC/gAAAQICAgD//P/8Af8A/QEAAQH/AgD+AQAA/QAAAP4AAAAEAAEAAP7/AAIAAf8AAQAAAQECAf8AAAACAAAAAQAAAP0BAgD/A////v8BAQH/AQECAAH/Av4A/wAAAv/9AgEBBgAABP79/f/6//sBBf0CAf8CAP0EA////AAD//gDAQEE/vwC//0AAPwB/AADAAAAAQMA/wIAAAABAAAA/gIAAAMAAAIAAAD/AP/+AP8AAQD//wL/AgIB/wAAAgAB/wAABP8AAAEAAQEAAQb//QL/AAEBAgEAAAEA/wD+AQT//wD+AAED///+//sAAQUC/wICAv77/wID/v/9AQP/AwQBAQP//wMAAQEAAAABAAEBAQADAf8AAgEA//7/AQAD/wAABAACAgAC/gD8/wABAgABAQP++/8AAQAB/AABAAEAAQD+/f8E/wT//wL/+wEE/gEBAAD//gP/AP8AAAL//gEBAQP+/wT/BP3//v/9AwT9AQIE//z9AgH+//0BAv79BAIBBP39/wAC/gD9AAAA/gAAAAMB/wAA/gAAAAD/AAIAAAABAAAB/wP+APz//QIA/wABA/v9/wIC///9AQAA/wMBAAL9/gEABAD//wD/AQH/AQEDAf//Af///wEBAf8CAgABAQAAAgAHAwAD//0AAQAAAAf/AgAB/AAAAwD/AAL/AAAC//8AAAP9AAIA//4DAf//BAIBAAEBAQABAAEAAwD/AAEBAwX+BQH+Af/8AgX8AgIFAf79AQAAAgAAAv7/BQEBA/8AAAD+AAAB//3/AQAAAAT/AAAC/wD/AAABAAH/AAAD/f8A/wMBAAD/AP8AAAAAAAD+/wL+AgACAQIA/wEAAP//Af8AAwEBAAD+/wABAP8DAQIBAP0BAAIAAAAD////AwAA/QABAQEDAAD+/wH///8BAQEA/gEAAwD/AQH+//8CAAL/AAEAAAAAAQD/Af8AAP3/AAD/AAABAP8BAgAAAAH/AAL/AP3/+//9AwP9AQEA/f/+AQT+/wABAf/9BAEBA/7/Av8A/wAB//sAAv8AAAX///8DA//+/wEBAP0AAQAF/QD+AQD/AQT//v4BAP//AwMDAQICAP8DAAABAf7//wQBAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABv4H+v37/vkFB/sDAwf6+vsEBP39/QIE/vwHAwIG+/kD/wMB/v3/+wMC/QEBAv7//gED///+AQL//wMBAQP9/QAAAgEBAQD+/wABAAAD/gABAf8BAAAB/wAD/wAA/wEBAAAEBAAB//0AAQEAAAQAAQEBAAD/AgAAAAL//wABAQAJ/Qn4/fn99wcL+QMDCfv3+gUF/P36AgX9+goDAwr49/8AAAUAAwD+//4CAAAAAgQB/wD/AAQA/wAA/v///wH9AQABAAD///4BAv8AAAP+/wACAAD//wEBAP8BAAACAP8C/wMA/v3//AID/gEBA/39/gID///8AQL//gMBAQL+/gT/BP/+/f/7BAP+AgID+vr9AgP+//sBAv7+BAICAf37/gECBAACAAD9/wIA/wMDAAL//QIABAAAAQP8//8AAgMBAAH//wD//wP//wEBAPv9/gAB/wD9AQH/AQECAf3//v8ABAMAAwD8/wQDAQACAQACAP4B/wMAAQD//QD/AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAE/gUA//z9AAIB/QICAwD9/QMAAAAAAAD+/QQAAQX+/gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABv4H/P77/vkFB/sCAwb6+vwEBP39/QIE/voGAgIG+/oD/wMA/////gIE/QEBA//9/gIC/wP/AQL//QQBAQP+/QECAwQA/wAAAP8AAAABAgEB/wIBAgIA/wAD/wAA/wEAAQAEBQAE//8BAAEAAAP/AP8B/gD/Av8BAAAAAAADAf4J/Qr4/Pj99wcL+AQECPr5+QYF+/z5Awb8+AkEBAv49v8B/wIBAgAD/v4C//8BAwMB//4CAgH//wED/f///wICAQABAQD/AP7/AQEAAAAAAAAAAQD/AQAAAAD/AAAAAQECAAMA/////QIC/wEBA/39/gIC/wH/AQL//gMBAQP+/gL/AwD+/P//BAP9AQIF/Pz9AwL+AP0BA/79AwECAv36/gD/AwEDAP///gMA////AQH//AL/A/8AAQD+//8AAgP//wAA//wABAMA/gEBBP79/gIB/wD9AAL//wEBAf7//AL//gAB/wAA/gMC/wAAAAD+AP0B/wIAAQH9/f//AQADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/wMA/wD/AAEB/gEDAP78/gAAAP8AAAL/AAEBAAAA/gP/BP///f79AwT9AQIDAP39AgL+//4BAv/9BAIBA/39AAAEAwAAAP8CAf8AAAAAAgABAQD/AgABAP4BAAAA/wAC/wb/////+wIB/gEBBfz9/wMA/v7/AAL//QUBAQT8/AMABf7/Af8AAQL+AQECAAAAAwIAAAEAAv//AwEBAQAAAgAFAgAD//8AAf8AAAYA/gAB+wD/AgAAAAEBAAAE/wAC/wD/AAEAAP8B/wAA/wAC/wACAQACAAEAAgAAAP8AAgX/Af/+/v//AQf8AgL///7+AgD//wEAAv7/BAEBAf0AAQAAAAAA//3/AQEAAAACAgAAAQD/AAEBAP//AAACAP4A/wABAAEAAP8CAQAAAQEAAAD9Af8DAP8A////AP0AAf//AgIAAv8A//4BAAABAwQAAv8B/wIAAAACAAAAAwEC/wAAAQEDAP3+/QH//v4CBQL//wEAAwD/AQP+//8BAgH+AAABAQIAAP//Af//AQMCAf4DAAACAP8AAQD/AP8C/gL/AP3//P/9AwL+AQEA/P3/AQT+//wAAv/9BAIBAv7+Af8A/wAA//0AAv8AAAD///8CAP7//gABAPwAAAAE/v//AAAAAQMA/v8AAP///QIDAQD///8CAAABAf7//wIBAwb+Bvv++v78BQj7AgME+/r8BAX9/voCBP76BgICBvv5AwAE/gD8AP8AAQAAAAIA/f8CAf///QAAAAEBAAAAAAAG/gP//fr+/AMF+wIDBfv9/AMB//79AAT++QQCAgT8+wQAA/3/+gD+/wX+AQECAvz+AAD///8AAP8FAwAAAQACAAH//gD8AAICAf8AAP4B/f8AAQAA/v8AAP8BAQABAAL+AP4AAQAAAAD/Af//AQICAf8B/wADAP8BAP////7/AgAAAv//AP//AQL+AAD/Av//Af///wAAAv/8AQAAA/4BA/8B+//+APwBBP4BAQIA//4BBP/+/AEC//oBAQEF/vwCAAD9AAAB/AECAAAA/wEAAAD/AP/+AAEA/gEAAAL+/wIAAP4AAf/9/wT+AAABAAH/AQP//wAAAAD/AAAABP8AAQEB/wEDAAH+/gL+//0C/gEBAAAAAP//Af79////AQICAAD9//oA/AME/QICAvz7/gIC/v78AQH/AgUBAQP+AAMBAgD//wD8/wX+AQEAAP/+AwEA/wAAAP8DAwAABf8BAQAD/f/9AP8BAv8BAQL9/P8CAQD//QAA/wECAAAB/v4G/v/9/gD++QAF/QEC/wH8/AQA//7/AAT/+gABAQb+/Qf+B/r9+v76Bgn6AwMG+/r7BAT9/foCBP76BwMCB/r5AQD5BAACAQAAAv8AAPj9AQAAAQAB/QAAAPgAAAAAAAECAAABAAQA/wACAAAA/wL+/wH9AAABAQAAAAEAAAIBBAT/Bf/+/f/7BAb9AgIE/v39A/////8BA/78BQICB/39AAABAwEDAAD/AgD/AAEA/v8B/AEAAQD/AAH///8BAAEE/gT//vz/+QMF/QICBf/+/QP//v7/AAP+/QQCAgb8/QEAAgMABQAC/////wAEAv4AAfsB/wQAAQAAAQAAAwAEBP4C//78/vsDBf0CAgL++/0DAf///QED/v0EAgIF/P0CAAb+/wD+/AAD/wEBA/7+AAIAAP4AAAEAAAAAAAX8+wL/Af7//gD7AQT+AQED/wH+AgH///4AAf//AgEBAv79AAAB/wAC//4A/wAAAAIAAP8BAQAAAQD/AAIBAAAC/QAD/wAA////+gID/gEBAv39/gIAAP8AAQL/AAMBAQT8+/8AAAABAf///wAB/wACAAABAf8BAAIA/wED////Af0AAf8DAf8A/vwBA/8BAQP8/v4C/wAAAQEB/wECAAEE/v8D/wP//////QIF/QIBA/79/gMB///8AAP/+wIBAQT//gb+Bvz+/P77BQf7AgMG/fz8BAH9/v0CBP78BgICBvv6BP8A/f78//wDBP0CAgD+/v0CA/7+/QEC//wEAgEE/fz9AAH+AP4BBAL8AAABAvv8AAEB/wD9AQAA/gIBAAD/AQT/Avz+/P/6BAP9AgIC/fr8AgP+//oBAv79BQICBf38/wAA/QD//wEC/AAAAf3+/QAAAf8B/AD/AP8CAQH//wEC/wH///7/AAEC/gEBAgD9/wED/wD/AAD//gEBAQH///4B/wQCBAEB+/wE/v4AAQYC/vwBAQQA/QIE+f7+AQUHAgAD/QAB/wAAAv8AAAMAAP4BAAEAAAABAAABAAACAP8BAAAA/wD//AMB/wEBA//9/gEB///9AQH//gABAQD//gMBAgAAAv/+/wABAAADBAMBAAABAAP/AAAD/gAAAgEC/QD+/wD+AAAC/AEBAf0A/gD/Av8B/QH+//4BAQH9/wIC/wD//wH+/P8D/gEAAwID//7+AQAA/wIABP8AAAD+/wL/A/7////9AQL+AgED/v/+AAAAAP4AAf/9AgEBAP7///8BAgAC/wAA/P8BAAAB/gD+/wAA/wAAAP4BAAD//wP+AQAEAgYAAPz7A/7+AAQFAwD8AgAE/wACAvz+/gAFAwb+B/r9+f75BQf7AwMH+fn7BAb8/fkCBP35BwMDB/v5BP8B/f/8//0CAv4BAQH8//8BBf7//AEB//sCAQEC/v0AAAEBAP0AAQIB/wAAAQD9AAEBAQD9AAAAAwMBAAL/+wP+BPz++//6BQX8AgIF+/z+AwT9/vsCA/78BgICBPz7AAACBAAD//0BAAAAAAACAQEA/QAAAwAAAAQCAAD/AP4E/gP8/vv/+gUF/QICBPz6/QME/v77AQP+/AUDAgT7/AT/Bf3/+/79AwX9AQEF+/7+AgT+/vwBAv/8BAIBA/3/Af8A//////0BAv4AAQH+/v4CAf/+/wAC/wADAgED//8D/gb///v+AAIF/QEBBvv9/QEC/v/8AQL//QQBAQQAAAEAAQAA/gAAAQAA/gAA/v4AAAEAAP8AAAIAAQAA/wEBAgADAP/+AP8CA/8AAQD8/P8BAf8A/gAA//0DAQEC//8CAAMA//4A/wID/v8BAP39/gIB/wD+AAH//QMBAQP/AP8AAwAAAAABAAMBAQAAAwD/Av0BAAMAAAD+Af//AP0EA/8DAP/+AAACA/7/AQD+/v4CAQAA/wAC/v0EAAEDAAAC/wIA//4AAAIC/v8BAP7+/gEBAAD/AAH//QMAAQIAAAL/Av///gD+AQL/AQEC/v7/AQL///4AAf/+AgEBAv/+Bv4I+f37//cFB/sCAgb6+/wEBv39+AIE/fkGAwIH+vn/AAD+//z//wMD/gEAAf78/gAC/gD8AAD/AgUBAQT+/P8BA/sA/gD9AAX/AAAEAAD/Af8AAAIBAAADAQAABf3+Af8B/f/9//sDA/0BAgP//f0CAf/+/QEC//0EAQED/f3//wD8AAL//gEC/gH/AgMB/gECAP8DAAEAAgEAAAL/AAAA/gEA/wAAAf4AAAD+Av//AAAAAP4AAAD/AgAAAP/+BP0A+/78//sDBPwCAwj+/vsEAP/+/QED/voFAQEE+/sA/wH9//7/AgID/QEBA/39/gIB/wD9AQL//gQBAQL+/AL/AQD//P8AAQP+AgEC////AwEB/gAAAf//AwEBAP/9A/8AAP/+//0DBP8BAQL//v8BAP///wAD//4EAQEC/v3+AAD/AP4AAQL/AAD+AP7/Af8D/wD+AAAAAwIAAP8AAAH/A////P//AgH/AQED/fz/AQL/AP0AAAD/BAEBAv79AAAB/gD+AAECAf8BAAH9/gABA/7//QEBAP8DAQEB//4B/wEAAAL/AP4BAAD/AgL+AAP+Av4DAAEA//7//wMAAgL/Av//AP/+AgL+AQECAAD/AQD//wABAf//AgEBAv/+Bv0F/f36//oECPwDAwX6+vwCBP7//AEF/f0HAwIE/fsD/wL8//7//gIBAAABAfv7/wIE/v/9AQH//AEBAQH9/AH/Av3//f/8AgP+AgEB/fv+AwL//v0BAf/9AwEBAv3+Av8F//7//vsDBP0CAgL8+/0CAf/+/gED//8EAgED/f7+AAICAQMA+/8AAP8AAv/+AAIC/v8CAP8B/v4AAAIBAQL/Av3//v/9AgP+AQAC/fz9AgT/APwBAv//AwEBBP79/QH/AgEFAQP+/AP///4EBQL+/QEAA//+AQb8///+AQQCAAIA/wIA/wP+//4AAv7//wABAAD9AAH//wEAAQP/AAIAAQD/AP8DAQEA/wD//fz+AgEAAf8AAf/7AgABAQD+AAAEAP///wACBP8BAQD+/P4AAgAA/QAD/wQDAAECAf8AAf39AAECAAH7AAAA/gAB//8A/wD/AP4ABAEAAfz+AAMBAAAAAgAA//4AAQAAAwIAAgAB/gMA/wAC//8AAP//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT/BP3+/v/9AwT9AQIE/P79AgL+//4BAv/9BAIBBP38/QH8AwEFAQX+/gL///wBAwP+/wEBAAD+Afz9///9AwP+AAH/APwA/gD/AAAAA/7+AAAE/v/9AAAA/AAAAAIA/wL/A/7/Af8AAQP+AQEB/v7/AgP/APwBAf/7AQEBAf4B/f8C/AD7AAMBAf8BAQP///8BA/7//QEB//wBAAADAP4D/gb6//v+/QIE/gEBBf76/gQB/v/9AQL/+gIBAQX9/gP/A/r////8AgP+AQIE/v//AQL+APwCAv/9AgEBA//+Av79/v/8AAACBP4AAQD9/f4CAv8A/QD//wEAAQEA/f0B/wT9//0A/AID/gEBAv3+/gEC///9AQL//wMBAQL9/f8A+wEBBQAA/QADAP4ABAED//wDAAUAAAEA//4A/gMEAv8AAP///wABAgACAQMA//8BAQAAAAAAAP4CAAADAAAAAP4AAQIAAP8AAP8BAQIC/wD+AQEBAP8B/////wEAAP8AAAABAQAAAP8B/wD/AAEB/wAAAAAAAAAB/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+Af8AAQMBAP79Av/+/wMDAQD+AgADAP8BAv3//gACAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/8A//////0CA/4BAgMA//4CAAD/AAEC/wACAQIA/gD/AAL+AP8BAQD/AAAAAgT/Af/+AAAAAAAA/f8BAP7+/AAAAv8A/gL+AAIA/gAB//8A/wH/AAAA/wAA/gAA/v8CAQABAgAEAP8B/gAAAP4AAAAA+wEAAwAAAAUBAAABAAUB/wT+//0A/AIE/gAAA//+/wIB/v78AAH//AEAAQEA/QAAAf4A/QD/AQAAAAABAP8AAAAAAAABAAD/AQEAAP/+Af8D//8B//wBAv8BAQQA/v8C/wD//gACAP8BAQAD/gEC/wIA/wH//gEB/wEBAf8A/wEAAP//AAH/AAIBAQL/AQICAP///v8AAf4DAQH//v8E/gD9AP8AAv/8/gEBAwAAAAAC/wD+AgAAAv/+AP//////Af8A/wD/AP//AAD9/wAAAAAAAAAAAAAAAAAA/wABAAAAAAABAAAAAAAAAAAAAAP/BP7//v78AgX+AQEC/Pz+AwL+//sBAv/8AwEBBP7+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/Av//////AgP+AQEC/v7+AgH+//4BAv//AwEBA/79/wD//wD/AAMB+wEAAP4A/AD/Af8B/wD+AAABAAD+AQMC/wQA/wD//AMD/gEBAwAA/gIAAP8AAQL/AAMBAQH+/wAA//4A/gAFAgH/AAH9/wAAAAH/Av4AAAD/AQEAAQECAv4G/P78AAIDBfwBAgT6+v4CA/7/+wEB//sFAgEF/PsAAAP+Af8AAf8C//4A/gIBAQAB/wAD//8BAP8AAAICAf4B/gABAAEAAP0A////AQIA/wAAAgAA/wD/AAAA/gAAAAAAAAD/AAABAv8BAAD+/wAAAQAA/gEB/wACAAAAAP//AfwAAgAAAP4AAf/+AAMDAgAAAAAE/v8BBP/+/v8ABAIABAD/AAD/AQP/AgMA//7/AgAAAP4AAAD9AgAAAwAAA/8F/wD/AP//AAABAAIAAf8C/wD+AAH+APwA/wAD/wAC/wX+AP8A//8AAAABAAAC/wL/AQEBAP4A/P///wP/AgQABQQABP///wEAAAAAAAEAA/0BAAH/AAD9////BAIBAv8DAP///wABAv8BAQD+/v8DAf8A/wAA//4BAAAD/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAb+Bvr9+v76BQf7AwMG+vr8BAb9/fkCBP76BgMCBvv6BP8F/P78//oEBvwCAgX9+/0DBP7+/AED/vwFAgIF/PwAAAAAAP8AAAEB/wABAv38AAEAAP/+AAEA/wIBAAL//P//BPz//v79AgT/AQEF/f//AgD///8BAv/+AwEBA/3+//8D/P/8//0CA/4BAQL+/v4CA/7//QEB/v8DAQEC/v39AP8AAAEAAP///wAAAgD9AAD+AAAAAAAA/v8AAP////wA/QABAgEA//wA/v/+AAAB//4BAAH//wH9/v///gAABP8D/f79//0CBf4BAgP8+/0CA/7+/AEC//wDAQEE/v7/AAAAAQMB/v7/AP7/AAICAP/9AQACAAACAf////8AAgAAAAAA/gAAAQD/AAEA/wD/AQAAAP4AAP/9AgAAAAD+AAABAAACAP7/AP///wEAAgAA/gEAAQAAAf///wD/AP4C/wIB/////gED/wEBA//8/wIB//3/AAH/AAEAAAMA/v8B/wABAwH+/v8A//8AAgMB//4BAAEAAAEC/v///wAAAv8CAP/+AP4CA/4AAQD//v8BAf8A/wAB//8DAAECAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAL/Av///wD/AQL/AQEB////AQH/AP4AAf/+AQEBAf//A/8C/f/8//wBBP0CAgT+/v4DA/7+AAAB//8EAAAE/QD/AAAAAAEA/wAAAAAAAAEBAAAAAAAAAAAA/wAAAAD/AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/wAAAAD/AAABAQAAAAEAAAAAAQAA/wAAAP8AAAAAAf8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP+BP3+/P79AwX9AgIF/P39AwP9/vwBA/77BQIBBP39Av8B/v/+AAACA/4BAQL+/v4BAv///QEB//0CAQEC/v4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT/BP3//P8AAwT9AQEE/AD+AgT+/vwCAv/8BAEBBP4AA/8C//////0CA/0BAQH+AP4CAv7//QEC//0DAQEC/gAAAAAA//0AAAEA/wEAAP4AAAAA/wAAAAAAAAIBAQAA/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABf4H+v35/vkFCPsDAwb5+fsEBv39+QIE/fkHAwIH+/n/AAH/AAABAwH/AAAAAP4AAP7//wH/AP8AAP8AAgEA/gMAAwD+/QAAAwAAAAAA/PwAAAAAAAAAAP78AAACAAD8AP8DAP///wABBP4BAQQAAP4CAAD/AAAC/wADAAEA/wD+AP4AAQIBAP//Af8A/wECAf8AAAACAP8AAv//AP4AAgAA/wEAAQAA//8BAAD/AQAB//8AAQEA/wEB//8A/wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/Av///gAAAQL/AQEC/v7/AQL///4BAf/+AgEBAv/+AQAFAP7//gADBP4CAgL9/P0CAv///AAC/wAEAAEDAP/9Af0AAQIBAP7/Af/+/QMBAv8AAAEEAP8BAf//Af4AAwH/AAD//f8AAQT+AAAA/vz9AwIAAP4AA/8AAwAABAD/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv8C/v/+//4CAv4BAQL+/v4CAv7//gEC//4CAQEC/v4AAP4DAAIA/gD+AAAAAAABAAEA/wACAAEAAgEAAQD/AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/wAAAP//AAAB/wAAAAD+/wEBAAAAAAEAAAEAAAIAAP4A/wABAgAA/v4B/v8AAwMBAAAAAAIAAAEC/wD/AAAD/AH9AwMCAQD/+wL+/v8BAwH+/QQBA///AgH9//7+AAMB/gMA/v/+AAAB/wEABPz+AAMB/v7/AAL9AAIBAQMA/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/wAA//4AAAIA/gEBAP7+AAAC/wD+AQH/AAICAQD+/gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/wH+AgECAQL//gH///8CAgH//wEBAgD/AQL+///+AQIB/wABAP8A/gABAAAAAAD+/gEAAAABAAEAAQAAAAEB/gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC/wT//wD//wID/gEBA//+/wEA//8BAQH//wIBAQL+/wAAAwD/AP8AAgP+AQEA/f3+AAEAAP0AAv8AAwEBAAAABP8FAP8A/wACAv8BAAT//f8DAP/7/gEB//4BAAAEAP7+AP4BAgIB//37A//+AAQFAP/8AgAG//4B//z+/v7+BAL/AQD/AP8AAQL/AQEBAAD/AQD//wAAAf/+AgEBAv7+/QIAAwEAAQP+/AL//wADAAL/AAEBAP//AQP9///9AgMBAP4A//4AAAIB/gAB/v4A/wEC/wD+AQH//wMBAQH//gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+Af8BAQABAP/+Af///wECAf8AAAEAAP8AAv//AP4CAAAB/gIBAwEA/v0C///+AAMC/v4CAQD//gEC/v///wIC+wL6BgIHAgb7+gX+/vsGBgT9+wMCBf78Agf6/v76BQb+AAD/AAMABP//Av8AAf4B////AQD/Af8AAf4C//sC/gAA/QABAAEA/v0C//8AAAMCAAAAAAMA/gEA/QD//wAAAAD9AAACAAD//gEA//8CAgEA/wEAAgAAAAAA/wD/AQED/wMA//7//gEE/gAAAv79/gIBAP/+AAL//gMAAQMA/wEAAf8A/wD/AQD/AAEA/wD/AQH/AP8AAQD+AQAAAQD//gEAAAEAAQL+/gL/AAAAAgL+/gIBAQD+AQD+AP/+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP8A/wABAQAB//8BAP//AQEB//8BAAEA/wAB/wD//wEB/QH/AAEEAQD9/gP+//8EBAL+/wAAAwD/AgT+//8AAAQC//4A//z/AAED////Af78/gMDAP//AAP/AAMAAAQA//0B/QABAgEA/v8B//7+AgIC/wAAAAMA/wEC//8A/gADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP4B/gIBAwEC/v4B///+AgMC/v4BAQMA/wED/f///gED/gL/BQIGAQP8/gP+/vsGAwIA+wICBv79AQP8/f4ABAf/AQACAQIAAv//Af///gICAf8BAQECAP8BAv7//wEBAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA/wD+AAABAP8AAQD+AP8CAf8A/gAAAP4CAAEA//4AAAACAAAAAgABAP8AAAABAP//AAEBAAAAAgEAAAACAf4B/gIBAQAB//8B////AwQAAP0BAAMAAAEEAAD//wID/gL9AgEDAQP+/wL///0DAAL//gEBA///AQL9//8AAgMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPwB+wQDBQEC/f0D//78BQMD/vwCAQX//gEF/P7+/AMF/QD8AwEEAQT+/QL///wEAAL+/gEBA//+AQP9///+AgT/AP8AAAAAAAD+AAD/AAACAf8AAAAAAAAAAP8AAP8AAAAAAv8AAAD/AAAAAAABAP8AAAAAAAAAAAD/AQAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPsC+gUCBwIG+/oF/v76BgUE/foDAgf+/AIG+v7++gQG/AL7BQIGAgX8/AT+/vsFBQP9/AICBf/9Agb7/v/9BAX/AP4BAAEAAgD/AQD/AAAAAAD/AAABAAAAAAAAAP8BAP4AAQMBBAED/vsCAP8BAQEB/fwCAAT//wED/v//+wID/gD/AgECAQUA/wEAAP8BAgH//wEBAQD/AAL//wABAgH/AAECAAAAAv/+AQAAAAECAQD/AQACAAAAAv7//wECAQH/AP///f8AAQD/AQAB//8AAgL+//4AAf/+AgAAAQD8/AH7BAIGAQL8+wT+/vwCBAL+/AIBBP/+AQX8/v7+AgMB/wMC//8A/AEEAAEBA/3/AAEBAP//AAD/AAEAAAP//v4BAAIBAgEB//4C///+AgICAP4BAgL//gEB/f//AAICAf8CAgD/APwBBAAAAQP+/wABAAD/AAAAAAACAAAC/v7+Af0CAgQBAf3+BP79AQIBAv79AwEE//8BBPz+/v4AAgH/AgIAAAD+AQMBAQEA/QABAQEAAP4AAP8BAAAAAgH+/gD+AgECAQD/AAH//wECAgH//gEAAv//AQL/////AAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPwB+wQCBgEE/PsE/v78BQQD/vwCAgb//QIF+/7+/AQF/wH8/QD+AAAB/gAAAP79/v7/AQAA+wAAAP4CAAD+/v8AAAADAQP/AP7+Av7/AAMDAAD+AQAD/wABAQD//wACBP4BAf4BAAED//0B/wAAAf4AAP8BAQAA/wH//////gAA/QH+AAIEAQX9/AP///4EBQL9+wICBf/+AgT8/v79AwMDAAH+//0AAgAA/gEB/gEA/gL+Af//AP//AQAAAQD9/AIAAP4A/gAAAQMAAQEAAP0AAAL/AP4AAP//AgEBAQD//QH9AQECAQL+/gH///4CAQH+/wEBAv//AQL/AAD+AQIDAgQA/wEA/wIE/gID/v/+/gIAAAL+AP///QIAAQQAAAAB/QABAAAA/gAB//8AAgICAAAAAAP//wEC/////wAD/QD9AwEBAAD+/AH+/gEAAwIA/gAABP4AAQP/AP8AAAT/AP8AAAIA//0AAf7/AgADAgH/AP8D/wABBAD//wEABAEA/wABAgAA/gIA/wABAgMBAP0BAAT/AAECAP//AP8EAAH/AAEDAf/+AQH/AP8CAgL//gEBA///AQL+///+AAMC/wL///8A/wEC/wABAf4A/wEBAAD+AAH//wIAAQL//wT/BP3/Av//AgT+AQEBAv3+Av///wIBAv8CAwEBBP4AAAACAQACAP7/AQEAAAICAQEA/wD/AgAAAAD/AAABAQL+AAAAAQIAAv/9AAD/AP/8Af8BAAD/AAAAAAABAP4A//8A/gEAAQAC//4BAAAAAQH///8BAv8AAAAA/wAB/wD+BAAC/gD+//sBAP8AAAb+/v4DAf///wACAPwBAAAC////Af8BAQEBAP78Av//AQH+Av78AQEAAP4BA/4AAP4DA/4AAAAAAAD/AP4AAAD/AP4A/gEAAP4AAAD+AQAA/wD/AQACAgACAP//AQAA/wMBAAAB/QH/Af8AAAD///8AAQMAAP8BAQEB///+Af//AAEBAAD/AQAAAAABAP7//wEBAP4B/gACAQEA//0B////AgEB/gABAAIA/wED//8A/gAA/wD+AAABAAL//gEAAAEBAf///wAC/gAAAAH/AAH/AAAAAP8AAAAAAAD/AAAAAAAAAP//AAAB/wAAAQAAAAAAAf4B/QIBAwEC/v4C//8AAgMC//4CAQP//gED/f///wIDAQAB/wD/AP8BAf8AAAH/AP8BAv8A/wABAP8BAAAA////Av4CAQIBAv/+Af///gL/Af/+AQECAP8BAv7///8BAv4B/QMBBAEC/v0C///+AwQC//0CAQP//wEE/f///gIDAv8A/P/8//sCBP0BAf38/P3+BP7//AEC//4AAQED/fz+AfsAAAABAQD+AQAA/QECAP7+AAH+AP8AAgAAAPz/AP8A/gABAQAB/gABAP8BAQAA/wEBAgEAAAEB+///AAIBAAH8AAABAQD/+wH///0BAAH//wABAQD/AAL/AAD8AQH9AQICAQUCBf37Av7/AgQDAf38AgEE//4BBv3//wECAwEB/QMBAQEB/v0B//8AAgABAP4BAgL//wEE/v///AEC/gH+/wEDAQT//QH///3/AgL/AQAB/wD/AQT+AP/+Av8AAf4BAQQAAv78Av///wECAv//AAADAP8BAv3///8DAwP+BP7+/f//Agb8AgIEAP/9AgD//wAAA/7/BAECAP4AAAD/AgEBAP//AAEAAAABAgAAAQEAAgAAAQEA//8BAAMC/wIBAAD//QEBAgEBAP8AAAECAP//AAAAAAAAAAEA/wEB/QEAAAACAP4BAAD+AAAB//4AAQD/AAAA/gAA/gECAv8C/v/+//4CA/4BAQL9/v8BAv///gAB//4CAQEB/v4B/wL//wD//gAB/wEBAAD///7/AQH/AAH//gIBAf7//wAB/gIBAgEA/v4C////AAIB//4BAQD//wEC/v///wIC/AH7BgIIAgX7+gX+/fsDBAX9+wMCBP/9Agb7/v78BAb9Af0DAAMBBP79Av7/+wEAAv3+AQIDAP4AA/3/AP8CAv4B/gMBAgED/f0C/v79BQYD/f4BAQb//gEE+v7+/wQF/gH/BQIGAQP9+gP+/v4FBQH++wMBBv/+AgT7/v7/BAQD//8C/wIA/QADAgEB/gEAAf7+AAECAAL/AAIBAQAAAf0B/QMBAgED/gAC///+AgQB/v4BAQP//wED/////QIDBP8E/v78//8DBf0BAQL8+/0DAv8A/AED/vwEAQIE/v0AAf/+Af4BAf8BAf8A/wEAAP8B/wACAAAAAv7/AP39Af8A/wAAAAEB//wB/wD/AAMB/wAAAAIA/wAC/wAA/v8B/gH+AAECAQL+/QH///8CAwH/AP4AAQAAAQUAAP/+AAMBAAUA//0AAgEBAP8AAv7//wEBAAD7AAD/BQEAAAIB/v8CAAAB/v8A/gP/AAD/BP8AAQIAAQQAAAD9//8AAQMDAAAAAgECAAD+/gL//wACAgAA/gEAAv8AAQL+//8AAgIC/gD+//8A/gEC/gEBAAAA/gAA/wAAAAL/AAIBAQD/AP8AAAAAAQAA//8AAP//AAEB/wABAAEAAAAB/wAAAAAB/QL4BgMGAgf8+wT9/fkHBQT9+gMCB/77Agf7/v77BQb9AfwCAf4BA//7Af/+/v4EAf7/AAAA//4B//3///0BAgEAAQMAAwAE/v4A//8CAgMAAP4AAAL//gAB/wAA/QIC/gD9BAIDAQT+/AP///4EAwH//gEBBf/+AQT+////BAb/AP4BAAMAA/4AAAAA/wEAAP/+Af8CAAEAAP7//wIAAP8AAAEA/gAAAQD/AAAB/gQAAAEAAP8AAAD+/wAA///++gH/BQEFAAP8+gT+/gICBAMA/AIABv/+AgX8/v7/BAf9Af4CAQIBBP7+Av7+/wEEA//+AgEB//4BAv3//wACBP//Af8BAwEA//wB///+AgIB//wB/wP/AAEC/v//Av8D/QH/BAEEAAP9/wP//gAAAAL//QIAA/8AAQP9/v8AAQH+Af0CAf8AAP8BA////v4AAv4BAAD+AAAAAP8AAP8AAP4A/gMBAAAC//8BAP3+AAMC/v4BAAD//wH//v///gAA/QH8AwEBAQP+AAP///0BAAP+/wEBAf//AQL+////AgIC/wMAAAT/Af8CAAEABAL8AAL9AgEB/wIAAf/////+AvsC+gUCBgIG/PoE/v76BgUE/fsDAgb//QIG+/7++gQG/AH7AwEDAgT+/gL///sDAwT9/gECBP/9AQL+///+AgL/Af8CAAAAAAD/AwAAAQAAAAH/AAABAP8A/wAAAP4AAfwB/AUCBQEE/PsE/v78BQQD/vsCAQT//QIC/P7+/QMFAv8DAAD/AP0BAgAAAAT///8CAAD//gABAP4CAAADAP/9AfsEAgQBBP37A//+/AYEAv38AgEE//0BAP3+//4DA/wB+wUCBQEC/fwF//76BAMD/fsCAgX//QIF/f//+wME/QH//QEBAQL+/gL+/wACAwH+AAEBAgH+AQL+///+AQH7AP8DAQIBAf78BP//AAMAAv78AQEEAP4BA/////0BA/4BAP4AAAD9AAABAAAAAP8A/wAAAAAA/wAAAgAAAP7//gL8/wEAAf3//QP//wABAQL+/wAAAv/+AQP////8AAL+Af3+AQAB/f//Av//AgEBAf8AAAAC//8BAgD///7/AgMAAv4A/v8A/wX+AAEFAAH//wAAAAIAAQADA/8ABf8D/gEAAAECAAD+/wH///8CAgH//QEAAgD/AQP/////AAL+AQAAAAEA//8AAf//AAICAf/+AQABAP8BAv8AAP8AAv4B/gMBBAEB/fwD///9BQIC/vsCAQX//gEE/P///gME/gH+AAEDAQL+/AL///0DAwL+/QECA//+AQT9///+AgT+AfwAAf8BBAD6BAAA/wD9AwEAAAEAAQEBAP8A//0B/v4C/AMCBAEB/fwE/v78AwIB/fsCAgP//QED/P7+/gMF/gH//wABAAEA/QEAAP7/AP/+AAAB/wAAAP8AAAD+AQD+Af//AAIAAv79AAAA/AEBAf//AAABAP8AAf4AAAABAAL/Af/+/gD/BAX8AgL//v/8AQH///4BAf7+AwICBf38/gD/AQAEAAT+/wAAAP8DAAAA/QEAAwD/AP/+/wACAAL+Af8BAQEAAQH+AwD//wABAf4AAAEAAAAAAv8AAAD/Av8AAf8AAf//AQP/AAD/A/7+AAAA/wIAAQD8/wEABAAA/gH9AAD/AAIB/AAAAP7/Af3+AQAB/QAB/wIBAAD+/wABAP4AAAIA/v8A/wAAAAID/gAAAAEEAP8AAf//AAMBAf0BAAABAwAA/v0B//4ABAQCAP0BAAMA/wED/f/+AAIDAf8C/wD//wMA/v8BAQICAv8C/gAAAgAB/wABAAEC/gAE/gUC/v7+AgIC/gECBf7//QMAAP//AAH//AQBAQT+/fsC/AUDBwIG+/sF/f38CAcE/foDAwj+/AIG+f3++wUH/wAA/gH/APz/AgEA//4AAAD8AAEAAAAAAAL9AAACAAABAAUAAQEA//8CAQD//QIA//7+AQEBAAAAAf3/AAMBAP4B/QQCBQED/PoD//7/BQQD//0DAAb//QIG+v7+/gQFBP8AAf8BAP4BAwEAAP4A//8B/gAAAQAC/wAAAAEC/wH+AgAGAgQBAv37A/7+/gUEA/76AwEG//0BBPv+//8FBgP/AAEA/wD+/wQBAAAAAgH/Af4B/wQAAQD//wAAAgAC/gH9BAIEAQH9+gP//v4EBQP/+wIBBQD/AgT7/v8AAwUAAQACAQMABP7/BP///wQBAf/5AgEG//8BA/7///sDBP4B/wQBAwEF/v8D/v79BAIB//wCAQP//gEG/v//AAQFAAH9AQD/AP8AAgEAAP8B/gAA/gEBAAABAAAAAAAAAAEAAf0CAQMBAf7/BP///gUCAv77AQEF//8BA/3///4BAwEA/gEA/wD+AQMBAQH/Av4AAAAAAQEAAv//AQAA//4AAQD/AwABAf4AAAUAAP8DAAH//wEBAwABAAH/AAD/AAL+Af0BAgUBAvz9Af7+/QQFAv78AgEF//4BBPv+//8CBP4B/QMCBgED/PsE/v79BQYD/vwCAgX//gIE+/7+/AQFAP8CAAD/AAYAAAEAAAIDAQL//gAAAgABAP//AAAAAAD9Af0EAwcBBPv6Bf79/QYIAv77AgEG//4CBvz+/vsFBf0ABQAAAAAD//7//wAB/wQA///+/wIAAAD///8ABQIB/wEA/QD/AP7//QIAAAH+AAP+AgAB/wEAAAH8AAADAQIDAAMA//4A/QAC/wABBfz//wAA/wD+AAEAAAIAAQP//v8A//8ABP/9AAAA/wD9AwMB/wABAAMAAQEE/AAAAAEC/gABAwEFAAT+AgD///0FAQL+/QICBQD+AQT6////AwMAAAACAAP/BP8CAAD/AAQAAQD8AQAEAAABA/z///8CAgEAAf//AwAFAAH/AQH+AAX+/v4AAAMAAQAEAAAB/wECAQAB/f///wICAv8BAQD/AQEAA/8A/wEBAAEAAAEB/v76Av3/AQEBBf77A////AIBAv8BAAADAf4CAf3///4CAgD/AP4AAQAD/wL/AAAAAwMA/wAAAAIAAAAC/v8AAgIC/wH8/QH/AAH//QMAAAABAAP/AP4BAQAAAgH/AP/+/wD+AQACAQYAAf0BAv///wQFAfz8AgIF//4BBP3///0DBP8BAwIBAwAB/v4C//8CAwMB//4BAQP//wEC/v///gIDAQD+AAAAAPz//wEAAAEAAAD/AAAAAAAAAAL/AAAB/wD+Af8DAQEBA/74A///AAICAv/9AQAB//4BA/3///sDAAIAAgABAf/+/gAA//8AA///AQABAAAAAAEA/wD//QABAwAFAQAAAAD//wEA/wYDAAAB+gEAAQD/AAMA/wAAAQMC/wMCAAD//wEC/gABAAAA/gAD/wD/AAIA/wMBAP4AAQABAAACBgD8/QAB/v4CBAIBAAECAAMB/gIG/P7/AgMF/wACAwAB/wEA/wIAAAABAQD//QEBAwAAAAEAAAD/AAIA/wH/AAH//v/+AgAAAAAAAAL+AQACAQEAAQEAAAD//AD/BQMA//8BAAIBAAABAP8AAf4BAAD/AAD+AAAAAAABAQACAAD+/wIBAf8BAQEAAP8CAQD///8BAP0BAAAB////AP4CAP3/AwECAAEB/v7//wEB/wD/AAH//gIAAP///v0B/wABAgAB/v4C///+AQIC/v8BAQH//wEB/v///wIC/wABBAADAAP+/AH/AAEDAwH/+wIABP/9AAP+//8AAgIC/wQC/wD/AQAB/wEBAQEB/wEAAP4B/wD//gEAAQAAAAEAA/8A/wD+AQL/AAAC//7/AQEAAP8AAQD+AQAAAv///gL7BgMHAgT7+QX+/fsHBwT9+gMCB/79Agb6/v76BQb+Af4CAQMA//7/Af///gEDAQD+AQABAP8BAf7///4CAwIA/gAAAAD9Af8AAAD+/wD/AAAA/wAAAgD/AAAAAP//AAEDAAABAPsAAgAAAAD+//8B/wH/AP8CAP8AAAD//gD9AQEEAgUBAvz7BP7+AAUGBP78AgIF//0CBvv+/v0EBQAABQAA///+AQP/AAEB/f/+AwAA//4AAQD/AQAAAf8DAf//Af//AP8B//8AAf//AP//AAAAAAABAAEBAAAB////AP4DAAMAAf79Av///QEDAf7+AQEDAAABA/7///0CA/4B/gYCBAED/voE//8ABQMC/vwBAgUA/QED/f//+gMDAQADAf//AP4BAv4BAQD+//8BAQD//QACAP8BAAAB/wD/AAEDAP8BAQD+AQAAAAH+AQD/AAEAAP4A/wAAAP0A/wIBA/7/AgEAAAH/AAEA/QD+AgEBAv//AgADAQAAA/7/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD9AAD+AgQBA/0AA/7+AAQFAwD8AgAEAAABAv3+/gAAAwH/Av7////9AQL+AQEC/v3+AQL///8AAf/9AgEBAv7+/AL7BQIHAgb7+QT9/f0GBwP++gMCB/79Agf6/f77BQX+AP4CAQAA/f7/Af///QICAf8AAf8CAP8BAv7///8CAgH+AQD//v8AAQIAAQEDAf/+Av//AP8BAv/+AgEBAv79/P/+AAADAfz/Af4BAQX///0C/gAAAAACAPwAAAD//v7+AgEEAgUBA/z9BP7+AQQFAv78AgIE//0CBfv+/v4EBf0ABQH//wD9AQP+AQID/f7/AQH/AP8BAP/7AQEBAv78AAD/AgABAAL/AAAAAAACAQAA/gABAQAAAAAAAAAAAAD+AAACAAMAAP79AAAA/wIC////AQACAAAAA/4A//4CAf8CAAQBAwIE//wF/v7/AQYC/vwBAQP//gEE/v///AMBAf4AAP7//wABA/0CAQX9/P4DAP///wAD//sBAQED/v4AAQIBAAABAgL+AgAA//0CAAD/AAD/AAAAAAEAAAAA/QP/////AAAA//7/AQH+/wD/Av4B/wEAAv/+/f8AA/7+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+AQECAQMA//7+AgD/AQMCAf/9AQAC/wABAAD//wD/AwEBAwIBBAEB/v0C//8CAwIB//4BAQL//wEC/f///wIDAAD+AAAAAAL/AQAAAAEB/wAC/wD/AAD/AP7/AAAA/wH+AQADAQMBBv34A///AgQDAQD8AgED//4BAv3+//oDAgEA/v8BAwD+/gAB///+AAAB//4BAQMAAAEA/////f8DAQAGAQABAAD//gEAAAUDAQAA/QEAAv//AAT/////AQEDAPwC/wH//v8D/gEB//8B/wP/AQD/AAIAAAIAAAAAAAAB/gABA////gEB///+AgMCAgAB/wMBAAED/v//AgAEAAD/AQAAAAAAAQEAAAABAwD//QEBAgAAAAMAAAAAAQAB/wP/AAD/AAEBAQAAAgMCAAECAAACAAIAAAAAAP///wP/AAH////9AQIAAQEA/v//Av0BAP8AAv8AAgAABP//Av/+//////0AAv8BAf7//v8C/wD//wAC//0BAAEBAP8AAP3/AP4A/gABAwAB/wD9Af//AAD/AAD//wEAAP3+/v4B/gEBAwED/vwD////BAIB/v4BAgL//wED/f///wIC/wACAQACAAQA/gAAAAICAP8A/QEBAf8AAAL/AAD/AAAD/wL//wD//gED/gEBA//+/gL/AAD/AAL//gEAAQD//vsC/AUDBwIG+/sF/f38BwcE/foEAgf+/AIF+f3+/AUH/wEAAgABAAQA/wEA//4CAgP+/QABAgD/AAD/AAD+/wH7Av0DAwYBBPv5Bf39BQUHBP78AgAG//0DBPr9/fwEBv4BAAIBAf8CAf4AAP/8AQMCAf4AAAL/AQH/AAD/AAED/QEB/QH+AAL/AgH///0AAAH+AgABAf/9AQL/AP8CAQMCAQL/////AwEA/wABAP4A/wL/AP8AAAEA/wEAAQD/AAAAAgABAgAA/v4B////BAEBAQABAAQA/gEB/f///wED/gIBAwEFAAP+/QH///0FAwP+/AICBAD9AQX8///9AwMBAQQDAAIABAD/AQAAAAIDAQD9AQAEAP4AAv8AAAABAQIAAv4ABAAFAAEB//8AA/4A//4CAQP//wD//wAA/wIDAQAD//8B/wQCAv4AAAD//wABAv8AAAEA//sBAQEB/v78Afv9AgIBAwD7BP//+gMBAwAAAQECAAADAfwA/v0CAwAAAgABAgAEAQIB/wD/BAECAP8BAAMAAAEA/wD/AQEE/wH//wEBAQIB/AMAAP4CAQMAAP8AAQEAAgH+AP8A/gEAAAAAAQb/AP0BAf///gQF///6AwEG//8BAv3//wMCA/4C/QQCBgEC/PwE/v79BgYD/fsCAgb//QIF+/7+/QQF/AAF/wAAAgf/BAD/AAcA//8AAAACAAEAAP8BAAADAAMAAAUAAAEA/P8DAP8ABgAC/wH+AAEBAAAA//8AAAT///8CAwQCBQEC/fsC//7/AgMBAPwDAAT//wEF/P7/AAIFAv8BAf/+APsCA/8BAf8BAf8C/v//Af8C//4CAQED/v4AAv8FAgYBBP39Av7+AAQDAf/6AwEE//4CBP3+/v8DBAP/AP//AP/8AQL/AAH/AgL9AwEA/wEBAgAAAgAAA/8A/gH/BQIFAQL9/AL//v4DAQIB/AIABP/+AgP9/v4BAQQAAAADAQQABf/+BAAA/wMGAf78AgAD//8BBv7///0BBP8B/gUCAwAF/v4D///9AwEB/vwBAQT//wED/v7//gMEAAD/AQH+AAMA/wIAAP4AAQAA/wAAAP8AAAEA/wAC/wH/APwEAQEAA/4ABP///QMCAQD8AQED//8BAv////4BAwH/AAEA/QABAAMBAQH+/wD/AQD///8AAAAAAgAAAv3/AAD9AgEBAAD/AAMAAP8BAwAA/wAAAv8AAQP//wD+/wL+Af4DAQQBAf8CAf///gIEAf39AgEE//8BBP7//gEBBP8B/gICBAEC/f0D//7+BAQC/vsCAQT//gEE/P7//gME/QH+AQEEAQT9/AP///0EAgP9/AIBBP/+AQT8///+AwQAAf0BAP8BAQH8AwAA//4AAv4BAAD/AAEB/v0AAP4A/vwB/wMCBAED/vsC///8BAMD/PwCAgP//QEC/f//+wMF/wD+/gABAAIB/AAAAPz//wAA/wAB/gD+AAAAAQD//wL+AP7/AAEAAf//AQAA+wECAf8A/wAAAAAAAAAAAP4BAQf//wD9/v/8AwL+AgIE/v7+AgL//vwBBP7+AgICBv39/gD/AwAF/wP+/wAAAP0DAwD//AECAv8BAAD//wACAQD+AP0CAAEAAAD+AAAAAAMBAv/+AAABAP8A/wAAAAEAAv/////////9AAX+AQH+/wL+AgEAAAAAAf/8AQABBP8A/gD+/gD/AAAC/AABAf8A/wD/Af8B/QD+/wABAQH9/gEAAAP+AgT//v0A//7/AAT///8AAAEE//4BA/////0DBP8BAAABAwAA/v0A//8ABAQC//0BAAL//gEE/f//AAAEAP8CAgAD//8B/QIBAP4A/v7//wD/AAABAAECAAD9AAIE/wIB//7//QID/QEC/v39/QIA//7+AQL//gMBAQT+/foC+AYDCAII+/gF/f35BwcF/PkDAwj+/AIH+f3++QUH/gH7AgECAgX+/QL///wCAgT9/gECAv//AQL+////AgL/AAACAQEA/AD+/wAAAAAD//7/AAEBAP4AAQAAAP0AAfwC+wUCBQED/PoE/v78BQID/foDAgb+/QIG/P7++wQDAAAB/wD9APwBAQAAAAT//v8CAAD/AAABAAABAAACAAD7AfsEAgQBAfv7A/39/AUCAv78AgIF//4CBfz+//wEAf4B/QQCBQAE/vwD//78BAUB/vsCAgX//gEF/f///AMF/wD9AAIAAAAA/wH///8BAQH+AQEAAgD+AQD/AP/+AQP9AfwEAgYABP37A//+/wQEA/36AwEF//8BBP7+//0DBf8A//8A/wD/AQD/AQEA/wD//wD8AQABAP4AAgAA/////gD9AwECAAL//QIA/wABAgH//wEBAgD/AQP////+AAL+AAACAQEAAv8AAv//AQIBAQD/AQACAP8BA/7///8BAv//AwIA//8BAAP/AQEEAgH+AgEB/wIAAgACAv8AAP4D/gD/AQIAAAD+AAL//wADAwL//AABA/4AAgP+////AgP+AP8AAgAAAP4AAv//AAICAv/8/wEC/gABA/7///8CAv8AAP8AAAD/AAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAA/QH9BAIGAQX8+gP+/v4GBQL++wICBf/+Agb7/v78BAX+Af0CAf8AAgH7Af///AQAAP/+AQH///8BAP8A//0CAgIAAwMBAf8CAP7/AP8ABAP/Af0B/wH//QAA/gD/AAID/QH/BAEEAQH9/AP+/gEEBAL9/QIBBf//AQT+/v/9AwT/APsBAAEAA/8AAAAA/QMB/wH+AQACAAAA/wAAAAAA/gAAAAAA/gD9AgH/AAAA//7/AQL/AP0AAQD+AwEBAv79/QEBBgMHAQT8+QT+/f8EBwP+/QMABgD8Agb6/v7/BAUAAf8CAf8B///+Av///gIBAf//AQEB/wABAf////8BAf8A/wMBBP8E/v8B//8BAwIBAfwBAQQAAAEA////AgAC/gD/BQEEAAP+/gL//gAFAgH//gIABf//AQP+////BP8AAf0AAP4BAgD8AgAA/wH///8AAAP/AAAA/gEAAP0A//8A/gIAAQADAP4BAP/9BAIA//4BAQEAAAD/AAAA/wAA/wL+AgEBAQP//QP///4CAQH+/gECAf//AQH////9AgEB/wADAAP/Av/+/wAAAgIAAAL+Af8D/v8A/gH/AAD+AQEBAAIBAgEA//4B//8BAgMA//4BAQIA/wAC/v///wEC+wH6BgIGAgb7+wT+/vsGBgT9/AMCBP/+Agf6/v77BQb/Af8DAAQAAgD8Av8A+wICAv/+AAECAP8AAf0A//0BA/4B/AMBBQED//wC///9BAUD//0BAQX//gIF+////wIF/QH+BAEFAQL++gP+//4EBQH+/AIBBP/+AQP7///9BAP+AP0C/wIABAEA/wAA/gAA/v/+AAECAAH/BAEAAf8AAf4B/wQCAwIC/vwE////BAQCAP0BAQX//wED/////wIDBP8E/f78//0CBf0CAgP9/P0DAv/+/QEC//wEAQEE/f0AAf/+AQIBAf/8Af///wICAQEBAP8CAP8BAAAA//0AAv8C//8BBAIC/voC/v//AQIC/wH/AQMA/gEA/QL/+wAC/gH9AAECAf/9/QEA//0DAQL/AP8BBAD/AQIA/wD+/wQBAQICAAEAAAL+AgAB/gD/AP8B/wAAAAAAAAAAAP8A/wD/A/4AAwABAQIAAAAABAIA/wEA/wIA/gAEAAAAA/4DAAEAAQEBAAD//wH//wABAQD//wEAAf8AAQH///8AAQEBAAAA//8AAAEB/wAAAP8A/wAA/wAAAAAAAAEAAAD/AP8B/gMBAwEB/v0C///+AwMB//0BAAP//gIC/f///gID//8D/f/9AgMCBP4AAQX9/v0CA/8C/AEA//0CAQEC/v4AAfoAAAECA//9Av8A/gICAv/+AAIBAP4A//0AAPwBAf0BAv8BAgED/gAB/v8BAgIB/wEAAQIA/gEC/v//AAICAAH7AQH+AQT//wH///4CAgEA/wECAQD/AQD9AP/8AgH/Af0BAQMAAv36AQAA/QQDAf79AQAC/wABAf7///wBAgIB/QIBAgEA//8B/v/8Av8A//4BAAH//gEE/f///QIC/gD9/gH/AAL++wH///wDAQH7AgAA/wD/Af/9///9AwEAAf4CAQIAAv78Av///gICAv/+AQEC//4BAv3///4CAQT/BgD+//8AAQX9AQIF/QD9AAAA/wAAA///BQACBQD/AAD9/gH/AAD/AgAAAAABAQEBAv8AAv8AAQIA//8A/wIBAP/+AP4A/v8B/wABAgAAAAEB//8BAAAAAAD//wH/AQEAAAEAAQABAP8BAAD/AAAB//8AAAAAAAD//wAA/wD/Av8C/v/+//4CA/4BAQL+/v4CAv///QEC//4CAQEC/v4D//7+//7//QMD/gEB/v0C/QP/AAH/AQL//gMBAQT9/QL/Af7////+AgP+AQECAAH/AgD//wABAf8AAgEBA/7+AAABAwEDAAD+/gL//wADAQEB/QEABP8AAQP9//8BAgP/AAL/AP4CA///Af3/////Af4B/wAAAP4AAP3///3/A/4BAQEAAQEB//8B//8CBQAB//4BAQEA/wD//gAA/gEAAwAD/gAA//wAAP8AAAT+AP4BAAD//wACAAAAAAADAAD+AQABAQMCA/79Af//AQMAAf78AQEA//4BAv4A//8CAP4AAAAA/wEC/wEB///9/wEB/gAAAQAA/wAB/gAA/gACAQACAgADAAL/AAAAAAEBAQAA/QEAAP8BAAH///8BAQIBAf8AAQEAAP//Af8AAQACAAD/AQABAAABAv7/AAIBAv8B/gAB/wEAAP4B///+/wIB/wEBAAIA/wEC/gH//wAB/wECAAH+AQD//wEA/wEEAAH//gEBAAD/AAD+AAD+AP8AAP8BAAEBAAAAAAAAAAEBAf//AQEBAAAAAf8A/wAAAf8B/QMBBAH//fwD/v//AwQC/v0CAgP//gED/f7+/wMEAQAB/wD/AP8BAf8AAAH///8BAQAA/wAAAP8BAAAB///+Af4CAQIAAf/+Af///gICAf/+AQEC//8BAv7///4BAv0C/AQCBQID/PsE/v77BgYD/fsCAQb//QIE+/7+/AQF///+/AD/AAAC/v8BAf7//P8AAv8A/AEA//0DAQH9/gAAAQACAQIAAP7+Av3/AAIEAAD9AgAE/wACBP7//gADBAAA/wAA/wABAP0A/wD//gEAAAAAAP4A/wABAAEA///+/QH9AwIEAQX8+wP+/v0EBAP9/QMCBf/9AgL8//78AgQCAQAAAAIAAQAE/gEA/QEA/v4AAQICAAH//wEBAP/9/AAAAP///gABAgP+AAH//f0AAAL/AP0AAf8BAwEB/wH+/gH9AgECAQH//QH///4CAwH//gEBAgD/AQL+AAD9AQIE//4A/f3/AQIE/wMBBAD9/gIB/gH/AP///AMAAQQBAP8B/wMBAgAA/gAC//8AAgMBAAABAAMAAAEC////AAADAAABAgIDAAH9/AEA/gICBAAA/QH+AwD+AQL+//8AAAMB/wICAQMAAP7/AAD/AgEDAAH+Af4CAP8AAgAA/wH/AgEA/gMAA//+/wEAAAD/AQT+Av0C/gIAAQACAQAAA/4C/wEB/gECAQD//wH//wH/AgAA/gEAAf//AQL/Af///gEBAAH///8A/wEB/wEBAf///wEB////AAH//wEBAAH///oC+AcDCAII+/gG/f35CAgF/PkEAwj+/AMI+f39+QUI/wL8AwIFAQP9/AT//vwFBAP+/QIBBf/9AQX8///8BAX9Af8CAQIBAP/9Av///wEEAv/+AQED//4BAv0A//0CAgIAAAQBAv8D//wBAAABAwMBAfwCAQT//gADAP8A/wEC/wEAAgEBAQMA/QL+//0CAQL//gEBAv//AgP/////AwMA/wMAAP///gEB/wAAAf7/AAAAAP/+AAAA/gAAAAIA//8AAQD//wD+AQH/AAED/f4AAAEAAP0AAQD8AAAAAAD8/gH9AgIEAQH9/QL+//0DBAH+/QIBA//+AwL8/v/+AgEA/wIA/////QACAAEBBP7+/gEA////AAH//wIAAAIA//8B/wEBAQAA//4B////AQIB//8AAQAA/wEB/wAA/wEBAAAB/gAAAPwAAwABAQT9/f8B/wD//wACAP8BAAAC////Av4DAgMB//38Af3//wQEAP78AgED/wACAf3+//4AAQEAAQIAAAAD/wAAAAECAP//AP8A//8AAQAAAAAAAQIA/gH+AAIAAAD+/QL//wADAgL+/gABAv4AAQP+///+AgIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP4B/gIBAwEC/v4B///+AwIB//4BAQL//wED/v///gID/wD8BAEEAAH9/gL///8FBAIA/AIBBf/+AQL9/v//AwYBAAH/AP4A/wAB/wAAAf///wECAAD/AAEA/wEAAAL//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v/+AAABAv4BAQAAAP8CAv//AAAA//4CAAEB//7/AAECAAAAAgAB/wAAAAEB/wH/AAABAAEAAQEAAP//Af0B/QMBAgEA/v4B///+AgEB//4CAQL//wED/wD//wID/gH+AgECAQL+/gL///4CAgL+/gEBAv//AQL+///+AgIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP8B/QMBBAAD/v0C///9BAMC//0CAQT//wEE/f///QME/QH8AwEEAAL+/QL///4EBAL+/gEBA///AQT+///9AgT/AP4AAAAAAAD+AAD//wECAP4AAAIAAAAAAAEAAP4A/wAAAAAA/wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP8A/wEAAQAB//8BAAD/AQEB//8BAAEA/wAB/wAA/wEB+wL5BgMHAgX7+QX+/fkHBwT8+gMDB/78AwX6/f75BQf/Af8AAP8ABAD9Av7/Af37Af4AAAEA//8A+/4CAP4A/gABAAAAAwEAAP0B//8AAAQD//4AAAAA/gAA/gAA/QAAAQAA/gD/AP8BAgABAAD///8BAAD/AAEBAP4BAAAA/wECAAP//////gIC/wEBAfz+/gID///+AAL//gMBAQIA/wP/Av3//f/9AgT+AQEC/Pz+AgP//vwBAv/9AwEBA/7+AAAAAAAAAAAA/wAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP4B/gIBAAEC/v4C///+AgIC/v4CAQL//gEA/v///gIC/AH8AQEDAQD9/wH+/v0EAwL//wEBBAD/AQP//wD+AAQBAAUA/wH//gID/wECAv39/v8BAAD8AAL//gMAAf8A//4B/wABAwEA/v4C/v//AwMCAAAAAAMA/wED/wD/AAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/gH+AgECAQL//gH///4CAgH//gEBAv//AQL+///+AQIAAAIA/wD//AEBAAEBAP///wEAAAD/AAEAAAIAAAAA/gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/wT//wH//AIE/QEBBAEA/QIA/v8BAQL//wMCAQT9/wD/AAAA/f8AAAL/AAAAAP3+AgIAAP4AAgAAAgAAAgAAAf4DAP7//gAAAv8BAAT8/v8DAf7+/wAC/QACAgEEAP/8AP0DAgEBAAD8Av/+/wADAf79BAEC//8CAP0A/v7/AwEAAQAAAAAAAQH/AAABAAD/AQD/AAAAAQAAAQAAAf8A/QEAAwEAAQL+/QL//wAAAwL+AAIBAP/+AQL9///9AgMBAAAAAP8A/wEC/wAAAP///wEB/wD/AAEA/wIAAAL//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+Af4CAQEBAv7+Av//AAAAAf8AAQAA//8BAf7+//8BAv4A/wD//gEBAgAAAQH+/v8A/wH/Af4BAP//AgEBAP/+AAAAAAD/AAAAAAAAAAAA/wAAAAAA/wAAAP8AAAAAAAABAAEAAQEA//8AAP//AgEAAAAAAAABAAAAAf///wABAQEBAAABAQAA//8B//8BAgEBAAABAAEAAAEB/v///wEC/gH/AAH/AQAA/QL//wD/AQH/AgAA/gD/Af/+AP/+AgD/AP8AAAAAAQAA/wEA/wD/AAD/AAABAAAAAAAAAAD/AAEAAQH/AQAAAAH/AQEBAQAAAP4AAAIAAP8BAQABAf8AAv8BAP8AAAABA/4BAQD///8AAP8AAAEB/wADAQEB/v//AAEAAAAB/wAAAAAAAQEAAP8BAAEAAP8AAf8AAP8AAR4AIQAfACAAIAAfACIAIAAhAB8AIAAgACAAHwAeAB8AIQAfACEAHwAiAB8AIQAfACAAHwAhACEAIAAfACAAIAAOAP7/DwD1//3/9P/9//b/BgANAPv/BgAFABAA8//1//v/DAAIAPn//f/z/wQACAD9//T/CAAFAAUAEAD3//X/jQAAAA==",
    };
    // NETWORK END

    // ------------------------------------------------------------------ alpha-beta search

    /** The score of the position for the side to move, searched to the given depth within (alpha, beta). */
    int negamax(int depth, int alpha, int beta, int ply) {
        if ((++nodesSearched & 1023) == 0 && System.nanoTime() > deadlineNanos) outOfTime = true;
        if (outOfTime) return 0;

        int mover = sideToMove;
        if (hasLine()) return ply - WIN; // any line wins for the last mover: the opponent
        if (quarkCount == cellCount) return 0;

        // Forced moves, from the playable threats (cells at a column's top that complete a line).
        int forced = NOT_FORCED;
        if (ply > 0) {
            int opponentThreats = 0;
            for (int column = 0; column < columns; column++) {
                if (completesLine(mover, column)) return WIN - ply - 1; // we complete a line next move
                if (completesLine(1 - mover, column)) {
                    opponentThreats++;
                    forced = column;
                }
            }
            boolean mayBomb = bombCooldown[mover] == 0;
            if (opponentThreats >= 2 && !mayBomb) return ply + 2 - WIN; // a drop blocks only one of them
            if (opponentThreats >= 2) forced = BOMBS_ONLY;
            if (opponentThreats == 1 && !mayBomb && depth == 0) depth = 1; // search the forced block, don't evaluate
        }
        if (depth == 0 || ply == MAX_PLY - 1) return evaluate();

        long key = positionKey();
        int slot = tableSlot(key), tableAction = NO_ACTION;
        if (tableKey[slot] == key) {
            long entry = tableEntry[slot];
            tableAction = entryAction(entry);
            if (entryDepth(entry) >= depth) {
                int score = scoreFromTable(entryScore(entry), ply), bound = entryBound(entry);
                if (bound == EXACT || bound == LOWER_BOUND && score >= beta || bound == UPPER_BOUND && score <= alpha) return score;
            }
        }

        int[] actions = actionLists[ply];
        int actionCount = orderedLegalActions(actions, tableAction, forced);
        int originalAlpha = alpha, bestScore = -WIN - 1, bestAction = NO_ACTION;
        for (int i = 0; i < actionCount; i++) {
            int action = actions[i], score;
            if (isBomb(action)) { // a blast rearranges several columns: save the whole state
                if (undoStates[ply] == null) undoStates[ply] = new int[stateSize];
                saveState(undoStates[ply]);
                play(action);
                score = -negamax(depth - 1, -beta, -alpha, ply + 1);
                restoreState(undoStates[ply]);
            } else {
                int column = columnOf(action), previousCooldown = drop(column);
                score = -negamax(depth - 1, -beta, -alpha, ply + 1);
                undoDrop(column, previousCooldown);
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

        int bound = bestScore <= originalAlpha ? UPPER_BOUND : bestScore >= beta ? LOWER_BOUND : EXACT;
        tableKey[slot] = key;
        tableEntry[slot] = packEntry(scoreToTable(bestScore, ply), depth, bound, bestAction);
        return bestScore;
    }

    /** {@link #orderedLegalActions}'s {@code forced}: any drop may be played, none (bombs only), or only the column given. */
    static final int NOT_FORCED = -1, BOMBS_ONLY = -2;

    /** Fills {@code actions} with the legal actions, best guess first, and returns how many there are. */
    int orderedLegalActions(int[] actions, int preferred, int forced) {
        boolean mayBomb = bombCooldown[sideToMove] == 0;
        int count = 0;
        for (int column : columnsCentreOut) {
            if (columnHeight[column] == rows) continue;
            if (forced == NOT_FORCED || forced == column) actions[count++] = action(column, false);
            if (mayBomb) actions[count++] = action(column, true);
        }
        int[] history = historyScore[sideToMove];
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

    /** Resizes the table to 2^bits entries of 16 bytes each, forgetting what it held. */
    public void resizeTable(int bits) {
        tableKey = new long[1 << bits];
        tableEntry = new long[1 << bits];
    }

    int tableSlot(long key) {
        return (int) key & tableKey.length - 1;
    }

    /** The best action stored for a position, or NO_ACTION. */
    int tableAction(long key) {
        int slot = tableSlot(key);
        return tableKey[slot] == key ? entryAction(tableEntry[slot]) : NO_ACTION;
    }

    // An entry packs the score (32 bits), depth (8), bound (2) and action + 1 (the rest) into one long.
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
