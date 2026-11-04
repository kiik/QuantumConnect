package ch.qc.meelik.bots;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.Move;
import ch.cern.quantumconnect.core.QuantumField;
import ch.cern.quantumconnect.core.SpaceOwnership;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * SerenaBot: QuarkBot's alpha-beta search, plus a loss book learned from the tournament: the bot
 * that learns from its own lost games.
 *
 * <h2>The loss book</h2>
 * Every revision of SerenaBot is uploaded to the tournament as {@code SerenaBot-DD-HHMM}. Its lost
 * games are fetched from the site ({@code qcsite losses}) and replayed offline ({@code qc.sh learn}).
 * At each of our moves in the run-up to the loss, a search with far more time than the arena allows
 * looks for a better move; where it finds one that clearly beats the move we played (above all one
 * that escapes a forced loss), the position and that move go into the book. The book is built into
 * the class ({@link #LOSS_BOOK}), so the next revision plays the better move at once, without
 * searching, whenever a game reaches a booked position again. Tournament opponents are bots, mostly
 * deterministic, so the same lines recur. The book only grows: each revision keeps every lesson of the
 * ones before it (params/serena/book.txt holds them, with where each came from).
 *
 * <p>Positions are keyed by {@link #positionKey}, from our side (we are player 0) with both bomb
 * cooldowns, and the board size is part of each entry, so a book learned on the tournament's board
 * never fires on another size.
 *
 * <h2>The rest is QuarkBot's</h2>
 * Alpha-beta search on a model of QuantumConnect (Connect-4 with extra steps), with line
 * weights evolved by self-play. It is one self-contained class, the game model, the search and the
 * arena interface together, so the upload JAR holds nothing of ours but this class. It uses only
 * the engine API and java.lang/java.util: the arena forbids java.io, java.net, reflection and the
 * like, and the bot loses on any forbidden access (java.lang.System is allowed since 2026-11-04,
 * so this bot keeps time with System.nanoTime). No nested, anonymous or lambda classes either:
 * lambdas compile to java.lang.invoke calls, which the sandbox may count as reflection.
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
 * {@link #EMPTY}. On the bot's own board, player 0 is always this bot ({@link #ME}). A move, called
 * an action, packs a column and the bomb decision: {@code action = column * 2 + (bomb ? 1 : 0)}.
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
 *   <li>a win scores {@code WIN - ply}, so faster wins are preferred and slower losses resisted;
 *   <li>forced moves, when threats are tracked (the arena's bot does): a side that can complete a
 *       line wins; one facing two playable threats without its bomb has lost; one facing a single
 *       threat may only block it or bomb, and at the search's horizon that block is searched
 *       rather than the position evaluated, so a forcing sequence isn't cut off halfway.
 * </ul>
 *
 * <h2>Talking to the arena</h2>
 * Each turn the arena calls {@link #accelerateQuark} (pick a column) and then {@link #useSpecialMove}
 * (bomb or not), both before anything is placed. The bot decides both at once, as one action, and
 * remembers the bomb decision between the two calls. It counts its own bomb cooldown, and the
 * opponent's from the move history, which marks every bomb. Every turn counts toward a cooldown,
 * also one lost to a timeout. The arena may use another board size than the standard 8x12: on the
 * first look at a field of another size, the bot rebuilds its board to that size, with the same
 * rules and line weights ({@link #fitBoardTo}).
 *
 * <h2>Also a plain board</h2>
 * The tools use SerenaBot as their game model too: {@code new SerenaBot(rows, columns, ...)} is an
 * empty board to play actions on, which can also search.
 */
public final class SerenaBot implements Algorithm {

    // ------------------------------------------------------------------ rules

    public static final int EMPTY = -1;
    /** Own turns from one bomb to the next. */
    public static final int BOMB_COOLDOWN = 10;
    /** Own turns before the first bomb: the arena's counter starts at 0 and allows a bomb at 10. */
    public static final int FIRST_BOMB_WAIT = 10;
    /** Quarks in a row that win, on every board size. */
    public static final int LINE_LENGTH = 6;

    /** Line directions as (column step, height step); both diagonals share a weight by symmetry. */
    static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    public static final int HORIZONTAL = 0, VERTICAL = 1, DIAGONAL = 2;

    // ------------------------------------------------------------------ evaluation settings

    /**
     * Value of a line holding 0..6 quarks of one player and none of the other, per direction class,
     * evolved by self-play ({@code ./qc.sh tune}; the run is recorded in params/evolved-d4.txt).
     * Against the hand-written weights (16, 64, 256, 1024, 8192) it scored 65% at the real time limit.
     */
    static final int[][] EVOLVED_LINE_WEIGHTS = {
            {0, 41, 131, 241, 540, 6367, 0},
            {0, 17, 75, 204, 658, 8300, 0},
            {0, 15, 105, 261, 615, 7863, 0}};
    /** Static-evaluation bonus for having the bomb ready while the opponent doesn't. */
    static final int EVOLVED_BOMB_READY_BONUS = 20;

    // ------------------------------------------------------------------ search constants

    public static final int WIN = 1 << 28;
    public static final int ME = 0, OPPONENT = 1;
    public static final int MAX_PLY = 64;
    public static final int NO_ACTION = -1;
    /** After this much thinking in one match, search with an eighth of the budget (the match limit is 30 s). */
    static final long MATCH_TIME_GUARD_NANOS = 20_000_000_000L;

    // ------------------------------------------------------------------ board geometry (set per board size)

    public int rows, columns, lineLength, cellCount, lineCount;
    public boolean bombsEnabled, tracksThreats;
    /** Columns ordered from the centre outwards: the usual best-first guess for move ordering. */
    public int[] columnsCentreOut;
    int[][] cellsOfLine, linesThroughCell;
    int[] lineDirection;

    /** Value of a line holding n quarks of one player and none of the other, per direction class. */
    public int[][] lineWeight;

    // ------------------------------------------------------------------ position

    /** Owner of each cell: player 0, player 1 or EMPTY. */
    public int[] owner;
    public int[] columnHeight;
    /** Own moves each player must still make before they may bomb. */
    public final int[] bombCooldown = new int[2];
    /** Number of complete lines each player has; above zero means that player has won. */
    public final int[] completedLines = new int[2];
    /** Quarks of each player in each line, at {@code 2 * line + player}. */
    int[] lineQuarks;
    /** Lines in which a cell is the only one a player still needs, at {@code player * cellCount + cell}. */
    public int[] threatCount;
    public int sideToMove, quarkCount;
    /** Sum of the line values, from player 0's point of view. */
    public int staticScore;

    // ------------------------------------------------------------------ hashing and saving

    long[] zobristQuark, zobristCooldown;
    long zobristSideToMove;
    long boardHash;
    int[] fallingQuarks;
    int stateSize;

    // ------------------------------------------------------------------ search settings

    public int bombReadyBonus = EVOLVED_BOMB_READY_BONUS;
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
    int myCooldown = FIRST_BOMB_WAIT, opponentCooldown = FIRST_BOMB_WAIT;
    int myTurns;
    long timeUsedNanos;

    // ------------------------------------------------------------------ construction

    /** The standard QuantumConnect board: 8 rows, 12 columns, six in a row, bombs on, threats tracked. The arena calls this one. */
    public SerenaBot() {
        this(8, 12, LINE_LENGTH, true, true);
    }

    /** The standard rules (six in a row, bombs on) on a board of another size. */
    public SerenaBot(int rows, int columns) {
        this(rows, columns, LINE_LENGTH, true, false);
    }

    public SerenaBot(int rows, int columns, int lineLength, boolean bombsEnabled, boolean tracksThreats) {
        setUp(rows, columns, lineLength, bombsEnabled, tracksThreats);
    }

    /**
     * Builds an empty board of this geometry and sizes the search memory to it, forgetting what it
     * held. The line weights are the evolved ones for six in a row, the hand-written formula for
     * other line lengths.
     */
    private void setUp(int rows, int columns, int lineLength, boolean bombsEnabled, boolean tracksThreats) {
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
        for (int d = 0; d < 3; d++) lineWeight[d] = lineLength == LINE_LENGTH ? EVOLVED_LINE_WEIGHTS[d].clone() : defaultLineWeights(lineLength);

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

        for (int ply = 0; ply < MAX_PLY; ply++) {
            actionLists[ply] = new int[actionCount()];
            undoStates[ply] = null;
        }
        historyScore = new int[2][actionCount()];
        Arrays.fill(tableKey, 0);
    }

    /** The cells of the line starting at (column, height) in a direction, or null if it leaves the board. */
    private int[] lineFrom(int column, int height, int[] direction) {
        int lastColumn = column + (lineLength - 1) * direction[0], lastHeight = height + (lineLength - 1) * direction[1];
        if (lastColumn < 0 || lastColumn >= columns || lastHeight < 0 || lastHeight >= rows) return null;
        int[] line = new int[lineLength];
        for (int i = 0; i < lineLength; i++) line[i] = (column + i * direction[0]) * rows + height + i * direction[1];
        return line;
    }

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

    /** The hand-written weights: 16, 64, 256, ... for 1, 2, 3, ... quarks, doubled for a line one quark short. */
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

    // ------------------------------------------------------------------ the arena interface

    @Override
    public int accelerateQuark(QuantumField field) {
        long started = nanoTime();
        SpaceOwnership[][] grid = field.getField();
        if (myColour == null) myColour = whoAmI(field, grid);
        fitBoardTo(grid);
        load(grid, myColour);
        updateOpponentCooldown(field.getMoveHistory());
        myTurns++;
        bombCooldown[ME] = myCooldown;
        bombCooldown[OPPONENT] = opponentCooldown;
        int booked = bookAction(positionKey());
        plannedAction = booked != NO_ACTION && isLegal(booked) && (bombCooldown[ME] == 0 || !isBomb(booked))
                ? booked
                : chooseAction(timeUsedNanos > MATCH_TIME_GUARD_NANOS ? budgetNanos / 8 : budgetNanos);
        timeUsedNanos += nanoTime() - started;
        return columnOf(plannedAction);
    }

    @Override
    public boolean useSpecialMove(QuantumField field) {
        if (plannedAction == NO_ACTION) return false;
        play(plannedAction); // counts the turn toward our cooldown, and resets it after a bomb
        myCooldown = bombCooldown[ME];
        return isBomb(plannedAction);
    }

    /** Rebuilds the board if the arena's field has another size: same line length, bomb rule and weights. */
    void fitBoardTo(SpaceOwnership[][] grid) {
        int fieldRows = grid.length, fieldColumns = grid[0].length;
        if (fieldRows == rows && fieldColumns == columns) return;
        int[][] weights = lineWeight;
        setUp(fieldRows, fieldColumns, lineLength, bombsEnabled, tracksThreats);
        setLineWeights(weights);
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
        if (last != null && last.player() != myColour && last.isSpecialMove()) opponentCooldown = BOMB_COOLDOWN - 1;
        else if (opponentCooldown > 0) opponentCooldown--;
    }

    /** Starts a new game on an empty board, forgetting the match and all search memory. Weights are kept. */
    public void newGame() {
        clear();
        Arrays.fill(tableKey, 0);
        for (int[] scores : historyScore) Arrays.fill(scores, 0);
        myColour = null;
        plannedAction = NO_ACTION;
        myCooldown = opponentCooldown = FIRST_BOMB_WAIT;
        myTurns = 0;
        timeUsedNanos = 0;
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

    // ------------------------------------------------------------------ board queries

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
        for (int d = 0; d < lineWeight.length; d++)
            for (int n = 0; n < lineWeight[d].length; n++) lineWeight[d][n] = weights[d][n];
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
        for (int i = 0; i < part.length; i++) state[at + i] = part[i];
        return at + part.length;
    }

    private static int copyFrom(int[] state, int at, int[] part) {
        for (int i = 0; i < part.length; i++) part[i] = state[at + i];
        return at + part.length;
    }

    // ------------------------------------------------------------------ the loss book

    /**
     * Positions from lost tournament games, each with the move a deep offline search prefers. One
     * entry per {@code ;}: {@code rows x columns, position key in hex, action} (column, {@code b} for a
     * bomb). Generated by {@code qc.sh learn} from params/serena/book.txt; split into chunks because a
     * class-file string constant holds at most 64 KB. Do not edit by hand.
     */
    // LOSS BOOK BEGIN (generated by qc.sh learn)
    static final String[] LOSS_BOOK = {
            "15x30 932b0e9364d6fe0c 18;15x30 92382d2081853b72 12;15x30 3d1c8eea0249f64f 26;15x30 3884c5e144be7c05 11;15x30 4421266e6206a710 12;15x30 e36afd9b38c0a153 14;15x30 7413ca505f75a080 5;15x30 1b239c79eae243af 13;15x30 0c9c0ff82d3b4c69 12;15x30 ce407cc4bc0bb54e 12;15x30 5a524f0c377d069e 11;15x30 24adcc89ea8396c6 2;15x30 73b66f65fd39ca77 2;15x30 e6b989518f1e743e 2;15x30 797a53d710f909fd 11;15x30 ccc5aa438dd1f6a6 11;15x30 f720a2910eecf2d9 11;15x30 4bdf36e477c5c089 15;15x30 34e4b228f1a7f79a 11;15x30 10484206bcd36e4a 8;15x30 78cebae845423dbc 15;15x30 fa5162239353f4e7 10;15x30 368a813bcf491982 1;15x30 805f44e1a039031d 20;15x30 15c078bd63e6fb19 17;15x30 1a26907beafff58b 17;15x30 8b920d506b2933a7 17;15x30 fa328e40b5dcc236 17;15x30 0cc9ee5c0e331e75 17;15x30 3796ccd014f04a99 17;15x30 0e08818d513da5c0 12;15x30 dd8965f7963022b9 11;15x30 5520b702e64c98e8 12;15x30 586c35e777a122bc 12;15x30 f0b11f8769eaa894 12;15x30 4a94ea8a168cb0f9 9b;15x30 786a03bbf68b126f 9b;15x30 2c451f03b44b58d5 9b;15x30 78c3a124fd2793e7 9b;15x30 011041f74c5be783 5;15x30 ce0554a97b050072 22;15x30 b51e2e57441e0577 9b;15x30 5708a12ed49e9267 12b;15x30 1a687bd0ff7a4d6a 12b;15x30 2b6cea86b7c54e68 12b;15x30 dc7b4c52f259b75b 11;15x30 a30e16801d5f2be6 11;15x30 d687323e2dcbc0ea 18;15x30 3397b98bcc4a8189 12;15x30 2482829221816e5d 21b;15x30 7f8c76e3b209f675 29;15x30 a95304f456a10b80 11;15x30 4b9de0477d974315 23;15x30 d3bd696ce15476e2 8;15x30 527c17bed4f1b349 7;15x30 4f07809c2071110c 7;15x30 d18019c2ba79594c 7;15x30 c1b9715e3e3cef71 1;15x30 2cbcc50822cf9f5f 4;15x30 a9adec14bad53fd2 4;15x30 8a5d04177527435a 10;15x30 d2c5538426401b51 10;15x30 d56ed27ded32ff94 18;15x30 3e47bb238001d260 3;15x30 6920751cd12c9972 9;15x30 b9c8e6cb108f0c98 10;15x30 48a1ada210707c63 10;15x30 b2390363237ff708 10;15x30 f7855c4cf43fc461 10;15x30 736fcebf5f64cafc 10;15x30 68d0f8a7c0eea675 4;15x30 9582c92cfc84f49c 3;15x30 74e29a4657058cac 18;15x30 87405f6517898d0c 3;15x30 3361e73adb8d6ff7 12;15x30 1d23ce6425714c39 26;15x30 6cebc2009d310cac 1;15x30 473e9e169f646fdf 2;15x30 980bb0ea879eb4f3 10;15x30 27830eeee972fc74 10;15x30 63049895dbeae80a 10;15x30 b7c56ae70017dc30 12;15x30 a44c5b4e7f560510 16;15x30 e344e11136eb070b 11;15x30 8ef9f92715a8a248 8b;15x30 a6d41f1d66fc6d45 10b;15x30 2f42ab1f70680d13 10b;15x30 a38787b2a4699a35 10b;15x30 a31b310bae1efd05 10b;15x30 4ffbc6ea7204986f 8;15x30 bc7e14be4f98257a 2;15x30 8c44dd088487cdc0 25;15x30 4c4efdb6d5417524 22;15x30 7ff1e4cbe06e2136 13;15x30 e68e57eb0c211583 22;15x30 fbcfdfee7ef4c572 19;15x30 215ddcc755d0dcee 26;15x30 7494c587b681284b 7;15x30 5bed41f61c976f88 29;15x30 ed6c9c98da2d01b6 29;15x30 74bdf351f1a4cc9a 29;15x30 fc419ee31c8a4ab1 25;15x30 7dff2a727cd3036f 25;15x30 0e607aa438a375c9 13;15x30 17cfd63022929fc5 13;15x30 43626b6791a90d44 13;15x30 857e9aba1cd20059 24;15x30 ae3e1494580c286f 13;15x30 400a03055084e09a 3b;15x30 e590a2aef7b19906 3b;15x30 e8efd1d1058299f1 3b;15x30 1d22176943b0cadb 3b;15x30 f10c7e9a9961693e 3b;15x30 3ac0e82399d2d61d 27b;15x30 088fe34b8cf8480b 18;15x30 18ad00213e90b824 18;15x30 fc5d9432ff3b5237 13;15x30 62da4cf187783d34 19;15x30 b56d8a41c09abc60 13;15x30 b0edf2eb29b2840a 19;15x30 c178ec713b87e0a5 13;15x30 a870645ed8f980a5 19;15x30 2837b1a629cebd49 19;15x30 9164d6eab6cb1fa3 19;15x30 775780d3aeb97075 7;15x30 ce80d0a400d38e52 21;15x30 2475474b3716e209 17;15x30 f917272ccfe85e4e 20;15x30 c62a3c95b0a6eda1 13;15x30 bbb9a94190822fb9 22;15x30 8ebc28913c0da43c 7;15x30 35ed60136d90db44 23;15x30 19fc126a936282fd 21;15x30 a2b8663341a762c9 23b;15x30 442992c1bf991813 23b;15x30 6f1a7822f11f3cf6 17;15x30 d43856861b83ba54 17;15x30 d3aa2eba62bdae74 17;15x30 da0062d2131a4bf8 10;15x30 bee1cbf897100357 13;15x30 a23571a471f471e0 23;15x30 f1fb10e728ac5ce0 25;15x30 f00913ca67e265cd 12;15x30 4d21ff2d3822f10a 18;15x30 6eb80342934d5299 8;15x30 677ce1e5e27dcbb4 8;15x30 60876d221d3a0374 18;15x30 86c0e08a81a0206b 6;15x30 1b6f81d8788f6204 26;15x30 1f77a64b13ed6889 6;15x30 e5167f910444869d 6;15x30 012b6008af2ce4f2 18;15x30 685712cede5efbbd 13;15x30 14bc2ab951bfefb2 18;15x30 b966368b018fe6de 15;15x30 eaacf2138e1aad10 19;15x30 f24f741bc61412a2 19;15x30 fca8cac212ef0e76 15;15x30 1d1ab7b822a47682 18;15x30 a9baa12596b0997c 20;15x30 a6d6ee0dfb2d7e56 20;15x30 367aaf9b1693b105 8;15x30 4192e4d657a46dc7 24;15x30 537ec2f0150245cf 2;15x30 b74636b7ba975981 12;15x30 48407d1667116577 12;15x30 778ac39a656a00bb 11;15x30 ea1758d4a2be319c 10;15x30 e098cd14e7f14457 5b;15x30 ef3e5e60dd7984d5 5b;15x30 59aa6fa51fcc702f 5b;15x30 fb94f8675798ba69 5b;15x30 9f2596d5cb024d11 8;15x30 f7963e7312d6d085 8;15x30 d98be0de49621a3f 13;15x30 493b7f120871a15a 8;15x30 680bac500e369c0b 8;15x30 305419263fc02222 15;15x30 adc0d540012f184f 10b;15x30 bfd9979407c5aaaf 11;15x30 f3b9bbb050e7141d 11;15x30 edfd8f5f054ce234 9b;15x30 8efc2c5480363f14 15;15x30 ca3163e3bc4f1d9f 15;",
    };
    // LOSS BOOK END

    /** The book for this board size, sorted by key for binary search; built on first use per size. */
    long[] bookKeys = new long[0];
    int[] bookActions = new int[0];
    int bookRows = -1, bookColumns = -1;

    /** The booked action for a position of this board, or NO_ACTION. */
    public int bookAction(long key) {
        if (bookRows != rows || bookColumns != columns) loadBook();
        int at = Arrays.binarySearch(bookKeys, key);
        return at >= 0 ? bookActions[at] : NO_ACTION;
    }

    /** Parses the entries of LOSS_BOOK that are for this board size. */
    private void loadBook() {
        String size = rows + "x" + columns;
        int count = 0;
        long[] keys = new long[64];
        int[] actions = new int[64];
        for (String chunk : LOSS_BOOK) {
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
        }
        sortByKey(keys, actions, count);
        bookKeys = Arrays.copyOf(keys, count);
        bookActions = Arrays.copyOf(actions, count);
        bookRows = rows;
        bookColumns = columns;
    }

    /** An action written as the column, with {@code b} appended for a bomb: "14", "14b". */
    public static int parseAction(String text) {
        boolean bomb = text.endsWith("b");
        return action(Integer.parseInt(bomb ? text.substring(0, text.length() - 1) : text), bomb);
    }

    /** Insertion sort of the first count keys, carrying the actions along; books are small. */
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

    // ------------------------------------------------------------------ the clock

    /**
     * Now, in nanoseconds. The arena allows java.lang.System since 2026-11-04; QuarkBot, from before,
     * reads java.util.Date at millisecond resolution instead.
     */
    static long nanoTime() {
        return System.nanoTime();
    }

    // ------------------------------------------------------------------ choosing an action

    /** The best action for the side to move, by iterative deepening within the time budget. */
    public int chooseAction(long budget) {
        deadlineNanos = nanoTime() + budget;
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
        for (int column : columnsCentreOut) if (columnHeight[column] < rows) return action(column, false);
        return NO_ACTION;
    }

    // ------------------------------------------------------------------ evaluation

    /** Static evaluation for the side to move: line values and bomb readiness. */
    public int evaluate() {
        int score = sideToMove == 0 ? staticScore : -staticScore;
        if (bombsEnabled) score += bombReadyBonus * (isBombReady(sideToMove) - isBombReady(1 - sideToMove));
        return score;
    }

    private int isBombReady(int player) {
        return bombCooldown[player] == 0 ? 1 : 0;
    }

    // ------------------------------------------------------------------ alpha-beta search

    int negamax(int depth, int alpha, int beta, int ply) {
        if ((++nodesSearched & 1023) == 0 && nanoTime() > deadlineNanos) outOfTime = true;
        if (outOfTime) return 0;

        int mover = sideToMove;
        if (hasLine()) return ply - WIN; // any line wins for the last mover: the opponent
        if (isFull()) return 0;

        // Forced moves, from the playable threats (cells at a column's top that complete a line).
        int forced = NOT_FORCED;
        if (tracksThreats && ply > 0) {
            int opponentThreats = 0;
            for (int column = 0; column < columns; column++) {
                if (completesLine(mover, column)) return WIN - ply - 1; // we complete a line next move
                if (completesLine(1 - mover, column)) {
                    opponentThreats++;
                    forced = column;
                }
            }
            boolean mayBomb = bombsEnabled && bombCooldown[mover] == 0;
            if (opponentThreats >= 2 && !mayBomb) return ply + 2 - WIN; // a drop blocks only one of them
            if (opponentThreats >= 2) forced = BOMBS_ONLY;
            if (opponentThreats == 1 && !mayBomb && depth == 0) depth = 1; // search the forced block, don't evaluate
        }
        if (depth == 0 || ply == MAX_PLY - 1) return evaluate();

        long key = positionKey();
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

        if (useTranspositionTable) {
            int bound = bestScore <= originalAlpha ? UPPER_BOUND : bestScore >= beta ? LOWER_BOUND : EXACT;
            tableKey[slot] = key;
            tableEntry[slot] = packEntry(scoreToTable(bestScore, ply), depth, bound, bestAction);
        }
        return bestScore;
    }

    /** {@link #orderedLegalActions}'s {@code forced}: any drop may be played, or none (only bombs), or else the one column given. */
    static final int NOT_FORCED = -1, BOMBS_ONLY = -2;

    /**
     * Fills {@code actions} with the legal actions, best guess first, and returns how many there are.
     * When forced to block a column, the only drop is there; bombs stay allowed.
     */
    int orderedLegalActions(int[] actions, int preferred, int forced) {
        boolean mayBomb = bombsEnabled && bombCooldown[sideToMove] == 0;
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
