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
 * Quarks drop into columns and stack from the bottom. {@link #LINE_LENGTH} in a row (six so far),
 * horizontally, vertically or diagonally, wins; on a board narrower or lower than that, the lines
 * in that direction simply do not exist. Instead of a plain drop a player may bomb: the quark drops
 * and the 3x3 cells around it, itself included, are cleared, and the quarks above fall down. A
 * player may bomb once
 * every {@link #BOMB_COOLDOWN} of their own turns, the first time after {@link #FIRST_BOMB_WAIT}.
 * After every turn, a line anywhere on the board wins for the player who just moved, whoever's line
 * it is.
 *
 * <h2>The model</h2>
 * Cells are numbered column by column, {@code cell = column * rows + height}, with height 0 at the
 * bottom. Player 0 is always this bot. A move, called an <i>action</i>, packs the column and the bomb
 * decision: {@code action = 2 * column + (bomb ? 1 : 0)}.
 *
 * <p>A <i>line</i> is any run of {@link #LINE_LENGTH} cells. For every line the board keeps how many
 * quarks each player has in it, and updates the counts quark by quark. Everything else follows from
 * the counts:
 * <ul>
 *   <li>the win test: a count reaching LINE_LENGTH;
 *   <li>the evaluation: a line open to one player only is worth a weight to that player;
 *   <li>the threats: empty cells that would complete a line;
 *   <li>the fours: empty cells of open lines two quarks short (four of six), where a drop makes a threat.
 * </ul>
 * A bomb is carried out as single-quark removals and additions, so it needs no bookkeeping of its own.
 *
 * <h2>The search</h2>
 * Negamax with alpha-beta pruning, deepened one ply at a time until the time budget runs out. A
 * transposition table reuses positions reached by other move orders, and moves are tried best guess
 * first: the table's move, then moves that caused cut-offs before, then centre columns. The threats
 * force moves: a side that can complete a line wins; one facing two threats without its bomb has
 * lost; one facing a single threat must block it or bomb. Past the depth, the search follows only
 * forcing moves ({@link #forcingPlies}): drops that make a threat to complete at once, and the blocks
 * they force. So a threat, block, double-threat combination is seen however deep it starts.
 *
 * <p>The refinements of game engines make the search reach about one ply further in the same time
 * ({@link #refinedSearch}): principal variation search (after the first move of a node, the others
 * only ask "is it better?" with a null window, and are searched in full only when it is), late move
 * reductions (quiet moves late in the order are searched shallower first, in full only if they then
 * look better), and two killer moves per ply. A root move found better in an unfinished iteration is
 * played. Bombs keep their place by history: ordered after the drops, they cost twice the nodes, as
 * a blast is often the move that refutes.
 *
 * <h2>The clock</h2>
 * The arena skips a turn answered after its limit ({@link #budgetNanos} keeps to about 80% of it),
 * and its machine stalls where a laptop does not: a cold JVM on the first turns, a shared CPU later. So the search keeps well inside the limit, thinks
 * a quarter as long on the first turns ({@link #COLD_TURNS}), and watches the move history: an answer
 * of ours missing from it came too late, and halves this match's budget; each answer placed gives a
 * little back ({@link #noteLostTurns}).
 *
 * <h2>The network (off by default)</h2>
 * A small integer network can take over the evaluation ({@link #useNetwork}). Its inputs are 0/1
 * features seen from one player: own and opponent quarks, playable cells, own and opponent threat
 * cells, both bomb cooldowns in four buckets, and whether that player moved first. The first layer
 * sums the weight rows of the active features into an accumulator per player, kept up to date by
 * the two incremental updates as quarks come and go; a clipped ReLU and one output weight per hidden
 * unit give the score. The weights are trained offline (tools/qc-net) and embedded as base64.
 *
 * <h2>The pattern table (on by default)</h2>
 * The line weights see only how many quarks a line holds, and only in a line one player owns. The
 * pattern table values each line by its exact contents instead: each of its six cells empty, own or
 * the opponent's, 3^6 = 729 patterns per direction class ({@link #usePatterns}). It, the network
 * and the loss book were trained with six in a row and play only at {@link #TRAINED_LINE_LENGTH}.
 * A line's pattern is kept up to date like its counts, so the evaluation stays one table lookup per changed line. The
 * table never refers to a cell or to the board's size, so one table plays on any board. It starts
 * from the line weights and is trained offline (tools/qc-net, {@code qcpatterns}) on archived and
 * self-play positions of any size, then embedded as base64. Without a table the line weights play.
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
public class SerenaBot implements Algorithm {

    // ------------------------------------------------------------------ rules

    public static final int EMPTY = -1;
    /**
     * Quarks in a row that win, at least 3. The organisers announce it for each tournament and the
     * engine keeps it to itself, so it is set here; everything else follows from it. The board size
     * is read from the arena's field.
     */
    public static final int LINE_LENGTH = 6;
    /**
     * The line length the line weights, the pattern table, the network and the loss book were
     * trained for. At any other, the pattern table, the network and the book stay off and the line
     * weights are stretched to the new length ({@link #lineWeightsFor}).
     */
    static final int TRAINED_LINE_LENGTH = 6;
    /** Own turns from one bomb to the next. */
    public static final int BOMB_COOLDOWN = 10;
    /** Own turns before the first bomb. */
    public static final int FIRST_BOMB_WAIT = 10;
    /** Line directions as (column step, height step). */
    static final int[][] DIRECTIONS = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    /** Direction classes for the weights; both diagonals share one. */
    static final int HORIZONTAL = 0, VERTICAL = 1, DIAGONAL = 2;

    // ------------------------------------------------------------------ evaluation

    /**
     * Value of a line of {@link #TRAINED_LINE_LENGTH} cells holding 0..6 quarks of one player and none
     * of the other, per direction class, evolved by self-play.
     */
    static final int[][] LINE_WEIGHT = {
            {0, 41, 131, 241, 540, 6367, 0},
            {0, 17, 75, 204, 658, 8300, 0},
            {0, 15, 105, 261, 615, 7863, 0}};
    /** Bonus for having the bomb ready while the opponent hasn't. */
    static final int BOMB_READY_BONUS = 20;
    /** This bot's weights (0..LINE_LENGTH quarks) and bonus: the constants above, unless a tuning tool sets others. */
    public int[][] lineWeight = lineWeightsFor(LINE_LENGTH);
    public int bombReadyBonus = BOMB_READY_BONUS;

    /**
     * The trained weights for lines of another length, counted from the full end: a line one quark
     * short of complete takes the trained weight of a line one short (five quarks of six), two short
     * the weight of two short, and so on down. Counts with no trained match take the one-quark weight;
     * an empty or complete line is worth nothing. For the trained length these are the trained weights.
     */
    static int[][] lineWeightsFor(int lineLength) {
        int[][] weights = new int[LINE_WEIGHT.length][lineLength + 1];
        for (int d = 0; d < weights.length; d++)
            for (int quarks = 1; quarks <= lineLength; quarks++)
                weights[d][quarks] = LINE_WEIGHT[d][Math.max(1, quarks + TRAINED_LINE_LENGTH - lineLength)];
        return weights;
    }

    // ------------------------------------------------------------------ search constants

    /** Score of a won position; a win in n plies scores WIN - n, so faster wins are preferred. */
    public static final int WIN = 1 << 28;
    public static final int ME = 0, OPPONENT = 1;
    public static final int MAX_PLY = 64;
    public static final int NO_ACTION = -1;
    /**
     * After this much thinking in one match, think an eighth as long per move (a match allows 30 s).
     * It does not depend on the per-move limit: on 15x30 the 225 turns of ours that fill the board
     * stay under it up to a budget of 88 ms; past it, the rest of the match fits in the 30 s.
     */
    static final long MATCH_TIME_GUARD_NANOS = 20_000_000_000L;
    /** Our first turns run on a cold JVM, classes still loading and nothing compiled: they think a quarter as long. */
    static final int COLD_TURNS = 2;
    /** The least a turn thinks after late answers, and what each answer placed gives back. */
    static final long MIN_BUDGET_NANOS = 2_000_000L, RECOVERY_NANOS = 500_000L;
    /**
     * The search reads the clock once per this many nodes (a power of two). On a wide board a node
     * with a bomb ready tries a blast in every column, each saving and restoring the whole state,
     * so 1024 nodes could overrun the budget by 50 ms. A clock read costs about as much as 1% of a node.
     */
    static final int CLOCK_CHECK_NODES = 16;

    // ------------------------------------------------------------------ board geometry

    public int rows, columns;
    int cellCount, lineCount;
    /**
     * The cells of each line, lowest first (leftmost first when level), the lines through each cell,
     * and each line's direction class. Lowest first makes both diagonals read alike, so a diagonal and
     * its mirror image have the same pattern.
     */
    int[][] cellsOfLine, linesThroughCell;
    /** The place value of each cell in each line through it, 3^(index in the line), parallel to linesThroughCell. */
    int[][] placeInLine;
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
    /**
     * Open lines a drop at a cell would turn into a threat: lines two quarks short, holding
     * LINE_LENGTH - 2 quarks of a player and none of the other ("fours" when six win), counted at both
     * their empty cells, at {@code player * cellCount + cell}.
     */
    int[] fourCount;
    public int sideToMove;
    int quarkCount;
    /** Pattern of each line: the sum over its cells of 0 (empty), 1 (player 0) or 2 (player 1) times 3^place. */
    int[] linePattern;
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
    /**
     * Thinking time per move: about 80% of the announced per-move limit (40 ms at 50 ms), leaving
     * room for reading the field and for a stalling machine. The cold turns, the match guard and the
     * late-answer cuts all scale from it.
     */
    public long budgetNanos = 40_000_000L;
    /** Deepest iteration of the search; a tuning tool may fix it. */
    public int maxDepth = MAX_PLY - 2;
    /**
     * Drops are searched only in columns at most this far from a column holding a quark (a forced
     * block is always searched). On a 100-column board this keeps the search near the play.
     */
    public int dropReach = 2;
    /** Per column: whether a drop there is within {@link #dropReach} of a quark; refilled at each node. */
    boolean[] nearQuarks;
    /**
     * How many plies past the search depth forcing moves are still followed: drops that make a threat
     * to complete at once, and the blocks they force. 0 ends the search at its depth.
     */
    public int forcingPlies = 8;
    /**
     * Percent of {@link #bombReadyBonus} the evaluation counts; 100 counts it as trained. 600 keeps the
     * bomb for a better moment: on 15x30 at the 50 ms limit it beat 100 by 72 and 119 Elo in two runs
     * (results/sparring-2026-11-05.md), and the top bots win most games against us with a held bomb.
     */
    public int bombBonusPercent = 600;
    /**
     * Which bombs the search tries. 0: every blast that hits a quark. Above 0: only blasts that hit
     * or drop a quark of a line holding this many quarks of one player and none of the other, so the
     * bomb is kept for defence and for strokes, not spent on small gains.
     */
    public int bombLineQuarks = 0;
    /**
     * The search refinements of {@link #negamax} (on by default): principal variation search, late
     * move reductions ({@link #lateMoveReductions}), killer moves, and the previous iteration's best
     * move first at the root, played if an unfinished iteration finds none better. Off: plain
     * alpha-beta with the table and history ordering.
     */
    public boolean refinedSearch = true;
    /** Late move reductions, part of the refined search; off, the refined search reduces nothing. */
    public boolean lateMoveReductions = true;
    /** Two killer moves per ply: drops that caused a cut-off at that ply in a sibling position. */
    final int[][] killerActions = new int[MAX_PLY][2];
    /** Scratch space for {@link #orderedLegalActions}: the sort priority of each action in the list. */
    long[] actionPriority;
    /**
     * The best root action of the iteration under way, once one has been searched in full, and the
     * best of the last finished iteration, which the refined search tries first at the root.
     */
    int rootBestAction, previousBestAction = NO_ACTION;
    long deadlineNanos;
    public long nodesSearched;
    boolean outOfTime;
    /** Score and depth of the last completed iteration, for the side that was to move. */
    public int lastScore, depthReached;

    // ------------------------------------------------------------------ match memory

    SpaceOwnership myColour;
    int plannedAction = NO_ACTION;
    int myCooldown = FIRST_BOMB_WAIT, opponentCooldown = FIRST_BOMB_WAIT;
    int myTurns;
    long timeUsedNanos;
    /** Our answers so far that should be in the move history, and how many of them were not (too late). */
    int answeredTurns;
    public int lostTurns;
    /** This match's budget per move: {@link #budgetNanos}, cut by late answers ({@link #noteLostTurns}). */
    public long matchBudgetNanos;
    /** The player who moved first in this match: player 0 unless we are the second player. */
    public int firstMover = ME;

    // ------------------------------------------------------------------ construction

    /** The standard 8x12 board. The arena calls this one; the bot resizes to the arena's field. */
    public SerenaBot() {
        this(8, 12);
    }

    /** An empty board of the given size. Sparring partners extend the bot but only add to its setup. */
    @SuppressWarnings("this-escape")
    public SerenaBot(int rows, int columns) {
        setUp(rows, columns);
    }

    /** Builds an empty board of this size and sizes the search memory to it. */
    public final void setUp(int rows, int columns) {
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
                    if (DIRECTIONS[d][1] < 0) reverse(line); // lowest first
                    lines.add(line);
                    directions.add(Math.min(d, DIAGONAL));
                }
        lineCount = lines.size();
        cellsOfLine = lines.toArray(new int[0][]);
        lineDirection = new int[lineCount];
        for (int line = 0; line < lineCount; line++) lineDirection[line] = directions.get(line);
        linesThroughCell = invert(cellsOfLine, cellCount);
        placeInLine = new int[cellCount][];
        for (int cell = 0; cell < cellCount; cell++) {
            placeInLine[cell] = new int[linesThroughCell[cell].length];
            for (int k = 0; k < placeInLine[cell].length; k++) placeInLine[cell][k] = placeOf(cell, cellsOfLine[linesThroughCell[cell][k]]);
        }
        columnsCentreOut = centreOut(columns);
        nearQuarks = new boolean[columns];

        // A fixed seed: the loss book stores position hashes, so they must not change between runs.
        Random random = new Random(20261102L);
        zobristQuark = randomLongs(random, 2 * cellCount);
        zobristCooldown = randomLongs(random, 2 * (BOMB_COOLDOWN + 1));
        zobristSideToMove = random.nextLong();

        owner = new int[cellCount];
        columnHeight = new int[columns];
        lineQuarks = new int[2 * lineCount];
        linePattern = new int[lineCount];
        threatCount = new int[2 * cellCount];
        fourCount = new int[2 * cellCount];
        fallingQuarks = new int[rows];
        networkOn = withNetwork && LINE_LENGTH == TRAINED_LINE_LENGTH && loadNetwork();
        patternsOn = withPatterns && LINE_LENGTH == TRAINED_LINE_LENGTH && loadPatterns();
        stateSize = owner.length + columnHeight.length + lineQuarks.length + linePattern.length + threatCount.length + fourCount.length + 9 + (networkOn ? 2 * hidden : 0);
        clear();

        for (int ply = 0; ply < MAX_PLY; ply++) {
            actionLists[ply] = new int[2 * columns];
            undoStates[ply] = null;
        }
        historyScore = new int[2][2 * columns];
        actionPriority = new long[2 * columns];
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

    private static void reverse(int[] cells) {
        for (int i = 0, j = cells.length - 1; i < j; i++, j--) {
            int swapped = cells[i];
            cells[i] = cells[j];
            cells[j] = swapped;
        }
    }

    /** 3^i for the cell at index i of the line. */
    private static int placeOf(int cell, int[] line) {
        int place = 1;
        for (int i = 0; line[i] != cell; i++) place *= 3;
        return place;
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
        if (myTurns == 0) matchBudgetNanos = budgetNanos;
        noteLostTurns(field.getMoveHistory());
        myTurns++;
        bombCooldown[ME] = myCooldown;
        bombCooldown[OPPONENT] = opponentCooldown;

        int booked = bookAction(positionKey());
        boolean bookPlayable = booked != NO_ACTION && isLegal(booked);
        long budget = Math.min(matchBudgetNanos, timeUsedNanos > MATCH_TIME_GUARD_NANOS ? budgetNanos / 8 : budgetNanos);
        if (myTurns <= COLD_TURNS) budget /= 4;
        long setUpNanos = System.nanoTime() - started; // reading the field, and building the board on the first move
        plannedAction = bookPlayable ? booked : chooseAction(Math.max(budget - setUpNanos, budget / 4));
        timeUsedNanos += System.nanoTime() - started;
        answeredTurns++;
        return columnOf(plannedAction);
    }

    /**
     * Counts our answers the arena did not place. However short our search, an answer can arrive late
     * when the machine stalls; each late one halves this match's budget, and each placed one gives
     * {@link #RECOVERY_NANOS} back, up to {@link #budgetNanos}.
     */
    void noteLostTurns(List<Move> history) {
        int placed = 0;
        for (Move move : history) if (move.player() == myColour) placed++;
        if (placed < answeredTurns) {
            lostTurns += answeredTurns - placed;
            answeredTurns = placed;
            matchBudgetNanos = Math.max(MIN_BUDGET_NANOS, matchBudgetNanos / 2);
        } else {
            matchBudgetNanos = Math.min(budgetNanos, matchBudgetNanos + RECOVERY_NANOS);
        }
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
        if (networkOn) addRows(quarkRows, (2 * cell + player) * 2 * hidden);
        int[] lines = linesThroughCell[cell], places = placeInLine[cell];
        for (int k = 0; k < lines.length; k++) {
            int line = lines[k];
            adjustThreat(line, cell, -1); // this cell was the line's gap
            adjustFours(line, cell, -1); // ... or one of its two empty cells
            staticScore -= lineValue(line);
            if (++lineQuarks[2 * line + player] == LINE_LENGTH) completedLines[player]++;
            linePattern[line] += (player + 1) * places[k];
            staticScore += lineValue(line);
            adjustThreat(line, FIND_GAP, +1);
            adjustFours(line, NO_CELL, +1);
        }
    }

    /** Takes the top quark off a column, updates every line through its cell, and returns its owner. */
    int removeTopQuark(int column) {
        int cell = column * rows + --columnHeight[column], player = owner[cell];
        quarkCount--;
        boardHash ^= zobristQuark[2 * cell + player];
        if (networkOn) subtractRows(quarkRows, (2 * cell + player) * 2 * hidden);
        int[] lines = linesThroughCell[cell], places = placeInLine[cell];
        for (int k = 0; k < lines.length; k++) {
            int line = lines[k];
            adjustThreat(line, FIND_GAP, -1);
            adjustFours(line, NO_CELL, -1);
            staticScore -= lineValue(line);
            if (lineQuarks[2 * line + player]-- == LINE_LENGTH) completedLines[player]--;
            linePattern[line] -= (player + 1) * places[k];
            staticScore += lineValue(line);
            adjustThreat(line, cell, +1); // this cell becomes the line's gap
            adjustFours(line, cell, +1); // ... or one of its two empty cells
        }
        owner[cell] = EMPTY;
        return player;
    }

    /** Value of a line from player 0's point of view: its pattern's, or with the line weights, only a line open to one player counts. */
    int lineValue(int line) {
        if (patternsOn) return patternValue[lineDirection[line] * PATTERNS + linePattern[line]];
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
            if (delta > 0) addRows(threatRows, (2 * gap + player) * 2 * hidden);
            else subtractRows(threatRows, (2 * gap + player) * 2 * hidden);
        }
    }

    private static final int NO_CELL = -1;

    /**
     * If the line holds four quarks of one player and none of the other, adds delta to the four count
     * of its empty cells. The cell given counts as empty though its quark is still on the board.
     */
    private void adjustFours(int line, int alsoEmpty, int delta) {
        int quarks0 = lineQuarks[2 * line], quarks1 = lineQuarks[2 * line + 1], player;
        if (quarks0 == LINE_LENGTH - 2 && quarks1 == 0) player = 0;
        else if (quarks1 == LINE_LENGTH - 2 && quarks0 == 0) player = 1;
        else return;
        for (int cell : cellsOfLine[line])
            if (owner[cell] == EMPTY || cell == alsoEmpty) fourCount[player * cellCount + cell] += delta;
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
        Arrays.fill(linePattern, 0);
        Arrays.fill(threatCount, 0);
        Arrays.fill(fourCount, 0);
        completedLines[0] = completedLines[1] = 0;
        bombCooldown[0] = bombCooldown[1] = FIRST_BOMB_WAIT;
        sideToMove = quarkCount = 0;
        boardHash = 0;
        staticScore = 0;
        for (int line = 0; line < lineCount; line++) staticScore += lineValue(line);
        if (networkOn) for (int player = 0; player < 2; player++) System.arraycopy(emptyBoardRow, 0, accumulator, player * hidden, hidden);
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
        at = copyInto(state, at, linePattern);
        at = copyInto(state, at, threatCount);
        at = copyInto(state, at, fourCount);
        state[at] = completedLines[0];
        state[at + 1] = completedLines[1];
        state[at + 2] = bombCooldown[0];
        state[at + 3] = bombCooldown[1];
        state[at + 4] = sideToMove;
        state[at + 5] = quarkCount;
        state[at + 6] = staticScore;
        state[at + 7] = (int) boardHash;
        state[at + 8] = (int) (boardHash >>> 32);
        if (networkOn) for (int i = 0; i < 2 * hidden; i++) state[at + 9 + i] = accumulator[i];
    }

    void restoreState(int[] state) {
        int at = copyFrom(state, 0, owner);
        at = copyFrom(state, at, columnHeight);
        at = copyFrom(state, at, lineQuarks);
        at = copyFrom(state, at, linePattern);
        at = copyFrom(state, at, threatCount);
        at = copyFrom(state, at, fourCount);
        completedLines[0] = state[at];
        completedLines[1] = state[at + 1];
        bombCooldown[0] = state[at + 2];
        bombCooldown[1] = state[at + 3];
        sideToMove = state[at + 4];
        quarkCount = state[at + 5];
        staticScore = state[at + 6];
        boardHash = state[at + 7] & 0xffffffffL | (long) state[at + 8] << 32;
        if (networkOn) for (int i = 0; i < 2 * hidden; i++) accumulator[i] = (short) state[at + 9 + i];
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
            "15x30 932b0e9364d6fe0c 18;15x30 92382d2081853b72 12;15x30 3d1c8eea0249f64f 26;15x30 3884c5e144be7c05 11;15x30 4421266e6206a710 12;15x30 e36afd9b38c0a153 14;15x30 7413ca505f75a080 5;15x30 1b239c79eae243af 13;15x30 0c9c0ff82d3b4c69 12;15x30 ce407cc4bc0bb54e 12;15x30 5a524f0c377d069e 11;15x30 24adcc89ea8396c6 2;15x30 73b66f65fd39ca77 2;15x30 e6b989518f1e743e 2;15x30 797a53d710f909fd 11;15x30 ccc5aa438dd1f6a6 11;15x30 f720a2910eecf2d9 11;15x30 4bdf36e477c5c089 15;15x30 34e4b228f1a7f79a 11;15x30 10484206bcd36e4a 8;15x30 78cebae845423dbc 15;15x30 fa5162239353f4e7 10;15x30 368a813bcf491982 1;15x30 805f44e1a039031d 20;15x30 15c078bd63e6fb19 17;15x30 1a26907beafff58b 17;15x30 8b920d506b2933a7 17;15x30 fa328e40b5dcc236 17;15x30 0cc9ee5c0e331e75 17;15x30 3796ccd014f04a99 17;15x30 0e08818d513da5c0 12;15x30 dd8965f7963022b9 11;15x30 5520b702e64c98e8 12;15x30 586c35e777a122bc 12;15x30 f0b11f8769eaa894 12;15x30 4a94ea8a168cb0f9 9b;15x30 786a03bbf68b126f 9b;15x30 2c451f03b44b58d5 9b;15x30 78c3a124fd2793e7 9b;15x30 011041f74c5be783 5;15x30 ce0554a97b050072 22;15x30 b51e2e57441e0577 9b;15x30 5708a12ed49e9267 12b;15x30 1a687bd0ff7a4d6a 12b;15x30 2b6cea86b7c54e68 12b;15x30 dc7b4c52f259b75b 11;15x30 a30e16801d5f2be6 11;15x30 d687323e2dcbc0ea 12;15x30 3397b98bcc4a8189 12;15x30 2482829221816e5d 21b;15x30 7f8c76e3b209f675 29;15x30 a95304f456a10b80 11;15x30 4b9de0477d974315 23;15x30 d3bd696ce15476e2 8;15x30 527c17bed4f1b349 7;15x30 4f07809c2071110c 7;15x30 d18019c2ba79594c 7;15x30 c1b9715e3e3cef71 1;15x30 2cbcc50822cf9f5f 4;15x30 a9adec14bad53fd2 4;15x30 8a5d04177527435a 10;15x30 d2c5538426401b51 10;15x30 d56ed27ded32ff94 18;15x30 3e47bb238001d260 3;15x30 6920751cd12c9972 9;15x30 b9c8e6cb108f0c98 10;15x30 48a1ada210707c63 10;15x30 b2390363237ff708 10;15x30 f7855c4cf43fc461 10;15x30 736fcebf5f64cafc 10;15x30 68d0f8a7c0eea675 4;15x30 9582c92cfc84f49c 3;15x30 74e29a4657058cac 18;15x30 87405f6517898d0c 3;15x30 3361e73adb8d6ff7 12;15x30 1d23ce6425714c39 26;15x30 6cebc2009d310cac 1;15x30 473e9e169f646fdf 2;15x30 980bb0ea879eb4f3 10;15x30 27830eeee972fc74 10;15x30 63049895dbeae80a 10;15x30 b7c56ae70017dc30 12;15x30 a44c5b4e7f560510 16;15x30 e344e11136eb070b 11;15x30 8ef9f92715a8a248 8b;15x30 a6d41f1d66fc6d45 10b;15x30 2f42ab1f70680d13 10b;15x30 a38787b2a4699a35 10b;15x30 a31b310bae1efd05 10b;15x30 4ffbc6ea7204986f 8;15x30 bc7e14be4f98257a 2;15x30 8c44dd088487cdc0 25;15x30 4c4efdb6d5417524 22;15x30 7ff1e4cbe06e2136 13;15x30 e68e57eb0c211583 22;15x30 fbcfdfee7ef4c572 19;15x30 215ddcc755d0dcee 26;15x30 7494c587b681284b 7;15x30 5bed41f61c976f88 29;15x30 ed6c9c98da2d01b6 29;15x30 74bdf351f1a4cc9a 29;15x30 fc419ee31c8a4ab1 25;15x30 7dff2a727cd3036f 25;15x30 0e607aa438a375c9 13;15x30 17cfd63022929fc5 13;15x30 43626b6791a90d44 13;15x30 857e9aba1cd20059 24;15x30 ae3e1494580c286f 13;15x30 400a03055084e09a 3b;15x30 e590a2aef7b19906 3b;15x30 e8efd1d1058299f1 3b;15x30 1d22176943b0cadb 3b;15x30 f10c7e9a9961693e 3b;15x30 3ac0e82399d2d61d 27b;15x30 088fe34b8cf8480b 18;15x30 18ad00213e90b824 18;15x30 fc5d9432ff3b5237 13;15x30 62da4cf187783d34 19;15x30 b56d8a41c09abc60 13;15x30 b0edf2eb29b2840a 19;15x30 c178ec713b87e0a5 13;15x30 a870645ed8f980a5 19;15x30 2837b1a629cebd49 19;15x30 9164d6eab6cb1fa3 19;15x30 775780d3aeb97075 7;15x30 ce80d0a400d38e52 21;15x30 2475474b3716e209 17;15x30 f917272ccfe85e4e 20;15x30 c62a3c95b0a6eda1 13;15x30 bbb9a94190822fb9 22;15x30 8ebc28913c0da43c 7;15x30 35ed60136d90db44 23;15x30 19fc126a936282fd 21;15x30 a2b8663341a762c9 23b;15x30 442992c1bf991813 23b;15x30 6f1a7822f11f3cf6 17;15x30 d43856861b83ba54 17;15x30 d3aa2eba62bdae74 17;15x30 da0062d2131a4bf8 10;15x30 bee1cbf897100357 13;15x30 a23571a471f471e0 23;15x30 f1fb10e728ac5ce0 25;15x30 f00913ca67e265cd 12;15x30 4d21ff2d3822f10a 18;15x30 6eb80342934d5299 8;15x30 677ce1e5e27dcbb4 8;15x30 60876d221d3a0374 18;15x30 86c0e08a81a0206b 6;15x30 1b6f81d8788f6204 26;15x30 1f77a64b13ed6889 6;15x30 e5167f910444869d 6;15x30 012b6008af2ce4f2 18;15x30 685712cede5efbbd 13;15x30 14bc2ab951bfefb2 18;15x30 b966368b018fe6de 15;15x30 eaacf2138e1aad10 19;15x30 f24f741bc61412a2 19;15x30 fca8cac212ef0e76 15;15x30 1d1ab7b822a47682 18;15x30 a9baa12596b0997c 20;15x30 a6d6ee0dfb2d7e56 20;15x30 367aaf9b1693b105 8;15x30 4192e4d657a46dc7 24;15x30 537ec2f0150245cf 2;15x30 b74636b7ba975981 12;15x30 48407d1667116577 12;15x30 778ac39a656a00bb 11;15x30 ea1758d4a2be319c 10;15x30 e098cd14e7f14457 5b;15x30 ef3e5e60dd7984d5 5b;15x30 59aa6fa51fcc702f 5b;15x30 fb94f8675798ba69 5b;15x30 9f2596d5cb024d11 8;15x30 f7963e7312d6d085 8;15x30 d98be0de49621a3f 13;15x30 493b7f120871a15a 8;15x30 680bac500e369c0b 8;15x30 305419263fc02222 15;15x30 adc0d540012f184f 10b;15x30 bfd9979407c5aaaf 11;15x30 f3b9bbb050e7141d 11;15x30 edfd8f5f054ce234 9b;15x30 8efc2c5480363f14 15;15x30 ca3163e3bc4f1d9f 15;15x100 42e2de873a988d1e 44;15x100 3b6b2b7ebbd87df6 53;15x100 2e9bd59f91ca4516 58;15x100 5030c3edcc54f12a 56;15x100 39446fcf63f895f1 57;15x100 2b3ece218de1e867 57;15x100 a7c36e1a0ca0ba1f 57;15x100 80c66dd4c48e94e0 57;15x100 2a2ec9a45b4f322a 57;15x100 e91b07ce9b27e1b6 57;15x100 8b3b9986b33658e0 53;15x100 67572a8ff7db7a4d 53;15x100 8748e8c2fc509fbd 53;15x100 decab78693492762 57;15x100 62366ebe94d3ebdd 31b;15x100 7599ddb1cbc164ca 57;15x100 700a796a8e337542 57;15x100 37d4b2e4e0eb3904 48;15x100 67a292a86b90c1be 48;15x100 da750fbad4ae9127 25;15x100 bdbec0b9eeffef3e 25;15x100 3fe004e751191644 25;15x100 221431243942a3b6 25;15x100 e97f0e551af24135 25;15x100 8e4d05e3cc9b3c84 25;15x100 017f17b55c7c46f5 19;15x100 a40b31ca4e8f2e0b 28;15x100 2f6d1b31f14c777b 19;15x100 46513a9227721d4b 19;15x100 7ca8d78964c707f8 19;15x100 29879c449b1469c3 19b;15x100 0fa50a02eb185c50 79b;15x100 dbaac228fda58fd7 31;15x100 74e07400abf248c6 51;15x100 0fb5c9aee2748f19 49;15x100 9826d53210aac4a1 49;15x100 d1857c4eef954340 13;15x100 54da368718aad483 13;15x100 f305205c04fe4c9e 13;15x100 cf9a138f091f5b57 13;15x100 13ebc9028f310bb1 13;15x100 3102302d5786f290 13;15x100 eebcd3ef70df5ab6 96b;15x100 d1267075346bd2f3 13;15x100 fd460d8ad35f0e0d 24;15x100 1a245facf8ddb02f 52b;15x100 d8030c89faabfe45 48;15x100 569c4b552174de85 47;15x100 d0cf6ccd0c127e06 50;15x100 7e3df2772b11b848 49;15x100 45246ffb3e6ea3ca 48b;15x100 58e54034a17aa0d7 48b;15x100 d4a67f6dafab96a4 57;15x100 aad7af765c173cfa 57;15x100 1a2e5ceacf04394f 57;15x100 233763f89abc62ad 45;15x100 43d0e58a7393bf72 58;15x100 60f79deb03923b9f 58;15x100 75b9055ca799f220 58;15x100 2be9efc4e71d79fb 55b;15x100 515e2850349164ee 58;15x100 2537257b4549be1a 53;15x100 dbfab8b4bdbb7ca7 35;15x100 434ba28bb02d7bcd 35;15x100 429fa5af369c46c3 35;15x100 d44dae58f9eadeb0 35;15x100 74e96c5464405d4a 43b;15x100 602a21efaa44b5c3 43b;15x100 d920b8f7779a7f6c 43b;15x100 13223dfff40b8e93 43b;15x100 67ef5f2355df9d33 0;15x100 fe3573e00cfaab4c 49;15x100 96153b57a7a47357 49;15x100 b1446ae709f9603c 51;15x100 3c822d4630a49845 48;15x100 9f1239920c39c592 50b;15x100 d9317ca23f07ae7e 50b;15x100 6d8e534b003e2b39 48;15x100 5c4c73520e3fdd47 47;15x100 baaedeea81ff19d1 48;15x100 b48651ac30de33a7 50;15x100 3bcb2548b2accec0 52;15x100 78db3b454b73946f 51;15x100 1ce419bbff6231e7 48;15x100 2082fdd0cce8f107 50b;15x100 0e4382a18403a486 48b;15x100 ac311569d19bcf51 47b;15x100 4249e87de34a02e1 0;15x100 e937b2d73f14f226 48;15x100 9c91606f0698def9 48;15x100 73f1c63050fd2c5a 49;15x30 19ea9290fead20a7 17;15x30 030513b3d1f08da9 15;15x100 9dfaf35c60e4d589 47;15x100 8d59e84ee1d3f232 47;15x100 d2d5b7b8f619137f 45b;15x100 6e4e0d4e87fd43d2 45b;15x100 713f94b4d88b9618 45b;15x100 5c27cf4acb281ee8 44;15x100 972117155cfe2e6f 43;15x100 e53aab6619ce2672 43;15x100 4cfc062ea5d0f59c 44;15x100 e7813662109d64b2 53;15x100 eba6fd9f70817852 53;15x100 9426a700a0c4f097 55;15x100 7af363cbd410dfdc 9;15x100 c79afaa4b26125b9 7b;15x100 e2d55d953c1d3900 55;15x100 6758dd1ac25b1b6f 55;15x100 0ad8d8490bb78957 55;15x100 e1e09ca74838d3ec 55;15x100 5b088ac9d2e69f1c 49;15x100 120c65c8908bc5e7 47;15x100 7722f27146648649 49;15x100 2ea39773babdc7fa 49;15x100 9b9883aaa73440dd 51;15x100 ec48ca08838051bb 48;15x100 14b192be28310f12 53;15x100 7ac091163c161db9 53;15x100 2ac58d24c93f189e 53;15x100 9460b449f9ea3f01 39;15x100 84b955a42ca6c129 39;15x100 32beb7fec2998f14 39;15x100 ce135d5f2ce5a49c 39;15x100 a2a8cb29d9245629 39;15x100 0138dffde5b90bfe 50;15x100 4a133f8c040ba28f 0;15x100 e7ceecda9b21f14e 0;15x100 634ef9bd1edb6cfb 85;15x100 c92725a87320fd05 52;15x100 bc3dd753f49d2f97 52;15x100 58a3444cf1be9409 52;15x100 761099c36ba6d4ff 53;15x100 5a04ec634a4f65dc 48;15x100 bbaec47eab057b9a 9b;15x100 f5e36c8f8d8a2fb1 50b;15x100 e888af007c5b92b0 55;15x100 35365a5a3a188eb3 55;15x100 7d6d2c392996ff5e 55;15x100 5ba465787b0d6321 55;15x30 a93d1258d2a89b04 14;15x30 20abf187786009fb 21;15x100 489c71a17448ff1b 11;15x100 41773fde1a3adba8 11;15x100 e1e3ee40cbbe3aac 11;15x100 ab3b4e345aecfb3f 10;15x100 19972f8e4e2fa6ba 10;15x100 6a7269095e5de695 5;15x100 34c688c996e0343b 6;15x100 c1dc5b94d232c1cf 5;15x100 5e7fbedbc2275224 28;15x100 b4f30011f30c3091 27;15x100 539e9f95db9092b5 27;15x100 6e3e7bb02062abaa 26;15x100 a960e08ffe652b95 26;15x100 e68dab44e3021c67 65;15x100 53dddb1fe9f4773a 65;15x100 42135a590bf21d6c 65;15x100 8c419ba073bb8210 41b;15x100 ec46b8f52ecc5d7b 41b;15x100 91d0d0575cec61d5 41b;15x100 52820f79c72795f7 41b;15x100 4dc94bef89ef027a 41b;15x100 5829d25b504b09dd 41b;15x100 cf9397f31d9de08d 41b;15x100 5d4a67487236af56 50b;15x100 d1d26f0a3ee4e7b5 50b;15x100 7f46253647e543a5 50b;15x100 bdccdebf719561d8 51;15x100 a7ac03474eed62ff 0;15x100 87abbce48ee71b84 0;15x100 3d5c97dc4a17a99d 47b;15x100 f922bdaedd1f2a26 50b;15x100 3b2b7fdd9fa38c8b 50b;15x100 2cad04f6051c7d61 39b;15x100 25cfa9ba549d0a21 39b;15x100 3d94bb8360c64aef 39b;15x100 332e8851c94768bf 39b;15x100 762f5476b7d21806 39b;15x100 ea1dfef35351b361 39b;15x100 7c93f84922337f32 39b;15x100 3b07fd73e2c5b86e 39b;15x100 1a45bf7973d469d8 39b;15x100 2f83109d6c943542 46;15x100 468d6e8437433ef8 30;15x100 aa7bd1efb34d5446 40;15x100 49fed696a14d9192 40;15x100 d5c248648d55279e 40;15x100 547448c5f97359a2 23;15x100 c2f90d0a49aa9946 49;15x100 687b4975157c30e7 16;15x100 3599c74bccc5dccd 16;15x100 1bef84d36b307d36 50b;15x100 fb4691e1c3280932 46;15x100 ac35c42ed5ef6d10 27;15x100 f227b6db8c312fd4 27;15x100 b251627c880d5e12 27;15x100 bd9a15232decb910 27;15x100 0d4047fa997bfa7f 28;15x100 23e41b9a47e9f92d 19;15x100 6fb0e1cf6b54f7f1 26b;15x100 7a573cb60617adab 26b;15x100 07c090fb3e46200b 23;15x100 0e8094dbb3aeff6a 26b;15x100 60d49301c4b52ec3 50;15x100 f41450d98419387c 46;15x100 f3b9704d3616f6c2 47;15x100 e28dcfa1dfd27dc8 47;15x100 026ece4804361b72 47;15x100 0a44a923a35cf876 55;15x100 83028d4b09c13298 48b;15x100 f85d4f473fad0c2a 48b;15x100 e224f332e48e40cf 48b;15x100 47ef1a109a3545e8 51;15x100 5b8c114435c11918 51;15x100 e500478c2bd0b945 51;",
            "15x100 c3c505d61058bace 48b;15x100 4fe51ea380c09fc9 44b;15x100 d4c96122c3b7119f 50;15x100 cda4949afc70a636 36;15x100 92a7f46e649a67c9 51b;15x100 f1a151f9fc7febb2 36;15x100 a49f34b9296279f3 36;15x100 fd1e221c6c17c183 50;15x100 a92413816e2b4c3c 50;15x100 9539a5ad6c8d30e8 47b;15x100 05355fa4b924435c 53;15x100 187dd4be874bb165 49;15x100 5ebc082f043180f6 52b;15x100 bd8b1990be71464d 52b;15x100 a958040670b48896 52b;15x100 90cf365ad13ee70d 53;15x100 44176e2b7d6e7d70 28;15x100 dadc8b83d487619a 28;15x100 3fa8bfe9cbc8171f 28;15x100 585ab02c18194d84 89;15x100 cb416e249687a45b 89;15x100 77a9aed9e33158c8 7;15x100 2fa23bda016e4097 7;15x100 98ba35921caccc9e 4;15x100 9ec60393531868b3 56b;15x100 c7c18f63f1913f11 47;15x100 7a77205f846743c0 53;15x100 2ec621d6ae57bb3f 52;15x100 2d87024f33a24aee 49;15x100 ebe8e08277703591 50b;15x100 37a304abddc60bd6 57;15x100 ca9f706ea2d2be64 25;15x100 9d5e150cd0990379 32;15x100 5eb674c8bf48891e 25;15x100 002b5bde529cc2f8 25;15x100 96f8450a6a51ffce 25;15x100 380197d6cb0f1341 25;15x100 97314c6a33bb4b01 25;15x100 9fc925fd08641160 64;15x100 70ef9cc5c83af7a8 28;15x30 cd146bcc913691b5 15;15x100 c51a2f164b1ed365 49;15x100 d00d597d0194db0d 49;15x100 070a82376d0623c8 49;15x100 5bc0ecf01906a1da 49;15x100 a940eed38f32fc4a 46b;15x100 b38bc2621b25bdc5 47;15x100 c35040ca553edc45 47;15x100 2690cb297d98d0e3 43;15x100 f8d7482a0b994145 57;15x100 d038528ecc43d8b0 6b;15x100 bdb857dd05af4a88 6b;15x100 82f47d17a62808ea 63;15x100 339753917f87d950 48b;15x100 dd4b54404d545e6d 63;15x100 164e80ab26a7b68f 63;15x100 7ae78bb96e4bdae6 63;15x100 16bfdda9a81a42b1 63;15x100 0fe8e6a77f3e3f36 48b;15x100 22d0ba1ca3f34d31 0;15x100 03f4af04eae91cd8 5;15x100 5a84297876fb2939 5;15x100 7c46008f9e819b2a 53;15x100 f22c4dbc68c43dbf 47;15x100 39d7c200603c750c 50;15x100 3ccbbc94f29a6383 49;15x100 abee4bcc68950f63 49;15x100 efa509a535859e5b 54;15x100 6113799ee3919065 54;15x100 f27167afc84dd766 50;15x100 0f98c05c1a3de77b 49;15x100 2f0285fe97ad62d4 53;15x100 e5e3daf733a10995 0;15x100 dd556c35de6481f5 63;15x100 a29e8eacf1ebbef6 34b;15x100 f27e01a82c278c2d 30b;15x100 8170425de2c41c3a 63;15x100 74e6798900e0ce49 63;15x100 11c303d9e1862ece 63;15x100 71f5c8ac23739e9e 0;15x100 d871d57c11631bb5 49b;15x100 ef4945bbc37b7863 49b;15x100 a74b04460d3fa0d1 49;15x100 5767a307fc170594 49;15x100 745752d84960b627 49;15x100 a139437c2d57b8c4 49;15x100 f3eafce50317f3c4 1;15x100 ca4f871634d122ca 11;15x100 6aaf4a15a7896955 48;15x100 c942d77b3602cd46 48;15x100 d43aeb58c60f48b8 28;15x100 e069760cbf9e5a48 31;15x100 b09f8a6d4413db67 31;15x100 2cbc4e7f5530394d 67;15x100 686fb7a62ae2cab3 68b;15x100 0bfe5ba9012d64e0 49;15x100 3f1e06082b787b6c 49;15x100 3e86b7501754a8d7 53;15x100 d6c42861e29441a5 37;15x100 48d80472f12d599f 35;15x100 2df588115ab7eb66 40;15x100 010233fe84abdfef 34;15x100 d0981a599f33ce9d 5;15x100 26819097dcbfc867 54;15x100 52d28cce514021e1 19b;15x100 f84d3b60fc4d1992 19b;15x100 fa10ee33d2581b23 33b;15x100 3181d66d7600de61 33b;15x100 01392f211487df7c 16b;15x100 7fac41ba794357df 16b;15x100 ffcd39b11650f3e0 15b;15x100 80b050aa97fd0d5f 69b;15x100 06568ead667d9b40 69b;15x100 d1ebdafa69af1473 69b;15x100 3e5513f781e50119 69b;15x100 1cb2a7ebf04a8fe9 26;15x100 7e64aedf0a9a550d 26;15x100 930a3c5252ca31e0 26;15x100 bc04817955a9e5f3 60;15x100 802a8376064b56fe 5;15x100 d9ce2f256b4c0cfa 5;15x100 066ded41d852be6d 5;15x100 ae09968f9c52d6f1 5;15x100 e07ebd7118dcff17 46;15x100 7b60d3019b804ef6 47;15x100 5ef0cf1274762190 46;15x100 f8916668e5f762c8 42;15x100 9ded5cb8397ee216 46;15x100 b5fe8bfd23895461 0;15x100 53cb89af585ee953 49;15x100 e6cc1801e2ee5368 49;15x100 1e1358e333cd3597 50;15x100 682945dcfbf6a56b 50;15x100 851add2c8fa4071f 0;15x100 46a784484f8a6e91 24;15x100 d4e961985dc4a482 5;15x100 1a642d63a225d225 5;15x100 4e09aa31e14b49fb 7;15x100 74bbae960a9f40bf 7;15x100 9b9becdcc61486de 4b;15x100 bd1fbfccaadeb694 56;15x100 4dac6f9bec4dded8 56;15x100 ea63be986ae502a3 56;15x100 915dec23ea6ae869 56;15x100 a95d3110d106f804 56;15x100 1079b46a2cf23e1f 56;15x100 a6e9c21319dec706 48;15x100 fce396d7437ef28e 48;15x100 3e2e6f2806565347 48;15x100 b7416c6b22420716 38;15x100 bc4765bc730273a9 48;15x100 4b9b9d46dea5a003 24;15x100 86d68d898067b938 24;15x100 9e1ff715890c71fc 49;15x100 8a9d843a6e4f29fe 49;15x100 32e6af99bcf53a72 50;15x100 e84e26803cef2b62 1;15x100 9810bcac6acc5c80 4;15x100 4ce7aa3f7d881529 1;15x100 698d82d132845241 1;15x100 9ded94b8c995b861 6;15x100 91643c8ae719b0ef 5;15x100 8f1dfb771018769b 50;15x100 c7ce104e4861942a 50;15x100 a0ea9b8d00792038 49;15x100 3b26cfa4ba9db9bd 51;15x100 ffcb88c070e056d6 50b;15x100 eb1eada289ebdf2b 52;15x100 db25178c7001b850 49;15x100 56a09ccfb17ca919 0;15x100 bbf4e4e319a6a453 50;15x100 ac8376eb16918b74 49;15x100 4132c5ab01752199 49;15x100 20c938b06de7de67 2b;15x100 10bf0936d1aee8e0 49;15x100 c98b5ef3017f5d1d 75;15x100 ff3236a6865155d9 77;15x100 e501ea4652f68ab1 77;15x100 8bc95c4f4b40c45f 57b;15x100 7ec468a0f13cbad2 77;15x100 07d0ed0167aa2491 77;15x100 9fe87f46d504c122 15;15x100 fc9bbd6171e664ed 18b;15x100 0d70bddae2c97826 46;15x100 afb2e6ecb5563390 4;15x100 4b7e958a98f1167a 75;15x100 3f894a7ac3814b3d 76;15x100 67f65bf2fa0b7517 76;15x100 daa4beed04f20a33 75;15x100 ff24e0d848ae894e 34;15x100 987d9cfabbd143d0 0;15x100 734eb9a4c8b36b31 0;15x100 17656820ae9e7a1c 52b;15x100 cc6e700a1dcd122a 44;15x100 f58a6cfbebbd20b7 48;15x100 68a7d256fbb63700 47b;15x100 597fcd50a7e81372 47b;15x100 51e024f559ed965a 46;15x100 68f0cf0b3b021613 7;15x100 f900c764cf64c906 10;15x100 43322dd75999320d 9;15x100 5d7cb29bfccc7461 61;15x100 4be71f6d19c3d982 59;15x100 5c78503b2e598ef0 59;15x100 7da9d986a821fcae 59;15x100 aac507d678009f7e 53;15x100 45062c6e4a61eda3 56;15x100 b1712188874cdfae 50;15x100 6691b95d07141590 46;15x100 5dfae259215050fa 48;15x100 a2a6b1473f85515b 51;15x100 b4baad6c1708be4f 52;15x100 90e91a1c821fec08 52;15x100 4bfdbf7c463703e5 70;15x100 e9855bc874b2f8f6 70;15x100 54142a4191d3b505 48b;15x100 e0cc4bc7f44801b8 70;15x100 236082cc308ad893 70;15x100 5b2db09562b75f3a 70;15x100 07257cd93203a3ef 70;15x100 c13f0984f3dd7bb5 70;15x100 f922dd6e00b35fb2 46;15x100 cb1245c5597d9474 5;15x100 30aa900288258074 46b;15x100 adbdd6b040ddedb3 47;15x100 0bebc71321c46e8b 55b;15x100 6d152d646470fb09 55b;15x100 d0fdd289477b22d0 60;15x100 12503f871c53a448 60;15x100 2e8d7bd47f0b34e5 31;15x100 61add17f47502a16 54;15x100 2c7bbcf71c9459be 54;15x100 d7a72ed1f039ae33 54;15x100 6429a2ac4016462a 61;15x100 ed8461121a8a95f0 49b;15x100 d1d511cbabb68705 49b;15x100 a5a286dc97edf068 49b;15x100 70bdaccd800e6cf1 49b;15x100 5701a1803784efc2 43;15x100 fa45988b2c4df6f5 43;15x100 083e4df3bf6643fe 43;15x100 a44adc0db7e5c814 14;15x100 4131f2f9e2752350 17b;15x100 9f1926be993b315b 57;15x100 70d833296c09d818 51;15x100 3e8d90866337575f 51b;15x100 d18d0c8339330b60 55b;15x100 64095ce197f67420 33;15x100 efbac4d0e0142ba7 55b;15x100 883817743b89397e 57;15x100 8d667c0d9bcf93d9 51b;15x100 91916f0d6bc4c2e1 5;15x100 769383f4f806e850 5;15x100 13f1cc3f91b2fc10 5;15x100 af90f71c0b7e57a4 5;15x100 4ed4630b8fcfb9a9 5;15x100 f224e8a71fb0cb0a 4b;15x100 42b79af2f82e02b7 50;15x100 910d5880d32e54ec 54;15x100 542344388d47cc93 53;15x100 85b877597d46eea7 45;15x100 7d8f55f680600d63 45;15x100 5cd621162c24cf4d 51;15x100 8f26ed0a75a35a9c 48;15x100 08da66768f3efb7a 53;15x100 3a13ad31996ea76d 54;15x100 a1f6960f3d1258d4 52;15x100 efc773d1e45891d1 51;15x100 d632dc0214669b9a 5;15x100 41ce6bf511826eb0 5;15x100 d2cbd2b9137c95a4 5;15x100 0ef7f16e2b0ce1e4 5;15x100 3d80a6ccd0564fed 7;15x100 e8d9486acfc99e54 7;15x100 9ff20d9af8918707 4;15x100 6bd3fcc30af4baf1 4;15x100 a56d6d439333363d 0;15x100 bbb79557945fee7a 54;15x100 0afc26e207df6d64 50;15x100 508c03c7cea98282 53;15x100 239181f658129881 53;15x100 0e56aeb88fe5f175 50;15x100 efadc9a81d234048 52;15x100 ee493a25d1187190 50;15x100 99a526b818091370 8;15x100 2b998ceec8448602 8;15x100 938633e5a0d90f81 8;15x100 fca7b0ad3e743326 8;15x100 65e18da12d485149 8;15x100 e4172579da5036c2 8;15x100 54e1d1e07f2d6279 8;15x100 b400ec5eaad0fdd4 4;15x100 906cc261a5879b9d 10;15x100 b61f6f10962faaca 5;15x100 8613c50f9059f173 7b;15x100 d17db2beae4e0fc8 5;15x100 6c142bd1c83ff5ad 4;15x100 68ed256acf25fb93 4;15x100 1a118ae9661f5ddf 8;15x100 77ed4f9e5a73db50 8;15x100 ce33438403b65028 9b;15x100 537f8dbde001ee66 3;15x30 254480cdd4ccfcd8 11;15x30 f50a0dddf1b0f51b 11;15x30 93a36dcda9981b82 7;15x30 827a4bbcbddeae12 15;15x30 80a4934da75e9e88 18;15x30 91bfb5a91a2d257a 12;15x30 5ac7832545838a75 5;15x30 ad53fa6eb61bf1ab 17;15x30 beaff8715b7a0371 17;15x30 c07011e9312e9dca 17;15x30 3e2f9868b3d428e1 20;15x30 4a67dded7a7314b5 13;15x30 6b4d02a80093c36d 9;15x30 60c7315ff6099a02 9;15x30 e7cebfe30bf66468 9;15x30 634ba788bc5c8b31 9;15x30 998b855a568443eb 9;15x30 e91bbd7a1ac17904 19;15x30 3fd1fec6fcf1122a 23;15x30 2e607fa7676daf7e 16;15x30 9af982667c4c9cb0 9;15x30 8dab1ca7d220a0f0 15;15x30 0fc92b94734ba208 16;15x30 3a58c664b7b5d776 9;15x30 29c55b33b0b08b08 11;15x30 d415450bc2b514b5 13;15x30 6b5246bf107e3fc7 13;15x30 b6d2b53c603d405d 13;15x30 58ee9eb8b6448417 9;15x30 c62058644eb1c394 9;15x30 32936ea2a7477d76 19;15x30 d7394f405106dbb3 16b;15x30 47ecbc3442af23bd 19;15x30 9ee1a394b780b481 19;15x30 8c25f8b6c81f9bd0 0;15x30 d4144b4f136ff8f8 21;15x30 6c9ad418c4bb9bcf 21;15x30 e8a22a6736ef30a9 12;15x30 e0812d4971aa3d12 9;15x30 5da7412f5e50f841 9;15x30 e4ee191b1de1ec17 9;15x30 31593a90dad407d2 21;15x30 e6fadad92097c237 13;15x30 317a2a9beb69b166 13;15x30 480795031d41a2f0 15;15x30 7d634106f6da3b93 15;15x30 52a61a9fd93dcec9 15;15x30 04760877d33c8f1b 13;15x30 a3d579e927f4135f 13;15x30 132a607e0dc71ef6 0;15x30 fa237d00725e5b9f 0;15x30 59001d353a650f2a 0;15x30 79247bb215042a76 18;15x30 8f7211c127195574 19;15x30 ed0a6d99439f395e 5;15x30 1301c1a12f301d5b 3;15x30 51365e97e2d4f517 3;15x30 4093a53dc7d8f27b 2;15x30 318017cbb4681624 2;15x30 f11fea08cfe1fbe9 2;15x30 e1bf73aa44ae801d 2;15x30 4ed5671b0747ff8e 2;15x30 a4207833b033c01e 2;15x30 55728b5923ed67a1 3;15x30 a63bf494b3481ac2 5;15x30 3b661fc8f337cc53 5;15x30 afba8fbbeca6e394 5;15x30 8304598b7a082efa 5;15x30 cca847fba66b4655 20;15x30 bae45adf7e1d7878 19;15x30 6015f6adf7e60248 5;15x30 178462086b0e2791 13b;15x30 a6d5bdcc4ffdff03 13b;15x30 13a9e7802864891f 16;15x30 5470f44b160160ce 15;15x30 a5986e73d90d0476 15;15x30 a24525aca1309da0 15;15x30 5c867bbf3a4e7549 15;15x30 b9ec156d5206f678 17;15x30 f0c7324c155bf7bd 17;15x30 5698af4aaf3ec302 11;15x30 554a0de155d13276 11;15x30 31911165eb00dff9 11;15x30 1837bc007340ce24 11;15x30 cf8f936cce549851 13b;15x30 43b03bea16e6e087 13b;15x30 6028b938a8cec53f 13b;15x30 b14412c079a85e86 17;15x30 23ad829e43a76500 17;15x30 cedbee169acb1283 19;15x30 53ccb7e8d099a83e 5;15x30 5f900be30069e10e 21;15x30 89a543130613f74b 13;15x30 c0c17aaba53a00a8 13;15x30 7867a94a470b304d 9;15x30 184ddc95dd24c065 12;15x30 01055f5003590638 24;15x30 964d526acebae1e6 5;",
            "15x30 561b46f388355c6c 15;15x30 c17d18ba6e12b10b 15;15x30 b0e9132f44dc3fc6 5;15x30 1c1b877c2a4084a3 5;15x30 3fe0a519fc5e864b 21;15x100 bc841e34e78fcd1e 50;15x30 85be3ff888ba34a6 23;15x30 69d79e53f91edcdc 15;15x30 11d4205cf9d140d7 22;15x30 2b3143f53feb8434 22;15x30 32f051919fad7b07 22;15x30 2e2f69ede78cc2bf 22;15x30 e1a4e2fcae86a463 22;15x30 83a321764090ff4b 22;15x30 3f3d39f4f71e5233 22;15x30 513dffd451e54e23 7;15x30 cbf2b04857b7e02a 21;15x30 70660d88f90c7dec 10;15x30 65936fcc8ab4292b 12;15x30 aaa40aec60a62d7d 10b;15x30 53f38004cd282ade 18;15x30 8f4b004f8d204176 12;15x30 ac89fa002e3dd20c 7;15x30 8b1a9db32de5a437 10;15x30 2ed24d9c9e39ad67 10;15x30 1e8751ab04801470 18;15x30 400dc1519a640fd7 21;15x30 9871645599be4ebf 17;15x30 68511d57534ef5a0 15;15x30 669e1f0a4d07ccfe 5;15x30 86d7fb8cc81e2752 7;15x30 3f287cbb526c71e1 7;15x30 b901d83a849fd8cf 5;15x30 52473eb5530b4741 16;15x30 dc7527bae585cc26 18;15x30 48669e638aa65fe8 18;15x30 f092a5ee77985b90 18;15x30 095f29f2517d9f8f 18;15x30 199d3e368bd2c2df 18;15x30 8bf52bc4a4e51e2a 18;15x30 6950330fce081773 18;15x30 ec9705792ab8dc65 18;15x30 2e76c00c9fdab725 15;15x30 f25804e1c2ebca55 10;15x30 efa3b18acfd010bb 20;15x30 089d46ac5c9166b0 11;15x30 c29853a89be8ed87 19;15x30 5992206b57142882 20;15x30 bc07453ac5587146 17;15x30 c2d181a98e0c7a80 16;15x30 0977312b6955ad6e 16;15x30 50a098b7b58981bf 21;15x30 2699b90515eec0e3 10b;15x30 3316c6ac4b796957 7;15x30 05046fd657e97bc6 5;15x30 d892bcd0a125899f 4;15x30 9b7868edad7c2f5f 11;15x30 41a00e1a10ca13b3 3;15x30 bf70944ab3d88424 8;15x30 055787c02e96d900 8;15x30 8a0d8b22f021d0cf 9;15x30 ed416a8476f7d1ec 8;15x30 4c91ba4cc6647784 11;15x30 40cebf7ceb4fe2f0 12;15x30 aa76e514b8fe947b 18;15x30 4380d1a9e8189f7a 12;15x30 7fdd8fd91f57104d 24;15x30 7b236dfac73fb9b5 23;15x30 61f811d3621d0c78 11;15x30 2f2e63dc57ec725d 11;15x30 4a3e853e5dc176d3 18;15x30 dcfd40e4fd8b59d3 19;15x30 05f8578ed4c6e76a 20;15x30 c5c07320da8f7c2f 21;15x30 d17edd1662c76114 13;15x30 bc2c703b52a21690 21;15x30 36a361b40c217875 10;15x30 a82212b7b5965084 11;15x30 fb19a6460801bcd1 22;15x30 8c6bc02e11644b0c 8;15x30 4ddbb02c0cc3595d 8;15x30 a466180dfc818ff7 27;15x30 7bab5465434c06ef 24b;15x30 c593d327536a3f95 13;15x30 7f31897e35513b90 20;15x100 166b8831586c03be 24;15x100 241b084edeb31475 26;15x100 8e8dfe3a581aca7b 26;15x100 e04f2259380e7843 23b;15x100 80a3d0ed4e3fe1b6 23b;15x100 56f67abbdcf721c0 23b;15x100 d041a8523e41867a 23b;15x30 31829823dbca91be 9;15x30 ebd82a7246664cb3 9;15x30 617a743973203ba5 13;15x30 9dd8bf11dbeb059d 13;15x30 98a51bd37ada144f 16;15x30 b7023e98328fd896 17;15x30 5af96087f4bde3f1 11;15x30 f4188305eab4b35a 12;15x30 e2c07bb48a84aca6 17;15x30 f3a0eee0f6a88562 16;15x30 222009426ce42a68 28;15x30 651b52f0277a26d4 17;15x30 d7e04357bfc8ab6f 17;15x30 af703196608d7e7b 28;15x30 7090b398b7f5359f 13b;15x30 a150d62c5731d184 12;15x30 482811f3ea353637 12;15x30 c6347e1fe3112433 10;15x30 f6e7cad810c7d5f0 8;15x30 31b69dc4b0a4a7ae 11;15x30 e39d881022a1516b 10;15x30 9003522c430014de 22b;15x30 be5860ad6a6c9fa1 23;15x30 0cbc80ff7b64b0f0 0;15x30 8e246e258204d053 10;15x30 8038220fe0dd0259 8;15x30 96a0aaaf97bdb31e 10;15x30 45fdaeabcf8a24dc 6;15x30 2caa9fb6d7dafacf 9;15x30 2975c799b6a353aa 16;15x30 042b8e89014eee23 12;15x30 58da2f3a06ab1feb 24;15x30 1b4750680601d5f1 5;15x30 3f2dd84ec5981fab 5;15x30 9e972c918b8dd9a2 7;15x30 37cd4ce05e4577a4 8;15x30 70fd4abb316f41c0 5;15x30 7183a04cefadcbf2 5;15x30 a88297fd6fe85217 19;15x30 667a51377849902d 25;15x30 1e43b3a1f3cb1c62 25;15x30 67bf4310c333b636 26;15x30 cfa4fa8b87c25152 27;15x30 b261498878ae91a3 0;15x30 df6ca67701b87dd1 15;15x30 e8e733264990b10d 12;15x30 de28058dc6082064 0;15x30 4de91ace830a90f1 0;15x30 e853d3ce377432d6 0;15x30 9fb815fbaeb688a1 0;15x30 eef060376310dbb5 0;15x30 983acaf86e24a8d9 6;15x30 1adaf9dd299d59d5 6;15x30 a48ccf86ee53c935 9;15x30 88eb1dc4c462f870 17;15x30 f9fd1276ea67db65 4;15x30 dfad3770e1ae3659 17;15x30 fb0560a55a91a87e 17;15x30 0ffd400e888fc47b 18;15x30 d8c2bb64a16f538a 18;15x30 2d1961427e12bf8e 5;15x30 c8e7bd1a61beef09 7;15x30 572e68353af56bb3 3;15x30 5ef94c757184944e 7;15x30 44a28966d67c8041 7;15x30 b053758b88caa0de 7;15x30 b855da67a553af60 10;15x30 fb93f343c3958d88 10;15x30 ef27e16c3429d685 10;15x30 8e834ad523ab2884 10;15x30 91cab912e4437ffd 10;15x30 7f66afe10d589cac 16;15x30 72545fb33253f28c 0;15x30 85dab32ab938315c 2;15x30 d87a770e0185d23d 14;15x30 7240d51347b23f6f 15;15x30 f313f54d11ef5979 15b;15x30 a5c3e7a51bee18ab 15;15x30 0260963bef2684ef 15b;15x30 ffd96640241d5677 15;15x30 3196f64ac216cc07 6;15x30 be552235067e95a9 4;15x30 75380ba1da73b3e5 4;15x30 8dc4c4e83cada4b7 4;15x30 8399c1a9d7aa77e7 23;15x30 a5488bc3aff6197a 24;15x30 e70f6760268f30cc 23;15x30 b42bfb881c976497 23;15x30 da723d666b851fb4 25;15x30 5b74a22692b98046 26;15x30 7c477442817899a9 23;15x30 bb75406c6b5f34fa 12;15x30 af54addfa70bc4fd 12b;15x30 ce87212d60f1f4d2 6;15x30 f8f74fcc48c18aa8 27;15x30 3040647b2a128d19 11;15x30 f84f2416855f3c16 11;15x30 3877a3389f2563b8 16;15x30 9d69602aae89c050 4;15x30 916f2d7239e944a8 29;15x30 543f31ff573ee7bd 2;15x30 e1486404bd28377f 2;15x30 9c2f61db66b3aa79 2;15x30 1380e4234e4f854e 14;15x30 72137a441770cf97 14;15x30 67d5c397292aec4e 13;15x100 767375deba9be9a2 49;15x100 478d1e6025abf8ce 48;15x100 8f845a8215bc1a1a 48;15x30 f40a6b460b1f2b5a 10;15x30 b0baaebab6d622c0 10;15x30 390a21c1fd69e5b9 18;15x30 f355a326e14ea57d 24;15x30 eb8457ec7922f8e2 20;15x30 82fa7fac627b0e2f 19;15x30 f70af70a48ac5957 15b;15x30 447269d5ec5b83cd 11;15x30 fabb19dc14b884e1 18;15x30 fb01856a3de71c94 18;15x30 e77b4ee9b1d90d2b 10;15x30 2ee3c33226761f21 7;15x30 88ed708cd3b61cb7 5;15x30 cd655aad5c4ab2c6 24;15x30 e2462cf223922733 23;15x30 da75b6c2a7c9f9c1 21;15x30 48411839242708a8 21;15x30 e4f8664e49e2386b 23;15x30 f2223235aff8fc23 17;15x30 610280d1847693f0 8;15x30 03bea66239be752f 17;15x30 347acb7b04f638c4 8;15x30 82f0182d11e244a8 8;15x30 84f585379d5dd97e 9;15x30 0db73ac8911d8f6b 13;15x30 95b3cf8f1fceb15d 10;15x30 cb061266a527fb6e 10;15x30 6790d02aae9418f6 10;15x30 1af6530e9a34e48e 10;15x30 e8712c249be0102c 10;15x30 5310a43a6b59966b 0;15x30 c38be7d1cc8181d0 13;15x30 f79d077312ca84a0 15;15x30 e09c37ede20739b3 16;15x30 cbb0ab97b2d8a9ac 16;15x30 e06249f8afe8949c 12;15x30 6f398e33ec52bd9a 12;15x30 7fb5e92883f216a6 7;15x30 55b174deb5222bf9 12;15x30 5b9bb7b9d263386b 7;15x30 55f1714fb502e6c7 7;15x30 0ab7ee2f61c294fc 14;15x30 3b2475bf0cd5c7e9 14;15x30 887960065d39acbd 18;15x30 a50a7380bda067c0 18;15x30 a58830cd45648f0b 11;15x30 6242b1348c600c1d 19;15x30 7bdb68ddbf086448 18;15x30 07aac11f95d8c22f 9;15x30 55d14448ba8f73df 17;15x30 32525fae2764cfbe 6;15x30 0ca8fefcff30f829 5;15x30 2ac01d67f14cc6be 9;15x30 e65d91d8f9a7e47e 14;15x30 e7095f473955fd08 14;15x30 53028548a37e7f6b 14;15x30 8eb6ff4589d7335d 13;15x30 81e46dda81234e15 12;15x30 a2f593aa1dcf4bbe 18;15x30 ed18ca38f832b75b 18;15x30 9fa9dc8b2fa66841 25;15x30 3c133066c37f5f26 20;15x30 0d1d3298faf82c78 25;15x30 1624ef3818c3a4cf 11;15x30 24d202041d211fc3 7;15x30 8ed9b6b354e5491b 7;15x30 1448c178bf295499 7;15x30 1ed7ca87d3f85e2c 12;15x30 db0fac996cb13b60 7;15x30 d254489471f7576c 7;15x30 47f53135e9f5110c 25;15x30 02f16b8673f92cc1 16;15x30 a50cefdc1add0f51 16;15x30 7b33d2c3ba25006f 16;15x30 93ce86d53dcf90d7 11;15x30 4207537012b11fb9 8;15x30 eabcc888e44d0f98 8;15x30 b91f564449d176b0 8;15x30 42cd256bbdf3367b 27;15x30 53a439209d646063 2;15x30 6fd75f4a67c6b273 2;15x30 35c880b1252b31c3 2;15x30 07d90835551f7756 7;15x30 87984b25e17d9b41 16;15x30 2ca9387967a78193 7;15x30 5d25349b06144a71 4;15x30 538fc7c87687b5a8 4;",
    };
    // LOSS BOOK END

    /** This board size's entries, sorted by key for binary search; parsed on first use. */
    long[] bookKeys = new long[0];
    int[] bookActions = new int[0];
    int bookRows = -1, bookColumns = -1;

    /**
     * The booked action for a position, or NO_ACTION. Entries carry their board size, so a key is
     * looked up only among its own size's; the book holds no line length, so it is used only at the
     * one it was learned at.
     */
    public int bookAction(long key) {
        if (LINE_LENGTH != TRAINED_LINE_LENGTH) return NO_ACTION;
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
        depthReached = 0;
        for (int[] scores : historyScore) for (int a = 0; a < scores.length; a++) scores[a] >>= 2; // age old knowledge
        for (int[] killers : killerActions) killers[0] = killers[1] = NO_ACTION;

        int best = firstLegalDrop();
        previousBestAction = NO_ACTION;
        for (int depth = 1; depth <= maxDepth; depth++) {
            rootBestAction = NO_ACTION;
            int score = negamax(depth, -WIN - 1, WIN + 1, 0);
            if (outOfTime) {
                // The unfinished iteration searched the previous best first: a root move that beat it in full is better still.
                if (refinedSearch && rootBestAction != NO_ACTION) best = rootBestAction;
                break;
            }
            int fromTable = tableAction(positionKey());
            if (refinedSearch && rootBestAction != NO_ACTION) best = rootBestAction; // even if the root's table slot was taken since
            else if (fromTable != NO_ACTION) best = fromTable;
            previousBestAction = best;
            lastScore = score;
            depthReached = depth;
            if (Math.abs(score) > WIN - MAX_PLY) break; // a forced result: deeper search can't change it
        }
        return best;
    }

    private int firstLegalDrop() {
        for (int column : columnsCentreOut) if (columnHeight[column] < rows) return action(column, false);
        return NO_ACTION;
    }

    /** Static evaluation for the side to move: line values, bomb readiness and (with the pattern table) the move itself. */
    public int evaluate() {
        int score;
        if (networkOn) score = networkScore();
        else score = (sideToMove == 0 ? staticScore : -staticScore)
                + bombReadyBonus * bombBonusPercent / 100 * (isBombReady(sideToMove) - isBombReady(1 - sideToMove)) + tempo;
        if (varietyNoise > 0 && quarkCount < varietyQuarks) score += varietyNoise();
        return score;
    }

    // ------------------------------------------------------------------ variety

    /**
     * Largest evaluation noise in the opening, in points; 0 plays alike every match. Rivals can read
     * our matches: a bot that plays alike every time loses again to a line that beat it once. The
     * noise is fixed per position within a match, so the search stays consistent, and new each match.
     */
    public int varietyNoise = 0;
    /** The noise applies while the board holds fewer quarks than this. */
    public int varietyQuarks = 30;
    /** This match's noise: a new seed for each bot made. */
    public long varietySeed = System.nanoTime();

    private int varietyNoise() {
        long mixed = (positionKey() ^ varietySeed) * 0x9E3779B97F4A7C15L;
        return (int) ((mixed >>> 33) % (2 * varietyNoise + 1)) - varietyNoise;
    }

    private int isBombReady(int player) {
        return bombCooldown[player] == 0 ? 1 : 0;
    }

    // ------------------------------------------------------------------ the pattern table

    /** Evaluate lines by their patterns where a table is embedded and LINE_LENGTH is the trained one. Off: the line weights play. */
    public static boolean usePatterns = true;
    /** This bot's copy of {@link #usePatterns}, taken when it is made and read when a board is set up. */
    public boolean withPatterns = usePatterns;
    /**
     * Patterns of a line of the trained length, 3^6. The table is used only at that length; at others
     * {@link #linePattern} is still kept (3^LINE_LENGTH, wrapping harmlessly past 20) but never read.
     */
    public static final int PATTERNS = 729;
    public boolean patternsOn;
    /** The value of each pattern per direction class, from player 0's point of view, at {@code direction * PATTERNS + pattern}. */
    int[] patternValue;
    /** Score of having the move, from the pattern table; 0 with the line weights. */
    int tempo;

    /**
     * Decodes {@link #PATTERN_TABLE}, if one is embedded. Layout, little-endian: version (a byte); the
     * value of every pattern of every direction class (int32, 3 x 729); the bomb readiness bonus and
     * the tempo (int32 each).
     */
    private boolean loadPatterns() {
        byte[] blob = Base64.getDecoder().decode(String.join("", PATTERN_TABLE));
        int values = 3 * PATTERNS;
        if (blob.length != 1 + 4 * (values + 2) || blob[0] != 1) return false;
        patternValue = new int[values];
        for (int i = 0; i < values; i++) patternValue[i] = littleEndianInt(blob, 1 + 4 * i);
        bombReadyBonus = littleEndianInt(blob, 1 + 4 * values);
        tempo = littleEndianInt(blob, 5 + 4 * values);
        return true;
    }

    private static int littleEndianInt(byte[] bytes, int at) {
        return bytes[at] & 0xff | (bytes[at + 1] & 0xff) << 8 | (bytes[at + 2] & 0xff) << 16 | bytes[at + 3] << 24;
    }

    /** The pattern table in base64. Generated; do not edit by hand. */
    // PATTERNS BEGIN (generated by tools/qc-net)
    static final String[] PATTERN_TABLE = {
            "AQAAAAArAQAA1f7//+8AAACTAgAAlP7//xH///9sAQAAbf3//+QAAAC7AQAAUf///5EBAACyAwAAdf7///j///9fAAAAkP///xz///+vAAAARf7//wgAAABwAAAAof///2/+//+LAQAATvz//+QAAACzAQAAv////2cBAABIAgAAHAAAAOz////n////RQAAAAIBAABlAgAATf///5UCAAAbBQAA0v7//+////9F////mP///wAAAAA5AAAAZwAAANb///+QAAAAAQAAAMX///80AAAA7v///xz///9BAAAATf7//xQAAAC7////GQAAAJn+///k////uP3//wAAAACZ////x////zsAAAASAAAAzP///yoAAAD/////cP////7+//+zAAAAm/3//xEAAABoAAAAuwAAAGv9//8uAQAA5fr//+8AAAALAQAAEAAAAFQBAACCAQAAFQAAAAAAAADl////PwAAAGcBAADJAAAAEwAAAIcBAADUAgAAhQAAAKX///9+AAAAmf///xQAAAA2AAAAov///+T///+XAAAABAAAANf///9yAAAARAAAAJEBAAC0AQAArP///4cBAADGAgAAagEAACkAAABIAAAARQAAAJUCAACJAQAA8f///04AAACGGgAAmQAAANkAAACi////w////zsAAAAgAAAAbQAAAGIAAAB2AQAAZAAAAAAAAABFAAAAq/7//wgAAABfAAAAGP///+T///8lAAAACAAAAFsAAAAtAQAAif///9b///+x////uP///2IAAABBAAAAVQAAAAAAAADtAAAANAAAABEAAABuAAAAsf7//8L///+2AAAA8f///yf///8nAAAAmv///xH////w////9f7//wAAAADB////GwAAAKz+///r////fv7//+z///9eAAAAyv///ykAAAC8////jv///xwAAAD8////af///5n+///t////N////1sAAABnAAAAgv///3n+//97////LP3///j////oAAAAof///6X///93AAAA0/7//xwAAAD4////2////+////9PAQAAkv///9kAAABmAAAA2f///z4AAAAPAAAASv///yoAAABIAAAATwAAAAAAAADM////E////57///+r////v////2/+//9UAAAATP7//9f///+7////uP///3n+//+W/v//Ov3//8X///+T////4P///wAAAABVAQAAu////57///+c////iv7//2v9//8PAAAAd/7//yf///89AAAAXgAAALL///9n////euX//ysBAAB8AAAAAAAAAAsBAABpAQAA2gAAAPD////y////FwAAALMBAACyAQAAQwAAALQBAAAjAgAA/P///+gAAABGAAAA7v///0EAAAAx////KAAAAF8AAACb////YAAAAFQAAAABAAAAa////7sBAACyAQAA2P///8kAAABVAgAApf///14AAAD5////JAAAAGUCAAD7AQAAdAEAAIkBAACgEgAAGAAAAE8BAACwAgAAPQAAAJn///+/////AAAAALH///+8AAAAvgAAAJP////lAAAAOwAAAK8AAAAx////vf///zYAAAD1////fP///+3////x////u////zkAAAC/////AAAAACAAAABBAQAALwEAAEgAAACc////dwAAALMAAAA7AAAAjP7//24AAAD0AAAAzf///w8AAADd/v//NP7//5MCAABpAQAA6f///4IBAABaAgAAp////8H///9fAAAAAAAAAEgCAABVAgAARQAAAMYCAAC7EQAA1wEAAHcAAADEAAAAxAEAALv////1////3P///yUAAABuAQAALAEAALv////bAAAAav///7IDAAAjAgAAlQAAANQCAAC7EQAA2f///7z///+PAgAAlgAAABsFAACgEgAAzAEAAIYaAAAAAAAAQQIAAGYAAAALAwAA8wAAABIAAABBAQAAxf///0EAAAB9AgAARf///1UBAABc////AAAAAHAAAACb////EgAAAJcAAABuAQAAmAAAAGcAAACSAAAAPP7//5AAAAC8AAAAif///3YBAAB9AgAA8////8z///+tAQAAAAAAAGgAAAD0AAAAw////7YAAADG////5v///z0AAAATAAAADf///2wBAADy////Jv///+X///9fAAAAAAAAAOv///8/////WQAAAOf////5////hAAAAEgAAACPAgAA6P////j///+VAAAAaP///+T////x////WwAAAC0BAACSAAAA6P///5b+//86/v//JwAAAF8AAABGAAAAoP///34AAADEAAAAGAAAAPz///+VAAAA1P7//0X///+wAgAAMwAAAKL///8LAwAAhP///w8AAAC9AAAAGgAAAP////+c////Qv///+0AAACtAQAAAAAAAJz///9u/v//uwAAAIsBAAABAAAABAAAAHIAAADbAAAAGAAAAHv///86/v//Kf7//zQAAADlAAAA0f7//0UAAABc////AAAAAKv///9u/v//DQAAAC4BAADd/v//6P///ycAAAATAAAAfAAAAGf///85////v/3//9X+//8AAAAAhP///xAAAADp////DgAAAPX+//8m////l/7//7/////Y////zwAAAKz///+VAAAA/////6H///+g////ZQAAAE3+//+9////Tv7//xj///8SAAAAuv///0z+//8EAAAA3f3//1H///9DAAAAzwAAABMAAABFAAAADwAAAMr///+EAAAACwAAAE3///90AQAAxf////H////MAQAAIwEAAJL///8zAAAADP///8f///8AAAAAQQAAALj///+J////ZAAAAOD////R/v//v/7//0X+//8oAAAATv7//6L////c////BwAAADf///9bAAAAq/3//2cAAAAAAAAAQQAAAG0AAADF////G////08AAABC////RP///5v9//+M/v//Bf7//7H+///D////UP3//3f+///o////YO3//5T+///aAAAADgAAABUAAACn////wQAAABsAAAAAAAAAof///xwAAACl////DwAAAGoBAADZ////xgEAANP+//8YAAAAbv///xkAAAB8////BwAAAAgAAACYAAAAa////7j///8YAAAAcf3//3X+///8/////////4UAAADXAQAAxgEAAI7////o////Jf///9L+//8YAAAAIwEAAJkAAABBAgAAxwAAANn///+E////7f///8z///8vAQAAG////1UAAADz////kgEAALv///8AAAAApAAAAKH///9gAAAAuv///wQAAAAsAQAAa////4L////o////PP///wEAAAC+AAAAZAAAAGQAAABF////kgEAABP///8AAAAAU/7//7sAAADN////UP3///H////m////Q////14AAAB8AAAA9fz//239//8XAAAAl/7//z8AAAAAAAAAof///37+//9ZAAAApv3//0UAAAAkAAAACwAAAEUAAACWAAAAJf///9v////U/v//kv7//7j9//+7////q/3//4n///88/v//PP///zr9//8p/v//Re7//5D////u////ZQAAAJn////EAQAAbv///2n///9o////kv7//5j///89AAAADP///8P////zAAAA7f///0r///8aAAAAOgAAAHD///93AAAARP///zQAAAAAAAAAU/7//4r+//8NAAAAg/3//078//9r////3f3//0QAAABq////cf3//yz9//8nAAAARe7//+7///87AAAAv/7//6v+//8AAAAApAAAAL////+7AAAAg/3//+X6//80/v//YO3//5r///8N////9fz//3rl//+//f//AAAAAAAAAABDAAAAvf///xEAAAAXAwAAzP///+////80AAAA6fz//xEAAABLAAAAAAAAAEsAAADBAwAApwAAAAAAAACkAAAAMv///+////8AAAAAtf///wAAAADOAAAAXP///7X///9Z////P/z//xEAAABLAAAAAAAAAEsAAADMAAAAAAAAAAAAAAAAAAAAAAAAAEsAAADMAAAAAAAAAMwAAABBAwAAZwAAAAAAAAAbAAAAIv///wAAAAAAAAAAAAAAAAAAAAD3AAAA6v7//wAAAAAoAAAAiv7//+////8AAAAAtf///wAAAAAAAAAAAAAAALX///8AAAAANP///wAAAAAAAAAAAAAAAAAAAAB2AQAA2P///wAAAAAWAQAACf///7X///8AAAAANP///wAAAADeAAAA5f///zT///+Z////v/z//xEAAABLAAAAAAAAAEsAAADMAAAAAAAAAAAAAAAAAAAAAAAAAEsAAADMAAAAAAAAAMwAAACSAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEsAAADMAAAAAAAAAMwAAACSAgAAAAAAAAAAAAAAAAAAAAAAAMwAAACSAgAAAAAAAJICAAAuIAAAcgAAAAAAAAAWAAAA8v///wAAAAAAAAAAAAAAAAAAAAAaAAAAuv///wAAAADO////rP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWAQAATAAAAAAAAACA////BgAAAAAAAAAAAAAAAAAAAAAAAAAsAAAAGQAAAAAAAAD4////z/7//+////8AAAAAtf///wAAAAAAAAAAAAAAALX///8AAAAANP//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/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB6AwAA+AAAAAAAAABY////Vv7//wAAAAAAAAAAAAAAAAAAAAD6////6P7//wAAAACi/v//vP7//7X///8AAAAANP///wAAAAAAAAAAAAAAADT///8AAAAAbv3//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADT///8AAAAAbv3//wAAAAAAAAAAAAAAAG79//8AAAAAlN///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABrAQAALAAAAAAAAADH////jv///wAAAAAAAAAAAAAAAAAAAAA3AQAApf7//wAAAADg/v//Uf///zT///8AAAAAbv3//wAAAAAAAAAAAAAAAG79//8AAAAAlN///wAAAAAAAAAAAAAAAAAAAABuAAAAD////wAAAABSAAAAH////279//8AAAAAlN///wAAAAD/AAAAXAAAAJTf//+jAAAAAAAAAAAAAAAMAAAA9P///3EAAADyAQAABAAAAI/////8////Dv7//14AAAC3AQAA7f///w4CAACIBAAAAQEAALz///+HAAAAMv7//6L///8TAAAASf7//0QAAADOAQAAef////L9/////v//ePv///H///+JAQAAYf///yUBAABbAwAAmf///54AAADM////9/7///EAAABRAQAAzP///3gBAAC9BQAAiP///2r///8i////LAAAAAgAAAB0////pf///0z///8bAAAAsv///4X////0////h/7//w8AAACfAAAAd/7//2L///8JAQAANAAAANv+//9nAAAApfz///j///9bAAAAjAAAAHsAAAB5AQAADAAAALQAAABOAAAA5f///w////80AAAAr/7//5YAAADU////3gAAAIj+//94AAAAQ/r///7///8DAQAAk////38BAACfAQAALgAAAEAAAABpAAAAO////z0BAAA1AQAArv///y8BAACUAwAALQAAAIwAAABZ////nf///9H///8TAAAALAAAAJj///9aAAAAv////63///99AAAABf///+oAAADjAQAA9f///6gBAACxAgAA4v///zkAAAA5AAAAXP///4kBAACBAgAAxv///ywCAABNIAAAngAAAJX///8DAAAATf///3MAAABe////m////zYAAACHAQAAHAEAAIz+//83AAAAkP///yYAAABl////fP///2T///9tAAAABQEAAMv////wAAAA+f7//3wAAADy////7f///5X///8iAAAA0QAAAL3///9IAAAA0////y3////V/v//nv///+L///8kAQAAzf///xIAAAA3////Bv///wIAAABtAAAA/f7//8D////FAAAAl////4H+///S////Yf7//y8AAADU////7f///1MAAAD7AAAAg////2gAAABBAAAApv///8P+//9SAAAAy/7//3T///9jAAAApwAAANH+///T////bPz//9r///+EAAAAmwAAADUAAAAHAQAAEP///5wAAAD7/v//k////9MAAABiAAAAKwEAAO7////6AAAAyQAAAB4AAAAzAAAA3P7//4T///8TAAAADgAAAEMAAAAtAAAAuP///2sAAAAv////3v///xb///8LAAAAHf7//8f///+kAAAAx////1j+//8eAAAAT/3//43///9lAAAAogAAAHQBAABwAAAAyf///8r////k/v//ef7//3f+//86AAAAf/3//2sAAACzAAAA/f///9T9//9i////s9///zcAAABtAAAAqQAAAEsAAADiAQAAawAAAHH///+g////P////+EAAACYAQAAUQAAAK4BAABIAwAAi////30AAAC3////zgAAAIsAAADl////AwAAAJX///9xAAAAdP///4gAAABcAAAAbP////gBAAArAQAAOAAAAEoBAAB6AgAA3v///4D///8kAAAAbP///4ABAADdAgAArP///wwDAACnHQAAPf///6L///9/AQAAMf///+////8ZAAAAyP///ygAAABxAAAArP///63+///D////5P7//2cAAABu////EQAAAEwAAAADAAAAHQAAAIMAAABBAAAAtP///4IAAACp////owAAADIAAADhAAAA5P///6r///+a////Fv///+kAAABd////Vf///1gAAAC6AAAALf///0wAAAAhAAAA+v3//+IAAADhAQAA0v///5wAAABBAwAAFgAAALb///+g////ugAAAOEBAABbAgAA1////3sCAACrHAAAOwAAALP///9bAAAAr/7//yEAAADY////2AAAAJcAAAACAQAAn////8r///+g////MgAAALoBAAB7AgAAb////6kCAABPHAAAxf///wsAAAAGAAAAY////2YDAABKHQAAYf///+oeAAAAAAAAnQAAAGsAAAC4/v//8AAAAHb////j////Uf///9v///8iAAAA7f7//z/////C////RwEAAHsAAABAAAAAagAAAL/////EAAAAv/7//5L///8+AAAAvf7///z////sAAAACgAAABYAAABHAAAAigEAADwAAABSAAAAi////2YAAAC7/v//yf7//1oBAACU////hwAAAMAAAADCAAAAu////8f+///B////YQAAAAoAAABn////AwAAANf///9B////7P///yb///80AAAAxP///0YAAABLAAAApwAAAIP///9sAQAAfv7//yn///+1////HQAAAAkAAACdAAAA2f///1v///9i////eP7//3QAAAAoAAAASQAAAE0AAAArAAAAoAAAAGcAAAAVAAAALf///yD///9MAAAADP///z4AAADb////rP///6EAAADHAAAA2QAAAC4AAAAuAAAACQAAALf////W////lAAAANoAAACj////tv///6wAAAAFAAAAwv///8P////S////Rf///wMAAAAwAQAAzv3//zcAAABwAAAAyv////////8x////0////+z+///JAAAAZP///2AAAACsAAAA5v///3oBAAAh/////P///47////O////W////8n///9X////k////48AAADBAAAAYAAAALX///+V////Hv7//3X////9////GwAAAHj///+UAAAApP///2sAAACMAAAAj////x////+v////aP7//4P///8y////SQAAAFL+//91AAAAuPz//5n////v////kgAAAH3///9MAAAAv////7T////j/////f///xf///+rAAAAowAAALT///8GAgAA3////6j////TAAAARv///37///9d////VwAAAFYAAADqAAAAZgAAAM7///8cAAAAH////wj+///I////1f7//4AAAACUAAAA3P///7b+//8iAAAAhv3//xEAAAA4AAAA5////1MBAAAcAQAAPQAAANj///9UAAAAj////4D+//9UAAAAI/3//14AAADPAAAAgf7///T8///DAAAAWeL//zkBAACf////PwAAACkAAAAUAAAAvwAAAPb////9////mQAAANcAAADj////SwAAAKUAAACIAQAAngAAAPf///8nAAAAY////9oAAAA8AAAAzP///30AAACCAQAAlP7//7r///9Z////tf///1T///8+AAAA+/////3///8yAgAA0P7//z0AAAC7AAAALgAAAKD///8aAAAAVP///3IAAAClAAAAMgAAAIb+//8EAAAA3wAAAMn///82AAAAkP///xQBAACcAAAAN////wEAAAAtAAAAzwAAAIz///+3////2P///5n////TAAAA6////7P///9g////1f///9L////3////0v///yb///9KAAAAXQAAAEkAAABs////KgAAAOAAAAD0AAAAtP///1////8n////Of///8L///9UAAAAJQAAAB7///8uAAAAH/7//0oAAABG////YAAAAGT////q////v/z//9////8o////KAAAADYAAADO////YAAAAGn///9hAAAA/v7//x/+//8pAAAApf3//00AAABRAQAApf///4X9///F////VeP//4X///+W////wP///24AAABDAQAAwv///0EAAABBAQAAPP///5r///83AQAARQEAAED///9FAAAAPv///6b+//95////bAAAAAQAAAD2////FP///8T///91AAAArv///+r///92/v//uf///0b+//+RAAAAhf3///X///+dAAAA+v///1f9//87AAAAseP//4oAAACvAAAAHQAAAMEAAAC5/v//PgAAACUAAAATAQAA3v///5r8//+fAAAAtuL//5X///8Q////SAEAABbh//9j////AAAAAI8IAAAHBgAA",
    };
    // PATTERNS END

    // ------------------------------------------------------------------ the network

    /** Evaluate with the network where its board size and LINE_LENGTH match its training. Off: the line weights play. */
    public static boolean useNetwork = false;
    /** This bot's copy of {@link #useNetwork}, taken when it is made and read when a board is set up. */
    public boolean withNetwork = useNetwork;
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
    /** Weight row of each feature (int8 values, while loading), output weights (int16), output bias. */
    short[] inputWeights, outputWeights;
    int outputBias;
    /**
     * Rows summed ahead for speed. The empty board: hidden biases plus the bottom row's playable
     * cells. A quark, and a threat, of a player at a cell: what it adds to both accumulators at once,
     * player 0's half then player 1's, at {@code (2 * cell + player) * 2 * hidden}; a quark's row
     * includes its column's playable cell moving up. The cooldowns and first mover, at
     * {@code ((own bucket * buckets + opponent bucket) * 2 + 1 if first mover) * hidden}.
     */
    short[] emptyBoardRow, quarkRows, threatRows, situationRows;
    /**
     * Both players' accumulators in one array, player 0's half then player 1's: the empty-board row
     * plus the rows of the player's active cell features (int16).
     */
    short[] accumulator;

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
        int base = mover * hidden;
        int situation = ((cooldownBucket(bombCooldown[mover]) * COOLDOWN_BUCKETS + cooldownBucket(bombCooldown[1 - mover])) * 2
                + (mover == firstMover ? 1 : 0)) * hidden;
        int output = outputBias;
        for (int i = 0; i < hidden; i++) output += Math.min(Math.max(accumulator[base + i] + situationRows[situation + i], 0), ACTIVATION_ONE) * outputWeights[i];
        return (int) ((long) output * SCORE_PER_LOG_ODDS / (ACTIVATION_ONE * OUTPUT_ONE));
    }

    /** Adds a double row (both players' halves) to the accumulators: one flat loop, which the JIT can vectorise. */
    private void addRows(short[] rows, int at) {
        for (int i = 0; i < accumulator.length; i++) accumulator[i] = (short) (accumulator[i] + rows[at + i]);
    }

    private void subtractRows(short[] rows, int at) {
        for (int i = 0; i < accumulator.length; i++) accumulator[i] = (short) (accumulator[i] - rows[at + i]);
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
        quarkRows = new short[4 * cellCount * hidden];
        threatRows = new short[4 * cellCount * hidden];
        for (int cell = 0; cell < cellCount; cell++)
            for (int player = 0; player < 2; player++)
                for (int half = 0; half < 2; half++) { // half: whose accumulator, the player's own or the other's
                    int row = ((2 * cell + player) * 2 + half) * hidden;
                    boolean own = half == player;
                    addRow(quarkRows, row, (own ? OWN_QUARK : OPPONENT_QUARK) * cellCount + cell, 1);
                    addRow(quarkRows, row, PLAYABLE * cellCount + cell, -1);
                    if ((cell + 1) % rows != 0) addRow(quarkRows, row, PLAYABLE * cellCount + cell + 1, 1);
                    addRow(threatRows, row, (own ? OWN_THREAT : OPPONENT_THREAT) * cellCount + cell, 1);
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
        accumulator = new short[2 * hidden];
        inputWeights = null; // summed into the rows above
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
            "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AP//Af//AAD//wH//gH9APwIAPn//QT6Af8AAQQA/v4DAAQB/QD5//4D/wIB/wEAAP/9BQII//0AAP0BAwH8AgIA+/z9AgX/AgUBAPsB/gQC+P8BA/oDA/4G/v7///oEAgcA+/0DAgEAAgAD/QT9/v4A/gMBBfz+/QD//QYC+v8ABf0A/AEGAPz//wT/AP8E/gD9+wH+Bv//AgH9Awf/AQECBAH//v79AgAAAQH+/wIBA/z+Bf4BAgD8/AEB/wH+/P73AwD7Af/6/wECA/oBAP8A/wIB//gAA/r/AQH9A/4A/v77Af4DBQAD/wL/CP3/Av/7/QMDAgX///79AAIB/gAF/f0B//8D//0AAv8G/f3/AP75A/4CBAP/+QX+AAQAA/4GAQUFA/7+AP4B/wX+AwAC/vcD/wj+//8A/gT/+wMBA/n5AQL9/f/7/wP9AQYB/fj9/wEACgH+/P7+AQECA/8C/wH8AgT9/wIEAf38BAD9Av79/QD8APsB/PoCAgEEBAcAAgUGBf3//wQBAQL9/f8B+vQBAwgCAvz+/gH+AAQC/QEDAP4F/v4CAgICAAL/AQP7AgL9BQAC+fr6Af7+AfwA/gcC/gD/AP4DAf8BBAIA/gP8AP//AP/+AgD/BQEABP7/AP4BAgH//gMD+f7/Bf0B+gT/AQL/AwMA9v8DAAMDBf0F/QMB/f7/A///+wf9AAEB/gT+/v7///kD/fz7/gAE+wcAAAj+//8E//wB/wIDBPv+AgAFAAMH/AL/AQH+/wD9AQMBBP0CBQQEA/7//f/9AAUEAwUG/gP+AP4CAv0B+voEAvwBAP0A/gH7/wACAwUB/QP+BwEGAvwAAQX+/v78+fz/AAT8AP8H/gAB/gP/Agf+BPsDAP/8+vsE/AIAAPf/BwEAAP3/AP8C/PwBAv/+/fsB+gT4//75+v8C/wIB/f8D/AP9/QL/AQD/BAf/Af/7AP0EAQH//AABAgkC/wAG+gL+AAX8/wAAAAADAQAA//73AP76Av/9+/8CAwEAAv399P4AAP7//gX+AggEBAMB/v4D/QAE/gD9AAP3AQEC+/z8Af0B/fwAAwP9AQMBBf3+/AIAA/8A+f/8Afr8AgT/9vj9AAL6/wACB/7+AAQA/f38AgEC/f/9AP4B+gL/AgP7//8BAwD+AAcE//4CAQP9AfsC/f8C/AMC/gP/AgMC+//6AP8EAP/9/wIA/gEBAP/4/QH+Av0A/QMA//cBBQIFAfz9AAD9/AECAQAA/gUC//79B/0ABwADAQAC9wIAAvX9+//+AAD8/QQGBAEBAQMAAQP/+QMC/wH+Av7+AQMEBPgBAv76/fr+AAEAAgT/+wIE+v/4/wD7Av8ABfgAAAD5AQD9/gUD/f4CAAD+AP8FA/sA/v/8/P4E/f4D/QEAAP8C/P3+Af37AP//AwcD+///+/8G/wEAAP39+/33/f0AAfz7AP4A/v8DAAICBvwDAwP+/Qb+/f0AAP3///38/v/9+gP//wf8+wX/Bv4C/wIAAP78BfsMAgD/Af7+/QH8/Pz7AAcB//3+/P0DAwADAwP+AwMF/AH8Af4C/P73/wL7+gL9APsB//8BAQEE+gQAAAP///z+/gT//QIBBP8D/wL//QAB/QH/AP//AAD7/gUB/AD/APz4AQMBAQEAAv4BAwP8/gABAP4C/AD//QAAAgL8+v3+APcBAP0D/AEAAAMBBQAB/QMC/f/9BAP+AAMDBvgB/QH//v7+/vz//AMBAv/+/gQFAAEE/wT/AP77//8E/Pr8/wP9/v0F/QIB+/sDAP379/8DAfv9/AL//v0CAwD5/wL//v/8//0B/v/9AQMC/QMF/AAEAvz9Av3/9wMCAwD7+f7+/QH//wACAv/+AP/9+v4ABP4C/PsF/f38BQP+AfoFAwEBAAP8/wH//P8BAP8CAP3//wD7/AIABAD//wH///r9/vf//v4CAf/9/gQB/PwCAAAB/v7+AP/9AQD//gP6BAIABf/++f/6/gT+AAEF+v0B/wP+//4DBAD9/v4C/f3+APoAA/0CAQQA/wICAwX+A/39/v8A/gEA/v38AAH/+/sB/v4A//3//v3//f/+AP/2/QAD/wP9Av3+A/4DAv////oD/AIDAv/9BAMJAf0EA/0BAQT8AP4C/gEBAAQBAv7/BP/5AwL/+/3+/P/9AwL/+v0DAf79/QL9Afn+BAb/+/4DBgUB//76+wH6/AIAAQH0AAcD/wD8AgT+AQIBAvcCCAABCPwBBQEBBAL/AgP8AAED/AH/Af8E/wL/AAP+AP8CAgMC/AD+/P0C//sFAAIBAv4BAwP+AQD+/gIBAAP//v///f4A/wQDAgUL+v8BAf8AAAADAgEDAQQC/wIF/QP/+gMC/vsC/wH7Av3/AQIDBPr7/wX/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BAADAf0AEP8E/vj9A/z8APwFA/r9/gb9Av8D+/0A/f8EAAL6AAEF/QACAf7/+QMGAAP8AQMCCP/7/wMB//v/AP7/AfkF///7/f8A/AT9/f7/BQL4+wIG/QT6AP37AP0EAP4A/QL/A/39/vwC/wH+AQEA//8C/QT/AwUBAP77AwD/AwD+/v4BAPv9/wABAgD/AQQA//3/BgAB/QL//v0AAQP+/QIA/wAB/wQA//4F/wQAAAD3/AAB//z/AQH+9wAEAQL/+AL9/Pj8+gIAAAP+/wH9AgL8AwQAAAAEAwL/AgIDAAIBAP4C+wADBfsABAP9+wIB+v7/Afz9/wL//PgDAAMCAfkB/vz7/gMBAQYFAv///P/9+wMFAQP/AgH+BQIEAgT9+P/5BP/+Bv/7/QEB/wMF/vYAAgMAAfv/APz4AAL+Af78/gH7/gD/APkBCfkD////+/0DA/4B+gAC/vwF/QMDAf0A//sA/QEC/AIDAwD////+AAIF/wAB/wEA+wT/AgcC+AUB/gH9Av8B/wD/+gL++vsCBAP7Afz//QH5BQICAf8BAwMB//wCAv0CAQH9//38A////vwFBP8EAP8DAQL/AfwAAQQA/P4HAgL9AP8C9f/6/Pv/+wIE/AH///wCAfwA/wEAAP8DA/f+/PwBAv4EA/oCAQH8BAAAAwH+/v8AAAH+AP0C/wP8AAH7/wQEAQP+9gEAAPwF/gD9/f0A/QX8BQL8AwQBAQUAAgAGAgQBBvz4/f/9AAICA//++wAAAAj8AAMABfr7/v8CAQIBA/7+AfL//v0AAvwDAP7///4DBwL6Av0C/P8D/QH9A/wAAP//Af79AgECAQQC/PwB//8DAP78AAP+Afz+AgP+BQIAAAH+/gH+Av8BAv8ABQIA/gEB9gcDA/3+/vsG/gT+/wH8//gAAAP/+f/+AwT8AAH+AgH8A/4B/gD8Af0B/AQDA/z//wAC//wC/wIAAv4A/wECAAD/AwP6/fz8/v7//wIBBP4CA/39//0B/wAB+wQEAAMC/AL6BfwA/AH//v8E/AP//fX7BP0D/P3/AwP6AAAC//78Bf799f4A/P/9AAID/gf//QABAP/+Af77/AD+AQAD//8CAQX9//4C/P3/AQIDCAL7/wH/BAL7/wH/BQAFAgIBBAEE+/4BAP38BP/6Bv4D/fwAAv74/f4CAQACBv8DAP4B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APz7/v8EAQUAAP8FBQUE/QAF+///+wIAAgT8AAEDA/sBBAD+AAD9AQD//wMA/gH//P/7BwD8AQH8/AAEAwAA//4EAwD//gAAA/0B/AEB/gD/AAAAAvwABAAD/wEA/wQC/P/9/QD5AfsA/P0CBAQE/gIIAAT//AECAAACAQD8Bf39AQAB+wb+AQIBAP8EBv79/gD+Av/7AQADAwQC/vwCA//7BAAB/wH5//r+AP4C///7Af/8/gL//v0EAv39AAD/+/78/AIBAgEA/vz9Bv3/AgH9AAP/APkC/v7+AQD7AQQB/fz/AgL8AQH//QH/+/v9//8A////AwH8//8DAP/+BP4E/wD//v39AwT/BAH7//wAA/0H+wADAQUCAAUE+wX9AAQE+v4A/gMD/v3+/wL8/f4C/QP+/gH//AD9BP4CBQEB/QH8A/v8/v7//AICBgH8/gMAAf4AAAYBAwMCBf8AAf38+/8B+P4CAv4BAf4A//0CBQX/Av79BAEE9f8F//4A/v4B//3/APwFAwACAAAA/wEF/AEA/AP9AAL9AgP5BgUBAAL+/QD9/P4A/wMABf77AQb8/wH9AwQAAf8G/f0E/P8BAf/9/AMCA///AgIBAP7+A/7+AfsE/wMC/gQABAH++/7+/f/4BPr+AP8BAAAC/AIE/v/8/v8GAAL++wEC/wP/Af39/foF/v4FAPwA/P4C/QICAf4EAQYB/QQG/wADAgAA/v38Av0F/gICAQD9+//+BAL7Af0BBf0AAP8D/AQCAAH+//0B//wAAf4F/AT+//wDAPv99wL7/wP+AAECAQH++wL+AP0CBQD/Av0EA//9/QoDAAAB/fwBCAcDAvoC+/75AvsAAwD8/f4BAgAC/f3//f8AAP8BCPoE+wICB//9AP0AAQMCAwEEAQEDAAD//wAB+P0DAPv6/wEABgIBAP8D/v0A/AD///4AAAAF/P79/QUAAf4A/v//AgL9/AEBAP73/QP///4BBP3+AQD+/wH8Bv7+AfsC/QEB+QMB/AL/AvX8/AX/A/v6/P8CAgEA/QAGAgAA/wIB/gEDAAAFAwUC/QUC/gAB9gAA+wMCAP79AAj+AAP8AAgEAwX+/gMB/P76//7++//9/vz+Bf4BAf7/AwAEBP4H+/8CAAACBP4GAP0B/wP9AP0BAP0BBQQB/QAAAP4CAP/6/QH9///8AwIE/P8D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",
    };
    // NETWORK END

    // ------------------------------------------------------------------ alpha-beta search

    /** The score of the position for the side to move, searched to the given depth within (alpha, beta). */
    int negamax(int depth, int alpha, int beta, int ply) {
        if ((++nodesSearched & (CLOCK_CHECK_NODES - 1)) == 0 && System.nanoTime() > deadlineNanos) outOfTime = true;
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
        if (ply == MAX_PLY - 1) return evaluate();
        if (depth <= 0) return forcingLine(depth, alpha, beta, ply, forced);

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

        if (ply == 0 && refinedSearch && previousBestAction != NO_ACTION) tableAction = previousBestAction;
        int[] actions = actionLists[ply];
        int actionCount = orderedLegalActions(actions, tableAction, forced, ply);
        int originalAlpha = alpha, bestScore = -WIN - 1, bestAction = NO_ACTION;
        for (int i = 0; i < actionCount; i++) {
            int action = actions[i], score, column = columnOf(action), previousCooldown = 0;
            boolean bomb = isBomb(action);
            int reduction = refinedSearch ? lateMoveReduction(i, depth, action, tableAction, forced, ply) : 0;
            if (bomb) { // a blast rearranges several columns: save the whole state
                if (undoStates[ply] == null) undoStates[ply] = new int[stateSize];
                saveState(undoStates[ply]);
                play(action);
            } else previousCooldown = drop(column);

            if (i == 0 || !refinedSearch) score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            else { // principal variation search: is this move better than the best so far?
                score = -negamax(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
                if (score > alpha && reduction > 0) score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1);
                if (score > alpha && score < beta) score = -negamax(depth - 1, -beta, -alpha, ply + 1); // yes: by how much?
            }

            if (bomb) restoreState(undoStates[ply]);
            else undoDrop(column, previousCooldown);
            if (outOfTime) return 0;
            if (score > bestScore) {
                bestScore = score;
                bestAction = action;
            }
            if (score > alpha) {
                alpha = score;
                if (ply == 0) rootBestAction = action;
            }
            if (alpha >= beta) {
                historyScore[mover][action] += depth * depth;
                if (!bomb && killerActions[ply][0] != action) {
                    killerActions[ply][1] = killerActions[ply][0];
                    killerActions[ply][0] = action;
                }
                break;
            }
        }

        int bound = bestScore <= originalAlpha ? UPPER_BOUND : bestScore >= beta ? LOWER_BOUND : EXACT;
        tableKey[slot] = key;
        tableEntry[slot] = packEntry(scoreToTable(bestScore, ply), depth, bound, bestAction);
        return bestScore;
    }

    /**
     * Past the search depth: follows only forcing moves, to {@link #forcingPlies} plies deep. A side
     * facing a threat without its bomb must block it. Any other side may take the evaluation instead
     * ("stand pat"), or play a drop that makes a threat to complete at once. A side facing a threat
     * with its bomb ready stands pat: a bomb answers in too many ways to follow here.
     */
    int forcingLine(int depth, int alpha, int beta, int ply, int forced) {
        int mover = sideToMove;
        if (forced >= 0 && bombCooldown[mover] > 0) { // the one answer: block
            int previousCooldown = drop(forced);
            int score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            undoDrop(forced, previousCooldown);
            return score;
        }
        int standPat = evaluate();
        if (forced != NOT_FORCED || depth <= -forcingPlies || standPat >= beta) return standPat;
        int bestScore = standPat;
        if (standPat > alpha) alpha = standPat;
        for (int column : columnsCentreOut) {
            if (!makesThreat(mover, column)) continue;
            int previousCooldown = drop(column);
            int score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            undoDrop(column, previousCooldown);
            if (outOfTime) return 0;
            if (score > bestScore) bestScore = score;
            if (score > alpha) alpha = score;
            if (alpha >= beta) break;
        }
        return bestScore;
    }

    /**
     * Whether a drop in this column gives the player a threat it can complete next move: the cell
     * above was a threat already, or the drop fills a four whose other empty cell is then playable.
     */
    boolean makesThreat(int player, int column) {
        int height = columnHeight[column];
        if (height == rows) return false;
        int cell = column * rows + height;
        if (height + 1 < rows && threatCount[player * cellCount + cell + 1] > 0) return true;
        if (fourCount[player * cellCount + cell] == 0) return false;
        int opponent = 1 - player;
        for (int line : linesThroughCell[cell]) {
            if (lineQuarks[2 * line + player] != LINE_LENGTH - 2 || lineQuarks[2 * line + opponent] != 0) continue;
            for (int gap : cellsOfLine[line]) {
                if (gap == cell || owner[gap] != EMPTY) continue;
                int gapColumn = gap / rows, gapHeight = gap % rows;
                if (gapHeight == (gapColumn == column ? height + 1 : columnHeight[gapColumn])) return true;
            }
        }
        return false;
    }

    /** {@link #orderedLegalActions}'s {@code forced}: any drop may be played, none (bombs only), or only the column given. */
    static final int NOT_FORCED = -1, BOMBS_ONLY = -2;

    /** Moves before this one in a node's order are searched in full; later quiet moves may be reduced. */
    static final int FIRST_LATE_MOVE = 3, FIRST_VERY_LATE_MOVE = 8;

    /**
     * Late move reductions: with the best guesses first, a move far down the order rarely turns out
     * best, so it is first searched one ply shallower, two if very late, and in full only if it then
     * looks better than the best so far. Not reduced: forced answers, the table's move, the killers,
     * and drops that make a threat to complete at once. Returns the plies to reduce by.
     */
    int lateMoveReduction(int index, int depth, int action, int tableAction, int forced, int ply) {
        if (!lateMoveReductions || index < FIRST_LATE_MOVE || depth < 3 || forced != NOT_FORCED || action == tableAction) return 0;
        if (action == killerActions[ply][0] || action == killerActions[ply][1]) return 0;
        if (!isBomb(action) && makesThreat(sideToMove, columnOf(action))) return 0;
        return index >= FIRST_VERY_LATE_MOVE && depth >= 4 ? 2 : 1;
    }

    /**
     * Fills {@code actions} with the legal actions, best guess first, and returns how many there are.
     * The guesses: the table's move, then (refined search) this ply's killer moves, then the others
     * by their history, drops and bombs alike: a blast is often the move that refutes.
     */
    int orderedLegalActions(int[] actions, int preferred, int forced, int ply) {
        boolean mayBomb = bombCooldown[sideToMove] == 0;
        markColumnsNearQuarks();
        int count = 0;
        for (int column : columnsCentreOut) {
            if (columnHeight[column] == rows) continue;
            if (forced == NOT_FORCED && nearQuarks[column] || forced == column) actions[count++] = action(column, false);
            if (mayBomb && blastHitsQuark(column) && (bombLineQuarks == 0 || blastMovesLine(column))) actions[count++] = action(column, true);
        }
        if (count == 0) { // an empty board, or nothing near is playable: every legal action
            for (int column : columnsCentreOut) {
                if (columnHeight[column] == rows) continue;
                if (forced == NOT_FORCED || forced == column) actions[count++] = action(column, false);
                if (mayBomb) actions[count++] = action(column, true);
            }
        }
        int[] history = historyScore[sideToMove];
        int firstKiller = refinedSearch ? killerActions[ply][0] : NO_ACTION, secondKiller = refinedSearch ? killerActions[ply][1] : NO_ACTION;
        for (int i = 0; i < count; i++) {
            int action = actions[i];
            if (action == preferred) actionPriority[i] = Long.MAX_VALUE;
            else if (action == firstKiller) actionPriority[i] = Long.MAX_VALUE - 1;
            else if (action == secondKiller) actionPriority[i] = Long.MAX_VALUE - 2;
            else actionPriority[i] = history[action];
        }
        for (int i = 1; i < count; i++) { // insertion sort: stable, so ties stay centre-out
            int action = actions[i], j = i - 1;
            long priority = actionPriority[i];
            while (j >= 0 && actionPriority[j] < priority) {
                actions[j + 1] = actions[j];
                actionPriority[j + 1] = actionPriority[j];
                j--;
            }
            actions[j + 1] = action;
            actionPriority[j + 1] = priority;
        }
        return count;
    }

    /** Fills {@link #nearQuarks}: a column is near when a column at most {@link #dropReach} away holds a quark. */
    void markColumnsNearQuarks() {
        int lastWithQuark = -columns - dropReach - 1; // the nearest column to the left that holds a quark
        for (int column = 0; column < columns; column++) {
            if (columnHeight[column] > 0) lastWithQuark = column;
            nearQuarks[column] = column - lastWithQuark <= dropReach;
        }
        int nextWithQuark = 2 * columns + dropReach + 1; // and to the right
        for (int column = columns - 1; column >= 0; column--) {
            if (columnHeight[column] > 0) nextWithQuark = column;
            if (nextWithQuark - column <= dropReach) nearQuarks[column] = true;
        }
    }

    /**
     * Whether a bomb in this column would blast a quark besides its own: one below it, or one beside
     * it in a neighbouring column. A blast that hits nothing only spends the bomb.
     */
    boolean blastHitsQuark(int column) {
        if (columnHeight[column] > 0) return true;
        return column > 0 && columnHeight[column - 1] > 0 || column < columns - 1 && columnHeight[column + 1] > 0;
    }

    /**
     * Whether a bomb in this column would blast or drop a quark of a developing line: one holding at
     * least {@link #bombLineQuarks} quarks of one player and none of the other. The blast clears the
     * three columns around the drop from one below its landing height; the quarks above fall.
     */
    boolean blastMovesLine(int column) {
        int from = Math.max(0, columnHeight[column] - 1);
        for (int c = Math.max(0, column - 1); c <= Math.min(columns - 1, column + 1); c++) {
            for (int height = from; height < columnHeight[c]; height++) {
                for (int line : linesThroughCell[c * rows + height]) {
                    int first = lineQuarks[2 * line], second = lineQuarks[2 * line + 1];
                    if (first >= bombLineQuarks && second == 0 || second >= bombLineQuarks && first == 0) return true;
                }
            }
        }
        return false;
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
