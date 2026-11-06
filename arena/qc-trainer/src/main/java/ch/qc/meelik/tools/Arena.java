package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.Experiment;
import ch.cern.quantumconnect.core.ExperimentStatus;
import ch.cern.quantumconnect.core.GameConstants;
import ch.cern.quantumconnect.core.Move;
import ch.cern.quantumconnect.core.QuantumField;
import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.bots.QuarkBot;
import ch.qc.meelik.bots.QuarkV2Bot;
import ch.qc.meelik.bots.SerenaBot;
import ch.qc.meelik.bots.ThreatBot;

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.StringJoiner;
import java.util.jar.JarFile;

/**
 * Headless local arena: plays bots against each other in the engine's own Experiment (the
 * qc-starter library), the referee the engine ships with, and checks every game against our rule
 * model, {@link QuarkBot}.
 *
 * <p>Usage: Arena [botA=quark] [botB=random] [games=20] [size=8x12], the size as rows x columns.
 * A bot spec is one of:
 * <ul>
 *   <li>{@code random}: a random legal column, never bombs;
 *   <li>a bot name, the class {@code ch.qc.meelik.bots.<Name>Bot} ({@code quark},
 *       ...) or any fully qualified bot class, optionally followed by {@code @<ms>} (time budget per
 *       move), {@code /<depth>} (fixed search depth) and {@code :<file>} (evaluation parameters from
 *       Tune), e.g. {@code quark@10}, {@code quark/2}, {@code quark:params/evolved-d4.txt}; the same
 *       options work for {@code quarkV2}; {@code serena+net} is SerenaBot evaluating with its network
 *       (it loads when the bot sets up the arena's board, for the size it was trained on),
 *       {@code serena+pat} with its pattern table (any board size), {@code serena+lines} with its
 *       line weights; plain {@code serena} evaluates as the bot ships;
 *   <li>{@code tuned}: short for {@code quark:out/tune-params.txt};
 *   <li>{@code jar:<path>}: the bot in a packaged JAR (its Bot-Class), in its own class loader, so
 *       two versions of the same bot can play each other.
 * </ul>
 */
public final class Arena {
    /**
     * The server's limit per move: 50 ms since 2026-11-05, while the engine library still says
     * {@link GameConstants#TIMEOUT_MS} (100 ms) and referees local games by it. Answers over this limit
     * count as late moves; set another with {@code -Dqc.limitMs=}.
     */
    static final int MOVE_LIMIT_MS = Integer.getInteger("qc.limitMs", 50);
    static final long MOVE_LIMIT_NANOS = MOVE_LIMIT_MS * 1_000_000L;

    /**
     * One game's outcome: winner (0 first mover, 1 second, -1 draw), turns, per-player statistics,
     * and the quarks placed, written like "5,6,5b" (skipped turns place none).
     */
    record Result(int winner, int plies, PlayerStats[] players, String moves) { }

    /** A player's totals; a late move is an answer (column or bomb) over the time limit. */
    static final class PlayerStats {
        long calls, totalNanos, maxNanos, lateMoves, skippedTurns, bombs;

        void add(PlayerStats other) {
            calls += other.calls;
            totalNanos += other.totalNanos;
            maxNanos = Math.max(maxNanos, other.maxNanos);
            lateMoves += other.lateMoves;
            skippedTurns += other.skippedTurns;
            bombs += other.bombs;
        }

        String summary() {
            return String.format("avg %.1f ms/move, max %.1f ms, late moves %d, skipped turns %d, bombs %d",
                    totalNanos / 1e6 / Math.max(1, calls), maxNanos / 1e6, lateMoves, skippedTurns, bombs);
        }
    }

    public static void main(String[] args) {
        String botA = args.length > 0 ? args[0] : "quark", botB = args.length > 1 ? args[1] : "random";
        int games = args.length > 2 ? Integer.parseInt(args[2]) : 20;
        int[] size = Tools.size(args.length > 3 ? args[3] : GameConstants.DEFAULT_ROWS + "x" + GameConstants.DEFAULT_COLUMNS);
        int winsA = 0, draws = 0, winsB = 0;
        long plies = 0;
        PlayerStats statsA = new PlayerStats(), statsB = new PlayerStats();
        for (int game = 0; game < games; game++) {
            boolean aStarts = game % 2 == 0;
            Algorithm first = make(aStarts ? botA : botB, game), second = make(aStarts ? botB : botA, game);
            Result result = play(first, second, null, size[0], size[1]);
            int seatOfA = aStarts ? 0 : 1;
            statsA.add(result.players[seatOfA]);
            statsB.add(result.players[1 - seatOfA]);
            plies += result.plies;
            if (result.winner < 0) draws++;
            else if (result.winner == seatOfA) winsA++;
            else winsB++;
        }
        System.out.printf("%s vs %s, %d games on %dx%d (alternating who starts)%n", botA, botB, games, size[0], size[1]);
        System.out.printf("  %s wins %d, draws %d, %s wins %d, avg %.1f plies%n", botA, winsA, draws, botB, winsB, (double) plies / games);
        System.out.printf("  %-12s %s%n  %-12s %s%n", botA, statsA.summary(), botB, statsB.summary());
    }

    // ------------------------------------------------------------------ one game

    /**
     * Plays one game in the engine's own {@link Experiment}, the referee it ships with: it asks
     * each bot on a fresh thread, within the time limit, and applies the engine's rules. Each bot is
     * wrapped in a {@link Recorder} that notes what it asked for and how long it took.
     *
     * <p>Afterwards the game is replayed on our model ({@link QuarkBot}): for the replay frames and
     * as a running check of our rules. If our model ends with a different position or winner than the
     * engine, the game throws. When {@code replay} is non-null, one JSON frame per turn is appended
     * (see Replay).
     */
    static Result play(Algorithm first, Algorithm second, StringBuilder replay) {
        return play(first, second, replay, GameConstants.DEFAULT_ROWS, GameConstants.DEFAULT_COLUMNS);
    }

    /** One game as above, on a board of rows x columns. */
    static Result play(Algorithm first, Algorithm second, StringBuilder replay, int rows, int columns) {
        List<Turn> turns = Collections.synchronizedList(new ArrayList<>());
        Experiment experiment = new Experiment(new Recorder(first, 0, turns, replay != null), new Recorder(second, 1, turns, replay != null), false, false, rows, columns);
        try {
            experiment.runExperiment();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", e);
        } catch (RuntimeException e) {
            return forfeit(turns, e);
        }
        int winner = experiment.getExperimentStatus() == ExperimentStatus.COLLAPSED ? seatOf(experiment.getCurrentPlayer()) : -1;

        // Replay the engine's history on our model, turn by turn. A turn the engine skipped (late
        // answer, full or invalid column) has no entry in the history.
        PlayerStats[] stats = {new PlayerStats(), new PlayerStats()};
        QuarkBot board = new QuarkBot(rows, columns);
        List<Move> history = experiment.getHistory();
        int placed = 0;
        if (replay != null) replay.append(frame(board, -1, -1, false, false, 0, null, -1));
        for (Turn turn : turns) {
            int player = board.sideToMove;
            PlayerStats mine = stats[player];
            mine.calls++;
            mine.totalNanos += turn.columnNanos;
            mine.maxNanos = Math.max(mine.maxNanos, Math.max(turn.columnNanos, turn.bombNanos));
            if (Math.max(turn.columnNanos, turn.bombNanos) > MOVE_LIMIT_NANOS) mine.lateMoves++;
            if (turn.player != player) throw new AssertionError("turn " + turns.indexOf(turn) + ": the engine asked player " + turn.player + ", our model expects " + player);

            boolean wasPlaced = placed < history.size() && seatOf(history.get(placed).player()) == player;
            if (!wasPlaced) {
                mine.skippedTurns++; // nothing is placed, but the turn counts toward the cooldown
                boolean bombAnswered = turn.wantsBomb && turn.bombNanos <= MOVE_LIMIT_NANOS;
                if (bombAnswered && board.bombCooldown[player] == 0) board.bombCooldown[player] = QuarkBot.BOMB_COOLDOWN; // the engine resets it even so
                board.skipTurn();
                if (replay != null) replay.append(',').append(frame(board, player, turn.column, false, true, turn.columnNanos, turn.search, -1));
                continue;
            }

            Move move = history.get(placed++);
            int column = move.column(), action = QuarkBot.action(column, move.isSpecialMove());
            if (!board.isLegal(action)) throw new AssertionError("turn " + turns.indexOf(turn) + ": the engine played " + Tools.describeAction(action) + ", our rules don't allow it");
            int landingRow = board.rows - 1 - board.columnHeight[column];
            board.play(action);
            if (move.isSpecialMove()) mine.bombs++;
            if (replay != null) {
                String json = frame(board, player, column, move.isSpecialMove(), false, turn.columnNanos, turn.search, board.winner());
                replay.append(',').append(move.isSpecialMove() ? json.replace("\"blast\":null", "\"blast\":[" + landingRow + "," + column + "]") : json);
            }
        }

        if (placed != history.size()) throw new AssertionError("the engine placed " + history.size() + " quarks, our replay " + placed);
        if (board.winner() != winner) throw new AssertionError("the engine's winner is " + winner + ", our model's " + board.winner());
        if (!Arrays.deepEquals(Referee.finalField(experiment), Tools.field(board)))
            throw new AssertionError("the final boards differ; ours:\n" + Tools.render(board));
        StringJoiner moves = new StringJoiner(",");
        for (Move move : history) moves.add(move.column() + (move.isSpecialMove() ? "b" : ""));
        return new Result(winner, turns.size(), stats, moves.toString());
    }

    /**
     * The engine's referee can throw after a late answer (it goes on with column -1). The game is then
     * lost by the last player who answered late; any other failure is rethrown.
     */
    static Result forfeit(List<Turn> turns, RuntimeException failure) {
        PlayerStats[] stats = {new PlayerStats(), new PlayerStats()};
        int late = -1;
        synchronized (turns) {
            for (Turn turn : turns) {
                PlayerStats mine = stats[turn.player];
                mine.calls++;
                mine.totalNanos += turn.columnNanos;
                mine.maxNanos = Math.max(mine.maxNanos, Math.max(turn.columnNanos, turn.bombNanos));
                if (Math.max(turn.columnNanos, turn.bombNanos) > MOVE_LIMIT_NANOS) { mine.lateMoves++; late = turn.player; }
            }
        }
        if (late < 0) throw failure;
        System.err.printf("the referee failed after a late answer of player %d (%s): player %d wins%n", late, failure, 1 - late);
        return new Result(1 - late, turns.size(), stats, "");
    }

    static int seatOf(SpaceOwnership colour) {
        return colour == SpaceOwnership.FIRST_PLAYER ? 0 : 1;
    }

    /** One turn as a bot saw it: what it answered and how long each answer took. */
    static final class Turn {
        final int player;
        volatile int column = -1;
        volatile boolean wantsBomb;
        volatile long columnNanos, bombNanos;
        volatile String search;

        Turn(int player) {
            this.player = player;
        }
    }

    /**
     * Wraps a bot for the Experiment and notes each turn. The turn is listed before the bot is
     * asked, so the list keeps the engine's order even if an answer comes too late.
     */
    static final class Recorder implements Algorithm {
        final Algorithm bot;
        final int player;
        final List<Turn> turns;
        final boolean withSearchInfo;
        Turn current;

        Recorder(Algorithm bot, int player, List<Turn> turns, boolean withSearchInfo) {
            this.bot = bot;
            this.player = player;
            this.turns = turns;
            this.withSearchInfo = withSearchInfo;
        }

        @Override
        public int accelerateQuark(QuantumField field) {
            Turn turn = current = new Turn(player);
            turns.add(turn);
            long started = System.nanoTime();
            int column = bot.accelerateQuark(field);
            turn.columnNanos = System.nanoTime() - started;
            turn.column = column;
            return column;
        }

        @Override
        public boolean useSpecialMove(QuantumField field) {
            Turn turn = current;
            long started = System.nanoTime();
            boolean wantsBomb = bot.useSpecialMove(field);
            turn.bombNanos = System.nanoTime() - started;
            turn.wantsBomb = wantsBomb;
            if (withSearchInfo) turn.search = searchInfo(bot, player);
            return wantsBomb;
        }
    }

    /** One replay frame: who moved where, the board after it (rows top to bottom), and search details. */
    static String frame(QuarkBot board, int player, int column, boolean bombed, boolean skipped, long nanos, String search, int winner) {
        StringBuilder cells = new StringBuilder();
        for (int height = board.rows - 1; height >= 0; height--)
            for (int c = 0; c < board.columns; c++) {
                int owner = board.owner[c * board.rows + height];
                cells.append(owner == QuarkBot.EMPTY ? '.' : owner == 0 ? 'X' : 'O');
            }
        return String.format(Locale.ROOT, "{\"p\":%d,\"c\":%d,\"bomb\":%b,\"skip\":%b,\"ms\":%.1f,\"board\":\"%s\",\"blast\":null,\"w\":%d%s}",
                player, column, bombed, skipped, nanos / 1e6, cells, winner, search == null ? "" : "," + search);
    }

    /** Depth, score (first mover's view) and nodes of the last search of a QuarkBot, also one loaded from a JAR. */
    static String searchInfo(Algorithm bot, int player) {
        try {
            Class<?> type = bot.getClass();
            int score = type.getField("lastScore").getInt(bot);
            return String.format("\"d\":%d,\"v\":%d,\"n\":%d", type.getField("lastDepth").getInt(bot),
                    player == 0 ? score : -score, type.getField("nodesSearched").getLong(bot));
        } catch (ReflectiveOperationException notAnEngine) {
            return null;
        }
    }

    // ------------------------------------------------------------------ making bots from specs

    static Algorithm make(String spec, long seed) {
        if (spec.equals("random")) return new RandomBot(seed);
        if (spec.equals("tuned")) spec = "quark:" + Tune.OUTPUT;
        if (spec.startsWith("jar:")) return fromJar(Path.of(spec.substring(4)));
        String settings = spec.contains("#") ? spec.substring(spec.indexOf('#') + 1) : null; // SerenaBot fields, name=value,...
        if (settings != null) spec = spec.substring(0, spec.indexOf('#'));

        String evaluation = spec.contains("+") ? spec.substring(spec.lastIndexOf('+') + 1) : null; // net, pat, lines; none: as shipped
        if (evaluation != null) spec = spec.substring(0, spec.lastIndexOf('+'));
        String parameterFile = null;
        if (spec.contains(":")) {
            parameterFile = spec.substring(spec.indexOf(':') + 1);
            spec = spec.substring(0, spec.indexOf(':'));
        }
        String name = spec.split("[@/]")[0];
        Algorithm bot = instantiate(className(name));
        if (bot instanceof QuarkBot engine) {
            if (parameterFile != null) Tools.applyParameters(engine, Tune.load(Path.of(parameterFile)));
            if (spec.contains("@")) engine.budgetNanos = Long.parseLong(spec.replaceAll(".*@(\\d+).*", "$1")) * 1_000_000L;
            if (spec.contains("/")) {
                engine.maxDepth = Integer.parseInt(spec.replaceAll(".*/(\\d+).*", "$1"));
                engine.budgetNanos = Long.MAX_VALUE / 4;
            }
        }
        if (bot instanceof QuarkV2Bot v2) {
            if (parameterFile != null) Tools.applyParameters(v2, Tune.load(Path.of(parameterFile)));
            if (spec.contains("@")) v2.budgetNanos = Long.parseLong(spec.replaceAll(".*@(\\d+).*", "$1")) * 1_000_000L;
            if (spec.contains("/")) {
                v2.maxDepth = Integer.parseInt(spec.replaceAll(".*/(\\d+).*", "$1"));
                v2.budgetNanos = Long.MAX_VALUE / 4;
            }
        }
        if (bot instanceof SerenaBot serena && parameterFile != null) Tune.applySerena(serena, Tune.load(Path.of(parameterFile)));
        if (bot instanceof SerenaBot serena && spec.contains("@")) serena.budgetNanos = Long.parseLong(spec.replaceAll(".*@(\\d+).*", "$1")) * 1_000_000L;
        if (bot instanceof SerenaBot serena && evaluation != null) {
            serena.withNetwork = evaluation.equals("net");
            serena.withPatterns = evaluation.equals("pat");
        }
        if (bot instanceof SerenaBot serena && settings != null) applySettings(serena, settings);
        return bot;
    }

    /**
     * SerenaBot's tuning fields from a spec's {@code #name=value,...}: {@code bomb} (bombBonusPercent),
     * {@code bombLine} (bombLineQuarks), {@code bombHold} (bombHoldPercent, 0: the flat bomb bonus only),
     * {@code holdThreat}, {@code holdFour}, {@code holdCap} (bombHoldThreat, bombHoldFour, bombHoldCap),
     * {@code forcing} (forcingPlies), {@code reach} (dropReach),
     * {@code variety} (varietyNoise), {@code varietyQuarks}, {@code refined} (refinedSearch, 0 or 1:
     * 0 plays the plain alpha-beta search), {@code lmr} (lateMoveReductions, 0 or 1), {@code lmrLog} (reductionLogDivisor, 0: the steps),
     * {@code lmrLogBase} (reductionLogBase), {@code lowThreat} (lowestThreatWeight, 0: off),
     * {@code lowThreatParity} (lowestThreatParity, 0 or 1),
     * {@code bombHorizon} (0 to 3), {@code tss} (threatSpaceSearch, 0 or 1), {@code tssShare}
     * (threatSharePercent), {@code tssPlies} (threatSearchPlies), {@code tssBombs} (bombThreats, 0 or 1);
     * for ThreatBot also {@code threat} (threatPlies). ThreatBot shares {@code refined} and {@code lmr}
     * (its own refined search now is SerenaBot's) and keeps {@code tss} off, running its own proofs.
     */
    static void applySettings(SerenaBot bot, String settings) {
        for (String setting : settings.split(",")) {
            String name = setting.substring(0, setting.indexOf('='));
            int value = Integer.parseInt(setting.substring(setting.indexOf('=') + 1));
            switch (name) {
                case "bomb" -> bot.bombBonusPercent = value;
                case "bombLine" -> bot.bombLineQuarks = value;
                case "bombHold" -> bot.bombHoldPercent = value;
                case "holdThreat" -> bot.bombHoldThreat = value;
                case "holdFour" -> bot.bombHoldFour = value;
                case "holdCap" -> bot.bombHoldCap = value;
                case "forcing" -> bot.forcingPlies = value;
                case "reach" -> bot.dropReach = value;
                case "variety" -> bot.varietyNoise = value;
                case "varietyQuarks" -> bot.varietyQuarks = value;
                case "bombHorizon" -> bot.bombHorizon = value;
                case "tss" -> bot.threatSpaceSearch = value != 0;
                case "tssShare" -> bot.threatSharePercent = value;
                case "tssPlies" -> bot.threatSearchPlies = value;
                case "tssBombs" -> bot.bombThreats = value != 0;
                case "threat" -> ((ThreatBot) bot).threatPlies = value;
                case "refined" -> bot.refinedSearch = value != 0;
                case "lmr" -> bot.lateMoveReductions = value != 0;
                case "lmrLog" -> bot.reductionLogDivisor = value;
                case "lmrLogBase" -> bot.reductionLogBase = value;
                case "lowThreat" -> bot.lowestThreatWeight = value;
                case "lowThreatParity" -> bot.lowestThreatParity = value != 0;
                default -> throw new IllegalArgumentException("unknown SerenaBot setting " + name);
            }
        }
    }

    /** The class of a bot name: {@code quark} is {@code ch.qc.meelik.bots.QuarkBot}; a qualified name is itself. */
    static String className(String name) {
        return name.contains(".") ? name : "ch.qc.meelik.bots." + Character.toUpperCase(name.charAt(0)) + name.substring(1) + "Bot";
    }

    static Algorithm instantiate(String className) {
        try {
            return (Algorithm) Class.forName(className).getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalArgumentException("unknown bot " + className, e);
        }
    }

    static Algorithm fromJar(Path jar) {
        try (JarFile file = new JarFile(jar.toFile())) {
            String botClass = file.getManifest().getMainAttributes().getValue("Bot-Class");
            if (botClass == null) throw new IllegalArgumentException(jar + " has no Bot-Class in its manifest");
            ClassLoader loader = new BotLoader(new URL[]{jar.toUri().toURL()}, Arena.class.getClassLoader());
            return (Algorithm) loader.loadClass(botClass).getDeclaredConstructor().newInstance();
        } catch (java.io.IOException | ReflectiveOperationException e) {
            throw new IllegalArgumentException("can't load a bot from " + jar, e);
        }
    }

    /** Loads a bot's own classes from its JAR first; the JDK and the arena's engine API come from the parent. */
    static final class BotLoader extends URLClassLoader {
        BotLoader(URL[] urls, ClassLoader parent) {
            super(urls, parent);
        }

        @Override
        protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            synchronized (getClassLoadingLock(name)) {
                Class<?> type = findLoadedClass(name);
                if (type == null && !name.startsWith("java.") && !name.startsWith("ch.cern.quantumconnect.")) {
                    try {
                        type = findClass(name);
                    } catch (ClassNotFoundException notInJar) {
                        // fall through to the parent
                    }
                }
                if (type == null) return super.loadClass(name, resolve);
                if (resolve) resolveClass(type);
                return type;
            }
        }
    }

    /** A random legal column, never bombs. */
    static final class RandomBot implements Algorithm {
        final Random random;

        RandomBot(long seed) {
            random = new Random(seed);
        }

        @Override
        public int accelerateQuark(QuantumField field) {
            List<Integer> open = new ArrayList<>();
            for (int column = 0; column < field.getNumberOfColumns(); column++) if (!field.isColumnFull(column)) open.add(column);
            return open.isEmpty() ? 0 : open.get(random.nextInt(open.size()));
        }

        @Override
        public boolean useSpecialMove(QuantumField field) {
            return false;
        }
    }
}
