package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.Move;
import ch.cern.quantumconnect.core.QuantumField;
import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.engine.Board;
import ch.qc.meelik.engine.Engine;

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.jar.JarFile;

/**
 * Headless local arena: plays bots against each other under the rules as we understand them. The
 * referee is our own {@link Board}, so this can't catch a misread rule; the starter project's
 * Experiment is the reference for that.
 *
 * <p>Usage: Arena [botA=quark] [botB=random] [games=20]. A bot spec is one of:
 * <ul>
 *   <li>{@code random}: a random legal column, never bombs;
 *   <li>a bot name, the class {@code ch.qc.meelik.bots.<Name>Bot} ({@code quark}, {@code evolved},
 *       ...) or any fully qualified bot class, optionally followed by {@code @<ms>} (time budget per
 *       move), {@code /<depth>} (fixed search depth) and {@code :<file>} (evaluation parameters from
 *       Tune), e.g. {@code quark@10}, {@code quark/2}, {@code quark:params/evolved-d4.txt};
 *   <li>{@code tuned}: short for {@code quark:out/tune-params.txt};
 *   <li>{@code jar:<path>}: the bot in a packaged JAR (its Bot-Class), in its own class loader, so
 *       two versions of the same bot can play each other.
 * </ul>
 */
public final class Arena {
    static final long MOVE_LIMIT_NANOS = 100_000_000L;
    static final int MOVE_LIMIT_PER_MATCH = 1000;

    /** One game's outcome: winner (0 first mover, 1 second, -1 draw), plies, and per-player statistics. */
    record Result(int winner, int plies, PlayerStats[] players) { }

    static final class PlayerStats {
        long calls, totalNanos, maxNanos, skippedTurns, bombs;

        void add(PlayerStats other) {
            calls += other.calls;
            totalNanos += other.totalNanos;
            maxNanos = Math.max(maxNanos, other.maxNanos);
            skippedTurns += other.skippedTurns;
            bombs += other.bombs;
        }

        String summary() {
            return String.format("avg %.1f ms/move, max %.1f ms, skipped turns %d, bombs %d",
                    totalNanos / 1e6 / Math.max(1, calls), maxNanos / 1e6, skippedTurns, bombs);
        }
    }

    public static void main(String[] args) {
        String botA = args.length > 0 ? args[0] : "quark", botB = args.length > 1 ? args[1] : "random";
        int games = args.length > 2 ? Integer.parseInt(args[2]) : 20;
        int winsA = 0, draws = 0, winsB = 0;
        long plies = 0;
        PlayerStats statsA = new PlayerStats(), statsB = new PlayerStats();
        for (int game = 0; game < games; game++) {
            boolean aStarts = game % 2 == 0;
            Result result = aStarts ? play(make(botA, game), make(botB, game), null) : play(make(botB, game), make(botA, game), null);
            int seatOfA = aStarts ? 0 : 1;
            statsA.add(result.players[seatOfA]);
            statsB.add(result.players[1 - seatOfA]);
            plies += result.plies;
            if (result.winner < 0) draws++;
            else if (result.winner == seatOfA) winsA++;
            else winsB++;
        }
        System.out.printf("%s vs %s, %d games (alternating who starts)%n", botA, botB, games);
        System.out.printf("  %s wins %d, draws %d, %s wins %d, avg %.1f plies%n", botA, winsA, draws, botB, winsB, (double) plies / games);
        System.out.printf("  %-12s %s%n  %-12s %s%n", botA, statsA.summary(), botB, statsB.summary());
    }

    // ------------------------------------------------------------------ one game

    /** Plays one game; when {@code replay} is non-null, appends one JSON frame per ply (see Replay). */
    static Result play(Algorithm first, Algorithm second, StringBuilder replay) {
        Algorithm[] bots = {first, second};
        PlayerStats[] stats = {new PlayerStats(), new PlayerStats()};
        Board referee = new Board();
        List<Move> history = new ArrayList<>();
        int winner = -1, ply = 0;
        if (replay != null) replay.append(frame(referee, -1, -1, false, false, 0, null, -1));

        for (; ply < MOVE_LIMIT_PER_MATCH && winner < 0 && !referee.isFull(); ply++) {
            int player = referee.sideToMove;
            SpaceOwnership colour = player == 0 ? SpaceOwnership.FIRST_PLAYER : SpaceOwnership.SECOND_PLAYER;
            PlayerStats mine = stats[player];

            long started = System.nanoTime();
            int column = bots[player].accelerateQuark(new QuantumField(Tools.field(referee), colour, history));
            long moveNanos = System.nanoTime() - started;
            mine.calls++;
            mine.totalNanos += moveNanos;
            mine.maxNanos = Math.max(mine.maxNanos, moveNanos);
            if (moveNanos > MOVE_LIMIT_NANOS || column < 0 || column >= referee.columns || referee.columnHeight[column] == referee.rows) {
                mine.skippedTurns++;
                referee.skipTurn();
                if (replay != null) replay.append(',').append(frame(referee, player, column, false, true, moveNanos, null, -1));
                continue;
            }

            int[] beforeDrop = referee.saveState();
            boolean mayBomb = referee.isLegal(Board.action(column, true));
            int landingRow = referee.rows - 1 - referee.columnHeight[column];
            referee.play(Board.action(column, false));
            winner = referee.winner(); // the win check comes before the arena asks about the bomb
            String search = replay == null ? null : searchInfo(bots[player], player);

            boolean bombed = false;
            if (winner < 0) {
                started = System.nanoTime();
                boolean wantsBomb = bots[player].useSpecialMove(new QuantumField(Tools.field(referee), colour, history));
                long bombNanos = System.nanoTime() - started;
                mine.maxNanos = Math.max(mine.maxNanos, bombNanos);
                if (wantsBomb && mayBomb && bombNanos <= MOVE_LIMIT_NANOS) {
                    referee.restoreState(beforeDrop);
                    referee.play(Board.action(column, true));
                    mine.bombs++;
                    bombed = true;
                    winner = referee.winner();
                }
                history.add(new Move(column, bombed));
            }
            if (replay != null) {
                String json = frame(referee, player, column, bombed, false, moveNanos, search, winner);
                replay.append(',').append(bombed ? json.replace("\"blast\":null", "\"blast\":[" + landingRow + "," + column + "]") : json);
            }
        }
        return new Result(winner, ply, stats);
    }

    /** One replay frame: who moved where, the board after it (rows top to bottom), and search details. */
    static String frame(Board board, int player, int column, boolean bombed, boolean skipped, long nanos, String search, int winner) {
        StringBuilder cells = new StringBuilder();
        for (int height = board.rows - 1; height >= 0; height--)
            for (int c = 0; c < board.columns; c++) {
                int owner = board.owner[c * board.rows + height];
                cells.append(owner == Board.EMPTY ? '.' : owner == 0 ? 'X' : 'O');
            }
        return String.format(Locale.ROOT, "{\"p\":%d,\"c\":%d,\"bomb\":%b,\"skip\":%b,\"ms\":%.1f,\"board\":\"%s\",\"blast\":null,\"w\":%d%s}",
                player, column, bombed, skipped, nanos / 1e6, cells, winner, search == null ? "" : "," + search);
    }

    /** Depth, score (first mover's view) and nodes of the last search of an Engine bot, also one loaded from a JAR. */
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

        String parameterFile = null;
        if (spec.contains(":")) {
            parameterFile = spec.substring(spec.indexOf(':') + 1);
            spec = spec.substring(0, spec.indexOf(':'));
        }
        String name = spec.split("[@/]")[0];
        Algorithm bot = instantiate(name.contains(".") ? name : "ch.qc.meelik.bots." + Character.toUpperCase(name.charAt(0)) + name.substring(1) + "Bot");
        if (bot instanceof Engine engine) {
            if (parameterFile != null) Tools.applyParameters(engine, Tune.load(Path.of(parameterFile)));
            if (spec.contains("@")) engine.budgetNanos = Long.parseLong(spec.replaceAll(".*@(\\d+).*", "$1")) * 1_000_000L;
            if (spec.contains("/")) {
                engine.maxDepth = Integer.parseInt(spec.replaceAll(".*/(\\d+).*", "$1"));
                engine.budgetNanos = Long.MAX_VALUE / 4;
            }
        }
        return bot;
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
