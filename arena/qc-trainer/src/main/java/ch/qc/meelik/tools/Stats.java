package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.GameConstants;
import ch.cern.quantumconnect.core.QuantumField;

import java.io.BufferedWriter;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Many games between many bots, and what they say: a rating with error bars for every bot, the
 * score of every pairing, game lengths and move times. Writes a page of charts to out/stats.html
 * and the ranking to results/stats.md.
 *
 * <p>The concepts:
 * <ul>
 *   <li><b>The field</b> is the entrants of a league file, as in {@link League}. A directory line
 *       enters every parameter file and every bot JAR in it, so {@code jars/} brings in teammates'
 *       JARs and snapshots of older versions ({@code ./qc.sh snapshot}).
 *   <li><b>The game log,</b> out/stats/games.tsv, keeps every game ever played, one line each. A run
 *       only tops up each pairing to the requested number of games, so the statistics sharpen with
 *       every run, and a new entrant plays only the pairings it is missing.
 *   <li><b>A fingerprint</b> names each bot in the log: a hash of everything that decides how it plays,
 *       its spec's options and the bytes of its class, JAR or parameter file. Change a bot's code and
 *       its old games stop counting for it (they stay in the log). To keep an old version in the
 *       field, snapshot it as a JAR first.
 *   <li><b>The rating</b> is an Elo scale fitted to all games at once (Bradley-Terry, a draw counted
 *       as half a win), with 0 at QuarkBot as shipped ({@code quark}) or else the first entrant other than random. Its
 *       95% interval comes from refitting on the games resampled 200 times (bootstrap).
 *   <li><b>Openings:</b> a bot that searches to a fixed depth plays the same game every time, and 100
 *       copies of one game say no more than one. The page counts the distinct games. Random
 *       openings (the first plies chosen at random, the same opening for both colours of a pairing)
 *       make every game different, at the price of no longer starting from the empty board as the
 *       tournament does. A negative number of opening plies takes them from real games instead: the
 *       start of an archived match on the same board ({@link Openings}), mirrored and shifted.
 * </ul>
 *
 * <p>Usage: Stats [file=league.txt] [games per pair=10] [threads=cores/2] [opening plies=0, negative: archived] [size=8x12].
 * Games per pair 0 plays nothing and only writes the report. As in League, keep threads at or below
 * half the cores so time budgets stay realistic; the page shows each bot's worst move against the
 * server's limit ({@link Arena#MOVE_LIMIT_MS}, 50 ms).
 */
public final class Stats {
    static final Path LOG = Path.of("out/stats/games.tsv"), REPORT = Path.of("results/stats.md"),
            PAGE = Path.of("out/stats.html"), TEMPLATE = Path.of("qc-trainer/src/main/resources/stats.html");
    static final String LOG_HEADER = "# date\tsize\topening plies\tX\tO\twinner\tplies\tmoves\t"
            + "X avg ms\tX max ms\tX late\tX skipped\tX bombs\tO avg ms\tO max ms\tO late\tO skipped\tO bombs\tX label\tO label";
    static final int RESAMPLES = 200;

    /**
     * One logged game: the bots by fingerprint, X (first mover) then O, a hash of the moves, and
     * per-seat statistics. A late move is an answer over the time limit: a late column skips the
     * turn, a late bomb answer only cancels the bomb.
     */
    record Game(String size, int openingPlies, String[] bots, int winner, int plies, String moves,
                double[] avgMs, double[] maxMs, int[] late, int[] skipped, int[] bombs) {
        /** The points of the bot in a seat: 1 for a win, 0.5 for a draw. */
        double points(int seat) {
            return winner < 0 ? 0.5 : winner == seat ? 1 : 0;
        }
    }

    /** A game still to play: the entrants as X and O, and which opening of their pairing it uses. */
    record Task(int first, int second, int opening) { }

    public static void main(String[] args) throws Exception {
        Path file = Path.of(args.length > 0 ? args[0] : "league.txt");
        int gamesPerPair = args.length > 1 ? Integer.parseInt(args[1]) : 10;
        int threads = args.length > 2 ? Integer.parseInt(args[2]) : Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        int openingPlies = args.length > 3 ? Integer.parseInt(args[3]) : 0;
        String size = args.length > 4 ? args[4] : GameConstants.DEFAULT_ROWS + "x" + GameConstants.DEFAULT_COLUMNS;

        List<League.Entrant> field = new ArrayList<>();
        List<String> fingerprints = new ArrayList<>();
        for (League.Entrant entrant : League.readEntrants(file)) {
            String fingerprint = fingerprint(entrant.spec());
            int twin = fingerprints.indexOf(fingerprint);
            if (twin >= 0) {
                System.out.println(entrant.label() + " plays exactly like " + field.get(twin).label() + "; left out");
                continue;
            }
            field.add(entrant);
            fingerprints.add(fingerprint);
        }

        List<Game> log = readLog();
        playMissingGames(field, fingerprints, log, gamesPerPair, threads, openingPlies, size);
        report(field, fingerprints, log, openingPlies, size);
    }

    // ------------------------------------------------------------------ fingerprints

    /**
     * A short hash of everything that decides how a spec plays: for a JAR its bytes; otherwise the
     * spec's options ({@code @ms}, {@code /depth}, {@code +net}), the bot's class file and its parameter file.
     */
    static String fingerprint(String spec) throws Exception {
        if (spec.equals("random")) return "random";
        if (spec.equals("tuned")) spec = "quark:" + Tune.OUTPUT;
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        if (spec.startsWith("jar:")) {
            digest.update(Files.readAllBytes(Path.of(spec.substring(4))));
        } else {
            String bot = spec.contains(":") ? spec.substring(0, spec.indexOf(':')) : spec;
            String name = bot.split("[@/+#]")[0], className = Arena.className(name);
            digest.update(bot.substring(name.length()).getBytes());
            try (InputStream classFile = Stats.class.getClassLoader().getResourceAsStream(className.replace('.', '/') + ".class")) {
                if (classFile == null) throw new IllegalArgumentException("unknown bot " + className);
                digest.update(classFile.readAllBytes());
            }
            if (spec.contains(":")) digest.update(Files.readAllBytes(Path.of(spec.substring(spec.indexOf(':') + 1))));
        }
        return HexFormat.of().formatHex(digest.digest()).substring(0, 12);
    }

    // ------------------------------------------------------------------ the game log

    static List<Game> readLog() throws Exception {
        List<Game> games = new ArrayList<>();
        if (!Files.exists(LOG)) return games;
        for (String line : Files.readAllLines(LOG)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] f = line.split("\t");
            games.add(new Game(f[1], Integer.parseInt(f[2]), new String[]{f[3], f[4]}, Integer.parseInt(f[5]), Integer.parseInt(f[6]), f[7],
                    new double[]{Double.parseDouble(f[8]), Double.parseDouble(f[13])}, new double[]{Double.parseDouble(f[9]), Double.parseDouble(f[14])},
                    new int[]{Integer.parseInt(f[10]), Integer.parseInt(f[15])}, new int[]{Integer.parseInt(f[11]), Integer.parseInt(f[16])},
                    new int[]{Integer.parseInt(f[12]), Integer.parseInt(f[17])}));
        }
        return games;
    }

    static String logLine(Game game, String firstLabel, String secondLabel) {
        return String.join("\t", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME).substring(0, 19), game.size(),
                Integer.toString(game.openingPlies()), game.bots()[0], game.bots()[1], Integer.toString(game.winner()), Integer.toString(game.plies()), game.moves(),
                decimal(game.avgMs()[0]), decimal(game.maxMs()[0]), Integer.toString(game.late()[0]), Integer.toString(game.skipped()[0]), Integer.toString(game.bombs()[0]),
                decimal(game.avgMs()[1]), decimal(game.maxMs()[1]), Integer.toString(game.late()[1]), Integer.toString(game.skipped()[1]), Integer.toString(game.bombs()[1]),
                firstLabel, secondLabel);
    }

    /** Whether a logged game was played under these conditions, between two bots of the field. */
    static boolean counts(Game game, Map<String, Integer> fieldIndex, int openingPlies, String size) {
        return game.size().equals(size) && game.openingPlies() == openingPlies
                && fieldIndex.containsKey(game.bots()[0]) && fieldIndex.containsKey(game.bots()[1]);
    }

    // ------------------------------------------------------------------ playing the missing games

    /**
     * Tops up every pairing to the requested games, half with each bot first. The schedule goes
     * round by round over all pairings, so a run stopped early still leaves them evenly played;
     * every finished game is appended to the log at once.
     */
    static void playMissingGames(List<League.Entrant> field, List<String> fingerprints, List<Game> log,
                                 int gamesPerPair, int threads, int openingPlies, String size) throws Exception {
        int n = field.size(), perColour = (gamesPerPair + 1) / 2;
        Map<String, Integer> fieldIndex = indexOf(fingerprints);
        int[][] playedAsFirst = new int[n][n]; // [X][O]
        for (Game game : log)
            if (counts(game, fieldIndex, openingPlies, size)) playedAsFirst[fieldIndex.get(game.bots()[0])][fieldIndex.get(game.bots()[1])]++;

        List<Task> schedule = new ArrayList<>();
        for (int round = 0; round < perColour; round++)
            for (int a = 0; a < n; a++)
                for (int b = a + 1; b < n; b++) {
                    if (playedAsFirst[a][b] <= round) schedule.add(new Task(a, b, round));
                    if (playedAsFirst[b][a] <= round) schedule.add(new Task(b, a, round));
                }
        if (schedule.isEmpty()) {
            if (gamesPerPair > 0) System.out.printf("%d bots, every pairing has %d games already%n", n, 2 * perColour);
            return;
        }
        System.out.printf("%d bots, %d games to play on %d threads (%d per pairing on %s, %s)%n",
                n, schedule.size(), threads, 2 * perColour, size, describeOpenings(openingPlies));

        int[] dimensions = Tools.size(size);
        Openings archived = openingPlies < 0 ? Openings.fromArchive(dimensions[0], dimensions[1], -openingPlies) : null;
        if (archived != null && archived.isEmpty()) throw new IllegalStateException("no archived openings of " + -openingPlies + " plies on " + size);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CompletionService<Game> games = new ExecutorCompletionService<>(pool);
        for (Task task : schedule) {
            String firstSpec = field.get(task.first()).spec(), secondSpec = field.get(task.second()).spec();
            String firstPrint = fingerprints.get(task.first()), secondPrint = fingerprints.get(task.second());
            long pairing = firstPrint.hashCode() ^ secondPrint.hashCode();
            int[] opening = archived != null ? archived.draw(new Random(pairing * 1_000_003L + task.opening()))
                    : opening(openingPlies, dimensions[1], pairing, task.opening());
            long seed = 1000L * task.opening() + 31L * task.first() + task.second();
            games.submit(() -> {
                Arena.Result result = Arena.play(withOpening(Arena.make(firstSpec, seed), opening), withOpening(Arena.make(secondSpec, seed + 1), opening),
                        null, dimensions[0], dimensions[1]);
                Arena.PlayerStats x = result.players()[0], o = result.players()[1];
                return new Game(size, openingPlies, new String[]{firstPrint, secondPrint}, result.winner(), result.plies(),
                        Integer.toHexString(result.moves().hashCode()),
                        new double[]{averageMs(x), averageMs(o)}, new double[]{x.maxNanos / 1e6, o.maxNanos / 1e6}, new int[]{(int) x.lateMoves, (int) o.lateMoves},
                        new int[]{(int) x.skippedTurns, (int) o.skippedTurns}, new int[]{(int) x.bombs, (int) o.bombs});
            });
        }
        pool.shutdown();

        Files.createDirectories(LOG.toAbsolutePath().getParent());
        boolean newLog = !Files.exists(LOG);
        long started = System.nanoTime();
        int failures = 0, progressStep = Math.max(1, schedule.size() / 20);
        try (BufferedWriter writer = Files.newBufferedWriter(LOG, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (newLog) writer.write(LOG_HEADER + "\n");
            for (int done = 1; done <= schedule.size(); done++) {
                try {
                    Game game = games.take().get();
                    log.add(game);
                    writer.write(logLine(game, field.get(fieldIndex.get(game.bots()[0])).label(), field.get(fieldIndex.get(game.bots()[1])).label()) + "\n");
                    writer.flush();
                } catch (ExecutionException e) {
                    if (failures++ == 0) System.out.println("a game failed and is not logged: " + e.getCause());
                }
                if (done % progressStep == 0 || done == schedule.size()) {
                    double seconds = (System.nanoTime() - started) / 1e9;
                    System.out.printf("  %d/%d games, %.0f s, about %.0f s left%n", done, schedule.size(), seconds, seconds / done * (schedule.size() - done));
                }
            }
        }
        if (failures > 0) System.out.println(failures + " games failed");
    }

    static double averageMs(Arena.PlayerStats stats) {
        return stats.totalNanos / 1e6 / Math.max(1, stats.calls);
    }

    static String describeOpenings(int openingPlies) {
        return openingPlies < 0 ? -openingPlies + " archived opening plies" : openingPlies + " random opening plies";
    }

    /** The columns of a pairing's random opening; the same pairing and index give the same opening. */
    static int[] opening(int plies, int columns, long pairing, int index) {
        Random random = new Random(pairing * 1_000_003L + index);
        int[] opening = new int[plies];
        for (int ply = 0; ply < plies; ply++) opening[ply] = random.nextInt(columns);
        return opening;
    }

    static Algorithm withOpening(Algorithm bot, int[] opening) {
        return opening.length == 0 ? bot : new Opening(bot, opening);
    }

    /**
     * Plays the given columns for the first plies of the game instead of the bot's own choice.
     * The bot is still asked every turn, so its own turn count and bomb cooldown stay right, and
     * reads the opening from the field like any other position.
     */
    static final class Opening implements Algorithm {
        final Algorithm bot;
        final int[] columns;

        Opening(Algorithm bot, int[] columns) {
            this.bot = bot;
            this.columns = columns;
        }

        @Override
        public int accelerateQuark(QuantumField field) {
            int column = bot.accelerateQuark(field), ply = field.getMoveHistory().size();
            return ply < columns.length && !field.isColumnFull(columns[ply]) ? columns[ply] : column;
        }

        @Override
        public boolean useSpecialMove(QuantumField field) {
            boolean wantsBomb = bot.useSpecialMove(field);
            return field.getMoveHistory().size() >= columns.length && wantsBomb;
        }
    }

    // ------------------------------------------------------------------ ratings

    /**
     * Elo ratings fitted to the games by the Bradley-Terry model: bot i beats bot j with probability
     * s_i / (s_i + s_j), a draw counting as half a win each. The strengths come from the classic
     * fixed-point iteration s_i = points_i / sum_j games_ij / (s_i + s_j). Each bot also gets two
     * virtual draws, spread over the pairings it played, which keeps a bot that lost or won every
     * game at a finite rating while barely moving the others.
     * Rating = 400 log10(s_i / s_anchor); NaN for a bot without games.
     */
    static double[] ratings(int n, List<Game> games, Map<String, Integer> fieldIndex, int anchor) {
        double[][] met = new double[n][n];
        double[] points = new double[n];
        for (Game game : games) {
            int x = fieldIndex.get(game.bots()[0]), o = fieldIndex.get(game.bots()[1]);
            met[x][o]++;
            met[o][x]++;
            points[x] += game.points(0);
            points[o] += game.points(1);
        }
        int[] opponents = new int[n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (met[i][j] > 0) opponents[i]++;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (met[i][j] > 0) {
                    double virtualDraws = 2.0 / Math.max(opponents[i], opponents[j]);
                    met[i][j] += virtualDraws;
                    met[j][i] += virtualDraws;
                    points[i] += virtualDraws / 2;
                    points[j] += virtualDraws / 2;
                }

        double[] strength = new double[n];
        Arrays.fill(strength, 1);
        for (int iteration = 0; iteration < 2000; iteration++) {
            double[] next = new double[n];
            double logSum = 0, change = 0;
            for (int i = 0; i < n; i++) {
                double expected = 0;
                for (int j = 0; j < n; j++) if (met[i][j] > 0) expected += met[i][j] / (strength[i] + strength[j]);
                next[i] = expected > 0 ? points[i] / expected : 1;
                logSum += Math.log(next[i]);
            }
            double mean = Math.exp(logSum / n);
            for (int i = 0; i < n; i++) {
                next[i] /= mean;
                change = Math.max(change, Math.abs(Math.log(next[i] / strength[i])));
            }
            strength = next;
            if (change < 1e-10) break;
        }

        double[] elo = new double[n];
        for (int i = 0; i < n; i++) {
            boolean played = false;
            for (int j = 0; j < n; j++) played |= met[i][j] > 0;
            elo[i] = played ? 400 * Math.log10(strength[i] / strength[anchor]) : Double.NaN;
        }
        return elo;
    }

    /** The 2.5% and 97.5% points of each bot's rating over refits on resampled games: [bot][low, high]. */
    static double[][] intervals(int n, List<Game> games, Map<String, Integer> fieldIndex, int anchor) {
        double[][] samples = new double[n][RESAMPLES];
        Random random = new Random(1);
        for (int sample = 0; sample < RESAMPLES; sample++) {
            List<Game> resampled = new ArrayList<>(games.size());
            for (int g = 0; g < games.size(); g++) resampled.add(games.get(random.nextInt(games.size())));
            double[] elo = ratings(n, resampled, fieldIndex, anchor);
            for (int i = 0; i < n; i++) samples[i][sample] = elo[i];
        }
        double[][] interval = new double[n][];
        for (int i = 0; i < n; i++) {
            double[] sorted = Arrays.stream(samples[i]).filter(v -> !Double.isNaN(v)).sorted().toArray();
            interval[i] = sorted.length == 0 ? new double[]{Double.NaN, Double.NaN} : new double[]{quantile(sorted, 0.025), quantile(sorted, 0.975)};
        }
        return interval;
    }

    // ------------------------------------------------------------------ the report

    /** Everything one bot did in the counted games. */
    static final class Tally {
        int games, wins, draws, losses, gamesAsFirst, late, skipped, bombs;
        double pointsAsFirst, pointsAsSecond, sumAvgMs;
        final List<Double> plies = new ArrayList<>(), worstMs = new ArrayList<>();

        double score() {
            return games == 0 ? Double.NaN : (wins + draws / 2.0) / games;
        }
    }

    static void report(List<League.Entrant> field, List<String> fingerprints, List<Game> log, int openingPlies, String size) throws Exception {
        int n = field.size();
        Map<String, Integer> fieldIndex = indexOf(fingerprints);
        List<Game> games = new ArrayList<>();
        for (Game game : log) if (counts(game, fieldIndex, openingPlies, size)) games.add(game);
        if (games.isEmpty()) {
            System.out.println("no games yet between these bots on " + size + " with " + openingPlies + " opening plies");
            return;
        }

        Tally[] tallies = new Tally[n];
        for (int i = 0; i < n; i++) tallies[i] = new Tally();
        int[][] wins = new int[n][n], draws = new int[n][n]; // [row bot][column bot]
        Map<String, Set<String>> distinctByPairing = new HashMap<>();
        Set<String> distinct = new HashSet<>();
        double firstMoverPoints = 0;
        for (Game game : games) {
            int[] seats = {fieldIndex.get(game.bots()[0]), fieldIndex.get(game.bots()[1])};
            for (int seat = 0; seat < 2; seat++) {
                Tally tally = tallies[seats[seat]];
                int opponent = seats[1 - seat];
                tally.games++;
                if (game.winner() < 0) {
                    tally.draws++;
                    draws[seats[seat]][opponent]++;
                } else if (game.winner() == seat) {
                    tally.wins++;
                    wins[seats[seat]][opponent]++;
                } else {
                    tally.losses++;
                }
                if (seat == 0) {
                    tally.gamesAsFirst++;
                    tally.pointsAsFirst += game.points(0);
                } else {
                    tally.pointsAsSecond += game.points(1);
                }
                tally.plies.add((double) game.plies());
                tally.worstMs.add(game.maxMs()[seat]);
                tally.sumAvgMs += game.avgMs()[seat];
                tally.late += game.late()[seat];
                tally.skipped += game.skipped()[seat];
                tally.bombs += game.bombs()[seat];
            }
            String key = String.join("|", game.bots()), pairing = Math.min(seats[0], seats[1]) + "," + Math.max(seats[0], seats[1]);
            distinct.add(key + "|" + game.moves());
            distinctByPairing.computeIfAbsent(pairing, k -> new HashSet<>()).add(key + "|" + game.moves());
            firstMoverPoints += game.points(0);
        }

        int anchor = anchor(field);
        double[] elo = ratings(n, games, fieldIndex, anchor);
        double[][] interval = intervals(n, games, fieldIndex, anchor);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(Double.isNaN(elo[b]) ? -1e9 : elo[b], Double.isNaN(elo[a]) ? -1e9 : elo[a]));

        // results/stats.md and the console: the ranking
        String when = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"));
        StringBuilder table = new StringBuilder();
        table.append(String.format(Locale.ROOT, "Stats: %d bots, %d games (%d distinct), %s, %s, %s%n%n", n, games.size(), distinct.size(), size, describeOpenings(openingPlies), when));
        table.append(String.format(Locale.ROOT, "Elo relative to %s (0), with 95%% intervals from %d resamples of the games. First mover's score: %.1f%%.%n%n",
                field.get(anchor).label(), RESAMPLES, 100 * firstMoverPoints / games.size()));
        table.append("| # | Bot | Elo | 95% | Games | W | D | L | Score | As X | As O | Avg ms | Worst ms | Late moves | Skipped turns |\n");
        table.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (int place = 0; place < n; place++) {
            int i = order[place];
            Tally t = tallies[i];
            table.append(String.format(Locale.ROOT, "| %d | %s | %s | %s | %d | %d | %d | %d | %s | %s | %s | %.1f | %.1f | %d | %d |%n", place + 1, field.get(i).label(),
                    whole(elo[i]), Double.isNaN(interval[i][0]) ? "" : whole(interval[i][0]) + " to " + whole(interval[i][1]),
                    t.games, t.wins, t.draws, t.losses, percent(t.score()), percent(t.pointsAsFirst / t.gamesAsFirst),
                    percent(t.pointsAsSecond / (t.games - t.gamesAsFirst)), t.sumAvgMs / Math.max(1, t.games), max(t.worstMs), t.late, t.skipped));
        }
        System.out.print("\n" + table);
        Files.createDirectories(REPORT.toAbsolutePath().getParent());
        Files.writeString(REPORT, table);
        System.out.println("\nwritten to " + REPORT);

        // out/stats.html: the same numbers and the distributions, for the charts
        List<String> bots = new ArrayList<>();
        for (int place = 0; place < n; place++) {
            int i = order[place];
            Tally t = tallies[i];
            bots.add(String.format(Locale.ROOT, "{\"label\":%s,\"spec\":%s,\"approach\":%s,\"print\":\"%s\",\"games\":%d,\"w\":%d,\"d\":%d,\"l\":%d,"
                            + "\"score\":%s,\"scoreX\":%s,\"scoreO\":%s,\"elo\":%s,\"lo\":%s,\"hi\":%s,\"plies\":%s,\"worstMs\":%s,\"avgMs\":%s,\"late\":%d,\"skipped\":%d,\"bombs\":%s}",
                    json(field.get(i).label()), json(field.get(i).spec()), json(Replay.approach(field.get(i).spec())), fingerprints.get(i), t.games, t.wins, t.draws, t.losses,
                    number(t.score()), number(t.pointsAsFirst / t.gamesAsFirst), number(t.pointsAsSecond / (t.games - t.gamesAsFirst)),
                    number(elo[i]), number(interval[i][0]), number(interval[i][1]), fiveNumbers(t.plies), fiveNumbers(t.worstMs),
                    number(t.sumAvgMs / Math.max(1, t.games)), t.late, t.skipped, number((double) t.bombs / Math.max(1, t.games))));
        }
        List<String> rows = new ArrayList<>();
        for (int r = 0; r < n; r++) {
            List<String> cells = new ArrayList<>();
            for (int c = 0; c < n; c++) {
                int row = order[r], column = order[c];
                int w = wins[row][column], d = draws[row][column], l = wins[column][row], played = w + d + l;
                if (row == column || played == 0) {
                    cells.add("null");
                    continue;
                }
                double score = (w + d / 2.0) / played;
                double variance = (w * Math.pow(1 - score, 2) + d * Math.pow(0.5 - score, 2) + l * Math.pow(score, 2)) / played;
                double superiority = w + l == 0 ? 0.5 : normalCdf((w - l) / Math.sqrt(w + l));
                int distinctGames = distinctByPairing.get(Math.min(row, column) + "," + Math.max(row, column)).size();
                cells.add(String.format(Locale.ROOT, "{\"w\":%d,\"d\":%d,\"l\":%d,\"score\":%s,\"ci\":%s,\"los\":%s,\"distinct\":%d}",
                        w, d, l, number(score), number(1.96 * Math.sqrt(variance / played)), number(superiority), distinctGames));
            }
            rows.add("[" + String.join(",", cells) + "]");
        }
        String data = String.format(Locale.ROOT, "{\"generated\":\"%s\",\"size\":\"%s\",\"openingPlies\":%d,\"anchor\":%s,\"games\":%d,\"distinct\":%d,"
                        + "\"ignored\":%d,\"firstMoverScore\":%s,\"resamples\":%d,\"limitMs\":%d,\"bots\":[%s],\"pairs\":[%s]}",
                when, size, openingPlies, json(field.get(anchor).label()), games.size(), distinct.size(), log.size() - games.size(),
                number(firstMoverPoints / games.size()), RESAMPLES, Arena.MOVE_LIMIT_MS, String.join(",\n", bots), String.join(",\n", rows));
        Files.createDirectories(PAGE.toAbsolutePath().getParent());
        Files.writeString(PAGE, Files.readString(TEMPLATE).replace("/*DATA*/null", data));
        System.out.println("charts written to " + PAGE);
    }

    /** The bot at Elo 0: QuarkBot as shipped, else the first entrant that isn't random (which only has a floor). */
    static int anchor(List<League.Entrant> field) {
        for (int i = 0; i < field.size(); i++) if (field.get(i).spec().equals("quark")) return i;
        for (int i = 0; i < field.size(); i++) if (!field.get(i).spec().equals("random")) return i;
        return 0;
    }

    // ------------------------------------------------------------------ small helpers

    static Map<String, Integer> indexOf(List<String> fingerprints) {
        Map<String, Integer> index = new HashMap<>();
        for (int i = 0; i < fingerprints.size(); i++) index.put(fingerprints.get(i), i);
        return index;
    }

    /** The value at fraction q of sorted values, interpolating between neighbours. */
    static double quantile(double[] sorted, double q) {
        double position = q * (sorted.length - 1);
        int below = (int) Math.floor(position), above = Math.min(sorted.length - 1, below + 1);
        return sorted[below] + (position - below) * (sorted[above] - sorted[below]);
    }

    /** Minimum, lower quartile, median, upper quartile and maximum, as a JSON array. */
    static String fiveNumbers(List<Double> values) {
        if (values.isEmpty()) return "null";
        double[] sorted = values.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        return String.format(Locale.ROOT, "[%.1f,%.1f,%.1f,%.1f,%.1f]", sorted[0], quantile(sorted, 0.25), quantile(sorted, 0.5), quantile(sorted, 0.75), sorted[sorted.length - 1]);
    }

    static double max(List<Double> values) {
        double max = 0;
        for (double value : values) max = Math.max(max, value);
        return max;
    }

    /** The standard normal distribution function, from the Abramowitz-Stegun approximation of erf. */
    static double normalCdf(double z) {
        double x = Math.abs(z) / Math.sqrt(2), t = 1 / (1 + 0.3275911 * x);
        double erf = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-x * x);
        return z >= 0 ? (1 + erf) / 2 : (1 - erf) / 2;
    }

    static String decimal(double value) {
        return String.format(Locale.ROOT, "%.1f", value);
    }

    static String whole(double value) {
        return Double.isNaN(value) ? "" : String.format(Locale.ROOT, "%+.0f", value);
    }

    static String percent(double fraction) {
        return Double.isNaN(fraction) ? "" : String.format(Locale.ROOT, "%.0f%%", 100 * fraction);
    }

    static String number(double value) {
        return Double.isNaN(value) || Double.isInfinite(value) ? "null" : String.format(Locale.ROOT, "%.4f", value);
    }

    static String json(String text) {
        return "\"" + text.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }
}
