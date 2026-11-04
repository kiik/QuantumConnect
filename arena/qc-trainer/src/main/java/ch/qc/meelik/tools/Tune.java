package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.QuantumField;
import ch.qc.meelik.bots.QuarkBot;
import ch.qc.meelik.bots.QuarkV2Bot;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Evolves the evaluation parameters (line weights per direction and the bomb bonus) by self-play.
 *
 * <h2>Method: an evolution strategy with mirrored pairs</h2>
 * Each generation draws {@code pairs} random directions e in parameter space. For each one, the
 * parameters moved forward along e play the parameters moved backward along e, from the same
 * random openings and with both colours, at a fixed search depth. The score difference says which
 * way along e is better, and the parameters take a step along the average of those directions,
 * each weighted by its score difference. Parameters are kept in log space, so steps are relative.
 *
 * <p>Every {@link #VALIDATE_EVERY} generations the current parameters play the untouched defaults
 * on a fixed set of openings. The best validated parameters are saved, with how they were made, to
 * the output file (out/tune-params.txt unless given), which Arena can use as
 * "quark:&lt;file&gt;" ("tuned" for the default file). If no generation beats the defaults, the final
 * parameters are saved anyway, with their score, so that every run leaves a model behind.
 *
 * <p>Starting from another model's file instead of the defaults is transfer learning: the run
 * refines that model, which may be tuned at a different search depth.
 *
 * <h2>Which bot, and validating against a field</h2>
 * The bot is QuarkBot ({@code quark}), whose games run on its own board, or QuarkV2Bot
 * ({@code quarkV2}), whose games run in the engine's Experiment with both players opening on the
 * same random drops. Both take the same 16 parameters. Given a {@code field} of bot specs (as in
 * Arena, comma separated), validation instead plays the parameters against each of those bots at
 * the real time limit, {@link #FIELD_OPENINGS} openings with both colours, and the best score
 * against the field is kept. The gradient still comes from self-play, which is cheap and
 * deterministic; the field only decides which checkpoint is best, so that a slow opponent, such as
 * a teammate's JAR, can steer training without playing thousands of games.
 *
 * <p>Usage: Tune [generations=40] [depth=4] [pairs=16] [openings=8] [threads=cores] [seed=1]
 * [out=out/tune-params.txt] [start=defaults] [bot=quark] [field=none] [field threads=6].
 * {@code ./qc.sh train} runs the recipes in training.txt.
 */
public final class Tune {
    static final Path OUTPUT = Path.of("out/tune-params.txt");
    static final double STEP_SIZE = 0.2, LEARNING_RATE = 3;
    static final int MAX_PLIES = 400, OPENING_PLIES = 4, VALIDATION_OPENINGS = 100, VALIDATE_EVERY = 10, FIELD_OPENINGS = 16;
    static final ThreadLocal<QuarkBot[]> PLAYERS = ThreadLocal.withInitial(() -> new QuarkBot[]{new QuarkBot(), new QuarkBot()});

    static int searchDepth;
    /** The bot being tuned: "quark" or "quarkV2". */
    static String bot = "quark";
    /** Turns lost to the time limit (or an illegal column) in engine games, by either player. */
    static final AtomicLong SKIPPED_TURNS = new AtomicLong();

    public static void main(String[] args) throws Exception {
        int generations = intArg(args, 0, 40), pairs = intArg(args, 2, 16), openingsPerPair = intArg(args, 3, 8);
        int threads = intArg(args, 4, Runtime.getRuntime().availableProcessors()), seed = intArg(args, 5, 1);
        searchDepth = intArg(args, 1, 4);
        Path output = args.length > 6 ? Path.of(args[6]) : OUTPUT;
        String start = args.length > 7 ? args[7] : "defaults";
        bot = args.length > 8 ? args[8] : "quark";
        List<String> field = args.length > 9 && !args[9].equals("none") ? List.of(args[9].split(",")) : List.of();
        if (!bot.equals("quark") && !bot.equals("quarkV2")) throw new IllegalArgumentException("bot is quark or quarkV2: " + bot);
        Random random = new Random(seed);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        ExecutorService fieldPool = Executors.newFixedThreadPool(intArg(args, 10, 6)); // few at once: real time limits

        double[] defaults = bot.equals("quark") ? Tools.parameters(new QuarkBot()) : Tools.parameters(new QuarkV2Bot());
        double[] current;
        if (start.equals("defaults")) {
            current = defaults.clone();
            current[current.length - 1] = Math.log(16); // start the bomb bonus where it can move
        } else {
            current = load(Path.of(start));
        }
        double bestScore = 0.5, lastScore = 0.5;
        boolean improved = false;
        String recipe = String.format("%s, depth %d, %d pairs x %d openings, seed %d, start %s", bot, searchDepth, pairs, openingsPerPair, seed, start);
        String against = field.isEmpty() ? "defaults" : String.join(" + ", field);
        int validationGames = field.isEmpty() ? 2 * VALIDATION_OPENINGS : 2 * FIELD_OPENINGS * field.size();
        System.out.printf("tuning %d parameters: %s; %d threads -> %s%n", current.length, recipe, threads, output);
        if (!field.isEmpty()) {
            bestScore = lastScore = validateAgainstField(fieldPool, current, field);
            System.out.printf("  start vs %s: %.1f%%%n", against, 100 * bestScore);
        }

        for (int generation = 1; generation <= generations; generation++) {
            long started = System.nanoTime();
            List<int[]> openings = randomOpenings(random, openingsPerPair);
            double[][] directions = new double[pairs][];
            List<Future<Double>> differences = new ArrayList<>();
            for (int p = 0; p < pairs; p++) {
                double[] direction = gaussian(random, current.length);
                directions[p] = direction;
                double[] forward = step(current, direction, STEP_SIZE), backward = step(current, direction, -STEP_SIZE);
                differences.add(pool.submit(() -> 2 * match(forward, backward, openings) - 1)); // in [-1, 1]
            }
            double[] update = new double[current.length];
            double meanAbsDifference = 0;
            for (int p = 0; p < pairs; p++) {
                double difference = differences.get(p).get();
                meanAbsDifference += Math.abs(difference) / pairs;
                for (int i = 0; i < update.length; i++) update[i] += LEARNING_RATE * STEP_SIZE * difference * directions[p][i] / pairs;
            }
            current = step(current, update, 1);
            System.out.printf("gen %3d  mean |score difference| %.2f  %.1f s%n", generation, meanAbsDifference, (System.nanoTime() - started) / 1e9);

            if (generation % VALIDATE_EVERY == 0 || generation == generations) {
                lastScore = field.isEmpty() ? validate(pool, current, defaults) : validateAgainstField(fieldPool, current, field);
                System.out.printf("  vs %s: %.1f%%  %s%n", against, 100 * lastScore, describe(current));
                if (lastScore > bestScore) {
                    bestScore = lastScore;
                    improved = true;
                    save(output, current, String.format("Tune gen %d/%d, %s: %.1f%% vs %s (%d games)%n# %s",
                            generation, generations, recipe, 100 * lastScore, against, validationGames, describe(current)));
                    System.out.println("  new best -> " + output);
                }
            }
        }
        pool.shutdown();
        fieldPool.shutdown();
        if (!improved) {
            save(output, current, String.format("Tune final gen %d, %s: %.1f%% vs %s (%d games), no improvement found%n# %s",
                    generations, recipe, 100 * lastScore, against, validationGames, describe(current)));
            System.out.printf("no improvement found; final parameters (%.1f%% vs %s) -> %s%n", 100 * lastScore, against, output);
        } else {
            System.out.printf("best: %.1f%% vs %s -> %s%n", 100 * bestScore, against, output);
        }
    }

    static String describe(double[] parameters) {
        QuarkBot shown = new QuarkBot();
        Tools.applyParameters(shown, parameters);
        return Tools.describeWeights(shown);
    }

    /** Score of the parameters against the defaults on a fixed set of openings. */
    static double validate(ExecutorService pool, double[] parameters, double[] defaults) throws Exception {
        List<int[]> openings = randomOpenings(new Random(12345), VALIDATION_OPENINGS);
        List<Future<Double>> scores = new ArrayList<>();
        for (int[] opening : openings) scores.add(pool.submit(() -> match(parameters, defaults, List.of(opening))));
        double total = 0;
        for (Future<Double> score : scores) total += score.get();
        return total / openings.size();
    }

    /** Score of the parameters, at the real time limit, against each bot of the field on a fixed set of openings. */
    static double validateAgainstField(ExecutorService pool, double[] parameters, List<String> field) throws Exception {
        SKIPPED_TURNS.set(0);
        List<int[]> openings = randomOpenings(new Random(54321), FIELD_OPENINGS);
        List<Future<Double>> scores = new ArrayList<>();
        for (String opponent : field)
            for (int[] opening : openings)
                for (int seat = 0; seat < 2; seat++) {
                    boolean tunedFirst = seat == 0;
                    scores.add(pool.submit(() -> {
                        Algorithm tuned = player(parameters, 0), other = Arena.make(opponent, 0);
                        double firstScore = engineGame(tunedFirst ? tuned : other, tunedFirst ? other : tuned, opening);
                        return tunedFirst ? firstScore : 1 - firstScore;
                    }));
                }
        double total = 0;
        for (Future<Double> score : scores) total += score.get();
        if (SKIPPED_TURNS.get() > 0) System.out.printf("  (%d turns skipped in these games: the field's timing may be off)%n", SKIPPED_TURNS.get());
        return total / scores.size();
    }

    // ------------------------------------------------------------------ playing

    /** Score of a against b (win 1, draw 0.5), averaged over the openings, each played with both colours. */
    static double match(double[] a, double[] b, List<int[]> openings) {
        double total = 0;
        for (int[] opening : openings) total += game(a, b, opening) + 1 - game(b, a, opening);
        return total / (2 * openings.size());
    }

    /** One game at fixed depth; first moves first. Returns first's score. */
    static double game(double[] first, double[] second, int[] opening) {
        if (bot.equals("quarkV2")) return engineGame(player(first, searchDepth), player(second, searchDepth), opening);
        QuarkBot[] players = PLAYERS.get();
        Tools.applyParameters(players[0], first);
        Tools.applyParameters(players[1], second);
        for (QuarkBot player : players) {
            player.newGame();
            player.maxDepth = searchDepth;
        }
        QuarkBot board = players[0]; // each player keeps its own board; both see the same moves
        for (int action : opening) for (QuarkBot player : players) player.play(action);
        for (int ply = 0; ply < MAX_PLIES; ply++) {
            int winner = board.winner();
            if (winner >= 0) return winner == 0 ? 1 : 0;
            if (board.isFull()) break;
            int action = players[board.sideToMove].chooseAction(Long.MAX_VALUE / 4);
            for (QuarkBot player : players) player.play(action);
        }
        return 0.5;
    }

    /** The tuned bot with the parameters, at a fixed depth, or at its time budget if depth is 0. */
    static Algorithm player(double[] parameters, int depth) {
        if (bot.equals("quarkV2")) {
            QuarkV2Bot player = new QuarkV2Bot();
            Tools.applyParameters(player, parameters);
            if (depth > 0) {
                player.maxDepth = depth;
                player.budgetNanos = Long.MAX_VALUE / 4;
            }
            return player;
        }
        QuarkBot player = new QuarkBot();
        Tools.applyParameters(player, parameters);
        if (depth > 0) {
            player.maxDepth = depth;
            player.budgetNanos = Long.MAX_VALUE / 4;
        }
        return player;
    }

    /** One game in the engine's Experiment, both players opening with the given drops. Returns first's score. */
    static double engineGame(Algorithm first, Algorithm second, int[] opening) {
        Arena.Result result = Arena.play(new Opening(first, opening), new Opening(second, opening), null);
        SKIPPED_TURNS.addAndGet(result.players()[0].skippedTurns + result.players()[1].skippedTurns);
        int winner = result.winner();
        return winner == 0 ? 1 : winner == 1 ? 0 : 0.5;
    }

    /**
     * A bot that plays the opening's drops while the history is shorter than the opening, and its own
     * moves after that. It asks the bot every turn even so, so that the bot sees the game from the start.
     */
    static final class Opening implements Algorithm {
        final Algorithm bot;
        final int[] opening;
        boolean inOpening;

        Opening(Algorithm bot, int[] opening) {
            this.bot = bot;
            this.opening = opening;
        }

        @Override
        public int accelerateQuark(QuantumField field) {
            int ply = field.getMoveHistory().size();
            int column = bot.accelerateQuark(field);
            inOpening = ply < opening.length;
            return inOpening ? QuarkBot.columnOf(opening[ply]) : column;
        }

        @Override
        public boolean useSpecialMove(QuantumField field) {
            boolean bomb = bot.useSpecialMove(field);
            return !inOpening && bomb;
        }
    }

    /** Random plain drops, so that games differ even though the engines are deterministic. */
    static List<int[]> randomOpenings(Random random, int count) {
        List<int[]> openings = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int[] opening = new int[OPENING_PLIES];
            for (int ply = 0; ply < OPENING_PLIES; ply++) opening[ply] = QuarkBot.action(random.nextInt(12), false);
            openings.add(opening);
        }
        return openings;
    }

    // ------------------------------------------------------------------ vectors and files

    static double[] gaussian(Random random, int length) {
        double[] vector = new double[length];
        for (int i = 0; i < length; i++) vector[i] = random.nextGaussian();
        return vector;
    }

    static double[] step(double[] from, double[] direction, double size) {
        double[] to = from.clone();
        for (int i = 0; i < to.length; i++) to[i] += size * direction[i];
        return to;
    }

    static int intArg(String[] args, int index, int fallback) {
        return args.length > index ? Integer.parseInt(args[index]) : fallback;
    }

    /** A parameter file: '#' lines saying how it was made, then the log-space parameters on one line. */
    static void save(Path output, double[] parameters, String provenance) throws Exception {
        Files.createDirectories(output.toAbsolutePath().getParent());
        Files.writeString(output, "# " + provenance + "\n# written " + LocalDateTime.now().withNano(0) + "\n"
                + Arrays.toString(parameters).replaceAll("[\\[\\],]", "") + "\n");
    }

    static double[] load(Path file) {
        try {
            return Files.readAllLines(file).stream()
                    .filter(line -> !line.isBlank() && !line.startsWith("#"))
                    .flatMap(line -> Arrays.stream(line.trim().split("\\s+")))
                    .mapToDouble(Double::parseDouble)
                    .toArray();
        } catch (Exception e) {
            throw new IllegalStateException("can't read parameters from " + file, e);
        }
    }
}
