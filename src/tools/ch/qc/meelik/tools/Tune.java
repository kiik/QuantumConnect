package ch.qc.meelik.tools;

import ch.qc.meelik.bots.QuarkBot;
import ch.qc.meelik.engine.Board;
import ch.qc.meelik.engine.Engine;

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
 * out/tune-params.txt, which Arena can use as "tuned" or "quark:out/tune-params.txt".
 *
 * <p>Usage: Tune [generations=40] [depth=4] [pairs=16] [openings=8] [threads=cores] [seed=1]
 */
public final class Tune {
    static final Path OUTPUT = Path.of("out/tune-params.txt");
    static final double STEP_SIZE = 0.2, LEARNING_RATE = 3;
    static final int MAX_PLIES = 400, OPENING_PLIES = 4, VALIDATION_OPENINGS = 100, VALIDATE_EVERY = 10;
    static final ThreadLocal<Engine[]> PLAYERS = ThreadLocal.withInitial(() -> new Engine[]{new QuarkBot(), new QuarkBot()});

    static int searchDepth;

    public static void main(String[] args) throws Exception {
        int generations = intArg(args, 0, 40), pairs = intArg(args, 2, 16), openingsPerPair = intArg(args, 3, 8);
        int threads = intArg(args, 4, Runtime.getRuntime().availableProcessors()), seed = intArg(args, 5, 1);
        searchDepth = intArg(args, 1, 4);
        Random random = new Random(seed);
        ExecutorService pool = Executors.newFixedThreadPool(threads);

        double[] defaults = Tools.parameters(new QuarkBot());
        double[] current = defaults.clone();
        current[current.length - 1] = Math.log(16); // start the bomb bonus where it can move
        double bestScore = 0.5;
        System.out.printf("tuning %d parameters: depth %d, %d pairs x %d openings x 2 colours per generation, %d threads%n",
                current.length, searchDepth, pairs, openingsPerPair, threads);

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
                double score = validate(pool, current, defaults);
                Engine shown = new QuarkBot();
                Tools.applyParameters(shown, current);
                System.out.printf("  vs defaults: %.1f%%  %s%n", 100 * score, Tools.describeWeights(shown));
                if (score > bestScore) {
                    bestScore = score;
                    save(current, String.format("Tune gen %d/%d, depth %d, %d pairs x %d openings, seed %d: %.1f%% vs defaults (%d games)%n# %s",
                            generation, generations, searchDepth, pairs, openingsPerPair, seed, 100 * score, 2 * VALIDATION_OPENINGS, Tools.describeWeights(shown)));
                    System.out.println("  new best -> " + OUTPUT);
                }
            }
        }
        pool.shutdown();
        System.out.printf("best: %.1f%% vs defaults at depth %d%s%n", 100 * bestScore, searchDepth, bestScore == 0.5 ? " (no improvement found)" : "");
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

    // ------------------------------------------------------------------ playing

    /** Score of a against b (win 1, draw 0.5), averaged over the openings, each played with both colours. */
    static double match(double[] a, double[] b, List<int[]> openings) {
        double total = 0;
        for (int[] opening : openings) total += game(a, b, opening) + 1 - game(b, a, opening);
        return total / (2 * openings.size());
    }

    /** One game at fixed depth; first moves first. Returns first's score. */
    static double game(double[] first, double[] second, int[] opening) {
        Engine[] players = PLAYERS.get();
        Tools.applyParameters(players[0], first);
        Tools.applyParameters(players[1], second);
        for (Engine player : players) {
            player.newGame();
            player.maxDepth = searchDepth;
        }
        Board board = players[0].board; // each player keeps its own board; both see the same moves
        for (int action : opening) for (Engine player : players) player.board.play(action);
        for (int ply = 0; ply < MAX_PLIES; ply++) {
            int winner = board.winner();
            if (winner >= 0) return winner == 0 ? 1 : 0;
            if (board.isFull()) break;
            int action = players[board.sideToMove].chooseAction(Long.MAX_VALUE / 4);
            for (Engine player : players) player.board.play(action);
        }
        return 0.5;
    }

    /** Random plain drops, so that games differ even though the engines are deterministic. */
    static List<int[]> randomOpenings(Random random, int count) {
        List<int[]> openings = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int[] opening = new int[OPENING_PLIES];
            for (int ply = 0; ply < OPENING_PLIES; ply++) opening[ply] = Board.action(random.nextInt(12), false);
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
    static void save(double[] parameters, String provenance) throws Exception {
        Files.createDirectories(OUTPUT.toAbsolutePath().getParent());
        Files.writeString(OUTPUT, "# " + provenance + "\n# written " + LocalDateTime.now().withNano(0) + "\n"
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
