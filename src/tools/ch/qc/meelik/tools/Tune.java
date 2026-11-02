package ch.qc.meelik.tools;

import ch.qc.meelik.QuarkBot;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Evolves QuarkBot's evaluation parameters by self-play (an evolution strategy with antithetic
 * pairs, SPSA-style).
 *
 * <p>Each generation draws {@code pairs} random directions e. For each one, theta + sigma*e plays
 * theta - sigma*e from {@code openings} random openings, both colours, at a fixed search depth. The
 * score difference r says which way is better, and theta moves along the mean of r*e. Every
 * {@code validateEvery} generations theta plays the untouched defaults; the best validated theta
 * goes to out/tune-params.txt (used by "./qc.sh arena tuned ...").
 *
 * <p>Usage: Tune [generations=40] [depth=4] [pairs=16] [openings=8] [threads=cores] [seed=1]
 */
public final class Tune {
    static final Path OUT = Path.of("out/tune-params.txt");
    static final double SIGMA = 0.2, STEP = 3;
    static final int MAX_PLIES = 400, OPENING_PLIES = 4, VALIDATION_OPENINGS = 100, validateEvery = 10;
    static final ThreadLocal<QuarkBot[]> ENGINES = ThreadLocal.withInitial(() -> new QuarkBot[]{new QuarkBot(), new QuarkBot()});

    static int depth;

    public static void main(String[] args) throws Exception {
        int gens = arg(args, 0, 40), pairs = arg(args, 2, 16), openings = arg(args, 3, 8);
        depth = arg(args, 1, 4);
        int threads = arg(args, 4, Runtime.getRuntime().availableProcessors());
        Random rnd = new Random(arg(args, 5, 1));
        ExecutorService pool = Executors.newFixedThreadPool(threads);

        double[] base = Tools.params(new QuarkBot()), theta = base.clone();
        theta[theta.length - 1] = Math.log(16); // start the bomb bonus where it can move
        double[] best = base.clone();
        double bestScore = 0.5;
        System.out.printf("tuning %d parameters: depth %d, %d pairs x %d openings x 2 colours per generation, %d threads%n",
                theta.length, depth, pairs, openings, threads);
        for (int g = 1; g <= gens; g++) {
            long t0 = System.nanoTime();
            List<int[]> ops = openings(rnd, openings);
            double[][] eps = new double[pairs][theta.length];
            List<Future<Double>> rs = new ArrayList<>();
            for (double[] e : eps) {
                for (int i = 0; i < e.length; i++) e[i] = rnd.nextGaussian();
                double[] plus = add(theta, e, SIGMA), minus = add(theta, e, -SIGMA);
                rs.add(pool.submit(() -> match(plus, minus, ops) * 2 - 1)); // in [-1, 1]
            }
            double[] step = new double[theta.length];
            double moved = 0;
            for (int p = 0; p < pairs; p++) {
                double r = rs.get(p).get();
                moved += Math.abs(r);
                for (int i = 0; i < step.length; i++) step[i] += STEP * SIGMA * r * eps[p][i] / pairs;
            }
            theta = add(theta, step, 1);
            System.out.printf("gen %3d  mean |r| %.2f  %.1f s%n", g, moved / pairs, (System.nanoTime() - t0) / 1e9);
            if (g % validateEvery == 0 || g == gens) {
                double[] cur = theta.clone();
                List<int[]> vops = openings(new Random(12345), VALIDATION_OPENINGS); // fixed set
                List<Future<Double>> vs = new ArrayList<>();
                for (int[] op : vops) vs.add(pool.submit(() -> match(cur, base, List.of(op))));
                double score = 0;
                for (Future<Double> v : vs) score += v.get() / vops.size();
                QuarkBot show = new QuarkBot();
                Tools.apply(show, cur);
                System.out.printf("  vs defaults: %.1f%%  %s%n", 100 * score, Tools.describe(show));
                if (score > bestScore) {
                    bestScore = score;
                    best = cur;
                    save(best, String.format("Tune gen %d/%d, depth %d, %d pairs x %d openings, seed %d: %.1f%% vs defaults (%d games)%n# %s",
                            g, gens, depth, pairs, openings, arg(args, 5, 1), 100 * score, 2 * VALIDATION_OPENINGS, Tools.describe(show)));
                    System.out.println("  new best -> " + OUT);
                }
            }
        }
        pool.shutdown();
        QuarkBot show = new QuarkBot();
        Tools.apply(show, best);
        System.out.printf("best: %.1f%% vs defaults at depth %d: %s%n", 100 * bestScore, depth, Tools.describe(show));
        if (bestScore == 0.5) System.out.println("no improvement over the defaults found");
    }

    /** Score of a against b (1 win, 0.5 draw) averaged over the openings, each played with both colours. */
    static double match(double[] a, double[] b, List<int[]> openings) {
        double s = 0;
        for (int[] op : openings) s += game(a, b, op) + 1 - game(b, a, op);
        return s / (2 * openings.size());
    }

    /** One game; first moves first. Returns first's score. */
    static double game(double[] first, double[] second, int[] opening) {
        QuarkBot[] e = ENGINES.get();
        Tools.apply(e[0], first);
        Tools.apply(e[1], second);
        for (QuarkBot q : e) {
            q.newGame();
            q.maxDepth = depth;
        }
        for (int a : opening) for (QuarkBot q : e) q.play(a);
        for (int ply = 0; ply < MAX_PLIES; ply++) {
            int w = e[0].winner();
            if (w >= 0) return w == 0 ? 1 : 0;
            if (e[0].pieces == e[0].size) break;
            int a = e[e[0].turn].think(Long.MAX_VALUE / 4);
            for (QuarkBot q : e) q.play(a);
        }
        return 0.5;
    }

    /** Random plain drops, so games differ even though the engines are deterministic. */
    static List<int[]> openings(Random rnd, int n) {
        List<int[]> ops = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int[] op = new int[OPENING_PLIES];
            for (int j = 0; j < op.length; j++) op[j] = 2 * rnd.nextInt(12);
            ops.add(op);
        }
        return ops;
    }

    static double[] add(double[] x, double[] e, double s) {
        double[] y = x.clone();
        for (int i = 0; i < y.length; i++) y[i] += s * e[i];
        return y;
    }

    static int arg(String[] a, int i, int def) {
        return a.length > i ? Integer.parseInt(a[i]) : def;
    }

    /** Parameter file: '#' comment lines (provenance), then the log-space parameters on one line. */
    static void save(double[] t, String about) throws Exception {
        Files.createDirectories(OUT.toAbsolutePath().getParent());
        Files.writeString(OUT, "# " + about + "\n# written " + java.time.LocalDateTime.now().withNano(0) + "\n"
                + Arrays.toString(t).replaceAll("[\\[\\],]", "") + "\n");
    }

    static double[] load(Path file) {
        try {
            return Files.readAllLines(file).stream().filter(l -> !l.isBlank() && !l.startsWith("#"))
                    .flatMap(l -> Arrays.stream(l.trim().split("\\s+"))).mapToDouble(Double::parseDouble).toArray();
        } catch (Exception ex) {
            throw new IllegalStateException("can't read parameters from " + file, ex);
        }
    }
}
