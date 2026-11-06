package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.QuantumField;
import ch.qc.meelik.bots.QuarkBot;
import ch.qc.meelik.bots.QuarkV2Bot;
import ch.qc.meelik.bots.SerenaBot;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
 *
 * <h2>Island evolution of SerenaBot ({@code --islands N})</h2>
 * A genetic algorithm on SerenaBot's genome: its 16 evaluation parameters (log line weights and
 * bomb bonus, as above) and one structural gene, the transposition table's size in bits. N islands
 * each grow a population from Serena's constants with their own seed. Each generation, every genome
 * plays the untouched SerenaBot from fresh random openings with both colours, on the tournament's
 * 15x30 board at the real time limit (or at a fixed depth). Its fitness is that score minus a size
 * penalty per MB of table, so that memory grows only where it wins games. The best quarter of an
 * island survives; the rest are crossovers of its better half, mutated. Every K generations each
 * island's best is copied into the next island (a ring), replacing its worst. Each island's
 * generation is a row of results/islands.tsv; the best genome of the last generation goes to the
 * output file, which Arena plays as "serena:&lt;file&gt;".
 *
 * <p>Usage: Tune --islands 4 [--migrate 3] [--generations 6] [--population 6] [--openings 3]
 * [--size 15x30] [--depth 0 = real time] [--cap-ms none: a fixed depth's time cap] [--penalty 0.002 per MB] [--table-mutation 0.25] [--threads cores-2]
 * [--seed 1] [--out out/serena-islands.txt]. At another LINE_LENGTH the genome has 3 x (LINE_LENGTH - 1) + 1
 * evaluation parameters instead of 16.
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
        if (args.length > 0 && args[0].startsWith("--")) {
            islands(args);
            return;
        }
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

    // ------------------------------------------------------------------ island evolution of SerenaBot

    static final Path ISLAND_ROWS = Path.of("results/islands.tsv");
    /** The evaluation parameters (3 x (LINE_LENGTH - 1) log line weights and the bomb bonus; 16 at six in a row), then the table gene. */
    static final int TABLE_GENE = 3 * (SerenaBot.LINE_LENGTH - 1) + 1, GENES = TABLE_GENE + 1, MIN_TABLE_BITS = 14, MAX_TABLE_BITS = 21;
    static final double MUTATION = 0.15;
    /** Chance of a step in table size per mutation; {@code --table-mutation 0} keeps SerenaBot's size. */
    static double tableMutation = 0.25;

    static void islands(String[] args) throws Exception {
        Map<String, String> options = new HashMap<>();
        for (int i = 0; i + 1 < args.length; i += 2) options.put(args[i].substring(2), args[i + 1]);
        int islandCount = Integer.parseInt(options.getOrDefault("islands", "4"));
        int migrateEvery = Integer.parseInt(options.getOrDefault("migrate", "3"));
        int generations = Integer.parseInt(options.getOrDefault("generations", "6"));
        int population = Integer.parseInt(options.getOrDefault("population", "6"));
        int openingCount = Integer.parseInt(options.getOrDefault("openings", "3"));
        int depth = Integer.parseInt(options.getOrDefault("depth", "0"));
        int seed = Integer.parseInt(options.getOrDefault("seed", "1"));
        int threads = Integer.parseInt(options.getOrDefault("threads", String.valueOf(Math.max(1, Runtime.getRuntime().availableProcessors() - 2))));
        double penalty = Double.parseDouble(options.getOrDefault("penalty", "0.002"));
        tableMutation = Double.parseDouble(options.getOrDefault("table-mutation", "0.25"));
        if (options.containsKey("cap-ms")) depthCapNanos = Long.parseLong(options.get("cap-ms")) * 1_000_000L;
        int[] size = Tools.size(options.getOrDefault("size", "15x30"));
        Path output = Path.of(options.getOrDefault("out", "out/serena-islands.txt"));
        String recipe = String.format("%d islands, migrate every %d, %d generations x %d genomes, %d openings x 2 colours, %dx%d, %s, penalty %s/MB, seed %d",
                islandCount, migrateEvery, generations, population, openingCount, size[0], size[1],
                depth > 0 ? "depth " + depth : "real time", penalty, seed);
        String run = LocalDateTime.now().withNano(0).toString();
        System.out.printf("island evolution of SerenaBot: %s; %d threads -> %s, %s%n", recipe, threads, output, ISLAND_ROWS);
        appendRow("# " + run + " Tune " + String.join(" ", args) + ": " + recipe + "; fitness = score vs SerenaBot - penalty x table MB");

        double[] defaults = serenaGenome();
        Random[] randoms = new Random[islandCount];
        List<List<double[]>> islands = new ArrayList<>();
        for (int island = 0; island < islandCount; island++) {
            randoms[island] = new Random(1000L * seed + island);
            List<double[]> members = new ArrayList<>();
            members.add(defaults.clone());
            while (members.size() < population) members.add(mutate(defaults, randoms[island]));
            islands.add(members);
        }
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        double[][] fitness = new double[islandCount][];
        for (int generation = 1; generation <= generations; generation++) {
            long started = System.nanoTime();
            List<List<Future<Double>>> scores = new ArrayList<>();
            for (int island = 0; island < islandCount; island++) {
                List<int[]> openings = openingsOn(randoms[island], openingCount, size[1]);
                List<Future<Double>> islandScores = new ArrayList<>();
                for (double[] genome : islands.get(island)) islandScores.add(pool.submit(() -> serenaScore(genome, openings, depth, size)));
                scores.add(islandScores);
            }
            for (int island = 0; island < islandCount; island++) {
                List<double[]> members = islands.get(island);
                fitness[island] = new double[population];
                double mean = 0;
                for (int m = 0; m < population; m++) {
                    fitness[island][m] = scores.get(island).get(m).get() - penalty * tableMegabytes(members.get(m));
                    mean += fitness[island][m] / population;
                }
                sortByFitness(members, fitness[island]);
                double[] leader = members.get(0);
                appendRow(String.format("%s\t%d\t%d\t%.4f\t%.4f\t%.4f\t%d\t%.1f", run, island, generation, fitness[island][0], mean,
                        fitness[island][0] + penalty * tableMegabytes(leader), tableBits(leader), (System.nanoTime() - started) / 1e9));
            }
            boolean migrating = generation % migrateEvery == 0 && islandCount > 1;
            System.out.printf("gen %2d  best per island", generation);
            for (double[] islandFitness : fitness) System.out.printf(" %.3f", islandFitness[0]);
            System.out.printf("%s  %.1f s%n", migrating && generation < generations ? "  (migrating)" : "", (System.nanoTime() - started) / 1e9);
            if (generation == generations) break;

            List<List<double[]>> next = new ArrayList<>();
            for (int island = 0; island < islandCount; island++) next.add(breed(islands.get(island), randoms[island]));
            if (migrating)
                for (int island = 0; island < islandCount; island++)
                    next.get((island + 1) % islandCount).set(population - 1, islands.get(island).get(0).clone());
            islands = next;
        }
        pool.shutdown();

        int bestIsland = 0;
        for (int island = 1; island < islandCount; island++) if (fitness[island][0] > fitness[bestIsland][0]) bestIsland = island;
        double[] best = islands.get(bestIsland).get(0);
        String weights = describe(Arrays.copyOf(best, TABLE_GENE));
        save(output, best, String.format("Tune --islands, %s: fitness %.3f (island %d, last generation), table %d bits%n# %s",
                recipe, fitness[bestIsland][0], bestIsland, tableBits(best), weights));
        System.out.printf("best: fitness %.3f, table %d bits, %s -> %s%n", fitness[bestIsland][0], tableBits(best), weights, output);
    }

    /** The next population: the best quarter as is, the rest crossovers of two of the better half, mutated. */
    static List<double[]> breed(List<double[]> sorted, Random random) {
        int elite = Math.max(1, sorted.size() / 4), parents = Math.max(1, sorted.size() / 2);
        List<double[]> next = new ArrayList<>(sorted.subList(0, elite));
        while (next.size() < sorted.size()) {
            double[] mother = sorted.get(random.nextInt(parents)), father = sorted.get(random.nextInt(parents));
            double[] child = new double[GENES];
            for (int i = 0; i < GENES; i++) child[i] = random.nextBoolean() ? mother[i] : father[i];
            next.add(mutate(child, random));
        }
        return next;
    }

    /** Gaussian steps on the log weights; now and then one step up or down in table size. */
    static double[] mutate(double[] genome, Random random) {
        double[] child = genome.clone();
        for (int i = 0; i < TABLE_GENE; i++) child[i] += MUTATION * random.nextGaussian();
        if (random.nextDouble() < tableMutation)
            child[TABLE_GENE] = Math.max(MIN_TABLE_BITS, Math.min(MAX_TABLE_BITS, child[TABLE_GENE] + (random.nextBoolean() ? 1 : -1)));
        return child;
    }

    /** Sorts the members, and their fitness alongside, best first. */
    static void sortByFitness(List<double[]> members, double[] fitness) {
        Integer[] order = new Integer[members.size()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble(i -> -fitness[i]));
        List<double[]> sorted = new ArrayList<>();
        double[] sortedFitness = new double[fitness.length];
        for (int i = 0; i < order.length; i++) {
            sorted.add(members.get(order[i]));
            sortedFitness[i] = fitness[order[i]];
        }
        members.clear();
        members.addAll(sorted);
        System.arraycopy(sortedFitness, 0, fitness, 0, fitness.length);
    }

    /** Score of the genome against the untouched SerenaBot, each opening played with both colours. */
    static double serenaScore(double[] genome, List<int[]> openings, int depth, int[] size) {
        double total = 0;
        for (int[] opening : openings)
            for (int tunedSeat = 0; tunedSeat < 2; tunedSeat++) {
                Algorithm tuned = new Opening(serena(genome, depth), opening), reference = new Opening(serena(null, depth), opening);
                int winner;
                try {
                    winner = Arena.play(tunedSeat == 0 ? tuned : reference, tunedSeat == 0 ? reference : tuned, null, size[0], size[1]).winner();
                } catch (RuntimeException | AssertionError e) { // a game the referee could not finish (a late answer): a draw, so one bad turn does not end the run
                    System.out.println("  game not finished (" + e.getMessage() + "): counted as a draw");
                    winner = -1;
                }
                total += winner == tunedSeat ? 1 : winner < 0 ? 0.5 : 0;
            }
        return total / (2 * openings.size());
    }

    /** SerenaBot with the genome (its constants if null), at a fixed depth, or at its time budget if depth is 0. */
    static SerenaBot serena(double[] genome, int depth) {
        SerenaBot bot = new SerenaBot();
        if (genome != null) applySerena(bot, genome);
        if (depth > 0) {
            bot.maxDepth = depth;
            bot.budgetNanos = depthCapNanos;
        }
        return bot;
    }

    /**
     * Thinking time of a fixed-depth SerenaBot ({@code --cap-ms}, none by default). The engine skips a
     * turn answered after 100 ms, and a depth that is cheap in most positions can blow up in a few
     * (bombs on a wide board), so a cap keeps those turns from deciding games.
     */
    static long depthCapNanos = Long.MAX_VALUE / 4;

    /** SerenaBot's own genome: its 16 parameters in QuarkBot's layout, then its table size in bits. */
    static double[] serenaGenome() {
        SerenaBot bot = new SerenaBot();
        QuarkBot shown = new QuarkBot();
        for (int d = 0; d < 3; d++) System.arraycopy(bot.lineWeight[d], 0, shown.lineWeight[d], 0, shown.lineWeight[d].length);
        shown.bombReadyBonus = bot.bombReadyBonus;
        double[] genome = Arrays.copyOf(Tools.parameters(shown), GENES);
        genome[TABLE_GENE] = SerenaBot.TABLE_BITS;
        return genome;
    }

    static void applySerena(SerenaBot bot, double[] genome) {
        QuarkBot weights = new QuarkBot();
        Tools.applyParameters(weights, Arrays.copyOf(genome, TABLE_GENE));
        bot.lineWeight = new int[3][];
        for (int d = 0; d < 3; d++) bot.lineWeight[d] = Arrays.copyOf(weights.lineWeight[d], SerenaBot.LINE_LENGTH + 1);
        bot.bombReadyBonus = weights.bombReadyBonus;
        if (genome.length > TABLE_GENE) bot.resizeTable(tableBits(genome));
    }

    static int tableBits(double[] genome) {
        return (int) Math.round(genome[TABLE_GENE]);
    }

    /** The transposition table's memory: two longs per entry. */
    static double tableMegabytes(double[] genome) {
        return 16.0 * (1L << tableBits(genome)) / (1 << 20);
    }

    /** Random plain drops on a board of the given width. */
    static List<int[]> openingsOn(Random random, int count, int columns) {
        List<int[]> openings = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int[] opening = new int[OPENING_PLIES];
            for (int ply = 0; ply < OPENING_PLIES; ply++) opening[ply] = QuarkBot.action(random.nextInt(columns), false);
            openings.add(opening);
        }
        return openings;
    }

    /** Appends a line to results/islands.tsv, writing the header first if the file is new. */
    static void appendRow(String line) throws Exception {
        Files.createDirectories(ISLAND_ROWS.toAbsolutePath().getParent());
        if (!Files.exists(ISLAND_ROWS))
            Files.writeString(ISLAND_ROWS, "run\tisland\tgeneration\tbest_fitness\tmean_fitness\tbest_score\tbest_table_bits\tseconds\n");
        Files.writeString(ISLAND_ROWS, line + "\n", StandardOpenOption.APPEND);
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
