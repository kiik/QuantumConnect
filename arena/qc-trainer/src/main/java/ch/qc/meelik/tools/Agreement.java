package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.bots.SerenaBot;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Agreement with a deep search on the positions that lost us games against the top bots: a cheap
 * benchmark for changes to SerenaBot's search and evaluation, which tracks the depth problem
 * directly and is far less noisy than a short match.
 *
 * <h2>The positions</h2>
 * The archived matches (archive/matches/*.json, read as {@link Network} reads them) of one board
 * size that one of our bots (a name starting with {@code Tank-} or {@code Serena}) lost to one of the
 * top bots ({@link #TOP_BOTS}, or the {@code --opponent} names). Of each, our last {@code window}
 * moves, each position built as the bot sees it in the arena ({@link Learn#decisionsOf},
 * {@link Learn#positionOf}).
 *
 * <h2>The reference</h2>
 * SerenaBot as it ships searches each position for the deep budget: its move and score are the
 * reference. A move's deep score is the score of a search as long of the position after it, from the
 * opponent's side. Both are computed once and cached in the cache directory: {@code reference.tsv}
 * and {@code moves.tsv}, one line per position (and move), with the position key and the budget, so
 * a line whose position or budget no longer matches is computed again.
 *
 * <h2>Per bot spec</h2>
 * A spec is written as in the arena's field lines ({@code serena}, {@code serena#lmrLog=0},
 * {@code serena@100}, {@code threat}); it must make a SerenaBot. Each position is searched with the
 * spec's own budget (40 ms unless {@code @ms}); the loss book is not consulted. Reported:
 * <ul>
 *   <li>{@code agree}: the share of positions where it chose the reference move;
 *   <li>{@code open}: the same among the positions the reference does not already see lost, where
 *       any move would do;
 *   <li>{@code blunder}: the share where the move it chose scores as a forced loss in the deep
 *       search while the reference move does not;
 *   <li>{@code depth}: the mean depth its searches reached.
 * </ul>
 * Each spec's moves go to {@code <cache>/choices-<spec>.tsv}, one line per position.
 *
 * <p>Usage: {@code Agreement [spec...=serena] [--threads n=3] [--deep ms=2000] [--window moves=12]
 * [--size RxC=15x30] [--opponent name]... [--cache dir=out/agree]}
 */
public final class Agreement {

    /** The bots that beat us most, as the archive names them. */
    static final List<String> TOP_BOTS = List.of("GitHub CHET45", "Savi24 - \"The King\"", "coffeeAtMidNight", "anysize11054", "1104-1-1105-4");
    /** Our bots' name prefixes in the archive. */
    static final List<String> OUR_PREFIXES = List.of("Tank-", "Serena");
    /** Positions searched once beforehand by every spec, so that the JVM has compiled the search. */
    static final int WARM_UP_POSITIONS = 30;
    static final int LOST = SerenaBot.MAX_PLY - SerenaBot.WIN;

    // ------------------------------------------------------------------ settings

    final List<String> specs = new ArrayList<>();
    int threads = 3;
    long deepNanos = 2_000_000_000L;
    int window = 12;
    int[] size = {15, 30};
    final Set<String> opponents = new LinkedHashSet<>();
    Path cache = Path.of("out/agree");

    // ------------------------------------------------------------------ data

    /** A deep search's verdict on a position, or on a move in it: its key, the move, the score for us. */
    record Verdict(long key, int action, int score, int depth) {}

    /** One spec's choice in one position. */
    record Choice(int action, int score, int depth) {}

    // ------------------------------------------------------------------ main

    public static void main(String[] args) throws Exception {
        Agreement agreement = new Agreement();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--threads" -> agreement.threads = Integer.parseInt(args[++i]);
                case "--deep" -> agreement.deepNanos = Long.parseLong(args[++i]) * 1_000_000L;
                case "--window" -> agreement.window = Integer.parseInt(args[++i]);
                case "--size" -> agreement.size = Tools.size(args[++i]);
                case "--opponent" -> agreement.opponents.add(args[++i]);
                case "--cache" -> agreement.cache = Path.of(args[++i]);
                default -> {
                    if (!agreement.specs.contains(args[i])) agreement.specs.add(args[i]);
                }
            }
        }
        if (agreement.specs.isEmpty()) agreement.specs.add("serena");
        if (agreement.opponents.isEmpty()) agreement.opponents.addAll(TOP_BOTS);
        agreement.run();
    }

    void run() throws Exception {
        Learn learn = new Learn();
        learn.window = window;
        List<Learn.LostGame> games = lostGames();
        List<Learn.Decision> decisions = new ArrayList<>();
        for (Learn.LostGame game : games) decisions.addAll(learn.decisionsOf(game));
        System.out.printf("%d positions: our last %d moves of %d lost %dx%d games vs %s%n",
                decisions.size(), window, games.size(), size[0], size[1], opponents);
        if (decisions.isEmpty()) return;
        Files.createDirectories(cache);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            Map<String, Verdict> references = references(decisions, pool);
            Map<String, List<Choice>> choices = new HashMap<>();
            for (String spec : specs) choices.put(spec, choose(spec, decisions, pool));
            Map<String, Verdict> moveScores = moveScores(decisions, references, choices, pool);
            report(decisions, references, moveScores, choices);
        } finally {
            pool.shutdown();
        }
    }

    // ------------------------------------------------------------------ the lost games

    /** Our lost games of this size against the opponents, in file name order. */
    List<Learn.LostGame> lostGames() throws IOException {
        List<Path> files;
        try (var stream = Files.list(Path.of("archive/matches"))) {
            files = stream.filter(file -> file.toString().endsWith(".json")).sorted().toList();
        }
        List<Learn.LostGame> games = new ArrayList<>();
        for (Path file : files) {
            String json = Files.readString(file, StandardCharsets.UTF_8);
            String first = botName(json, "first"), second = botName(json, "second");
            int ourSeat = isOurs(first) && !isOurs(second) ? 0 : isOurs(second) && !isOurs(first) ? 1 : -1;
            if (ourSeat < 0 || !opponents.contains(ourSeat == 0 ? second : first)) continue;
            String winner = Network.field(json, "winner");
            if (!winner.equals(ourSeat == 0 ? "second" : "first")) continue;
            String moveText = Network.field(json, "moves");
            if (moveText.isBlank()) continue;
            String[] moves = moveText.split("\\s+");
            int[] players = new int[moves.length], actions = new int[moves.length];
            for (int i = 0; i < moves.length; i++) {
                players[i] = moves[i].charAt(0) == 'f' ? 0 : 1;
                actions[i] = SerenaBot.parseAction(moves[i].substring(1));
            }
            int rows = Integer.parseInt(Network.field(json, "rows"));
            int columns = Network.playedColumns(rows, Integer.parseInt(Network.field(json, "columns")), actions);
            if (rows != size[0] || columns != size[1]) continue;
            games.add(new Learn.LostGame(Network.field(json, "match_id"), rows, columns, ourSeat, players, actions));
        }
        return games;
    }

    /** The bot name of one side of a match, JSON escapes undone. */
    static String botName(String json, String side) {
        Matcher matcher = Pattern.compile("\"" + side + "\":\\s*\\{\\s*\"bot\":\\s*\"((?:[^\"\\\\]|\\\\.)*)\"").matcher(json);
        if (!matcher.find()) return "";
        return matcher.group(1).replace("\\\"", "\"").replace("\\\\", "\\");
    }

    static boolean isOurs(String name) {
        for (String prefix : OUR_PREFIXES) if (name.startsWith(prefix)) return true;
        return false;
    }

    static String idOf(Learn.Decision decision) {
        return decision.game().matchId() + " " + decision.moveNumber();
    }

    // ------------------------------------------------------------------ the deep search, cached

    /** The reference of every position: from the cache where its key and budget match, else searched now. */
    Map<String, Verdict> references(List<Learn.Decision> decisions, ExecutorService pool) throws Exception {
        Path file = cache.resolve("reference.tsv");
        Map<String, Verdict> cached = readVerdicts(file, false);
        Map<String, Verdict> references = new HashMap<>();
        List<Learn.Decision> missing = new ArrayList<>();
        for (Learn.Decision decision : decisions) {
            Verdict verdict = cached.get(idOf(decision));
            if (verdict != null && verdict.key() == Learn.positionOf(decision).positionKey()) references.put(idOf(decision), verdict);
            else missing.add(decision);
        }
        System.out.printf("reference (SerenaBot, %d ms): %d positions cached in %s, %d to search (about %.0f s with %d threads)%n",
                deepNanos / 1_000_000, references.size(), file, missing.size(), missing.size() * deepNanos / 1e9 / threads, threads);
        List<Future<Verdict>> jobs = new ArrayList<>();
        for (Learn.Decision decision : missing) jobs.add(pool.submit(() -> deepSearch(decision)));
        for (int i = 0; i < missing.size(); i++) {
            Verdict verdict = jobs.get(i).get();
            references.put(idOf(missing.get(i)), verdict);
            appendVerdict(file, idOf(missing.get(i)), verdict);
            if ((i + 1) % 50 == 0) System.out.printf("  %d of %d searched%n", i + 1, missing.size());
        }
        return references;
    }

    /** The deep score of every spec's move that is not the reference's, from the cache or searched now. */
    Map<String, Verdict> moveScores(List<Learn.Decision> decisions, Map<String, Verdict> references,
                                    Map<String, List<Choice>> choices, ExecutorService pool) throws Exception {
        Path file = cache.resolve("moves.tsv");
        Map<String, Verdict> scores = readVerdicts(file, true);
        Set<String> queued = new LinkedHashSet<>();
        List<Learn.Decision> missingDecisions = new ArrayList<>();
        List<Integer> missingActions = new ArrayList<>();
        for (String spec : specs)
            for (int i = 0; i < decisions.size(); i++) {
                Learn.Decision decision = decisions.get(i);
                Verdict reference = references.get(idOf(decision));
                int action = choices.get(spec).get(i).action();
                String id = idOf(decision) + " " + action;
                if (action == reference.action()) continue;
                Verdict known = scores.get(id);
                if (known != null && known.key() == reference.key()) continue;
                if (queued.add(id)) {
                    missingDecisions.add(decision);
                    missingActions.add(action);
                }
            }
        if (!missingDecisions.isEmpty())
            System.out.printf("deep scores of %d other moves to search (about %.0f s)%n", missingDecisions.size(),
                    missingDecisions.size() * deepNanos / 1e9 / threads);
        List<Future<Verdict>> jobs = new ArrayList<>();
        for (int i = 0; i < missingDecisions.size(); i++) {
            Learn.Decision decision = missingDecisions.get(i);
            int action = missingActions.get(i);
            jobs.add(pool.submit(() -> deepScoreOf(decision, action)));
        }
        for (int i = 0; i < jobs.size(); i++) {
            Verdict verdict = jobs.get(i).get();
            String id = idOf(missingDecisions.get(i)) + " " + missingActions.get(i);
            scores.put(id, verdict);
            appendVerdict(file, id, verdict);
        }
        return scores;
    }

    /** SerenaBot as it ships, searching the position for the deep budget. */
    Verdict deepSearch(Learn.Decision decision) {
        SerenaBot position = Learn.positionOf(decision);
        long key = position.positionKey();
        int action = position.chooseAction(deepNanos);
        return new Verdict(key, action, position.lastScore, position.depthReached);
    }

    /** The deep score of a move: the position after it, searched from the opponent's side. */
    Verdict deepScoreOf(Learn.Decision decision, int action) {
        SerenaBot after = Learn.positionOf(decision);
        long key = after.positionKey();
        after.play(action);
        if (after.hasLine()) return new Verdict(key, action, SerenaBot.WIN, 0); // our move made a line
        after.chooseAction(deepNanos);
        return new Verdict(key, action, -after.lastScore, after.depthReached);
    }

    /**
     * A cache file: {@code <match> <move number>}, then tab-separated the position key, action, score,
     * depth and budget in ms, per line; lines of another budget are left out. With {@code byAction}
     * (moves.tsv), the id also holds the action.
     */
    Map<String, Verdict> readVerdicts(Path file, boolean byAction) throws IOException {
        Map<String, Verdict> verdicts = new HashMap<>();
        if (!Files.exists(file)) return verdicts;
        for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] p = line.strip().split("\t");
            if (Long.parseLong(p[5]) != deepNanos / 1_000_000) continue;
            Verdict verdict = new Verdict(Long.parseUnsignedLong(p[1], 16), SerenaBot.parseAction(p[2]), Integer.parseInt(p[3]), Integer.parseInt(p[4]));
            verdicts.put(byAction ? p[0] + " " + verdict.action() : p[0], verdict);
        }
        return verdicts;
    }

    /** Appends a line to a cache file, writing the header first if the file is new. */
    synchronized void appendVerdict(Path file, String id, Verdict verdict) throws IOException {
        StringBuilder text = new StringBuilder();
        if (!Files.exists(file))
            text.append("# ./qc.sh agree: deep searches of positions from our lost games (Agreement). Columns: match and move number,\n")
                    .append("# position key (our side to move), action, score for us, depth reached, budget ms.\n");
        String position = id.split(" ")[0] + " " + id.split(" ")[1];
        text.append(String.format("%s\t%016x\t%s\t%d\t%d\t%d%n", position, verdict.key(), Tools.describeAction(verdict.action()),
                verdict.score(), verdict.depth(), deepNanos / 1_000_000));
        Files.writeString(file, text, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    // ------------------------------------------------------------------ the specs' searches

    /** Every position searched by the spec's bot within its budget, after a warm-up. */
    List<Choice> choose(String spec, List<Learn.Decision> decisions, ExecutorService pool) throws Exception {
        List<Future<Choice>> warmUp = new ArrayList<>();
        for (int i = 0; i < Math.min(WARM_UP_POSITIONS, decisions.size()); i++) {
            Learn.Decision decision = decisions.get(i);
            warmUp.add(pool.submit(() -> fastSearch(spec, decision)));
        }
        for (Future<Choice> job : warmUp) job.get();
        List<Future<Choice>> jobs = new ArrayList<>();
        for (Learn.Decision decision : decisions) jobs.add(pool.submit(() -> fastSearch(spec, decision)));
        List<Choice> choices = new ArrayList<>();
        for (Future<Choice> job : jobs) choices.add(job.get());
        return choices;
    }

    static Choice fastSearch(String spec, Learn.Decision decision) {
        SerenaBot bot = botFor(spec, decision);
        int action = bot.chooseAction(bot.budgetNanos);
        return new Choice(action, bot.lastScore, bot.depthReached);
    }

    /** The spec's bot, settings applied as in the arena, holding the decision's position. */
    static SerenaBot botFor(String spec, Learn.Decision decision) {
        if (!(Arena.make(spec, 1) instanceof SerenaBot bot)) throw new IllegalArgumentException(spec + " is not a SerenaBot");
        Learn.LostGame game = decision.game();
        if (bot.rows != game.rows() || bot.columns != game.columns()) bot.setUp(game.rows(), game.columns());
        bot.load(decision.field(), SpaceOwnership.FIRST_PLAYER);
        bot.bombCooldown[SerenaBot.ME] = decision.ourCooldown();
        bot.bombCooldown[SerenaBot.OPPONENT] = decision.opponentCooldown();
        return bot;
    }

    // ------------------------------------------------------------------ the report

    void report(List<Learn.Decision> decisions, Map<String, Verdict> references, Map<String, Verdict> moveScores,
                Map<String, List<Choice>> choices) throws IOException {
        int referenceLost = 0;
        long referenceDepth = 0;
        for (Learn.Decision decision : decisions) {
            Verdict reference = references.get(idOf(decision));
            if (reference.score() < LOST) referenceLost++;
            referenceDepth += reference.depth();
        }
        int n = decisions.size();
        System.out.printf("reference: mean depth %.2f; already a forced loss in %d positions (%.1f%%)%n",
                (double) referenceDepth / n, referenceLost, 100.0 * referenceLost / n);
        System.out.printf("%-40s %7s %7s %9s %7s%n", "spec", "agree%", "open%", "blunder%", "depth");
        for (String spec : specs) {
            List<String> lines = new ArrayList<>();
            lines.add("# " + spec + ": match, move number, reference move and score, chosen move, its depth and search score, its deep score");
            int agree = 0, agreeOpen = 0, blunders = 0;
            long depth = 0;
            for (int i = 0; i < n; i++) {
                Learn.Decision decision = decisions.get(i);
                Verdict reference = references.get(idOf(decision));
                Choice choice = choices.get(spec).get(i);
                int deepScore = choice.action() == reference.action() ? reference.score()
                        : moveScores.get(idOf(decision) + " " + choice.action()).score();
                if (choice.action() == reference.action()) {
                    agree++;
                    if (reference.score() >= LOST) agreeOpen++;
                } else if (deepScore < LOST && reference.score() >= LOST) blunders++;
                depth += choice.depth();
                lines.add(String.format("%s\t%d\t%s\t%d\t%s\t%d\t%d\t%d", decision.game().matchId(), decision.moveNumber(),
                        Tools.describeAction(reference.action()), reference.score(), Tools.describeAction(choice.action()),
                        choice.depth(), choice.score(), deepScore));
            }
            System.out.printf("%-40s %7.1f %7.1f %9.1f %7.2f%n", spec, 100.0 * agree / n, 100.0 * agreeOpen / Math.max(1, n - referenceLost),
                    100.0 * blunders / n, (double) depth / n);
            Files.write(cache.resolve("choices-" + spec.replaceAll("[^A-Za-z0-9=._-]", "_") + ".tsv"), lines, StandardCharsets.UTF_8);
        }
    }
}
