package ch.qc.meelik.tools;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Round robin between the entrants of a league file, ranked like the real tournament: wins, then
 * draws. Every pair plays the given number of games, alternating who starts, so with 2 games each
 * bot has the first move once. Writes the standings and a head-to-head table to results/league.md,
 * so they can be committed alongside the change that moved them, and every game to
 * out/league.html, the replay viewer with highscores and a table of all matches.
 *
 * <p>Usage: League [file=league.txt] [games per pair=8] [threads=cores/2]. Each non-comment line of
 * the file is "label spec", with spec as in {@link Arena}, or a directory ending in "/": every
 * parameter file in it is an entrant, labelled by its file name and played by QuarkBot's search.
 * Games run in parallel, which slows each bot down a little; keep threads at or below half the
 * cores so time budgets stay realistic.
 */
public final class League {
    record Entrant(String label, String spec) { }

    record Pairing(int first, int second, StringBuilder frames, Future<Arena.Result> result) { }

    static final Path REPORT = Path.of("results/league.md"), REPLAY = Path.of("out/league.html");

    public static void main(String[] args) throws Exception {
        Path file = Path.of(args.length > 0 ? args[0] : "league.txt");
        int gamesPerPair = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        int threads = args.length > 2 ? Integer.parseInt(args[2]) : Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        List<Entrant> entrants = readEntrants(file);
        int n = entrants.size();

        long started = System.nanoTime();
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Pairing> pairings = new ArrayList<>();
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++)
                for (int game = 0; game < gamesPerPair; game++) {
                    int first = game % 2 == 0 ? a : b, second = first == a ? b : a;
                    String firstSpec = entrants.get(first).spec(), secondSpec = entrants.get(second).spec();
                    long seed = 1000L * a + 31L * b + game;
                    StringBuilder frames = new StringBuilder();
                    pairings.add(new Pairing(first, second, frames, pool.submit(
                            () -> Arena.play(Arena.make(firstSpec, seed), Arena.make(secondSpec, seed + 1), frames))));
                }

        int[][] winsDrawsLosses = new int[n][3];
        double[][] points = new double[n][n]; // points[row][column]: row bot's points against column bot
        Arena.PlayerStats[] stats = new Arena.PlayerStats[n];
        for (int i = 0; i < n; i++) stats[i] = new Arena.PlayerStats();
        List<String> games = new ArrayList<>();
        for (Pairing pairing : pairings) {
            Arena.Result result = pairing.result().get();
            games.add(Replay.gameJson(entrants.get(pairing.first()).label(), entrants.get(pairing.second()).label(), result, pairing.frames()));
            int[] seats = {pairing.first(), pairing.second()};
            for (int seat = 0; seat < 2; seat++) {
                int bot = seats[seat], opponent = seats[1 - seat];
                boolean draw = result.winner() < 0, won = result.winner() == seat;
                winsDrawsLosses[bot][draw ? 1 : won ? 0 : 2]++;
                points[bot][opponent] += draw ? 0.5 : won ? 1 : 0;
                stats[bot].add(result.players()[seat]);
            }
        }
        pool.shutdown();

        Integer[] ranking = new Integer[n];
        for (int i = 0; i < n; i++) ranking[i] = i;
        Arrays.sort(ranking, Comparator.<Integer>comparingInt(i -> -winsDrawsLosses[i][0]).thenComparingInt(i -> -winsDrawsLosses[i][1]));

        StringBuilder report = new StringBuilder();
        report.append(String.format("League: %d bots, %d games per pair, %s, %.0f s%n%n", n, gamesPerPair, LocalDate.now(), (System.nanoTime() - started) / 1e9));
        report.append("| # | Bot | Spec | W | D | L | Max ms/move | Skipped turns |\n|---|---|---|---|---|---|---|---|\n");
        for (int place = 0; place < n; place++) {
            int bot = ranking[place];
            report.append(String.format("| %d | %s | `%s` | %d | %d | %d | %.1f | %d |%n", place + 1, entrants.get(bot).label(), entrants.get(bot).spec(),
                    winsDrawsLosses[bot][0], winsDrawsLosses[bot][1], winsDrawsLosses[bot][2], stats[bot].maxNanos / 1e6, stats[bot].skippedTurns));
        }
        report.append("\nHead to head (points of the row bot out of ").append(gamesPerPair).append("; a draw is half a point):\n\n| |");
        for (int column = 0; column < n; column++) report.append(' ').append(entrants.get(ranking[column]).label()).append(" |");
        report.append("\n|---|").append("---|".repeat(n)).append('\n');
        for (int row = 0; row < n; row++) {
            report.append("| ").append(entrants.get(ranking[row]).label()).append(" |");
            for (int column = 0; column < n; column++)
                report.append(row == column ? " - |" : " " + formatPoints(points[ranking[row]][ranking[column]]) + " |");
            report.append('\n');
        }

        System.out.print(report);
        Files.createDirectories(REPORT.toAbsolutePath().getParent());
        Files.writeString(REPORT, report);
        System.out.println("\nwritten to " + REPORT);

        List<String> bots = new ArrayList<>();
        for (Entrant entrant : entrants) bots.add(Replay.botJson(entrant.label(), entrant.spec()));
        Replay.write(REPLAY, bots, games);
    }

    static List<Entrant> readEntrants(Path file) throws Exception {
        List<Entrant> entrants = new ArrayList<>();
        for (String line : Files.readAllLines(file)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] parts = line.trim().split("\\s+", 2);
            if (parts.length == 1 && parts[0].endsWith("/")) entrants.addAll(modelsIn(Path.of(parts[0])));
            else entrants.add(new Entrant(parts[0], parts[1]));
        }
        return entrants;
    }

    /** Every parameter file in a directory, as an entrant "name quark:dir/name.txt", sorted by name. */
    static List<Entrant> modelsIn(Path directory) throws Exception {
        List<Entrant> models = new ArrayList<>();
        try (var files = Files.list(directory)) {
            for (Path file : files.sorted().toList()) {
                String name = file.getFileName().toString();
                if (name.endsWith(".txt")) models.add(new Entrant(name.substring(0, name.length() - 4), "quark:" + directory.resolve(name).toString().replace('\\', '/')));
            }
        }
        return models;
    }

    static String formatPoints(double points) {
        return points == Math.floor(points) ? Integer.toString((int) points) : String.format("%.1f", points);
    }
}
