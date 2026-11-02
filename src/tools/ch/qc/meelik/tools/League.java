package ch.qc.meelik.tools;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Round robin between all entrants of a league file, ranked like the real tournament (wins, then
 * draws). Writes a markdown table to results/league.md so the standings can be committed.
 *
 * <p>Usage: League [file=league.txt] [games per pair=8] [threads=cores/2]. Each non-comment line
 * of the file is "label spec", with spec as in {@link Arena}. Games run in parallel, which slows
 * each bot down a little; keep threads at or below half the cores so time budgets stay realistic.
 */
public final class League {
    static String fmt(double x) {
        return x == Math.floor(x) ? Integer.toString((int) x) : String.format("%.1f", x);
    }

    public static void main(String[] args) throws Exception {
        Path file = Path.of(args.length > 0 ? args[0] : "league.txt");
        int games = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        int threads = args.length > 2 ? Integer.parseInt(args[2]) : Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        List<String[]> bots = new ArrayList<>();
        for (String line : Files.readAllLines(file))
            if (!line.isBlank() && !line.startsWith("#")) bots.add(line.trim().split("\\s+", 2));
        int n = bots.size();
        int[][] wdl = new int[n][3];
        long[] maxNs = new long[n], skipped = new long[n];
        double[][] points = new double[n][n]; // head to head: points of row bot against column bot

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<int[]> jobs = new ArrayList<>(); // {i, j, game}
        List<Future<Arena.Result>> results = new ArrayList<>();
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                for (int g = 0; g < games; g++) {
                    String first = bots.get(g % 2 == 0 ? i : j)[1], second = bots.get(g % 2 == 0 ? j : i)[1];
                    long seed = 1000L * i + 31L * j + g;
                    jobs.add(new int[]{i, j, g});
                    results.add(pool.submit(() -> Arena.game(Arena.make(first, seed), Arena.make(second, seed + 1), null)));
                }
        long t0 = System.nanoTime();
        for (int k = 0; k < jobs.size(); k++) {
            int[] job = jobs.get(k);
            Arena.Result r = results.get(k).get();
            int first = job[2] % 2 == 0 ? job[0] : job[1], second = first == job[0] ? job[1] : job[0];
            int[] seat = {first, second};
            for (int p = 0; p < 2; p++) {
                int me = seat[p];
                wdl[me][r.winner() < 0 ? 1 : r.winner() == p ? 0 : 2]++;
                points[me][seat[1 - p]] += r.winner() < 0 ? 0.5 : r.winner() == p ? 1 : 0;
                maxNs[me] = Math.max(maxNs[me], r.stats()[p][2]);
                skipped[me] += r.stats()[p][3];
            }
        }
        pool.shutdown();

        Integer[] rank = new Integer[n];
        for (int i = 0; i < n; i++) rank[i] = i;
        Arrays.sort(rank, Comparator.<Integer>comparingInt(i -> -wdl[i][0]).thenComparingInt(i -> -wdl[i][1]));
        StringBuilder md = new StringBuilder();
        md.append(String.format("League: %d bots, %d games per pair, %s, %.0f s%n%n", n, games,
                java.time.LocalDate.now(), (System.nanoTime() - t0) / 1e9));
        md.append("| # | Bot | Spec | W | D | L | Max ms/move | Skipped turns |\n|---|---|---|---|---|---|---|---|\n");
        for (int r = 0; r < n; r++) {
            int i = rank[r];
            md.append(String.format("| %d | %s | `%s` | %d | %d | %d | %.1f | %d |%n", r + 1, bots.get(i)[0], bots.get(i)[1],
                    wdl[i][0], wdl[i][1], wdl[i][2], maxNs[i] / 1e6, skipped[i]));
        }
        md.append("\nHead to head (points of the row bot out of ").append(games).append("; a draw is half a point):\n\n| |");
        for (int c = 0; c < n; c++) md.append(' ').append(bots.get(rank[c])[0]).append(" |");
        md.append("\n|---|").append("---|".repeat(n)).append('\n');
        for (int r = 0; r < n; r++) {
            md.append("| ").append(bots.get(rank[r])[0]).append(" |");
            for (int c = 0; c < n; c++) md.append(r == c ? " - |" : String.format(" %s |", fmt(points[rank[r]][rank[c]])));
            md.append('\n');
        }
        System.out.print(md);
        Files.createDirectories(Path.of("results"));
        Files.writeString(Path.of("results/league.md"), md);
        System.out.println("\nwritten to results/league.md");
    }
}
