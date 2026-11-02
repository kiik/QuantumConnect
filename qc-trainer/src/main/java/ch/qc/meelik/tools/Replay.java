package ch.qc.meelik.tools;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Records games for the browser replay viewer. The games are played here, in Java, exactly as in
 * {@link Arena}; the page only plays the recording back. Writes out/replay.html: the viewer
 * (qc-trainer/src/main/resources/replay.html) with the bots and the games embedded. {@link League}
 * writes its whole round robin the same way, to out/league.html.
 *
 * <p>Usage: Replay [botA=quark] [botB=random] [games=2] [out=out/replay.html], bot specs as in Arena.
 */
public final class Replay {
    static final Path TEMPLATE = Path.of("qc-trainer/src/main/resources/replay.html");

    public static void main(String[] args) throws Exception {
        String botA = args.length > 0 ? args[0] : "quark", botB = args.length > 1 ? args[1] : "random";
        int games = args.length > 2 ? Integer.parseInt(args[2]) : 2;
        Path output = Path.of(args.length > 3 ? args[3] : "out/replay.html");

        List<String> gameRecords = new ArrayList<>();
        for (int game = 0; game < games; game++) {
            String first = game % 2 == 0 ? botA : botB, second = game % 2 == 0 ? botB : botA;
            StringBuilder frames = new StringBuilder();
            Arena.Result result = Arena.play(Arena.make(first, game), Arena.make(second, game), frames);
            System.out.printf("game %d: %s (X) vs %s (O): %s in %d plies%n", game + 1, first, second, outcome(first, second, result), result.plies());
            gameRecords.add(gameJson(first, second, result, frames));
        }
        write(output, List.of(botJson(botA, botA), botJson(botB, botB)), gameRecords);
    }

    // ------------------------------------------------------------------ the recording format

    /** A bot as the viewer lists it: its label, its spec, and the approach it stands for. */
    static String botJson(String label, String spec) {
        return String.format("{\"label\":\"%s\",\"spec\":\"%s\",\"approach\":\"%s\"}", label, spec, approach(spec));
    }

    /** One game: who played X and O, the outcome (winner 0 = X, 1 = O, -1 = draw), and one frame per ply. */
    static String gameJson(String first, String second, Arena.Result result, CharSequence frames) {
        return String.format("{\"x\":\"%s\",\"o\":\"%s\",\"winner\":%d,\"result\":\"%s\",\"plies\":[%s]}",
                first, second, result.winner(), outcome(first, second, result), frames);
    }

    static String outcome(String first, String second, Arena.Result result) {
        return result.winner() < 0 ? "draw" : (result.winner() == 0 ? first : second) + " wins";
    }

    /** The viewer page with the bots and games filled in. */
    static void write(Path output, List<String> bots, List<String> games) throws Exception {
        String page = Files.readString(TEMPLATE)
                .replace("/*BOTS*/[]", "[" + String.join(",", bots) + "]")
                .replace("/*GAMES*/[]", "[" + String.join(",\n", games) + "]");
        Files.createDirectories(output.toAbsolutePath().getParent());
        Files.writeString(output, page);
        System.out.println("replay of " + games.size() + " games written to " + output);
    }

    // ------------------------------------------------------------------ approaches

    /**
     * The solution method behind a bot spec, so that results can be grouped by method as well as
     * by model. A parameter file's method is read from its provenance: Tune writes "start
     * defaults" for weights evolved from scratch and "start &lt;file&gt;" for transfer learning.
     */
    static String approach(String spec) {
        if (spec.equals("random")) return "random moves";
        if (spec.startsWith("jar:")) return "packaged JAR";
        if (spec.equals("tuned")) spec = "quark:" + Tune.OUTPUT;
        if (spec.contains(":")) {
            String provenance = "";
            try {
                provenance = Files.readAllLines(Path.of(spec.substring(spec.indexOf(':') + 1))).get(0);
            } catch (Exception unreadable) {
                // treated as weights evolved from scratch
            }
            return provenance.contains("start params") ? "transfer-learned weights" : "self-play evolved weights";
        }
        if (spec.contains("/")) return "fixed-depth search";
        switch (spec.split("[@/]")[0]) {
            case "quark": return "hand-written weights";
            case "evolved": return "self-play evolved weights";
            case "threat": return "threat-sequence search";
            case "parity": return "odd/even parity rules";
            default: return spec.split("[@/]")[0];
        }
    }
}
