package ch.qc.meelik.tools;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Records games for the browser replay viewer. The games are played here, in Java, exactly as in
 * {@link Arena}; the page only plays the recording back. Writes out/replay.html: the viewer
 * (src/tools/replay.html) with the games embedded.
 *
 * <p>Usage: Replay [botA=quark] [botB=random] [games=2] [out=out/replay.html], bot specs as in Arena.
 */
public final class Replay {
    public static void main(String[] args) throws Exception {
        String botA = args.length > 0 ? args[0] : "quark", botB = args.length > 1 ? args[1] : "random";
        int games = args.length > 2 ? Integer.parseInt(args[2]) : 2;
        Path output = Path.of(args.length > 3 ? args[3] : "out/replay.html");

        StringBuilder json = new StringBuilder("[");
        for (int game = 0; game < games; game++) {
            String first = game % 2 == 0 ? botA : botB, second = game % 2 == 0 ? botB : botA;
            StringBuilder frames = new StringBuilder();
            Arena.Result result = Arena.play(Arena.make(first, game), Arena.make(second, game), frames);
            String outcome = result.winner() < 0 ? "draw" : (result.winner() == 0 ? first : second) + " wins";
            System.out.printf("game %d: %s (X) vs %s (O): %s in %d plies%n", game + 1, first, second, outcome, result.plies());
            if (game > 0) json.append(',');
            json.append(String.format("{\"x\":\"%s\",\"o\":\"%s\",\"result\":\"%s\",\"plies\":[%s]}", first, second, outcome, frames));
        }
        json.append(']');

        String page = Files.readString(Path.of("src/tools/replay.html")).replace("/*GAMES*/[]", json);
        Files.createDirectories(output.toAbsolutePath().getParent());
        Files.writeString(output, page);
        System.out.println("written to " + output);
    }
}
