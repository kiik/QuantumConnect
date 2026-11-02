package ch.qc.meelik.tools;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Records games for the browser replay viewer: plays them headless, then writes
 * out/replay.html (the viewer from src/tools/replay.html with the games embedded). Open it in any
 * browser to step through the games at human speed.
 *
 * <p>Usage: Replay [botA=quark] [botB=random] [games=2] [out=out/replay.html], bot specs as in Arena.
 */
public final class Replay {
    public static void main(String[] args) throws Exception {
        String a = args.length > 0 ? args[0] : "quark", b = args.length > 1 ? args[1] : "random";
        int games = args.length > 2 ? Integer.parseInt(args[2]) : 2;
        Path out = Path.of(args.length > 3 ? args[3] : "out/replay.html");
        StringBuilder json = new StringBuilder("[");
        for (int g = 0; g < games; g++) {
            boolean aFirst = g % 2 == 0;
            String first = aFirst ? a : b, second = aFirst ? b : a;
            StringBuilder plies = new StringBuilder();
            Arena.Result r = Arena.game(Arena.make(first, g), Arena.make(second, g), plies);
            String result = r.winner() < 0 ? "draw" : (r.winner() == 0 ? first : second) + " wins";
            System.out.printf("game %d: %s (X) vs %s (O): %s in %d plies%n", g + 1, first, second, result, r.plies());
            if (g > 0) json.append(',');
            json.append(String.format("{\"x\":\"%s\",\"o\":\"%s\",\"result\":\"%s\",\"plies\":[%s]}", first, second, result, plies));
        }
        json.append(']');
        String html = Files.readString(Path.of("src/tools/replay.html")).replace("/*GAMES*/[]", json);
        Files.createDirectories(out.toAbsolutePath().getParent());
        Files.writeString(out, html);
        System.out.println("written to " + out);
    }
}
