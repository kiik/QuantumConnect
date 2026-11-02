package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.Move;
import ch.cern.quantumconnect.core.QuantumField;
import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.QuarkBot;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Headless local arena: plays two bots against each other under the rules as we understand them
 * (QuarkBot's own rule model is the referee, so this can't catch a misread rule; the starter
 * project's Experiment is the reference for that).
 *
 * <p>Usage: Arena <botA> <botB> [games=20]. A bot spec is one of:
 * <ul>
 *   <li>{@code random}: random legal column, never bombs;
 *   <li>{@code quark}, optionally {@code @<ms>} (time budget) and/or {@code /<depth>} (fixed depth),
 *       optionally {@code :<params file>} (evaluation parameters, e.g. params/evolved-d4.txt);
 *   <li>{@code tuned}: shorthand for {@code quark:out/tune-params.txt};
 *   <li>a fully qualified class name: any other bot on the classpath (public no-arg constructor);
 *   <li>{@code jar:<path>}: the bot packaged in that JAR (its Bot-Class), in its own class loader,
 *       so e.g. two versions of QuarkBot can play each other.
 * </ul>
 */
public final class Arena {
    static final long MOVE_LIMIT = 100_000_000L;

    /** Outcome of one game: winner (0 first, 1 second, -1 draw), plies, and per-player stats. */
    record Result(int winner, int plies, long[][] stats, int[] bombs) { }

    public static void main(String[] args) {
        String a = args.length > 0 ? args[0] : "quark", b = args.length > 1 ? args[1] : "random";
        int games = args.length > 2 ? Integer.parseInt(args[2]) : 20;
        int[] score = new int[3]; // A wins, draws, B wins
        long[][] stats = new long[2][4]; // per bot: calls, total ns, max ns, skipped turns
        int[] bombs = new int[2];
        long plies = 0;
        for (int g = 0; g < games; g++) {
            boolean aFirst = g % 2 == 0;
            Result r = game(make(aFirst ? a : b, g), make(aFirst ? b : a, g), null);
            for (int p = 0; p < 2; p++) {
                int who = aFirst ? p : 1 - p;
                for (int i = 0; i < 4; i++) stats[who][i] = i == 2 ? Math.max(stats[who][i], r.stats[p][i]) : stats[who][i] + r.stats[p][i];
                bombs[who] += r.bombs[p];
            }
            plies += r.plies;
            score[r.winner < 0 ? 1 : (r.winner == 0) == aFirst ? 0 : 2]++;
        }
        System.out.printf("%s vs %s, %d games (alternating who starts)%n", a, b, games);
        System.out.printf("  %s wins %d, draws %d, %s wins %d, avg %.1f plies%n", a, score[0], score[1], b, score[2], (double) plies / games);
        for (int i = 0; i < 2; i++)
            System.out.printf("  %-10s avg %.1f ms/move, max %.1f ms, skipped turns %d, bombs %d%n", i == 0 ? a : b,
                    stats[i][1] / 1e6 / Math.max(1, stats[i][0]), stats[i][2] / 1e6, stats[i][3], bombs[i]);
    }

    /** Plays one game; when {@code log} is non-null, appends one JSON object per ply (see Replay). */
    static Result game(Algorithm first, Algorithm second, StringBuilder log) {
        Algorithm[] bots = {first, second};
        long[][] stats = new long[2][4];
        int[] bombs = new int[2];
        QuarkBot ref = new QuarkBot();
        List<Move> history = new ArrayList<>();
        int winner = -1, move = 0;
        if (log != null) log.append(ply(ref, -1, -1, false, false, 0, null, -1));
        for (; move < 1000 && winner < 0 && ref.pieces < ref.size; move++) {
            int p = ref.turn;
            long[] st = stats[p];
            SpaceOwnership me = p == 0 ? SpaceOwnership.FIRST_PLAYER : SpaceOwnership.SECOND_PLAYER;
            long t0 = System.nanoTime();
            int col = bots[p].accelerateQuark(new QuantumField(Tools.grid(ref), me, history));
            long dt = System.nanoTime() - t0;
            st[0]++; st[1] += dt; st[2] = Math.max(st[2], dt);
            if (dt > MOVE_LIMIT || col < 0 || col >= ref.cols || ref.height[col] == ref.rows) {
                st[3]++;
                ref.turn ^= 1; // turn skipped
                if (log != null) log.append(',').append(ply(ref, p, col, false, true, dt, null, -1));
                continue;
            }
            int[] before = ref.snapshot();
            boolean canBomb = ref.legal(2 * col + 1);
            int row = ref.rows - 1 - ref.height[col];
            ref.play(2 * col);
            winner = ref.winner();
            long moveNs = dt;
            String info = log == null ? null : searchInfo(bots[p], p);
            if (winner >= 0) {
                if (log != null) log.append(',').append(ply(ref, p, col, false, false, moveNs, info, winner));
                break;
            }
            t0 = System.nanoTime();
            boolean boom = bots[p].useSpecialMove(new QuantumField(Tools.grid(ref), me, history));
            dt = System.nanoTime() - t0;
            st[2] = Math.max(st[2], dt);
            if (boom && canBomb && dt <= MOVE_LIMIT) {
                ref.restore(before);
                ref.play(2 * col + 1);
                bombs[p]++;
                winner = ref.winner();
            }
            if (log != null) {
                String frame = ply(ref, p, col, boom && canBomb, false, moveNs, info, winner);
                log.append(',').append(boom && canBomb ? frame.replace("\"blast\":null", "\"blast\":[" + row + "," + col + "]") : frame);
            }
            history.add(new Move(col, boom && canBomb));
        }
        return new Result(winner, move, stats, bombs);
    }

    /** Search depth, value (first mover's view) and nodes of a QuarkBot, also one loaded from a JAR. */
    static String searchInfo(Algorithm bot, int p) {
        try {
            Class<?> c = bot.getClass();
            if (!c.getSimpleName().equals("QuarkBot")) return null;
            int v = c.getField("lastValue").getInt(bot);
            return String.format("\"d\":%d,\"v\":%d,\"n\":%d", c.getField("lastDepth").getInt(bot), p == 0 ? v : -v, c.getField("nodes").getLong(bot));
        } catch (ReflectiveOperationException e) {
            return null;
        }
    }

    /** One replay frame: who moved where, the board after it (rows top to bottom), search info. */
    static String ply(QuarkBot ref, int p, int col, boolean bomb, boolean skipped, long ns, String info, int winner) {
        StringBuilder b = new StringBuilder();
        for (int r = ref.rows - 1; r >= 0; r--)
            for (int c = 0; c < ref.cols; c++) {
                int o = ref.cell[c * ref.rows + r];
                b.append(o < 0 ? '.' : o == 0 ? 'X' : 'O');
            }
        return String.format(java.util.Locale.ROOT, "{\"p\":%d,\"c\":%d,\"bomb\":%b,\"skip\":%b,\"ms\":%.1f,\"board\":\"%s\",\"blast\":null,\"w\":%d%s}",
                p, col, bomb, skipped, ns / 1e6, b, winner, info == null ? "" : "," + info);
    }

    static Algorithm make(String spec, long seed) {
        if (spec.equals("random")) return new RandomBot(seed);
        if (spec.equals("tuned")) spec = "quark:" + Tune.OUT;
        if (spec.startsWith("jar:")) return fromJar(Path.of(spec.substring(4)));
        if (!spec.startsWith("quark")) {
            try {
                return (Algorithm) Class.forName(spec).getDeclaredConstructor().newInstance();
            } catch (ReflectiveOperationException e) {
                throw new IllegalArgumentException("unknown bot " + spec, e);
            }
        }
        QuarkBot q = new QuarkBot();
        int colon = spec.indexOf(':');
        if (colon >= 0) {
            Tools.apply(q, Tune.load(Path.of(spec.substring(colon + 1))));
            spec = spec.substring(0, colon);
        }
        if (spec.contains("@")) q.budgetNanos = Long.parseLong(spec.replaceAll(".*@(\\d+).*", "$1")) * 1_000_000L;
        if (spec.contains("/")) {
            q.maxDepth = Integer.parseInt(spec.replaceAll(".*/(\\d+).*", "$1"));
            q.budgetNanos = Long.MAX_VALUE / 4;
        }
        return q;
    }

    static Algorithm fromJar(Path jar) {
        try (java.util.jar.JarFile f = new java.util.jar.JarFile(jar.toFile())) {
            String cls = f.getManifest().getMainAttributes().getValue("Bot-Class");
            if (cls == null) throw new IllegalArgumentException(jar + " has no Bot-Class in its manifest");
            ClassLoader loader = new BotLoader(new java.net.URL[]{jar.toUri().toURL()}, Arena.class.getClassLoader());
            return (Algorithm) loader.loadClass(cls).getDeclaredConstructor().newInstance();
        } catch (java.io.IOException | ReflectiveOperationException e) {
            throw new IllegalArgumentException("can't load a bot from " + jar, e);
        }
    }

    /** Loads the bot's own classes from its JAR first; the JDK and the engine API come from the parent. */
    static final class BotLoader extends java.net.URLClassLoader {
        BotLoader(java.net.URL[] urls, ClassLoader parent) { super(urls, parent); }

        @Override
        protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            synchronized (getClassLoadingLock(name)) {
                Class<?> c = findLoadedClass(name);
                if (c == null && !name.startsWith("java.") && !name.startsWith("ch.cern.quantumconnect.")) {
                    try { c = findClass(name); } catch (ClassNotFoundException ignored) { }
                }
                if (c == null) return super.loadClass(name, resolve);
                if (resolve) resolveClass(c);
                return c;
            }
        }
    }

    /** Uniformly random legal column, never bombs. */
    static final class RandomBot implements Algorithm {
        final Random rnd;

        RandomBot(long seed) { rnd = new Random(seed); }

        @Override
        public int accelerateQuark(QuantumField f) {
            List<Integer> open = new ArrayList<>();
            for (int c = 0; c < f.getNumberOfColumns(); c++) if (!f.isColumnFull(c)) open.add(c);
            return open.isEmpty() ? 0 : open.get(rnd.nextInt(open.size()));
        }

        @Override
        public boolean useSpecialMove(QuantumField f) { return false; }
    }
}
