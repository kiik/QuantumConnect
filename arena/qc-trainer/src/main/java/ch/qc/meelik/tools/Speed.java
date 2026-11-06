package ch.qc.meelik.tools;

import ch.qc.meelik.bots.SerenaBot;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * SerenaBot's speed on archived positions, for work on the search's speed that must not change its play.
 *
 * <h2>The positions</h2>
 * The position before every n-th move of the archived matches of one board size (archive/matches,
 * read as {@link Network} reads them), in file name order, up to a limit. Each is built afresh on a
 * new bot, so every search starts from the same empty table and history.
 *
 * <h2>The two measurements</h2>
 * <ul>
 *   <li>{@code fixed}: a search of each position to a fixed depth, with no time limit. It prints the
 *       nodes per second and writes one line per position, its chosen action, score and nodes, to a
 *       file. Given a reference file written before a change, it reports every position whose
 *       action, score or node count differs: a change that only makes the search faster changes none.
 *   <li>{@code timed}: a search of each position within a time budget, as in a game. It prints the
 *       mean depth reached, the nodes per second and the slowest search, and writes one line per
 *       position to out/speed-timed.txt: match, move, action, score, depth, nodes, microseconds.
 * </ul>
 * The environment variable {@code SERENA} may give settings for every searched bot, as a field line
 * does ({@code SERENA=bombNear=1 ./qc.sh speed timed}), to measure a toggle against its default.
 * The first positions are searched once beforehand, untimed, so that the JVM has compiled the search.
 *
 * <p>Usage: {@code Speed fixed [depth] [RxC] [every n] [limit] [out file] [reference file]} and
 * {@code Speed timed [ms] [RxC] [every n] [limit]}.
 */
public final class Speed {

    /** Positions searched untimed first, to compile the search. */
    static final int WARM_UP_POSITIONS = 20;

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "fixed";
        int[] size = Tools.size(args.length > 2 ? args[2] : "15x30");
        int every = args.length > 3 ? Integer.parseInt(args[3]) : 11;
        int limit = args.length > 4 ? Integer.parseInt(args[4]) : 300;
        List<Position> positions = positions(size, every, limit);
        System.out.printf("%d positions on %dx%d, every %d-th move of the archived matches%n", positions.size(), size[0], size[1], every);
        if (mode.equals("fixed")) {
            int depth = args.length > 1 ? Integer.parseInt(args[1]) : 5;
            fixed(positions, depth, Path.of(args.length > 5 ? args[5] : "out/speed-fixed.txt"), args.length > 6 ? Path.of(args[6]) : null);
        } else if (mode.equals("timed")) {
            timed(positions, (args.length > 1 ? Long.parseLong(args[1]) : 40) * 1_000_000L);
        } else throw new IllegalArgumentException("usage: Speed fixed [depth] [RxC] [every n] [limit] [out file] [reference file] | Speed timed [ms] [RxC] [every n] [limit]");
    }

    // ------------------------------------------------------------------ the positions

    /** A position: its match, the move before which it stands, and the game to replay to it. */
    record Position(Network.ArchivedGame game, int end) {
        /** The board at this position, with the SerenaBot settings of SPEED_SETTINGS (as in a field file's {@code #...}) if set. */
        SerenaBot board() {
            SerenaBot board = Network.replay(game, end);
            if (board != null && SETTINGS != null && !SETTINGS.isBlank()) Arena.applySettings(board, SETTINGS);
            return board;
        }
    }

    /** SerenaBot settings for every searched board, as in a field line ({@link Arena#applySettings}): environment variable SERENA (or SPEED_SETTINGS). */
    static final String SETTINGS = System.getenv("SERENA") != null ? System.getenv("SERENA") : System.getenv("SPEED_SETTINGS");

    static List<Position> positions(int[] size, int every, int limit) throws Exception {
        List<Position> positions = new ArrayList<>();
        for (Network.ArchivedGame game : Network.readArchive(Path.of("archive/matches"))) {
            if (game.rows() != size[0] || game.columns() != size[1]) continue;
            for (int end = every; end < game.actions().length && positions.size() < limit; end += every) {
                if (Network.replay(game, end) == null) break; // over, or broken by our rules
                positions.add(new Position(game, end));
            }
            if (positions.size() >= limit) break;
        }
        return positions;
    }

    // ------------------------------------------------------------------ fixed depth

    static void fixed(List<Position> positions, int depth, Path out, Path reference) throws Exception {
        for (int i = 0; i < Math.min(WARM_UP_POSITIONS, positions.size()); i++) search(positions.get(i), depth, Long.MAX_VALUE / 4);
        List<String> lines = new ArrayList<>();
        long nodes = 0, nanos = 0, slowest = 0;
        for (Position position : positions) {
            SerenaBot board = position.board();
            board.maxDepth = depth;
            long started = System.nanoTime();
            int action = board.chooseAction(Long.MAX_VALUE / 4);
            long took = System.nanoTime() - started;
            nodes += board.nodesSearched;
            nanos += took;
            slowest = Math.max(slowest, took);
            lines.add(String.format("%s %d %d %d %d", position.game().matchId(), position.end(), action, board.lastScore, board.nodesSearched));
        }
        Files.write(out, lines, StandardCharsets.UTF_8);
        System.out.printf("fixed depth %d: %.0fk nodes/s, %.1f ms per position (slowest %.1f ms), %d nodes -> %s%n",
                depth, nodes / (nanos / 1e6), nanos / 1e6 / positions.size(), slowest / 1e6, nodes, out);
        if (reference == null) return;
        List<String> expected = Files.readAllLines(reference, StandardCharsets.UTF_8);
        int differ = 0, actionsDiffer = 0;
        for (int i = 0; i < Math.min(expected.size(), lines.size()); i++) {
            if (expected.get(i).equals(lines.get(i))) continue;
            differ++;
            String[] was = expected.get(i).split(" "), is = lines.get(i).split(" ");
            if (!was[2].equals(is[2])) actionsDiffer++;
            if (differ <= 10) System.out.printf("  differs: %s  (was %s)%n", lines.get(i), expected.get(i));
        }
        System.out.printf("against %s: %d of %d positions differ, %d in the action chosen%n", reference, differ, lines.size(), actionsDiffer);
    }

    // ------------------------------------------------------------------ timed

    static void timed(List<Position> positions, long budget) throws Exception {
        for (int i = 0; i < Math.min(WARM_UP_POSITIONS, positions.size()); i++) search(positions.get(i), SerenaBot.MAX_PLY - 2, budget);
        long nodes = 0, nanos = 0, slowest = 0, depths = 0;
        List<String> lines = new ArrayList<>();
        for (Position position : positions) {
            SerenaBot board = position.board();
            long started = System.nanoTime();
            int action = board.chooseAction(budget);
            long took = System.nanoTime() - started;
            nodes += board.nodesSearched;
            nanos += took;
            slowest = Math.max(slowest, took);
            depths += board.depthReached;
            lines.add(String.format("%s %d %d %d %d %d %d", position.game().matchId(), position.end(), action, board.lastScore,
                    board.depthReached, board.nodesSearched, took / 1000));
        }
        Files.write(Path.of("out/speed-timed.txt"), lines, StandardCharsets.UTF_8); // match, move, action, score, depth, nodes, microseconds
        System.out.printf("timed %d ms: mean depth %.3f, %.0fk nodes/s, %.0f nodes per move, slowest %.1f ms%n",
                budget / 1_000_000, (double) depths / positions.size(), nodes / (nanos / 1e6), (double) nodes / positions.size(), slowest / 1e6);
    }

    private static void search(Position position, int depth, long budget) {
        SerenaBot board = position.board();
        board.maxDepth = depth;
        board.chooseAction(budget);
    }
}
