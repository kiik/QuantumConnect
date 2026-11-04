package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.bots.SerenaBot;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The data side of SerenaBot's network (the trainer is tools/qc-net).
 *
 * <p>{@code Network dataset [out dir] [--search ms] [--self-play games] [--threads n] [--seed s]}
 * replays every archived match (archive/matches/*.json) on SerenaBot's model and writes, for the
 * position before each move, the network's input features from the mover's point of view
 * ({@link SerenaBot#features}) and the game's result for the mover. With {@code --search}, each
 * position is also labelled with SerenaBot's own score for it after a search of that many ms (line
 * weights, no network), and {@code --self-play} adds games of SerenaBot against itself from random
 * openings, each move searched once to label the position and pick the move. Games run in parallel:
 * <ul>
 *   <li>{@code planes.bin}: one bit per feature, {@code ceil(features / 8)} bytes per position,
 *       most significant bit first (numpy's unpackbits order);
 *   <li>{@code results.bin}: one byte per position, 0 lost, 1 drawn, 2 won;
 *   <li>{@code scores.bin}: with {@code --search}, the search score for the mover (int32, little-endian;
 *       a won position scores near {@link SerenaBot#WIN});
 *   <li>{@code games.bin}: the match number of each position (int32, little-endian), to split by match;
 *       self-play games are numbered after the archive's;
 *   <li>{@code meta.txt}: the board size and feature layout, as {@code key value} lines.
 * </ul>
 *
 * <p>{@code Network bench [every n] [budget ms]} checks, with the network on, that the incremental
 * accumulators after a replay and a search equal those of the same position built afresh, then
 * measures nodes/s and the mean depth reached with the network off and on, searching every n-th
 * position of the archive.
 */
public final class Network {

    record ArchivedGame(String matchId, int rows, int columns, int winner, int[] players, int[] actions) {}

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "bench";
        List<ArchivedGame> games = readArchive(Path.of("archive/matches"));
        if (mode.equals("dataset")) dataset(games, settings(args));
        else if (mode.equals("bench")) {
            int every = args.length > 1 ? Integer.parseInt(args[1]) : 40;
            long budget = (args.length > 2 ? Long.parseLong(args[2]) : 50) * 1_000_000L;
            checkAccumulators(games, every);
            bench(games, every, budget);
        } else throw new IllegalArgumentException("usage: Network dataset [out dir] [--search ms] [--self-play games] [--threads n] [--seed s] | Network bench [every n] [budget ms]");
    }

    // ------------------------------------------------------------------ dataset

    /** Dataset settings: search time per label (0: no labels), self-play games, threads. */
    record Settings(Path directory, long searchNanos, int selfPlayGames, int threads, long seed) {}

    /** One game's positions: feature bits, result and search score for the mover, the summed depth of its searches. */
    record GameRows(int number, byte[] planes, byte[] results, int[] scores, long depthSum, int positions) {}

    static Settings settings(String[] args) {
        Path directory = Path.of("out/dataset");
        long search = 0, seed = 1;
        int selfPlay = 0, threads = Runtime.getRuntime().availableProcessors();
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--search" -> search = Long.parseLong(args[++i]) * 1_000_000L;
                case "--self-play" -> selfPlay = Integer.parseInt(args[++i]);
                case "--threads" -> threads = Integer.parseInt(args[++i]);
                case "--seed" -> seed = Long.parseLong(args[++i]);
                default -> directory = Path.of(args[i]);
            }
        }
        return new Settings(directory, search, selfPlay, threads, seed);
    }

    static void dataset(List<ArchivedGame> games, Settings settings) throws Exception {
        int rows = games.get(0).rows(), columns = games.get(0).columns();
        SerenaBot.useNetwork = false; // the labels come from a search with the line weights
        int features = new SerenaBot(rows, columns).featureCount();
        long started = System.nanoTime();

        ExecutorService pool = Executors.newFixedThreadPool(settings.threads());
        List<Future<GameRows>> jobs = new ArrayList<>();
        int archived = 0;
        for (int number = 0; number < games.size(); number++) {
            ArchivedGame game = games.get(number);
            if (game.rows() != rows || game.columns() != columns) continue;
            archived++;
            int id = number;
            jobs.add(pool.submit(() -> replayAndLabel(id, game, features, settings.searchNanos())));
        }
        for (int i = 0; i < settings.selfPlayGames(); i++) {
            int id = games.size() + i; // self-play games are numbered after the archive
            Random random = new Random(settings.seed() * 1_000_003L + i);
            jobs.add(pool.submit(() -> selfPlay(id, rows, columns, features, settings.searchNanos(), random)));
        }

        ByteBuffer planes = ByteBuffer.allocate(0), results = ByteBuffer.allocate(0), scores = ByteBuffer.allocate(0), matches = ByteBuffer.allocate(0);
        int positions = 0, selfPlayPositions = 0;
        long depthSum = 0;
        for (Future<GameRows> job : jobs) {
            GameRows game = job.get();
            if (planes.remaining() < game.planes().length) planes = grow(planes, game.planes().length);
            if (results.remaining() < game.positions()) results = grow(results, game.positions());
            if (scores.remaining() < 4 * game.positions()) scores = grow(scores, 4 * game.positions());
            if (matches.remaining() < 4 * game.positions()) matches = grow(matches, 4 * game.positions());
            planes.put(game.planes());
            results.put(game.results());
            for (int i = 0; i < game.positions(); i++) {
                scores.putInt(game.scores()[i]);
                matches.putInt(game.number());
            }
            positions += game.positions();
            if (game.number() >= games.size()) selfPlayPositions += game.positions();
            depthSum += game.depthSum();
        }
        pool.shutdown();
        double seconds = (System.nanoTime() - started) / 1e9;

        Path directory = settings.directory();
        long searchMs = settings.searchNanos() / 1_000_000;
        double meanDepth = (double) depthSum / Math.max(1, positions);
        Files.createDirectories(directory);
        Files.write(directory.resolve("planes.bin"), contents(planes));
        Files.write(directory.resolve("results.bin"), contents(results));
        Files.write(directory.resolve("games.bin"), contents(matches));
        if (searchMs > 0) Files.write(directory.resolve("scores.bin"), contents(scores));
        else Files.deleteIfExists(directory.resolve("scores.bin"));
        Files.writeString(directory.resolve("meta.txt"), String.format(
                "# %s: ./qc.sh dataset, %d archived matches and %d self-play games of %dx%d; search %d ms per position (mean depth %.2f), %d threads, %.0f s%n"
                        + "rows %d%ncolumns %d%nfeatures %d%ncell_planes %d%ncooldown_buckets %d%npositions %d%ngames %d%narchive_games %d%nsearch_ms %d%nwin_score %d%n",
                LocalDateTime.now().truncatedTo(ChronoUnit.SECONDS), archived, settings.selfPlayGames(), rows, columns, searchMs, meanDepth,
                settings.threads(), seconds, rows, columns, features, SerenaBot.CELL_PLANES, SerenaBot.COOLDOWN_BUCKETS, positions,
                archived + settings.selfPlayGames(), games.size(), searchMs, SerenaBot.WIN));
        System.out.printf("%d positions (%d from %d self-play games) and %d archived matches (%dx%d, %d features) -> %s%n",
                positions, selfPlayPositions, settings.selfPlayGames(), archived, rows, columns, features, directory);
        if (searchMs == 0) return;
        double perSecond = positions / seconds;
        System.out.printf("labelled by a %d ms search each, mean depth %.2f: %.0f s, %.0f positions/s on %d threads%n",
                searchMs, meanDepth, seconds, perSecond, settings.threads());
        System.out.printf("a longer run on these threads: 100k positions take %.0f min at %d ms, %.0f min at %d ms;"
                        + " 1M positions at %d ms take %.1f h%n",
                1e5 / perSecond / 60, searchMs, 4e5 / perSecond / 60, 4 * searchMs, searchMs, 1e6 / perSecond / 3600);
    }

    /** An archived game's positions, each labelled by a search of its own when searchNanos > 0. */
    static GameRows replayAndLabel(int number, ArchivedGame game, int features, long searchNanos) {
        SerenaBot board = new SerenaBot(game.rows(), game.columns());
        Rows rows = new Rows(features);
        for (int move = 0; move < game.actions().length; move++) {
            if (board.sideToMove != game.players()[move]) board.skipTurn(); // a turn lost to a timeout
            if (board.hasLine() || !board.isLegal(game.actions()[move])) break;
            rows.add(board, searchNanos);
            board.play(game.actions()[move]);
        }
        return rows.finish(number, game.winner());
    }

    /**
     * SerenaBot against itself from a random opening of 2 to 8 drops. Each position after the opening
     * is searched once: the search labels it and picks the move.
     */
    static GameRows selfPlay(int number, int rowCount, int columnCount, int features, long searchNanos, Random random) {
        SerenaBot board = new SerenaBot(rowCount, columnCount);
        for (int opening = 2 + random.nextInt(7); opening > 0; opening--) {
            int drop = SerenaBot.action(random.nextInt(columnCount), false);
            if (board.isLegal(drop)) board.play(drop);
        }
        Rows rows = new Rows(features);
        for (int turn = 0; !board.hasLine() && turn < rowCount * columnCount; turn++) {
            int action = rows.add(board, Math.max(searchNanos, 1_000_000L));
            if (action == SerenaBot.NO_ACTION) break; // the board is full
            board.play(action);
        }
        return rows.finish(number, board.hasLine() ? 1 - board.sideToMove : -1); // a line wins for the last mover
    }

    /** The positions of one game as they are collected. */
    static final class Rows {
        final int rowBytes;
        final int[] active;
        final List<byte[]> bits = new ArrayList<>();
        final List<Integer> movers = new ArrayList<>(), scores = new ArrayList<>();
        long depthSum;

        Rows(int features) {
            rowBytes = (features + 7) / 8;
            active = new int[features];
        }

        /** Records the position; when searchNanos > 0, searches it too and returns the search's action, else NO_ACTION. */
        int add(SerenaBot board, long searchNanos) {
            byte[] row = new byte[rowBytes];
            for (int i = 0, count = board.features(board.sideToMove, active); i < count; i++)
                row[active[i] >> 3] |= (byte) (0x80 >> (active[i] & 7));
            bits.add(row);
            movers.add(board.sideToMove);
            if (searchNanos == 0) {
                scores.add(0);
                return SerenaBot.NO_ACTION;
            }
            int action = board.chooseAction(searchNanos);
            scores.add(board.lastScore);
            depthSum += board.depthReached;
            return action;
        }

        /** The game's rows, with each position's result for its mover (winner 0, 1 or -1 for a draw). */
        GameRows finish(int number, int winner) {
            int positions = bits.size();
            byte[] planes = new byte[positions * rowBytes], results = new byte[positions];
            int[] scoreArray = new int[positions];
            for (int i = 0; i < positions; i++) {
                System.arraycopy(bits.get(i), 0, planes, i * rowBytes, rowBytes);
                results[i] = (byte) (winner < 0 ? 1 : winner == movers.get(i) ? 2 : 0);
                scoreArray[i] = scores.get(i);
            }
            return new GameRows(number, planes, results, scoreArray, depthSum, positions);
        }
    }

    /** A little-endian buffer with room for at least {@code more} bytes beyond the old one's content. */
    static ByteBuffer grow(ByteBuffer buffer, int more) {
        ByteBuffer grown = ByteBuffer.allocate(Math.max(2 * buffer.capacity(), buffer.position() + 1024 * more)).order(ByteOrder.LITTLE_ENDIAN);
        return grown.put(buffer.flip());
    }

    static byte[] contents(ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.position()];
        buffer.flip().get(bytes);
        return bytes;
    }

    // ------------------------------------------------------------------ bench

    /** With the network on: after a replay and a search, the network's scores equal those of the same position loaded afresh. */
    static void checkAccumulators(List<ArchivedGame> games, int every) {
        SerenaBot.useNetwork = true;
        int checked = 0, wrong = 0;
        for (ArchivedGame game : games)
            for (int end = every; end < game.actions().length; end += every) {
                SerenaBot played = replay(game, end);
                if (played == null) break;
                if (!played.networkOn) {
                    System.out.printf("no network trained for %dx%d: train one with tools/qc-net%n", game.rows(), game.columns());
                    SerenaBot.useNetwork = false;
                    return;
                }
                played.chooseAction(5_000_000L);
                SerenaBot fresh = new SerenaBot(game.rows(), game.columns());
                fresh.load(Learn.fieldFrom(played, 0), SpaceOwnership.FIRST_PLAYER);
                fresh.bombCooldown[0] = played.bombCooldown[0];
                fresh.bombCooldown[1] = played.bombCooldown[1];
                for (int side = 0; side < 2; side++) {
                    played.sideToMove = fresh.sideToMove = side;
                    if (played.networkScore() != fresh.networkScore()) wrong++;
                    checked++;
                }
            }
        SerenaBot.useNetwork = false;
        System.out.printf("accumulators: %d of %d network scores after replay + search differ from a fresh load%n", wrong, checked);
        if (wrong > 0) throw new AssertionError("incremental accumulators drift");
    }

    static void bench(List<ArchivedGame> games, int every, long budget) {
        for (int round = 0; round < 4; round++) {
            SerenaBot.useNetwork = round % 2 == 1;
            long nodes = 0, nanos = 0, depths = 0;
            int positions = 0;
            boolean networkOn = false;
            for (ArchivedGame game : games)
                for (int end = every; end < game.actions().length; end += every) {
                    SerenaBot position = replay(game, end);
                    if (position == null) break;
                    networkOn = position.networkOn;
                    long started = System.nanoTime();
                    position.chooseAction(budget);
                    nanos += System.nanoTime() - started;
                    nodes += position.nodesSearched;
                    depths += position.depthReached;
                    positions++;
                }
            System.out.printf("network %-3s: %.0fk nodes/s, mean depth %.2f (%d positions, %d ms each)%n", networkOn ? "on" : "off",
                    nodes / (nanos / 1e6), (double) depths / positions, positions, budget / 1_000_000);
        }
        SerenaBot.useNetwork = false;
    }

    /** The position before move {@code end}, or null if the game was over or broke our rules by then. */
    static SerenaBot replay(ArchivedGame game, int end) {
        SerenaBot board = new SerenaBot(game.rows(), game.columns());
        for (int move = 0; move <= end; move++) {
            if (board.sideToMove != game.players()[move]) board.skipTurn();
            if (board.hasLine() || !board.isLegal(game.actions()[move])) return null;
            if (move < end) board.play(game.actions()[move]);
        }
        return board;
    }

    // ------------------------------------------------------------------ the archive

    /** Every archived match, in file name order; the winner is 0 (first), 1 (second) or -1 (draw). */
    static List<ArchivedGame> readArchive(Path directory) throws IOException {
        List<Path> files;
        try (var stream = Files.list(directory)) {
            files = stream.filter(file -> file.toString().endsWith(".json")).sorted().toList();
        }
        List<ArchivedGame> games = new ArrayList<>();
        for (Path file : files) {
            String json = Files.readString(file, StandardCharsets.UTF_8);
            String[] moves = field(json, "moves").trim().split("\\s+");
            int[] players = new int[moves.length], actions = new int[moves.length];
            for (int i = 0; i < moves.length; i++) {
                players[i] = moves[i].charAt(0) == 'f' ? 0 : 1;
                actions[i] = SerenaBot.parseAction(moves[i].substring(1));
            }
            String winner = field(json, "winner");
            games.add(new ArchivedGame(field(json, "match_id"), Integer.parseInt(field(json, "rows")), Integer.parseInt(field(json, "columns")),
                    winner.equals("first") ? 0 : winner.equals("second") ? 1 : -1, players, actions));
        }
        return games;
    }

    /** A top-level value of the archive's flat JSON, string or number. */
    static String field(String json, String name) {
        Matcher matcher = Pattern.compile("\"" + name + "\":\\s*\"?([^\",}]*)").matcher(json);
        if (!matcher.find()) throw new IllegalArgumentException("no " + name + " in " + json);
        return matcher.group(1).trim();
    }
}
