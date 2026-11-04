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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The data side of SerenaBot's network (the trainer is tools/qc-net).
 *
 * <p>{@code Network dataset [out dir]} replays every archived match (archive/matches/*.json) on
 * SerenaBot's model and writes, for the position before each move, the network's input features from
 * the mover's point of view ({@link SerenaBot#features}) and the game's result for the mover:
 * <ul>
 *   <li>{@code planes.bin}: one bit per feature, {@code ceil(features / 8)} bytes per position,
 *       most significant bit first (numpy's unpackbits order);
 *   <li>{@code results.bin}: one byte per position, 0 lost, 1 drawn, 2 won;
 *   <li>{@code games.bin}: the match number of each position (int32, little-endian), to split by match;
 *   <li>{@code meta.txt}: the board size and feature layout, as {@code key value} lines.
 * </ul>
 *
 * <p>{@code Network bench [every n] [budget ms]} checks, with the network on, that the incremental
 * accumulators after a replay and a search equal those of the same position built afresh, then
 * measures nodes/s with the network off and on, searching every n-th position of the archive.
 */
public final class Network {

    record ArchivedGame(String matchId, int rows, int columns, int winner, int[] players, int[] actions) {}

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "bench";
        List<ArchivedGame> games = readArchive(Path.of("archive/matches"));
        if (mode.equals("dataset")) dataset(games, Path.of(args.length > 1 ? args[1] : "out/dataset"));
        else if (mode.equals("bench")) {
            int every = args.length > 1 ? Integer.parseInt(args[1]) : 40;
            long budget = (args.length > 2 ? Long.parseLong(args[2]) : 50) * 1_000_000L;
            checkAccumulators(games, every);
            bench(games, every, budget);
        } else throw new IllegalArgumentException("usage: Network dataset [out dir] | Network bench [every n] [budget ms]");
    }

    // ------------------------------------------------------------------ dataset

    static void dataset(List<ArchivedGame> games, Path directory) throws IOException {
        ArchivedGame sizeOf = games.get(0);
        SerenaBot layout = new SerenaBot(sizeOf.rows(), sizeOf.columns());
        int features = layout.featureCount(), rowBytes = (features + 7) / 8;
        int[] active = new int[features];
        ByteBuffer planes = ByteBuffer.allocate(0), results = ByteBuffer.allocate(0), matches = ByteBuffer.allocate(0);
        int positions = 0, used = 0;
        for (int number = 0; number < games.size(); number++) {
            ArchivedGame game = games.get(number);
            if (game.rows() != sizeOf.rows() || game.columns() != sizeOf.columns()) continue;
            used++;
            SerenaBot board = new SerenaBot(game.rows(), game.columns());
            for (int move = 0; move < game.actions().length; move++) {
                if (board.sideToMove != game.players()[move]) board.skipTurn(); // a turn lost to a timeout
                if (board.hasLine() || !board.isLegal(game.actions()[move])) break;
                if (planes.remaining() < rowBytes) {
                    planes = grow(planes, rowBytes);
                    results = grow(results, 1);
                    matches = grow(matches, 4);
                }
                byte[] bits = new byte[rowBytes];
                for (int i = 0, count = board.features(board.sideToMove, active); i < count; i++)
                    bits[active[i] >> 3] |= (byte) (0x80 >> (active[i] & 7));
                planes.put(bits);
                results.put((byte) (game.winner() < 0 ? 1 : game.winner() == board.sideToMove ? 2 : 0));
                matches.putInt(number);
                positions++;
                board.play(game.actions()[move]);
            }
        }
        Files.createDirectories(directory);
        Files.write(directory.resolve("planes.bin"), contents(planes));
        Files.write(directory.resolve("results.bin"), contents(results));
        Files.write(directory.resolve("games.bin"), contents(matches));
        Files.writeString(directory.resolve("meta.txt"), String.format(
                "# %s: ./qc.sh dataset, %d archived matches of %dx%d%nrows %d%ncolumns %d%nfeatures %d%ncell_planes %d%ncooldown_buckets %d%npositions %d%ngames %d%n",
                LocalDateTime.now().truncatedTo(ChronoUnit.SECONDS), used, sizeOf.rows(), sizeOf.columns(), sizeOf.rows(), sizeOf.columns(),
                features, SerenaBot.CELL_PLANES, SerenaBot.COOLDOWN_BUCKETS, positions, used));
        System.out.printf("%d positions from %d matches (%dx%d, %d features) -> %s%n", positions, used, sizeOf.rows(), sizeOf.columns(), features, directory);
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
            long nodes = 0, nanos = 0;
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
                    positions++;
                }
            System.out.printf("network %-3s: %.0fk nodes/s (%d positions, %d ms each)%n", networkOn ? "on" : "off",
                    nodes / (nanos / 1e6), positions, budget / 1_000_000);
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
