package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.bots.SerenaBot;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Learns SerenaBot's loss book from lost tournament games.
 *
 * <h2>Input</h2>
 * Files written by {@code qcsite losses}: one lost game per line, {@code <match id> <rows>x<columns>
 * <our side> <moves>}, a move being {@code f} or {@code s} (first or second player), the column, and
 * {@code b} for a bomb. Lines starting with {@code #} are comments.
 *
 * <h2>What a lesson is</h2>
 * Each game is replayed on SerenaBot's model. At each of our last {@code window} moves before the
 * loss, the position is built exactly as the bot sees it in the arena (we are player 0, both bomb
 * cooldowns), and two searches run with the offline budget, far beyond the arena's 100 ms:
 * <ul>
 *   <li>the position itself: the move a deep search prefers, and its score;
 *   <li>the position after the move we actually played, from the opponent's side: what that move
 *       was really worth.
 * </ul>
 * If the preferred move differs and is better by at least {@code min-gap}, or escapes a forced loss
 * the played move walked into, the position and the preferred move become a book entry.
 *
 * <h2>Output</h2>
 * The entries are merged into the book file (params/serena/book.txt; a newer lesson on the same
 * position replaces the older one), with a {@code #} line recording the run, and the whole book is
 * written into SerenaBot.java's LOSS_BOOK block. Games already in the book (by match id) are skipped,
 * so learning from the same file twice adds nothing.
 *
 * <p>Usage: {@code Learn <losses file>... [--budget ms] [--window moves] [--min-gap score]
 * [--threads n] [--book file] [--bot SerenaBot.java]}
 */
public final class Learn {

    static final String DEFAULT_BOOK = "params/serena/book.txt";
    static final String DEFAULT_BOT_SOURCE = "qc-starter/src/main/java/ch/qc/meelik/bots/SerenaBot.java";
    static final String BOOK_BEGIN = "    // LOSS BOOK BEGIN", BOOK_END = "    // LOSS BOOK END";
    /** Book entries per string constant in the generated source; a constant holds at most 64 KB. */
    static final int ENTRIES_PER_CHUNK = 400;

    // ------------------------------------------------------------------ settings

    long budgetNanos = 2_000_000_000L;
    int window = 12;
    int minimumGap = 500;
    int threads = Runtime.getRuntime().availableProcessors();
    Path bookFile = Path.of(DEFAULT_BOOK);
    Path botSource = Path.of(DEFAULT_BOT_SOURCE);
    final List<Path> lossFiles = new ArrayList<>();

    // ------------------------------------------------------------------ data

    /** A lost game as qcsite wrote it. */
    record LostGame(String matchId, int rows, int columns, int ourSeat, int[] players, int[] actions) {}

    /** One of our moves in a lost game, ready to analyse: the position from our side and what we played. */
    record Decision(LostGame game, int moveNumber, SpaceOwnership[][] field, int ourCooldown, int opponentCooldown, int played) {}

    /** A lesson: in this position play {@code action}, not {@code played}. */
    record Entry(String size, long key, int action, int played, int gap, String matchId, int moveNumber) {
        String line() {
            return String.format("%s %016x %s %d %s %d %s", size, key, Tools.describeAction(action), gap, matchId, moveNumber, Tools.describeAction(played));
        }

        String bookEntry() {
            return String.format("%s %016x %s", size, key, Tools.describeAction(action));
        }
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) throws Exception {
        Learn learn = new Learn();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--budget" -> learn.budgetNanos = Long.parseLong(args[++i]) * 1_000_000L;
                case "--window" -> learn.window = Integer.parseInt(args[++i]);
                case "--min-gap" -> learn.minimumGap = Integer.parseInt(args[++i]);
                case "--threads" -> learn.threads = Integer.parseInt(args[++i]);
                case "--book" -> learn.bookFile = Path.of(args[++i]);
                case "--bot" -> learn.botSource = Path.of(args[++i]);
                default -> learn.lossFiles.add(Path.of(args[i]));
            }
        }
        if (learn.lossFiles.isEmpty()) throw new IllegalArgumentException("usage: Learn <losses file>... [--budget ms] [--window moves] [--min-gap score] [--threads n]");
        learn.run();
    }

    void run() throws Exception {
        Map<String, Entry> book = readBook();
        Set<String> learnedMatches = new LinkedHashSet<>();
        for (Entry entry : book.values()) learnedMatches.add(entry.matchId());
        List<String> history = readHistory();

        List<LostGame> games = new ArrayList<>();
        int skipped = 0;
        for (Path file : lossFiles)
            for (LostGame game : readGames(file)) {
                if (learnedMatches.contains(game.matchId())) skipped++;
                else games.add(game);
            }
        System.out.printf("%d new lost games (%d already in the book); budget %d ms, window %d moves, min gap %d, %d threads%n",
                games.size(), skipped, budgetNanos / 1_000_000, window, minimumGap, threads);

        List<Decision> decisions = new ArrayList<>();
        for (LostGame game : games) decisions.addAll(decisionsOf(game));
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Future<Entry>> lessons = new ArrayList<>();
        for (Decision decision : decisions) lessons.add(pool.submit(() -> analyse(decision)));
        int added = 0, examined = 0;
        for (Future<Entry> lesson : lessons) {
            Entry entry = lesson.get();
            examined++;
            if (entry == null) continue;
            book.put(entry.size() + " " + entry.key(), entry);
            added++;
            System.out.printf("  %s move %d: play %s, not %s (gap %s)%n", entry.matchId().substring(0, 8), entry.moveNumber(),
                    Tools.describeAction(entry.action()), Tools.describeAction(entry.played()), describeGap(entry.gap()));
        }
        pool.shutdown();

        String stamp = LocalDateTime.now().truncatedTo(ChronoUnit.SECONDS).toString();
        history.add(String.format("# %s: %d games from %s; %d of our moves examined, %d lessons; budget %d ms, window %d, min gap %d",
                stamp, games.size(), lossFiles, examined, added, budgetNanos / 1_000_000, window, minimumGap));
        writeBook(history, book);
        writeBotSource(book);
        System.out.printf("%d lessons from %d games -> %s (%d entries), %s%n", added, games.size(), bookFile, book.size(), botSource);
    }

    static String describeGap(int gap) {
        return gap >= SerenaBot.WIN ? "escapes a forced loss" : String.valueOf(gap);
    }

    // ------------------------------------------------------------------ replaying a lost game

    /** Our last {@code window} moves of the game, each with the position the bot saw before it. */
    List<Decision> decisionsOf(LostGame game) {
        SerenaBot board = new SerenaBot(game.rows(), game.columns());
        List<Decision> ours = new ArrayList<>();
        for (int move = 0; move < game.actions().length; move++) {
            if (board.sideToMove != game.players()[move]) board.skipTurn(); // a turn lost to a timeout
            if (board.hasLine()) break;
            if (!board.isLegal(game.actions()[move])) { // checked first: analysing it would play it on a broken board
                System.out.printf("  %s: move %d is illegal on our model; replay stops there%n", game.matchId(), move + 1);
                break;
            }
            if (game.players()[move] == game.ourSeat())
                ours.add(new Decision(game, move + 1, fieldFrom(board, game.ourSeat()),
                        board.bombCooldown[game.ourSeat()], board.bombCooldown[1 - game.ourSeat()], game.actions()[move]));
            board.play(game.actions()[move]);
        }
        boolean weLost = board.hasLine() && board.sideToMove == game.ourSeat();
        if (!weLost) System.out.printf("  %s: our replay doesn't end in our loss; learning from it anyway%n", game.matchId());
        return ours.subList(Math.max(0, ours.size() - window), ours.size());
    }

    /** The board as the arena's field, with our quarks as FIRST_PLAYER: player 0 when the bot loads it. */
    static SpaceOwnership[][] fieldFrom(SerenaBot board, int ourSeat) {
        SpaceOwnership[][] field = new SpaceOwnership[board.rows][board.columns];
        for (int column = 0; column < board.columns; column++)
            for (int height = 0; height < board.rows; height++) {
                int owner = board.owner[column * board.rows + height];
                field[board.rows - 1 - height][column] = owner == SerenaBot.EMPTY ? SpaceOwnership.EMPTY
                        : owner == ourSeat ? SpaceOwnership.FIRST_PLAYER : SpaceOwnership.SECOND_PLAYER;
            }
        return field;
    }

    /** The position as SerenaBot builds it in accelerateQuark: loaded from the field, cooldowns set. */
    static SerenaBot positionOf(Decision decision) {
        LostGame game = decision.game();
        SerenaBot board = new SerenaBot(game.rows(), game.columns());
        board.load(decision.field(), SpaceOwnership.FIRST_PLAYER);
        board.bombCooldown[SerenaBot.ME] = decision.ourCooldown();
        board.bombCooldown[SerenaBot.OPPONENT] = decision.opponentCooldown();
        return board;
    }

    // ------------------------------------------------------------------ analysing one decision

    /** A book entry if a deep search clearly prefers another move to the one we played, else null. */
    Entry analyse(Decision decision) {
        SerenaBot position = positionOf(decision);
        long key = position.positionKey();
        int preferred = position.chooseAction(budgetNanos);
        int preferredScore = position.lastScore;
        if (preferred == decision.played() || preferred == SerenaBot.NO_ACTION) return null;

        SerenaBot afterPlayed = positionOf(decision);
        afterPlayed.play(decision.played());
        int playedScore;
        if (afterPlayed.hasLine()) playedScore = SerenaBot.WIN; // our move made a line: we won there
        else {
            afterPlayed.chooseAction(budgetNanos);
            playedScore = -afterPlayed.lastScore;
        }
        int lost = SerenaBot.MAX_PLY - SerenaBot.WIN;
        boolean escapesForcedLoss = playedScore < lost && preferredScore >= lost;
        int gap = escapesForcedLoss ? SerenaBot.WIN : preferredScore - playedScore;
        if (!escapesForcedLoss && gap < minimumGap) return null;
        LostGame game = decision.game();
        return new Entry(game.rows() + "x" + game.columns(), key, preferred, decision.played(), gap, game.matchId(), decision.moveNumber());
    }

    // ------------------------------------------------------------------ files

    static List<LostGame> readGames(Path file) throws IOException {
        List<LostGame> games = new ArrayList<>();
        for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            line = line.strip();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] parts = line.split("\\s+");
            int[] size = Tools.size(parts[1]);
            int ourSeat = parts[2].equals("first") ? 0 : 1;
            int moves = parts.length - 3;
            int[] players = new int[moves], actions = new int[moves];
            for (int i = 0; i < moves; i++) {
                String move = parts[3 + i];
                players[i] = move.charAt(0) == 'f' ? 0 : 1;
                actions[i] = SerenaBot.parseAction(move.substring(1));
            }
            games.add(new LostGame(parts[0], size[0], size[1], ourSeat, players, actions));
        }
        return games;
    }

    /** The book, keyed by size and position; in file order, so later lessons replace earlier ones. */
    Map<String, Entry> readBook() throws IOException {
        Map<String, Entry> book = new LinkedHashMap<>();
        if (!Files.exists(bookFile)) return book;
        for (String line : Files.readAllLines(bookFile, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] p = line.strip().split("\\s+");
            Entry entry = new Entry(p[0], Long.parseUnsignedLong(p[1], 16), SerenaBot.parseAction(p[2]),
                    SerenaBot.parseAction(p[6]), Integer.parseInt(p[3]), p[4], Integer.parseInt(p[5]));
            book.put(entry.size() + " " + entry.key(), entry);
        }
        return book;
    }

    List<String> readHistory() throws IOException {
        List<String> history = new ArrayList<>();
        if (Files.exists(bookFile))
            for (String line : Files.readAllLines(bookFile, StandardCharsets.UTF_8)) if (line.startsWith("#")) history.add(line);
        if (history.isEmpty()) {
            history.add("# SerenaBot's loss book: positions from lost tournament games and the move a deep offline search prefers.");
            history.add("# Written by ./qc.sh learn (qc-trainer Learn) from qcsite losses files in params/serena/losses/.");
            history.add("# Columns: size, position key (our side is player 0), move to play, score gap (" + SerenaBot.WIN + " = escapes a forced loss),");
            history.add("#          match id, move number in that match, the move we played there.");
        }
        return history;
    }

    void writeBook(List<String> history, Map<String, Entry> book) throws IOException {
        List<String> lines = new ArrayList<>(history);
        for (Entry entry : book.values()) lines.add(entry.line());
        Files.createDirectories(bookFile.toAbsolutePath().getParent());
        Files.write(bookFile, lines, StandardCharsets.UTF_8);
    }

    /** Rewrites the LOSS_BOOK block of SerenaBot.java with the whole book. */
    void writeBotSource(Map<String, Entry> book) throws IOException {
        List<String> source = Files.readAllLines(botSource, StandardCharsets.UTF_8);
        int begin = -1, end = -1;
        for (int i = 0; i < source.size(); i++) {
            if (source.get(i).startsWith(BOOK_BEGIN)) begin = i;
            if (source.get(i).startsWith(BOOK_END)) end = i;
        }
        if (begin < 0 || end < begin) throw new IllegalStateException(botSource + " has no LOSS BOOK BEGIN/END block");
        List<String> block = new ArrayList<>();
        block.add("    static final String[] LOSS_BOOK = {");
        List<Entry> entries = new ArrayList<>(book.values());
        for (int from = 0; from < entries.size(); from += ENTRIES_PER_CHUNK) {
            StringBuilder chunk = new StringBuilder();
            for (Entry entry : entries.subList(from, Math.min(entries.size(), from + ENTRIES_PER_CHUNK)))
                chunk.append(entry.bookEntry()).append(';');
            block.add("            \"" + chunk + "\",");
        }
        block.add("    };");
        List<String> updated = new ArrayList<>(source.subList(0, begin + 1));
        updated.addAll(block);
        updated.addAll(source.subList(end, source.size()));
        Files.write(botSource, updated, StandardCharsets.UTF_8);
    }
}
