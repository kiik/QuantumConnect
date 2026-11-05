package ch.qc.meelik.tools;

import ch.qc.meelik.bots.SerenaBot;
import ch.qc.meelik.bots.ThreatBot;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Sparring: a candidate SerenaBot against a partner that may think as long as it likes, such as
 * {@code threat@1000} ({@link ThreatBot}, a second per move). The engine's referee would skip the
 * partner's late answers, so these games are refereed on our own model, SerenaBot's, whose rules
 * {@code ./qc.sh conformance} checks against the engine's. The candidate still keeps the arena's
 * limit ({@link Arena#MOVE_LIMIT_MS}, 50 ms): a later answer skips its turn, as on the server.
 *
 * <p>The concepts:
 * <ul>
 *   <li><b>Clean games.</b> Every move of both sides is known, with no skipped turn that the move
 *       history hides, unlike the tournament's archive. The candidate's lost games are written in
 *       qcsite's losses format, so {@code ./qc.sh learn} can take lessons from them.
 *   <li><b>Pairs.</b> Each opening, the first plies of an archived match (mirrored and shifted, see
 *       {@link Openings}), is played twice, the candidate moving first in one game and second in the
 *       other.
 * </ul>
 *
 * <p>Usage: Sparring candidate partner [games=20] [size=15x30] [threads=8] [opening plies=6] [losses file].
 * Bot specs as in {@link Arena}; both must be SerenaBot or a subclass.
 */
public final class Sparring {
    static final long LATE_NANOS = Arena.MOVE_LIMIT_NANOS;
    static final int MAX_PLIES = 1000;

    /** One game: the winner by seat (-1 a draw), the candidate's seat, the moves as "f12 s13b", and statistics. */
    record Game(int winner, int candidateSeat, String moves, int plies, int candidateLate, long candidateMaxNanos,
                int[] bombs, int partnerThreatWins) { }

    public static void main(String[] args) throws Exception {
        String candidate = args[0], partner = args[1];
        int games = args.length > 2 ? Integer.parseInt(args[2]) : 20;
        int[] size = Tools.size(args.length > 3 ? args[3] : "15x30");
        int threads = args.length > 4 ? Integer.parseInt(args[4]) : 8;
        int openingPlies = args.length > 5 ? Integer.parseInt(args[5]) : 6;
        Path lossFile = args.length > 6 ? Path.of(args[6]) : null;

        Openings openings = Openings.fromArchive(size[0], size[1], openingPlies);
        Random random = new Random(1);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Future<Game>> results = new ArrayList<>();
        for (int pair = 0; pair < (games + 1) / 2; pair++) {
            int[] opening = openings.isEmpty() ? new int[0] : openings.draw(random);
            for (int seat = 0; seat < 2 && results.size() < games; seat++) {
                int candidateSeat = seat;
                results.add(pool.submit(() -> play(candidate, partner, size, opening, candidateSeat)));
            }
        }

        int wins = 0, draws = 0, losses = 0, late = 0, partnerThreatWins = 0;
        long plies = 0, maxNanos = 0;
        int[] bombs = new int[2];
        List<String> lossLines = new ArrayList<>();
        lossLines.add(String.format("# %s lost to %s, sparring on our model, %dx%d", candidate, partner, size[0], size[1]));
        for (int i = 0; i < results.size(); i++) {
            Game game = results.get(i).get();
            if (game.winner < 0) draws++;
            else if (game.winner == game.candidateSeat) wins++;
            else {
                losses++;
                lossLines.add(String.format("spar-%04d %dx%d %s %s", i, size[0], size[1], game.candidateSeat == 0 ? "first" : "second", game.moves));
            }
            plies += game.plies;
            late += game.candidateLate;
            maxNanos = Math.max(maxNanos, game.candidateMaxNanos);
            bombs[0] += game.bombs[0];
            bombs[1] += game.bombs[1];
            partnerThreatWins += game.partnerThreatWins;
            if ((i + 1) % 10 == 0) System.out.printf("  %d/%d games: %d-%d-%d%n", i + 1, results.size(), wins, draws, losses);
        }
        pool.shutdown();
        System.out.printf("%s vs %s, %d games on %dx%d from %d-ply archived openings%n", candidate, partner, results.size(), size[0], size[1], openingPlies);
        System.out.printf("  candidate wins %d, draws %d, losses %d: score %.1f%%, avg %.1f plies%n", wins, draws, losses,
                100.0 * (wins + draws / 2.0) / results.size(), (double) plies / results.size());
        System.out.printf("  candidate: late answers %d, slowest %.1f ms, bombs %d; partner: bombs %d, threat-space wins %d%n",
                late, maxNanos / 1e6, bombs[0], bombs[1], partnerThreatWins);
        if (lossFile != null) {
            Files.write(lossFile, lossLines, StandardCharsets.UTF_8);
            System.out.printf("  %d lost games written to %s%n", losses, lossFile);
        }
    }

    // ------------------------------------------------------------------ one game

    static SerenaBot make(String spec, int[] size) {
        if (!(Arena.make(spec, 0) instanceof SerenaBot bot)) throw new IllegalArgumentException(spec + " is not a SerenaBot");
        bot.setUp(size[0], size[1]);
        return bot;
    }

    static Game play(String candidateSpec, String partnerSpec, int[] size, int[] opening, int candidateSeat) {
        SerenaBot[] bots = new SerenaBot[2];
        bots[candidateSeat] = make(candidateSpec, size);
        bots[1 - candidateSeat] = make(partnerSpec, size);
        StringBuilder moves = new StringBuilder();
        int plies = 0, late = 0;
        long maxNanos = 0;
        int[] bombs = new int[2];
        for (int column : opening) {
            int drop = SerenaBot.action(column, false);
            if (!bots[0].isLegal(drop)) continue;
            for (SerenaBot bot : bots) bot.play(drop);
            append(moves, plies % 2, drop);
            plies++;
        }
        SerenaBot referee = bots[0];
        while (!referee.hasLine() && plies < MAX_PLIES) {
            int seat = referee.sideToMove;
            SerenaBot mover = bots[seat];
            long started = System.nanoTime();
            int action = mover.chooseAction(mover.budgetNanos);
            long took = System.nanoTime() - started;
            if (action == SerenaBot.NO_ACTION) break; // the board is full
            if (!referee.isLegal(action)) throw new IllegalStateException(Tools.describeAction(action) + " is illegal");
            if (seat == candidateSeat) {
                maxNanos = Math.max(maxNanos, took);
                if (took > LATE_NANOS) { // the server would skip this turn
                    late++;
                    for (SerenaBot bot : bots) bot.skipTurn();
                    plies++;
                    continue;
                }
            }
            for (SerenaBot bot : bots) bot.play(action);
            if (SerenaBot.isBomb(action)) bombs[seat == candidateSeat ? 0 : 1]++;
            append(moves, seat, action);
            plies++;
        }
        int winner = referee.hasLine() ? 1 - referee.sideToMove : -1; // a line wins for the last mover
        int threatWins = bots[1 - candidateSeat] instanceof ThreatBot threat ? threat.threatWins : 0;
        return new Game(winner, candidateSeat, moves.toString().strip(), plies, late, maxNanos, bombs, threatWins);
    }

    static void append(StringBuilder moves, int seat, int action) {
        moves.append(seat == 0 ? " f" : " s").append(Tools.describeAction(action));
    }
}
