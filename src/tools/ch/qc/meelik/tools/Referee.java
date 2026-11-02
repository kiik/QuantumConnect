package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.engine.Board;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

/**
 * Plays bots against each other in the engine's own {@code Experiment}, the reference referee,
 * instead of our model of the rules. Use it to confirm that a bot behaves in the real engine as it
 * does in {@link Arena}: no skipped turns, bombs where it meant them, and similar results.
 *
 * <p>Needs the real engine JAR (qc-starter/libs) and Java 25 at run time; {@code ./qc.sh referee}
 * sets both up. Experiment is reached by reflection, so the tools still compile against the
 * stand-ins in src/stub, which have no Experiment.
 *
 * <p>Usage: Referee [botA=jar:out/EvolvedBot.jar] [botB=ch.cern.quantumconnect.bots.LinearBot]
 * [games=10], bot specs as in Arena; the engine's own bots are in {@code ch.cern.quantumconnect.bots}
 * (RandomBot, SpecialMoveBot, LinearBot).
 *
 * <p>{@code Referee conformance [games=200]} checks our rule model against the engine instead: the
 * engine's SpecialMoveBot (random columns, bombing whenever it may) plays itself, and each game's
 * history is replayed on our {@link Board}. Every bomb the engine allowed must be legal for us, and
 * the final board and the winner must be the same.
 *
 * <p>{@code Referee watch [botA] [botB]} plays one game in the engine's window instead, as the
 * starter's EmulatorRunner does: the engine paints each move and prints its messages.
 */
public final class Referee {
    static final String EXPERIMENT = "ch.cern.quantumconnect.core.Experiment";

    public static void main(String[] args) throws Exception {
        String botA = args.length > 0 ? args[0] : "jar:out/EvolvedBot.jar";
        String botB = args.length > 1 ? args[1] : "ch.cern.quantumconnect.bots.LinearBot";
        Class<?> experimentClass = Class.forName(EXPERIMENT);
        if (botA.equals("conformance")) {
            conformance(experimentClass, args.length > 1 ? Integer.parseInt(args[1]) : 200);
            return;
        }
        if (botA.equals("watch")) {
            watch(experimentClass, args.length > 1 ? args[1] : "jar:out/EvolvedBot.jar", args.length > 2 ? args[2] : "ch.cern.quantumconnect.bots.LinearBot");
            return;
        }
        int games = args.length > 2 ? Integer.parseInt(args[2]) : 10;

        int winsA = 0, draws = 0, winsB = 0;
        for (int game = 0; game < games; game++) {
            boolean aStarts = game % 2 == 0;
            String first = aStarts ? botA : botB, second = aStarts ? botB : botA;
            long started = System.nanoTime();
            Outcome outcome = play(experimentClass, Arena.make(first, game), Arena.make(second, game));
            int winnerSeat = outcome.winnerSeat();
            if (winnerSeat < 0) draws++;
            else if ((winnerSeat == 0) == aStarts) winsA++;
            else winsB++;
            System.out.printf("game %d: %s (X) vs %s (O): %s, %d quarks placed, bombs X %d O %d, %.1f s%n", game + 1, first, second,
                    winnerSeat < 0 ? "draw" : (winnerSeat == 0 ? first : second) + " wins", outcome.placed(), outcome.bombs()[0], outcome.bombs()[1],
                    (System.nanoTime() - started) / 1e9);
        }
        System.out.printf("%s vs %s in the real Experiment, %d games: %s wins %d, draws %d, %s wins %d%n", botA, botB, games, botA, winsA, draws, botB, winsB);
    }

    // ------------------------------------------------------------------ one game in the real engine

    /** Winner seat (0 first, 1 second, -1 draw), quarks placed, and bombs per seat. */
    record Outcome(int winnerSeat, int placed, int[] bombs) { }

    /**
     * Runs one Experiment without a window or console output. When it ends COLLAPSED (a line was
     * made), its current player is the one who made the last move, the winner; any other ending
     * (full board, the iteration cap) is a draw.
     */
    static Outcome play(Class<?> experimentClass, Algorithm first, Algorithm second) throws Exception {
        Object experiment = experimentClass.getConstructor(Algorithm.class, Algorithm.class, boolean.class, boolean.class)
                .newInstance(first, second, false, false);
        experimentClass.getMethod("runExperiment").invoke(experiment);
        String status = experimentClass.getMethod("getExperimentStatus").invoke(experiment).toString();
        String lastMover = experimentClass.getMethod("getCurrentPlayer").invoke(experiment).toString();
        int winnerSeat = status.equals("COLLAPSED") ? lastMover.equals("FIRST_PLAYER") ? 0 : 1 : -1;

        List<?> history = (List<?>) experimentClass.getMethod("getHistory").invoke(experiment);
        int[] bombs = new int[2];
        for (Object move : history) {
            boolean special = (Boolean) move.getClass().getMethod("isSpecialMove").invoke(move);
            String player = move.getClass().getMethod("player").invoke(move).toString();
            if (special) bombs[player.equals("FIRST_PLAYER") ? 0 : 1]++;
        }
        return new Outcome(winnerSeat, history.size(), bombs);
    }

    /**
     * One game in the engine's window: the two-bot constructor paints and prints. The engine plays on
     * its own thread, so the program keeps running until the window is closed.
     */
    static void watch(Class<?> experimentClass, String first, String second) throws Exception {
        System.out.printf("%s (first) vs %s (second) in the engine's window%n", first, second);
        Object experiment = experimentClass.getConstructor(Algorithm.class, Algorithm.class).newInstance(Arena.make(first, 0), Arena.make(second, 0));
        experimentClass.getMethod("runExperiment").invoke(experiment);
    }

    // ------------------------------------------------------------------ our rules against the engine's

    static void conformance(Class<?> experimentClass, int games) throws Exception {
        int moves = 0, bombs = 0, wins = 0;
        for (int game = 0; game < games; game++) {
            Algorithm bomber = (Algorithm) Class.forName("ch.cern.quantumconnect.bots.SpecialMoveBot").getConstructor(double.class).newInstance(1.0);
            Algorithm other = (Algorithm) Class.forName("ch.cern.quantumconnect.bots.SpecialMoveBot").getConstructor(double.class).newInstance(1.0);
            Object experiment = experimentClass.getConstructor(Algorithm.class, Algorithm.class, boolean.class, boolean.class)
                    .newInstance(bomber, other, false, false);
            experimentClass.getMethod("runExperiment").invoke(experiment);
            String status = experimentClass.getMethod("getExperimentStatus").invoke(experiment).toString();
            String lastMover = experimentClass.getMethod("getCurrentPlayer").invoke(experiment).toString();
            int engineWinner = status.equals("COLLAPSED") ? lastMover.equals("FIRST_PLAYER") ? 0 : 1 : -1;

            Board board = new Board();
            List<?> history = (List<?>) experimentClass.getMethod("getHistory").invoke(experiment);
            for (Object move : history) {
                boolean special = (Boolean) move.getClass().getMethod("isSpecialMove").invoke(move);
                int column = (Integer) move.getClass().getMethod("column").invoke(move);
                int player = move.getClass().getMethod("player").invoke(move).toString().equals("FIRST_PLAYER") ? 0 : 1;
                if (player != board.sideToMove) throw new AssertionError("game " + game + ": a turn was skipped; replay needs every turn");
                int action = Board.action(column, special);
                if (!board.isLegal(action)) throw new AssertionError("game " + game + ", move " + moves + ": engine allowed " + Tools.describeAction(action) + ", our rules don't");
                board.play(action);
                moves++;
                if (special) bombs++;
            }

            SpaceOwnership[][] engineField = finalField(experimentClass, experiment);
            if (!Arrays.deepEquals(engineField, Tools.field(board)))
                throw new AssertionError("game " + game + ": final boards differ\nengine:\n" + Arrays.deepToString(engineField) + "\nours:\n" + Tools.render(board));
            if (board.winner() != engineWinner)
                throw new AssertionError("game " + game + ": engine winner " + engineWinner + ", ours " + board.winner());
            if (engineWinner >= 0) wins++;
        }
        System.out.printf("conformance: %d engine games replayed on our Board (%d moves, %d bombs, %d won, %d drawn): boards and winners identical%n",
                games, moves, bombs, wins, games - wins);
    }

    /** The engine's board at the end of the game; Experiment keeps it private, and tools may use reflection. */
    static SpaceOwnership[][] finalField(Class<?> experimentClass, Object experiment) throws Exception {
        Field quantumField = experimentClass.getDeclaredField("quantumField");
        quantumField.setAccessible(true);
        Object field = quantumField.get(experiment);
        return (SpaceOwnership[][]) field.getClass().getMethod("getField").invoke(field);
    }
}
