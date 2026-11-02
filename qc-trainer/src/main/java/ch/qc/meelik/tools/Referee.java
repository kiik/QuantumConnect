package ch.qc.meelik.tools;

import ch.cern.quantumconnect.bots.SpecialMoveBot;
import ch.cern.quantumconnect.core.Experiment;
import ch.cern.quantumconnect.core.ExperimentStatus;
import ch.cern.quantumconnect.core.Move;
import ch.cern.quantumconnect.core.QuantumField;
import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.engine.Board;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

/**
 * Tools around the engine's own {@code Experiment} (the qc-starter library). Every Arena game is
 * already played in it; these two need it directly.
 *
 * <p>{@code Referee conformance [games=200]} checks our rule model, {@link Board}, against the
 * engine: the engine's SpecialMoveBot (random columns, bombing whenever it may) plays itself, and
 * each game's history is replayed on our Board. Every bomb the engine allowed must be legal for us,
 * and the final board and the winner must be the same. Arena games check the same, but our bots
 * bomb far less often than this.
 *
 * <p>{@code Referee watch [botA] [botB]} plays one game in the engine's window, as the starter's
 * EmulatorRunner does: the engine paints each move and prints its messages. Bot specs as in Arena.
 */
public final class Referee {

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "conformance";
        if (mode.equals("watch")) watch(args.length > 1 ? args[1] : "jar:qc-starter/build/libs/EvolvedBot.jar", args.length > 2 ? args[2] : "ch.cern.quantumconnect.bots.LinearBot");
        else if (mode.equals("conformance")) conformance(args.length > 1 ? Integer.parseInt(args[1]) : 200);
        else throw new IllegalArgumentException("usage: Referee conformance [games] | Referee watch [botA] [botB]");
    }

    /**
     * One game in the engine's window: the two-bot constructor paints and prints. The engine plays on
     * its own thread, so the program keeps running until the window is closed.
     */
    static void watch(String first, String second) throws InterruptedException {
        System.out.printf("%s (first) vs %s (second) in the engine's window%n", first, second);
        new Experiment(Arena.make(first, 0), Arena.make(second, 0)).runExperiment();
    }

    // ------------------------------------------------------------------ our rules against the engine's

    static void conformance(int games) throws Exception {
        int moves = 0, bombs = 0, wins = 0;
        for (int game = 0; game < games; game++) {
            Experiment experiment = new Experiment(new SpecialMoveBot(1.0), new SpecialMoveBot(1.0), false, false);
            experiment.runExperiment();
            int engineWinner = experiment.getExperimentStatus() == ExperimentStatus.COLLAPSED ? Arena.seatOf(experiment.getCurrentPlayer()) : -1;

            Board board = new Board();
            List<Move> history = experiment.getHistory();
            for (Move move : history) {
                int player = Arena.seatOf(move.player());
                if (player != board.sideToMove) throw new AssertionError("game " + game + ": a turn was skipped; replay needs every turn");
                int action = Board.action(move.column(), move.isSpecialMove());
                if (!board.isLegal(action)) throw new AssertionError("game " + game + ", move " + moves + ": engine allowed " + Tools.describeAction(action) + ", our rules don't");
                board.play(action);
                moves++;
                if (move.isSpecialMove()) bombs++;
            }

            SpaceOwnership[][] engineField = finalField(experiment);
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
    static SpaceOwnership[][] finalField(Experiment experiment) {
        try {
            Field quantumField = Experiment.class.getDeclaredField("quantumField");
            quantumField.setAccessible(true);
            return ((QuantumField) quantumField.get(experiment)).getField();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("this engine version keeps its board elsewhere", e);
        }
    }
}
