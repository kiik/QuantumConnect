package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.bots.QuarkBot;
import ch.qc.meelik.engine.Board;
import ch.qc.meelik.engine.Engine;

import java.util.Arrays;

/** Helpers shared by the tools. On tool boards, player 0 is the first mover (X), player 1 is O. */
final class Tools {
    private Tools() { }

    // ------------------------------------------------------------------ boards and engines

    /** A board from "rows x columns x lineLength", e.g. "8x12x6", with optional "-nobomb" suffix. */
    static Board board(String geometry, boolean tracksThreats) {
        boolean bombs = !geometry.endsWith("-nobomb");
        int[] size = Arrays.stream(geometry.replace("-nobomb", "").split("x")).mapToInt(Integer::parseInt).toArray();
        return new Board(size[0], size[1], size[2], bombs, tracksThreats);
    }

    /** A plain search engine (QuarkBot's) on a board of the given geometry. */
    static Engine engine(String geometry) {
        return new QuarkBot(board(geometry, false));
    }

    /** The board as an arena field: row 0 at the top. */
    static SpaceOwnership[][] field(Board board) {
        SpaceOwnership[][] field = new SpaceOwnership[board.rows][board.columns];
        for (int column = 0; column < board.columns; column++)
            for (int height = 0; height < board.rows; height++) {
                int owner = board.owner[column * board.rows + height];
                field[board.rows - 1 - height][column] = owner == Board.EMPTY ? SpaceOwnership.EMPTY
                        : owner == 0 ? SpaceOwnership.FIRST_PLAYER : SpaceOwnership.SECOND_PLAYER;
            }
        return field;
    }

    // ------------------------------------------------------------------ actions and scores as text

    /** Actions written like "5,6,5b" (column, b for bomb); "-" or "" is no actions. */
    static int[] parseActions(String text) {
        if (text.isBlank() || text.equals("-")) return new int[0];
        return Arrays.stream(text.split("[,\\s]+"))
                .mapToInt(t -> Board.action(Integer.parseInt(t.replace("b", "")), t.endsWith("b")))
                .toArray();
    }

    static String describeAction(int action) {
        return Board.columnOf(action) + (Board.isBomb(action) ? "b" : "");
    }

    static String describeScore(int score) {
        if (score > Engine.WIN - Engine.MAX_PLY) return "WIN in " + (Engine.WIN - score);
        if (score < Engine.MAX_PLY - Engine.WIN) return "LOSS in " + (Engine.WIN + score);
        return Integer.toString(score);
    }

    /** X = first mover, O = second mover, with column numbers underneath. */
    static String render(Board board) {
        StringBuilder text = new StringBuilder();
        for (int height = board.rows - 1; height >= 0; height--) {
            for (int column = 0; column < board.columns; column++) {
                int owner = board.owner[column * board.rows + height];
                text.append(owner == Board.EMPTY ? " ." : owner == 0 ? " X" : " O");
            }
            text.append('\n');
        }
        for (int column = 0; column < board.columns; column++) text.append(String.format("%2d", column % 10));
        return text.append('\n').toString();
    }

    // ------------------------------------------------------------------ evaluation parameters (for Tune)

    /** The tunable parameters in log space: line weights for 1..lineLength-1 quarks per direction, then the bomb bonus. */
    static double[] parameters(Engine engine) {
        int perDirection = engine.board.lineLength - 1;
        double[] parameters = new double[3 * perDirection + 1];
        for (int d = 0; d < 3; d++)
            for (int n = 1; n <= perDirection; n++) parameters[d * perDirection + n - 1] = Math.log(Math.max(1, engine.board.lineWeight[d][n]));
        parameters[parameters.length - 1] = Math.log(Math.max(1, engine.bombReadyBonus));
        return parameters;
    }

    static void applyParameters(Engine engine, double[] parameters) {
        int perDirection = engine.board.lineLength - 1;
        int[][] weights = new int[3][engine.board.lineLength + 1];
        for (int d = 0; d < 3; d++)
            for (int n = 1; n <= perDirection; n++) weights[d][n] = (int) Math.min(1 << 20, Math.round(Math.exp(parameters[d * perDirection + n - 1])));
        engine.board.setLineWeights(weights);
        engine.bombReadyBonus = (int) Math.min(1 << 20, Math.round(Math.exp(parameters[parameters.length - 1])));
    }

    static String describeWeights(Engine engine) {
        String[] directions = {"horizontal", "vertical", "diagonal"};
        StringBuilder text = new StringBuilder();
        for (int d = 0; d < 3; d++) text.append(directions[d]).append(' ').append(Arrays.toString(engine.board.lineWeight[d])).append("  ");
        return text.append("bombReadyBonus ").append(engine.bombReadyBonus).toString();
    }
}
