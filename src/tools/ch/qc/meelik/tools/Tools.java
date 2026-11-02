package ch.qc.meelik.tools;

import ch.cern.quantumconnect.core.SpaceOwnership;
import ch.qc.meelik.QuarkBot;

/** Helpers shared by the tools. Engine player 0 is the first mover here. */
final class Tools {
    private Tools() { }

    /** "8x12x6" (rows x cols x k), optional "-nobomb" suffix. */
    static QuarkBot engine(String geom) {
        String[] p = geom.replace("-nobomb", "").split("x");
        return new QuarkBot(Integer.parseInt(p[0]), Integer.parseInt(p[1]), Integer.parseInt(p[2]), !geom.endsWith("-nobomb"));
    }

    static SpaceOwnership[][] grid(QuarkBot e) {
        SpaceOwnership[][] g = new SpaceOwnership[e.rows][e.cols];
        for (int c = 0; c < e.cols; c++)
            for (int h = 0; h < e.rows; h++) {
                int p = e.cell[c * e.rows + h];
                g[e.rows - 1 - h][c] = p < 0 ? SpaceOwnership.EMPTY : p == 0 ? SpaceOwnership.FIRST_PLAYER : SpaceOwnership.SECOND_PLAYER;
            }
        return g;
    }

    /** Actions like "5,6,5b" (b = bomb); "-" or "" is the empty board. */
    static int[] parseMoves(String s) {
        if (s.isBlank() || s.equals("-")) return new int[0];
        String[] t = s.split("[,\\s]+");
        int[] a = new int[t.length];
        for (int i = 0; i < t.length; i++)
            a[i] = t[i].endsWith("b") ? 2 * Integer.parseInt(t[i].substring(0, t[i].length() - 1)) + 1 : 2 * Integer.parseInt(t[i]);
        return a;
    }

    static String action(int a) {
        return (a >> 1) + ((a & 1) != 0 ? "b" : "");
    }

    /** X = first mover, O = second mover. */
    static String board(QuarkBot e) {
        StringBuilder sb = new StringBuilder();
        for (int h = e.rows - 1; h >= 0; h--) {
            for (int c = 0; c < e.cols; c++) {
                int p = e.cell[c * e.rows + h];
                sb.append(p < 0 ? " ." : p == 0 ? " X" : " O");
            }
            sb.append('\n');
        }
        for (int c = 0; c < e.cols; c++) sb.append(String.format("%2d", c % 10));
        return sb.append('\n').toString();
    }

    /** Evaluation parameters in log space: weight[dir][1..k-1] for the 3 directions, then bombBonus. */
    static double[] params(QuarkBot e) {
        double[] t = new double[3 * (e.k - 1) + 1];
        for (int d = 0; d < 3; d++)
            for (int n = 1; n < e.k; n++) t[d * (e.k - 1) + n - 1] = Math.log(Math.max(1, e.weight[d][n]));
        t[t.length - 1] = Math.log(Math.max(1, e.bombBonus));
        return t;
    }

    static void apply(QuarkBot e, double[] t) {
        for (int d = 0; d < 3; d++)
            for (int n = 1; n < e.k; n++) e.weight[d][n] = (int) Math.min(1 << 20, Math.round(Math.exp(t[d * (e.k - 1) + n - 1])));
        e.bombBonus = (int) Math.min(1 << 20, Math.round(Math.exp(t[t.length - 1])));
    }

    static String describe(QuarkBot e) {
        StringBuilder sb = new StringBuilder();
        String[] names = {"horizontal", "vertical", "diagonal"};
        for (int d = 0; d < 3; d++) sb.append(names[d]).append(' ').append(java.util.Arrays.toString(e.weight[d])).append("  ");
        return sb.append("bombBonus ").append(e.bombBonus).toString();
    }

    static String value(int v) {
        if (v > QuarkBot.WIN - 1000) return "WIN in " + (QuarkBot.WIN - v);
        if (v < 1000 - QuarkBot.WIN) return "LOSS in " + (QuarkBot.WIN + v);
        return Integer.toString(v);
    }
}
