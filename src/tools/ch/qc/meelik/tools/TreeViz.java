package ch.qc.meelik.tools;

import ch.qc.meelik.QuarkBot;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Shows how the engine sees a position: the board, a score for every root action (a per-column
 * "heatmap"), the principal variation, and the top of the game tree as a Graphviz file.
 *
 * <p>Usage: TreeViz [moves=-] [plies=2] [evalDepth=6] [width=3] [geom=8x12x6] [out=out/tree.dot]
 * where moves is e.g. "5,6,5b" (b = bomb). Every node is valued by a full-width search of
 * evalDepth from that node; values are from the root player's point of view. All root actions
 * are expanded; deeper levels keep the best {@code width} replies.
 */
public final class TreeViz {
    QuarkBot e;
    int evalDepth, width, rootTurn, ids;
    StringBuilder dot = new StringBuilder();

    public static void main(String[] args) throws Exception {
        TreeViz t = new TreeViz();
        String moves = args.length > 0 ? args[0] : "-";
        int plies = args.length > 1 ? Integer.parseInt(args[1]) : 2;
        t.evalDepth = args.length > 2 ? Integer.parseInt(args[2]) : 6;
        t.width = args.length > 3 ? Integer.parseInt(args[3]) : 3;
        t.e = Tools.engine(args.length > 4 ? args[4] : "8x12x6");
        Path out = Path.of(args.length > 5 ? args[5] : "out/tree.dot");
        for (int a : Tools.parseMoves(moves)) {
            if (!t.e.legal(a)) throw new IllegalArgumentException("illegal action " + Tools.action(a));
            t.e.play(a);
        }
        t.rootTurn = t.e.turn;
        System.out.print(Tools.board(t.e));
        System.out.printf("%s to move, bomb ready: X %s, O %s%n", t.e.turn == 0 ? "X" : "O",
                t.e.wait[0] == 0 && t.e.bombs, t.e.wait[1] == 0 && t.e.bombs);

        t.dot.append("digraph tree {\n  rankdir=LR; node [shape=box, style=\"rounded,filled\", fontname=Helvetica];\n");
        t.dot.append("  n0 [label=\"root\\n").append(moves).append("\", fillcolor=white];\n");
        List<int[]> root = t.expand(0, plies);

        System.out.println("\nroot actions (value for the side to move, depth " + t.evalDepth + "):");
        StringBuilder heat = new StringBuilder("  drop:");
        StringBuilder bomb = new StringBuilder("  bomb:");
        for (int c = 0; c < t.e.cols; c++) {
            heat.append(cellValue(root, 2 * c));
            bomb.append(cellValue(root, 2 * c + 1));
        }
        System.out.println(heat);
        if (t.e.bombs) System.out.println(bomb);
        StringBuilder pv = new StringBuilder();
        int[] s = t.e.snapshot();
        t.e.maxDepth = t.evalDepth;
        for (int d = 0; d < t.evalDepth && t.e.winner() < 0 && t.e.pieces < t.e.size; d++) {
            int a = t.e.think(Long.MAX_VALUE / 4);
            pv.append(Tools.action(a)).append(' ');
            t.e.play(a);
        }
        t.e.restore(s);
        System.out.println("principal variation: " + pv.toString().trim());

        t.dot.append("}\n");
        Files.createDirectories(out.toAbsolutePath().getParent());
        Files.writeString(out, t.dot);
        System.out.println("tree written to " + out + " (" + t.ids + " nodes)");
    }

    static String cellValue(List<int[]> root, int a) {
        for (int[] r : root) if (r[0] == a) return String.format("%8s", Tools.value(r[1]).replace("WIN in ", "W").replace("LOSS in ", "L"));
        return String.format("%8s", "-");
    }

    /** Values every action at node {@code id}, writes them to the graph, recurses into the best ones. */
    List<int[]> expand(int id, int plies) {
        List<int[]> kids = new ArrayList<>(); // {action, value for the root player, node id}
        int[] s = e.snapshot();
        boolean rootMoves = e.turn == rootTurn;
        for (int a = 0; a < 2 * e.cols; a++) {
            if (!e.legal(a)) continue;
            e.play(a);
            int w = e.winner();
            int v = w >= 0 ? (w == rootTurn ? QuarkBot.WIN - 1 : 1 - QuarkBot.WIN)
                    : e.pieces == e.size ? 0 : (e.turn == rootTurn ? 1 : -1) * e.evaluate(evalDepth);
            kids.add(new int[]{a, v, ++ids});
            e.restore(s);
        }
        kids.sort((x, y) -> rootMoves ? y[1] - x[1] : x[1] - y[1]); // best for the mover first
        for (int i = 0; i < kids.size(); i++) {
            int[] k = kids.get(i);
            boolean show = id == 0 || i < width;
            if (!show) continue;
            dot.append(String.format("  n%d [label=\"%s\\n%s\", fillcolor=\"%s\"];%n", k[2], Tools.action(k[0]), Tools.value(k[1]), color(k[1])));
            dot.append(String.format("  n%d -> n%d%s;%n", id, k[2], i == 0 ? " [penwidth=3]" : ""));
            if (plies > 1 && (id == 0 ? i < width : true)) {
                e.play(k[0]);
                if (e.winner() < 0 && e.pieces < e.size) expand(k[2], plies - 1);
                e.restore(s);
            }
        }
        return kids;
    }

    static String color(int v) {
        if (v > QuarkBot.WIN - 1000) return "#7bc47f";
        if (v < 1000 - QuarkBot.WIN) return "#e57373";
        double t = Math.max(-1, Math.min(1, v / 6400.0));
        int r = t < 0 ? 255 : (int) (255 - 80 * t), g = t > 0 ? 255 : (int) (255 + 80 * t);
        return String.format("#%02x%02x%02x", r, g, (int) (255 - 80 * Math.abs(t)));
    }
}
