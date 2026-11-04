package ch.qc.meelik.tools;


import ch.qc.meelik.bots.QuarkBot;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Shows how the engine sees a position: the board, a score for every possible action (a per-column
 * "heatmap"), the expected line of play, and the top of the game tree as a Graphviz file.
 *
 * <p>Usage: TreeViz [actions=-] [plies=2] [evalDepth=6] [width=3] [geometry=8x12x6] [out=out/tree.dot]
 * where actions is e.g. "5,6,5b" (b = bomb). Every node is scored by a full search of evalDepth
 * from it, from the root player's point of view. All root actions are drawn; deeper levels keep
 * the best {@code width} replies.
 */
public final class TreeViz {
    record Child(int action, int score, int node) { }

    final QuarkBot engine;
    final QuarkBot board;
    final int evalDepth, width, rootPlayer;
    final StringBuilder dot = new StringBuilder();
    int nodeCount;

    TreeViz(QuarkBot engine, int evalDepth, int width) {
        this.engine = engine;
        this.board = engine;
        this.evalDepth = evalDepth;
        this.width = width;
        this.rootPlayer = board.sideToMove;
    }

    public static void main(String[] args) throws Exception {
        String actions = args.length > 0 ? args[0] : "-";
        int plies = args.length > 1 ? Integer.parseInt(args[1]) : 2;
        int evalDepth = args.length > 2 ? Integer.parseInt(args[2]) : 6;
        int width = args.length > 3 ? Integer.parseInt(args[3]) : 3;
        QuarkBot engine = Tools.engine(args.length > 4 ? args[4] : "8x12x6");
        Path output = Path.of(args.length > 5 ? args[5] : "out/tree.dot");

        for (int action : Tools.parseActions(actions)) {
            if (!engine.isLegal(action)) throw new IllegalArgumentException("illegal action " + Tools.describeAction(action));
            engine.play(action);
        }
        TreeViz tree = new TreeViz(engine, evalDepth, width);
        tree.printPosition();
        tree.dot.append("digraph tree {\n  rankdir=LR; node [shape=box, style=\"rounded,filled\", fontname=Helvetica];\n");
        tree.dot.append("  n0 [label=\"root\\n").append(actions).append("\", fillcolor=white];\n");
        List<Child> rootActions = tree.expand(0, plies);
        tree.printHeatmap(rootActions);
        tree.printExpectedLine();
        tree.dot.append("}\n");

        Files.createDirectories(output.toAbsolutePath().getParent());
        Files.writeString(output, tree.dot);
        System.out.println("tree written to " + output + " (" + tree.nodeCount + " nodes)");
    }

    void printPosition() {
        System.out.print(Tools.render(board));
        System.out.printf("%s to move, bomb ready: X %s, O %s%n", board.sideToMove == 0 ? "X" : "O",
                board.bombsEnabled && board.bombCooldown[0] == 0, board.bombsEnabled && board.bombCooldown[1] == 0);
    }

    /** One row of scores per action kind, one column per board column. */
    void printHeatmap(List<Child> rootActions) {
        System.out.println("\nscores of each action for the side to move (search depth " + evalDepth + "):");
        StringBuilder drops = new StringBuilder("  drop:"), bombs = new StringBuilder("  bomb:");
        for (int column = 0; column < board.columns; column++) {
            drops.append(scoreOf(rootActions, QuarkBot.action(column, false)));
            bombs.append(scoreOf(rootActions, QuarkBot.action(column, true)));
        }
        System.out.println(drops);
        if (board.bombsEnabled) System.out.println(bombs);
    }

    static String scoreOf(List<Child> children, int action) {
        for (Child child : children)
            if (child.action() == action) return String.format("%8s", Tools.describeScore(child.score()).replace("WIN in ", "W").replace("LOSS in ", "L"));
        return String.format("%8s", "-");
    }

    void printExpectedLine() {
        int[] start = board.saveState();
        engine.maxDepth = evalDepth;
        StringBuilder line = new StringBuilder();
        for (int ply = 0; ply < evalDepth && board.winner() < 0 && !board.isFull(); ply++) {
            int action = engine.chooseAction(Long.MAX_VALUE / 4);
            line.append(Tools.describeAction(action)).append(' ');
            board.play(action);
        }
        board.restoreState(start);
        System.out.println("expected line of play: " + line.toString().trim());
    }

    /** Scores every action at a node, draws the best ones, and recurses into them. */
    List<Child> expand(int node, int plies) {
        List<Child> children = new ArrayList<>();
        int[] state = board.saveState();
        boolean rootPlayerMoves = board.sideToMove == rootPlayer;
        for (int action = 0; action < board.actionCount(); action++) {
            if (!board.isLegal(action)) continue;
            board.play(action);
            children.add(new Child(action, scoreForRootPlayer(), ++nodeCount));
            board.restoreState(state);
        }
        children.sort((a, b) -> rootPlayerMoves ? b.score() - a.score() : a.score() - b.score()); // best for the mover first

        for (int rank = 0; rank < children.size(); rank++) {
            if (node != 0 && rank >= width) break;
            Child child = children.get(rank);
            dot.append(String.format("  n%d [label=\"%s\\n%s\", fillcolor=\"%s\"];%n", child.node(),
                    Tools.describeAction(child.action()), Tools.describeScore(child.score()), colour(child.score())));
            dot.append(String.format("  n%d -> n%d%s;%n", node, child.node(), rank == 0 ? " [penwidth=3]" : ""));
            if (plies > 1 && rank < width) {
                board.play(child.action());
                if (board.winner() < 0 && !board.isFull()) expand(child.node(), plies - 1);
                board.restoreState(state);
            }
        }
        return children;
    }

    int scoreForRootPlayer() {
        int winner = board.winner();
        if (winner >= 0) return winner == rootPlayer ? QuarkBot.WIN - 1 : 1 - QuarkBot.WIN;
        if (board.isFull()) return 0;
        int score = engine.searchValue(evalDepth); // for the side to move
        return board.sideToMove == rootPlayer ? score : -score;
    }

    /** Green for good for the root player, red for bad, shading by size. */
    static String colour(int score) {
        if (score > QuarkBot.WIN - QuarkBot.MAX_PLY) return "#7bc47f";
        if (score < QuarkBot.MAX_PLY - QuarkBot.WIN) return "#e57373";
        double t = Math.max(-1, Math.min(1, score / 6400.0));
        int red = t < 0 ? 255 : (int) (255 - 80 * t), green = t > 0 ? 255 : (int) (255 + 80 * t), blue = (int) (255 - 80 * Math.abs(t));
        return String.format("#%02x%02x%02x", red, green, blue);
    }
}
