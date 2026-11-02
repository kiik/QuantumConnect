package ch.qc.meelik.bots;

import ch.qc.meelik.engine.Board;
import ch.qc.meelik.engine.Engine;

/**
 * The simple solution: the shared alpha-beta engine with hand-written line weights, the same in
 * every direction (16, 64, 256, 1024 and 8192 for one to five quarks in a line). Keep this one
 * uploaded as the safe fallback.
 */
public class QuarkBot extends Engine {
    public QuarkBot() {
        this(new Board());
    }

    public QuarkBot(Board board) {
        super(board);
    }
}
