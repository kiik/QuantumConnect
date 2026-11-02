package ch.qc.meelik.bots;

import ch.qc.meelik.engine.Board;
import ch.qc.meelik.engine.Engine;

/**
 * The evolved solution: QuarkBot's engine with line weights evolved by self-play
 * ({@code ./qc.sh tune}; the run is recorded in params/evolved-d4.txt). Against QuarkBot at the
 * real time limit it scored 14 wins, 24 draws and 2 losses.
 *
 * <p>Compared with the hand-written weights, single quarks and pairs in horizontal lines count for
 * far more, four-in-a-line counts for less, and having the bomb ready is worth a bonus.
 */
public class EvolvedBot extends Engine {
    static final int[] HORIZONTAL = {0, 41, 131, 241, 540, 6367, 0};
    static final int[] VERTICAL = {0, 17, 75, 204, 658, 8300, 0};
    static final int[] DIAGONAL = {0, 15, 105, 261, 615, 7863, 0};
    static final int BOMB_READY_BONUS = 20;

    public EvolvedBot() {
        this(new Board());
    }

    public EvolvedBot(Board board) {
        super(board);
        board.setLineWeights(new int[][]{HORIZONTAL, VERTICAL, DIAGONAL});
        bombReadyBonus = BOMB_READY_BONUS;
    }
}
