package ch.qc.meelik.tools;

import ch.qc.meelik.bots.SerenaBot;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Openings from the tournament: the first plies of the archived matches on one board size, so that
 * gates and self-play start where real games start. Random drops scatter quarks over a wide board
 * as no real game does: on 15x100 half the matches open in the centre column, and a quarter are two
 * fights far apart.
 *
 * <p>An opening is kept only while the players alternate and only drop (no turn lost to a timeout,
 * no bomb), and each distinct opening once, so that rare openings are drawn as often as common ones.
 * A draw may mirror it, and on a wide board shift it sideways by up to {@link #MAX_SHIFT} columns,
 * which changes nothing but its distance to the edges.
 */
final class Openings {
    static final int MAX_SHIFT = 10;

    final int columns;
    /** The distinct openings, as the column of each ply. */
    final List<int[]> openings;

    private Openings(int columns, List<int[]> openings) {
        this.columns = columns;
        this.openings = openings;
    }

    /** The openings of at least {@code plies} plies (cut to that length) of the archived matches on this board. */
    static Openings fromArchive(int rows, int columns, int plies) throws IOException {
        Set<String> seen = new LinkedHashSet<>();
        List<int[]> openings = new ArrayList<>();
        for (Network.ArchivedGame game : Network.readArchive(Path.of("archive/matches"))) {
            if (game.rows() != rows || game.columns() != columns || game.actions().length <= plies) continue;
            int[] opening = new int[plies];
            boolean clean = true;
            for (int ply = 0; ply < plies && clean; ply++) {
                int action = game.actions()[ply];
                clean = game.players()[ply] == ply % 2 && !SerenaBot.isBomb(action);
                opening[ply] = SerenaBot.columnOf(action);
            }
            if (clean && seen.add(Arrays.toString(opening))) openings.add(opening);
        }
        return new Openings(columns, openings);
    }

    boolean isEmpty() {
        return openings.isEmpty();
    }

    /** One opening, mirrored or not, and shifted sideways on a board wide enough. */
    int[] draw(Random random) {
        int[] opening = openings.get(random.nextInt(openings.size())).clone();
        if (random.nextBoolean()) for (int ply = 0; ply < opening.length; ply++) opening[ply] = columns - 1 - opening[ply];
        if (columns >= 30) {
            int left = columns, right = -1;
            for (int column : opening) {
                left = Math.min(left, column);
                right = Math.max(right, column);
            }
            int lowest = Math.max(-MAX_SHIFT, -left), highest = Math.min(MAX_SHIFT, columns - 1 - right);
            int shift = lowest + random.nextInt(highest - lowest + 1);
            for (int ply = 0; ply < opening.length; ply++) opening[ply] += shift;
        }
        return opening;
    }
}
