package ch.qc.meelik.bots;

import java.util.HashMap;
import java.util.Map;

/**
 * A sparring partner for SerenaBot, never uploaded: SerenaBot's board model and evaluation, with
 * the search refinements of game engines and without the arena's time limit, so that it can show
 * SerenaBot what a stronger player does. It lives with the tools, so the one-class rules of an
 * upload don't bind it.
 *
 * <p>The concepts:
 * <ul>
 *   <li><b>Threat-space search</b> (as gomoku solvers use it, "victory by continuous forcing"): the
 *       attacker plays only moves that make a threat to complete a line at once, a drop or a bomb;
 *       the defender answers each with every defence it has: the block, and with its bomb ready every
 *       blast. The attacker wins when it completes a line, or faces a defender without its bomb with
 *       two threats. Proven at the root, the bot plays the first move of the proof.
 *   <li><b>Principal variation search:</b> after the first move of a node, the others are searched
 *       with a null window, which only asks "is it better?", and re-searched only when it is.
 *   <li><b>Late move reductions:</b> quiet drops late in the move order are first searched two plies
 *       shallower, and in full only when they look better than the best so far.
 * </ul>
 */
public class ThreatBot extends SerenaBot {
    /** How many plies, attacker and defender together, the threat-space search may look ahead. */
    public int threatPlies = 21;
    /** Positions the threat-space search may visit per move, so a long proof doesn't stall the game. */
    public long threatNodeLimit = 3_000_000L;
    /** Search refinements: principal variation search with late move reductions. */
    public boolean refinedSearch = true;
    /** Moves won by the threat-space search, for the reports. */
    public int threatWins;

    long threatNodes;
    /** The length of the last proof found, in plies. */
    int proofPlies;
    /** Positions where the attacker has no win in the given number of plies or fewer. */
    final Map<Long, Integer> noWinWithin = new HashMap<>();

    public ThreatBot() {
        super();
    }

    public ThreatBot(int rows, int columns) {
        super(rows, columns);
    }

    // ------------------------------------------------------------------ choosing an action

    @Override
    public int chooseAction(long budget) {
        long started = System.nanoTime();
        int win = threatSpaceWin();
        if (win != NO_ACTION) {
            threatWins++;
            lastScore = WIN - proofPlies; // the proof's length bounds the win's
            depthReached = proofPlies;
            return win;
        }
        return super.chooseAction(Math.max(1_000_000L, budget - (System.nanoTime() - started)));
    }

    /** The first move of a proven threat-space win for the side to move, or NO_ACTION. */
    public int threatSpaceWin() {
        threatNodes = 0;
        noWinWithin.clear();
        int attacker = sideToMove;
        for (int depth = 1; depth <= threatPlies; depth += 2) { // shortest proofs first
            int[] actions = attackingActions(attacker);
            for (int action : actions) {
                if (action == NO_ACTION) break;
                if (attackerWinsAfter(action, attacker, depth)) {
                    proofPlies = depth;
                    return action;
                }
                if (threatNodes > threatNodeLimit) return NO_ACTION;
            }
        }
        return NO_ACTION;
    }

    // ------------------------------------------------------------------ the threat-space search

    /**
     * The attacking moves: a drop that completes a line or makes a threat, and with the bomb ready, a
     * blast that hits a quark (the search then checks it makes a line or a threat).
     */
    int[] attackingActions(int attacker) {
        int[] actions = new int[2 * columns + 1];
        int count = 0;
        for (int column : columnsCentreOut) if (completesLine(attacker, column)) actions[count++] = action(column, false);
        for (int column : columnsCentreOut) if (!completesLine(attacker, column) && makesThreat(attacker, column)) actions[count++] = action(column, false);
        if (bombCooldown[attacker] == 0) for (int column : columnsCentreOut) if (columnHeight[column] < rows && blastHitsQuark(column)) actions[count++] = action(column, true);
        actions[count] = NO_ACTION;
        return actions;
    }

    /** Plays the attacker's action and asks whether every defence still loses within {@code plies}. */
    boolean attackerWinsAfter(int action, int attacker, int plies) {
        threatNodes++;
        int[] saved = new int[stateSize];
        saveState(saved);
        play(action);
        boolean wins;
        if (hasLine()) wins = true; // any line after a turn wins for its mover
        else if (playableThreats(attacker) == 0) wins = false; // not forcing
        else wins = everyDefenceLoses(attacker, plies - 1);
        restoreState(saved);
        return wins;
    }

    /** The defender is to move and faces at least one playable threat. */
    boolean everyDefenceLoses(int attacker, int plies) {
        int defender = 1 - attacker;
        for (int column = 0; column < columns; column++) if (completesLine(defender, column)) return false;
        boolean mayBomb = bombCooldown[defender] == 0;
        int threats = playableThreats(attacker);
        if (threats >= 2 && !mayBomb) return true;
        if (plies <= 0) return false;
        Long key = positionKey();
        Integer known = noWinWithin.get(key);
        if (known != null && known >= plies) return false;

        boolean allLose = true;
        if (threats == 1) {
            int block = NO_ACTION;
            for (int column = 0; column < columns; column++) if (completesLine(attacker, column)) block = column;
            allLose = attackerWinsAfterDefence(action(block, false), attacker, plies);
        }
        if (allLose && mayBomb) {
            for (int column : columnsCentreOut) {
                if (columnHeight[column] == rows || !blastHitsQuark(column)) continue;
                if (!attackerWinsAfterDefence(action(column, true), attacker, plies)) {
                    allLose = false;
                    break;
                }
                if (threatNodes > threatNodeLimit) {
                    allLose = false;
                    break;
                }
            }
        }
        if (!allLose) noWinWithin.put(key, plies);
        return allLose;
    }

    /** Plays a defence and asks whether the attacker still has a winning continuation. */
    boolean attackerWinsAfterDefence(int defence, int attacker, int plies) {
        threatNodes++;
        int[] saved = new int[stateSize];
        saveState(saved);
        play(defence);
        boolean wins = false;
        if (!hasLine()) {
            int[] actions = attackingActions(attacker);
            for (int action : actions) {
                if (action == NO_ACTION || threatNodes > threatNodeLimit) break;
                if (attackerWinsAfter(action, attacker, plies - 1)) {
                    wins = true;
                    break;
                }
            }
        }
        restoreState(saved);
        return wins;
    }

    int playableThreats(int player) {
        int threats = 0;
        for (int column = 0; column < columns; column++) if (completesLine(player, column)) threats++;
        return threats;
    }

    // ------------------------------------------------------------------ the refined search

    /** SerenaBot's negamax, with principal variation search and late move reductions. */
    @Override
    int negamax(int depth, int alpha, int beta, int ply) {
        if (!refinedSearch) return super.negamax(depth, alpha, beta, ply);
        if ((++nodesSearched & (CLOCK_CHECK_NODES - 1)) == 0 && System.nanoTime() > deadlineNanos) outOfTime = true;
        if (outOfTime) return 0;

        int mover = sideToMove;
        if (hasLine()) return ply - WIN;
        if (quarkCount == cellCount) return 0;

        int forced = NOT_FORCED;
        if (ply > 0) {
            int opponentThreats = 0;
            for (int column = 0; column < columns; column++) {
                if (completesLine(mover, column)) return WIN - ply - 1;
                if (completesLine(1 - mover, column)) {
                    opponentThreats++;
                    forced = column;
                }
            }
            boolean mayBomb = bombCooldown[mover] == 0;
            if (opponentThreats >= 2 && !mayBomb) return ply + 2 - WIN;
            if (opponentThreats >= 2) forced = BOMBS_ONLY;
            if (opponentThreats == 1 && !mayBomb && depth == 0) depth = 1;
        }
        if (ply == MAX_PLY - 1) return evaluate();
        if (depth <= 0) return forcingLine(depth, alpha, beta, ply, forced);

        long key = positionKey();
        int slot = tableSlot(key), tableAction = NO_ACTION;
        if (tableKey[slot] == key) {
            long entry = tableEntry[slot];
            tableAction = entryAction(entry);
            if (entryDepth(entry) >= depth) {
                int score = scoreFromTable(entryScore(entry), ply), bound = entryBound(entry);
                if (bound == EXACT || bound == LOWER_BOUND && score >= beta || bound == UPPER_BOUND && score <= alpha) return score;
            }
        }

        int[] actions = actionLists[ply];
        int actionCount = orderedLegalActions(actions, tableAction, forced);
        int originalAlpha = alpha, bestScore = -WIN - 1, bestAction = NO_ACTION;
        for (int i = 0; i < actionCount; i++) {
            int action = actions[i], score;
            boolean bomb = isBomb(action);
            int column = columnOf(action), previousCooldown = 0;
            if (bomb) {
                if (undoStates[ply] == null) undoStates[ply] = new int[stateSize];
                saveState(undoStates[ply]);
                play(action);
            } else previousCooldown = drop(column);

            if (i == 0) score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            else {
                boolean reduce = i >= 3 && depth >= 3 && !bomb && forced == NOT_FORCED && action != tableAction;
                score = -negamax(depth - 1 - (reduce ? 2 : 0), -alpha - 1, -alpha, ply + 1);
                if (score > alpha && reduce) score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1);
                if (score > alpha && score < beta) score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            }

            if (bomb) restoreState(undoStates[ply]);
            else undoDrop(column, previousCooldown);
            if (outOfTime) return 0;
            if (score > bestScore) {
                bestScore = score;
                bestAction = action;
            }
            if (score > alpha) alpha = score;
            if (alpha >= beta) {
                historyScore[mover][action] += depth * depth;
                break;
            }
        }

        int bound = bestScore <= originalAlpha ? UPPER_BOUND : bestScore >= beta ? LOWER_BOUND : EXACT;
        tableKey[slot] = key;
        tableEntry[slot] = packEntry(scoreToTable(bestScore, ply), depth, bound, bestAction);
        return bestScore;
    }
}
