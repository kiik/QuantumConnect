package ch.cern.quantumconnect.core;

import java.util.List;

/** Stand-in for the engine JAR (API as shown in the slides; row 0 is the top). Not shipped. */
public class QuantumField {
    private final SpaceOwnership[][] grid;
    private final SpaceOwnership me;
    private final List<Move> history;

    public QuantumField(SpaceOwnership[][] grid, SpaceOwnership me, List<Move> history) {
        this.grid = grid;
        this.me = me;
        this.history = history;
    }

    public int getNumberOfRows() { return grid.length; }

    public int getNumberOfColumns() { return grid[0].length; }

    public boolean isColumnFull(int col) { return grid[0][col] != SpaceOwnership.EMPTY; }

    public boolean isSpaceEmpty(int row, int col) { return grid[row][col] == SpaceOwnership.EMPTY; }

    public SpaceOwnership[][] getField() {
        SpaceOwnership[][] copy = new SpaceOwnership[grid.length][];
        for (int r = 0; r < grid.length; r++) copy[r] = grid[r].clone();
        return copy;
    }

    public boolean belongsToMe(int row, int col) { return grid[row][col] == me; }

    public List<Move> getMoveHistory() { return List.copyOf(history); }
}
