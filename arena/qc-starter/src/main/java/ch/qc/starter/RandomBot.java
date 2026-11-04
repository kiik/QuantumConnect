package ch.qc.starter;

import ch.cern.quantumconnect.core.Algorithm;
import ch.cern.quantumconnect.core.QuantumField;
import java.util.Random;


public class RandomBot implements Algorithm {
  private final Random random;

  public RandomBot() {
    this.random = new Random();
  }

  @Override
  public int accelerateQuark(QuantumField quantumField) {
    int columns = quantumField.getNumberOfColumns();
    int[] validColumns = new int[columns];
    int validCount = 0;

    // Collect all non-full columns
    for (int col = 0; col < columns; col++) {
      if (!quantumField.isColumnFull(col)) {
        validColumns[validCount++] = col;
      }
    }

    // Pick a random column from the valid ones
    if (validCount > 0) {
      return validColumns[random.nextInt(validCount)];
    }

    throw new IllegalStateException("No valid moves available");
  }
}