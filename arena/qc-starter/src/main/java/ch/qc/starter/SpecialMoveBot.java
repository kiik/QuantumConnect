package ch.qc.starter;

import ch.cern.quantumconnect.bots.RandomBot;
import ch.cern.quantumconnect.core.QuantumField;
import java.util.Random;

/**
 * A bot that occasionally uses special moves to destroy surrounding discs.
 * The Experiment class controls how often special moves can be used.
 */
public class SpecialMoveBot extends RandomBot {
  private final Random random;
  private final double specialMoveProbability;

  /**
   * Creates a new SpecialMoveBot with the default special move probability of 0.2 (20%).
   */
  public SpecialMoveBot() {
    this(0.2);
  }

  /**
   * Creates a new SpecialMoveBot with the specified special move probability.
   * @param specialMoveProbability the probability of using a special move (0.0 to 1.0)
   */
  public SpecialMoveBot(double specialMoveProbability) {
    super();
    this.random = new Random();
    this.specialMoveProbability = specialMoveProbability;
  }

  @Override
  public int accelerateQuark(QuantumField quantumField) {
    return super.accelerateQuark(quantumField);
  }

  @Override
  public boolean useSpecialMove(QuantumField quantumField) {
    // Randomly decide whether to use a special move based on the probability
    // The Experiment class will enforce the restriction on how often special moves can be used
    return random.nextDouble() < specialMoveProbability;
  }
}
