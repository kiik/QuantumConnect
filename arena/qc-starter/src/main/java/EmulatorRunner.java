import ch.cern.quantumconnect.core.Experiment;
import ch.qc.starter.RandomBot;
import ch.qc.starter.SpecialMoveBot;

import ch.qc.meelik.bots.QuarkBot;

void main() throws InterruptedException {
  var exp = new Experiment(new RandomBot(), new QuarkBot());
  exp.runExperiment();
}
