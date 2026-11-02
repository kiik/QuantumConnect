import ch.cern.quantumconnect.core.Experiment;
import ch.qc.starter.RandomBot;
import ch.qc.starter.SpecialMoveBot;

void main() throws InterruptedException {
  var exp = new Experiment(new RandomBot(), new SpecialMoveBot());
  exp.runExperiment();
}
