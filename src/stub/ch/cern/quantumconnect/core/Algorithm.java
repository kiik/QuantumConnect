package ch.cern.quantumconnect.core;

/** Stand-in for the engine JAR (API as shown in the info-session slides). Not shipped. */
public interface Algorithm {
    int accelerateQuark(QuantumField field);

    boolean useSpecialMove(QuantumField field);
}
