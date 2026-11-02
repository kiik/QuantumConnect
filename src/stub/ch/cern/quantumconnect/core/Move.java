package ch.cern.quantumconnect.core;

/** Stand-in for the engine JAR, with the real Move's signature (core-2.0). Not shipped. */
public record Move(boolean isSpecialMove, int column, SpaceOwnership player) { }
