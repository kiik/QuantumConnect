# Quantum Connect starter project

This is a ready-to-use project to write, test and package your own Quantum Connect bot. It already
contains the game engine (`libs/core-*.jar`), so you do not need anything else.

## Pre-requisites

- Java 25 (the Gradle toolchain downloads it for you if it is missing and your Gradle is allowed to)
- Nothing else: use the included `./gradlew` (`gradlew.bat` on Windows)

## What is in the project

| Path | What it is |
| ---- | ---------- |
| `src/main/java/ch/qc/starter/RandomBot.java` | A bot that plays a random free column. Start from here. |
| `src/main/java/ch/qc/starter/SpecialMoveBot.java` | A bot that also uses the special move now and then. |
| `src/main/java/EmulatorRunner.java` | Plays two bots against each other on your machine. |
| `libs/core-*.jar` | The game engine (`Algorithm`, `QuantumField`, `Experiment`, ...). |
| `build.gradle` | Build file. This is where you tell the game which class is your bot. |

## Writing your bot

A bot is a class that implements `ch.cern.quantumconnect.core.Algorithm`:

```java
public class MyBot implements Algorithm {

  // Required: return the column (0-based) in which to drop your quark
  @Override
  public int accelerateQuark(QuantumField quantumField) {
    return 0;
  }

  // Optional: return true to use your special move (destroys surrounding discs).
  // The game only allows it once every few moves.
  @Override
  public boolean useSpecialMove(QuantumField quantumField) {
    return false;
  }
}
```

Useful methods of `QuantumField`: `getNumberOfColumns()` and `isColumnFull(int column)`; browse the
rest of the class in your IDE (the jar contains the sources' signatures).

Rules to keep in mind:

- Your bot needs a public constructor without arguments.
- Always return a column that is not full, otherwise you lose the game.
- You only have a limited time per move; a bot that is too slow loses the move.
- Only use the Java standard library and the classes that are in `libs/`.

## Testing your bot locally

1. Create your class, for example `src/main/java/ch/qc/starter/MyBot.java`.
2. Open `src/main/java/EmulatorRunner.java` and put your bot in the experiment:

   ```java
   var exp = new Experiment(new MyBot(), new RandomBot());
   exp.runExperiment();
   ```

   Any `Algorithm` can be your opponent: the `RandomBot` and `SpecialMoveBot` of this project, a
   second version of your own bot, or the bots of `ch.cern.quantumconnect.bots` in the core jar.
3. Run it:

   ```
   ./gradlew run
   ```

   A window opens and shows the match. Run it several times: results vary because of the random
   bots. A good bot should beat `RandomBot` almost every time.

## Packaging and uploading

1. In `build.gradle`, set the fully qualified class name of your bot:

   ```
   'Bot-Class': 'ch.qc.starter.MyBot',
   ```

2. Build the jar:

   ```
   ./gradlew build
   ```

3. Upload `build/libs/quantumconnect-starter-1.0-SNAPSHOT.jar` on the "Bots" page of the game.

If the upload is rejected, check that `Bot-Class` points to a class that exists, implements
`Algorithm`, and has a public no-argument constructor.

## Updating this project

New versions of the game engine are published with a new copy of this starter project. To update an
existing bot, download the starter project again and copy your own sources (`src/main/java/...`) and
your `Bot-Class` line into it, instead of replacing `libs/` by hand.
