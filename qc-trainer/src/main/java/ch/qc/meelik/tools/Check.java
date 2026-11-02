package ch.qc.meelik.tools;

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/**
 * Pre-upload check of a bot JAR against the format of qc-starter's own upload JAR: Bot-Class in the
 * manifest and in the JAR, the engine library bundled (the starter's JAR bundles its whole runtime
 * classpath), size within 5 MB, and no references from our own classes to the packages the arena
 * forbids (forbidden access loses the match). The bundled libraries (engine, slf4j, logback) are
 * not scanned. It reads the class files' constant pools for package names, so it can't see through
 * reflection tricks, which are forbidden anyway.
 *
 * <p>Then it loads the JAR on its own, with nothing else on the classpath, as an arena that runs
 * uploaded JARs might: Bot-Class must implement the bundled Algorithm and have a public no-argument
 * constructor (the starter's README names these as the reasons for a rejected upload), and the bot
 * plays one game in the JAR's own Experiment against the engine's RandomBot.
 *
 * <p>Usage: Check <jar>...
 */
public final class Check {
    static final String[] FORBIDDEN = {"java/net/", "java/io/", "java/nio/file/", "java/lang/reflect/", "sun/", "com/sun/",
            "java/security/", "javax/crypto/", "java/lang/Thread", "java/util/concurrent/", "java/lang/ProcessBuilder",
            "java/lang/Process", "java/lang/Runtime", "java/lang/invoke/MethodHandles"};
    /** Libraries the starter's build bundles into every upload JAR. */
    static final String[] BUNDLED = {"ch/cern/quantumconnect/", "org/slf4j/", "ch/qos/logback/", "META-INF/", "module-info.class"};

    public static void main(String[] args) throws Exception {
        int problems = 0;
        for (String path : args) {
            List<String> issues = new ArrayList<>();
            java.io.File file = new java.io.File(path);
            if (file.length() > 5L << 20) issues.add("larger than 5 MB");
            String cls;
            try (JarFile jar = new JarFile(file)) {
                cls = jar.getManifest() == null ? null : jar.getManifest().getMainAttributes().getValue("Bot-Class");
                if (cls == null) issues.add("no Bot-Class in the manifest");
                else if (jar.getEntry(cls.replace('.', '/') + ".class") == null) issues.add("Bot-Class " + cls + " is not in the JAR");
                if (jar.getEntry("ch/cern/quantumconnect/core/Algorithm.class") == null) issues.add("the engine library is not bundled, as qc-starter's JAR does (build with ./qc.sh build)");
                int ownClasses = 0;
                for (JarEntry e : java.util.Collections.list(jar.entries())) {
                    if (!e.getName().endsWith(".class") || isBundled(e.getName())) continue;
                    ownClasses++;
                    String pool = new String(jar.getInputStream(e).readAllBytes(), StandardCharsets.ISO_8859_1);
                    for (String f : FORBIDDEN) if (pool.contains(f)) issues.add(e.getName() + " references " + f.replace('/', '.'));
                }
                System.out.printf("%s: Bot-Class %s, %d own class file(s) plus the bundled libraries, %d KB%n", path, cls, ownClasses, file.length() / 1024);
            }
            if (issues.isEmpty()) {
                String alone = playAlone(file, cls);
                if (alone.startsWith("PROBLEM")) issues.add(alone.substring(9));
                else System.out.println("  " + alone);
            }
            for (String i : issues) System.out.println("  PROBLEM: " + i);
            if (issues.isEmpty()) System.out.println("  ok");
            problems += issues.size();
        }
        if (problems > 0) System.exit(1);
    }

    /** Loads the JAR alone and plays one game in its bundled Experiment; a summary, or "PROBLEM: ...". */
    static String playAlone(java.io.File file, String botClass) {
        try (URLClassLoader alone = new URLClassLoader(new URL[]{file.toURI().toURL()}, ClassLoader.getPlatformClassLoader())) {
            Class<?> algorithm = alone.loadClass("ch.cern.quantumconnect.core.Algorithm");
            Class<?> type = alone.loadClass(botClass);
            if (!algorithm.isAssignableFrom(type)) return "PROBLEM: Bot-Class " + botClass + " doesn't implement Algorithm";
            Object bot;
            try {
                bot = type.getConstructor().newInstance();
            } catch (NoSuchMethodException none) {
                return "PROBLEM: Bot-Class " + botClass + " has no public no-argument constructor";
            }
            Object opponent = alone.loadClass("ch.cern.quantumconnect.bots.RandomBot").getConstructor().newInstance();
            Class<?> experimentClass = alone.loadClass("ch.cern.quantumconnect.core.Experiment");
            Object experiment = experimentClass.getConstructor(algorithm, algorithm, boolean.class, boolean.class).newInstance(bot, opponent, false, false);
            experimentClass.getMethod("runExperiment").invoke(experiment);
            String status = experimentClass.getMethod("getExperimentStatus").invoke(experiment).toString();
            String lastMover = experimentClass.getMethod("getCurrentPlayer").invoke(experiment).toString();
            int placed = ((List<?>) experimentClass.getMethod("getHistory").invoke(experiment)).size();
            String result = status.equals("COLLAPSED") ? (lastMover.equals("FIRST_PLAYER") ? "the bot wins" : "RandomBot wins") : "draw";
            return String.format("loaded alone, played its bundled Experiment against RandomBot: %s after %d quarks", result, placed);
        } catch (Exception | LinkageError e) {
            return "PROBLEM: can't run on its own: " + e;
        }
    }

    static boolean isBundled(String entry) {
        for (String prefix : BUNDLED) if (entry.startsWith(prefix)) return true;
        return false;
    }
}
