package ch.qc.meelik.tools;

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/**
 * Pre-upload check of a bot JAR against the format of qc-starter's own upload JAR: Bot-Class in the
 * manifest and in the JAR, the engine library bundled (the starter's JAR bundles its whole runtime
 * classpath), size within 5 MB, the bot a single class (the arena ran the starter's one-class
 * samples but no games for our bot with helper classes), and no references from it to the
 * packages the arena forbids (forbidden access loses the match). The bundled libraries (engine,
 * slf4j, logback) are not scanned. It reads the class files' constant pools for package names, so it can't see through
 * reflection tricks, which are forbidden anyway.
 *
 * <p>Then it loads the JAR on its own, with nothing else on the classpath, as an arena that runs
 * uploaded JARs might: Bot-Class must implement the bundled Algorithm and have a public no-argument
 * constructor (the starter's README names these as the reasons for a rejected upload), and the bot
 * plays one game in the JAR's own Experiment against the engine's RandomBot.
 *
 * <p>Last it prints the bot's size, tracked per revision: {@code size: class=..KB book=.. params=..
 * heap=..MB jar=..KB}. class is the bot's class file, book the entries of its LOSS_BOOK (if any),
 * params the numbers in its static numeric arrays (weights, tables, a network), heap what a bot
 * built for the tournament board (15x30) retains after its first move. Above 60 MB of heap fails:
 * the bot's budget is 80 MB.
 *
 * <p>Usage: Check <jar>...
 */
public final class Check {
    static final String[] FORBIDDEN = {"java/net/", "java/io/", "java/nio/file/", "java/lang/reflect/", "sun/", "com/sun/",
            "java/security/", "javax/crypto/", "java/lang/Thread", "java/util/concurrent/", "java/lang/ProcessBuilder",
            "java/lang/Process", "java/lang/Runtime", "java/lang/invoke/MethodHandles"};
    /** The tournament board, on which the heap is measured, and the most heap a bot may retain there. */
    static final int TOURNAMENT_ROWS = 15, TOURNAMENT_COLUMNS = 30;
    static final long HEAP_LIMIT_BYTES = 60L << 20;
    // java.lang.System was forbidden too until 2026-11-04, when the organisers allowed it; QuarkBot,
    // from before, still keeps time with java.util.Date, SerenaBot with System.nanoTime.
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
                if (ownClasses != 1) issues.add(ownClasses + " class files besides the bundled libraries: a bot must be one self-contained class, like the starter's samples");
                System.out.printf("%s: Bot-Class %s, %d own class file(s) plus the bundled libraries, %d KB%n", path, cls, ownClasses, file.length() / 1024);
            }
            if (issues.isEmpty()) {
                String alone = playAlone(file, cls);
                if (alone.startsWith("PROBLEM")) issues.add(alone.substring(9));
                else System.out.println("  " + alone);
            }
            if (issues.isEmpty()) {
                String size = size(file, cls);
                if (size.startsWith("PROBLEM")) issues.add(size.substring(9));
                else System.out.println("  " + size);
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
            boolean declaresAlgorithm = Arrays.asList(type.getInterfaces()).contains(algorithm);
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
            return String.format("implements Algorithm %s; loaded alone, played its bundled Experiment against RandomBot: %s after %d quarks",
                    declaresAlgorithm ? "itself" : "through its superclass", result, placed);
        } catch (Exception | LinkageError e) {
            return "PROBLEM: can't run on its own: " + e;
        }
    }

    // ------------------------------------------------------------------ size

    /** "size: class=..KB book=.. params=.. heap=..MB jar=..KB", or "PROBLEM: ..." above the heap limit. */
    static String size(java.io.File file, String botClass) {
        try (JarFile jar = new JarFile(file);
             URLClassLoader alone = new URLClassLoader(new URL[]{file.toURI().toURL()}, ClassLoader.getPlatformClassLoader())) {
            long classBytes = jar.getEntry(botClass.replace('.', '/') + ".class").getSize();
            Class<?> type = alone.loadClass(botClass);
            int bookEntries = 0;
            long parameters = 0;
            for (java.lang.reflect.Field field : type.getDeclaredFields()) {
                if (!java.lang.reflect.Modifier.isStatic(field.getModifiers()) || !field.getType().isArray()) continue;
                field.setAccessible(true);
                Object value = field.get(null);
                if (field.getName().equals("LOSS_BOOK"))
                    for (String chunk : (String[]) value) bookEntries += chunk.split(";").length;
                else parameters += numbers(value);
            }
            long heap = heapAfterFirstMove(alone, type);
            String line = String.format("size: class=%dKB book=%d params=%d heap=%dMB jar=%dKB",
                    (classBytes + 1023) / 1024, bookEntries, parameters, (heap + (1 << 19)) >> 20, file.length() / 1024);
            return heap > HEAP_LIMIT_BYTES ? "PROBLEM: " + line + ": heap above " + (HEAP_LIMIT_BYTES >> 20) + " MB" : line;
        } catch (Exception | LinkageError e) {
            return "PROBLEM: can't measure the size: " + e;
        }
    }

    /** The numbers in a numeric array, nested or not; 0 for other arrays. */
    static long numbers(Object array) {
        if (array == null || !array.getClass().isArray()) return 0;
        Class<?> component = array.getClass().getComponentType();
        int length = java.lang.reflect.Array.getLength(array);
        if (component.isPrimitive()) return component == boolean.class || component == char.class ? 0 : length;
        long count = 0;
        for (int i = 0; i < length; i++) count += numbers(java.lang.reflect.Array.get(array, i));
        return count;
    }

    /**
     * Bytes the bot retains after one move on an empty tournament board: built with (rows, columns)
     * if it has that constructor, else with no arguments (it resizes to the field), and kept alive.
     */
    static long heapAfterFirstMove(ClassLoader alone, Class<?> type) throws Exception {
        Class<?> algorithm = alone.loadClass("ch.cern.quantumconnect.core.Algorithm");
        Class<?> fieldClass = alone.loadClass("ch.cern.quantumconnect.core.QuantumField");
        Object opponent = alone.loadClass("ch.cern.quantumconnect.bots.RandomBot").getConstructor().newInstance();
        long before = usedHeap();
        Object bot;
        try {
            bot = type.getConstructor(int.class, int.class).newInstance(TOURNAMENT_ROWS, TOURNAMENT_COLUMNS);
        } catch (NoSuchMethodException none) {
            bot = type.getConstructor().newInstance();
        }
        var fieldConstructor = fieldClass.getDeclaredConstructor(int.class, int.class, algorithm, algorithm);
        fieldConstructor.setAccessible(true);
        Object field = fieldConstructor.newInstance(TOURNAMENT_ROWS, TOURNAMENT_COLUMNS, bot, opponent);
        algorithm.getMethod("accelerateQuark", fieldClass).invoke(bot, field);
        long after = usedHeap();
        java.lang.ref.Reference.reachabilityFence(bot);
        return Math.max(0, after - before);
    }

    static long usedHeap() {
        Runtime runtime = Runtime.getRuntime();
        for (int i = 0; i < 3; i++) System.gc();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    static boolean isBundled(String entry) {
        for (String prefix : BUNDLED) if (entry.startsWith(prefix)) return true;
        return false;
    }
}
