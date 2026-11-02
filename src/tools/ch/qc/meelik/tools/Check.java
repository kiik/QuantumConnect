package ch.qc.meelik.tools;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/**
 * Pre-upload check of a bot JAR: manifest Bot-Class present, size within 5 MB, one bot class, and
 * no references to the packages the arena forbids (forbidden access loses the match). It scans
 * the class files' constant pools for package names, so it can't see through reflection tricks,
 * which are forbidden anyway.
 *
 * <p>Usage: Check <jar>...
 */
public final class Check {
    static final String[] FORBIDDEN = {"java/net/", "java/io/", "java/nio/file/", "java/lang/reflect/", "sun/", "com/sun/",
            "java/security/", "javax/crypto/", "java/lang/Thread", "java/util/concurrent/", "java/lang/ProcessBuilder",
            "java/lang/Process", "java/lang/Runtime", "java/lang/invoke/MethodHandles"};

    public static void main(String[] args) throws Exception {
        int problems = 0;
        for (String path : args) {
            List<String> issues = new ArrayList<>();
            java.io.File file = new java.io.File(path);
            if (file.length() > 5L << 20) issues.add("larger than 5 MB");
            try (JarFile jar = new JarFile(file)) {
                String cls = jar.getManifest() == null ? null : jar.getManifest().getMainAttributes().getValue("Bot-Class");
                if (cls == null) issues.add("no Bot-Class in the manifest");
                else if (jar.getEntry(cls.replace('.', '/') + ".class") == null) issues.add("Bot-Class " + cls + " is not in the JAR");
                int classes = 0;
                for (JarEntry e : java.util.Collections.list(jar.entries())) {
                    if (!e.getName().endsWith(".class")) continue;
                    classes++;
                    String pool = new String(jar.getInputStream(e).readAllBytes(), StandardCharsets.ISO_8859_1);
                    for (String f : FORBIDDEN) if (pool.contains(f)) issues.add(e.getName() + " references " + f.replace('/', '.'));
                    if (pool.contains("ch/cern/quantumconnect/core/") && e.getName().startsWith("ch/cern/")) issues.add("bundles the engine classes: " + e.getName());
                }
                System.out.printf("%s: Bot-Class %s, %d class file(s), %d KB%n", path, cls, classes, file.length() / 1024);
            }
            for (String i : issues) System.out.println("  PROBLEM: " + i);
            if (issues.isEmpty()) System.out.println("  ok");
            problems += issues.size();
        }
        if (problems > 0) System.exit(1);
    }
}
