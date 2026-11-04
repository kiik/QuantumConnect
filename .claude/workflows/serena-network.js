export const meta = {
  name: 'serena-network',
  description: 'SerenaBot next steps: size tracking, Tank revision loop, guarded MCP upload, planes+network, island evolution; each reviewed',
  whenToUse: 'Run from the QuantumConnect repo root after committing; each task works in its own worktree and commits on its own branch',
  phases: [
    { title: 'Build', detail: 'five tasks in parallel, one git worktree each' },
    { title: 'Review', detail: 'one independent reviewer per task branch' },
    { title: 'Explain', detail: 'methodology document with diagrams and charts' },
  ],
}

// Rules every task shares, on top of CLAUDE.md (which agents get anyway).
const SHARED = `
Context: QuantumConnect tournament (https://quantum-connect.web.cern.ch/), our team is WilyQuarks.
The Java side lives in arena/ (run ./qc.sh from there); the site CLI/MCP in tools/qc-site (uv run qcsite ...).
The tournament board is 15x30 (six in a row, 3x3 bomb, 100 ms per move). Our bot is SerenaBot
(arena/qc-starter/src/main/java/ch/qc/meelik/bots/SerenaBot.java): QuarkBot's search plus a loss book
learned from lost games (./qc.sh learn, arena/params/serena/book.txt). Matches are archived in arena/archive/.

Hard rules:
- Bot source is visible to the tournament organisers: keep it concise, textbook-style (class doc explaining
  the concepts, labelled sections, descriptive names, short comments) and annotated "@author WilyQuarks".
- Bot code: one class, java.lang and java.util only, no lambdas/nested/anonymous classes, and no string "+"
  concatenation (javac compiles it to java.lang.invoke, which the arena forbids; use String.concat or a
  StringBuilder). ./qc.sh check qc-starter/build/libs/<Bot>.jar must print "ok".
- Never print, log or store the credentials in .env (USER_NAME, USER_PSK). Show only key names.
- Never upload to the live site and never call the upload endpoint during this task; test upload code with
  a dry run or a fake.
- Memory budget: the bot must stay under 80 MB of heap; grow any network sparingly.
- Keep any training or game run under 10 minutes; say what a longer run would need instead of running it.
- We must be able to explain and illustrate the methodology: every training, evolution or gate run you add
  appends its numbers to a TSV under arena/results/ (one row per generation/epoch/revision, with a header and
  a '#' line saying what produced it), so the methodology document can chart it. Name the files in your report.
- When done: ./qc.sh build && ./qc.sh test pass, then commit on your worktree branch with a plain message
  (no Co-Authored-By or other AI attribution lines). Return the branch name, the commit, what you did,
  what you verified (with numbers) and what you did not do.`

const TASKS = [
  {
    key: 'size',
    prompt: `Task: size and memory tracking for every SerenaBot revision.
Extend Check (arena/qc-trainer/.../tools/Check.java) so ./qc.sh check <jar> also reports: our class's KB, the
loss-book entry count, network parameter count if the class has one (0 now), an estimated heap of a fresh
instance at 15x30 (measure: allocate one with new SerenaBot(15, 30) and one accelerateQuark, Runtime before/after,
GC'd; or count the arrays it allocates) and the JAR KB. Fail above 60 MB estimated heap (headroom under 80 MB).
Print one machine-readable summary line too (e.g. "size: class=..KB book=.. params=.. heap=..MB jar=..KB"),
which the revision loop will append to arena/params/serena/revisions.tsv.`,
  },
  {
    key: 'tank-loop',
    prompt: `Task: the revision loop, as ./qc.sh tank (in arena/qc.sh, documented in its usage header and README).
Revisions are uploaded under the codename Tank-DD-HHMM (date-time, e.g. Tank-04-1835); older ones were
QuarkBot-* and SerenaBot-*. Steps, stopping at the first failure:
1. (cd ../tools/qc-site && uv run qcsite archive) - saves new matches of ours + top 3 (must precede any upload,
   which deletes our matches);
2. archive-losses for losers 'Tank-*' 'SerenaBot-*' 'QuarkBot-*' plus the current top-3 bot names
   (read them from qcsite --json leaderboard) -> out/serena/losses.txt;
3. ./qc.sh learn out/serena/losses.txt (default budget) -> book;
4. ./qc.sh build, test, check SerenaBot.jar;
5. a local gate: ./qc.sh stats on 15x30 with the candidate against the last uploaded Tank JAR in Bots/ (add
   "serena serena" to league.txt if missing; a small league file just for the gate is fine) and report the score;
6. copy the candidate JAR to Bots/Tank-DD-HHMM.jar (the local snapshot of exactly what would be uploaded),
   append a line to arena/params/serena/revisions.tsv (name, git rev, book entries, gate score, sizes),
   and PRINT the upload command (uv run qcsite upload Tank-.. ../../arena/Bots/Tank-...jar --yes && uv run
   qcsite wait Tank-..) without running it.
Also make qcsite archive-losses' default loser patterns include 'Tank-*'. Test steps 1-4 and 6 for real
(archive and archive-losses only read the site); keep the gate run short (few games).`,
  },
  {
    key: 'mcp-upload',
    prompt: `Task: model upload through the MCP server, plus an opt-in autopilot that re-uploads when we fall
on the leaderboard (tools/qc-site/qcsite/: client.py has upload_bot, cli.py the upload/wait commands,
mcp_server.py is read-only today).
- Add an MCP tool upload_bot(jar_path, name, confirm) that refuses unless: name matches Tank-DD-HHMM
  (5..20 chars), the JAR passes ./qc.sh check (run it), confirm == name, and the archive was refreshed in this
  call first (run the archive step before uploading). Keep a local log arena/archive/uploads.jsonl
  (name, jar sha256, time, our rank before). Copy the JAR to arena/Bots/<name>.jar as the snapshot.
- Add "qcsite autopilot" (CLI, and an MCP tool that only reports what it would do): it reads the leaderboard,
  and if our rank got worse than at the last upload (or we are not first) AND a candidate JAR exists whose
  local gate result (arena/params/serena/revisions.tsv, written by ./qc.sh tank) beats the deployed Tank by a
  clear margin (default >= 60% over >= 20 games), it proposes the upload. It uploads only with --yes or an
  environment opt-in (QC_AUTOPILOT=1), at most once per 30 minutes, and never while the last upload's games
  are still incomplete. Default is a dry run that prints the decision and why.
- If revisions.tsv does not exist yet, define its columns in one place and document them; the tank-loop task
  writes it in parallel, so agree on: name, git_rev, book_entries, gate_games, gate_score, class_kb, jar_kb, heap_mb.
Test with the upload endpoint faked (monkeypatch client.Site.upload or httpx); never call the real one.`,
  },
  {
    key: 'network',
    prompt: `Task: the input planes and a small integer evaluation network for SerenaBot, switched off by default.
1. Data: a qc-trainer tool (./qc.sh dataset) that replays archived matches (arena/archive/matches/*.json,
   moves like "f15 s14 f14b"; replay on SerenaBot as Learn does) and writes positions as integer planes:
   own quarks, opponent quarks, playable top cells, own threats, opponent threats (from threatCount),
   plus bomb-cooldown buckets and side to move; labels = game result from the mover's side and, optionally,
   a short search score. Output a compact binary or .npz-ready format under arena/out/dataset/.
2. Training: a small Python trainer under tools/ (uv, numpy; torch only if it stays light) for an NNUE-style net:
   sparse inputs -> int16 accumulator (32-64 units) -> clipped ReLU -> 1 output, quantised to ints.
3. Bot: SerenaBot gains the accumulator, updated incrementally in addQuark/removeTopQuark like lineQuarks,
   weights embedded as base64 String chunks (same pattern as LOSS_BOOK, with BEGIN/END markers), blended with or
   replacing the line evaluation behind a static flag that stays false until a gate shows a gain.
   Must keep nodes/s within 30% of today's (./qc.sh test prints it) and pass ./qc.sh check and conformance.
Report parameter count, embedded KB, nodes/s with and without the net, and a short local match result.`,
  },
  {
    key: 'islands',
    prompt: `Task: island-model evolution in Tune (arena/qc-trainer/.../tools/Tune.java).
Today Tune runs one evolution strategy with mirrored pairs on QuarkBot's line weights. Add --islands N (4-8)
and --migrate K: N independent populations from different seeds evolve in parallel; every K generations each
island's best is copied to the others (ring or all-to-all), so islands stuck at a local optimum learn from the
rest. Make it work with SerenaBot as the bot (line weights, bomb-ready bonus, and later a network's scale knobs;
SerenaBot's weights are static final today, so add the narrowest hook the tool needs without bloating the bot)
and play fitness games on 15x30 at the real time limit (a fixed-depth option may stay for speed). Add a size
penalty hook for structural parameters (e.g. hidden units) so growth must pay for itself. Run a short smoke test
(e.g. 2 islands x 2 generations on a small board) and report it; describe the full run's command and duration.`,
  },
]

const RESULT = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    commit: { type: 'string' },
    worktree: { type: 'string', description: 'absolute path of the worktree' },
    summary: { type: 'string' },
    verified: { type: 'string' },
    not_done: { type: 'string' },
  },
  required: ['branch', 'commit', 'worktree', 'summary', 'verified', 'not_done'],
}

const REVIEW = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['merge', 'fix-first', 'reject'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string' }, line: { type: 'integer' },
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          problem: { type: 'string' }, fix: { type: 'string' },
        },
        required: ['file', 'severity', 'problem', 'fix'],
      },
    },
    checks_run: { type: 'string' },
  },
  required: ['verdict', 'findings', 'checks_run'],
}

const results = await pipeline(
  TASKS,
  task => agent(`${task.prompt}\n${SHARED}`, {
    label: `build:${task.key}`, phase: 'Build', isolation: 'worktree', schema: RESULT,
  }),
  (built, task) => built && agent(
    `Review the branch ${built.branch} (commit ${built.commit}, worktree ${built.worktree}) of task "${task.key}".
The task was:\n${task.prompt}\n${SHARED}
Read git diff main...${built.branch} in that worktree. Run, in its arena/: ./qc.sh build, ./qc.sh test, and
./qc.sh check qc-starter/build/libs/SerenaBot.jar if the bot changed (./qc.sh conformance too if its rules or
board code changed). Look for: anything the arena forbids in bot code (including string "+" concatenation),
credential leaks, any path that could upload without the stated guards, behaviour changes to the search that the
task did not ask for, broken position hashing (the loss book depends on positionKey being unchanged), heap over
budget, and code that is not concise/textbook or lacks the WilyQuarks annotation. Do not modify the branch.`,
    { label: `review:${task.key}`, phase: 'Review', schema: REVIEW },
  ).then(review => ({ task: task.key, built, review })),
)

// Barrier on purpose: the methodology document needs every branch's result and review together.
const done = results.filter(Boolean)
log(`${done.length}/${TASKS.length} tasks built and reviewed; writing the methodology`)

const methodology = await agent(
  `Write docs/methodology.md: how SerenaBot (team WilyQuarks) learns and how its network is evolved, so the team
can explain and illustrate it to the tournament organisers and others. Audience: technically literate readers
who don't know our code. Cover, each with a figure:
1. the game model and search (line counts, threats, alpha-beta) - one diagram;
2. the learning loop: tournament -> archive -> lost games -> deep offline search -> loss book -> Tank revision
   -> local gate -> upload - one flow diagram;
3. the input planes - a diagram of a 15x30 position split into its planes;
4. the network: sparse planes -> int16 accumulator updated per quark -> hidden -> score, with sizes and memory;
5. island evolution: islands, migration, size penalty, fitness at the real time limit - a diagram, and charts
   from the run logs (fitness per generation per island, network size per revision, gate scores);
6. budgets: 80 MB heap, JAR size, per-revision tracking.
Figures: hand-written SVG files under docs/figures/ (readable in light and dark) for diagrams; charts generated
by a small reproducible script (tools/ or arena/, Python via uv) from the TSVs under arena/results/, also as SVG.
Where a run log is empty or a task was not done, say so in the text instead of inventing numbers.
The task branches, their worktrees, summaries and reviews:\n${JSON.stringify(done, null, 1)}
Work on a new branch "methodology" from main in your own worktree; read the task branches' files from their
worktrees (git show <branch>:<path>) without merging them. Commit with a plain message (no AI attribution).
Return the branch, the commit, and the list of figures.`,
  { label: 'explain:methodology', phase: 'Explain', isolation: 'worktree' },
)

return { tasks: done, methodology }
