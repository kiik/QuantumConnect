export const meta = {
  name: 'serena-network',
  description: 'SerenaBot next steps: Tank loop with size tracking, guarded MCP upload, planes+network, island evolution, methodology doc',
  whenToUse: 'Run from the QuantumConnect repo root after committing; each task commits on its own worktree branch',
  phases: [
    { title: 'Build', detail: 'four tasks in parallel, one git worktree each' },
    { title: 'Explain', detail: 'methodology document with diagrams and charts' },
  ],
}

// Rules every task shares, on top of CLAUDE.md.
const SHARED = `
Context: QuantumConnect tournament, team WilyQuarks. Java side in arena/ (./qc.sh), site CLI/MCP in tools/qc-site
(uv run qcsite). Tournament board 15x30. Our bot: arena/qc-starter/src/main/java/ch/qc/meelik/bots/SerenaBot.java
(search + loss book from ./qc.sh learn). Matches archived in arena/archive/.
Rules: bot source is read by the organisers - concise, textbook, "@author WilyQuarks"; no string "+" in bot code
(use String.concat); ./qc.sh check must print "ok". Never print .env credentials. Never call the real upload
endpoint. Bot heap < 80 MB. Runs < 10 min. Any training/evolution/gate run appends rows to a TSV under
arena/results/ (header + '#' provenance line) for the methodology charts. Finish with ./qc.sh build && ./qc.sh test,
commit on your branch (plain message, no AI attribution). Be brief in the report.`

const TASKS = [
  {
    key: 'tank-loop',
    prompt: `Task: ./qc.sh tank, the revision loop (codename Tank-DD-HHMM), plus size tracking.
- Check also prints "size: class=..KB book=.. params=.. heap=..MB jar=..KB" (heap of new SerenaBot(15,30) after one
  move); fail above 60 MB.
- tank: qcsite archive; archive-losses for 'Tank-*' 'SerenaBot-*' 'QuarkBot-*' + top-3 names -> out/serena/losses.txt;
  learn; build/test/check; short 15x30 gate vs the newest Bots/Tank-*.jar; copy to Bots/Tank-DD-HHMM.jar; append
  arena/params/serena/revisions.tsv (name, git_rev, book_entries, gate_games, gate_score, class_kb, jar_kb, heap_mb);
  PRINT the upload command, don't run it. Add 'Tank-*' to qcsite's default loser patterns.`,
  },
  {
    key: 'mcp-upload',
    prompt: `Task: upload via MCP + opt-in autopilot (tools/qc-site/qcsite/).
- MCP tool upload_bot(jar_path, name, confirm): refuse unless name is Tank-DD-HHMM, confirm == name, ./qc.sh check
  passes; archive first; log arena/archive/uploads.jsonl; snapshot to arena/Bots/<name>.jar.
- qcsite autopilot: dry run by default; proposes an upload when our rank fell (or we're not first) and a candidate in
  arena/params/serena/revisions.tsv (columns as above) scored >= 60% over >= 20 gate games. Uploads only with --yes or
  QC_AUTOPILOT=1, max once per 30 min, not while the last upload's games are incomplete.
Test with the upload faked.`,
  },
  {
    key: 'network',
    prompt: `Task: input planes + small integer net, off by default.
- ./qc.sh dataset: replay archived matches, write integer planes (own/opponent quarks, playable cells, own/opponent
  threats, cooldown buckets, side to move) + result labels under arena/out/dataset/.
- tools/ Python trainer (uv, numpy): NNUE-style sparse -> int16 accumulator (32-64) -> clipped ReLU -> 1, quantised.
- SerenaBot: incremental accumulator in addQuark/removeTopQuark, base64 weight chunks between BEGIN/END markers,
  behind a static flag left false. nodes/s within 30% of today; check and conformance pass.
Report params, embedded KB, nodes/s with/without.`,
  },
  {
    key: 'islands',
    prompt: `Task: island evolution in Tune (arena/qc-trainer/.../tools/Tune.java): --islands N --migrate K, N populations
from different seeds, best copied between islands every K generations. Works on SerenaBot's weights via the narrowest
hook; fitness on 15x30 at the real time limit (fixed depth optional); size penalty for structural params.
Smoke test: 2 islands x 2 generations on a small board; give the full run's command.`,
  },
]

const RESULT = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    commit: { type: 'string' },
    summary: { type: 'string', description: 'what was done and verified, with numbers; what was not done' },
    results_tsv: { type: 'array', items: { type: 'string' } },
  },
  required: ['branch', 'commit', 'summary'],
}

const built = (await parallel(TASKS.map(task => () =>
  agent(`${task.prompt}\n${SHARED}`, { label: `build:${task.key}`, phase: 'Build', isolation: 'worktree', schema: RESULT })
    .then(r => r && { task: task.key, ...r }),
))).filter(Boolean)

log(`${built.length}/${TASKS.length} tasks built; writing the methodology`)

const methodology = await agent(
  `Write docs/methodology.md (concise) explaining how SerenaBot (team WilyQuarks) learns and evolves its network, for
readers who don't know our code: game model and search; learning loop (tournament -> archive -> losses -> loss book
-> Tank revision -> gate -> upload); input planes; network and memory; island evolution; budgets. Diagrams as SVG
in docs/figures/ (light and dark readable); charts from arena/results/ TSVs via a small uv Python script. No invented
numbers: say when a log is empty. Read the task branches with git show <branch>:<path>; don't merge them.
Branches:\n${JSON.stringify(built, null, 1)}
Work on a new branch "methodology", commit plainly (no AI attribution). Return branch, commit, figures.`,
  { label: 'explain:methodology', phase: 'Explain', isolation: 'worktree' },
)

return { tasks: built, methodology }
