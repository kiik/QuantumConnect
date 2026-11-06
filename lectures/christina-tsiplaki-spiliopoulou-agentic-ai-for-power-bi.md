# Beyond Prompting: Agentic AI for Power BI Development

**Speaker:** Christina Tsiplaki Spiliopoulou
**Notes added:** 2026-11-06
**Theme:** "What if prompting was only the warm-up?"

---

## 1. The core idea

The shift: you used to **ask a model questions**. Now you **give a model a goal and let it act**.

| | Prompt-based assistant | Agent |
|---|---|---|
| **Input** | your question | a goal or task |
| **Context** | whatever you paste in | whatever it can fetch: files, schemas, docs, live data |
| **Output** | text you copy somewhere | actions: edits files, runs queries, checks results |
| **Loop** | one shot, you check it | plan → act → observe → correct → repeat |
| **Your role** | operator | reviewer and supervisor |

"Prompting was the warm-up": a clever prompt matters less than what surrounds the model: the
**tools** it can call, the **context** it can read, and the **feedback loop** that shows it whether
it was right.

## 2. The three ingredients of an agent

1. **Tools.** Functions the model can call: read a file, run a DAX query, list tables, write a
   measure. The **Model Context Protocol (MCP)** is now the standard plug: one MCP server exposes a
   system's capabilities to any agent that supports it (Claude, GitHub Copilot, Cursor, ...).
2. **Context.** The model only knows what's in its context window. Agents pull in what they need:
   schema, naming conventions, existing measures, business definitions. This is **context
   engineering**, which has largely replaced prompt engineering as the skill that matters.
3. **The loop.** The model acts, sees the result (an error, a wrong number, a passing test) and
   adjusts. The loop turns a *plausible* answer into a *verified* one.

## 3. Why Power BI is a good test case

Power BI used to be hostile to AI: most work was clicking in Desktop, inside a binary `.pbix` file.
Three changes opened it up:

- **PBIP / TMDL / PBIR formats.** Semantic models and reports saved as human-readable text. Text
  can be read, diffed and edited by an agent, and kept in Git.
- **MCP servers for Power BI.** Microsoft's Power BI Modeling MCP server, plus remote MCP endpoints
  for querying models. An agent can connect to a model, list tables and measures, create or change
  objects, and run DAX against real data.
- **Copilot and Fabric data agents.** Built-in assistants cover the end-user side ("ask your data a
  question"); external agents cover the developer side.

## 4. What tends to work

- **Bulk, repetitive model work.** 40 time-intelligence measures, descriptions on every measure,
  naming conventions, format strings, display folders.
- **DAX writing with verification.** When the agent can run the query and see the numbers, quality
  jumps. Without execution it's guessing.
- **Documentation and explanation.** Explaining an inherited model, finding unused measures,
  mapping dependencies.
- **Refactoring as code.** Edit TMDL files, review the Git diff before accepting.
- **Intent → first draft.** "Build a star schema for this sales data" gives a reasonable skeleton
  to refine.

## 5. What tends to fail

- **Ambiguous business logic.** "Active customer" or "net revenue" means something specific in
  your company; the agent will confidently invent a definition unless you give it one.
- **Hallucinated objects.** Without live schema access it references columns and tables that don't
  exist. Tools fix most of this.
- **Report visuals and layout.** Still weak. PBIR is newer and more complex, and "make it look
  good" is hard to verify.
- **Performance tuning.** Needs query plans, VertiPaq stats and data volumes; without them the
  advice is generic.
- **Large models.** Hundreds of tables and thousands of measures overflow the context window. The
  agent sees fragments and makes locally sensible, globally wrong changes.
- **Silent wrongness** (the dangerous one). DAX that runs but returns subtly wrong totals doesn't
  error, so the loop won't catch it unless you supply **expected values**.
- **Governance.** Write access to a production model is a risk: you need permissions, environments
  and review steps.

## 6. What agentic development really looks like

The realistic workflow, not the demo version:

1. Keep the model in **PBIP format under Git**, on a dev workspace or local copy.
2. Give the agent **context files**: naming conventions, business glossary, measure patterns
   (standing instruction files such as `CLAUDE.md` or `AGENTS.md`).
3. **Connect the MCP server** so it can inspect the schema and run queries.
4. Assign a **bounded task**, e.g. "add YoY measures for all base revenue measures, following our
   conventions".
5. The agent works in a loop and **tests its DAX against data**.
6. **You review** the diff, check numbers against known values, then merge and deploy through the
   normal pipeline.

> The human role shifts from writing every line to **specifying, reviewing and verifying**.
> That shift is the real story behind the hype.

## 7. Honest takeaways

- Agents make a competent developer faster; they don't replace understanding the data model. If you
  can't spot a wrong DAX result, the agent won't save you.
- Gains are uneven: **large** for tedious, structured work, **small** for judgment-heavy work.
- **Context and verification beat clever prompts.** Most failures come from the agent lacking
  information or a way to check itself.
- **Text formats and Git are now foundational.** Teams working only in `.pbix` can't really use
  agents.

## Questions to bring to the talk

- How did they verify agent-generated DAX? Against what ground truth?
- How do they stop an agent from touching production models?
- How do they handle context for very large models?
- Where did the agent clearly fail, and was the cause the model, missing context or missing tools?

## Glossary

| Term | Meaning |
|---|---|
| **MCP** | Model Context Protocol: standard interface exposing a system's tools to AI agents |
| **PBIP** | Power BI Project: folder-based, text-file save format for a Power BI project |
| **TMDL** | Tabular Model Definition Language: text format for the semantic model |
| **PBIR** | Power BI Enhanced Report format: text format for report definitions |
| **DAX** | Data Analysis Expressions: Power BI's formula/query language |
| **VertiPaq** | the in-memory columnar engine behind Power BI models |
