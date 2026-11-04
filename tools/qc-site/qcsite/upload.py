"""Guarded uploads, and the autopilot that proposes them.

An upload replaces our tournament bot and deletes its matches, so every upload here goes through
the same guards:

    name      a Tank revision, Tank-DD-HHMM (day, hour, minute), typed twice (name and confirm)
    check     `./qc.sh check <jar>` passes (prints "ok"): the arena's own rules on the JAR
    archive   our matches and the top three's are archived before they are deleted
    log       one JSON line per upload in arena/archive/uploads.jsonl
    snapshot  the JAR is kept as arena/Bots/<name>.jar, to replay against later

The autopilot reads the leaderboard and arena/params/serena/revisions.tsv (written by
`./qc.sh tank`). When we are not first, it proposes the newest revision that won at least 60% of
at least 20 gate games and is not uploaded yet. It only proposes (a dry run) unless given --yes or
QC_AUTOPILOT=1, and never uploads twice within 30 minutes or while the last upload is still
playing its tournament games.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from . import archive, client

TANK_NAME = re.compile(r"Tank-(0[1-9]|[12]\d|3[01])-([01]\d|2[0-3])[0-5]\d")
GATE_MIN_GAMES = 20
GATE_MIN_SCORE = 0.60
UPLOAD_SPACING = timedelta(minutes=30)
GAMES_PER_OPPONENT = 7


# ---------------------------------------------------------------------------------------------
# Places
# ---------------------------------------------------------------------------------------------

def arena_root() -> Path:
    """arena/ in the repository, or QC_ARENA."""
    override = os.environ.get("QC_ARENA")
    return Path(override) if override else Path(__file__).resolve().parents[3] / "arena"


def uploads_log() -> Path:
    return client.archive_root() / "uploads.jsonl"


def revisions_file() -> Path:
    return arena_root() / "params" / "serena" / "revisions.tsv"


# ---------------------------------------------------------------------------------------------
# The guarded upload
# ---------------------------------------------------------------------------------------------

def run_check(jar: Path) -> tuple[bool, str]:
    """`./qc.sh check <jar>`: passes when it exits 0 and prints "ok"."""
    done = subprocess.run(["./qc.sh", "check", str(jar.resolve())], cwd=arena_root(),
                          capture_output=True, text=True, timeout=300)
    output = done.stdout + done.stderr
    passed = done.returncode == 0 and any(line.strip() == "ok" for line in output.splitlines())
    return passed, output.strip()[-1200:]


def past_uploads() -> list[dict]:
    if not uploads_log().is_file():
        return []
    lines = uploads_log().read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def guarded_upload(site: client.Site, jar: Path, name: str, confirm: str, source: str) -> dict:
    """Uploads `jar` as `name` after every guard passed; raises SiteError at the first that fails."""
    if not TANK_NAME.fullmatch(name):
        raise client.SiteError(f"refused: {name!r} is not a Tank revision name (Tank-DD-HHMM)")
    if confirm != name:
        raise client.SiteError("refused: confirm must repeat the bot name exactly")
    if not jar.is_file():
        raise client.SiteError(f"refused: no such JAR: {jar}")
    passed, check_output = run_check(jar)
    if not passed:
        raise client.SiteError(f"refused: ./qc.sh check failed:\n{check_output}")
    archived = archive.archive(site, ["ours", "top:3"])
    result = client.upload_bot(site, name, jar)
    snapshot = arena_root() / "Bots" / f"{name}.jar"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if jar.resolve() != snapshot.resolve():
        shutil.copyfile(jar, snapshot)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "jar": str(jar),
        "sha256": hashlib.sha256(jar.read_bytes()).hexdigest(),
        "source": source,
        "archived_matches": archived["archived_matches"],
        "landed_on": result["landed_on"],
    }
    uploads_log().parent.mkdir(parents=True, exist_ok=True)
    with uploads_log().open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {**record, "snapshot": str(snapshot), "page_text": result["page_text"]}


# ---------------------------------------------------------------------------------------------
# The autopilot
# ---------------------------------------------------------------------------------------------

def read_revisions() -> list[dict]:
    """Rows of revisions.tsv (name, git_rev, book_entries, gate_games, gate_score, class_kb,
    jar_kb, heap_mb), oldest first; gate_score as a fraction even if written in percent."""
    if not revisions_file().is_file():
        return []
    lines = [line for line in revisions_file().read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    rows = []
    for row in csv.DictReader(lines, delimiter="\t"):
        try:
            games, score = int(row["gate_games"]), float(row["gate_score"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({**row, "gate_games": games, "gate_score": score / 100 if score > 1 else score})
    return rows


def best_candidate(revisions: list[dict], uploaded: set[str]) -> dict | None:
    """The newest revision that passed the gate, is not uploaded yet, and has its JAR in Bots/."""
    for row in reversed(revisions):
        jar = arena_root() / "Bots" / f"{row['name']}.jar"
        if (row["gate_games"] >= GATE_MIN_GAMES and row["gate_score"] >= GATE_MIN_SCORE
                and row["name"] not in uploaded and jar.is_file()):
            return {**row, "jar": str(jar)}
    return None


def games_played(entry: dict) -> int:
    return entry["wins"] + entry["losses"] + entry["draws"]


def autopilot(site: client.Site, really: bool, now: datetime | None = None) -> dict:
    """Decides whether to upload; uploads only when `really` and every guard allows it.
    The answer's "action" is one of: none, wait, propose, uploaded."""
    now = now or datetime.now()
    board = client.leaderboard(site)
    entries = board["entries"]
    team = client.team_name(site)
    ours = next((e for e in entries if e["team"] == team), None)
    state = {"rank": ours["rank"] if ours else None, "bot": ours["bot"] if ours else None,
             "leaderboard_hidden": board["hidden"]}

    def answer(action: str, reason: str, candidate: dict | None = None) -> dict:
        return {"action": action, "reason": reason, "candidate": candidate, **state}

    if ours is not None and ours["rank"] == 1:
        return answer("none", "we are first")
    uploads = past_uploads()
    candidate = best_candidate(read_revisions(), {u["name"] for u in uploads})
    if candidate is None:
        return answer("none", f"no unuploaded revision with >= {GATE_MIN_SCORE:.0%} "
                              f"over >= {GATE_MIN_GAMES} gate games")
    if uploads:
        last = uploads[-1]
        since = now - datetime.fromisoformat(last["time"])
        if since < UPLOAD_SPACING:
            return answer("wait", f"last upload {last['name']} only {int(since.total_seconds() // 60)} min ago", candidate)
        if board["hidden"]:
            return answer("wait", "leaderboard hidden: cannot tell whether the last upload finished its games", candidate)
        entry = next((e for e in entries if e["bot"] == last["name"]), None)
        expected = GAMES_PER_OPPONENT * (len(entries) - 1)
        if entry is None or games_played(entry) < expected:
            played = games_played(entry) if entry else 0
            return answer("wait", f"last upload {last['name']} has played {played}/{expected} games", candidate)
    if not really:
        return answer("propose", "dry run: repeat with --yes or QC_AUTOPILOT=1 to upload", candidate)
    result = guarded_upload(site, Path(candidate["jar"]), candidate["name"], candidate["name"], "autopilot")
    return {**answer("uploaded", f"uploaded {candidate['name']}", candidate), "upload": result}
