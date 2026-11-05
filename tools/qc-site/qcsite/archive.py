"""The local match archive: every match of the bots we follow, kept after the site forgets it.

The site keeps one bot per team, and re-uploading deletes the old bot's matches, so lost games
must be saved before every upload or they are gone. The archive (arena/archive/) holds:

    leaderboard/<time>.json   every leaderboard seen that differs from the one before
    matches/<match id>.json   one match: players, board size, moves, replayed winner, when fetched

A match never changes once played, so archiving is incremental: only new match ids are fetched.
By default it follows our bot and the leaderboard's top three: their losses are lessons too,
since SerenaBot's learner reads any lost game from the loser's side.
"""

from __future__ import annotations

import fnmatch
import json
from datetime import datetime
from pathlib import Path

from . import client


# ---------------------------------------------------------------------------------------------
# Storing matches
# ---------------------------------------------------------------------------------------------

def matches_dir() -> Path:
    return client.archive_root() / "matches"


def moves_text(moves: list[dict]) -> str:
    return " ".join(
        ("f" if move["player"] == "first" else "s") + str(move["column"]) + ("b" if move["bomb"] else "")
        for move in moves)


def moves_from_text(text: str) -> list[dict]:
    return [{"player": "first" if token[0] == "f" else "second",
             "column": int(token[1:].rstrip("b")), "bomb": token.endswith("b")}
            for token in text.split()]


def played_columns(match: dict) -> int:
    """The board width a match was played on. The site's match page gives the board size of the
    tournament at the time it is fetched, not of the match: after a switch between 15x100 and 15x30,
    older matches come back with the new width. On 15 rows the moves tell: a 15x100 match starts
    near the centre column 49 and a 15x30 match never reaches column 30."""
    if match["rows"] != 15 or match["columns"] not in (30, 100):
        return match["columns"]
    highest = max((move["column"] for move in moves_from_text(match["moves"])), default=0)
    return 100 if highest >= 30 else 30


def store_match(details: dict) -> None:
    record = {
        "match_id": details["match_id"],
        "first": details["first"],
        "second": details["second"],
        "rows": details["rows"],
        "columns": details["columns"],
        "winner": details["winner"],
        "deciding_move": details["deciding_move"],
        "moves": moves_text(details["moves"]),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    matches_dir().mkdir(parents=True, exist_ok=True)
    (matches_dir() / f"{details['match_id']}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


def stored_matches() -> list[dict]:
    if not matches_dir().is_dir():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(matches_dir().glob("*.json"))]


# ---------------------------------------------------------------------------------------------
# Archiving from the site
# ---------------------------------------------------------------------------------------------

def followed_bots(site: client.Site, selectors: list[str]) -> list[dict]:
    """Leaderboard entries for selectors: "ours", "top:N", a bot name or a bot id."""
    board = client.leaderboard(site)
    entries = board["entries"]
    chosen: list[dict] = []
    for selector in selectors:
        if selector.startswith("top:"):
            count = int(selector.split(":", 1)[1])
            chosen += sorted((e for e in entries if e["rank"]), key=lambda e: e["rank"])[:count]
        else:
            chosen.append(client.find_bot(site, selector))
    unique = {entry["bot_id"]: entry for entry in chosen}
    return list(unique.values())


def archive(site: client.Site, selectors: list[str]) -> dict:
    """Fetches and stores every not-yet-archived match of the followed bots."""
    board = client.leaderboard(site)
    known = {path.stem for path in matches_dir().glob("*.json")} if matches_dir().is_dir() else set()
    report = []
    for entry in followed_bots(site, selectors):
        _, html = site.page(f"/leaderboard/{entry['bot_id']}")
        listed = client.parse_bot_matches(html)["matches"]
        new = [match for match in listed if match["match_id"] not in known]
        for match in new:
            store_match(client.match_details(site, match["match_id"]))
            known.add(match["match_id"])
        lost = sum(match["result"].lower() == "lost" for match in listed)
        report.append({"bot": entry["bot"], "team": entry["team"], "rank": entry["rank"],
                       "listed": len(listed), "lost": lost, "new": len(new)})
    return {"leaderboard_hidden": board["hidden"], "leaderboard_as_of": board["as_of"],
            "archived_matches": len(known), "bots": report}


# ---------------------------------------------------------------------------------------------
# Reading lost games back
# ---------------------------------------------------------------------------------------------

# Our uploads over time: QuarkBot-*, SerenaBot-*, and now SerenaBot revisions codenamed Tank-DD-HHMM.
OUR_LOSER_PATTERNS = ["Tank-*", "SerenaBot-*"]


def top_names(count: int) -> list[str]:
    """The names of the top bots on the latest saved leaderboard, offline."""
    snapshots = sorted((client.archive_root() / "leaderboard").glob("*.json"))
    if not snapshots:
        return []
    entries = json.loads(snapshots[-1].read_text(encoding="utf-8"))["entries"]
    return [e["bot"] for e in sorted((e for e in entries if e["rank"]), key=lambda e: e["rank"])[:count]]


def archived_losses(loser_patterns: list[str]) -> list[dict]:
    """Archived matches whose loser's bot name matches one of the patterns (shell-style, e.g.
    "Tank-*", or "top:N" for the exact names of the latest saved leaderboard's top N), each with
    the loser's side."""
    names = {name for p in loser_patterns if p.startswith("top:") for name in top_names(int(p[4:]))}
    loser_patterns = [p for p in loser_patterns if not p.startswith("top:")]
    games = []
    for match in stored_matches():
        if match["winner"] not in ("first", "second"):
            continue
        loser_side = "second" if match["winner"] == "first" else "first"
        loser = match[loser_side]["bot"]
        if loser in names or any(fnmatch.fnmatchcase(loser, pattern) for pattern in loser_patterns):
            games.append({**match, "loser": loser, "loser_side": loser_side})
    return games


def learner_text(games: list[dict], patterns: list[str]) -> str:
    """Lost games in the learner's format: <match id> <rows>x<columns> <loser's side> <moves>."""
    stamp = datetime.now().isoformat(timespec="seconds")
    lines = [f"# lost games of {', '.join(patterns)} from arena/archive, {stamp} (qcsite archive-losses)"]
    for game in games:
        lines.append(f"# {game['loser']} lost to "
                     f"{game['first' if game['loser_side'] == 'second' else 'second']['bot']}")
        lines.append(f"{game['match_id']} {game['rows']}x{played_columns(game)} {game['loser_side']} {game['moves']}")
    return "\n".join(lines) + "\n"
