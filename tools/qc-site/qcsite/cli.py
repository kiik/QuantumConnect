"""Command line for the arena site.

Every command prints a readable table by default and JSON with --json, so the same command serves a
person at a terminal and a script or agent. `qcsite mcp` serves the same reads as MCP tools.

    qcsite leaderboard               standings: rank, bot, team, wins, losses, draws
    qcsite bot [BOT]                 a bot's matches (BOT: name, id, or "ours", the default)
    qcsite match MATCH_ID            players, moves, replayed winner and final board
    qcsite losses [BOT] -o FILE      a bot's lost games, live from the site
    qcsite archive [--bot SEL]...    save new matches of ours and the top 3 in arena/archive
    qcsite archive-losses -o FILE    archived lost games, as text for `qc.sh learn`
    qcsite upload NAME JAR --yes     upload a bot: replaces ours and deletes its matches
    qcsite autopilot [--yes]         upload a gated Tank revision when we are not first (dry run)
    qcsite wait NAME                 wait until the tournament has played the bot's games
    qcsite team                      our team's name
    qcsite login                     check the credentials in .env
    qcsite page /some/path           what any site page holds: headings, tables, links
    qcsite mcp                       run the MCP server on stdio
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from . import archive, client, upload


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------

def print_json(value) -> None:
    json.dump(value, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def print_table(headers: list[str], rows: list[list]) -> None:
    """Columns padded to their widest cell; the header underlined."""
    cells = [["" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in cells:
        for index, cell in enumerate(row[: len(widths)]):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)).rstrip())
    print("  ".join("-" * width for width in widths))
    for row in cells:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def describe_moves(moves: list[dict]) -> str:
    return " ".join(
        ("f" if move["player"] == "first" else "s") + str(move["column"]) + ("b" if move["bomb"] else "")
        for move in moves)


# ---------------------------------------------------------------------------------------------
# Commands: reading
# ---------------------------------------------------------------------------------------------

def command_leaderboard(site: client.Site, arguments) -> None:
    board = client.leaderboard(site)
    entries = board["entries"]
    if arguments.json:
        print_json(board)
        return
    if not entries:
        print("the leaderboard is hidden, and no copy of it is saved yet")
        return
    if board["hidden"]:
        print(f"the leaderboard is hidden; this is the copy saved at {board['as_of']}\n")
    print_table(
        ["rank", "bot", "team", "W", "L", "D", "games", "bot id"],
        [[e["rank"], e["bot"], e["team"], e["wins"], e["losses"], e["draws"],
          e["wins"] + e["losses"] + e["draws"], e["bot_id"]] for e in entries])


def command_bot(site: client.Site, arguments) -> None:
    history = client.bot_matches(site, arguments.bot)
    if arguments.json:
        print_json(history)
        return
    print(f"{history['bot']} ({history['team']}): rank {history['rank']}, "
          f"{history['wins']} W / {history['losses']} L / {history['draws']} D")
    print_table(["result", "first", "second", "match id"],
                [[m["result"], m["first"], m["second"], m["match_id"]] for m in history["matches"]])


def command_match(site: client.Site, arguments) -> None:
    match = client.match_details(site, arguments.match_id)
    if arguments.json:
        print_json(match)
        return
    first, second = match["first"], match["second"]
    winner = {"first": first["bot"], "second": second["bot"]}.get(match["winner"], "nobody")
    print(f"{first['bot']} ({first['team']}, first, 1) vs {second['bot']} ({second['team']}, second, 2)")
    print(f"{match['rows']}x{match['columns']} board, {len(match['moves'])} moves, "
          f"{sum(m['bomb'] for m in match['moves'])} bombs; winner (replayed): {winner}"
          + (f" on move {match['deciding_move'] + 1}" if match["deciding_move"] is not None else ""))
    print()
    for row in match["board"]:
        print(" ".join(row))
    print(" ".join(str(column % 10) for column in range(match["columns"])))
    if arguments.moves:
        print("\nmoves:", describe_moves(match["moves"]))


def command_team(site: client.Site, arguments) -> None:
    name = client.team_name(site)
    print_json({"team": name}) if arguments.json else print(name)


def command_losses(site: client.Site, arguments) -> None:
    losses = client.lost_games(site, arguments.bot)
    if arguments.json:
        text = json.dumps(losses, indent=2, ensure_ascii=False) + "\n"
    else:
        text = client.games_text(losses, datetime.now().isoformat(timespec="seconds"))
    if arguments.output:
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.output).write_text(text, encoding="utf-8")
        print(f"{len(losses['games'])} lost games of {losses['bot']} -> {arguments.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


# ---------------------------------------------------------------------------------------------
# Commands: the local archive
# ---------------------------------------------------------------------------------------------

def command_archive(site: client.Site, arguments) -> None:
    report = archive.archive(site, arguments.bot or ["ours", "top:3"])
    if arguments.json:
        print_json(report)
        return
    if report["leaderboard_hidden"]:
        print(f"leaderboard hidden; bots taken from the copy saved at {report['leaderboard_as_of']}")
    print_table(["bot", "team", "rank", "matches", "lost", "new"],
                [[b["bot"], b["team"], b["rank"], b["listed"], b["lost"], b["new"]] for b in report["bots"]])
    print(f"{report['archived_matches']} matches in {archive.matches_dir()}")


def command_archive_losses(site: client.Site, arguments) -> None:
    patterns = arguments.loser or archive.OUR_LOSER_PATTERNS
    games = archive.archived_losses(patterns)
    if arguments.json:
        text = json.dumps(games, indent=1, ensure_ascii=False) + "\n"
    else:
        text = archive.learner_text(games, patterns)
    if arguments.output:
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.output).write_text(text, encoding="utf-8")
        print(f"{len(games)} archived lost games of {', '.join(patterns)} -> {arguments.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


# ---------------------------------------------------------------------------------------------
# Commands: uploading and waiting
# ---------------------------------------------------------------------------------------------

def command_upload(site: client.Site, arguments) -> None:
    if not arguments.yes:
        raise client.SiteError("an upload replaces our tournament bot and deletes its matches; "
                               "archive them (qcsite losses) and repeat with --yes")
    result = client.upload_bot(site, arguments.name, Path(arguments.jar))
    if arguments.json:
        print_json(result)
    else:
        print(f"uploaded {result['jar']} as {result['bot']}; the site answered with {result['landed_on']}")
        print(result["page_text"])


def command_autopilot(site: client.Site, arguments) -> None:
    really = arguments.yes or os.environ.get("QC_AUTOPILOT") == "1"
    decision = upload.autopilot(site, really)
    if arguments.json:
        print_json(decision)
        return
    print(f"we are {decision['bot']} at rank {decision['rank']}: {decision['action']} ({decision['reason']})")
    candidate = decision["candidate"]
    if candidate:
        print(f"candidate {candidate['name']}: {candidate['gate_score']:.0%} over "
              f"{candidate['gate_games']} gate games, {candidate['jar']}")


def command_wait(site: client.Site, arguments) -> None:
    """Polls the leaderboard until the bot has played every other bot `games` times, or the
    timeout passes. No games at all after the timeout usually means the arena rejected the bot."""
    deadline = time.monotonic() + arguments.timeout * 60
    while True:
        board = client.leaderboard(site)
        entries = [] if board["hidden"] else board["entries"]
        entry = next((e for e in entries if e["bot"] == arguments.name), None)
        expected = arguments.games * (len(entries) - 1)
        played = entry["wins"] + entry["losses"] + entry["draws"] if entry else 0
        stamp = datetime.now().strftime("%H:%M:%S")
        if board["hidden"]:
            state = "leaderboard hidden"
        elif entry is None:
            state = "not on the leaderboard"
        else:
            state = f"{played}/{expected} games, {entry['wins']} W {entry['losses']} L, rank {entry['rank']}"
        print(f"{stamp} {arguments.name}: {state}", file=sys.stderr, flush=True)
        if entry is not None and played >= expected:
            status = "complete"
            break
        if time.monotonic() >= deadline:
            status = ("leaderboard hidden" if board["hidden"]
                      else "no games (rejected?)" if played == 0 else "incomplete")
            break
        time.sleep(arguments.interval)
    result = {"bot": arguments.name, "status": status, "played": played, "expected": expected, "entry": entry}
    print_json(result) if arguments.json else print(f"{arguments.name}: {status}, {played}/{expected} games")
    if status != "complete":
        raise SystemExit(2)


# ---------------------------------------------------------------------------------------------
# Commands: plumbing
# ---------------------------------------------------------------------------------------------

def command_login(site: client.Site, arguments) -> None:
    site.login()
    user_name, _ = client.load_credentials()
    print_json({"logged_in": True, "user": user_name}) if arguments.json else print(f"logged in as {user_name}")


def command_page(site: client.Site, arguments) -> None:
    path, html = site.page(arguments.path)
    if arguments.html:
        sys.stdout.write(html)
        return
    summary = client.page_summary(path, html)
    if arguments.json:
        print_json(summary)
        return
    print(f"{summary['path']}  —  {summary['title']}")
    for heading in summary["headings"]:
        print(f"# {heading}")
    for number, table in enumerate(summary["tables"], 1):
        print(f"\ntable {number}:")
        width = max([len(table["headers"])] + [len(row) for row in table["rows"]])
        print_table(table["headers"] or [f"col{index}" for index in range(width)], table["rows"])
    if summary["links"]:
        print("\nlinks:")
        for link in summary["links"]:
            print(f"  {link['href']}  {link['text']}")


def command_mcp(site: client.Site, arguments) -> None:
    from . import mcp_server

    site.close()
    mcp_server.run()


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qcsite", description="Read the QuantumConnect arena site.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 2)[2])
    parser.add_argument("--json", action="store_true", help="print JSON instead of tables")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("leaderboard", help="the standings")

    bot = commands.add_parser("bot", help="a bot's matches")
    bot.add_argument("bot", nargs="?", default="ours", help='bot name, bot id, or "ours" (default)')

    match = commands.add_parser("match", help="one match: players, replayed winner, final board")
    match.add_argument("match_id")
    match.add_argument("--moves", action="store_true", help="also list the moves")

    commands.add_parser("team", help="our team's name")

    losses = commands.add_parser("losses", help="every lost game with its moves (text for qc.sh learn)")
    losses.add_argument("bot", nargs="?", default="ours", help='bot name, bot id, or "ours" (default)')
    losses.add_argument("-o", "--output", help="write to this file instead of stdout")

    archive_parser = commands.add_parser("archive", help="save new matches of followed bots in arena/archive")
    archive_parser.add_argument("--bot", action="append",
                                help='bot to follow: "ours", "top:N", a name or an id; repeatable '
                                     '(default: ours and top:3)')

    archive_losses = commands.add_parser("archive-losses", help="archived lost games, for qc.sh learn")
    archive_losses.add_argument("--loser", action="append",
                                help='loser bot name pattern, e.g. "Tank-*", "King beater" or "top:3" (the latest '
                                     'saved top 3); repeatable; default "Tank-*" and "SerenaBot-*"')
    archive_losses.add_argument("-o", "--output", help="write to this file instead of stdout")

    upload = commands.add_parser("upload", help="upload a bot JAR: replaces our bot, deletes its matches")
    upload.add_argument("name", help="bot name, 5-20 characters, e.g. SerenaBot-04-1809")
    upload.add_argument("jar", help="the JAR, e.g. arena/qc-starter/build/libs/SerenaBot.jar")
    upload.add_argument("--yes", action="store_true", help="really upload")

    autopilot = commands.add_parser("autopilot", help="upload a gated Tank revision when we are not first")
    autopilot.add_argument("--yes", action="store_true", help="really upload (or QC_AUTOPILOT=1); else a dry run")

    wait = commands.add_parser("wait", help="wait until the tournament has played the bot's games")
    wait.add_argument("name", help="the bot name as uploaded")
    wait.add_argument("--games", type=int, default=7, help="games against each other bot (default 7)")
    wait.add_argument("--timeout", type=float, default=25, help="minutes to wait (default 25)")
    wait.add_argument("--interval", type=float, default=60, help="seconds between polls (default 60)")

    commands.add_parser("login", help="check the credentials (USER_NAME, USER_PSK in .env)")

    page = commands.add_parser("page", help="summarise any site page (headings, tables, links)")
    page.add_argument("path", help="site path, e.g. /leaderboard")
    page.add_argument("--html", action="store_true", help="print the raw HTML instead")

    commands.add_parser("mcp", help="run the MCP server on stdio")
    return parser


COMMANDS = {
    "leaderboard": command_leaderboard,
    "bot": command_bot,
    "match": command_match,
    "team": command_team,
    "losses": command_losses,
    "archive": command_archive,
    "archive-losses": command_archive_losses,
    "upload": command_upload,
    "autopilot": command_autopilot,
    "wait": command_wait,
    "login": command_login,
    "page": command_page,
    "mcp": command_mcp,
}


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        with client.Site() as site:
            COMMANDS[arguments.command](site, arguments)
    except client.SiteError as error:
        print(f"qcsite: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
