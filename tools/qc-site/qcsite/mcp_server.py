"""MCP server for the arena site: the CLI's reads as tools for agents.

One Site (one logged-in session) serves every tool call for the life of the process; it logs in
again by itself when the session expires. Tools return the same shapes as `qcsite <command> --json`.
The server only reads: uploading a bot replaces our tournament entry, so it stays a CLI command a
person runs on purpose.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import archive, client

server = FastMCP(
    "qc-site",
    instructions=(
        "Read-only access to the QuantumConnect arena site (quantum-connect.web.cern.ch), logged in "
        "as our team. Start with leaderboard; bot_matches lists a bot's games (bot = name, id or "
        "'ours'); match gives the moves, the replayed winner and the final board; lost_games gives "
        "every loss of a bot with its moves. Players are 'first' (1) and 'second' (2); columns count "
        "from 0 on the left; a move with bomb=true clears the 3x3 around the dropped quark."
    ),
)

_site: client.Site | None = None


def site() -> client.Site:
    global _site
    if _site is None:
        _site = client.Site()
    return _site


# ---------------------------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------------------------

@server.tool()
def leaderboard() -> dict:
    """The tournament standings: entries of rank, bot, bot_id, team, participants, wins, losses,
    draws. The organisers hide the leaderboard at times: then hidden=true and the entries are the
    latest saved copy (as_of), whose bot_ids still work in bot_matches."""
    return client.leaderboard(site())


@server.tool()
def our_team() -> dict:
    """Our team's name, to find our bot on the leaderboard."""
    return {"team": client.team_name(site())}


@server.tool()
def bot_matches(bot: str = "ours") -> dict:
    """A bot's leaderboard entry and its matches: match_id, first, second (bot names) and result
    ("Won"/"Lost") from that bot's side. `bot` is a bot name, a bot_id, or "ours"."""
    return client.bot_matches(site(), bot)


@server.tool()
def match(match_id: str, include_moves: bool = True) -> dict:
    """One match: both players (bot, bot_id, team), board size, the move list (player, column,
    bomb), the winner and deciding move as replayed by the game's rules, and the final board as
    rows from the top ("1" first player, "2" second, "." empty)."""
    details = client.match_details(site(), match_id)
    if not include_moves:
        details = {**details, "moves": None, "move_count": len(details["moves"])}
    return details


@server.tool()
def lost_games(bot: str = "ours") -> dict:
    """Every match the bot lost, with our side ("first"/"second"), the opponent, board size and
    full move list. Fetches one page per loss, so it takes a few seconds."""
    return client.lost_games(site(), bot)


@server.tool()
def archived_matches(bot_pattern: str = "*", lost_only: bool = False, include_moves: bool = False) -> list[dict]:
    """Matches saved in the local archive (arena/archive), which keeps games the site has deleted
    (a re-upload deletes a bot's matches). bot_pattern is a shell pattern on either player's bot
    name, e.g. "Tank-*"; with lost_only, only matches that bot lost. Refresh with `qcsite archive`."""
    if lost_only:
        matches = archive.archived_losses([bot_pattern])
    else:
        import fnmatch
        matches = [m for m in archive.stored_matches()
                   if fnmatch.fnmatchcase(m["first"]["bot"], bot_pattern) or fnmatch.fnmatchcase(m["second"]["bot"], bot_pattern)]
    if not include_moves:
        matches = [{**m, "moves": None, "move_count": len(m["moves"].split())} for m in matches]
    return matches


@server.tool()
def site_page(path: str) -> dict:
    """Summarise any page of the arena site: title, headings, tables, links. For pages the other
    tools don't cover. `path` is a site path such as "/rules"."""
    final_path, html = site().page(path)
    return client.page_summary(final_path, html)


def run() -> None:
    server.run()
