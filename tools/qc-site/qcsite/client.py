"""The arena site as Python data.

The site (https://quantum-connect.web.cern.ch) is a server-rendered Spring application: every page
except the home, rules, login and register pages redirects to /login until the session cookie
belongs to a logged-in user. There is no JSON API, so this module logs in with the form, fetches
pages and reads their HTML.

Concepts:
- Credentials come from the environment (USER_NAME, USER_PSK), else from the repository's .env.
  They are only ever sent to the site's /login form; nothing here prints or stores them.
- A Site holds one HTTP session. It logs in lazily, on the first page that needs it, and again
  if the session expires (a page answers with the login form).
- The session cookie lives in memory only: the CLI logs in once per run, the MCP server once per
  process.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

SITE_URL = "https://quantum-connect.web.cern.ch"
USER_AGENT = "qcsite/0.1 (QuantumConnect team tool)"


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------

class SiteError(Exception):
    """The site could not be read: network failure, unexpected answer, or bad credentials."""


class MissingCredentials(SiteError):
    pass


class LoginFailed(SiteError):
    pass


# ---------------------------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------------------------

def repository_env_file() -> Path:
    """The .env at the repository root: tools/qc-site/qcsite/client.py is four levels below it."""
    override = os.environ.get("QC_ENV_FILE")
    return Path(override) if override else Path(__file__).resolve().parents[3] / ".env"


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; blank lines, comments and an optional `export ` prefix are allowed."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


def load_credentials() -> tuple[str, str]:
    """(user name, password): the environment wins over the .env file."""
    env_file = repository_env_file()
    from_file = read_env_file(env_file)
    user_name = os.environ.get("USER_NAME") or from_file.get("USER_NAME")
    user_psk = os.environ.get("USER_PSK") or from_file.get("USER_PSK")
    if not user_name or not user_psk:
        raise MissingCredentials(f"set USER_NAME and USER_PSK in the environment or in {env_file}")
    return user_name, user_psk


# ---------------------------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------------------------

class Site:
    """One logged-in browsing session on the arena site."""

    def __init__(self, base_url: str = SITE_URL, timeout_seconds: float = 30.0):
        self.base_url = base_url
        self.http = httpx.Client(
            base_url=base_url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_seconds,
            follow_redirects=True,
        )
        self.logged_in = False

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "Site":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def login(self) -> None:
        user_name, user_psk = load_credentials()
        try:
            self.http.get("/login")  # the site wants its session cookie before the form post
            answer = self.http.post(
                "/login",
                data={"username": user_name, "password": user_psk},
                follow_redirects=False,
            )
        except httpx.HTTPError as error:
            raise SiteError(f"login request failed: {type(error).__name__}") from None
        location = answer.headers.get("location", "")
        if answer.status_code not in (302, 303) or "/login" in location:
            raise LoginFailed(f"login refused for user {user_name!r} (check USER_NAME / USER_PSK)")
        self.logged_in = True

    def page(self, path: str) -> tuple[str, str]:
        """(final path, HTML) of a site page, logging in first if the site asks for it."""
        path = site_path(path)
        if not self.logged_in:
            self.login()
        answer = self._get(path)
        if answer.url.path == "/login":  # the session expired: log in again, once
            self.login()
            answer = self._get(path)
            if answer.url.path == "/login":
                raise SiteError(f"GET {path} still redirects to /login after logging in")
        if answer.status_code >= 400:
            raise SiteError(f"GET {path} answered HTTP {answer.status_code}")
        query = answer.url.query.decode()
        return answer.url.path + (f"?{query}" if query else ""), answer.text

    def _get(self, path: str) -> httpx.Response:
        try:
            return self.http.get(path)
        except httpx.HTTPError as error:
            raise SiteError(f"GET {path} failed: {type(error).__name__}") from None


def site_path(path: str) -> str:
    """A path on the site; full URLs are accepted only for the site itself, so the session
    cookie never goes elsewhere."""
    if path.startswith(SITE_URL):
        path = path[len(SITE_URL):]
    if "://" in path or path.startswith("//"):
        raise SiteError(f"not a path on {SITE_URL}: {path}")
    return path if path.startswith("/") else "/" + path


# ---------------------------------------------------------------------------------------------
# Generic page reading
# ---------------------------------------------------------------------------------------------

def soup_of(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def clean_text(node) -> str:
    return " ".join(node.get_text(" ", strip=True).split())


def page_summary(path: str, html: str) -> dict:
    """What a page holds, without knowing its layout: title, headings, tables, links, htmx calls."""
    soup = soup_of(html)
    for noise in soup(["script", "style", "nav", "footer"]):
        noise.decompose()
    tables = []
    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        rows = []
        for row in table.find_all("tr"):
            cells = [clean_text(cell) for cell in row.find_all("td")]
            if cells:
                rows.append(cells)
        tables.append({"headers": [clean_text(cell) for cell in header_cells], "rows": rows})
    links = []
    for anchor in soup.find_all("a", href=True):
        links.append({"text": clean_text(anchor), "href": anchor["href"]})
    htmx = []
    for element in soup.find_all(attrs={"hx-get": True}):
        htmx.append({"hx-get": element["hx-get"], "hx-trigger": element.get("hx-trigger", "")})
    title = soup.title.get_text(strip=True) if soup.title else ""
    return {
        "path": path,
        "title": title,
        "headings": [clean_text(h) for h in soup.find_all(["h1", "h2", "h3", "h4"])],
        "tables": tables,
        "links": links,
        "htmx": htmx,
        "text": clean_text(soup.body or soup)[:4000],
    }


# ---------------------------------------------------------------------------------------------
# Leaderboard: /leaderboard
# ---------------------------------------------------------------------------------------------

TROPHY_RANKS = {"Gold Trophy": 1, "Silver Trophy": 2, "Bronze Trophy": 3}


def parse_leaderboard(html: str) -> list[dict]:
    """One entry per bot: rank, bot name and id, team, participants, wins, losses, draws.
    Ranks 1-3 are trophy icons on the page; they come back as numbers."""
    entries = []
    table = soup_of(html).find("table")
    if table is None:
        return entries  # the organisers hide the leaderboard at times
    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 6:
            continue
        trophy = cells[0].find("i")
        rank_text = clean_text(cells[0])
        rank = TROPHY_RANKS.get(trophy.get("title", "")) if trophy else None
        if rank is None and rank_text.isdigit():
            rank = int(rank_text)
        link = cells[1].find("a")
        team_text = clean_text(cells[2])
        team, participants = team_text, None
        team_and_count = re.fullmatch(r"(.*?)\s*\(\s*(\d+)\s+participant\(s\)\s*\)", team_text)
        if team_and_count:
            team, participants = team_and_count.group(1), int(team_and_count.group(2))
        entries.append({
            "rank": rank,
            "bot": clean_text(link) if link else clean_text(cells[1]),
            "bot_id": link["href"].rsplit("/", 1)[-1] if link else None,
            "team": team,
            "participants": participants,
            "wins": int(clean_text(cells[3])),
            "losses": int(clean_text(cells[4])),
            "draws": int(clean_text(cells[5])),
        })
    return entries


# ---------------------------------------------------------------------------------------------
# A bot's match history: /leaderboard/<bot id>
# ---------------------------------------------------------------------------------------------

def parse_bot_matches(html: str) -> dict:
    """The bot's name and its matches: match id, first and second player, and the result
    ("Won", "Lost", ...) seen from this bot's side."""
    soup = soup_of(html)
    heading = soup.find(["h2", "h3"], string=None)
    name_node = soup.select_one("h3 p") or soup.select_one("h2 p")
    matches = []
    for link in soup.find_all("a", href=True):
        if not link["href"].startswith("/matches/"):
            continue
        card = link.parent.parent
        spans = [clean_text(span) for span in card.find_all("span")]
        if len(spans) < 4 or spans[1].lower() != "vs":
            continue
        matches.append({
            "match_id": link["href"].rsplit("/", 1)[-1],
            "first": spans[0],
            "second": spans[2],
            "result": spans[3],
        })
    return {"bot": clean_text(name_node) if name_node else (clean_text(heading) if heading else ""), "matches": matches}


# ---------------------------------------------------------------------------------------------
# One match: /matches/<match id>
# ---------------------------------------------------------------------------------------------

def parse_match(html: str) -> dict:
    """Both players and the full move list, as the site's match viewer has them. The page states
    no winner; replay_match works it out."""
    soup = soup_of(html)
    players = []
    for link in soup.find_all("a", href=True):
        if link["href"].startswith("/leaderboard/"):
            team_node = link.find_parent("div").find_next_sibling("div")
            team = clean_text(team_node.find("span")) if team_node and team_node.find("span") else ""
            players.append({"bot": clean_text(link), "bot_id": link["href"].rsplit("/", 1)[-1], "team": team})
    history = re.search(r"const matchHistory = (\[.*?\]);", html, re.S)
    rows = re.search(r"const ROWS = (\d+);", html)
    columns = re.search(r"const COLS = (\d+);", html)
    if history is None or rows is None or columns is None:
        raise SiteError("match page without a move history (layout changed?)")
    moves = [
        {"player": "first" if move["player"] == "FIRST_PLAYER" else "second",
         "column": move["column"], "bomb": bool(move["isSpecialMove"])}
        for move in json.loads(history.group(1))
    ]
    match_id = re.search(r"Match ID:\s*<span[^>]*>([^<]+)</span>", html)
    return {
        "match_id": match_id.group(1).strip() if match_id else None,
        "first": players[0] if players else None,
        "second": players[1] if len(players) > 1 else None,
        "rows": int(rows.group(1)),
        "columns": int(columns.group(1)),
        "moves": moves,
    }


# ---------------------------------------------------------------------------------------------
# Replaying a match: who won, and the final board
# ---------------------------------------------------------------------------------------------

LINE_LENGTH = 6
LINE_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def replay_match(rows: int, columns: int, moves: list[dict]) -> dict:
    """Plays the move list by the engine's rules (QuarkBot's model, checked against the engine):
    a quark drops to the lowest free cell; a bomb drops it and clears the 3x3 around it, the quarks
    above falling down; after every move, any six in a row wins for the player who just moved.
    Returns the winner ("first", "second" or None), the deciding move's index and the final board
    (rows from the top, "1"/"2"/"."). A rough check, for reading results: our bots use the Java model."""
    stacks = [[] for _ in range(columns)]  # bottom first
    winner, deciding_move = None, None
    for index, move in enumerate(moves):
        column, player = move["column"], move["player"]
        if not 0 <= column < columns:  # an invalid answer the engine rejected
            continue
        stack = stacks[column]
        if len(stack) >= rows:
            continue
        stack.append(player)
        if move["bomb"]:
            height = len(stack) - 1
            for blasted in range(max(0, column - 1), min(columns, column + 2)):
                kept = [quark for h, quark in enumerate(stacks[blasted]) if not height - 1 <= h <= height + 1]
                stacks[blasted] = kept
        if has_line(stacks, rows, columns):
            winner, deciding_move = player, index
            break
    board = []
    for row in range(rows):
        height = rows - 1 - row
        board.append("".join(
            ("1" if stacks[c][height] == "first" else "2") if height < len(stacks[c]) else "."
            for c in range(columns)))
    return {"winner": winner, "deciding_move": deciding_move, "board": board}


def has_line(stacks: list[list[str]], rows: int, columns: int) -> bool:
    def owner(column: int, height: int):
        if 0 <= column < columns and 0 <= height < len(stacks[column]):
            return stacks[column][height]
        return None

    for column in range(columns):
        for height in range(len(stacks[column])):
            player = stacks[column][height]
            for column_step, height_step in LINE_DIRECTIONS:
                if all(owner(column + i * column_step, height + i * height_step) == player for i in range(1, LINE_LENGTH)):
                    return True
    return False


# ---------------------------------------------------------------------------------------------
# Reads that combine pages
# ---------------------------------------------------------------------------------------------

def team_name(site: Site) -> str:
    _, html = site.page("/team")
    heading = soup_of(html).find("h2", string=re.compile("My team"))
    name = heading.find_next_sibling("p") if heading else None
    if name is None:
        raise SiteError("team page without a team name (layout changed?)")
    return clean_text(name)


# The organisers hide the leaderboard at times. Every visible one is saved in the archive, and a
# hidden one is answered from the latest saved copy: its bot ids still open the bots' match pages.

def archive_root() -> Path:
    """arena/archive in the repository, or QC_ARCHIVE."""
    override = os.environ.get("QC_ARCHIVE")
    return Path(override) if override else Path(__file__).resolve().parents[3] / "arena" / "archive"


def leaderboard(site: Site) -> dict:
    """{"entries", "hidden", "as_of"}: the live leaderboard, saved to the archive when it changed,
    or the latest saved one (hidden=True) while the site hides it."""
    _, html = site.page("/leaderboard")
    entries = parse_leaderboard(html)
    snapshots = archive_root() / "leaderboard"
    saved = sorted(snapshots.glob("*.json")) if snapshots.is_dir() else []
    if entries:
        latest = json.loads(saved[-1].read_text(encoding="utf-8"))["entries"] if saved else None
        stamp = datetime.now().isoformat(timespec="seconds")
        if entries != latest:
            snapshots.mkdir(parents=True, exist_ok=True)
            (snapshots / f"{stamp.replace(':', '-')}.json").write_text(
                json.dumps({"as_of": stamp, "entries": entries}, indent=1, ensure_ascii=False), encoding="utf-8")
        return {"entries": entries, "hidden": False, "as_of": stamp}
    if not saved:
        return {"entries": [], "hidden": True, "as_of": None}
    snapshot = json.loads(saved[-1].read_text(encoding="utf-8"))
    return {"entries": snapshot["entries"], "hidden": True, "as_of": snapshot["as_of"]}


def find_bot(site: Site, which: str) -> dict:
    """A leaderboard entry by bot id, exact bot name, or "ours" (our team's bot). While the
    leaderboard is hidden, from its latest saved copy."""
    entries = leaderboard(site)["entries"]
    if which == "ours":
        team = team_name(site)
        matching = [entry for entry in entries if entry["team"] == team]
    else:
        matching = [entry for entry in entries if which in (entry["bot_id"], entry["bot"])]
    if not matching:
        raise SiteError(f"no bot {which!r} on the leaderboard")
    return matching[0]


def bot_matches(site: Site, which: str) -> dict:
    entry = find_bot(site, which)
    _, html = site.page(f"/leaderboard/{entry['bot_id']}")
    history = parse_bot_matches(html)
    return {**entry, "matches": history["matches"]}


def match_details(site: Site, match_id: str) -> dict:
    _, html = site.page(f"/matches/{match_id}")
    match = parse_match(html)
    replay = replay_match(match["rows"], match["columns"], match["moves"])
    return {**match, **replay}


# ---------------------------------------------------------------------------------------------
# Lost games, for learning
# ---------------------------------------------------------------------------------------------

def lost_games(site: Site, which: str) -> dict:
    """Every match the bot lost, with its side and full move list."""
    history = bot_matches(site, which)
    games = []
    for match in history["matches"]:
        if match["result"].lower() != "lost":
            continue
        details = match_details(site, match["match_id"])
        side = "first" if details["first"]["bot"] == history["bot"] else "second"
        opponent = details["second"] if side == "first" else details["first"]
        games.append({
            "match_id": match["match_id"],
            "side": side,
            "opponent": opponent["bot"],
            "rows": details["rows"],
            "columns": details["columns"],
            "moves": details["moves"],
            "replayed_winner": details["winner"],
        })
    return {"bot": history["bot"], "bot_id": history["bot_id"], "team": history["team"], "games": games}


def games_text(losses: dict, fetched_at: str) -> str:
    """The lost games as text for the Java learner, one game per line:
    <match id> <rows>x<columns> <our side> <moves>, a move being f|s, the column, and b for a bomb."""
    lines = [f"# lost games of {losses['bot']} ({losses['team']}), fetched {fetched_at} by qcsite losses"]
    for game in losses["games"]:
        moves = " ".join(
            ("f" if move["player"] == "first" else "s") + str(move["column"]) + ("b" if move["bomb"] else "")
            for move in game["moves"])
        lines.append(f"{game['match_id']} {game['rows']}x{game['columns']} {game['side']} {moves}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Uploading a bot (replaces our team's bot and deletes its matches)
# ---------------------------------------------------------------------------------------------

BOT_NAME_LENGTH = (5, 20)


def upload_bot(site: Site, bot_name: str, jar_path: Path) -> dict:
    """Uploads a bot JAR under a name, as the /bots form does. The site keeps one bot per team:
    the upload replaces our bot and deletes its matches, so archive them first."""
    if not BOT_NAME_LENGTH[0] <= len(bot_name) <= BOT_NAME_LENGTH[1]:
        raise SiteError(f"bot name must be {BOT_NAME_LENGTH[0]}-{BOT_NAME_LENGTH[1]} characters: {bot_name!r}")
    if not jar_path.is_file():
        raise SiteError(f"no such JAR: {jar_path}")
    if not site.logged_in:
        site.login()
    with jar_path.open("rb") as jar:
        try:
            answer = site.http.post(
                "/bots/upload",
                data={"botName": bot_name},
                files={"botFile": (jar_path.name, jar, "application/java-archive")},
            )
        except httpx.HTTPError as error:
            raise SiteError(f"upload failed: {type(error).__name__}") from None
    if answer.url.path == "/login":
        raise SiteError("upload bounced to the login page")
    if answer.status_code >= 400:
        raise SiteError(f"upload answered HTTP {answer.status_code}")
    summary = page_summary(answer.url.path, answer.text)
    if "no longer upload" in summary["text"]:
        raise SiteError("the site refused the upload: uploads are closed")
    return {"bot": bot_name, "jar": str(jar_path), "landed_on": summary["path"], "page_text": summary["text"][:600]}
