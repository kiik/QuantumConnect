"""The upload guards and the autopilot, with the site, the check and the upload faked.

    uv run python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from qcsite import client, upload

HEADER = "name\tgit_rev\tbook_entries\tgate_games\tgate_score\tclass_kb\tjar_kb\theap_mb\n"


def entry(rank: int, bot: str, team: str, wins: int = 40, losses: int = 2) -> dict:
    return {"rank": rank, "bot": bot, "bot_id": bot, "team": team, "participants": "",
            "wins": wins, "losses": losses, "draws": 0}


class UploadTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.arena = root / "arena"
        (self.arena / "Bots").mkdir(parents=True)
        (self.arena / "params" / "serena").mkdir(parents=True)
        self.environment = mock.patch.dict(os.environ, {"QC_ARENA": str(self.arena),
                                                        "QC_ARCHIVE": str(self.arena / "archive")})
        self.environment.start()
        self.uploaded: list[str] = []
        self.board = [entry(1, "King beater", "Kings"), entry(2, "Tank-04-1700", "WilyQuarks")]
        fakes = {
            "qcsite.client.upload_bot": self.fake_upload,
            "qcsite.client.leaderboard": lambda site: {"entries": self.board, "hidden": False, "as_of": "now"},
            "qcsite.client.team_name": lambda site: "WilyQuarks",
            "qcsite.archive.archive": lambda site, selectors: {"archived_matches": 12},
            "qcsite.upload.run_check": lambda jar: (True, "  ok"),
        }
        self.patches = [mock.patch(target, side_effect=fake) for target, fake in fakes.items()]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.environment.stop()
        self.temp.cleanup()

    def fake_upload(self, site, name, jar):
        self.uploaded.append(name)
        return {"bot": name, "jar": str(jar), "landed_on": "/bots", "page_text": "uploaded"}

    def revision(self, name: str, games: int, score: str) -> None:
        tsv = upload.revisions_file()
        if not tsv.exists():
            tsv.write_text("# revisions of SerenaBot, written by ./qc.sh tank\n" + HEADER)
        with tsv.open("a") as rows:
            rows.write(f"{name}\tabc123\t40\t{games}\t{score}\t30\t900\t60\n")
        (self.arena / "Bots" / f"{name}.jar").write_bytes(b"jar " + name.encode())

    # -- guarded upload --------------------------------------------------------------------------

    def test_refuses_bad_names_confirmations_and_failed_checks(self):
        jar = self.arena / "candidate.jar"
        jar.write_bytes(b"jar")
        for name, confirm in [("SerenaBot-04-1800", "SerenaBot-04-1800"), ("Tank-4-1800", "Tank-4-1800"),
                              ("Tank-04-1800", "Tank-04-1801")]:
            with self.assertRaises(client.SiteError):
                upload.guarded_upload(None, jar, name, confirm, "test")
        with mock.patch("qcsite.upload.run_check", return_value=(False, "PROBLEM: java.io")):
            with self.assertRaisesRegex(client.SiteError, "check failed"):
                upload.guarded_upload(None, jar, "Tank-04-1800", "Tank-04-1800", "test")
        self.assertEqual(self.uploaded, [])

    def test_upload_logs_and_snapshots(self):
        jar = self.arena / "candidate.jar"
        jar.write_bytes(b"jar")
        result = upload.guarded_upload(None, jar, "Tank-04-1800", "Tank-04-1800", "mcp")
        self.assertEqual(self.uploaded, ["Tank-04-1800"])
        self.assertEqual((self.arena / "Bots" / "Tank-04-1800.jar").read_bytes(), b"jar")
        logged = upload.past_uploads()
        self.assertEqual([(u["name"], u["source"]) for u in logged], [("Tank-04-1800", "mcp")])
        self.assertEqual(result["archived_matches"], 12)

    # -- autopilot -------------------------------------------------------------------------------

    def test_dry_run_proposes_the_newest_gated_revision(self):
        self.revision("Tank-04-1810", 30, "0.70")
        self.revision("Tank-04-1820", 10, "0.90")   # too few gate games
        self.revision("Tank-04-1830", 24, "55")      # below 60%
        decision = upload.autopilot(None, really=False)
        self.assertEqual((decision["action"], decision["candidate"]["name"]), ("propose", "Tank-04-1810"))
        self.assertEqual(self.uploaded, [])

    def test_nothing_while_first(self):
        self.revision("Tank-04-1810", 30, "0.70")
        self.board = [entry(1, "Tank-04-1700", "WilyQuarks"), entry(2, "King beater", "Kings")]
        self.assertEqual(upload.autopilot(None, really=True)["action"], "none")
        self.assertEqual(self.uploaded, [])

    def test_uploads_when_allowed_then_waits(self):
        self.revision("Tank-04-1810", 30, "62.5")
        decision = upload.autopilot(None, really=True)
        self.assertEqual(decision["action"], "uploaded")
        self.assertEqual(self.uploaded, ["Tank-04-1810"])
        self.revision("Tank-04-1840", 30, "0.80")
        soon = datetime.now() + timedelta(minutes=10)
        self.assertEqual(upload.autopilot(None, really=True, now=soon)["action"], "wait")
        later = datetime.now() + timedelta(minutes=40)
        self.board = [entry(1, "King beater", "Kings"), entry(2, "Tank-04-1810", "WilyQuarks", 3, 1)]
        self.assertIn("4/7 games", upload.autopilot(None, really=True, now=later)["reason"])
        self.board = [entry(1, "King beater", "Kings"), entry(2, "Tank-04-1810", "WilyQuarks", 5, 2)]
        self.assertEqual(upload.autopilot(None, really=True, now=later)["action"], "uploaded")
        self.assertEqual(self.uploaded, ["Tank-04-1810", "Tank-04-1840"])
        self.assertEqual(json.loads(upload.uploads_log().read_text().splitlines()[-1])["source"], "autopilot")


if __name__ == "__main__":
    unittest.main()
