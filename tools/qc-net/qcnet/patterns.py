"""Trains SerenaBot's pattern table on the planes of ./qc.sh dataset and embeds it in the bot.

The pattern table values every line of six cells by its exact contents, each cell empty, own or the
opponent's: 3^6 = 729 patterns per direction class (level, upright, diagonal). A position's value for
the side to move is the sum of its lines' values, plus a bomb readiness bonus and a tempo. The table
never refers to a cell or a board size, so positions of every board size train one table, and the
table plays on any board.

The values are shared where the game is symmetric: swapping the colours negates a value, and a level
line reversed (its mirror image) keeps it. Diagonals are read lowest cell first, so both diagonals
share one class without help. A line holding six of one colour has ended the game and gets 0.

The model is a logistic regression: the sum, divided by a scale, is the log-odds that the mover wins.
It starts from SerenaBot's line weights, which value only lines one player owns, by quark count, and
is pulled towards them (L2), so a pattern seen rarely keeps its line-weight value. The scale is fitted
first, with the line weights alone. Targets are those of qcnet: the search label blended with the
faded result. 10% of the matches of each data set are held out; the regularisation with the best
held-out loss is kept.

Writes params/serena/patterns.txt (provenance and base64), the PATTERNS block of SerenaBot.java, and
rows to results/pattern-training.tsv.

Usage: uv run qcpatterns [--data out/dataset-15x100 --data out/dataset-15x30 ...] [--reg 1e-4 ...]
                        [--lam 0.7] [--horizon 20] [--no-embed]
"""

import argparse
import base64
import datetime
import re
import struct
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

from qcnet.train import (ARENA, BOT_SOURCE, CHUNK, accuracy, bce, bce_probabilities, embedded, fit_calibration, load,
                         plies_left, quantised_logits, win_probability)

LENGTH, PATTERNS, CLASSES = 6, 3 ** 6, 3
LEVEL, UPRIGHT, DIAGONAL = 0, 1, 2
BEGIN, END = "    // PATTERNS BEGIN", "    // PATTERNS END"


# ------------------------------------------------------------------ the symmetries of the patterns

def digits(pattern):
    return [pattern // 3 ** i % 3 for i in range(LENGTH)]


def number(cells):
    return sum(cell * 3 ** i for i, cell in enumerate(cells))


def symmetries():
    """For each class and pattern, its feature (the class's smallest equivalent pattern) and its sign: +1, -1, or 0 for none."""
    feature, sign = np.zeros((CLASSES, PATTERNS), np.int64), np.zeros((CLASSES, PATTERNS), np.int64)
    features = {}
    for direction in range(CLASSES):
        for pattern in range(PATTERNS):
            cells = digits(pattern)
            if pattern == 0 or cells.count(1) == LENGTH or cells.count(2) == LENGTH:
                continue
            same = {pattern} | ({number(cells[::-1])} if direction == LEVEL else set())
            swapped = {number([(3 - c) % 3 for c in digits(p)]) for p in same}
            if same & swapped:  # a pattern worth its own negation is worth 0
                continue
            representative = min(same | swapped)
            feature[direction, pattern] = features.setdefault((direction, representative), len(features))
            sign[direction, pattern] = 1 if representative in same else -1
    return feature, sign, features


FEATURE, SIGN, REPRESENTATIVES = symmetries()
PATTERN_FEATURES = len(REPRESENTATIVES)
BOMB, TEMPO = PATTERN_FEATURES, PATTERN_FEATURES + 1  # the two extra features
FEATURES = PATTERN_FEATURES + 2


# ------------------------------------------------------------------ features of positions

def line_patterns(state):
    """The patterns of all lines of boards given as (positions, columns, rows) of 0 empty, 1 mover, 2 opponent, per class."""
    columns, rows = state.shape[1], state.shape[2]
    level = sum(3 ** i * state[:, i:columns - 5 + i, :] for i in range(LENGTH))
    upright = sum(3 ** i * state[:, :, i:rows - 5 + i] for i in range(LENGTH))
    rising = sum(3 ** i * state[:, i:columns - 5 + i, i:rows - 5 + i] for i in range(LENGTH))
    falling = sum(3 ** i * state[:, 5 - i:columns - i, i:rows - 5 + i] for i in range(LENGTH))  # lowest cell first
    n = len(state)
    return [(LEVEL, level.reshape(n, -1)), (UPRIGHT, upright.reshape(n, -1)),
            (DIAGONAL, np.concatenate([rising.reshape(n, -1), falling.reshape(n, -1)], axis=1))]


def position_features(meta, planes, chunk=2048):
    """A sparse matrix of the features of each position: signed pattern counts, bomb readiness difference, tempo."""
    rows, columns, cells = meta["rows"], meta["columns"], meta["rows"] * meta["columns"]
    extras = meta["cell_planes"] * cells
    parts = []
    for start in range(0, len(planes), chunk):
        bits = np.unpackbits(planes[start:start + chunk], axis=1)
        n = len(bits)
        state = (bits[:, :cells].astype(np.int16) + 2 * bits[:, cells:2 * cells]).reshape(n, columns, rows)
        row_ids, feature_ids, values = [], [], []
        for direction, patterns in line_patterns(state):
            signs = SIGN[direction][patterns]
            positions, lines = np.nonzero(signs)
            row_ids.append(positions)
            feature_ids.append(FEATURE[direction][patterns[positions, lines]])
            values.append(signs[positions, lines])
        bomb = bits[:, extras].astype(np.int64) - bits[:, extras + meta["cooldown_buckets"]]
        row_ids += [np.arange(n), np.arange(n)]
        feature_ids += [np.full(n, BOMB), np.full(n, TEMPO)]
        values += [bomb, np.ones(n, np.int64)]
        parts.append(sparse.csr_matrix((np.concatenate(values).astype(np.float64),
                                        (np.concatenate(row_ids), np.concatenate(feature_ids))), shape=(n, FEATURES)))
    return sparse.vstack(parts).tocsr()


# ------------------------------------------------------------------ the line weights as a pattern table

def line_weights():
    """SerenaBot's LINE_WEIGHT and BOMB_READY_BONUS, read from its source."""
    source = BOT_SOURCE.read_text()
    block = re.search(r"LINE_WEIGHT = \{(.*?)\};", source, re.S).group(1)
    weights = [[int(v) for v in row.split(",")] for row in re.findall(r"\{([^{}]*)\}", block)]
    bonus = int(re.search(r"BOMB_READY_BONUS = (-?\d+);", source).group(1))
    return weights, bonus


def prior(weights, bonus):
    """The features' weights in score units that reproduce the line weights: a line one player owns is worth its count's weight."""
    w = np.zeros(FEATURES)
    for (direction, pattern), feature in REPRESENTATIVES.items():
        cells = digits(pattern)
        own, opponent = cells.count(1), cells.count(2)
        w[feature] = weights[direction][own] if opponent == 0 else -weights[direction][opponent] if own == 0 else 0
    w[BOMB] = bonus
    return w


def table(w):
    """The full table of pattern values for SerenaBot, from player 0's point of view (digit 1 = player 0), in score units."""
    values = np.zeros((CLASSES, PATTERNS), np.int64)
    for direction in range(CLASSES):
        values[direction] = np.round(SIGN[direction] * w[FEATURE[direction]])
    return values


# ------------------------------------------------------------------ training

def loss_and_gradient(w, x, targets, w0, reg):
    logits = x @ w
    loss = bce(logits, targets) + 0.5 * reg * np.sum((w - w0) ** 2)
    gradient = x.T @ ((1 / (1 + np.exp(-logits)) - targets) / len(targets)) + reg * (w - w0)
    return loss, gradient


def fit(x, targets, w0, reg):
    result = optimize.minimize(loss_and_gradient, w0.copy(), args=(x, targets, w0, reg), jac=True, method="L-BFGS-B",
                               options={"maxiter": 2000})
    return result.x


def fit_scale(scores, targets):
    """Score units per log-odds for the line weights: the scale that best predicts the targets (a grid search)."""
    candidates = np.logspace(1, 5, 161)
    losses = [bce(scores / scale, targets) for scale in candidates]
    return float(candidates[int(np.argmin(losses))])


# ------------------------------------------------------------------ data sets

class DataSet:
    """One board size's positions: features, targets, the faded result, and which rows are held out."""

    def __init__(self, directory, lam, horizon, seed, search_scale=None):
        self.meta, self.planes, self.results, games, scores, self.comment = load(directory)
        self.size = f"{self.meta['rows']}x{self.meta['columns']}"
        self.faded = (0.5 + (self.results - 0.5) * np.exp(-plies_left(games) / horizon)).astype(np.float64)
        rng = np.random.default_rng(seed)
        archived = games < self.meta.get("archive_games", games.max() + 1)
        matches, self_play = np.unique(games[archived]), np.unique(games[~archived])
        held_out = rng.choice(matches, max(1, len(matches) // 10), replace=False)
        if len(self_play):
            held_out = np.concatenate([held_out, rng.choice(self_play, max(1, len(self_play) // 10), replace=False)])
        validation = np.isin(games, held_out)
        self.train, self.validation = np.flatnonzero(~validation), np.flatnonzero(validation)
        self.archived_validation = np.flatnonzero(validation & archived)
        self.matches, self.self_play = len(matches), len(self_play)
        self.lam = lam if scores is not None else 0.0
        self.label = "faded result"
        targets = self.faded
        if scores is not None:
            if search_scale:  # the search alone: no fit to results, which on 15x100 are decided by timeouts
                scale, confidence = search_scale, 1.0
            else:
                scale, confidence = fit_calibration(scores[self.train], self.results[self.train])
            targets = lam * win_probability(scores, scale, confidence) + (1 - lam) * self.faded
            self.label = f"{lam} x (0.5 + {confidence} x (sigmoid(search / {scale:.0f}) - 0.5)) + {1 - lam:.1f} x faded result"
        self.targets = targets.astype(np.float64)
        self.x = position_features(self.meta, self.planes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, action="append", help="a ./qc.sh dataset directory (repeat for more board sizes)")
    parser.add_argument("--reg", type=float, action="append", help="L2 pull towards the line weights (repeat to try several; the best held-out wins)")
    parser.add_argument("--lam", type=float, default=0.7, help="weight of the search label against the faded result")
    parser.add_argument("--horizon", type=float, default=20, help="plies over which the result fades to a draw")
    parser.add_argument("--search-scale", type=float, help="the search label is sigmoid(search / this), not calibrated to the results")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-embed", action="store_true", help="train and report, but leave the bot as it is")
    args = parser.parse_args()
    directories = args.data or [ARENA / "out/dataset"]
    regs = args.reg or [1e-3, 1e-4, 1e-5, 1e-6]

    sets = [DataSet(directory, args.lam, args.horizon, args.seed, args.search_scale) for directory in directories]
    for data in sets:
        print(f"{data.size}: {len(data.train)} training / {len(data.validation)} held-out positions from {data.matches} matches"
              f" and {data.self_play} self-play games; target {data.label}")
    x_train = sparse.vstack([d.x[d.train] for d in sets]).tocsr()
    t_train = np.concatenate([d.targets[d.train] for d in sets])
    x_validation = sparse.vstack([d.x[d.validation] for d in sets]).tocsr()
    t_validation = np.concatenate([d.targets[d.validation] for d in sets])

    weights, bonus = line_weights()
    w_lines = prior(weights, bonus)
    scale = fit_scale(x_train @ w_lines, t_train)
    w0 = w_lines / scale
    print(f"line weights: {scale:.0f} score units per log-odds; held-out loss {bce(x_validation @ w0, t_validation):.4f}")

    line_weights_loss = bce(x_validation @ w0, t_validation)
    best = (line_weights_loss, w0, 0.0)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    log = ARENA / "results/pattern-training.tsv"
    new_log = not log.exists()
    with log.open("a") as out:
        if new_log:
            out.write("run\tdata\treg\ttrain_loss\tvalidation_loss\tline_weights_loss\n")
            out.write("# SerenaBot's pattern table (tools/qc-net qcpatterns): mean binary cross-entropy on the target; "
                      "line_weights_loss = the line weights alone, scaled to log-odds\n")
        sizes = "+".join(d.size for d in sets)
        out.write(f"# {stamp}: uv run qcpatterns {' '.join('--data ' + str(d) for d in directories)} --lam {args.lam} "
                  f"--horizon {args.horizon:g} --search-scale {args.search_scale}; scale {scale:.0f}\n")
        for reg in regs:
            w = fit(x_train, t_train, w0, reg)
            train_loss, validation_loss = bce(x_train @ w, t_train), bce(x_validation @ w, t_validation)
            out.write(f"{stamp}\t{sizes}\t{reg:g}\t{train_loss:.4f}\t{validation_loss:.4f}\t{line_weights_loss:.4f}\n")
            print(f"reg {reg:g}: train {train_loss:.4f}  held-out {validation_loss:.4f}")
            if validation_loss < best[0]:
                best = (validation_loss, w, reg)
        validation_loss, w, reg = best
        values = table(w * scale)

        # the earlier runs' measure: held-out archived positions against the faded result
        report = []
        for data in sets:
            rows = data.archived_validation
            faded, results = data.faded[rows], data.results[rows]
            entries = [("line weights", data.x[rows] @ w0), ("pattern table", data.x[rows] @ w)]
            old = embedded_table()
            if old is not None:
                entries.append(("the table it replaces", data.x[rows] @ old / scale))
            network = embedded(data.meta)
            if network is not None:
                entries.append(("the network", quantised_logits(network, data.planes[rows], data.meta["features"])[0]))
            line = (f"{data.size} held-out archived positions ({len(rows)}), loss against the result faded over {args.horizon:g} plies: "
                    + ", ".join(f"{name} {bce(logits, faded):.4f} (accuracy {accuracy(logits, results):.3f})" for name, logits in entries)
                    + "; coin flip 0.6931")
            out.write(f"# {line}\n")
            report.append(line)
            print(line)

    changed = int(np.sum(np.abs(np.round(w * scale) - np.round(w0 * scale)) >= 1))
    print(f"kept reg {reg:g}: held-out loss {validation_loss:.4f}; {changed} of {FEATURES} weights moved off the line weights; "
          f"bomb readiness {w[BOMB] * scale:.0f}, tempo {w[TEMPO] * scale:.0f}; largest value {np.abs(values).max()}")
    if args.no_embed:
        return
    data = bytes([1]) + values.astype("<i4").tobytes() + struct.pack("<ii", int(round(w[BOMB] * scale)), int(round(w[TEMPO] * scale)))
    text = base64.b64encode(data).decode()
    provenance = [f"{stamp}: uv run qcpatterns {' '.join('--data ' + str(d) for d in directories)} --lam {args.lam} --horizon {args.horizon:g} --search-scale {args.search_scale}; "
                  f"reg {reg:g} (best held-out of {', '.join(f'{r:g}' for r in regs)}), scale {scale:.0f} score units per log-odds; "
                  f"data: " + "; ".join(f"{d.size}: {d.comment}" for d in sets)] + report
    print(f"{len(data) / 1024:.1f} KB of table, {len(text) / 1024:.1f} KB as base64 -> {embed(text, provenance)}, {BOT_SOURCE.name}")


def embedded_table():
    """The feature weights (score units) of the table SerenaBot embeds now, or None."""
    params = ARENA / "params/serena/patterns.txt"
    if not params.exists():
        return None
    data = base64.b64decode("".join(line.strip() for line in params.read_text().splitlines() if not line.startswith("#")))
    values = np.frombuffer(data, "<i4", CLASSES * PATTERNS, 1).reshape(CLASSES, PATTERNS)
    w = np.zeros(FEATURES)
    for (direction, pattern), feature in REPRESENTATIVES.items():
        w[feature] = values[direction, pattern]
    w[BOMB], w[TEMPO] = struct.unpack_from("<ii", data, 1 + 4 * CLASSES * PATTERNS)
    return w


def embed(text, provenance):
    chunks = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]
    params = ARENA / "params/serena/patterns.txt"
    params.write_text("".join(f"# {line}\n" for line in provenance) + "".join(chunk + "\n" for chunk in chunks))
    source = BOT_SOURCE.read_text().split("\n")
    begin = next(i for i, line in enumerate(source) if line.startswith(BEGIN))
    end = next(i for i, line in enumerate(source) if line.startswith(END))
    block = ["    static final String[] PATTERN_TABLE = {"] + [f'            "{chunk}",' for chunk in chunks] + ["    };"]
    BOT_SOURCE.write_text("\n".join(source[:begin + 1] + block + source[end:]))
    return params


if __name__ == "__main__":
    main()
