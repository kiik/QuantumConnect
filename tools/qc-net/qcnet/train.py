"""Trains SerenaBot's network on the planes of ./qc.sh dataset and embeds it in the bot.

The network, NNUE-style:
    sparse 0/1 features -> accumulator (hidden units) -> clipped ReLU [0, 1] -> one output,
the output being the log-odds that the side to move wins. It trains in floating point with Adam,
each position also mirrored left to right. The target is the mover's result in the archived match,
pulled towards a draw the further the position is from the end: 0.5 + (result - 0.5) * exp(-plies
left / horizon), since an early position says little about who wins. 10% of the matches are held
out, and the epoch with the best held-out loss is kept.

Quantised as SerenaBot computes it: input weights and hidden biases times 64 (int8, int16), the
clipped ReLU on 0..64, output weights times 64 (int16), the output bias times 64 * 64 (int32).

Writes params/serena/network.txt (provenance and base64), the NETWORK block of SerenaBot.java, and
one row per epoch to results/network-training.tsv.

Usage: uv run qcnet [--hidden 32] [--epochs 40] [--seed 1] [--rate 1e-3] [--decay 3] [--horizon 20] [--no-embed]
"""

import argparse
import base64
import datetime
import struct
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[3] / "arena"
BOT_SOURCE = ARENA / "qc-starter/src/main/java/ch/qc/meelik/bots/SerenaBot.java"
BEGIN, END = "    // NETWORK BEGIN", "    // NETWORK END"
ACTIVATION_ONE, OUTPUT_ONE = 64, 64  # SerenaBot's fixed-point ones
CHUNK = 60_000  # base64 characters per string constant (a constant holds at most 64 KB)


# ------------------------------------------------------------------ data

def load(directory):
    meta = {}
    for line in (directory / "meta.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            key, value = line.split()
            meta[key] = int(value)
    features = meta["features"]
    planes = np.fromfile(directory / "planes.bin", dtype=np.uint8).reshape(meta["positions"], -1)
    results = np.fromfile(directory / "results.bin", dtype=np.uint8).astype(np.float32) / 2
    games = np.fromfile(directory / "games.bin", dtype="<i4")
    return meta, planes, results, games


def plies_left(games):
    """Positions after each one in its match (the positions of a match are consecutive)."""
    _, match, counts = np.unique(games, return_inverse=True, return_counts=True)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return counts[match] - 1 - (np.arange(len(games)) - first[match])


def mirror_permutation(meta):
    """Feature index of each feature on the board mirrored left to right (cells are column * rows + height)."""
    rows, columns, cells = meta["rows"], meta["columns"], meta["rows"] * meta["columns"]
    permutation = np.arange(meta["features"])
    for plane in range(meta["cell_planes"]):
        for column in range(columns):
            for height in range(rows):
                permutation[plane * cells + column * rows + height] = plane * cells + (columns - 1 - column) * rows + height
    return permutation


def unpack(planes, features):
    return np.unpackbits(planes, axis=1)[:, :features].astype(np.float32)


# ------------------------------------------------------------------ the network in floating point

def forward(model, x):
    accumulator = x @ model["w1"] + model["b1"]
    hidden = np.clip(accumulator, 0, 1)
    return accumulator, hidden, hidden @ model["w2"] + model["b2"]


def bce(logits, targets):
    return float(np.mean(np.logaddexp(0, logits) - targets * logits))


def accuracy(logits, results):
    """How often the sign of the output names the winner, over positions of decided matches."""
    decisive = results != 0.5
    return float(np.mean((logits[decisive] > 0) == (results[decisive] > 0.5)))


def train(meta, planes, targets, results, train_rows, validation_rows, hidden, epochs, seed, rate, decay, log):
    rng = np.random.default_rng(seed)
    features = meta["features"]
    permutation = mirror_permutation(meta)
    model = {"w1": rng.normal(0, 0.03, (features, hidden)).astype(np.float32),
             "b1": np.full(hidden, 0.5, np.float32),
             "w2": rng.normal(0, 0.3, hidden).astype(np.float32),
             "b2": np.zeros(1, np.float32)}
    limits = {"w1": 127 / ACTIVATION_ONE, "b1": 32767 / ACTIVATION_ONE, "w2": 32767 / OUTPUT_ONE, "b2": 1000.0}
    moments = {name: (np.zeros_like(value), np.zeros_like(value)) for name, value in model.items()}
    batch, step = 256, 0
    best = (float("inf"), None, 0)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order), batch):
            rows = order[start:start + batch]
            x, batch_targets = unpack(planes[rows], features), targets[rows]
            mirrored = rng.random(len(rows)) < 0.5
            x[mirrored] = x[mirrored][:, permutation]
            accumulator, hidden_values, logits = forward(model, x)
            losses.append(bce(logits, batch_targets))
            d_logits = (1 / (1 + np.exp(-logits)) - batch_targets) / len(rows)
            d_accumulator = np.outer(d_logits, model["w2"]) * ((accumulator > 0) & (accumulator < 1))
            gradients = {"w1": x.T @ d_accumulator, "b1": d_accumulator.sum(0),
                         "w2": hidden_values.T @ d_logits, "b2": np.array([d_logits.sum()])}
            step += 1
            for name, gradient in gradients.items():  # Adam, with decoupled weight decay on the weights (AdamW)
                first, second = moments[name]
                first[:] = 0.9 * first + 0.1 * gradient
                second[:] = 0.999 * second + 0.001 * gradient ** 2
                corrected = first / (1 - 0.9 ** step) / (np.sqrt(second / (1 - 0.999 ** step)) + 1e-8)
                shrink = decay * model[name] if name in ("w1", "w2") else 0
                model[name] = np.clip(model[name] - rate * (corrected + shrink), -limits[name], limits[name]).astype(np.float32)
        logits = forward(model, unpack(planes[validation_rows], features))[2]
        validation_loss = bce(logits, targets[validation_rows])
        log(epoch, float(np.mean(losses)), validation_loss, accuracy(logits, results[validation_rows]), "float")
        if validation_loss < best[0]:
            best = (validation_loss, {name: value.copy() for name, value in model.items()}, epoch)
    return best[1], best[2]


# ------------------------------------------------------------------ quantised, as SerenaBot computes it

def quantise(model):
    return {"w1": np.round(model["w1"] * ACTIVATION_ONE).astype(np.int8),
            "b1": np.round(model["b1"] * ACTIVATION_ONE).astype(np.int16),
            "w2": np.round(model["w2"] * OUTPUT_ONE).astype(np.int16),
            "b2": np.int32(np.round(model["b2"][0] * ACTIVATION_ONE * OUTPUT_ONE))}


def quantised_logits(q, planes, features):
    x = np.unpackbits(planes, axis=1)[:, :features].astype(np.int64)
    accumulator = x @ q["w1"].astype(np.int64) + q["b1"]
    hidden = np.clip(accumulator, 0, ACTIVATION_ONE)
    return (hidden @ q["w2"].astype(np.int64) + int(q["b2"])) / (ACTIVATION_ONE * OUTPUT_ONE), int(np.abs(accumulator).max())


def blob(meta, q, hidden):
    return (bytes([meta["rows"], meta["columns"], hidden, 1]) + q["w1"].tobytes()
            + q["b1"].astype("<i2").tobytes() + q["w2"].astype("<i2").tobytes() + struct.pack("<i", int(q["b2"])))


# ------------------------------------------------------------------ output

def embed(text, provenance):
    chunks = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]
    params = ARENA / "params/serena/network.txt"
    params.write_text("".join(f"# {line}\n" for line in provenance) + "".join(chunk + "\n" for chunk in chunks))
    source = BOT_SOURCE.read_text().split("\n")
    begin = next(i for i, line in enumerate(source) if line.startswith(BEGIN))
    end = next(i for i, line in enumerate(source) if line.startswith(END))
    block = ["    static final String[] NETWORK = {"] + [f'            "{chunk}",' for chunk in chunks] + ["    };"]
    BOT_SOURCE.write_text("\n".join(source[:begin + 1] + block + source[end:]))
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=ARENA / "out/dataset")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rate", type=float, default=1e-3, help="Adam's learning rate")
    parser.add_argument("--decay", type=float, default=3, help="decoupled weight decay")
    parser.add_argument("--horizon", type=float, default=20, help="plies over which the result fades to a draw")
    parser.add_argument("--no-embed", action="store_true", help="train and report, but leave the bot as it is")
    args = parser.parse_args()

    meta, planes, results, games = load(args.data)
    targets = (0.5 + (results - 0.5) * np.exp(-plies_left(games) / args.horizon)).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    match_ids = np.unique(games)
    held_out = rng.choice(match_ids, max(1, len(match_ids) // 10), replace=False)
    validation = np.isin(games, held_out)
    train_rows, validation_rows = np.flatnonzero(~validation), np.flatnonzero(validation)

    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    run = f"{stamp} h{args.hidden} s{args.seed} d{args.decay}"
    provenance = [f"{stamp}: uv run qcnet --hidden {args.hidden} --epochs {args.epochs} --seed {args.seed} "
                  f"--rate {args.rate} --decay {args.decay} --horizon {args.horizon}; "
                  f"{len(train_rows)} training / {len(validation_rows)} held-out positions from {len(match_ids)} matches "
                  f"({meta['rows']}x{meta['columns']}, {meta['features']} features)"]
    table = ARENA / "results/network-training.tsv"
    new_table = not table.exists()
    with table.open("a") as out:
        if new_table:
            out.write("run\tepoch\thidden\ttrain_loss\tvalidation_loss\tvalidation_accuracy\tprecision\n")
            out.write("# SerenaBot's network (tools/qc-net): per epoch, mean binary cross-entropy on the target (the mover's result faded by plies left) "
                      "and held-out accuracy on decisive games; precision int = the embedded quantised network\n")
        out.write(f"# {provenance[0]}\n")

        def log(epoch, train_loss, validation_loss, accuracy, precision):
            out.write(f"{run}\t{epoch}\t{args.hidden}\t{train_loss:.4f}\t{validation_loss:.4f}\t{accuracy:.4f}\t{precision}\n")
            out.flush()
            print(f"epoch {epoch:>3}  train {train_loss:.4f}  held-out {validation_loss:.4f}  accuracy {accuracy:.3f}  ({precision})")

        model, best_epoch = train(meta, planes, targets, results, train_rows, validation_rows, args.hidden, args.epochs, args.seed, args.rate, args.decay, log)
        q = quantise(model)
        logits, largest = quantised_logits(q, planes[validation_rows], meta["features"])
        held_out_loss, held_out_accuracy = bce(logits, targets[validation_rows]), accuracy(logits, results[validation_rows])
        log(best_epoch, float("nan"), held_out_loss, held_out_accuracy, "int")

    data = blob(meta, q, args.hidden)
    text = base64.b64encode(data).decode()
    print(f"best epoch {best_epoch}; quantised held-out loss {held_out_loss:.4f}, accuracy {held_out_accuracy:.3f}; "
          f"largest accumulator {largest} (int16 holds 32767)")
    print(f"{len(data) / 1024:.1f} KB of weights, {len(text) / 1024:.1f} KB as base64")
    if not args.no_embed:
        provenance.append(f"best epoch {best_epoch}, held-out loss {held_out_loss:.4f}, accuracy {held_out_accuracy:.3f}; "
                          f"layout: rows, columns, hidden, version; int8 input weights; int16 biases, output weights; int32 output bias")
        print(f"-> {embed(text, provenance)}, {BOT_SOURCE.name}")


if __name__ == "__main__":
    main()
