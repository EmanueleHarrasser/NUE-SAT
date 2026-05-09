from __future__ import annotations

from pathlib import Path
import argparse
import csv
import random
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import torch
from torch import nn
from tqdm import tqdm
from training.model import FcNet


FEATURE_COLUMNS = [
    "pos_len_2",
    "neg_len_2",
    "pos_len_3",
    "neg_len_3",
    "pos_len_4p",
    "neg_len_4p",
    "conflict_heat",
    "recent_flips",
    "trail_depth",
    "active_clause_ratio",
]


def iter_pairs(csv_paths):
    for path in csv_paths:
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            current_id = None
            pos = None
            negatives = []

            for row in reader:
                decision_id = int(row["decision_id"])
                if current_id is None:
                    current_id = decision_id

                if decision_id != current_id:
                    assert pos is not None
                    for neg in negatives:
                        yield pos, neg
                    pos = None
                    negatives = []
                    current_id = decision_id

                features = [float(row[name]) for name in FEATURE_COLUMNS]
                label = int(row["label"])
                if label == 1:
                    pos = features
                else:
                    negatives.append(features)

            if pos is not None:
                for neg in negatives:
                    yield pos, neg


def compute_loss(model, pairs, device, margin):
    pos_batch = torch.tensor([p for p, _ in pairs], dtype=torch.float32, device=device)
    neg_batch = torch.tensor([n for _, n in pairs], dtype=torch.float32, device=device)
    pos_scores = model(pos_batch)
    neg_scores = model(neg_batch)
    target = torch.ones(len(pairs), device=device)
    criterion = nn.MarginRankingLoss(margin=margin)
    return criterion(pos_scores, neg_scores, target)


def eval_epoch_loss(model, csv_paths, device, margin, batch_pairs):
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    batch = []

    with torch.no_grad():
        for pair in iter_pairs(csv_paths):
            batch.append(pair)
            if len(batch) < batch_pairs:
                continue

            loss = compute_loss(model, batch, device, margin)
            total_loss += loss.item() * len(batch)
            total_pairs += len(batch)
            batch = []

        if batch:
            loss = compute_loss(model, batch, device, margin)
            total_loss += loss.item() * len(batch)
            total_pairs += len(batch)

    return total_loss / total_pairs


def save_model_weights(model: nn.Module, pth_path: Path, bin_path: Path) -> None:
    torch.save(model.state_dict(), pth_path)
    weights = [
        p.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
        for p in model.parameters()
    ]
    flat = torch.cat(weights) if weights else torch.empty(0, dtype=torch.float32)
    flat.numpy().tofile(bin_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-pairs", type=int, default=256)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def train_model(
    data_dir: Path,
    model_dir: Path,
    init_path: Path | None,
    epochs: int,
    batch_pairs: int,
    margin: float,
    lr: float,
    seed: int | None,
    best_name: str = "network",
    latest_name: str | None = None,
) -> tuple[Path, Path, float, float, float]:
    csv_paths = sorted(data_dir.rglob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {data_dir}")

    if seed is not None:
        random.seed(seed)
    random.shuffle(csv_paths)
    split_idx = int(len(csv_paths) * 0.8)
    train_paths = csv_paths[:split_idx]
    val_paths = csv_paths[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FcNet().to(device)
    if init_path is not None:
        model.load_state_dict(torch.load(init_path, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = None
    best_pth = model_dir / f"{best_name}.pth"
    best_bin = model_dir / f"{best_name}.bin"
    latest_pth = model_dir / f"{latest_name}.pth" if latest_name else None
    latest_bin = model_dir / f"{latest_name}.bin" if latest_name else None

    last_train_loss = 0.0
    last_val_loss = 0.0

    for epoch in range(epochs):
        model.train()
        batch = []
        total_loss = 0.0
        total_batches = 0

        for pair in tqdm(
            iter_pairs(train_paths),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="pair",
        ):
            batch.append(pair)
            if len(batch) < batch_pairs:
                continue

            loss = compute_loss(model, batch, device, margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            batch = []

        if batch:
            loss = compute_loss(model, batch, device, margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

        avg = total_loss / max(total_batches, 1)
        val_loss = eval_epoch_loss(model, val_paths, device, margin, batch_pairs)
        last_train_loss = avg
        last_val_loss = val_loss
        print(f"Epoch {epoch + 1}: loss {avg:.4f} val {val_loss:.4f}")

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            save_model_weights(model, best_pth, best_bin)

    if latest_pth and latest_bin:
        save_model_weights(model, latest_pth, latest_bin)

    print(f"Saved {best_pth} and {best_bin}")
    return best_pth, best_bin, best_val, last_train_loss, last_val_loss


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or (root / "output/sr_1")
    model_dir = args.model_dir or (root / "models")
    model_dir.mkdir(parents=True, exist_ok=True)

    train_model(
        data_dir=data_dir,
        model_dir=model_dir,
        init_path=args.init,
        epochs=args.epochs,
        batch_pairs=args.batch_pairs,
        margin=args.margin,
        lr=args.lr,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
