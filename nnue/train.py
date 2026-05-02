from pathlib import Path
import csv
import random

import torch
from torch import nn
from tqdm import tqdm

from model import NnueNet


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


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "output/sr_2"
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(data_dir.rglob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {data_dir}")

    random.shuffle(csv_paths)
    split_idx = int(len(csv_paths) * 0.8)
    train_paths = csv_paths[:split_idx]
    val_paths = csv_paths[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NnueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 35
    batch_pairs = 256
    margin = 1.0

    best_val = None
    best_pth = model_dir / "nnue.pth"
    best_bin = model_dir / "nnue.bin"

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
        print(f"Epoch {epoch + 1}: loss {avg:.4f} val {val_loss:.4f}")

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_pth)

            weights = [
                p.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
                for p in model.parameters()
            ]
            flat = torch.cat(weights) if weights else torch.empty(0, dtype=torch.float32)
            flat.numpy().tofile(best_bin)

    print(f"Saved {best_pth} and {best_bin}")


if __name__ == "__main__":
    main()
