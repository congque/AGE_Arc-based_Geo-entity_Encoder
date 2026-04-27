from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .entitysettransformer import EntitySetTransformer
    from .load_entities import load_gpkg
except ImportError:
    from entitysettransformer import EntitySetTransformer
    from load_entities import load_gpkg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/buildings/ShapeClassification.gpkg")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-blocks", type=int, default=2)
    parser.add_argument("--num-decoder-blocks", type=int, default=1)
    parser.add_argument("--output-dir", default="model_edges/results/entitysettransformer_shapeclassification")
    parser.add_argument("--xy-num-freqs", type=int, default=8)
    parser.add_argument("--length-fourier", action="store_true")
    parser.add_argument("--length-num-freqs", type=int, default=None)
    parser.add_argument("--second-harmonic", action="store_true")
    parser.add_argument("--use-endpoints", action="store_true")
    return parser.parse_args()


def collate_fn(batch):
    edge_sets, labels = zip(*batch)
    return list(edge_sets), torch.tensor(labels, dtype=torch.long)


def split_data(edge_sets, labels, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (
        [(edge_sets[i], labels[i]) for i in train_idx],
        [(edge_sets[i], labels[i]) for i in val_idx],
        [(edge_sets[i], labels[i]) for i in test_idx],
    )


def macro_f1(labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    f1s = []
    for c in sorted(set(labels.tolist()) | set(preds.tolist())):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def run_epoch(model, loader, optimizer, criterion, device):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_n = 0
    all_labels = []
    all_preds = []

    with torch.set_grad_enabled(train):
        for edge_sets, labels in loader:
            edge_sets = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in edge_sets]
            labels = labels.to(device)
            logits = model(edge_sets)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * labels.shape[0]
            total_n += labels.shape[0]
            all_labels.append(labels.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    return {
        "loss": total_loss / total_n,
        "accuracy": float((labels == preds).mean()),
        "macro_f1": macro_f1(labels, preds),
    }


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_sets, raw_labels = load_gpkg(
        args.input,
        args.label_column,
        xy_num_freqs=args.xy_num_freqs,
        length_fourier=args.length_fourier,
        length_num_freqs=args.length_num_freqs,
        second_harmonic=args.second_harmonic,
        use_endpoints=args.use_endpoints,
    )
    classes, labels = np.unique(raw_labels, return_inverse=True)
    train_data, val_data, test_data = split_data(edge_sets, labels, args.seed)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = EntitySetTransformer(
        input_dim=edge_sets[0].shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        output_dim=len(classes),
        num_heads=args.num_heads,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = run_epoch(model, val_loader, None, criterion, device)
        print(
            f"epoch {epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, None, criterion, device)

    summary = {
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test": test_metrics,
        "num_classes": len(classes),
        "num_samples": len(labels),
        "max_epochs": args.epochs,
        "input_dim": edge_sets[0].shape[1],
        "config": {
            "xy_num_freqs": args.xy_num_freqs,
            "length_fourier": args.length_fourier,
            "length_num_freqs": args.length_num_freqs,
            "second_harmonic": args.second_harmonic,
            "use_endpoints": args.use_endpoints,
            "num_heads": args.num_heads,
            "num_encoder_blocks": args.num_encoder_blocks,
            "num_decoder_blocks": args.num_decoder_blocks,
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("best_epoch:", best_epoch)
    print("test:", test_metrics)


if __name__ == "__main__":
    main()
