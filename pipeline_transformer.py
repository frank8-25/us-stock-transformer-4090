"""Small temporal Transformer with chronological holdouts and purged label boundaries."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pipeline_documents import save_json, write_csv


class PriceDirectionTransformer(nn.Module):
    def __init__(self, inputs, lookback, width=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads.")
        self.project = nn.Linear(inputs, width)
        self.position = nn.Parameter(torch.randn(1, lookback, width) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=width*2, dropout=dropout,
                                       batch_first=True), num_layers=layers, enable_nested_tensor=False)
        # Independently initialize cloned encoder layers.
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        self.classifier = nn.Linear(width, 1)

    def forward(self, inputs):
        sequence = self.project(inputs) + self.position[:, :inputs.shape[1]]
        return self.classifier(self.encoder(sequence)[:, -1]).squeeze(-1)


def temporal_samples(dataset, columns, lookback):
    if lookback < 1:
        raise ValueError("lookback must be positive.")
    if not columns or any(c.startswith("target_") or c in {"label", "future_return", "date"} for c in columns):
        raise ValueError("Explicit feature columns must exclude targets and dates.")
    data = dataset.copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data.date, errors="raise")
    data["target_end"] = pd.to_datetime(data.target_end, errors="coerce")
    if data.date.duplicated().any():
        raise ValueError("Duplicate feature dates.")
    matrix = data[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature values must be finite.")
    labels = pd.to_numeric(data.label, errors="coerce")
    indices = np.array([i for i in range(lookback-1, len(data))
                        if labels.iloc[i] in [0, 1] and pd.notna(data.target_end.iloc[i])], dtype=int)
    if len(indices) < 20:
        raise ValueError("Need at least 20 labeled sequences for chronological train/validation/test splits.")
    val_start = int(len(indices)*0.70)
    test_start = int(len(indices)*0.85)
    val_cutoff = data.date.iloc[indices[val_start]]
    test_cutoff = data.date.iloc[indices[test_start]]
    train = indices[:val_start]
    validation = indices[val_start:test_start]
    test = indices[test_start:]
    train = train[(data.target_end.iloc[train] < val_cutoff).to_numpy()]
    validation = validation[(data.target_end.iloc[validation] < test_cutoff).to_numpy()]
    if len(train) < 4 or len(validation) < 2 or len(test) < 2:
        raise ValueError("Insufficient samples after purging overlapping target horizons.")
    train_rows = np.unique(np.concatenate([np.arange(i-lookback+1, i+1) for i in train]))
    mean = matrix[train_rows].mean(axis=0)
    std = matrix[train_rows].std(axis=0)
    std[std < 1e-8] = 1
    scaled = (matrix - mean) / std
    groups = {}
    for name, selected in [("train", train), ("validation", validation), ("test", test)]:
        x = np.stack([scaled[i-lookback+1:i+1] for i in selected])
        y = labels.iloc[selected].to_numpy(dtype=np.float32)
        groups[name] = (selected, x, y)
    return data, groups, mean, std


def classification_metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    labels = labels.astype(int)
    recalls = [float((predictions[labels == c] == c).mean()) for c in (0, 1) if (labels == c).any()]
    return {"rows": len(labels), "accuracy": float((predictions == labels).mean()),
            "balanced_accuracy_present_classes": float(np.mean(recalls)),
            "confusion_matrix_actual_by_predicted": [
                [int(((labels == a) & (predictions == p)).sum()) for p in (0, 1)] for a in (0, 1)]}


def train_transformer(dataset, columns, output, lookback=8, epochs=20, batch_size=32,
                      learning_rate=1e-3, device="cpu", seed=42, width=32):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    data, groups, mean, std = temporal_samples(dataset, columns, lookback)
    model = PriceDirectionTransformer(len(columns), lookback, width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    tensors = {name: (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
               for name, (_, x, y) in groups.items()}
    best, state, history = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        x, y = tensors["train"]
        losses = []
        for start in range(0, len(x), batch_size):
            optimizer.zero_grad()
            loss = criterion(model(x[start:start+batch_size]), y[start:start+batch_size])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vx, vy = tensors["validation"]
            validation_loss = float(criterion(model(vx), vy).cpu())
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss < best:
            best = validation_loss
            state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    model.eval()
    majority = int(groups["train"][2].mean() >= 0.5)
    predictions, metrics = [], {}
    with torch.no_grad():
        for name, (indices, _, labels) in groups.items():
            probs = torch.sigmoid(model(tensors[name][0])).cpu().numpy()
            metrics[name] = classification_metrics(labels, probs)
            metrics[name]["majority_baseline_accuracy"] = float((labels == majority).mean())
            for index, label, probability in zip(indices, labels, probs):
                predictions.append({"date": data.date.iloc[index].strftime("%Y-%m-%d"), "split": name,
                                    "actual_label": int(label), "predicted_label": int(probability >= 0.5),
                                    "prediction_prob": float(probability)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                  "feature_columns": columns, "mean": mean.tolist(), "std": std.tolist(),
                  "lookback": lookback, "width": width, "heads": 4, "layers": 2, "seed": seed}
    torch.save(checkpoint, output / "transformer.pt")
    write_csv(pd.DataFrame(predictions), output / "transformer_predictions.csv")
    write_csv(pd.DataFrame(history), output / "training_history.csv")
    save_json(output / "metrics.json", metrics)
    save_json(output / "training_config.json",
              {"feature_columns": columns, "lookback": lookback, "epochs": epochs, "seed": seed,
               "device": str(device), "split": "70/15/15 chronological before horizon purging",
               "standardization": "training window rows only", "best_validation_loss": best,
               "mean": mean.tolist(), "std": std.tolist(),
               "limitations": "Research classification metrics only; no trading/backtest claim."})
    return metrics
