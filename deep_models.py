"""
Small PyTorch models for the optional embedding-based branch.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import RANDOM_SEED
from embedding_features import ENTITY_COLUMNS


EMBED_DIMS = {
    "canonical_drug": 32,
    "primary_target": 32,
    "disease": 16,
    "modality": 8,
    "moa": 16,
}


def set_torch_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrialTensorDataset(Dataset):
    def __init__(self, entity_ids, biology, labels):
        self.entity_ids = torch.as_tensor(np.array(entity_ids, copy=True), dtype=torch.long)
        self.biology = torch.as_tensor(np.array(biology, copy=True), dtype=torch.float32)
        self.labels = torch.as_tensor(np.array(labels, copy=True), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.entity_ids[idx], self.biology[idx], self.labels[idx]


class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        biology_dim: int,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.25,
        use_entities: bool = True,
        use_biology: bool = True,
    ):
        super().__init__()
        self.columns = ENTITY_COLUMNS
        self.use_entities = use_entities
        self.use_biology = use_biology
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], EMBED_DIMS[col], padding_idx=0)
            for col in self.columns
        })
        input_dim = 0
        if use_entities:
            input_dim += sum(EMBED_DIMS[col] for col in self.columns)
        if use_biology:
            input_dim += biology_dim
        if input_dim == 0:
            raise ValueError("EmbeddingMLP requires entity embeddings, biology features, or both.")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, entity_ids, biology):
        pieces = []
        if self.use_entities:
            for i, col in enumerate(self.columns):
                pieces.append(self.embeddings[col](entity_ids[:, i]))
        if self.use_biology:
            pieces.append(biology)
        return self.net(torch.cat(pieces, dim=1)).squeeze(1)


class TrialTransformer(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        biology_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.columns = ENTITY_COLUMNS
        self.entity_embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], d_model, padding_idx=0)
            for col in self.columns
        })
        self.biology_projection = nn.Linear(biology_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_embeddings = nn.Embedding(len(self.columns) + 2, d_model)
        self.position_embeddings = nn.Embedding(len(self.columns) + 2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, entity_ids, biology):
        batch_size = entity_ids.shape[0]
        tokens = [self.cls_token.expand(batch_size, -1, -1)]
        for i, col in enumerate(self.columns):
            tokens.append(self.entity_embeddings[col](entity_ids[:, i]).unsqueeze(1))
        tokens.append(self.biology_projection(biology).unsqueeze(1))
        x = torch.cat(tokens, dim=1)
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        token_types = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = x + self.position_embeddings(positions) + self.type_embeddings(token_types)
        encoded = self.encoder(x)
        return self.classifier(encoded[:, 0, :]).squeeze(1)


def _pos_weight(labels: np.ndarray) -> torch.Tensor:
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    if positives == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


@dataclass
class TrainingResult:
    model: nn.Module
    best_epoch: int
    best_val_auc: float
    history: list[dict]


def classification_metrics(y_true, probs) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "roc_auc": round(float(auc), 6),
        "pr_auc": round(float(average_precision_score(y_true, probs)), 6),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 6),
        "brier": round(float(brier_score_loss(y_true, probs)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 6),
    }


def predict_probs(model, dataset, batch_size: int = 256, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for entity_ids, biology, _ in loader:
            logits = model(entity_ids.to(device), biology.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def train_model(
    model,
    train_dataset,
    val_dataset,
    train_labels,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int = RANDOM_SEED,
    device: str | None = None,
) -> TrainingResult:
    set_torch_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(train_labels).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = deepcopy(model.state_dict())
    best_auc = -np.inf
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for entity_ids, biology, labels in loader:
            entity_ids = entity_ids.to(device)
            biology = biology.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(entity_ids, biology)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_probs = predict_probs(model, val_dataset, batch_size=batch_size, device=device)
        val_labels = val_dataset.labels.numpy()
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            "val_auc": float(val_auc),
        })

        if val_auc > best_auc:
            best_auc = float(val_auc)
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(model=model, best_epoch=best_epoch, best_val_auc=best_auc, history=history)
