# src/training/concept_trainer.py
"""
Concept Bottleneck Training Module (Week 3)

This module trains a simple concept prediction head on top of
precomputed visual features from a vision backbone (e.g., BiomedCLIP).
It is used to learn clinically meaningful concepts that will later
feed into the neuro-symbolic reasoning layer (LTN / rule engine).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ConceptDataset(Dataset):
    """
    Minimal dataset wrapper for:
      - visual_features: Tensor [N, feature_dim]
      - concept_labels: Tensor [N, num_concepts] in {0,1}
    """

    def __init__(self, visual_features: torch.Tensor, concept_labels: torch.Tensor):
        assert visual_features.shape[0] == concept_labels.shape[0], \
            "Features and labels must have the same batch size."
        self.visual_features = visual_features
        self.concept_labels = concept_labels

    def __len__(self):
        return self.visual_features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.visual_features[idx]
        y = self.concept_labels[idx]
        return x, y


class ConceptHead(nn.Module):
    """
    Simple MLP head that maps visual features -> concept probabilities.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_concepts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Binary concepts → sigmoid for multi-label probabilities
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return probs


class ConceptTrainer:
    """
    Handles training & evaluation of the concept head.
    This will be used as the "concept bottleneck" before the logic layer.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y.float())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Returns simple metrics:
          - loss
          - mean concept accuracy (threshold 0.5)
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        correct = 0
        total_labels = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(x)
            loss = self.criterion(y_hat, y.float())

            total_loss += loss.item()
            total_batches += 1

            # multi-label accuracy (per concept, threshold 0.5)
            preds = (y_hat > 0.5).long()
            correct += (preds == y).sum().item()
            total_labels += y.numel()

        avg_loss = total_loss / max(total_batches, 1)
        acc = correct / max(total_labels, 1)

        return {"loss": avg_loss, "concept_accuracy": acc}

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)


if __name__ == "__main__":
    # Demo with random tensors (for sanity checking the pipeline)
    num_samples = 256
    feature_dim = 768
    num_concepts = 14

    # Fake features and labels (replace with real ones in practice)
    visual_feats = torch.randn(num_samples, feature_dim)
    concept_labels = torch.randint(0, 2, (num_samples, num_concepts))

    dataset = ConceptDataset(visual_feats, concept_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ConceptHead(feature_dim=feature_dim, hidden_dim=256, num_concepts=num_concepts)
    trainer = ConceptTrainer(model, lr=1e-3)

    for epoch in range(3):
        train_loss = trainer.train_epoch(loader)
        metrics = trainer.evaluate(loader)
        print(
            f"Epoch {epoch+1} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"concept_acc={metrics['concept_accuracy']:.4f}"
        )

    # Example checkpoint save
    trainer.save_checkpoint("concept_head.ckpt")
