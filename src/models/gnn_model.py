"""
Day 4 — GNN Model: AST-based Graph Neural Network for defect prediction.
Converts Python source code → AST → PyTorch Geometric graph → 32-dim embedding.
"""

from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

from configs.config import FEATURES, MLFLOW, MODEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AST_NODE_TYPES: list[str] = FEATURES["ast_node_types"]
NODE_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(AST_NODE_TYPES)}
NUM_NODE_TYPES: int = len(AST_NODE_TYPES)  # 38
NODE_FEATURE_DIM: int = NUM_NODE_TYPES + 2  # 40  (one-hot + depth + n_children)

GNN_CFG = MODEL["gnn"]
HIDDEN_DIM: int = GNN_CFG["hidden_dim"]       # 64
EMBEDDING_DIM: int = GNN_CFG["embedding_dim"]  # 32
NUM_LAYERS: int = GNN_CFG["num_layers"]        # 3
DROPOUT: float = GNN_CFG["dropout"]            # 0.3
EPOCHS: int = GNN_CFG["epochs"]                # 100
LR: float = GNN_CFG["lr"]                      # 0.001
BATCH_SIZE: int = GNN_CFG["batch_size"]        # 32


# ---------------------------------------------------------------------------
# AST → PyG graph
# ---------------------------------------------------------------------------

def _node_type_onehot(node: ast.AST) -> list[float]:
    """38-dim one-hot over known node types; all zeros for unknown."""
    vec = [0.0] * NUM_NODE_TYPES
    type_name = type(node).__name__
    if type_name in NODE_TYPE_TO_IDX:
        vec[NODE_TYPE_TO_IDX[type_name]] = 1.0
    return vec


def _collect_nodes_edges(
    node: ast.AST,
    parent_idx: Optional[int],
    depth: int,
    node_list: list,
    edge_src: list[int],
    edge_dst: list[int],
) -> None:
    """DFS traversal — populates node_list and edge lists in-place."""
    current_idx = len(node_list)
    children = list(ast.iter_child_nodes(node))
    node_list.append((node, depth, len(children)))

    if parent_idx is not None:
        # parent → child
        edge_src.append(parent_idx)
        edge_dst.append(current_idx)
        # child → parent (bidirectional)
        edge_src.append(current_idx)
        edge_dst.append(parent_idx)

    for child in children:
        _collect_nodes_edges(child, current_idx, depth + 1, node_list, edge_src, edge_dst)


def file_to_pyg_graph(source_code: str) -> Optional[Data]:
    """
    Parse Python source → AST → PyG Data object.

    Node features (per AST node):
      - one-hot over FEATURES["ast_node_types"] (38 dims)
      - depth in tree  (1 dim, normalised 0-1)
      - number of children (1 dim, normalised 0-1)
    Total node feature dim: 40

    Edges: bidirectional parent ↔ child.

    Returns None for: syntax errors, empty source, < 5 nodes, > 5000 nodes.
    """
    if not source_code or not source_code.strip():
        return None

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None

    node_list: list = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    _collect_nodes_edges(tree, None, 0, node_list, edge_src, edge_dst)

    n_nodes = len(node_list)
    if n_nodes < 5 or n_nodes > 5000:
        return None

    max_depth = max(depth for _, depth, _ in node_list) or 1
    max_children = max(nc for _, _, nc in node_list) or 1

    features: list[list[float]] = []
    for node, depth, n_children in node_list:
        onehot = _node_type_onehot(node)
        norm_depth = depth / max_depth
        norm_children = n_children / max_children
        features.append(onehot + [norm_depth, norm_children])

    x = torch.tensor(features, dtype=torch.float)  # [N, 40]

    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)  # [2, E]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, num_nodes=n_nodes)


# ---------------------------------------------------------------------------
# GNN architecture
# ---------------------------------------------------------------------------

class CodeGNN(nn.Module):
    """
    3-layer GCN for file-level defect prediction.

    Architecture:
        GCNConv(40, 64) → BatchNorm → ReLU → Dropout(0.3)
        GCNConv(64, 64) → BatchNorm → ReLU → Dropout(0.3)
        GCNConv(64, 32) → global_mean_pool
        → embedding [B, 32]

    Classification head (training):
        Linear(32, 16) → ReLU → Linear(16, 1) → Sigmoid
    """

    def __init__(self) -> None:
        super().__init__()
        self.return_embedding = False

        # GCN layers
        self.conv1 = GCNConv(NODE_FEATURE_DIM, HIDDEN_DIM)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)

        self.conv2 = GCNConv(HIDDEN_DIM, HIDDEN_DIM)
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM)

        self.conv3 = GCNConv(HIDDEN_DIM, EMBEDDING_DIM)

        self.dropout = nn.Dropout(DROPOUT)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)

        # Global pooling → graph-level embedding
        embedding = global_mean_pool(x, batch)  # [B, 32]

        if self.return_embedding:
            return embedding

        return self.classifier(embedding)  # [B, 1]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GNNTrainer:
    """Wraps CodeGNN with dataset building, training, and embedding extraction."""

    def __init__(self, model: CodeGNN, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    # ------------------------------------------------------------------
    def build_dataset(
        self,
        source_codes: dict[str, str],
        labels: dict[str, int],
    ) -> list[Data]:
        """
        source_codes: {file_path: source_code_string}
        labels:       {file_path: 0 or 1}
        Returns list of PyG Data objects with .y set.
        Files that fail parsing are skipped entirely.
        """
        dataset: list[Data] = []
        skipped = 0

        for file_path, source in source_codes.items():
            if file_path not in labels:
                skipped += 1
                continue
            graph = file_to_pyg_graph(source)
            if graph is None:
                skipped += 1
                continue
            graph.y = torch.tensor([labels[file_path]], dtype=torch.float)
            graph.file_path = file_path
            dataset.append(graph)

        logger.info(
            f"Dataset built: {len(dataset)} parseable files, {skipped} skipped."
        )
        return dataset

    # ------------------------------------------------------------------
    def train(self, dataset: list[Data], epochs: int = EPOCHS) -> list[float]:
        """
        Train GNN as binary classifier.
        Returns list of per-epoch losses.
        Uses BCELoss + Adam. Logs to MLflow every 10 epochs.
        """
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        criterion = nn.BCELoss()

        self.model.return_embedding = False
        self.model.train()

        epoch_losses: list[float] = []

        mlflow.set_tracking_uri(MLFLOW["tracking_uri"])
        mlflow.set_experiment(MLFLOW["experiment_name"])

        with tqdm(range(epochs), desc="GNN Training", unit="epoch") as pbar:
            for epoch in pbar:
                total_loss = 0.0
                n_batches = 0

                for batch in loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out.squeeze(-1), batch.y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

                mean_loss = total_loss / max(n_batches, 1)
                epoch_losses.append(mean_loss)
                pbar.set_postfix(loss=f"{mean_loss:.4f}")

                if (epoch + 1) % 10 == 0:
                    try:
                        mlflow.log_metric("gnn_train_loss", mean_loss, step=epoch)
                    except Exception:
                        pass  # MLflow not active during standalone GNN train calls

        logger.info(f"GNN training complete. Final loss: {epoch_losses[-1]:.4f}")
        return epoch_losses

    # ------------------------------------------------------------------
    def get_embeddings(self, source_codes: dict[str, str]) -> dict[str, np.ndarray]:
        """
        Returns {file_path: array[32]} for every file.
        Files that fail parsing → np.zeros(32).
        """
        self.model.return_embedding = True
        self.model.eval()

        embeddings: dict[str, np.ndarray] = {}
        failed = 0

        graphs: list[tuple[str, Data]] = []
        for file_path, source in source_codes.items():
            graph = file_to_pyg_graph(source)
            if graph is None:
                embeddings[file_path] = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                failed += 1
            else:
                graphs.append((file_path, graph))

        # Batch inference
        if graphs:
            loader = DataLoader([g for _, g in graphs], batch_size=BATCH_SIZE, shuffle=False)
            file_paths_ordered = [fp for fp, _ in graphs]

            all_embeddings: list[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    emb = self.model(batch.x, batch.edge_index, batch.batch)
                    all_embeddings.append(emb.cpu().numpy())

            stacked = np.vstack(all_embeddings)  # [N_parseable, 32]
            for fp, emb in zip(file_paths_ordered, stacked):
                embeddings[fp] = emb

        logger.info(
            f"Embeddings extracted: {len(graphs)} files embedded, {failed} used zero-vector fallback."
        )
        self.model.return_embedding = False
        return embeddings

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"GNN model saved to {path}")

    def load(self, path: Path) -> None:
        path = Path(path)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"GNN model loaded from {path}")