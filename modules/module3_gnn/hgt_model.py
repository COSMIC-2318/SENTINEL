import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


class HGTModel(nn.Module):
    def __init__(self, metadata, hidden_dim=64, num_heads=2, num_layers=3):
        """
        metadata    — the graph's node types and edge types (from graph_builder)
        hidden_dim  — all nodes projected to this size (64)
        num_heads   — attention heads (2)
        num_layers  — rounds of message passing (3)
        """
        super().__init__()

        # ─────────────────────────────────────────
        # STEP 1 — Linear projection layers
        # Project every node type to the same hidden_dim (64)
        # Each node type gets its own projection layer
        # ─────────────────────────────────────────
        self.projections = nn.ModuleDict()

        node_feature_sizes = {
            'article': 262,
            'author':  4,
            'domain':  4,
            'claim':   768,
            'entity':  3
        }

        for node_type, feature_size in node_feature_sizes.items():
            self.projections[node_type] = Linear(feature_size, hidden_dim)

        # ─────────────────────────────────────────
        # STEP 2 — HGT Convolution layers
        # One HGTConv layer = one round of message passing
        # We stack num_layers (3) of them
        # ─────────────────────────────────────────
        self.hgt_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.hgt_layers.append(
                HGTConv(
                    in_channels=hidden_dim,   # input size (all nodes are 64 after projection)
                    out_channels=hidden_dim,  # output size (keep at 64)
                    metadata=metadata,        # tells HGT what node/edge types exist
                    heads=num_heads           # attention heads
                )
            )

        # ─────────────────────────────────────────
        # STEP 3 — Final classification head
        # Takes the enriched Article embedding (64-dim)
        # Outputs P(Fake) — a single probability
        # ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_dict, edge_index_dict):
        """
        x_dict          — dictionary of node features {node_type: feature_tensor}
        edge_index_dict — dictionary of edges {edge_type: edge_index_tensor}
        """

        # Step 1 — Project all node types to hidden_dim (64)
        projected = {}
        for node_type, features in x_dict.items():
            projected[node_type] = self.projections[node_type](features)

        # Step 2 — Run 3 rounds of HGT message passing
        for hgt_layer in self.hgt_layers:
            projected = hgt_layer(projected, edge_index_dict)

        # Step 3 — Take the Article node's final embedding
        # and pass through classifier to get P(Fake)
        article_embedding = projected['article']
        fake_probability = self.classifier(article_embedding)

        return fake_probability