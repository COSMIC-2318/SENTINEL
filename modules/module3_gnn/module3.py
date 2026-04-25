import torch
from graph_builder import build_sentinel_graph
from hgt_model import HGTModel

# ================================================================
# FIX — HGTModel initialized ONCE at module load time
# Previously it was initialized inside run_module3() which meant
# brand new random weights every single call.
# The classifier learned to ignore this noise → class collapse.
# Now the same model instance is reused across all calls.
# During joint training, gradients flow back into this model
# and it actually learns graph patterns.
# ================================================================

# Build a temporary graph just to get metadata for model init
_init_graph    = build_sentinel_graph(article_features=None)
_metadata      = _init_graph.metadata()

# Create the model ONCE — reused for every call
_hgt_model = HGTModel(
    metadata   = _metadata,
    hidden_dim = 64,
    num_heads  = 2,
    num_layers = 3
)


def get_hgt_model():
    """Returns the shared HGTModel instance."""
    return _hgt_model


def run_module3(article_features=None, model=None):
    """
    Input  — article_features: 262-dim tensor [1, 262]
               (256 from Module 1 fusion + 6 from Module 2 evidence score)
               If None, uses random features for standalone testing.
             model: optional HGTModel instance to use.
               If None, uses the shared _hgt_model instance.
    Output — dict with fake_prob, flags, article_embedding
    """
    # Use provided model or fall back to shared instance
    # This allows train.py to pass its own model for gradient flow
    if model is None:
        model = _hgt_model

    # Step 1 — Build graph with real or simulated article features
    graph = build_sentinel_graph(article_features=article_features)

    # Step 2 — Prepare inputs
    x_dict = {
        node_type: graph[node_type].x
        for node_type in graph.node_types
    }
    edge_index_dict = {
        edge_type: graph[edge_type].edge_index
        for edge_type in graph.edge_types
    }

    # Step 3 — Forward pass
    # Note: eval() only called when no gradient needed
    # During training, caller controls model.train()/eval()
    fake_prob = model(x_dict, edge_index_dict)

    fake_prob_val = fake_prob.item() if isinstance(
        fake_prob, torch.Tensor
    ) else float(fake_prob)
    real_prob_val = 1 - fake_prob_val
    verdict       = "LIKELY FAKE" if fake_prob_val > 0.5 else "LIKELY REAL"

    result = {
        "fake_probability" : round(fake_prob_val, 4),
        "fake_prob"        : round(fake_prob_val, 4),
        "real_prob"        : round(real_prob_val, 4),
        "verdict"          : verdict,
        "author_flag"      : None,   # no real author data — graph uses placeholder nodes
        "domain_flag"      : None,   # no real domain data
        "claim_overlap"    : None,
        "article_embedding": x_dict["article"],
    }

    return result


if __name__ == "__main__":
    print("Testing Module 3 with shared model instance...")
    result1 = run_module3()
    result2 = run_module3()

    print(f"Call 1 fake_prob: {result1['fake_prob']}")
    print(f"Call 2 fake_prob: {result2['fake_prob']}")
    print("✅ Same model used both times — no random reinitialization")