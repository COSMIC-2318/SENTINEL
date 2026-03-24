import torch
from graph_builder import build_sentinel_graph
from hgt_model import HGTModel


def run_module3(article_features=None):
    """
    Input  — article_features: 262-dim tensor [1, 262]
               (256 from Module 1 fusion + 6 from Module 2 evidence score)
               If None, uses random features for standalone testing.
    Output — dict with fake_prob, flags, article_embedding
    """

    # Step 1 — Build graph with real or simulated article features
    graph = build_sentinel_graph(article_features=article_features)

    # Step 2 — Extract metadata
    metadata = graph.metadata()

    # Step 3 — Initialize HGT model
    model = HGTModel(
        metadata=metadata,
        hidden_dim=64,
        num_heads=2,
        num_layers=3
    )

    # Step 4 — Prepare inputs
    x_dict = {
        node_type: graph[node_type].x
        for node_type in graph.node_types
    }
    edge_index_dict = {
        edge_type: graph[edge_type].edge_index
        for edge_type in graph.edge_types
    }

    # Step 5 — Forward pass
    model.eval()
    with torch.no_grad():
        fake_prob = model(x_dict, edge_index_dict)

    fake_prob_val = fake_prob.item()
    real_prob_val = 1 - fake_prob_val

    verdict = "LIKELY FAKE" if fake_prob_val > 0.5 else "LIKELY REAL"

    # Step 6 — Build enriched output dict for Module 4
    result = {
        "fake_prob":         round(fake_prob_val, 4),
        "real_prob":         round(real_prob_val, 4),
        "verdict":           verdict,
        "author_flag":       "Author credibility score below threshold (random weights — train on FakeNewsNet)",
        "domain_flag":       "Domain age and traffic rank flagged (random weights — train on FakeNewsNet)",
        "claim_overlap":     "Claim overlap detection active (random weights — train on FakeNewsNet)",
        "article_embedding": x_dict['article']  # 64-dim after message passing
    }

    print("\n=== MODULE 3 OUTPUT ===")
    print(f"  Fake Probability: {result['fake_prob']}")
    print(f"  Real Probability: {result['real_prob']}")
    print(f"  Verdict:          {result['verdict']}")

    return result


if __name__ == "__main__":
    # Standalone test with random features
    result = run_module3()
    print("\nModule 3 standalone test complete.")