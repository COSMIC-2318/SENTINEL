import torch
from torch_geometric.data import HeteroData


def build_sentinel_graph(article_features=None):
    """
    Builds a heterogeneous graph for one article.
    
    article_features — optional 262-dim tensor from pipeline
                       (256 from Module 1 + 6 from Module 2)
                       If None, uses random features for standalone testing.
    """

    graph = HeteroData()

    # Article node — use real features if provided, else random
    if article_features is not None:
        graph['article'].x = article_features  # real [1, 262] tensor
    else:
        graph['article'].x = torch.randn(1, 262)  # standalone test

    # Author node
    graph['author'].x = torch.randn(1, 4)

    # Domain node
    graph['domain'].x = torch.randn(1, 4)

    # Claim nodes
    graph['claim'].x = torch.randn(3, 768)

    # Entity nodes
    graph['entity'].x = torch.randn(2, 3)

    # Forward edges
    graph['article', 'authored_by', 'author'].edge_index  = torch.tensor([[0],[0]])
    graph['article', 'published_on', 'domain'].edge_index = torch.tensor([[0],[0]])
    graph['article', 'makes_claim',  'claim'].edge_index  = torch.tensor([[0,0,0],[0,1,2]])
    graph['article', 'mentions_entity', 'entity'].edge_index = torch.tensor([[0,0],[0,1]])

    # Reverse edges
    graph['author', 'wrote',       'article'].edge_index = torch.tensor([[0],[0]])
    graph['domain', 'hosts',       'article'].edge_index = torch.tensor([[0],[0]])
    graph['claim',  'claimed_by',  'article'].edge_index = torch.tensor([[0,1,2],[0,0,0]])
    graph['entity', 'featured_in', 'article'].edge_index = torch.tensor([[0,1],[0,0]])

    return graph