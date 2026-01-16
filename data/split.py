import random

def split_graphs(graphs, seed=0, train_ratio=0.8, val_ratio=0.1):
    """
    graphs: List[Data]
    return: train_graphs, val_graphs, test_graphs
    """
    assert train_ratio + val_ratio < 1.0

    num_graphs = len(graphs)
    indices = list(range(num_graphs))

    random.Random(seed).shuffle(indices)

    train_end = int(train_ratio * num_graphs)
    val_end = int((train_ratio + val_ratio) * num_graphs)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    return train_graphs, val_graphs, test_graphs
