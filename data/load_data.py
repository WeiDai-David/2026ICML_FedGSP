from torch_geometric.datasets import TUDataset


def load_domain(domain, root="data/raw"):
    """
    Load graphs for a single client.

    domain: dataset name, e.g.
        "MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",
        "PROTEINS", "ENZYMES",
        "IMDB-BINARY", "IMDB-MULTI", etc.

    return:
        List[torch_geometric.data.Data]
    """
    dataset = TUDataset(root=root, name=domain)
    return list(dataset)
