from torch_geometric.datasets import TUDataset

DOMAIN_DATASETS = {
    "SM": ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"],
    "BIO": ["PROTEINS", "ENZYMES"],
    "CV": ["Letter-low", "Letter-high", "Letter-med"],
    "SN": ["IMDB-BINARY", "IMDB-MULTI"]
}

def load_domain(domain, root="data/raw"):
    datasets = DOMAIN_DATASETS[domain]
    all_graphs = []
    for name in datasets:
        ds = TUDataset(root, name)
        all_graphs.extend(ds)
    return all_graphs
