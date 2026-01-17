import json
import yaml
import time
import copy
import torch
from tqdm import tqdm

from utils.seed import set_seed
from data.load_data import load_domain
from data.split import split_graphs

from models.gnn import GNN
from models.client_alpha import ClientAlpha
from models.bandpass import SlidingBandpassDictionary

from trainers.local_trainer_slide import train_one_epoch, evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Federated aggregation: ONLY aggregate band-pass (u, v)
# =========================================================
def aggregate_bandpass(global_bandpass, client_bandpasses):
    with torch.no_grad():
        for name, param in global_bandpass.named_parameters():
            param.data.zero_()
            for bp in client_bandpasses:
                param.data += dict(bp.named_parameters())[name].data
            param.data /= len(client_bandpasses)


# =========================================================
# Main federated training
# =========================================================
def run_federated(cfg, seed):
    set_seed(seed)

    domains = cfg["data"]["domains"]
    R = cfg["train"]["round"]
    local_epochs = cfg["train"]["local_epochs"]

    # -----------------------------------------------------
    # Prepare clients (one domain = one client)
    # -----------------------------------------------------
    clients = {}

    for domain in domains:
        graphs = load_domain(domain)
        train_g, val_g, test_g = split_graphs(
            graphs,
            seed=seed,
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
        )

        num_classes = len(set(g.y.item() for g in graphs))

        sample_graph = graphs[0]

        if sample_graph.x is None:
            in_dim = 1
        else:
            in_dim = sample_graph.x.size(1)

        model = GNN(
            in_dim=in_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            num_classes=num_classes,
        ).to(DEVICE)

        alpha = ClientAlpha(
            M=cfg["model"]["bandpass_M"]
        ).to(DEVICE)

        clients[domain] = {
            "train": train_g,
            "test": test_g,
            "model": model,
            "alpha": alpha,
        }

    # -----------------------------------------------------
    # Initialize GLOBAL band-pass dictionary (shared)
    # -----------------------------------------------------
    global_bandpass = SlidingBandpassDictionary(
        M=cfg["model"]["bandpass_M"]
    ).to(DEVICE)

    # -----------------------------------------------------
    # Federated rounds
    # -----------------------------------------------------
    round_bar = tqdm(range(R), desc=f"[Seed {seed}] Federated Rounds")

    for r in round_bar:
        client_bandpasses = []

        for domain, client in clients.items():
            # ---- local copy of global band-pass ----
            local_bandpass = copy.deepcopy(global_bandpass)

            optimizer = torch.optim.Adam(
                list(client["model"].parameters())
                + list(client["alpha"].parameters())
                + list(local_bandpass.parameters()),
                lr=cfg["train"]["lr"],
                weight_decay=cfg["train"]["weight_decay"],
            )

            # ---- local training ----
            for _ in range(local_epochs):
                train_one_epoch(
                    client["train"],
                    client["model"],
                    client["alpha"],
                    local_bandpass,
                    optimizer,
                    DEVICE,
                    spectral_k=cfg["model"]["spectral_k"],
                )

            client_bandpasses.append(local_bandpass)

        # ---- server aggregation (FedAvg on band-pass) ----
        aggregate_bandpass(global_bandpass, client_bandpasses)

    # -----------------------------------------------------
    # Final evaluation (with global band-pass)
    # -----------------------------------------------------
    results = {}

    for domain, client in clients.items():
        acc = evaluate(
            client["test"],
            client["model"],
            client["alpha"],
            global_bandpass,
            DEVICE,
            spectral_k=cfg["model"]["spectral_k"],
        )
        results[domain] = acc
        print(f"[Seed {seed}] {domain} Acc: {acc:.4f}")

    return results


# =========================================================
# Entry
# =========================================================
def main():
    with open("configs/SM_BIO_SN.yaml", "rb") as f:
        cfg = yaml.safe_load(f.read().decode("utf-8", errors="ignore"))

    all_results = {}
    seeds = cfg["experiment"]["seeds"]

    start = time.time()

    for seed in seeds:
        res = run_federated(cfg, seed)
        all_results[f"seed_{seed}"] = res

        with open(f"results/federated_seed_{seed}.json", "w") as f:
            json.dump(res, f, indent=2)

    # -------- summary --------
    summary = {}
    for domain in cfg["data"]["domains"]:
        vals = [all_results[f"seed_{s}"][domain] for s in seeds]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        summary[domain] = {"mean": mean, "std": std}

    with open("results/federated_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== Federated Summary =====")
    for d, v in summary.items():
        print(f"{d}: {v['mean']:.4f} ± {v['std']:.4f}")

    print(f"Total time: {(time.time() - start)/60:.1f} min")


if __name__ == "__main__":
    main()
