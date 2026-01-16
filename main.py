import json
import yaml
import torch
import time
from tqdm import tqdm

from utils.seed import set_seed
from data.load_data import load_domain
from data.split import split_graphs
from models.gnn import GNN
from models.client_alpha import ClientAlpha
from trainers.local_trainer import train_one_epoch, evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_one_seed(cfg, seed):
    set_seed(seed)

    results = {}

    domains = cfg["data"]["domains"]
    local_epochs = cfg["train"]["local_epochs"]

    # ---------- domain loop ----------
    for domain in tqdm(domains, desc=f"[Seed {seed}] Domains", leave=False):
        domain_start = time.time()

        # load & split data
        graphs = load_domain(domain)
        train_g, val_g, test_g = split_graphs(
            graphs,
            seed=seed,
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
        )

        num_classes = len(set(g.y.item() for g in graphs))

        # build model
        model = GNN(
            in_dim=1,  # unified constant feature
            hidden_dim=cfg["model"]["hidden_dim"],
            num_classes=num_classes,
        ).to(DEVICE)

        alpha = ClientAlpha(
            M=cfg["model"]["bandpass_M"]
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(alpha.parameters()),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )

        # ---------- local training ----------
        epoch_bar = tqdm(
            range(local_epochs),
            desc=f"  {domain} | Training",
            leave=False,
        )

        for epoch in epoch_bar:
            loss = train_one_epoch(
                train_g,
                model,
                alpha,
                optimizer,
                DEVICE,
                spectral_k=cfg["model"]["spectral_k"],
            )
            epoch_bar.set_postfix(loss=f"{loss:.4f}")

        # ---------- evaluation ----------
        acc = evaluate(
            test_g,
            model,
            alpha,
            DEVICE,
            spectral_k=cfg["model"]["spectral_k"],
        )

        elapsed = time.time() - domain_start
        results[domain] = acc

        tqdm.write(
            f"[Seed {seed}] {domain} Test Acc: {acc:.4f} "
            f"(time: {elapsed/60:.1f} min)"
        )

    return results


def main():
    # ---------- load config (Windows-safe) ----------
    with open("configs/multidomain.yaml", "rb") as f:
        content = f.read().decode("utf-8", errors="ignore")
    cfg = yaml.safe_load(content)

    all_results = {}

    seeds = cfg["experiment"]["seeds"]
    overall_start = time.time()

    # ---------- seed loop ----------
    for seed in tqdm(seeds, desc="Seeds"):
        seed_res = run_one_seed(cfg, seed)
        all_results[f"seed_{seed}"] = seed_res

        with open(f"results/seed_{seed}.json", "w") as f:
            json.dump(seed_res, f, indent=2)

    # ---------- summary ----------
    summary = {}
    for domain in cfg["data"]["domains"]:
        vals = [
            all_results[f"seed_{s}"][domain]
            for s in seeds
        ]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        summary[domain] = {"mean": mean, "std": std}

    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - overall_start

    print("\n===== Summary =====")
    for d, v in summary.items():
        print(f"{d}: {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
