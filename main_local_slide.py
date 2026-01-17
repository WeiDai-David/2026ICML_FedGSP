# main.py
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
from models.bandpass import SlidingBandpassDictionary

from trainers.local_trainer_slide import train_one_epoch, evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_one_seed(cfg, seed):
    set_seed(seed)
    results = {}

    domains = cfg["data"]["domains"]
    local_epochs = cfg["train"]["local_epochs"]

    # 只跑 sliding band-pass local
    methods = {
        "local_sliding": True,
    }

    for method_name, use_spectral in methods.items():
        tqdm.write(f"\n[Seed {seed}] Method: {method_name}")
        results[method_name] = {}

        for domain in tqdm(domains, desc=f"[{method_name}] Domains", leave=False):
            domain_start = time.time()

            # 1. load & split dataset
            graphs = load_domain(domain)

            # --- debug: check node features ---
            g0 = graphs[0]
            print(f"\n[DEBUG] {domain}")
            print("  x is None:", g0.x is None)
            if g0.x is not None:
                print("  x dtype/shape:", g0.x.dtype, tuple(g0.x.shape))
                if g0.x.dim() == 2:
                    print("  x unique rows:", torch.unique(g0.x, dim=0).size(0))
            print("  y unique:", torch.unique(torch.stack([g.y for g in graphs])).tolist())

            train_g, val_g, test_g = split_graphs(
                graphs,
                seed=seed,
                train_ratio=cfg["data"]["train_ratio"],
                val_ratio=cfg["data"]["val_ratio"],
            )

            num_classes = len(set(g.y.item() for g in graphs))

            # 2. build model
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

            # 3. spectral modules
            alpha = ClientAlpha(
                M=cfg["model"]["bandpass_M"]
            ).to(DEVICE)

            bandpass = SlidingBandpassDictionary(
                M=cfg["model"]["bandpass_M"],
                lambda_max=2.0,
            ).to(DEVICE)

            # 4. optimizer (CRITICAL: include bandpass params)
            params = (
                list(model.parameters())
                + list(alpha.parameters())
                + list(bandpass.parameters())
            )

            optimizer = torch.optim.Adam(
                params,
                lr=cfg["train"]["lr"],
                weight_decay=cfg["train"]["weight_decay"],
            )

            # 5. training loop
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
                    bandpass,
                    optimizer,
                    DEVICE,
                    spectral_k=cfg["model"]["spectral_k"],
                    use_spectral=True,
                )
                epoch_bar.set_postfix(loss=f"{loss:.4f}")

            # 6. evaluation
            acc = evaluate(
                test_g,
                model,
                alpha,
                bandpass,
                DEVICE,
                spectral_k=cfg["model"]["spectral_k"],
                use_spectral=True,
            )

            elapsed = time.time() - domain_start
            results[method_name][domain] = acc

            tqdm.write(
                f"[Seed {seed}] {method_name} | {domain} "
                f"Acc: {acc:.4f} ({elapsed/60:.1f} min)"
            )

    return results


def main():
    # load config
    with open("configs/Local_SM.yaml", "rb") as f:
        content = f.read().decode("utf-8", errors="ignore")
    cfg = yaml.safe_load(content)

    all_results = {}
    seeds = cfg["experiment"]["seeds"]
    overall_start = time.time()

    # run seeds
    for seed in tqdm(seeds, desc="Seeds"):
        seed_res = run_one_seed(cfg, seed)
        all_results[f"seed_{seed}"] = seed_res

        with open(f"results/seed_{seed}.json", "w") as f:
            json.dump(seed_res, f, indent=2)

    # --------------------------------------------------
    # summary
    # --------------------------------------------------
    summary = {}
    methods = ["local_sliding"]

    for method in methods:
        summary[method] = {}
        for domain in cfg["data"]["domains"]:
            vals = [
                all_results[f"seed_{s}"][method][domain]
                for s in seeds
            ]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            summary[method][domain] = {"mean": mean, "std": std}

    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== Summary =====")
    for method, domains in summary.items():
        print(f"\nMethod: {method}")
        for domain, stats in domains.items():
            print(f"  {domain}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    total_time = time.time() - overall_start
    print(f"Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
