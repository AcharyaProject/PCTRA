"""
PCTRA efficiency and scalability evaluation on processed Gowalla/Foursquare data.

Measures, for K in {10, 50, 100, 500}:
- Wall-clock time per federated round
- Client-to-server update communication volume
- Server-side memory for retained client updates
- Full aggregation versus 50% top-client selection

Results:
- grid_results/scalability_evaluation.csv
- grid_results/scalability_time_plot.png
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from improved_pctra_complete import ImprovedPCTRA, PCTRAConfig
from run_pctra_grid_search_processed import prepare_dataset
from run_pctra_processed_datasets import set_seed


CLIENT_COUNTS = [10, 50, 100, 500]
DATASETS = {
    "gowalla": {
        "label": "Gowalla",
        "tune": "processed/gowalla/Gowalla_tune_filtered.txt",
        "test": "processed/gowalla/Gowalla_test_filtered.txt",
    },
    "foursquare": {
        "label": "Foursquare",
        "tune": "processed/foursquare/Foursquare_tune_filtered.txt",
        "test": "processed/foursquare/Foursquare_test_filtered.txt",
    },
    "swarm": {
        "label": "Swarm",
        "tune": "processed/swarm/Swarm_tune_filtered.txt",
        "test": "processed/swarm/Swarm_test_filtered.txt",
    },
}


def load_best_rows(path: Path) -> Dict[str, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Best-results CSV not found: {path}")
    summary = pd.read_csv(path)
    rows: Dict[str, pd.Series] = {}
    for key in DATASETS:
        mask = summary["dataset"].astype(str).str.strip().str.lower() == key
        if not mask.any():
            raise ValueError(f"No {key} row found in {path}")
        rows[key] = summary.loc[mask].iloc[0]
    return rows


def make_config(bundle: Dict, best_row: pd.Series, n_clients: int, top_k_fraction: float) -> PCTRAConfig:
    return PCTRAConfig(
        n_users=bundle["n_users"],
        n_items=bundle["n_items"],
        embedding_dim=16,
        n_gcn_layers=1,
        n_clients=n_clients,
        n_rounds=1,
        local_epochs=int(float(best_row["training.local_epochs"])),
        batch_size=256,
        learning_rate=float(best_row["training.learning_rate"]),
        clipping_norm=1.0,
        noise_multiplier=0.2,
        delta=1e-5,
        lambda_B=0.3,
        lambda_M=0.25,
        lambda_Q=0.25,
        lambda_R=0.2,
        rho=2.0,
        eta=0.95,
        lambda_uncertainty=float(best_row["preference.lambda_uncertainty"]),
        beta_T=1.0,
        beta_Q=float(best_row["preference.beta_Q"]),
        beta_N=0.5,
        kappa=float(best_row["preference.kappa"]),
        top_k_fraction=top_k_fraction,
        ablation_mode="full",
    )


def measure_case(
    root: Path,
    dataset_key: str,
    best_row: pd.Series,
    n_clients: int,
    mode: str,
    top_k_fraction: float,
    num_negatives: int,
    seed: int,
    device: str,
) -> Dict[str, object]:
    set_seed(seed)
    dataset = DATASETS[dataset_key]
    bundle = prepare_dataset(
        tune_path=root / dataset["tune"],
        test_path=root / dataset["test"],
        seed=seed,
        n_clients=n_clients,
        num_negatives=num_negatives,
    )
    config = make_config(bundle, best_row, n_clients, top_k_fraction)
    model = ImprovedPCTRA(config, device=device)
    validation_data = (
        torch.from_numpy(bundle["val_user"]).long().to(device),
        torch.from_numpy(bundle["val_item"]).long().to(device),
        torch.ones(len(bundle["val_user"]), dtype=torch.float32).to(device),
    )

    parameter_count = sum(parameter.numel() for parameter in model.global_model.parameters())
    update_bytes_per_client = parameter_count * 8
    total_communication_bytes = update_bytes_per_client * n_clients
    update_memory_bytes = total_communication_bytes

    start = time.perf_counter()
    model.train_federated(
        client_data=bundle["client_data"],
        validation_data=validation_data,
        test_data=(bundle["test_user"], bundle["test_item"]),
        clients_with_attacks=None,
        verbose=False,
        poison_updates=False,
    )
    elapsed = time.perf_counter() - start

    selected_clients = int(round(n_clients * top_k_fraction))
    return {
        "dataset": dataset["label"],
        "client_count": n_clients,
        "aggregation_mode": mode,
        "top_k_fraction": top_k_fraction,
        "time_per_round_seconds": elapsed,
        "parameter_count": parameter_count,
        "update_bytes_per_client": update_bytes_per_client,
        "communication_bytes_per_round": total_communication_bytes,
        "communication_mb_per_round": total_communication_bytes / (1024 ** 2),
        "server_client_update_memory_mb": update_memory_bytes / (1024 ** 2),
        "selected_clients": selected_clients,
        "local_epochs": config.local_epochs,
        "device": device,
        "seed": seed,
    }


def save_plot(result_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for (dataset, mode), group in result_df.groupby(["dataset", "aggregation_mode"]):
        group = group.sort_values("client_count")
        ax.plot(
            group["client_count"],
            group["time_per_round_seconds"],
            marker="o",
            linewidth=2,
            label=f"{dataset} - {mode}",
        )
    ax.set_xlabel("Number of Clients (K)")
    ax.set_ylabel("Time per Federated Round (seconds)")
    ax.set_title("PCTRA Scalability: Round Time vs Number of Clients")
    ax.set_xticks(CLIENT_COUNTS)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scalability plot to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure PCTRA efficiency and scalability.")
    parser.add_argument("--output", default="grid_results/scalability_evaluation.csv")
    parser.add_argument("--plot-output", default="grid_results/scalability_time_plot.png")
    parser.add_argument("--best-summary", default="grid_results_full/grid_best_by_dataset.csv")
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--client-counts", nargs="+", type=int, default=CLIENT_COUNTS)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available.")

    root = Path(__file__).resolve().parent
    best_rows = load_best_rows(root / args.best_summary)
    rows: List[Dict[str, object]] = []
    total = len(args.datasets) * len(args.client_counts) * 2
    completed = 0

    for dataset_key in args.datasets:
        for n_clients in args.client_counts:
            for mode, fraction in [("full", 1.0), ("top_k_50_percent", 0.5)]:
                completed += 1
                print(f"[{completed}/{total}] {dataset_key} | K={n_clients} | {mode}")
                rows.append(
                    measure_case(
                        root=root,
                        dataset_key=dataset_key,
                        best_row=best_rows[dataset_key],
                        n_clients=n_clients,
                        mode=mode,
                        top_k_fraction=fraction,
                        num_negatives=args.num_negatives,
                        seed=args.seed,
                        device=args.device,
                    )
                )

    result_df = pd.DataFrame(rows)
    output_path = root / args.output
    plot_path = root / args.plot_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    save_plot(result_df, plot_path)
    print(f"Saved scalability results to: {output_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
