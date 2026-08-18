"""
Privacy-robustness tradeoff evaluation for processed Gowalla and Foursquare data.

Outputs one CSV containing:
1. Target privacy budget epsilon in {0.5, 1.0, 2.0, 5.0}, with the equivalent
   one-round RDP Gaussian noise multiplier.
2. Noise multiplier sensitivity sigma in {0.5, 1.0, 2.0}.

For every setting and dataset, both clean and 30% poisoned-client results are
computed for NDCG@10 and NDCG@20.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from improved_pctra_complete import ImprovedPCTRA, PCTRAConfig
from run_pctra_grid_search_processed import prepare_dataset
from run_pctra_processed_datasets import evaluate_model, set_seed


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

EPSILON_VALUES = [0.5, 1.0, 2.0, 5.0]
SIGMA_VALUES = [0.5, 1.0, 2.0]
RDP_ALPHA = 32
POISON_PERCENT = 30


def load_best_rows(path: Path) -> Dict[str, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Best-results CSV not found: {path}")
    summary = pd.read_csv(path)
    rows: Dict[str, pd.Series] = {}
    for dataset_key in DATASETS:
        mask = summary["dataset"].astype(str).str.strip().str.lower() == dataset_key
        if not mask.any():
            raise ValueError(f"No {dataset_key} row found in {path}")
        rows[dataset_key] = summary.loc[mask].iloc[0]
    return rows


def sigma_for_epsilon(epsilon: float, alpha: int = RDP_ALPHA) -> float:
    """Invert the core implementation's one-round epsilon formula."""
    return float(np.sqrt(alpha / (2.0 * epsilon)))


def make_attackers(n_clients: int, poison_percent: int, seed: int) -> Optional[List[int]]:
    count = int(round(n_clients * poison_percent / 100.0))
    if count <= 0:
        return None
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_clients, size=min(count, n_clients), replace=False).tolist())


def make_config(bundle: Dict, best_row: pd.Series, noise_multiplier: float, n_clients: int) -> PCTRAConfig:
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
        noise_multiplier=noise_multiplier,
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
        top_k_fraction=1.0,
        ablation_mode="full",
    )


def run_case(
    root: Path,
    dataset_key: str,
    best_row: pd.Series,
    setting_type: str,
    setting_value: float,
    poison_percent: int,
    n_clients: int,
    num_negatives: int,
    seed: int,
    device: str,
) -> Dict[str, object]:
    if setting_type == "epsilon":
        epsilon_target = float(setting_value)
        sigma = sigma_for_epsilon(epsilon_target)
    else:
        epsilon_target = float(RDP_ALPHA / (2.0 * setting_value**2))
        sigma = float(setting_value)

    run_seed = seed + int(round(setting_value * 100)) + poison_percent
    set_seed(run_seed)
    dataset = DATASETS[dataset_key]
    bundle = prepare_dataset(
        tune_path=root / dataset["tune"],
        test_path=root / dataset["test"],
        seed=seed,
        n_clients=n_clients,
        num_negatives=num_negatives,
    )
    config = make_config(bundle, best_row, sigma, n_clients)
    model = ImprovedPCTRA(config, device=device)
    attackers = make_attackers(n_clients, poison_percent, run_seed)
    validation_data = (
        torch.from_numpy(bundle["val_user"]).long().to(device),
        torch.from_numpy(bundle["val_item"]).long().to(device),
        torch.ones(len(bundle["val_user"]), dtype=torch.float32).to(device),
    )

    model.train_federated(
        client_data=bundle["client_data"],
        validation_data=validation_data,
        test_data=(bundle["test_user"], bundle["test_item"]),
        clients_with_attacks=attackers,
        verbose=False,
        poison_updates=True,
    )
    metrics = evaluate_model(model.global_model, bundle["eval_candidates"], device=device)

    return {
        "dataset": dataset["label"],
        "setting_type": setting_type,
        "setting_value": setting_value,
        "epsilon_target": epsilon_target,
        "epsilon_one_round_rdp": epsilon_target,
        "noise_multiplier_sigma": sigma,
        "poison_percent": poison_percent,
        "n_clients": n_clients,
        "n_attackers": 0 if attackers is None else len(attackers),
        "ndcg@10": metrics["ndcg@10"],
        "ndcg@20": metrics["ndcg@20"],
        "seed": run_seed,
        "best_training.local_epochs": best_row["training.local_epochs"],
        "best_training.learning_rate": best_row["training.learning_rate"],
        "best_preference.beta_Q": best_row["preference.beta_Q"],
        "best_preference.lambda_uncertainty": best_row["preference.lambda_uncertainty"],
        "best_preference.kappa": best_row["preference.kappa"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute privacy-robustness tradeoff results.")
    parser.add_argument("--output", default="grid_results/privacy_robustness_tradeoff.csv")
    parser.add_argument("--best-summary", default="grid_results_full/grid_best_by_dataset.csv")
    parser.add_argument("--n-clients", type=int, default=30)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available.")

    root = Path(__file__).resolve().parent
    best_rows = load_best_rows(root / args.best_summary)
    rows = []
    total = len(args.datasets) * (len(EPSILON_VALUES) + len(SIGMA_VALUES)) * 2
    completed = 0

    for dataset_key in args.datasets:
        for setting_type, values in [("epsilon", EPSILON_VALUES), ("sigma", SIGMA_VALUES)]:
            for setting_value in values:
                for poison_percent in [0, POISON_PERCENT]:
                    completed += 1
                    print(
                        f"[{completed}/{total}] {dataset_key} | {setting_type}={setting_value} | "
                        f"poison={poison_percent}%"
                    )
                    rows.append(
                        run_case(
                            root=root,
                            dataset_key=dataset_key,
                            best_row=best_rows[dataset_key],
                            setting_type=setting_type,
                            setting_value=setting_value,
                            poison_percent=poison_percent,
                            n_clients=args.n_clients,
                            num_negatives=args.num_negatives,
                            seed=args.seed,
                            device=args.device,
                        )
                    )

    result_df = pd.DataFrame(rows)
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Saved privacy-robustness results to: {output_path}")
    print(
        result_df.pivot_table(
            index=["dataset", "setting_type", "setting_value"],
            columns="poison_percent",
            values=["ndcg@10", "ndcg@20"],
        ).to_string()
    )


if __name__ == "__main__":
    main()
