"""
PCTRA ablation evaluation on the processed Gowalla and Foursquare datasets.

The runner evaluates:
- PCTRA FULL
- PCTRA w/o Marginal Utility
- PCTRA w/o Uncertainty Quantification
- PCTRA w/o Risk-Adjusted Aggregation
- PCTRA w/o Adaptive Caps
- PCTRA w/o GCN / matrix-factorization mode

Poisoning is performed by negating the privacy-protected update of a fixed,
seeded fraction of clients before server aggregation. Results are written to
`grid_results/ablation_evaluation.csv` by default.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from improved_pctra_complete import ImprovedPCTRA, PCTRAConfig
from run_pctra_grid_search_processed import prepare_dataset
from run_pctra_processed_datasets import evaluate_model, set_seed


VARIANTS = {
    "full": "full",
    "without_marginal_utility": "without_marginal_utility",
    "without_uncertainty": "without_uncertainty",
    "without_risk_adjustment": "without_risk_adjustment",
    "without_adaptive_caps": "without_adaptive_caps",
    "without_gcn": "without_gcn",
}

VARIANT_LABELS = {
    "full": "PCTRA FULL",
    "without_marginal_utility": "PCTRA w/o Marginal Utility",
    "without_uncertainty": "PCTRA w/o Uncertainty Quantification",
    "without_risk_adjustment": "PCTRA w/o Risk-Adjusted Aggregation",
    "without_adaptive_caps": "PCTRA w/o Adaptive Caps",
    "without_gcn": "PCTRA w/o GCN (Matrix Factorization)",
}

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
    if "dataset" not in summary.columns:
        raise ValueError(f"Best-results CSV has no dataset column: {path}")

    rows: Dict[str, pd.Series] = {}
    for dataset_key in DATASETS:
        mask = summary["dataset"].astype(str).str.strip().str.lower() == dataset_key
        if not mask.any():
            raise ValueError(f"No {dataset_key} row found in {path}")
        rows[dataset_key] = summary.loc[mask].iloc[0]
    return rows


def make_attackers(n_clients: int, poison_percent: int, seed: int) -> Optional[List[int]]:
    n_attackers = int(round(n_clients * poison_percent / 100.0))
    if n_attackers <= 0:
        return None
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_clients, size=min(n_attackers, n_clients), replace=False).tolist())


def make_config(bundle: Dict, best_row: pd.Series, variant: str, n_clients: int) -> PCTRAConfig:
    return PCTRAConfig(
        n_users=bundle["n_users"],
        n_items=bundle["n_items"],
        embedding_dim=16,
        n_gcn_layers=0 if variant == "without_gcn" else 1,
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
        top_k_fraction=1.0,
        ablation_mode=VARIANTS[variant],
    )


def evaluate_one(
    root: Path,
    dataset_key: str,
    best_row: pd.Series,
    variant: str,
    poison_percent: int,
    n_clients: int,
    num_negatives: int,
    seed: int,
    device: str,
) -> Dict[str, object]:
    dataset = DATASETS[dataset_key]
    # Match initialization, minibatch sampling, and attacker selection across variants.
    run_seed = seed + poison_percent
    set_seed(run_seed)

    bundle = prepare_dataset(
        tune_path=root / dataset["tune"],
        test_path=root / dataset["test"],
        seed=seed,
        n_clients=n_clients,
        num_negatives=num_negatives,
    )
    config = make_config(bundle, best_row, variant, n_clients)
    model = ImprovedPCTRA(config, device=device)
    attackers = make_attackers(n_clients, poison_percent, run_seed)
    validation_data = (
        __import__("torch").from_numpy(bundle["val_user"]).long().to(device),
        __import__("torch").from_numpy(bundle["val_item"]).long().to(device),
        __import__("torch").ones(len(bundle["val_user"]), dtype=__import__("torch").float32).to(device),
    )

    history = model.train_federated(
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
        "variant": VARIANT_LABELS[variant],
        "variant_key": variant,
        "poison_percent": poison_percent,
        "n_clients": n_clients,
        "n_attackers": 0 if attackers is None else len(attackers),
        "ndcg@10": metrics["ndcg@10"],
        "ndcg@20": metrics["ndcg@20"],
        "recall@20": metrics["recall@20"],
        "precision@20": metrics["precision@20"],
        "acc@10": metrics["acc@10"],
        "seed": run_seed,
        "best_training.local_epochs": best_row["training.local_epochs"],
        "best_training.learning_rate": best_row["training.learning_rate"],
        "best_preference.beta_Q": best_row["preference.beta_Q"],
        "best_preference.lambda_uncertainty": best_row["preference.lambda_uncertainty"],
        "best_preference.kappa": best_row["preference.kappa"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PCTRA ablation robustness results.")
    parser.add_argument("--output", default="grid_results/ablation_evaluation.csv")
    parser.add_argument("--best-summary", default="grid_results_full/grid_best_by_dataset.csv")
    parser.add_argument("--n-clients", type=int, default=30)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--poison-levels", nargs="+", type=int, default=[0, 10, 20, 30, 50])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_path = root / args.output
    best_rows = load_best_rows(root / args.best_summary)

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available.")

    rows = []
    total = len(args.datasets) * len(args.variants) * len(args.poison_levels)
    completed = 0
    for dataset_key in args.datasets:
        for variant in args.variants:
            for poison_percent in args.poison_levels:
                completed += 1
                print(f"[{completed}/{total}] {dataset_key} | {VARIANT_LABELS[variant]} | poison={poison_percent}%")
                rows.append(
                    evaluate_one(
                        root=root,
                        dataset_key=dataset_key,
                        best_row=best_rows[dataset_key],
                        variant=variant,
                        poison_percent=poison_percent,
                        n_clients=args.n_clients,
                        num_negatives=args.num_negatives,
                        seed=args.seed,
                        device=args.device,
                    )
                )

    result_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Saved ablation results to: {output_path}")
    print(result_df.pivot_table(index=["dataset", "variant"], columns="poison_percent", values="ndcg@10").to_string())


if __name__ == "__main__":
    main()
