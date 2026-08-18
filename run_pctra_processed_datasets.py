import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from improved_pctra_complete import ImprovedPCTRA, PCTRAConfig
from pctra_data_and_metrics import DataSplitter


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_interactions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1])
    user_ids = data[0].astype(np.int64).to_numpy()
    item_ids = data[1].astype(np.int64).to_numpy()
    return user_ids, item_ids


def remap_ids(
    train_user: np.ndarray,
    train_item: np.ndarray,
    test_user: np.ndarray,
    test_item: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    all_users = np.concatenate([train_user, test_user])
    _, user_inverse = np.unique(all_users, return_inverse=True)

    all_items = np.concatenate([train_item, test_item])
    _, item_inverse = np.unique(all_items, return_inverse=True)

    train_len = len(train_user)
    test_len = len(test_user)

    remap_train_user = user_inverse[:train_len]
    remap_test_user = user_inverse[train_len : train_len + test_len]

    remap_train_item = item_inverse[:train_len]
    remap_test_item = item_inverse[train_len : train_len + test_len]

    n_users = int(user_inverse.max()) + 1
    n_items = int(item_inverse.max()) + 1

    return (
        remap_train_user,
        remap_train_item,
        remap_test_user,
        remap_test_item,
        n_users,
        n_items,
    )


def build_user_item_dict(user_ids: np.ndarray, item_ids: np.ndarray) -> Dict[int, set]:
    result: Dict[int, set] = {}
    for u, i in zip(user_ids, item_ids):
        if u not in result:
            result[u] = set()
        result[u].add(int(i))
    return result


def sample_negatives(excluded: set, n_items: int, n_neg: int, rng: np.random.Generator) -> np.ndarray:
    target = min(n_neg, n_items - len(excluded))
    if target <= 0:
        return np.empty((0,), dtype=np.int64)

    negatives = set()
    # Rejection sampling is efficient here because excluded sets are small per user.
    while len(negatives) < target:
        need = target - len(negatives)
        draws = rng.integers(0, n_items, size=max(need * 3, 16), endpoint=False)
        for d in draws:
            if d not in excluded:
                negatives.add(int(d))
                if len(negatives) == target:
                    break

    return np.array(list(negatives), dtype=np.int64)


def build_eval_candidates(
    train_user_items: Dict[int, set],
    test_user_items: Dict[int, set],
    n_items: int,
    num_negatives: int,
    seed: int,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    eval_data: List[Tuple[int, np.ndarray, np.ndarray]] = []

    for user, rel_items_set in test_user_items.items():
        rel_items = np.array(sorted(rel_items_set), dtype=np.int64)
        if rel_items.size == 0:
            continue

        excluded = set(rel_items_set)
        if user in train_user_items:
            excluded.update(train_user_items[user])

        neg_items = sample_negatives(excluded, n_items, num_negatives, rng)
        if neg_items.size == 0:
            continue

        candidate_items = np.concatenate([rel_items, neg_items])
        ground_truth = np.concatenate(
            [np.ones(rel_items.size, dtype=np.int64), np.zeros(neg_items.size, dtype=np.int64)]
        )

        perm = rng.permutation(candidate_items.size)
        eval_data.append((user, candidate_items[perm], ground_truth[perm]))

    return eval_data


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    rel = labels[order].astype(np.float64)

    discounts = 1.0 / np.log2(np.arange(2, 2 + len(rel)))
    dcg = float(np.sum(rel * discounts))

    ideal_rel = np.sort(labels)[::-1][:k].astype(np.float64)
    ideal_discounts = 1.0 / np.log2(np.arange(2, 2 + len(ideal_rel)))
    idcg = float(np.sum(ideal_rel * ideal_discounts))

    if idcg <= 0:
        return 0.0
    return dcg / idcg


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    rel_top = labels[order]
    total_rel = int(labels.sum())
    if total_rel == 0:
        return 0.0
    return float(rel_top.sum() / total_rel)


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    rel_top = labels[order]
    if len(rel_top) == 0:
        return 0.0
    return float(rel_top.sum() / len(rel_top))


def acc_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    rel_top = labels[order]
    return 1.0 if rel_top.sum() > 0 else 0.0


def evaluate_model(
    model: torch.nn.Module,
    eval_candidates: List[Tuple[int, np.ndarray, np.ndarray]],
    device: str,
) -> Dict[str, float]:
    ks = [5, 10, 20]

    if len(eval_candidates) == 0:
        return {
            "ndcg@10": 0.0,
            "ndcg@20": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "precision@10": 0.0,
            "precision@20": 0.0,
            "acc@5": 0.0,
            "acc@10": 0.0,
            "acc@20": 0.0,
        }

    user_segments = []
    all_users = []
    all_items = []
    all_labels = []

    start = 0
    for user, items, labels in eval_candidates:
        end = start + len(items)
        user_segments.append((start, end))

        all_users.append(np.full(len(items), user, dtype=np.int64))
        all_items.append(items.astype(np.int64))
        all_labels.append(labels.astype(np.int64))

        start = end

    users_flat = np.concatenate(all_users)
    items_flat = np.concatenate(all_items)
    labels_flat = np.concatenate(all_labels)

    model.eval()
    with torch.no_grad():
        user_t = torch.from_numpy(users_flat).long().to(device)
        item_t = torch.from_numpy(items_flat).long().to(device)
        scores_flat = model.forward_simple(user_t, item_t).detach().cpu().numpy()

    metrics = {
        "ndcg@10": [],
        "ndcg@20": [],
        "recall@10": [],
        "recall@20": [],
        "precision@10": [],
        "precision@20": [],
        "acc@5": [],
        "acc@10": [],
        "acc@20": [],
    }

    for seg_start, seg_end in user_segments:
        scores = scores_flat[seg_start:seg_end]
        labels = labels_flat[seg_start:seg_end]

        for k in ks:
            metrics[f"acc@{k}"].append(acc_at_k(scores, labels, k))

        metrics["ndcg@10"].append(ndcg_at_k(scores, labels, 10))
        metrics["ndcg@20"].append(ndcg_at_k(scores, labels, 20))
        metrics["recall@10"].append(recall_at_k(scores, labels, 10))
        metrics["recall@20"].append(recall_at_k(scores, labels, 20))
        metrics["precision@10"].append(precision_at_k(scores, labels, 10))
        metrics["precision@20"].append(precision_at_k(scores, labels, 20))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def run_dataset(
    dataset_name: str,
    tune_path: Path,
    test_path: Path,
    epochs: int,
    num_negatives: int,
    seed: int,
    device: str,
) -> Dict:
    print(f"\n{'=' * 90}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 90}")

    train_user_raw, train_item_raw = load_interactions(tune_path)
    test_user_raw, test_item_raw = load_interactions(test_path)

    (
        train_user,
        train_item,
        test_user,
        test_item,
        n_users,
        n_items,
    ) = remap_ids(train_user_raw, train_item_raw, test_user_raw, test_item_raw)

    print(f"Interactions (tune/train source): {len(train_user)}")
    print(f"Interactions (test source): {len(test_user)}")
    print(f"Users: {n_users} | Items: {n_items}")

    train_split, val_split, _ = DataSplitter.split_train_val_test(
        train_user,
        train_item,
        ratios=(0.9, 0.1, 0.0),
        random_seed=seed,
    )

    train_user_split, train_item_split = train_split
    val_user, val_item = val_split

    n_clients = max(10, min(30, n_users // 100 + 10))

    client_data = DataSplitter.split_to_clients_iid(
        train_user_split,
        train_item_split,
        n_clients=n_clients,
        random_seed=seed,
    )

    val_tensor = (
        torch.from_numpy(val_user).long().to(device),
        torch.from_numpy(val_item).long().to(device),
        torch.ones(len(val_user), dtype=torch.float32).to(device),
    )

    train_user_items = build_user_item_dict(train_user, train_item)
    test_user_items = build_user_item_dict(test_user, test_item)

    eval_candidates = build_eval_candidates(
        train_user_items=train_user_items,
        test_user_items=test_user_items,
        n_items=n_items,
        num_negatives=num_negatives,
        seed=seed,
    )

    print(f"Users evaluated: {len(eval_candidates)}")
    print(f"Federated clients: {n_clients}")

    config = PCTRAConfig(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=16,
        n_gcn_layers=1,
        n_clients=n_clients,
        n_rounds=1,
        local_epochs=1,
        batch_size=256,
        learning_rate=0.01,
        clipping_norm=1.0,
        noise_multiplier=0.2,
        delta=1e-5,
        lambda_B=0.3,
        lambda_M=0.25,
        lambda_Q=0.25,
        lambda_R=0.2,
        rho=2.0,
        eta=0.95,
        lambda_uncertainty=0.5,
        beta_T=1.0,
        beta_Q=1.0,
        beta_N=0.5,
        kappa=0.5,
        top_k_fraction=1.0,
    )

    pctra = ImprovedPCTRA(config, device=device)

    epoch_history = []

    for epoch in range(1, epochs + 1):
        _ = pctra.train_federated(
            client_data=client_data,
            validation_data=val_tensor,
            test_data=(test_user, test_item),
            clients_with_attacks=None,
            verbose=False,
        )

        metrics = evaluate_model(pctra.global_model, eval_candidates, device=device)
        epoch_record = {"epoch": epoch, **metrics}
        epoch_history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"NDCG@10={metrics['ndcg@10']:.4f} "
            f"NDCG@20={metrics['ndcg@20']:.4f} "
            f"Recall@10={metrics['recall@10']:.4f} "
            f"Recall@20={metrics['recall@20']:.4f} "
            f"Precision@10={metrics['precision@10']:.4f} "
            f"Precision@20={metrics['precision@20']:.4f} "
            f"Acc@5={metrics['acc@5']:.4f} "
            f"Acc@10={metrics['acc@10']:.4f} "
            f"Acc@20={metrics['acc@20']:.4f}"
        )

    best = max(epoch_history, key=lambda x: (x["ndcg@20"], x["ndcg@10"]))

    print("-" * 90)
    print(f"Best epoch for {dataset_name}: {best['epoch']}")
    print(
        f"Best metrics | NDCG@10={best['ndcg@10']:.4f} NDCG@20={best['ndcg@20']:.4f} "
        f"Recall@10={best['recall@10']:.4f} Recall@20={best['recall@20']:.4f} "
        f"Precision@10={best['precision@10']:.4f} Precision@20={best['precision@20']:.4f} "
        f"Acc@5={best['acc@5']:.4f} Acc@10={best['acc@10']:.4f} Acc@20={best['acc@20']:.4f}"
    )

    return {
        "dataset": dataset_name,
        "n_users": n_users,
        "n_items": n_items,
        "n_clients": n_clients,
        "epochs": epoch_history,
        "best_epoch": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCTRA on processed datasets and report best epoch metrics.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of federated epochs (rounds) per dataset.")
    parser.add_argument("--num-negatives", type=int, default=100, help="Negatives sampled per user for ranking evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="pctra_processed_results.json",
        help="Path to save full per-epoch and best-epoch results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        choices=["cpu", "cuda"],
        help="Device for training/evaluation.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    root = Path(__file__).resolve().parent

    datasets = [
        (
            "Gowalla",
            root / "processed" / "gowalla" / "Gowalla_tune_filtered.txt",
            root / "processed" / "gowalla" / "Gowalla_test_filtered.txt",
        ),
        (
            "Foursquare",
            root / "processed" / "foursquare" / "Foursquare_tune_filtered.txt",
            root / "processed" / "foursquare" / "Foursquare_test_filtered.txt",
        ),
        (
            "Swarm",
            root / "processed" / "swarm" / "Swarm_tune_filtered.txt",
            root / "processed" / "swarm" / "Swarm_test_filtered.txt",
        ),
    ]

    all_results = []

    for dataset_name, tune_path, test_path in datasets:
        result = run_dataset(
            dataset_name=dataset_name,
            tune_path=tune_path,
            test_path=test_path,
            epochs=args.epochs,
            num_negatives=args.num_negatives,
            seed=args.seed,
            device=args.device,
        )
        all_results.append(result)

    summary = {"results": all_results}

    output_path = root / args.output_json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved detailed results to: {output_path}")

    print("\n" + "=" * 90)
    print("BEST EPOCH SUMMARY")
    print("=" * 90)
    for r in all_results:
        b = r["best_epoch"]
        print(
            f"{r['dataset']:<12} | epoch={b['epoch']:>2} | "
            f"NDCG@10={b['ndcg@10']:.4f} NDCG@20={b['ndcg@20']:.4f} "
            f"Recall@10={b['recall@10']:.4f} Recall@20={b['recall@20']:.4f} "
            f"Precision@10={b['precision@10']:.4f} Precision@20={b['precision@20']:.4f} "
            f"Acc@5={b['acc@5']:.4f} Acc@10={b['acc@10']:.4f} Acc@20={b['acc@20']:.4f}"
        )


if __name__ == "__main__":
    main()
