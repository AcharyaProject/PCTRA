import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from run_pctra_grid_search_processed import make_grid, prepare_dataset, run_one_combo


def dataset_paths(root: Path) -> Dict[str, Dict[str, Path]]:
    return {
        "gowalla": {
            "tune": root / "processed" / "gowalla" / "Gowalla_tune_filtered.txt",
            "test": root / "processed" / "gowalla" / "Gowalla_test_filtered.txt",
        },
        "foursquare": {
            "tune": root / "processed" / "foursquare" / "Foursquare_tune_filtered.txt",
            "test": root / "processed" / "foursquare" / "Foursquare_test_filtered.txt",
        },
        "swarm": {
            "tune": root / "processed" / "swarm" / "Swarm_tune_filtered.txt",
            "test": root / "processed" / "swarm" / "Swarm_test_filtered.txt",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a chunk of PCTRA grid search and append to CSV.")
    parser.add_argument("--dataset", type=str, required=True, choices=["gowalla", "foursquare", "swarm"])
    parser.add_argument("--start", type=int, required=True, help="Start combo index (inclusive).")
    parser.add_argument("--end", type=int, required=True, help="End combo index (exclusive).")
    parser.add_argument("--n-clients", type=int, default=30)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default="grid_results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combo_list = make_grid()
    if args.start < 0 or args.start >= len(combo_list):
        raise ValueError("start index out of range")
    if args.end <= args.start or args.end > len(combo_list):
        raise ValueError("end index out of range")

    paths = dataset_paths(root)[args.dataset]

    bundle = prepare_dataset(
        tune_path=paths["tune"],
        test_path=paths["test"],
        seed=args.seed,
        n_clients=args.n_clients,
        num_negatives=args.num_negatives,
    )

    csv_path = out_dir / f"grid_{args.dataset}_all_combinations.csv"
    existing = pd.DataFrame()
    if csv_path.exists():
        existing = pd.read_csv(csv_path)

    rows: List[Dict] = []

    def flush_rows(in_memory_rows: List[Dict]) -> pd.DataFrame:
        new_df_local = pd.DataFrame(in_memory_rows)
        if not existing.empty:
            combined_local = pd.concat([existing, new_df_local], ignore_index=True)
        else:
            combined_local = new_df_local

        combined_local = combined_local.drop_duplicates(subset=["combo_id"], keep="last")
        combined_local = combined_local.sort_values("combo_id").reset_index(drop=True)
        combined_local.to_csv(csv_path, index=False)
        return combined_local

    print(f"Running {args.dataset} combos {args.start}..{args.end - 1}")
    for idx in range(args.start, args.end):
        params = combo_list[idx]
        metrics = run_one_combo(
            dataset_bundle=bundle,
            params=params,
            n_clients=args.n_clients,
            seed=args.seed + idx + 1,
            device=args.device,
        )

        rows.append(
            {
                "combo_id": idx,
                "dataset": args.dataset,
                **params,
                **metrics,
            }
        )

        if (idx - args.start + 1) % 10 == 0 or idx == args.end - 1:
            combined_snapshot = flush_rows(rows)
            existing = combined_snapshot
            print(
                f"{args.dataset}: {idx + 1}/{len(combo_list)} "
                f"NDCG@20={metrics['ndcg@20']:.4f}"
            )

    combined = flush_rows(rows)

    print(f"Saved rows: {len(combined)} to {csv_path}")


if __name__ == "__main__":
    main()
