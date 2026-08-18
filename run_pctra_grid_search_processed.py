import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TORCH = None
IMPROVED_PCTRA = None
PCTRA_CONFIG = None
DATA_SPLITTER = None
SET_SEED = None
LOAD_INTERACTIONS = None
REMAP_IDS = None
BUILD_USER_ITEM_DICT = None
BUILD_EVAL_CANDIDATES = None
EVALUATE_MODEL = None


sns.set_style("darkgrid")
PLOT_COLORS = {"gowalla": "#2E86AB", "foursquare": "#A23B72", "swarm": "#F18F01"}
PARAM_COLS = [
    "preference.beta_Q",
    "preference.lambda_uncertainty",
    "preference.kappa",
    "training.local_epochs",
    "training.learning_rate",
]
METRIC_COLS = [
    "ndcg@10",
    "ndcg@20",
    "recall@10",
    "recall@20",
    "precision@10",
    "precision@20",
    "acc@5",
    "acc@10",
    "acc@20",
]


def _ensure_training_dependencies() -> None:
    global TORCH
    global IMPROVED_PCTRA
    global PCTRA_CONFIG
    global DATA_SPLITTER
    global SET_SEED
    global LOAD_INTERACTIONS
    global REMAP_IDS
    global BUILD_USER_ITEM_DICT
    global BUILD_EVAL_CANDIDATES
    global EVALUATE_MODEL

    if TORCH is not None:
        return

    import torch

    from improved_pctra_complete import ImprovedPCTRA, PCTRAConfig
    from pctra_data_and_metrics import DataSplitter
    from run_pctra_processed_datasets import (
        set_seed,
        load_interactions,
        remap_ids,
        build_user_item_dict,
        build_eval_candidates,
        evaluate_model,
    )

    TORCH = torch
    IMPROVED_PCTRA = ImprovedPCTRA
    PCTRA_CONFIG = PCTRAConfig
    DATA_SPLITTER = DataSplitter
    SET_SEED = set_seed
    LOAD_INTERACTIONS = load_interactions
    REMAP_IDS = remap_ids
    BUILD_USER_ITEM_DICT = build_user_item_dict
    BUILD_EVAL_CANDIDATES = build_eval_candidates
    EVALUATE_MODEL = evaluate_model


def make_grid() -> List[Dict[str, float]]:
    beta_q_values = [0.5, 1.0, 1.5, 2.0]
    lambda_unc_values = [0.3, 0.5, 0.7, 1.0]
    kappa_values = [0.2, 0.5, 1.0, 2.0]
    local_epochs_values = [1, 3, 5, 10]
    lr_values = [0.005, 0.01, 0.02]

    combinations = []
    for beta_q, lambda_unc, kappa, local_epochs, learning_rate in itertools.product(
        beta_q_values,
        lambda_unc_values,
        kappa_values,
        local_epochs_values,
        lr_values,
    ):
        combinations.append(
            {
                "preference.beta_Q": beta_q,
                "preference.lambda_uncertainty": lambda_unc,
                "preference.kappa": kappa,
                "training.local_epochs": local_epochs,
                "training.learning_rate": learning_rate,
            }
        )
    return combinations


def prepare_dataset(
    tune_path: Path,
    test_path: Path,
    seed: int,
    n_clients: int,
    num_negatives: int,
) -> Dict:
    _ensure_training_dependencies()

    train_user_raw, train_item_raw = LOAD_INTERACTIONS(tune_path)
    test_user_raw, test_item_raw = LOAD_INTERACTIONS(test_path)

    (
        train_user,
        train_item,
        test_user,
        test_item,
        n_users,
        n_items,
    ) = REMAP_IDS(train_user_raw, train_item_raw, test_user_raw, test_item_raw)

    train_split, val_split, _ = DATA_SPLITTER.split_train_val_test(
        train_user,
        train_item,
        ratios=(0.9, 0.1, 0.0),
        random_seed=seed,
    )

    train_user_split, train_item_split = train_split
    val_user, val_item = val_split

    client_data = DATA_SPLITTER.split_to_clients_iid(
        train_user_split,
        train_item_split,
        n_clients=n_clients,
        random_seed=seed,
    )

    train_user_items = BUILD_USER_ITEM_DICT(train_user, train_item)
    test_user_items = BUILD_USER_ITEM_DICT(test_user, test_item)

    eval_candidates = BUILD_EVAL_CANDIDATES(
        train_user_items=train_user_items,
        test_user_items=test_user_items,
        n_items=n_items,
        num_negatives=num_negatives,
        seed=seed,
    )

    return {
        "n_users": n_users,
        "n_items": n_items,
        "client_data": client_data,
        "val_user": val_user,
        "val_item": val_item,
        "test_user": test_user,
        "test_item": test_item,
        "eval_candidates": eval_candidates,
    }


def run_one_combo(
    dataset_bundle: Dict,
    params: Dict[str, float],
    n_clients: int,
    seed: int,
    device: str,
) -> Dict[str, float]:
    _ensure_training_dependencies()

    config = PCTRA_CONFIG(
        n_users=dataset_bundle["n_users"],
        n_items=dataset_bundle["n_items"],
        embedding_dim=16,
        n_gcn_layers=1,
        n_clients=n_clients,
        n_rounds=1,
        local_epochs=int(params["training.local_epochs"]),
        batch_size=256,
        learning_rate=float(params["training.learning_rate"]),
        clipping_norm=1.0,
        noise_multiplier=0.2,
        delta=1e-5,
        lambda_B=0.3,
        lambda_M=0.25,
        lambda_Q=0.25,
        lambda_R=0.2,
        rho=2.0,
        eta=0.95,
        lambda_uncertainty=float(params["preference.lambda_uncertainty"]),
        beta_T=1.0,
        beta_Q=float(params["preference.beta_Q"]),
        beta_N=0.5,
        kappa=float(params["preference.kappa"]),
        top_k_fraction=1.0,
    )

    SET_SEED(seed)

    pctra = IMPROVED_PCTRA(config, device=device)

    val_tensor = (
        TORCH.from_numpy(dataset_bundle["val_user"]).long().to(device),
        TORCH.from_numpy(dataset_bundle["val_item"]).long().to(device),
        TORCH.ones(len(dataset_bundle["val_user"]), dtype=TORCH.float32).to(device),
    )

    _ = pctra.train_federated(
        client_data=dataset_bundle["client_data"],
        validation_data=val_tensor,
        test_data=(dataset_bundle["test_user"], dataset_bundle["test_item"]),
        clients_with_attacks=None,
        verbose=False,
    )

    metrics = EVALUATE_MODEL(
        pctra.global_model,
        dataset_bundle["eval_candidates"],
        device=device,
    )

    return metrics


def run_dataset_grid(
    dataset_name: str,
    tune_path: Path,
    test_path: Path,
    combos: List[Dict[str, float]],
    n_clients: int,
    num_negatives: int,
    seed: int,
    device: str,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    print("\n" + "=" * 100)
    print(f"GRID SEARCH DATASET: {dataset_name}")
    print("=" * 100)

    bundle = prepare_dataset(
        tune_path=tune_path,
        test_path=test_path,
        seed=seed,
        n_clients=n_clients,
        num_negatives=num_negatives,
    )

    print(f"Users: {bundle['n_users']} | Items: {bundle['n_items']} | Clients: {n_clients}")
    print(f"Users evaluated: {len(bundle['eval_candidates'])}")
    print(f"Total combinations: {len(combos)}")

    rows: List[Dict] = []
    csv_path = output_dir / f"grid_{dataset_name.lower()}_all_combinations.csv"

    for idx, params in enumerate(combos, start=1):
        combo_seed = seed + idx
        metrics = run_one_combo(
            dataset_bundle=bundle,
            params=params,
            n_clients=n_clients,
            seed=combo_seed,
            device=device,
        )

        row = {
            "dataset": dataset_name,
            **params,
            **metrics,
        }
        rows.append(row)

        if idx % 20 == 0 or idx == len(combos):
            partial_df = pd.DataFrame(rows)
            partial_df.to_csv(csv_path, index=False)

        if idx % 10 == 0 or idx == len(combos):
            print(
                f"[{dataset_name}] Completed {idx}/{len(combos)} combos | "
                f"latest NDCG@20={metrics['ndcg@20']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    best_idx = df["ndcg@20"].idxmax()
    best_row = df.loc[best_idx].to_dict()

    return df, best_row


def _read_dataset_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {dataset_name} CSV: {path}")

    df = pd.read_csv(path)
    for col in PARAM_COLS + METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str)
    return df


def _save_figure(fig: plt.Figure, output_dir: Path, file_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


def _plot_distribution(
    gowalla_df: pd.DataFrame,
    foursquare_df: pd.DataFrame,
    output_dir: Path,
    swarm_df: Optional[pd.DataFrame] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.hist(
        gowalla_df["ndcg@10"].dropna(),
        bins=30,
        alpha=0.65,
        color=PLOT_COLORS["gowalla"],
        edgecolor="black",
        label="Gowalla",
    )
    ax.hist(
        foursquare_df["ndcg@10"].dropna(),
        bins=30,
        alpha=0.65,
        color=PLOT_COLORS["foursquare"],
        edgecolor="black",
        label="Foursquare",
    )
    if swarm_df is not None:
        ax.hist(
            swarm_df["ndcg@10"].dropna(),
            bins=30,
            alpha=0.55,
            color="#F18F01",
            edgecolor="black",
            label="Swarm",
        )
    ax.set_title("NDCG@10 Distribution")
    ax.set_xlabel("NDCG@10")
    ax.set_ylabel("Frequency")
    ax.legend()
    _save_figure(fig, output_dir, "01_ndcg10_distribution.png")


def _plot_param_impact(
    gowalla_df: pd.DataFrame,
    foursquare_df: pd.DataFrame,
    param_col: str,
    output_dir: Path,
    file_name: str,
    swarm_df: Optional[pd.DataFrame] = None,
) -> None:
    gowalla_stats = gowalla_df.groupby(param_col)["ndcg@10"].agg(["mean", "std"]).reset_index()
    foursquare_stats = foursquare_df.groupby(param_col)["ndcg@10"].agg(["mean", "std"]).reset_index()
    swarm_stats = None
    if swarm_df is not None:
        swarm_stats = swarm_df.groupby(param_col)["ndcg@10"].agg(["mean", "std"]).reset_index()

    x_vals = sorted(set(gowalla_stats[param_col].tolist()) | set(foursquare_stats[param_col].tolist()))
    if swarm_stats is not None:
        x_vals = sorted(set(x_vals) | set(swarm_stats[param_col].tolist()))
    x = np.arange(len(x_vals))
    width = 0.35

    g_map = gowalla_stats.set_index(param_col)
    f_map = foursquare_stats.set_index(param_col)

    g_mean = [g_map.loc[v, "mean"] if v in g_map.index else np.nan for v in x_vals]
    g_std = [g_map.loc[v, "std"] if v in g_map.index else 0.0 for v in x_vals]
    f_mean = [f_map.loc[v, "mean"] if v in f_map.index else np.nan for v in x_vals]
    f_std = [f_map.loc[v, "std"] if v in f_map.index else 0.0 for v in x_vals]
    s_mean = []
    s_std = []
    if swarm_stats is not None:
        s_map = swarm_stats.set_index(param_col)
        s_mean = [s_map.loc[v, "mean"] if v in s_map.index else np.nan for v in x_vals]
        s_std = [s_map.loc[v, "std"] if v in s_map.index else 0.0 for v in x_vals]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(
        x - width / 2,
        g_mean,
        width,
        yerr=g_std,
        capsize=4,
        alpha=0.85,
        color=PLOT_COLORS["gowalla"],
        label="Gowalla",
    )
    ax.bar(
        x + width / 2,
        f_mean,
        width,
        yerr=f_std,
        capsize=4,
        alpha=0.85,
        color=PLOT_COLORS["foursquare"],
        label="Foursquare",
    )
    if swarm_stats is not None:
        ax.bar(
            x + width * 1.5,
            s_mean,
            width,
            yerr=s_std,
            capsize=4,
            alpha=0.85,
            color="#F18F01",
            label="Swarm",
        )

    ax.set_title(f"NDCG@10 Impact: {param_col}")
    ax.set_xlabel(param_col)
    ax.set_ylabel("Mean NDCG@10")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x_vals])
    ax.legend()
    _save_figure(fig, output_dir, file_name)


def _plot_heatmap(df: pd.DataFrame, dataset_name: str, output_dir: Path, file_name: str) -> None:
    pivot = df.pivot_table(
        values="ndcg@10",
        index="preference.beta_Q",
        columns="preference.lambda_uncertainty",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(pivot, annot=True, fmt=".5f", cmap="YlOrRd", cbar_kws={"label": "NDCG@10"}, ax=ax)
    ax.set_title(f"{dataset_name}: beta_Q vs lambda_uncertainty")
    ax.set_xlabel("lambda_uncertainty")
    ax.set_ylabel("beta_Q")
    _save_figure(fig, output_dir, file_name)


def _plot_top10(
    gowalla_df: pd.DataFrame,
    foursquare_df: pd.DataFrame,
    output_dir: Path,
    swarm_df: Optional[pd.DataFrame] = None,
) -> None:
    g_top10 = gowalla_df.nlargest(10, "ndcg@10").reset_index(drop=True)
    f_top10 = foursquare_df.nlargest(10, "ndcg@10").reset_index(drop=True)
    s_top10 = None if swarm_df is None else swarm_df.nlargest(10, "ndcg@10").reset_index(drop=True)
    ranks = np.arange(1, 11)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(ranks, g_top10["ndcg@10"].values, marker="o", linewidth=2, color=PLOT_COLORS["gowalla"], label="Gowalla")
    ax.plot(
        ranks,
        f_top10["ndcg@10"].values,
        marker="s",
        linewidth=2,
        color=PLOT_COLORS["foursquare"],
        label="Foursquare",
    )
    if s_top10 is not None:
        ax.plot(
            ranks,
            s_top10["ndcg@10"].values,
            marker="^",
            linewidth=2,
            color="#F18F01",
            label="Swarm",
        )
    ax.set_title("Top 10 Configurations (NDCG@10)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("NDCG@10")
    ax.legend()
    _save_figure(fig, output_dir, "09_top10_configs.png")


def _plot_correlation(
    gowalla_df: pd.DataFrame,
    foursquare_df: pd.DataFrame,
    output_dir: Path,
    swarm_df: Optional[pd.DataFrame] = None,
) -> None:
    frames = [("Gowalla", gowalla_df), ("Foursquare", foursquare_df)]
    if swarm_df is not None:
        frames.append(("Swarm", swarm_df))
    fig, axes = plt.subplots(1, len(frames), figsize=(8 * len(frames), 6), squeeze=False)
    axes = axes[0]
    for ax, (name, frame) in zip(axes, frames):
        corr = frame[PARAM_COLS + ["ndcg@10"]].corr()
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0.0, vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"{name} Correlation")

    _save_figure(fig, output_dir, "10_correlation_heatmaps.png")


def _plot_cdf(
    gowalla_df: pd.DataFrame,
    foursquare_df: pd.DataFrame,
    output_dir: Path,
    swarm_df: Optional[pd.DataFrame] = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_triplet = ["ndcg@10", "ndcg@20", "recall@20"]

    for idx, metric in enumerate(metric_triplet):
        g_vals = np.sort(gowalla_df[metric].dropna().values)
        f_vals = np.sort(foursquare_df[metric].dropna().values)
        axes[idx].plot(g_vals, np.linspace(0, 1, len(g_vals)), color=PLOT_COLORS["gowalla"], linewidth=2, label="Gowalla")
        axes[idx].plot(
            f_vals,
            np.linspace(0, 1, len(f_vals)),
            color=PLOT_COLORS["foursquare"],
            linewidth=2,
            label="Foursquare",
        )
        if swarm_df is not None:
            s_vals = np.sort(swarm_df[metric].dropna().values)
            axes[idx].plot(
                s_vals,
                np.linspace(0, 1, len(s_vals)),
                color="#F18F01",
                linewidth=2,
                label="Swarm",
            )
        axes[idx].set_title(f"CDF: {metric}")
        axes[idx].set_xlabel(metric)
        axes[idx].set_ylabel("Cumulative Probability")
        axes[idx].legend()

    _save_figure(fig, output_dir, "11_cumulative_distributions.png")


def _plot_3d_ndcg_epoch_acc10(
    gowalla_df: pd.DataFrame,
    foursquare_df: pd.DataFrame,
    output_dir: Path,
    swarm_df: Optional[pd.DataFrame] = None,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    frames = [("Gowalla", gowalla_df, "Blues"), ("Foursquare", foursquare_df, "Reds")]
    if swarm_df is not None:
        frames.append(("Swarm", swarm_df, "Oranges"))
    fig = plt.figure(figsize=(8 * len(frames), 7))

    for index, (name, frame, cmap) in enumerate(frames, start=1):
        ax = fig.add_subplot(1, len(frames), index, projection="3d")
        x = frame["training.local_epochs"].astype(float).values
        y = frame["ndcg@10"].astype(float).values
        z = frame["acc@10"].astype(float).values
        scatter = ax.scatter(x, y, z, c=z, cmap=cmap, alpha=0.8, s=26, edgecolors="black", linewidths=0.2)
        ax.set_title(f"{name}: Epoch vs NDCG@10 vs Acc@10")
        ax.set_xlabel("Local Epochs")
        ax.set_ylabel("NDCG@10")
        ax.set_zlabel("Acc@10")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Acc@10")

    _save_figure(fig, output_dir, "14_3d_ndcg_epoch_acc10.png")


def _find_best_row(best_df: pd.DataFrame, dataset_key: str) -> Optional[pd.Series]:
    if "dataset" not in best_df.columns:
        return None
    mask = best_df["dataset"].astype(str).str.strip().str.lower() == dataset_key.lower()
    if not mask.any():
        return None
    return best_df.loc[mask].iloc[0]


def _extract_tsne_embeddings(
    root: Path,
    dataset_key: str,
    best_row: pd.Series,
    n_clients: int,
    num_negatives: int,
    seed: int,
    device: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    _ensure_training_dependencies()

    dataset_map = {
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
    if dataset_key not in dataset_map:
        return None

    bundle = prepare_dataset(
        tune_path=dataset_map[dataset_key]["tune"],
        test_path=dataset_map[dataset_key]["test"],
        seed=seed,
        n_clients=n_clients,
        num_negatives=num_negatives,
    )

    # Train a 64-dim model using best discovered hyperparameters for embedding visualization.
    config = PCTRA_CONFIG(
        n_users=bundle["n_users"],
        n_items=bundle["n_items"],
        embedding_dim=64,
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
        top_k_fraction=1.0,
    )

    SET_SEED(seed)
    pctra = IMPROVED_PCTRA(config, device=device)
    val_tensor = (
        TORCH.from_numpy(bundle["val_user"]).long().to(device),
        TORCH.from_numpy(bundle["val_item"]).long().to(device),
        TORCH.ones(len(bundle["val_user"]), dtype=TORCH.float32).to(device),
    )
    _ = pctra.train_federated(
        client_data=bundle["client_data"],
        validation_data=val_tensor,
        test_data=(bundle["test_user"], bundle["test_item"]),
        clients_with_attacks=None,
        verbose=False,
    )

    user_emb = pctra.global_model.user_embeddings.weight.detach().cpu().numpy()
    item_emb = pctra.global_model.item_embeddings.weight.detach().cpu().numpy()
    return user_emb, item_emb


def _plot_tsne_user_item_embeddings(
    root: Path,
    best_summary_path: Path,
    output_dir: Path,
    n_clients: int,
    num_negatives: int,
    seed: int,
    device: str,
    max_points_per_group: int = 1500,
) -> None:
    try:
        from sklearn.manifold import TSNE
    except Exception:
        print("Skipping t-SNE plot: scikit-learn is not installed.")
        return

    if not best_summary_path.exists():
        print(f"Skipping t-SNE plot: best summary CSV not found at {best_summary_path}")
        return

    try:
        best_df = pd.read_csv(best_summary_path)
    except Exception as exc:
        print(f"Skipping t-SNE plot: could not read summary CSV ({exc})")
        return

    try:
        if device == "cuda":
            _ensure_training_dependencies()
            if not TORCH.cuda.is_available():
                print("CUDA requested for t-SNE embedding extraction but not available. Falling back to CPU.")
                device = "cpu"
    except Exception as exc:
        print(f"Skipping t-SNE plot: training dependencies unavailable ({exc})")
        return

    rows = {dataset_key: _find_best_row(best_df, dataset_key) for dataset_key in ["gowalla", "foursquare", "swarm"]}
    if any(rows[dataset_key] is None for dataset_key in rows):
        print("Skipping t-SNE plot: Gowalla, Foursquare, or Swarm best row missing in summary CSV.")
        return

    dataset_keys = ["gowalla", "foursquare", "swarm"]
    fig, axes = plt.subplots(1, len(dataset_keys), figsize=(8 * len(dataset_keys), 7))
    plotted = 0

    for idx, dataset_key in enumerate(dataset_keys):
        try:
            extracted = _extract_tsne_embeddings(
                root=root,
                dataset_key=dataset_key,
                best_row=rows[dataset_key],
                n_clients=n_clients,
                num_negatives=num_negatives,
                seed=seed,
                device=device,
            )
        except Exception as exc:
            print(f"Skipping {dataset_key} t-SNE due to error: {exc}")
            continue

        if extracted is None:
            continue

        user_emb, item_emb = extracted
        rng = np.random.default_rng(seed)
        u_idx = np.arange(user_emb.shape[0])
        i_idx = np.arange(item_emb.shape[0])
        if len(u_idx) > max_points_per_group:
            u_idx = rng.choice(u_idx, size=max_points_per_group, replace=False)
        if len(i_idx) > max_points_per_group:
            i_idx = rng.choice(i_idx, size=max_points_per_group, replace=False)

        sampled_users = user_emb[u_idx]
        sampled_items = item_emb[i_idx]
        stacked = np.vstack([sampled_users, sampled_items])

        perplexity = min(30, max(5, (stacked.shape[0] - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=seed,
        )
        projected = tsne.fit_transform(stacked)

        user_2d = projected[: len(sampled_users)]
        item_2d = projected[len(sampled_users) :]

        ax = axes[idx]
        ax.scatter(
            user_2d[:, 0],
            user_2d[:, 1],
            s=10,
            alpha=0.55,
            color=PLOT_COLORS[dataset_key],
            label=f"{dataset_key.title()} Users",
        )
        ax.scatter(
            item_2d[:, 0],
            item_2d[:, 1],
            s=10,
            alpha=0.45,
            color="#555555",
            label=f"{dataset_key.title()} Items",
            marker="x",
        )
        ax.set_title(f"{dataset_key.title()} Embedding t-SNE (64D -> 2D)")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.legend(loc="best")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("Skipping t-SNE plot: no dataset embedding could be generated.")
        return

    _save_figure(fig, output_dir, "15_tsne_user_item_embeddings.png")


def _plot_statistical_comparison(
    gowalla_df: pd.DataFrame,
    foursquare_df: pd.DataFrame,
    output_dir: Path,
    swarm_df: Optional[pd.DataFrame] = None,
) -> None:
    metrics = ["ndcg@10", "ndcg@20", "recall@10", "recall@20", "precision@10", "precision@20"]
    x = np.arange(len(metrics))
    width = 0.35

    frames = [("Gowalla", gowalla_df, PLOT_COLORS["gowalla"]), ("Foursquare", foursquare_df, PLOT_COLORS["foursquare"])]
    if swarm_df is not None:
        frames.append(("Swarm", swarm_df, "#F18F01"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    offsets = np.linspace(-width, width, len(frames))
    for offset, (name, frame, color) in zip(offsets, frames):
        means = [frame[m].mean() for m in metrics]
        axes[0].bar(x + offset, means, width, color=color, alpha=0.85, label=name)
    axes[0].set_title("Mean Metrics")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=40)
    axes[0].legend()

    for offset, (name, frame, color) in zip(offsets, frames):
        means = [frame[m].mean() for m in metrics]
        iqr = [frame[m].quantile(0.75) - frame[m].quantile(0.25) for m in metrics]
        axes[1].bar(x + offset, means, width, yerr=iqr, capsize=4, color=color, alpha=0.85, label=name)
    axes[1].set_title("Means with IQR Error Bars")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, rotation=40)
    axes[1].legend()

    _save_figure(fig, output_dir, "12_statistical_comparison.png")


def _plot_best_by_dataset(best_df: pd.DataFrame, output_dir: Path) -> None:
    required_cols = ["dataset", "ndcg@10", "ndcg@20", "recall@20"]
    available = [c for c in required_cols if c in best_df.columns]
    if len(available) < 2:
        print("Skipping best-by-dataset plot: not enough columns in summary CSV.")
        return

    plot_df = best_df.copy()
    if "dataset" in plot_df.columns:
        plot_df["dataset"] = plot_df["dataset"].astype(str).str.title()

    metrics = [m for m in ["ndcg@10", "ndcg@20", "recall@20"] if m in plot_df.columns]
    if not metrics:
        print("Skipping best-by-dataset plot: no metric columns found.")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    plot_df.set_index("dataset")[metrics].plot(kind="bar", ax=ax)
    ax.set_title("Best Configuration Metrics by Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    _save_figure(fig, output_dir, "13_best_by_dataset_summary.png")


def generate_grid_search_figures(
    root: Path,
    results_dir: Path,
    best_summary_path: Path,
    figures_dir: Path,
    mode: str,
    n_clients_for_tsne: int,
    num_negatives_for_tsne: int,
    seed_for_tsne: int,
    device_for_tsne: str,
) -> None:
    gowalla_path = results_dir / "grid_gowalla_all_combinations.csv"
    foursquare_path = results_dir / "grid_foursquare_all_combinations.csv"
    swarm_path = results_dir / "grid_swarm_all_combinations.csv"

    gowalla_df = _read_dataset_csv(gowalla_path, "gowalla")
    foursquare_df = _read_dataset_csv(foursquare_path, "foursquare")
    swarm_df = _read_dataset_csv(swarm_path, "swarm")

    print("\n" + "=" * 100)
    print("GENERATING GRID SEARCH FIGURES")
    print("=" * 100)
    print(f"Using results from: {results_dir}")
    print(f"Saving figures to: {figures_dir}")

    if mode in ("all", "original"):
        _plot_distribution(gowalla_df, foursquare_df, figures_dir, swarm_df)
        _plot_param_impact(gowalla_df, foursquare_df, "preference.beta_Q", figures_dir, "02_impact_beta_q.png", swarm_df)
        _plot_param_impact(
            gowalla_df,
            foursquare_df,
            "preference.lambda_uncertainty",
            figures_dir,
            "03_impact_lambda_uncertainty.png",
            swarm_df,
        )
        _plot_param_impact(gowalla_df, foursquare_df, "preference.kappa", figures_dir, "04_impact_kappa.png", swarm_df)
        _plot_param_impact(
            gowalla_df,
            foursquare_df,
            "training.local_epochs",
            figures_dir,
            "05_impact_local_epochs.png",
            swarm_df,
        )
        _plot_param_impact(
            gowalla_df,
            foursquare_df,
            "training.learning_rate",
            figures_dir,
            "06_impact_learning_rate.png",
            swarm_df,
        )
        _plot_heatmap(gowalla_df, "Gowalla", figures_dir, "07_heatmap_gowalla_beta_lambda.png")
        _plot_heatmap(foursquare_df, "Foursquare", figures_dir, "08_heatmap_foursquare_beta_lambda.png")
        _plot_heatmap(swarm_df, "Swarm", figures_dir, "08b_heatmap_swarm_beta_lambda.png")
        _plot_top10(gowalla_df, foursquare_df, figures_dir, swarm_df)

    if mode in ("all", "additional"):
        _plot_correlation(gowalla_df, foursquare_df, figures_dir, swarm_df)
        _plot_cdf(gowalla_df, foursquare_df, figures_dir, swarm_df)
        _plot_3d_ndcg_epoch_acc10(gowalla_df, foursquare_df, figures_dir, swarm_df)
        _plot_tsne_user_item_embeddings(
            root=root,
            best_summary_path=best_summary_path,
            output_dir=figures_dir,
            n_clients=n_clients_for_tsne,
            num_negatives=num_negatives_for_tsne,
            seed=seed_for_tsne,
            device=device_for_tsne,
        )

    if mode in ("all", "stats"):
        _plot_statistical_comparison(gowalla_df, foursquare_df, figures_dir, swarm_df)

        if best_summary_path.exists():
            best_df = pd.read_csv(best_summary_path)
            _plot_best_by_dataset(best_df, figures_dir)
        else:
            print(f"Skipping best-by-dataset figure: summary CSV not found at {best_summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full hyperparameter grid search on processed datasets and save dataset-wise CSV outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="grid_results",
        help="Directory where CSV files will be saved.",
    )
    parser.add_argument(
        "--n-clients",
        type=int,
        default=30,
        help="Number of federated clients.",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=100,
        help="Negatives sampled per user for ranking evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training/evaluation.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip grid search and only generate figures from existing CSV files.",
    )
    parser.add_argument(
        "--plot-mode",
        type=str,
        choices=["all", "original", "additional", "stats"],
        default="all",
        help="Choose which set of figures to generate.",
    )
    parser.add_argument(
        "--figures-results-dir",
        type=str,
        default="grid_results",
        help="Directory containing grid_gowalla_all_combinations.csv and grid_foursquare_all_combinations.csv.",
    )
    parser.add_argument(
        "--best-summary-csv",
        type=str,
        default="grid_results_full/grid_best_by_dataset.csv",
        help="Path to grid_best_by_dataset.csv used for best-configuration summary figure.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="grid_results/figures",
        help="Directory to save generated figure files.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_results_dir = root / args.figures_results_dir
    best_summary_csv = root / args.best_summary_csv
    figures_dir = root / args.figures_dir

    combos = make_grid()

    if not args.figures_only:
        _ensure_training_dependencies()
        if args.device == "cuda" and not TORCH.cuda.is_available():
            raise RuntimeError("CUDA requested but not available in the current environment.")

        SET_SEED(args.seed)

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

        best_rows = []

        for dataset_name, tune_path, test_path in datasets:
            _, best_row = run_dataset_grid(
                dataset_name=dataset_name,
                tune_path=tune_path,
                test_path=test_path,
                combos=combos,
                n_clients=args.n_clients,
                num_negatives=args.num_negatives,
                seed=args.seed,
                device=args.device,
                output_dir=output_dir,
            )
            best_rows.append(best_row)

        summary_df = pd.DataFrame(best_rows)
        summary_path = output_dir / "grid_best_by_dataset.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 100)
        print("GRID SEARCH COMPLETE")
        print("=" * 100)
        print(f"Saved per-dataset CSVs and best-summary CSV in: {output_dir}")

    generate_grid_search_figures(
        root=root,
        results_dir=figures_results_dir,
        best_summary_path=best_summary_csv,
        figures_dir=figures_dir,
        mode=args.plot_mode,
        n_clients_for_tsne=args.n_clients,
        num_negatives_for_tsne=args.num_negatives,
        seed_for_tsne=args.seed,
        device_for_tsne=args.device,
    )


if __name__ == "__main__":
    main()
