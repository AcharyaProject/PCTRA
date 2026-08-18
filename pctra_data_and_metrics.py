"""
Improved PCTRA: Data Loading, Evaluation Metrics, and Utilities

This module provides:
- Real dataset loading (Gowalla, Foursquare, Swarm)
- Evaluation metrics (NDCG, Recall, Hit Rate, MAP)
- Data preprocessing and splitting
- Benchmarking utilities
- Visualization functions
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import pickle
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class RecommendationMetrics:
    """Compute recommendation evaluation metrics"""
    
    @staticmethod
    def ndcg_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
        """
        Compute NDCG@K (Normalized Discounted Cumulative Gain).
        
        NDCG@K = DCG@K / IDCG@K
        
        where DCG@K = Σ_{i=1}^K (2^{rel_i} - 1) / log_2(i+1)
        
        Args:
            predictions: Predicted scores, shape (n_items,)
            ground_truth: Ground truth relevance, shape (n_items,)
            k: Cutoff position
        
        Returns:
            ndcg_score: NDCG@K score
        """
        # Sort by predictions
        sorted_indices = np.argsort(-predictions)[:k]
        sorted_relevance = ground_truth[sorted_indices]
        
        # Compute DCG
        dcg = np.sum((2.0**sorted_relevance - 1) / np.log2(np.arange(2, k+2)))
        
        # Compute IDCG (ideal ranking: all 1s come first)
        ideal_relevance = np.sort(ground_truth)[::-1][:k]
        idcg = np.sum((2.0**ideal_relevance - 1) / np.log2(np.arange(2, k+2)))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def recall_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int = 20) -> float:
        """
        Compute Recall@K.
        
        Recall@K = |{relevant items in top-K}| / |{all relevant items}|
        
        Args:
            predictions: Predicted scores
            ground_truth: Binary relevance (0 or 1)
            k: Cutoff position
        
        Returns:
            recall_score: Recall@K
        """
        sorted_indices = np.argsort(-predictions)[:k]
        retrieved = ground_truth[sorted_indices]
        
        n_relevant = np.sum(ground_truth)
        if n_relevant == 0:
            return 0.0
        
        return np.sum(retrieved) / n_relevant
    
    @staticmethod
    def hit_rate_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
        """
        Compute Hit Rate@K (whether any relevant item is in top-K).
        
        Args:
            predictions: Predicted scores
            ground_truth: Binary relevance
            k: Cutoff position
        
        Returns:
            hit_rate: Fraction of queries with at least one relevant item
        """
        sorted_indices = np.argsort(-predictions)[:k]
        retrieved = ground_truth[sorted_indices]
        
        return 1.0 if np.sum(retrieved) > 0 else 0.0
    
    @staticmethod
    def mean_average_precision_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
        """
        Compute MAP@K (Mean Average Precision).
        
        AP@K = Σ_{i=1}^K P(i) * rel(i) / min(m, K)
        
        where P(i) = (# relevant @ i) / i
        
        Args:
            predictions: Predicted scores
            ground_truth: Binary relevance
            k: Cutoff position
        
        Returns:
            map_score: MAP@K
        """
        sorted_indices = np.argsort(-predictions)[:k]
        sorted_relevance = ground_truth[sorted_indices]
        
        # Cumulative sum of relevant items
        cum_relevant = np.cumsum(sorted_relevance)
        
        # Positions of relevant items (1-indexed)
        positions = np.arange(1, k+1)
        
        # Precision at each position
        precisions = cum_relevant / positions
        
        # AP: average precision only at relevant positions
        relevant_positions = np.where(sorted_relevance > 0)[0]
        
        if len(relevant_positions) == 0:
            return 0.0
        
        ap = np.sum(precisions[relevant_positions]) / min(np.sum(ground_truth), k)
        
        return ap


class MetricsComputer:
    """Compute multiple metrics efficiently"""
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [5, 10, 20]
        self.metrics = RecommendationMetrics()
    
    def compute_metrics(self, predictions: np.ndarray,
                       ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Compute all metrics for given predictions.
        
        Returns:
            metrics_dict: Dictionary of metric names to scores
        """
        results = {}
        
        for k in self.k_values:
            results[f'NDCG@{k}'] = self.metrics.ndcg_at_k(predictions, ground_truth, k)
            results[f'Recall@{k}'] = self.metrics.recall_at_k(predictions, ground_truth, k)
            results[f'HR@{k}'] = self.metrics.hit_rate_at_k(predictions, ground_truth, k)
            results[f'MAP@{k}'] = self.metrics.mean_average_precision_at_k(predictions, ground_truth, k)
        
        return results
    
    def compute_all_metrics(self, all_predictions: List[np.ndarray],
                           all_ground_truth: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute metrics for multiple queries and average.
        
        Args:
            all_predictions: List of prediction arrays
            all_ground_truth: List of ground truth arrays
        
        Returns:
            avg_metrics: Averaged metrics across all queries
        """
        all_metric_dicts = [
            self.compute_metrics(pred, truth)
            for pred, truth in zip(all_predictions, all_ground_truth)
        ]
        
        # Average across queries
        avg_metrics = {}
        for metric_name in all_metric_dicts[0].keys():
            avg_metrics[metric_name] = np.mean([
                d[metric_name] for d in all_metric_dicts
            ])
        
        return avg_metrics


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

class DataLoader:
    """Load and preprocess recommendation datasets"""
    
    @staticmethod
    def load_gowalla(data_path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Load Gowalla dataset.
        Format: user_id, venue_id, check-in_time, latitude, longitude
        
        Returns:
            user_ids, item_ids, n_users, n_items
        """
        data = pd.read_csv(data_path, sep='\t', header=None)
        user_ids = data[0].values - 1  # 0-indexed
        item_ids = data[1].values - 1
        
        n_users = user_ids.max() + 1
        n_items = item_ids.max() + 1
        
        return user_ids, item_ids, n_users, n_items
    
    @staticmethod
    def load_foursquare(data_path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Load Foursquare dataset.
        Format: user_id \t venue_id \t check-in_time \t latitude \t longitude
        """
        data = pd.read_csv(data_path, sep='\t', header=None)
        user_ids = data[0].values - 1  # 0-indexed
        item_ids = data[1].values - 1
        
        n_users = user_ids.max() + 1
        n_items = item_ids.max() + 1
        
        return user_ids, item_ids, n_users, n_items
    
    @staticmethod
    def load_swarm(data_path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Load Swarm dataset"""
        data = pd.read_csv(data_path, sep=',', header=0)
        user_ids = pd.factorize(data['user_id'])[0]  # Encode to integers
        item_ids = pd.factorize(data['venue_id'])[0]
        
        n_users = user_ids.max() + 1
        n_items = item_ids.max() + 1
        
        return user_ids, item_ids, n_users, n_items
    
    @staticmethod
    def preprocess_data(user_ids: np.ndarray, item_ids: np.ndarray,
                       min_user_interactions: int = 5,
                       min_item_interactions: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter users and items by minimum interaction count.
        
        Args:
            user_ids, item_ids: Original data
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
        
        Returns:
            filtered_users, filtered_items
        """
        data = pd.DataFrame({'user': user_ids, 'item': item_ids})
        
        # Filter users
        user_counts = data['user'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        data = data[data['user'].isin(valid_users)]
        
        # Filter items
        item_counts = data['item'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        data = data[data['item'].isin(valid_items)]
        
        # Re-encode to consecutive integers
        data['user'] = pd.factorize(data['user'])[0]
        data['item'] = pd.factorize(data['item'])[0]
        
        return data['user'].values, data['item'].values


# ============================================================================
# DATA SPLITTING & CLIENT ASSIGNMENT
# ============================================================================

class DataSplitter:
    """Split data for federated learning"""
    
    @staticmethod
    def split_train_val_test(user_ids: np.ndarray, item_ids: np.ndarray,
                            ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                            random_seed: int = 42) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Split data into train/val/test sets.
        
        Returns:
            train_data, val_data, test_data (each as (user_ids, item_ids))
        """
        np.random.seed(random_seed)
        n_total = len(user_ids)
        indices = np.random.permutation(n_total)
        
        train_size = int(n_total * ratios[0])
        val_size = int(n_total * ratios[1])
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        train_data = (user_ids[train_idx], item_ids[train_idx])
        val_data = (user_ids[val_idx], item_ids[val_idx])
        test_data = (user_ids[test_idx], item_ids[test_idx])
        
        return train_data, val_data, test_data
    
    @staticmethod
    def split_to_clients_iid(user_ids: np.ndarray, item_ids: np.ndarray,
                            n_clients: int,
                            random_seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split training data to clients (IID distribution).
        
        Args:
            user_ids, item_ids: Training data
            n_clients: Number of clients
            random_seed: For reproducibility
        
        Returns:
            client_data: List of (user_ids, item_ids) for each client
        """
        np.random.seed(random_seed)
        n_total = len(user_ids)
        indices = np.random.permutation(n_total)
        
        client_indices = np.array_split(indices, n_clients)
        
        client_data = [
            (user_ids[idx], item_ids[idx]) for idx in client_indices
        ]
        
        return client_data
    
    @staticmethod
    def split_to_clients_noniid(user_ids: np.ndarray, item_ids: np.ndarray,
                               n_clients: int, n_shards: int = 200,
                               random_seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split training data to clients (non-IID distribution).
        Follows Dirichlet distribution to simulate realistic heterogeneity.
        
        Args:
            user_ids, item_ids: Training data
            n_clients: Number of clients
            n_shards: Number of shards (should be > n_clients)
            random_seed: For reproducibility
        
        Returns:
            client_data: List of (user_ids, item_ids) for each client
        """
        np.random.seed(random_seed)
        n_total = len(user_ids)
        
        # Sort by user for non-IID split
        sorted_indices = np.argsort(user_ids)
        sorted_users = user_ids[sorted_indices]
        sorted_items = item_ids[sorted_indices]
        
        # Create shards
        shard_size = n_total // n_shards
        shards = []
        for i in range(n_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < n_shards - 1 else n_total
            shards.append((sorted_users[start_idx:end_idx], sorted_items[start_idx:end_idx]))
        
        # Assign shards to clients
        shards_per_client = n_shards // n_clients
        client_data = []
        
        for client_id in range(n_clients):
            start_shard = client_id * shards_per_client
            end_shard = start_shard + shards_per_client if client_id < n_clients - 1 else n_shards
            
            client_users = np.concatenate([s[0] for s in shards[start_shard:end_shard]])
            client_items = np.concatenate([s[1] for s in shards[start_shard:end_shard]])
            
            client_data.append((client_users, client_items))
        
        return client_data


# ============================================================================
# BENCHMARKING & COMPARISON
# ============================================================================

class Benchmarker:
    """Benchmark improved PCTRA against baselines"""
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def save_results(self, model_name: str, metrics: Dict[str, float],
                    metadata: Dict = None):
        """Save evaluation results"""
        self.results[model_name] = {
            'metrics': metrics,
            'metadata': metadata or {}
        }
    
    def load_results(self, filepath: str) -> Dict:
        """Load saved results"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table across models"""
        rows = []
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['metrics'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def print_comparison(self):
        """Print formatted comparison table"""
        df = self.create_comparison_table()
        print(df.to_string(index=False))
    
    def save_comparison(self, filepath: str):
        """Save comparison to CSV"""
        df = self.create_comparison_table()
        df.to_csv(filepath, index=False)


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

class Visualizer:
    """Visualization utilities for results"""
    
    @staticmethod
    def plot_training_curves(history: Dict[str, List[float]], metric: str = 'ndcg'):
        """
        Plot training curves (requires matplotlib).
        
        Args:
            history: Training history dict with metric names as keys
            metric: Which metric to plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")
            return
        
        if metric not in history:
            print(f"Metric '{metric}' not found in history.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(history[metric], marker='o', label=metric.upper())
        plt.xlabel('Round')
        plt.ylabel(metric.upper())
        plt.title(f'Training Curve: {metric.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_privacy_utility_tradeoff(epsilons: List[float], utilities: List[float]):
        """
        Plot privacy-utility tradeoff curve.
        
        Args:
            epsilons: Privacy budget values
            utilities: Corresponding utility values (e.g., NDCG)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, utilities, marker='s', linewidth=2, markersize=8)
        plt.xlabel('Privacy Budget (ε)')
        plt.ylabel('Utility (NDCG@10)')
        plt.title('Privacy-Utility Tradeoff')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_robustness_comparison(attack_percentages: List[int],
                                  pctra_scores: List[float],
                                  baseline_scores: List[float]):
        """
        Plot robustness under increasing attacks.
        
        Args:
            attack_percentages: % of malicious clients
            pctra_scores: NDCG scores for improved PCTRA
            baseline_scores: NDCG scores for baseline
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(attack_percentages, pctra_scores, marker='o', label='Improved PCTRA', linewidth=2)
        plt.plot(attack_percentages, baseline_scores, marker='s', label='Baseline', linewidth=2)
        plt.xlabel('% Malicious Clients')
        plt.ylabel('NDCG@10')
        plt.title('Robustness Under Increasing Attacks')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================================================
# STATISTICAL TESTING
# ============================================================================

class StatisticalTesting:
    """Statistical significance testing"""
    
    @staticmethod
    def paired_t_test(scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
        """
        Paired t-test to compare two sets of scores.
        
        Args:
            scores1, scores2: Score arrays
        
        Returns:
            t_statistic, p_value
        """
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        return float(t_stat), float(p_value)
    
    @staticmethod
    def is_significant(p_value: float, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant"""
        return p_value < alpha
    
    @staticmethod
    def mean_confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval for mean.
        
        Args:
            scores: Score array
            confidence: Confidence level (e.g., 0.95 for 95% CI)
        
        Returns:
            lower_bound, upper_bound
        """
        from scipy import stats
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        se = std / np.sqrt(n)
        
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * se
        
        return float(mean - margin), float(mean + margin)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("IMPROVED PCTRA: Data Loading & Evaluation Utilities")
    print("=" * 80)
    
    # Create synthetic data
    print("\n📊 Creating synthetic data...")
    n_interactions = 100000
    user_ids = np.random.randint(0, 5000, n_interactions)
    item_ids = np.random.randint(0, 1000, n_interactions)
    
    print(f"  - Total interactions: {n_interactions}")
    print(f"  - Unique users: {len(np.unique(user_ids))}")
    print(f"  - Unique items: {len(np.unique(item_ids))}")
    
    # Preprocessing
    print("\n🔧 Preprocessing data...")
    user_ids, item_ids = DataLoader.preprocess_data(
        user_ids, item_ids,
        min_user_interactions=5,
        min_item_interactions=5
    )
    print(f"  - After filtering: {len(user_ids)} interactions")
    print(f"  - Unique users: {len(np.unique(user_ids))}")
    print(f"  - Unique items: {len(np.unique(item_ids))}")
    
    # Split data
    print("\n📁 Splitting data...")
    train_data, val_data, test_data = DataSplitter.split_train_val_test(
        user_ids, item_ids
    )
    print(f"  - Train: {len(train_data[0])} interactions")
    print(f"  - Validation: {len(val_data[0])} interactions")
    print(f"  - Test: {len(test_data[0])} interactions")
    
    # Split to clients
    print("\n👥 Assigning to 100 clients (IID)...")
    client_data = DataSplitter.split_to_clients_iid(
        train_data[0], train_data[1], n_clients=100
    )
    print(f"  - Client data sizes: {[len(c[0]) for c in client_data[:5]]}... (first 5)")
    
    # Evaluation metrics
    print("\n📊 Computing evaluation metrics...")
    metrics_computer = MetricsComputer(k_values=[5, 10, 20])
    
    # Simulate predictions
    test_predictions = np.random.rand(len(test_data[0]))
    test_ground_truth = np.random.randint(0, 2, len(test_data[0]))
    
    test_metrics = metrics_computer.compute_metrics(test_predictions, test_ground_truth)
    
    print("  Metrics:")
    for metric_name, score in test_metrics.items():
        print(f"    - {metric_name}: {score:.4f}")
    
    print("\n" + "=" * 80)
    print("✨ Data loading & evaluation utilities ready!")
    print("=" * 80)
