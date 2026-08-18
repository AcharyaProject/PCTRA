"""
Improved PCTRA: Configuration Management & Advanced Utilities

This module provides:
- Configuration management with validation
- Hyperparameter tuning via grid search
- Experiment tracking and logging
- Privacy accounting utilities
- Model checkpointing and resumption
"""

import json
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from datetime import datetime
import pickle


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@dataclass
class DPConfig:
    """Differential Privacy configuration"""
    enabled: bool = True
    clipping_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1e-5
    
    def validate(self):
        """Validate DP configuration"""
        assert self.clipping_norm > 0, "clipping_norm must be positive"
        assert self.noise_multiplier >= 0, "noise_multiplier must be non-negative"
        assert 0 < self.delta < 1, "delta must be in (0, 1)"


@dataclass
class TrustConfig:
    """Trust score configuration"""
    lambda_B: float = 0.3    # Behavioral reliability
    lambda_M: float = 0.25   # Model consistency
    lambda_Q: float = 0.25   # Contribution quality
    lambda_R: float = 0.2    # Historical reputation
    
    # Uncertainty calibration
    rho: float = 2.0         # Trust penalty weight
    eta: float = 0.95        # DP calibration parameter
    
    def validate(self):
        """Validate trust configuration"""
        weights_sum = self.lambda_B + self.lambda_M + self.lambda_Q + self.lambda_R
        assert abs(weights_sum - 1.0) < 1e-6, f"Trust weights must sum to 1.0, got {weights_sum}"
        assert all(w >= 0 for w in [self.lambda_B, self.lambda_M, self.lambda_Q, self.lambda_R]), \
            "All trust weights must be non-negative"
        assert self.rho > 0, "rho must be positive"
        assert 0 < self.eta < 1, "eta must be in (0, 1)"


@dataclass
class PreferenceConfig:
    """Preference score configuration (NEW improvements)"""
    beta_T: float = 1.0      # Weight on trust
    beta_Q: float = 1.0      # Weight on utility
    beta_N: float = 0.5      # Weight on data size
    
    # Risk adjustment
    lambda_uncertainty: float = 0.5  # Uncertainty discount
    
    # Adaptive caps
    kappa: float = 0.5       # Sensitivity to usefulness
    
    # Client selection
    top_k_fraction: float = 1.0  # Select top-K% (1.0 = all)
    
    def validate(self):
        """Validate preference configuration"""
        assert all(b > 0 for b in [self.beta_T, self.beta_Q, self.beta_N]), \
            "All beta weights must be positive"
        assert 0 < self.lambda_uncertainty <= 1, "lambda_uncertainty must be in (0, 1]"
        assert self.kappa >= 0, "kappa must be non-negative"
        assert 0 < self.top_k_fraction <= 1, "top_k_fraction must be in (0, 1]"


@dataclass
class GCNConfig:
    """Graph Convolutional Network configuration"""
    embedding_dim: int = 64
    n_layers: int = 2
    
    def validate(self):
        """Validate GCN configuration"""
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.n_layers > 0, "n_layers must be positive"


@dataclass
class TrainingConfig:
    """Training configuration"""
    n_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Momentum/optimization
    momentum: float = 0.0
    weight_decay: float = 0.0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = 'NDCG@10'
    
    def validate(self):
        """Validate training configuration"""
        assert self.n_rounds > 0, "n_rounds must be positive"
        assert self.local_epochs > 0, "local_epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.momentum < 1, "momentum must be in [0, 1)"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"


@dataclass
class DataConfig:
    """Data configuration"""
    n_users: int
    n_items: int
    n_clients: int = 100
    
    # Preprocessing
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    
    # Splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Distribution type
    distribution_type: str = 'iid'  # 'iid' or 'non-iid'
    
    def validate(self):
        """Validate data configuration"""
        assert self.n_users > 0, "n_users must be positive"
        assert self.n_items > 0, "n_items must be positive"
        assert self.n_clients > 0, "n_clients must be positive"
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        assert self.distribution_type in ['iid', 'non-iid'], \
            "distribution_type must be 'iid' or 'non-iid'"


@dataclass
class PCTRAConfig:
    """Complete PCTRA configuration"""
    # Data
    data: DataConfig
    gcn: GCNConfig
    training: TrainingConfig
    dp: DPConfig
    trust: TrustConfig
    preference: PreferenceConfig
    
    # Experiment
    name: str = 'PCTRA_Experiment'
    description: str = ''
    seed: int = 42
    device: str = 'cpu'
    
    # Logging
    log_level: str = 'INFO'
    log_dir: str = './logs'
    checkpoint_dir: str = './checkpoints'
    results_dir: str = './results'
    
    def validate(self):
        """Validate all configurations"""
        self.data.validate()
        self.gcn.validate()
        self.training.validate()
        self.dp.validate()
        self.trust.validate()
        self.preference.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'PCTRAConfig':
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        data = DataConfig(**config_dict['data'])
        gcn = GCNConfig(**config_dict['gcn'])
        training = TrainingConfig(**config_dict['training'])
        dp = DPConfig(**config_dict['dp'])
        trust = TrustConfig(**config_dict['trust'])
        preference = PreferenceConfig(**config_dict['preference'])
        
        return cls(
            data=data, gcn=gcn, training=training,
            dp=dp, trust=trust, preference=preference,
            **{k: v for k, v in config_dict.items()
               if k not in ['data', 'gcn', 'training', 'dp', 'trust', 'preference']}
        )
    
    def to_yaml(self, filepath: str):
        """Save to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'PCTRAConfig':
        """Load from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses (same as JSON)
        data = DataConfig(**config_dict['data'])
        gcn = GCNConfig(**config_dict['gcn'])
        training = TrainingConfig(**config_dict['training'])
        dp = DPConfig(**config_dict['dp'])
        trust = TrustConfig(**config_dict['trust'])
        preference = PreferenceConfig(**config_dict['preference'])
        
        return cls(
            data=data, gcn=gcn, training=training,
            dp=dp, trust=trust, preference=preference,
            **{k: v for k, v in config_dict.items()
               if k not in ['data', 'gcn', 'training', 'dp', 'trust', 'preference']}
        )


# ============================================================================
# LOGGING & EXPERIMENT TRACKING
# ============================================================================

class ExperimentLogger:
    """Logging and experiment tracking"""
    
    def __init__(self, log_dir: str = './logs', experiment_name: str = 'pctra_exp'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Timestamp for experiment
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = f"{experiment_name}_{self.timestamp}"
        
        # Setup logging
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        log_file = self.log_dir / f"{self.experiment_name}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_config(self, config: PCTRAConfig):
        """Log configuration"""
        self.logger.info("=" * 80)
        self.logger.info("CONFIGURATION")
        self.logger.info("=" * 80)
        for key, value in config.to_dict().items():
            self.logger.info(f"{key}: {value}")
    
    def log_round(self, round_num: int, metrics: Dict[str, float], metadata: Dict = None):
        """Log round results"""
        msg = f"Round {round_num} | "
        for metric, value in metrics.items():
            msg += f"{metric}: {value:.4f} | "
        self.logger.info(msg.rstrip(' | '))
        
        if metadata:
            self.logger.debug(f"Metadata: {metadata}")
    
    def log_error(self, error_msg: str):
        """Log error"""
        self.logger.error(error_msg)
    
    def log_info(self, msg: str):
        """Log info"""
        self.logger.info(msg)
    
    def log_warning(self, msg: str):
        """Log warning"""
        self.logger.warning(msg)


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

class GridSearchTuner:
    """Grid search for hyperparameter tuning"""
    
    def __init__(self, base_config: PCTRAConfig, logger: ExperimentLogger):
        self.base_config = base_config
        self.logger = logger
        self.results = []
    
    def search(self, param_grid: Dict[str, List[Any]],
              eval_func: callable,
              n_runs: int = 1) -> List[Dict[str, Any]]:
        """
        Perform grid search.
        
        Args:
            param_grid: Dict mapping parameter names to lists of values
            eval_func: Function that takes config and returns metric score
            n_runs: Number of runs per configuration
        
        Returns:
            results: List of (config, score, variance) tuples
        """
        self.logger.log_info(f"Starting grid search with {self._count_configs(param_grid)} configs")
        
        configs_to_try = self._generate_configs(param_grid)
        
        for i, config in enumerate(configs_to_try):
            self.logger.log_info(f"Testing config {i+1}/{len(configs_to_try)}")
            
            # Run multiple times and compute mean/variance
            scores = []
            for run in range(n_runs):
                try:
                    score = eval_func(config)
                    scores.append(score)
                except Exception as e:
                    self.logger.log_error(f"Error in run {run+1}: {e}")
                    continue
            
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                result = {
                    'config': config,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'n_runs': len(scores)
                }
                self.results.append(result)
                
                self.logger.log_info(f"Config {i+1}: Score = {mean_score:.4f} ± {std_score:.4f}")
        
        # Sort by score (descending)
        self.results.sort(key=lambda x: x['mean_score'], reverse=True)
        
        self.logger.log_info(f"Grid search completed. Best score: {self.results[0]['mean_score']:.4f}")
        
        return self.results
    
    def _count_configs(self, param_grid: Dict[str, List[Any]]) -> int:
        """Count total number of configurations"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count
    
    def _generate_configs(self, param_grid: Dict[str, List[Any]]):
        """Generate all configurations from grid"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in itertools.product(*values):
            config = self._copy_config(self.base_config)
            
            # Apply parameters
            for key, value in zip(keys, combination):
                self._set_nested_param(config, key, value)
            
            yield config
    
    def _copy_config(self, config: PCTRAConfig) -> PCTRAConfig:
        """Deep copy configuration"""
        from copy import deepcopy
        return deepcopy(config)
    
    def _set_nested_param(self, config: PCTRAConfig, param_path: str, value: Any):
        """Set nested parameter (e.g., 'preference.beta_T' -> config.preference.beta_T = value)"""
        parts = param_path.split('.')
        obj = config
        
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        setattr(obj, parts[-1], value)
    
    def get_best_config(self) -> PCTRAConfig:
        """Get best configuration from search"""
        if not self.results:
            return self.base_config
        return self.results[0]['config']
    
    def print_top_k(self, k: int = 5):
        """Print top-K configurations"""
        print(f"\nTop {min(k, len(self.results))} Configurations:")
        print("=" * 80)
        for i, result in enumerate(self.results[:k]):
            print(f"\nRank {i+1}: Score = {result['mean_score']:.4f} ± {result['std_score']:.4f}")
            print(f"Config: {result['config'].to_dict()}")


# ============================================================================
# PRIVACY ACCOUNTING
# ============================================================================

class PrivacyAccountant:
    """Track and compute privacy budget"""
    
    def __init__(self, delta: float = 1e-5):
        self.delta = delta
        self.epsilon_history = []
    
    def compute_epsilon(self, n_rounds: int, noise_multiplier: float,
                       alpha: int = 32) -> float:
        """
        Compute ε using Rényi DP composition.
        
        ε_round(α) = α / (2σ²)
        ε_total = Σ ε_round
        """
        eps_round = alpha / (2 * (noise_multiplier**2))
        eps_total = n_rounds * eps_round
        
        return eps_total
    
    def add_round(self, noise_multiplier: float):
        """Add privacy cost for one round"""
        eps = self.compute_epsilon(1, noise_multiplier)
        self.epsilon_history.append(eps)
    
    def get_total_epsilon(self) -> float:
        """Get cumulative ε"""
        return sum(self.epsilon_history)
    
    def get_delta(self) -> float:
        """Get δ"""
        return self.delta
    
    def is_private(self, epsilon_threshold: float = 1.0) -> bool:
        """Check if current ε is below threshold"""
        return self.get_total_epsilon() <= epsilon_threshold


# ============================================================================
# MODEL CHECKPOINTING
# ============================================================================

class CheckpointManager:
    """Save and load model checkpoints"""
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def save_checkpoint(self, model, optimizer, round_num: int,
                       metrics: Dict[str, float], config: PCTRAConfig,
                       name: str = 'checkpoint'):
        """Save model checkpoint"""
        checkpoint = {
            'round': round_num,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer else None,
            'metrics': metrics,
            'config': config
        }
        
        filepath = self.checkpoint_dir / f"{name}_round{round_num}.pt"
        torch.save(checkpoint, filepath)
        
        return str(filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        import torch
        checkpoint = torch.load(filepath)
        
        return (checkpoint['round'], checkpoint['model_state'],
                checkpoint['optimizer_state'], checkpoint['metrics'],
                checkpoint['config'])
    
    def get_best_checkpoint(self, metric: str = 'NDCG@10') -> Optional[str]:
        """Get path to checkpoint with best metric"""
        best_checkpoint = None
        best_value = -float('inf')
        
        for checkpoint_file in self.checkpoint_dir.glob('*.pt'):
            checkpoint = torch.load(checkpoint_file)
            if metric in checkpoint['metrics']:
                value = checkpoint['metrics'][metric]
                if value > best_value:
                    best_value = value
                    best_checkpoint = str(checkpoint_file)
        
        return best_checkpoint


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

class ConfigFactory:
    """Factory for creating standard configurations"""
    
    @staticmethod
    def create_gowalla_config(n_clients: int = 100) -> PCTRAConfig:
        """Create configuration for Gowalla dataset"""
        return PCTRAConfig(
            data=DataConfig(
                n_users=3823,
                n_items=5972,
                n_clients=n_clients,
                min_user_interactions=5,
                min_item_interactions=5,
                distribution_type='iid'
            ),
            gcn=GCNConfig(embedding_dim=64, n_layers=2),
            training=TrainingConfig(
                n_rounds=50,
                local_epochs=5,
                batch_size=32,
                learning_rate=0.01
            ),
            dp=DPConfig(clipping_norm=1.0, noise_multiplier=1.0),
            trust=TrustConfig(lambda_B=0.3, lambda_M=0.25, lambda_Q=0.25, lambda_R=0.2),
            preference=PreferenceConfig(
                beta_T=1.0, beta_Q=1.0, beta_N=0.5,
                lambda_uncertainty=0.5, kappa=0.5,
                top_k_fraction=1.0
            ),
            name='PCTRA_Gowalla'
        )
    
    @staticmethod
    def create_foursquare_config(n_clients: int = 100) -> PCTRAConfig:
        """Create configuration for Foursquare dataset"""
        return PCTRAConfig(
            data=DataConfig(
                n_users=2427,
                n_items=2419,
                n_clients=n_clients,
                min_user_interactions=5,
                min_item_interactions=5,
                distribution_type='iid'
            ),
            gcn=GCNConfig(embedding_dim=64, n_layers=2),
            training=TrainingConfig(
                n_rounds=50,
                local_epochs=5,
                batch_size=32,
                learning_rate=0.01
            ),
            dp=DPConfig(clipping_norm=1.0, noise_multiplier=1.0),
            trust=TrustConfig(lambda_B=0.3, lambda_M=0.25, lambda_Q=0.25, lambda_R=0.2),
            preference=PreferenceConfig(
                beta_T=1.0, beta_Q=1.0, beta_N=0.5,
                lambda_uncertainty=0.5, kappa=0.5,
                top_k_fraction=1.0
            ),
            name='PCTRA_Foursquare'
        )
    
    @staticmethod
    def create_debug_config() -> PCTRAConfig:
        """Create small configuration for quick testing"""
        return PCTRAConfig(
            data=DataConfig(
                n_users=500,
                n_items=200,
                n_clients=10,
                distribution_type='iid'
            ),
            gcn=GCNConfig(embedding_dim=32, n_layers=1),
            training=TrainingConfig(
                n_rounds=5,
                local_epochs=1,
                batch_size=16,
                learning_rate=0.01
            ),
            dp=DPConfig(clipping_norm=1.0, noise_multiplier=1.0),
            trust=TrustConfig(),
            preference=PreferenceConfig(),
            name='PCTRA_Debug'
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("IMPROVED PCTRA: Configuration & Advanced Utilities")
    print("=" * 80)
    
    # Create configuration
    print("\n📋 Creating Gowalla configuration...")
    config = ConfigFactory.create_gowalla_config(n_clients=100)
    config.validate()
    print("✅ Configuration validated!")
    
    # Save to JSON
    print("\n💾 Saving configuration to JSON...")
    config.to_json('/tmp/pctra_config.json')
    print("✅ Saved to /tmp/pctra_config.json")
    
    # Load from JSON
    print("\n📂 Loading configuration from JSON...")
    loaded_config = PCTRAConfig.from_json('/tmp/pctra_config.json')
    print("✅ Loaded successfully!")
    
    # Setup logging
    print("\n📝 Setting up experiment logging...")
    logger = ExperimentLogger(experiment_name='pctra_test')
    logger.log_config(config)
    logger.log_info("Experiment started!")
    
    # Privacy accounting
    print("\n🔐 Privacy accounting...")
    accountant = PrivacyAccountant(delta=1e-5)
    eps_total = accountant.compute_epsilon(
        n_rounds=50,
        noise_multiplier=config.dp.noise_multiplier
    )
    print(f"  - Total ε for 50 rounds: {eps_total:.4f}")
    print(f"  - δ: {accountant.get_delta()}")
    
    print("\n" + "=" * 80)
    print("✨ Configuration management utilities ready!")
    print("=" * 80)
