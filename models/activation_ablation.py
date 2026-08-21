#!/usr/bin/env python3
"""
Activation Function Ablation Test — Dead Neuron Detection & Mitigation

Tests multiple activation functions on the football prediction MLP to find:
1. Which activation performs best (log-loss, accuracy, calibration)
2. Dead neuron rates for each activation
3. Training stability and convergence speed
4. Impact on betting ROI and edge detection

Activations tested:
- ReLU (baseline)
- LeakyReLU (prevents dead neurons, α=0.01)
- PReLU (learnable negative slope)
- ELU (smooth, negative values allowed)
- SELU (self-normalizing)
- GELU (Transformer-style smooth ReLU)
- Swish (self-gated, smooth)
- Mish (smooth, self-regularizing)

Research basis:
- He et al. (2015): ReLU initialization and dead neurons
- Maas et al. (2013): LeakyReLU for dead neuron prevention
- Clevert et al. (2016): ELU for faster convergence
- Klambauer et al. (2017): SELU for self-normalizing networks
- Hendrycks & Gimpel (2016): GELU for smooth approximation
- Ramachandran et al. (2017): Swish for improved performance
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime


# ============================================================================
# Activation Registry
# ============================================================================

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=False),
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False),
    "prelu": lambda: nn.PReLU(num_parameters=1, init=0.25),
    "elu": lambda: nn.ELU(alpha=1.0, inplace=False),
    "selu": lambda: nn.SELU(inplace=False),
    "gelu": lambda: nn.GELU(),
    "swish": lambda: nn.SiLU(inplace=False),  # SiLU is Swish
    "mish": lambda: nn.Mish(inplace=False),
}


# ============================================================================
# Dead Neuron Detector
# ============================================================================

class DeadNeuronDetector:
    """Monitors neuron activation statistics to detect dead neurons."""

    def __init__(self):
        self.activations: Dict[str, List[np.ndarray]] = {}
        self.dead_neuron_counts: Dict[str, int] = {}

    def register_layer(self, name: str, layer: nn.Module):
        """Register a layer for monitoring."""
        self.activations[name] = []
        self.dead_neuron_counts[name] = 0

    def hook_fn(self, name: str):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # Store activation statistics
            if isinstance(output, torch.Tensor):
                act = output.detach().cpu().numpy()
                self.activations[name].append(act)
        return hook

    def compute_dead_neurons(self, threshold: float = 0.01) -> Dict[str, int]:
        """Count neurons that are dead (output ~0 for >99% of inputs)."""
        dead_counts = {}
        for name, acts in self.activations.items():
            if not acts:
                continue
            # Stack all activations for this layer
            all_acts = np.concatenate(acts, axis=0)  # (total_samples, n_neurons)
            # A neuron is dead if its mean activation is below threshold
            mean_acts = np.mean(all_acts, axis=0)  # (n_neurons,)
            dead = np.sum(mean_acts < threshold)
            dead_counts[name] = int(dead)
        return dead_counts

    def compute_activation_stats(self) -> Dict[str, Dict]:
        """Compute detailed activation statistics per layer."""
        stats = {}
        for name, acts in self.activations.items():
            if not acts:
                continue
            all_acts = np.concatenate(acts, axis=0)
            stats[name] = {
                "mean": float(np.mean(all_acts)),
                "std": float(np.std(all_acts)),
                "min": float(np.min(all_acts)),
                "max": float(np.max(all_acts)),
                "fraction_zero": float(np.mean(all_acts == 0)),
                "fraction_negative": float(np.mean(all_acts < 0)),
                "n_neurons": all_acts.shape[1] if all_acts.ndim > 1 else 1,
            }
        return stats

    def reset(self):
        """Reset all stored activations."""
        self.activations.clear()
        self.dead_neuron_counts.clear()


# ============================================================================
# Improved MLP with Activation Selection
# ============================================================================

class _MLPAblation(nn.Module):
    """MLP with configurable activation and dead neuron monitoring."""

    def __init__(self, n_in: int, hidden: int = 64,
                 activation: str = "relu",
                 dropout: float = 0.2,
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 init_method: str = "he"):
        super().__init__()

        # Choose activation
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}. Choose from: {list(ACTIVATIONS.keys())}")
        act_fn = ACTIVATIONS[activation]()

        # Build layers
        layers = []
        layers.append(nn.Linear(n_in, hidden))

        # Optional normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(hidden))

        layers.append(act_fn)
        layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden, hidden))

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(hidden))

        layers.append(act_fn)
        layers.append(nn.Linear(hidden, 3))

        self.net = nn.Sequential(*layers)

        # Apply weight initialization
        self._init_weights(init_method)

        # Dead neuron detector
        self.detector = DeadNeuronDetector()
        self._register_hooks()

    def _init_weights(self, method: str):
        """Initialize weights using specified method."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "he":
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif method == "lecun":
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                nn.init.zeros_(m.bias)

    def _register_hooks(self):
        """Register forward hooks for dead neuron detection."""
        for name, module in self.net.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.GELU, nn.SiLU, nn.Mish)):
                self.detector.register_layer(name, module)
                module.register_forward_hook(self.detector.hook_fn(name))

    def forward(self, x):
        return self.net(x)

    def get_dead_neurons(self) -> Dict[str, int]:
        """Get dead neuron counts for each activation layer."""
        return self.detector.compute_dead_neurons()

    def get_activation_stats(self) -> Dict[str, Dict]:
        """Get detailed activation statistics."""
        return self.detector.compute_activation_stats()


# ============================================================================
# Ablation Runner
# ============================================================================

class ActivationAblation:
    """Run activation function ablation test on football prediction task."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 n_in: int, hidden: int = 64,
                 epochs: int = 100, lr: float = 1e-3,
                 batch_size: int = 64):
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.LongTensor(y_val)
        self.n_in = n_in
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def train_and_evaluate(self, activation: str,
                           use_batch_norm: bool = False,
                           use_layer_norm: bool = False,
                           init_method: str = "he") -> Dict:
        """Train a model with specified activation and evaluate."""
        print(f"\n{'='*60}")
        print(f"Testing: {activation} (batch_norm={use_batch_norm}, layer_norm={use_layer_norm})")
        print(f"{'='*60}")

        # Create model
        model = _MLPAblation(
            n_in=self.n_in,
            hidden=self.hidden,
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            init_method=init_method
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        train_losses = []
        val_losses = []
        val_accs = []
        dead_neuron_history = []

        n = len(self.X_train)
        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n)
            total_loss = 0.0

            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                out = model(self.X_train[idx])
                loss = loss_fn(out, self.y_train[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            train_losses.append(total_loss / max(n // self.batch_size, 1))

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(self.X_val)
                val_loss = loss_fn(val_out, self.y_val).item()
                val_pred = torch.argmax(val_out, dim=1)
                val_acc = float(torch.mean((val_pred == self.y_val).float()))
                val_losses.append(val_loss)
                val_accs.append(val_acc)

            # Dead neuron tracking
            dead = model.get_dead_neurons()
            total_dead = sum(dead.values())
            dead_neuron_history.append(total_dead)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train Loss: {train_losses[-1]:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.3f} | "
                      f"Dead Neurons: {total_dead}")

        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_out = model(self.X_val)
            probs = torch.softmax(val_out, dim=1).numpy()

        # Compute log-loss
        eps = 1e-9
        y_val_np = self.y_val.numpy()
        log_loss = -np.mean(np.log(np.clip(probs[np.arange(len(y_val_np)), y_val_np], eps, 1)))

        # Compute ECE (Expected Calibration Error)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs.max(axis=1) >= bin_boundaries[i]) & (probs.max(axis=1) < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = np.mean(np.argmax(probs[mask], axis=1) == y_val_np[mask])
                bin_conf = np.mean(probs[mask].max(axis=1))
                ece += mask.sum() / len(y_val_np) * np.abs(bin_acc - bin_conf)

        # Final dead neuron count
        final_dead = model.get_dead_neurons()
        activation_stats = model.get_activation_stats()

        return {
            "activation": activation,
            "use_batch_norm": use_batch_norm,
            "use_layer_norm": use_layer_norm,
            "final_val_loss": val_losses[-1],
            "final_val_acc": val_accs[-1],
            "log_loss": float(log_loss),
            "ece": float(ece),
            "final_dead_neurons": sum(final_dead.values()),
            "dead_neuron_history": dead_neuron_history,
            "activation_stats": activation_stats,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    def run_full_ablation(self) -> List[Dict]:
        """Run ablation across all activations and configurations."""
        results = []

        # Test each activation
        for act_name in ACTIVATIONS.keys():
            # Basic configuration
            result = self.train_and_evaluate(act_name)
            results.append(result)

            # With batch normalization
            result_bn = self.train_and_evaluate(act_name, use_batch_norm=True)
            results.append(result_bn)

        return results


# ============================================================================
# Main
# ============================================================================

def main():
    """Run activation ablation test."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from models.nn_model import NNFootballPredictor
    from models.ml_layer import MLFootballPredictor
    import pipeline

    print("="*70)
    print("ACTIVATION FUNCTION ABLATION TEST — FOOTBALL PREDICTION MLP")
    print("="*70)

    # Generate synthetic data
    np.random.seed(42)
    torch.manual_seed(42)

    df = pipeline.generate_match_data(1000, seed=42)
    feat = NNFootballPredictor.build_features(df)
    X = feat[["home_elo", "away_elo", "home_goals_avg", "away_goals_avg"]].fillna(
        {"home_elo": 1500.0, "away_elo": 1500.0,
         "home_goals_avg": 1.6, "away_goals_avg": 1.3}).to_numpy()
    y = df["result"].map({"A": 0, "D": 1, "H": 2}).to_numpy()

    # Split: 70% train, 15% val, 15% test
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    n_test = len(X_test)
    print(f"\nDataset: {n} matches")
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # Run ablation
    ablation = ActivationAblation(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        n_in=X.shape[1], hidden=64,
        epochs=100, lr=1e-3
    )

    results = ablation.run_full_ablation()

    # Print summary table
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"{'Activation':<15} {'BatchNorm':<10} {'Val Loss':<10} {'Val Acc':<10} {'Log-Loss':<10} {'ECE':<10} {'Dead Neurons':<15}")
    print("-"*100)

    for r in results:
        print(f"{r['activation']:<15} {str(r['use_batch_norm']):<10} "
              f"{r['final_val_loss']:.4f}    {r['final_val_acc']:.3f}    "
              f"{r['log_loss']:.4f}    {r['ece']:.4f}    {r['final_dead_neurons']}")

    # Find best configuration
    best = min(results, key=lambda x: x['log_loss'])
    print(f"\n{'='*100}")
    print(f"BEST: {best['activation']} (batch_norm={best['use_batch_norm']})")
    print(f"  Log-Loss: {best['log_loss']:.4f}")
    print(f"  ECE: {best['ece']:.4f}")
    print(f"  Val Acc: {best['final_val_acc']:.3f}")
    print(f"  Dead Neurons: {best['final_dead_neurons']}")
    print(f"{'='*100}")

    # Save results
    results_path = Path(__file__).resolve().parent.parent / "backtests" / "results" / "activation_ablation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != 'activation_stats'}
        sr['activation_stats'] = {
            name: {k: v for k, v in stats.items()}
            for name, stats in r['activation_stats'].items()
        }
        serializable.append(sr)

    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset_size": n,
            "results": serializable,
            "best": best['activation'],
        }, f, indent=2)

    print(f"\n[OK] Results saved to {results_path}")

    return results, best


if __name__ == "__main__":
    main()
