"""
Holographic State Representation

THEORETICAL FOUNDATION: IRH v21.1 §1.2-1.3

This module represents the state of the holographic resonance field,
tracking coupling constants, RG scales, and field configurations.

Adapted from AlphaGeometry's graph.py proof state representation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# Constants for numerical stability
SCALE_DIFF_TOLERANCE = 1e-12  # Minimum scale difference to avoid division by zero


@dataclass
class CouplingState:
    """
    State of coupling constants at a given RG scale.

    Represents the point (λ̃, γ̃, μ̃) in coupling space at scale k.

    Attributes
    ----------
    lambda_tilde : float
        Dimensionless cosmological constant
    gamma_tilde : float
        Dimensionless Newton's constant
    mu_tilde : float
        Dimensionless mass scale
    k : float
        RG scale
    level : int
        Depth in RG flow trajectory
    """
    lambda_tilde: float
    gamma_tilde: float
    mu_tilde: float
    k: float
    level: int = 0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML processing."""
        return np.array([
            self.lambda_tilde,
            self.gamma_tilde,
            self.mu_tilde,
            self.k
        ])

    def to_jax(self) -> Any:
        """Convert to JAX array for GPU acceleration."""
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available. Install with: pip install jax jaxlib")
        return jnp.array([
            self.lambda_tilde,
            self.gamma_tilde,
            self.mu_tilde,
            self.k
        ])

    def distance_to(self, other: CouplingState) -> float:
        """
        Compute distance to another coupling state.

        Parameters
        ----------
        other : CouplingState
            Another coupling state

        Returns
        -------
        float
            Euclidean distance in coupling space
        """
        delta = self.to_array()[:3] - other.to_array()[:3]
        return float(np.linalg.norm(delta))


class HolographicState:
    """
    Complete holographic field state representation.

    Analogous to AlphaGeometry's Graph class for proof states.
    Tracks the evolution of coupling constants through RG flow.

    Attributes
    ----------
    trajectory : List[CouplingState]
        List of coupling states through RG flow
    fixed_point : Optional[CouplingState]
        Converged fixed point (if reached)
    action : Optional[float]
        cGFT action value
    metadata : Dict[str, Any]
        Additional state information
    """

    def __init__(
        self,
        initial_couplings: Optional[CouplingState] = None,
        k_initial: float = 1.0
    ):
        """
        Initialize holographic state.

        Parameters
        ----------
        initial_couplings : Optional[CouplingState]
            Starting point in coupling space
        k_initial : float
            Initial RG scale
        """
        self.trajectory: List[CouplingState] = []
        self.fixed_point: Optional[CouplingState] = None
        self.action: Optional[float] = None
        self.metadata: Dict[str, Any] = {}

        if initial_couplings is not None:
            self.trajectory.append(initial_couplings)
        else:
            # Default initial state
            self.trajectory.append(
                CouplingState(
                    lambda_tilde=10.0,
                    gamma_tilde=10.0,
                    mu_tilde=10.0,
                    k=k_initial,
                    level=0
                )
            )

    def add_rg_step(
        self,
        new_couplings: CouplingState,
        beta_functions: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """
        Add a new RG flow step to the trajectory.

        Parameters
        ----------
        new_couplings : CouplingState
            New coupling state
        beta_functions : Optional[Tuple[float, float, float]]
            (β_λ, β_γ, β_μ) at this step
        """
        new_couplings.level = len(self.trajectory)
        self.trajectory.append(new_couplings)

        if beta_functions is not None:
            self.metadata[f'beta_{new_couplings.level}'] = beta_functions

    def check_fixed_point(self, tolerance: float = 1e-6) -> bool:
        """
        Check if current state is at a fixed point.

        A fixed point satisfies β_λ = β_γ = β_μ = 0

        Parameters
        ----------
        tolerance : float
            Numerical tolerance for beta functions

        Returns
        -------
        bool
            True if at fixed point
        """
        if len(self.trajectory) < 2:
            return False

        current = self.trajectory[-1]
        previous = self.trajectory[-2]

        # Check if couplings have stopped evolving
        delta_lambda = abs(current.lambda_tilde - previous.lambda_tilde)
        delta_gamma = abs(current.gamma_tilde - previous.gamma_tilde)
        delta_mu = abs(current.mu_tilde - previous.mu_tilde)

        if max(delta_lambda, delta_gamma, delta_mu) < tolerance:
            self.fixed_point = current
            return True

        return False

    def to_graph_representation(self) -> Dict[str, np.ndarray]:
        """
        Convert to graph representation for ML encoding.

        Mimics AlphaGeometry's graph structure for neural network input.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with node features, edge features, and adjacency
        """
        num_steps = len(self.trajectory)

        # Node features: each RG step is a node
        node_features = np.array([
            state.to_array() for state in self.trajectory
        ])  # Shape: (num_steps, 4)

        # Edge features: beta functions between steps
        edge_features = []
        adjacency = []

        for i in range(num_steps - 1):
            # Connect consecutive RG steps
            adjacency.append([i, i + 1])

            # Beta function as edge feature
            beta_key = f'beta_{i+1}'
            if beta_key in self.metadata:
                edge_features.append(self.metadata[beta_key])
            else:
                # Compute approximate beta from difference
                curr = self.trajectory[i + 1]
                prev = self.trajectory[i]
                dk = curr.k - prev.k

                if abs(dk) > SCALE_DIFF_TOLERANCE:
                    beta_lambda = (curr.lambda_tilde - prev.lambda_tilde) / dk
                    beta_gamma = (curr.gamma_tilde - prev.gamma_tilde) / dk
                    beta_mu = (curr.mu_tilde - prev.mu_tilde) / dk
                    edge_features.append([beta_lambda, beta_gamma, beta_mu])
                else:
                    edge_features.append([0.0, 0.0, 0.0])

        return {
            'node_features': node_features,
            'edge_features': np.array(edge_features) if edge_features else np.array([]).reshape(0, 3),
            'adjacency': np.array(adjacency) if adjacency else np.array([]).reshape(0, 2),
            'num_nodes': num_steps
        }

    def compute_action(self) -> float:
        """
        Compute cGFT action for current state.

        Returns
        -------
        float
            Action value
        """
        # Placeholder - should call actual IRH action computation
        if self.fixed_point is not None:
            # Action at fixed point
            fp = self.fixed_point
            self.action = (
                fp.lambda_tilde**2 +
                fp.gamma_tilde**2 +
                fp.mu_tilde**2
            )
        else:
            # Action for trajectory
            self.action = sum(
                state.lambda_tilde**2 +
                state.gamma_tilde**2 +
                state.mu_tilde**2
                for state in self.trajectory
            ) / len(self.trajectory)

        return self.action

    def get_current_state(self) -> CouplingState:
        """Get the most recent coupling state."""
        return self.trajectory[-1]

    def get_trajectory_length(self) -> int:
        """Get number of RG steps in trajectory."""
        return len(self.trajectory)

    def copy(self) -> HolographicState:
        """Create a deep copy of this state."""
        new_state = HolographicState()
        new_state.trajectory = [
            CouplingState(
                lambda_tilde=s.lambda_tilde,
                gamma_tilde=s.gamma_tilde,
                mu_tilde=s.mu_tilde,
                k=s.k,
                level=s.level
            )
            for s in self.trajectory
        ]
        new_state.fixed_point = self.fixed_point
        new_state.action = self.action
        new_state.metadata = dict(self.metadata)
        return new_state

    def __repr__(self) -> str:
        """String representation for debugging."""
        current = self.trajectory[-1]
        fp_str = f", FP={self.fixed_point is not None}" if self.fixed_point else ""
        return (
            f"HolographicState(λ̃={current.lambda_tilde:.4f}, "
            f"γ̃={current.gamma_tilde:.4f}, μ̃={current.mu_tilde:.4f}, "
            f"k={current.k:.4f}, steps={len(self.trajectory)}{fp_str})"
        )


# Example usage and validation
if __name__ == "__main__":
    print("Testing HolographicState implementation...")

    # Test 1: Basic initialization
    initial = CouplingState(
        lambda_tilde=10.0,
        gamma_tilde=10.0,
        mu_tilde=10.0,
        k=1.0
    )

    state = HolographicState(initial)
    print(f"✓ Initialization: {state}")

    # Test 2: RG flow simulation
    for i in range(10):
        k_new = 1.0 - 0.1 * (i + 1)

        # Simplified flow (should use actual beta functions)
        lambda_new = initial.lambda_tilde * (1 - 0.05 * (i + 1))
        gamma_new = initial.gamma_tilde * (1 - 0.05 * (i + 1))
        mu_new = initial.mu_tilde * (1 - 0.05 * (i + 1))

        new_state = CouplingState(
            lambda_tilde=lambda_new,
            gamma_tilde=gamma_new,
            mu_tilde=mu_new,
            k=k_new
        )

        beta_funcs = (-0.5, -0.5, -0.5)  # Dummy beta functions
        state.add_rg_step(new_state, beta_functions=beta_funcs)

    print(f"✓ RG Flow: {state.get_trajectory_length()} steps")

    # Test 3: Fixed point detection
    is_fp = state.check_fixed_point(tolerance=1e-2)
    print(f"✓ Fixed Point Detection: {is_fp}")

    # Test 4: Graph representation
    graph = state.to_graph_representation()
    print(f"✓ Graph Conversion: {graph['num_nodes']} nodes, "
          f"{len(graph['edge_features'])} edges")
    print(f"  Node features shape: {graph['node_features'].shape}")
    print(f"  Edge features shape: {graph['edge_features'].shape}")

    # Test 5: Action computation
    action = state.compute_action()
    print(f"✓ Action Computation: {action:.4f}")

    # Test 6: Copy
    state_copy = state.copy()
    print(f"✓ Copy: {state_copy.get_trajectory_length()} steps")

    # Test 7: Distance calculation
    state1 = CouplingState(10.0, 10.0, 10.0, 1.0)
    state2 = CouplingState(11.0, 11.0, 11.0, 1.0)
    distance = state1.distance_to(state2)
    print(f"✓ Distance: {distance:.4f}")

    print("\n✅ All basic tests passed!")
