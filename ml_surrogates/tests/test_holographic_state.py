"""
Tests for holographic_state.py

Run with: pytest ml_surrogates/tests/test_holographic_state.py
"""

import pytest
import numpy as np
from ml_surrogates.engines.holographic_state import CouplingState, HolographicState


class TestCouplingState:
    """Tests for CouplingState class."""

    def test_initialization(self):
        """Test basic initialization."""
        state = CouplingState(
            lambda_tilde=10.0,
            gamma_tilde=5.0,
            mu_tilde=2.0,
            k=1.0
        )
        assert state.lambda_tilde == 10.0
        assert state.gamma_tilde == 5.0
        assert state.mu_tilde == 2.0
        assert state.k == 1.0
        assert state.level == 0

    def test_to_array(self):
        """Test conversion to numpy array."""
        state = CouplingState(10.0, 5.0, 2.0, 1.0)
        arr = state.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4,)
        assert np.allclose(arr, [10.0, 5.0, 2.0, 1.0])

    def test_distance_to(self):
        """Test distance calculation."""
        state1 = CouplingState(10.0, 10.0, 10.0, 1.0)
        state2 = CouplingState(11.0, 11.0, 11.0, 1.0)
        distance = state1.distance_to(state2)
        expected = np.sqrt(3)  # sqrt((1^2 + 1^2 + 1^2))
        assert abs(distance - expected) < 1e-10


class TestHolographicState:
    """Tests for HolographicState class."""

    def test_initialization_with_state(self):
        """Test initialization with provided state."""
        initial = CouplingState(10.0, 5.0, 2.0, 1.0)
        hstate = HolographicState(initial)
        assert len(hstate.trajectory) == 1
        assert hstate.trajectory[0] == initial

    def test_initialization_default(self):
        """Test default initialization."""
        hstate = HolographicState()
        assert len(hstate.trajectory) == 1
        assert hstate.trajectory[0].k == 1.0

    def test_add_rg_step(self):
        """Test adding RG flow steps."""
        hstate = HolographicState()

        new_state = CouplingState(9.0, 9.0, 9.0, 0.9)
        betas = (-0.1, -0.1, -0.1)
        hstate.add_rg_step(new_state, beta_functions=betas)

        assert len(hstate.trajectory) == 2
        assert hstate.trajectory[1].level == 1
        assert 'beta_1' in hstate.metadata

    def test_fixed_point_detection_true(self):
        """Test fixed point detection when converged."""
        hstate = HolographicState()

        # Add very small steps to simulate convergence
        for i in range(5):
            new_state = CouplingState(
                lambda_tilde=10.0 - 0.00001 * i,
                gamma_tilde=10.0 - 0.00001 * i,
                mu_tilde=10.0 - 0.00001 * i,
                k=1.0 - 0.1 * i
            )
            hstate.add_rg_step(new_state)

        is_fp = hstate.check_fixed_point(tolerance=1e-3)
        assert is_fp
        assert hstate.fixed_point is not None

    def test_fixed_point_detection_false(self):
        """Test fixed point detection when not converged."""
        hstate = HolographicState()

        new_state = CouplingState(5.0, 5.0, 5.0, 0.5)
        hstate.add_rg_step(new_state)

        is_fp = hstate.check_fixed_point()
        assert not is_fp
        assert hstate.fixed_point is None

    def test_graph_representation(self):
        """Test conversion to graph representation."""
        hstate = HolographicState()

        for i in range(5):
            new_state = CouplingState(
                lambda_tilde=10.0 - i,
                gamma_tilde=10.0 - i,
                mu_tilde=10.0 - i,
                k=1.0 - 0.1 * i
            )
            hstate.add_rg_step(new_state)

        graph = hstate.to_graph_representation()

        assert 'node_features' in graph
        assert 'edge_features' in graph
        assert 'adjacency' in graph
        assert 'num_nodes' in graph

        assert graph['num_nodes'] == 6
        assert graph['node_features'].shape == (6, 4)
        assert graph['edge_features'].shape[0] == 5  # num_nodes - 1
        assert graph['adjacency'].shape == (5, 2)

    def test_compute_action(self):
        """Test action computation."""
        hstate = HolographicState()
        action = hstate.compute_action()
        assert isinstance(action, float)
        assert action > 0

    def test_copy(self):
        """Test state copying."""
        hstate = HolographicState()
        hstate.add_rg_step(CouplingState(9.0, 9.0, 9.0, 0.9))

        hstate_copy = hstate.copy()

        assert len(hstate_copy.trajectory) == len(hstate.trajectory)
        assert hstate_copy.trajectory[0] is not hstate.trajectory[0]  # Deep copy
        assert hstate_copy.trajectory[0].lambda_tilde == hstate.trajectory[0].lambda_tilde

    def test_get_current_state(self):
        """Test getting current state."""
        hstate = HolographicState()
        current = hstate.get_current_state()
        assert current == hstate.trajectory[-1]

    def test_get_trajectory_length(self):
        """Test trajectory length."""
        hstate = HolographicState()
        assert hstate.get_trajectory_length() == 1

        hstate.add_rg_step(CouplingState(9.0, 9.0, 9.0, 0.9))
        assert hstate.get_trajectory_length() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
