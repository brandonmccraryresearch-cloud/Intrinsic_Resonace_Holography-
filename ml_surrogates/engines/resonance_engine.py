"""
Resonance Engine - Symbolic Field Dynamics

THEORETICAL FOUNDATION: IRH v21.1 §1.2-1.3, Eqs. 1.12-1.14

Symbolic computation engine for holographic field dynamics.
Adapted from AlphaGeometry's DD+AR (Deductive Database + Algebraic Reasoning).

Key Operations:
- Beta function evaluation (Eq. 1.13)
- RG flow integration (Eq. 1.12 - Wetterich equation)
- Fixed point convergence (Eq. 1.14)
"""

from typing import Tuple, Optional, Callable
import numpy as np

try:
    from .holographic_state import CouplingState, HolographicState
except ImportError:
    from holographic_state import CouplingState, HolographicState


def _rk4_step(k1: float, k2: float, k3: float, k4: float, d_ln_k: float) -> float:
    """
    Compute RK4 weighted sum for a single coupling.
    
    Parameters
    ----------
    k1, k2, k3, k4 : float
        RK4 stage derivatives
    d_ln_k : float
        Step size in log scale
        
    Returns
    -------
    float
        Weighted sum for RK4 update
    """
    return (d_ln_k / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class ResonanceEngine:
    """
    Symbolic computation engine for IRH field dynamics.

    Analogous to AlphaGeometry's DDAR solver (ddar.py).
    Performs exact symbolic computations that ML surrogates will approximate.

    Attributes
    ----------
    tolerance : float
        Convergence tolerance for fixed points
    max_iterations : int
        Maximum RG flow integration steps
    _beta_lambda_fn : Optional[Callable]
        Custom beta function for λ̃
    _beta_gamma_fn : Optional[Callable]
        Custom beta function for γ̃
    _beta_mu_fn : Optional[Callable]
        Custom beta function for μ̃
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        beta_lambda_fn: Optional[Callable] = None,
        beta_gamma_fn: Optional[Callable] = None,
        beta_mu_fn: Optional[Callable] = None
    ):
        """
        Initialize resonance engine.

        Parameters
        ----------
        tolerance : float
            Numerical tolerance for convergence
        max_iterations : int
            Maximum RG flow steps
        beta_lambda_fn : Optional[Callable]
            Custom β_λ function (if None, uses default)
        beta_gamma_fn : Optional[Callable]
            Custom β_γ function (if None, uses default)
        beta_mu_fn : Optional[Callable]
            Custom β_μ function (if None, uses default)
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self._beta_lambda_fn = beta_lambda_fn
        self._beta_gamma_fn = beta_gamma_fn
        self._beta_mu_fn = beta_mu_fn

    def beta_lambda(
        self,
        lambda_tilde: float,
        gamma_tilde: float,
        mu_tilde: float,
        k: float
    ) -> float:
        """
        Beta function for λ̃ (IRH Eq. 1.13).

        β_λ = dλ̃/d(ln k)

        Parameters
        ----------
        lambda_tilde : float
            λ̃ coupling
        gamma_tilde : float
            γ̃ coupling
        mu_tilde : float
            μ̃ coupling
        k : float
            RG scale

        Returns
        -------
        float
            β_λ value
        """
        if self._beta_lambda_fn is not None:
            return self._beta_lambda_fn(lambda_tilde, gamma_tilde, mu_tilde, k)

        # Placeholder: simple running
        # TODO: Replace with IRH beta functions from theory
        return -0.1 * lambda_tilde * (1 + 0.01 * gamma_tilde)

    def beta_gamma(
        self,
        lambda_tilde: float,
        gamma_tilde: float,
        mu_tilde: float,
        k: float
    ) -> float:
        """
        Beta function for γ̃ (IRH Eq. 1.13).

        β_γ = dγ̃/d(ln k)

        Parameters
        ----------
        lambda_tilde : float
            λ̃ coupling
        gamma_tilde : float
            γ̃ coupling
        mu_tilde : float
            μ̃ coupling
        k : float
            RG scale

        Returns
        -------
        float
            β_γ value
        """
        if self._beta_gamma_fn is not None:
            return self._beta_gamma_fn(lambda_tilde, gamma_tilde, mu_tilde, k)

        # Placeholder
        return -0.1 * gamma_tilde * (1 + 0.01 * lambda_tilde)

    def beta_mu(
        self,
        lambda_tilde: float,
        gamma_tilde: float,
        mu_tilde: float,
        k: float
    ) -> float:
        """
        Beta function for μ̃ (IRH Eq. 1.13).

        β_μ = dμ̃/d(ln k)

        Parameters
        ----------
        lambda_tilde : float
            λ̃ coupling
        gamma_tilde : float
            γ̃ coupling
        mu_tilde : float
            μ̃ coupling
        k : float
            RG scale

        Returns
        -------
        float
            β_μ value
        """
        if self._beta_mu_fn is not None:
            return self._beta_mu_fn(lambda_tilde, gamma_tilde, mu_tilde, k)

        # Placeholder
        return -0.1 * mu_tilde * (1 + 0.01 * lambda_tilde)

    def compute_beta_functions(
        self,
        state: CouplingState
    ) -> Tuple[float, float, float]:
        """
        Compute all beta functions for current coupling state.

        Parameters
        ----------
        state : CouplingState
            Current coupling state

        Returns
        -------
        Tuple[float, float, float]
            (β_λ, β_γ, β_μ)
        """
        beta_l = self.beta_lambda(
            state.lambda_tilde,
            state.gamma_tilde,
            state.mu_tilde,
            state.k
        )
        beta_g = self.beta_gamma(
            state.lambda_tilde,
            state.gamma_tilde,
            state.mu_tilde,
            state.k
        )
        beta_m = self.beta_mu(
            state.lambda_tilde,
            state.gamma_tilde,
            state.mu_tilde,
            state.k
        )

        return (beta_l, beta_g, beta_m)

    def integrate_rg_flow(
        self,
        initial_state: CouplingState,
        k_final: float,
        num_steps: Optional[int] = None,
        method: str = 'euler'
    ) -> HolographicState:
        """
        Integrate RG flow from initial state to final scale.

        Solves Wetterich equation (IRH Eq. 1.12) numerically.
        Adapted from AlphaGeometry's solve() function in ddar.py.

        Parameters
        ----------
        initial_state : CouplingState
            Starting point in coupling space
        k_final : float
            Target RG scale
        num_steps : Optional[int]
            Number of integration steps (auto if None)
        method : str
            Integration method ('euler', 'rk4')

        Returns
        -------
        HolographicState
            Complete holographic state with trajectory
        """
        if num_steps is None:
            # Adaptive step count based on scale change
            num_steps = min(
                int(abs(np.log(initial_state.k / k_final)) * 100),
                self.max_iterations
            )

        holographic_state = HolographicState(initial_state)

        # Logarithmic integration (β = dk/d(ln k))
        ln_k_initial = np.log(initial_state.k)
        ln_k_final = np.log(k_final)
        d_ln_k = (ln_k_final - ln_k_initial) / num_steps

        current = initial_state

        for step in range(num_steps):
            # Compute beta functions
            beta_l, beta_g, beta_m = self.compute_beta_functions(current)

            # Integration step
            if method == 'euler':
                lambda_new = current.lambda_tilde + beta_l * d_ln_k
                gamma_new = current.gamma_tilde + beta_g * d_ln_k
                mu_new = current.mu_tilde + beta_m * d_ln_k

            elif method == 'rk4':
                # Runge-Kutta 4th order
                # k1
                k1_l, k1_g, k1_m = beta_l, beta_g, beta_m

                # k2
                mid_state = CouplingState(
                    lambda_tilde=current.lambda_tilde + 0.5 * k1_l * d_ln_k,
                    gamma_tilde=current.gamma_tilde + 0.5 * k1_g * d_ln_k,
                    mu_tilde=current.mu_tilde + 0.5 * k1_m * d_ln_k,
                    k=current.k * np.exp(0.5 * d_ln_k)
                )
                k2_l, k2_g, k2_m = self.compute_beta_functions(mid_state)

                # k3
                mid_state = CouplingState(
                    lambda_tilde=current.lambda_tilde + 0.5 * k2_l * d_ln_k,
                    gamma_tilde=current.gamma_tilde + 0.5 * k2_g * d_ln_k,
                    mu_tilde=current.mu_tilde + 0.5 * k2_m * d_ln_k,
                    k=current.k * np.exp(0.5 * d_ln_k)
                )
                k3_l, k3_g, k3_m = self.compute_beta_functions(mid_state)

                # k4
                end_state = CouplingState(
                    lambda_tilde=current.lambda_tilde + k3_l * d_ln_k,
                    gamma_tilde=current.gamma_tilde + k3_g * d_ln_k,
                    mu_tilde=current.mu_tilde + k3_m * d_ln_k,
                    k=current.k * np.exp(d_ln_k)
                )
                k4_l, k4_g, k4_m = self.compute_beta_functions(end_state)

                # Combined step using helper function
                lambda_new = current.lambda_tilde + _rk4_step(k1_l, k2_l, k3_l, k4_l, d_ln_k)
                gamma_new = current.gamma_tilde + _rk4_step(k1_g, k2_g, k3_g, k4_g, d_ln_k)
                mu_new = current.mu_tilde + _rk4_step(k1_m, k2_m, k3_m, k4_m, d_ln_k)

            else:
                raise ValueError(f"Unknown integration method: {method}")

            # Update scale
            k_new = current.k * np.exp(d_ln_k)

            current = CouplingState(
                lambda_tilde=lambda_new,
                gamma_tilde=gamma_new,
                mu_tilde=mu_new,
                k=k_new
            )

            holographic_state.add_rg_step(
                current,
                beta_functions=(beta_l, beta_g, beta_m)
            )

            # Check for fixed point convergence
            if holographic_state.check_fixed_point(self.tolerance):
                break

        return holographic_state

    def find_fixed_point(
        self,
        initial_guess: CouplingState,
        method: str = 'rg_flow',
        max_attempts: int = 5
    ) -> Optional[CouplingState]:
        """
        Find fixed point of RG flow (IRH Eq. 1.14).

        Fixed points satisfy β_λ = β_γ = β_μ = 0.

        Parameters
        ----------
        initial_guess : CouplingState
            Starting point for search
        method : str
            'rg_flow', 'newton', or 'hybrid'
        max_attempts : int
            Number of search attempts

        Returns
        -------
        Optional[CouplingState]
            Fixed point state if found, None otherwise
        """
        if method == 'rg_flow':
            # Flow to IR (k→0) and check for fixed point
            state = self.integrate_rg_flow(
                initial_guess,
                k_final=0.001,  # Near IR
                num_steps=self.max_iterations
            )
            return state.fixed_point

        elif method == 'newton':
            # Newton-Raphson on beta functions
            # TODO: Implement Jacobian and Newton iteration
            raise NotImplementedError("Newton method not yet implemented")

        elif method == 'hybrid':
            # Combine RG flow + Newton refinement
            # First, flow to get close
            state = self.integrate_rg_flow(
                initial_guess,
                k_final=0.001,
                num_steps=100
            )

            if state.fixed_point:
                return state.fixed_point

            # TODO: Then refine with Newton
            return None

        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_flow_jacobian(
        self,
        state: CouplingState
    ) -> np.ndarray:
        """
        Compute Jacobian matrix of beta functions.

        J_ij = ∂β_i/∂coupling_j

        Useful for stability analysis and Newton's method.

        Parameters
        ----------
        state : CouplingState
            Coupling state at which to evaluate Jacobian

        Returns
        -------
        np.ndarray
            3x3 Jacobian matrix
        """
        eps = 1e-8
        jacobian = np.zeros((3, 3))

        # Central difference approximation
        for j, (param, value) in enumerate([
            ('lambda', state.lambda_tilde),
            ('gamma', state.gamma_tilde),
            ('mu', state.mu_tilde)
        ]):
            # Perturb parameter
            state_plus = CouplingState(
                lambda_tilde=state.lambda_tilde + (eps if param == 'lambda' else 0),
                gamma_tilde=state.gamma_tilde + (eps if param == 'gamma' else 0),
                mu_tilde=state.mu_tilde + (eps if param == 'mu' else 0),
                k=state.k
            )

            state_minus = CouplingState(
                lambda_tilde=state.lambda_tilde - (eps if param == 'lambda' else 0),
                gamma_tilde=state.gamma_tilde - (eps if param == 'gamma' else 0),
                mu_tilde=state.mu_tilde - (eps if param == 'mu' else 0),
                k=state.k
            )

            beta_plus = self.compute_beta_functions(state_plus)
            beta_minus = self.compute_beta_functions(state_minus)

            jacobian[:, j] = [(bp - bm) / (2 * eps)
                             for bp, bm in zip(beta_plus, beta_minus)]

        return jacobian

    def check_stability(self, fixed_point: CouplingState) -> Tuple[bool, np.ndarray]:
        """
        Check stability of a fixed point.

        Stable if all eigenvalues of Jacobian have negative real parts.

        Parameters
        ----------
        fixed_point : CouplingState
            Fixed point to analyze

        Returns
        -------
        Tuple[bool, np.ndarray]
            (is_stable, eigenvalues)
        """
        jacobian = self.compute_flow_jacobian(fixed_point)
        eigenvalues = np.linalg.eigvals(jacobian)

        is_stable = np.all(np.real(eigenvalues) < 0)

        return is_stable, eigenvalues


# Example usage and validation
if __name__ == "__main__":
    print("Testing ResonanceEngine implementation...")

    engine = ResonanceEngine(tolerance=1e-4)

    initial = CouplingState(
        lambda_tilde=10.0,
        gamma_tilde=10.0,
        mu_tilde=10.0,
        k=1.0
    )

    # Test 1: Beta functions
    betas = engine.compute_beta_functions(initial)
    print(f"✓ Beta functions: β_λ={betas[0]:.4f}, β_γ={betas[1]:.4f}, β_μ={betas[2]:.4f}")

    # Test 2: RG flow integration (Euler)
    state_euler = engine.integrate_rg_flow(
        initial,
        k_final=0.1,
        num_steps=50,
        method='euler'
    )
    print(f"✓ Euler integration: {state_euler.get_trajectory_length()} steps")
    print(f"  Final: {state_euler.get_current_state()}")

    # Test 3: RG flow integration (RK4)
    state_rk4 = engine.integrate_rg_flow(
        initial,
        k_final=0.1,
        num_steps=50,
        method='rk4'
    )
    print(f"✓ RK4 integration: {state_rk4.get_trajectory_length()} steps")
    print(f"  Final: {state_rk4.get_current_state()}")

    # Test 4: Fixed point finding
    fp = engine.find_fixed_point(initial, method='rg_flow')
    if fp:
        print(f"✓ Fixed point found: {fp}")
    else:
        print("✓ No fixed point (expected for placeholder beta functions)")

    # Test 5: Jacobian
    jacobian = engine.compute_flow_jacobian(initial)
    print(f"✓ Jacobian shape: {jacobian.shape}")
    print(f"  Jacobian:\n{jacobian}")

    # Test 6: Stability
    test_fp = CouplingState(5.0, 5.0, 5.0, 0.1)
    is_stable, eigs = engine.check_stability(test_fp)
    print(f"✓ Stability analysis: stable={is_stable}")
    print(f"  Eigenvalues: {eigs}")

    print("\n✅ All ResonanceEngine tests completed!")
