"""
Cost and constraint objective for cost-guided diffusion inference.

This objective is now specialized for inference-time cylinder avoidance on
the can-with-obstacle image task. The diffusion model samples in normalized
action space, so this module first reconstructs world-frame end-effector
positions before applying any geometry-based penalties.
"""

from typing import Optional, Tuple

import torch


class CostConstraintObjective:
    """
    Augmented-Lagrangian style objective for cost-guided diffusion.

    J(x) = cost_weight * cost(x)
         + equality_penalty(x)
         + inequality_penalty(x)

    - x is typically a batch of trajectories: shape (B, T, D),
      but code only assumes the last dimension is the feature dim.

    To customize behavior, edit:
      - cost(self, x)
      - equality_penalty(self, x)
      - inequality_penalty(self, x)

    All gradients are obtained via autograd; no hardcoded derivatives.
    """

    # -------- Hyperparameters (single place to edit) --------
    # Overall weight multiplying the cost term
    cost_weight: float = 1.0

    # Penalty weights for equality and inequality constraints
    c_eq: float = 50.0
    c_ineq: float = 50.0

    # Smoothness for soft inequality constraints (softplus)
    softplus_beta: float = 10.0

    # Numerical epsilon
    eps_numerical: float = 1e-8

    # Optional target terminal state for equality constraint
    # (e.g. desired final state). Shape: (D,) or broadcastable to (B, D).
    target_state: Optional[torch.Tensor] = None

    def __init__(
        self,
        cost_weight: Optional[float] = None,
        c_eq: Optional[float] = None,
        c_ineq: Optional[float] = None,
        softplus_beta: Optional[float] = None,
        target_state: Optional[torch.Tensor] = None,
    ):
        if cost_weight is not None:
            self.cost_weight = cost_weight
        if c_eq is not None:
            self.c_eq = c_eq
        if c_ineq is not None:
            self.c_ineq = c_ineq
        if softplus_beta is not None:
            self.softplus_beta = softplus_beta
        if target_state is not None:
            self.target_state = target_state

        # Runtime context injected by the live inference policy.
        self.action_normalizer = None
        self.action_dim: Optional[int] = None

        # Hardcoded geometry for can_with_obs_image_abs.
        self.cylinder_center_xy = (0.15, 0.29)
        self.cylinder_radius = 0.02
        self.cylinder_epsilon = 0.02

    def set_runtime_context(self, action_normalizer, action_dim: Optional[int]) -> None:
        self.action_normalizer = action_normalizer
        self.action_dim = action_dim

    def _zeros(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def _as_like(self, values, ref: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(values, device=ref.device, dtype=ref.dtype)

    def _action_trajectory_world(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.action_normalizer is None:
            return None

        action_dim = self.action_dim if self.action_dim is not None else x.shape[-1]
        if action_dim <= 0 or x.shape[-1] < action_dim:
            return None

        action_traj = x[..., :action_dim]
        return self.action_normalizer.unnormalize(action_traj)

    def _eef_positions_world(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        action_world = self._action_trajectory_world(x)
        if action_world is None or action_world.shape[-1] < 3:
            return None
        return action_world[..., :3]

    def _cylinder_avoidance_penalty(self, position: torch.Tensor) -> torch.Tensor:
        cylinder_center_xy = self._as_like(self.cylinder_center_xy, position)
        effective_radius = self.cylinder_radius + self.cylinder_epsilon

        delta_xy = position[..., :2] - cylinder_center_xy
        squared_distance = delta_xy.square().sum(dim=-1)
        violation = torch.clamp(effective_radius ** 2 - squared_distance, min=0.0)
        return violation


    # ------------- EDIT THESE THREE METHODS ONLY -------------

    def cost(self, x: torch.Tensor) -> torch.Tensor:
        """
        Placeholder cost term.

        Uses world-frame end-effector smoothness over the full trajectory,
        but is disabled by multiplying the result by zero.
        """
        positions = self._eef_positions_world(x)
        if positions is None or positions.shape[1] < 2:
            return self._zeros(x)

        deltas = positions[:, 1:, :] - positions[:, :-1, :]
        cost_val = deltas.square().reshape(x.shape[0], -1).sum(dim=1)
        return cost_val * 0.0

    def equality_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Placeholder equality-like term.

        Penalizes terminal velocity as a simple terminal consistency proxy,
        but is disabled by multiplying the result by zero.
        """
        positions = self._eef_positions_world(x)
        if positions is None or positions.shape[1] < 2:
            return self._zeros(x)

        terminal_step = positions[:, -1, :] - positions[:, -2, :]
        eq_pen = self.c_eq * terminal_step.square().sum(dim=1)
        return eq_pen * 0.0

    def inequality_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trajectory-wide cylinder avoidance penalty in world coordinates.

        The cylinder is treated as infinitely tall, so only the xy-plane is
        constrained. All sampled action steps contribute to the penalty.
        """
        positions = self._eef_positions_world(x)
        if positions is None:
            return self._zeros(x)

        violation = self._cylinder_avoidance_penalty(positions)
        ineq_pen = self.c_ineq * violation.reshape(x.shape[0], -1).sum(dim=1)
        return ineq_pen

    # ------------- Derived helpers (do NOT edit) -------------

    def J(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augmented Lagrangian:
        J(x) = cost_weight * cost(x)
             + equality_penalty(x)
             + inequality_penalty(x)
        """
        return (
            self.cost_weight * self.cost(x)
            + self.equality_penalty(x)
            + self.inequality_penalty(x)
        )

    def J_and_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (J(x), grad_J(x)) using autograd.

        - J(x): (B,)
        - grad_J(x): same shape as x
        """
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            J_total = self.J(x_req)  # (B,)
            grad_J = torch.autograd.grad(
                J_total.sum(), x_req, create_graph=False
            )[0]
        return J_total.detach(), grad_J.detach()

    def J_only(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: J(x) without computing gradients."""
        return self.J(x)
