"""Tracking environment hook that preserves terminal online observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.envs import ManagerBasedRLEnv


class OnlineLearningManagerBasedRLEnv(ManagerBasedRLEnv):
    """Capture terminal tracking state before the base environment resets it.

    When module three is disabled this override immediately delegates to the
    base implementation and does not draw random numbers or change MDP values.
    """

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        command_manager = getattr(self, "command_manager", None)
        if command_manager is not None and "motion" in getattr(
            command_manager, "active_terms", ()
        ):
            command = command_manager.get_term("motion")
            if (
                getattr(command, "online_learning", None) is not None
                and not getattr(self, "_online_external_reset_in_progress", False)
            ):
                command.record_online_learning_step(env_ids)
        super()._reset_idx(env_ids)

    def reset(
        self,
        seed: int | None = None,
        env_ids: Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict]:
        """Reset once, then refresh online targets before computing observations."""

        command_manager = getattr(self, "command_manager", None)
        command = None
        if command_manager is not None and "motion" in getattr(
            command_manager, "active_terms", ()
        ):
            command = command_manager.get_term("motion")
        if command is None or getattr(command, "online_learning", None) is None:
            return super().reset(seed=seed, env_ids=env_ids, options=options)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self.recorder_manager.record_pre_reset(env_ids)
        if seed is not None:
            self.seed(seed)
        command.close_online_assignments_for_external_reset(env_ids)
        self._online_external_reset_in_progress = True
        try:
            self._reset_idx(env_ids)
        finally:
            self._online_external_reset_in_progress = False

        self.scene.write_data_to_sim()
        self.sim.forward()
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()
        command.refresh_online_state_after_external_reset(env_ids)
        self.recorder_manager.record_post_reset(env_ids)
        self.obs_buf = self.observation_manager.compute()
        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()
        return self.obs_buf, self.extras
