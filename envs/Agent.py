from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from envs.entities import (
    DRONE_TYPE_CONFIG,
    TIME_STEP,
    DroneStatus,
    DroneType,
    clamp,
    distance,
)


@dataclass
class MovementNoise:
    """Random maneuver noise to emulate dynamic regrouping/splitting."""

    lateral_bias: float = 0.0
    persist_timer: float = 0.0

    def sample(self) -> float:
        if self.persist_timer <= 0.0:
            self.lateral_bias = random.uniform(-15.0, 15.0)
            self.persist_timer = random.randint(3, 8)
        self.persist_timer -= 1.0
        return self.lateral_bias


class Agent:
    """
    Red-force drone agent with simple kinematics and HP model.
    The blue-side dispatcher interacts with these through generated orders.
    """

    def __init__(
        self,
        agent_id: int,
        init_pos: Tuple[float, float],
        env,
        drone_type: DroneType,
        target: Tuple[float, float],
    ):
        cfg = DRONE_TYPE_CONFIG[drone_type]
        self.env = env
        self.agent_id = agent_id
        self.type = drone_type
        self.position = np.array(init_pos, dtype=float)
        self.target = np.array(target, dtype=float)
        self.altitude = cfg["altitude"]
        self.speed = cfg["speed"]
        self.max_speed = cfg["speed"]
        self.hp_max = cfg["hp"]
        self.hp = cfg["hp"]
        self.rcs = cfg["rcs"]
        self.role_value = cfg["role_value"]
        self.status = DroneStatus.ACTIVE
        self.time_alive = 0.0
        self.ew_timer = 0.0
        self.split_timer = random.uniform(15.0, 30.0)
        self.movement_noise = MovementNoise()
        self.cluster_id = None
        self.wave_id = 0
        self.leaked = False

    def is_active(self) -> bool:
        return self.status == DroneStatus.ACTIVE

    def is_destroyed(self) -> bool:
        return self.status == DroneStatus.DESTROYED

    def is_infiltrated(self) -> bool:
        return self.status == DroneStatus.INFILTRATED

    def _direction_to(self, target: np.ndarray) -> np.ndarray:
        vec = target - self.position
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.zeros_like(vec)
        return vec / norm

    def heading_rad(self) -> float:
        vec = self._direction_to(self.target)
        return math.atan2(vec[1], vec[0])

    def time_to_target(self) -> float:
        if not self.is_active():
            return 0.0
        effective_speed = max(1.0, self.current_speed())
        return distance(tuple(self.position), tuple(self.target)) / effective_speed

    def current_speed(self) -> float:
        disruption = 0.7 if self.ew_timer > 0 else 1.0
        return clamp(self.speed * disruption, self.speed * 0.4, self.max_speed)

    def distance_to_point(self, point: Tuple[float, float]) -> float:
        return distance(tuple(self.position), point)

    def apply_damage(self, damage: float) -> bool:
        if not self.is_active():
            return False
        self.hp -= damage
        if self.hp <= 0:
            self.status = DroneStatus.DESTROYED
            self.hp = 0
            return True
        return False

    def apply_ew_effect(self, duration: float):
        self.ew_timer = max(self.ew_timer, duration)
        if self.status == DroneStatus.ACTIVE:
            self.status = DroneStatus.JAMMED

    def _update_split_behavior(self):
        """Simulate dynamic regrouping to challenge dispatcher."""
        if not self.is_active():
            return
        self.split_timer -= TIME_STEP
        if self.split_timer <= 0:
            offset_angle = self.movement_noise.sample()
            heading = self.heading_rad() + math.radians(offset_angle)
            rotation = np.array([math.cos(heading), math.sin(heading)])
            self.target = self.env.pick_secondary_target(self.type, rotation)
            self.split_timer = random.uniform(12.0, 24.0)

    def step(self):
        if self.status == DroneStatus.DESTROYED:
            return
        if self.ew_timer > 0:
            self.ew_timer = max(0.0, self.ew_timer - TIME_STEP)
            if self.ew_timer == 0.0 and self.status == DroneStatus.JAMMED:
                self.status = DroneStatus.ACTIVE

        self._update_split_behavior()

        direction = self._direction_to(self.target)
        displacement = direction * self.current_speed() * TIME_STEP

        # add mild noise to emulate evasive maneuvers
        bias = self.movement_noise.sample()
        bias_heading = math.atan2(direction[1], direction[0]) + math.radians(bias)
        drift = np.array([math.cos(bias_heading), math.sin(bias_heading)]) * 0.1 * TIME_STEP

        self.position += displacement + drift
        self.time_alive += TIME_STEP

        if self.distance_to_point(tuple(self.target)) <= 500.0:
            self.status = DroneStatus.INFILTRATED

    def snapshot(self) -> np.ndarray:
        """Returns observation vector for this drone used by heuristic red controller."""
        type_encoding = {
            DroneType.RECON: (1.0, 0.0, 0.0),
            DroneType.ATTACK: (0.0, 1.0, 0.0),
            DroneType.EW: (0.0, 0.0, 1.0),
        }[self.type]
        return np.array(
            [
                self.position[0] / 100000.0,
                self.position[1] / 100000.0,
                self.altitude / 5000.0,
                self.current_speed() / 200.0,
                self.hp / max(1.0, self.hp_max),
                float(self.status == DroneStatus.ACTIVE),
                float(self.status == DroneStatus.JAMMED),
                self.time_to_target() / 120.0,
                *type_encoding,
            ],
            dtype=np.float32,
        )
