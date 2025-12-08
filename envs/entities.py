from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


TIME_STEP = 1.0  # seconds


class DroneType(Enum):
    RECON = auto()
    ATTACK = auto()
    EW = auto()


class DroneStatus(Enum):
    ACTIVE = auto()
    DESTROYED = auto()
    INFILTRATED = auto()
    JAMMED = auto()


class OrderStatus(Enum):
    PENDING = auto()
    ASSIGNED = auto()
    COMPLETED = auto()
    FAILED = auto()


class WeaponType(Enum):
    CIWS = auto()
    SAM = auto()
    LASER = auto()
    EW = auto()
    IDLE = auto()


DRONE_TYPE_CONFIG = {
    DroneType.RECON: {
        "speed": 180.0,  # m/s
        "hp": 40.0,
        "rcs": 0.4,
        "altitude": 1500.0,
        "role_value": 0.3,
    },
    DroneType.ATTACK: {
        "speed": 150.0,
        "hp": 120.0,
        "rcs": 1.0,
        "altitude": 1200.0,
        "role_value": 1.0,
    },
    DroneType.EW: {
        "speed": 130.0,
        "hp": 80.0,
        "rcs": 0.8,
        "altitude": 2000.0,
        "role_value": 0.7,
    },
}


WEAPON_CONFIG = {
    WeaponType.CIWS: {
        "range": 4000.0,
        "max_altitude": 3000.0,
        "cooldown": 8.0,  # seconds between engagements
        "burst_rounds": 200,  # per engagement
        "ammo": 2000,
        "base_kill_prob": 0.55,
        "damage": 50.0,
        "cost": 1.0,  # relative cost per burst
    },
    WeaponType.SAM: {
        "range": 10000.0,
        "max_altitude": 6000.0,
        "cooldown": 20.0,
        "ammo": 8,
        "reload_time": 300.0,
        "base_kill_prob": 0.9,
        "damage": 120.0,
        "cost": 100.0,
    },
    WeaponType.LASER: {
        "range": 5000.0,
        "max_altitude": 3500.0,
        "cooldown": 15.0,
        "burst_time": 5.0,
        "base_kill_prob": 0.65,
        "damage": 80.0,
        "cost": 20.0,
        "duty_cycle": 30.0,
    },
    WeaponType.EW: {
        "range": 8000.0,
        "max_targets": 5,
        "spin_up": 3.0,
        "effect_duration": 10.0,
        "cost": 10.0,
    },
}


@dataclass
class WeaponModuleState:
    weapon_type: WeaponType
    ammo: float
    cooldown_timer: float = 0.0
    reload_timer: float = 0.0
    active_targets: List[int] = field(default_factory=list)
    energy_timer: float = 0.0  # for laser accumulated active time

    def is_available(self) -> bool:
        if self.weapon_type == WeaponType.EW:
            cfg = WEAPON_CONFIG[self.weapon_type]
            return len(self.active_targets) < cfg["max_targets"]
        if self.reload_timer > 0:
            return False
        return self.cooldown_timer <= 0 and self.ammo > 0

    def tick(self):
        if self.cooldown_timer > 0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - TIME_STEP)
        if self.reload_timer > 0:
            self.reload_timer = max(0.0, self.reload_timer - TIME_STEP)
        if self.weapon_type == WeaponType.LASER and self.energy_timer > 0:
            self.energy_timer = max(0.0, self.energy_timer - TIME_STEP)


@dataclass
class Order:
    order_id: int
    drone_id: int
    threat_level: float
    priority: float
    position: Tuple[float, float]
    heading: float
    velocity: float
    time_created: float
    estimated_time_to_target: float
    target_sector: int
    drone_type: DroneType
    status: OrderStatus = OrderStatus.PENDING
    assigned_to: Optional[int] = None
    attempts: int = 0

    def encode_features(self) -> Tuple[float, ...]:
        """Returns a normalized tuple used inside observations."""
        type_one_hot = {
            DroneType.RECON: (1.0, 0.0, 0.0),
            DroneType.ATTACK: (0.0, 1.0, 0.0),
            DroneType.EW: (0.0, 0.0, 1.0),
        }[self.drone_type]
        return (
            float(self.threat_level),
            float(self.priority),
            float(self.estimated_time_to_target / 120.0),
            float(self.position[0] / 10000.0),
            float(self.position[1] / 10000.0),
            float(self.heading / math.pi),
            *type_one_hot,
        )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance(p_a: Tuple[float, float], p_b: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(p_a) - np.array(p_b)))

