from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from envs.entities import (
    DRONE_TYPE_CONFIG,
    WEAPON_CONFIG,
    DroneStatus,
    DroneType,
    Order,
    OrderStatus,
    TIME_STEP,
    WeaponModuleState,
    WeaponType,
    clamp,
    distance,
)


ORDER_FEATURES = len(
    Order(
        order_id=-1,
        drone_id=-1,
        threat_level=0.0,
        priority=0.0,
        position=(0.0, 0.0),
        heading=0.0,
        velocity=0.0,
        time_created=0.0,
        estimated_time_to_target=0.0,
        target_sector=0,
        drone_type=DroneType.ATTACK,
    ).encode_features()
)
MAX_ORDERS_PER_AGENT = 6
WEAPON_CHOICES: Sequence[WeaponType] = (
    WeaponType.CIWS,
    WeaponType.SAM,
    WeaponType.LASER,
    WeaponType.EW,
)


def _pad_list(values: Iterable[float], length: int) -> List[float]:
    result = list(values)
    if len(result) < length:
        result.extend([0.0] * (length - len(result)))
    else:
        result = result[:length]
    return result


@dataclass
class EngagementOutcome:
    order_id: Optional[int]
    success: bool
    threat: float = 0.0
    cost: float = 0.0
    kill: bool = False
    weapon_type: WeaponType = WeaponType.IDLE
    message: str = ""


class DefenceAgent:
    """Blue-side brigade dispatcher controlled by reinforcement learning."""

    def __init__(
        self,
        agent_id: int,
        init_pos: Tuple[float, float],
        env,
        reward_weight: Dict[str, float],
    ):
        self.env = env
        self.agent_id = agent_id
        self.position = np.array(init_pos, dtype=float)
        self.action_dim = len(WEAPON_CHOICES) * MAX_ORDERS_PER_AGENT + 1
        self.weapon_modules: Dict[WeaponType, WeaponModuleState] = {
            WeaponType.CIWS: WeaponModuleState(WeaponType.CIWS, ammo=WEAPON_CONFIG[WeaponType.CIWS]["ammo"]),
            WeaponType.SAM: WeaponModuleState(WeaponType.SAM, ammo=WEAPON_CONFIG[WeaponType.SAM]["ammo"]),
            WeaponType.LASER: WeaponModuleState(WeaponType.LASER, ammo=WEAPON_CONFIG[WeaponType.LASER]["duty_cycle"]),
            WeaponType.EW: WeaponModuleState(WeaponType.EW, ammo=float("inf")),
        }
        self.reward_weight = reward_weight or {}
        self.reward = 0.0
        self.hp = 100.0
        self.attack_position = [self.position[0], self.position[1]]
        self._available_orders: List[Order] = []
        self.obs: List[float] = []
        self.metrics = {
            "order_completion": 0.0,
            "resource_cost": 0.0,
            "threat_neutralised": 0.0,
            "intercept_success": 0.0,
        }
        self._recent_outcome = 0.0

    def set_available_orders(self, orders: List[Order]):
        self._available_orders = orders[:MAX_ORDERS_PER_AGENT]

    def tick_modules(self):
        for module in self.weapon_modules.values():
            module.tick()

    def _weapon_available(self, weapon_type: WeaponType) -> bool:
        return self.weapon_modules[weapon_type].is_available()

    def _engage(
        self,
        order: Order,
        weapon_type: WeaponType,
        drone_lookup: Dict[int, "Agent"],
    ) -> EngagementOutcome:
        module = self.weapon_modules[weapon_type]
        cfg = WEAPON_CONFIG[weapon_type]
        drone = drone_lookup.get(order.drone_id)

        if drone is None or drone.is_destroyed():
            module.cooldown_timer = cfg.get("cooldown", 5.0)
            return EngagementOutcome(order.order_id, True, threat=order.threat_level, weapon_type=weapon_type, message="target already destroyed")

        threat = order.threat_level
        drone_pos = tuple(drone.position)
        dist = distance(drone_pos, tuple(self.position))

        if weapon_type != WeaponType.EW:
            if dist > cfg["range"]:
                return EngagementOutcome(order.order_id, False, threat=threat, weapon_type=weapon_type, message="target out of range")
            if drone.altitude > cfg.get("max_altitude", float("inf")):
                return EngagementOutcome(order.order_id, False, threat=threat, weapon_type=weapon_type, message="target altitude too high")

        module.cooldown_timer = cfg.get("cooldown", 5.0)
        module.ammo = max(0.0, module.ammo - cfg.get("burst_rounds", 1))
        cost = cfg.get("cost", 0.0)

        if weapon_type == WeaponType.LASER:
            module.energy_timer += cfg.get("burst_time", 3.0)
            if module.energy_timer >= WEAPON_CONFIG[WeaponType.LASER]["duty_cycle"]:
                module.reload_timer = cfg.get("cooldown", 15.0)
                module.energy_timer = 0.0

        if weapon_type == WeaponType.SAM and module.ammo <= 0:
            module.reload_timer = cfg.get("reload_time", 300.0)

        if weapon_type == WeaponType.EW:
            module.active_targets.append(order.drone_id)
            drone.apply_ew_effect(cfg["effect_duration"])
            return EngagementOutcome(
                order.order_id,
                success=True,
                threat=threat,
                cost=cost,
                weapon_type=weapon_type,
                message="EW disruption applied",
            )

        base_prob = cfg["base_kill_prob"]
        range_factor = clamp(1.0 - dist / cfg["range"], 0.1, 1.0)
        time_factor = clamp(1.0 - order.estimated_time_to_target / 180.0, 0.2, 1.1)
        type_weight = DRONE_TYPE_CONFIG[drone.type]["role_value"]
        kill_prob = clamp(base_prob * range_factor * (0.8 + 0.4 * threat) * time_factor, 0.05, 0.98)

        roll = np.random.random()
        kill = bool(roll < kill_prob)
        if kill:
            damage = cfg.get("damage", 50.0)
            destroyed = drone.apply_damage(damage)
            if destroyed:
                outcome = EngagementOutcome(order.order_id, True, threat=threat, cost=cost, kill=True, weapon_type=weapon_type)
                return outcome
        else:
            if weapon_type == WeaponType.CIWS:
                drone.apply_damage(cfg.get("damage", 30.0) * 0.5)

        return EngagementOutcome(
            order.order_id,
            success=kill,
            threat=threat,
            cost=cost,
            kill=kill,
            weapon_type=weapon_type,
            message="hit" if kill else "miss",
        )

    def _decode_action(self, action: np.ndarray) -> List[Tuple[int, WeaponType]]:
        ranked_idx = np.argsort(action)[::-1]
        decoded: List[Tuple[int, WeaponType]] = []
        for idx in ranked_idx:
            if idx >= self.action_dim - 1:
                decoded.append((-1, WeaponType.IDLE))
                continue
            order_slot = idx // len(WEAPON_CHOICES)
            weapon_idx = idx % len(WEAPON_CHOICES)
            decoded.append((order_slot, WEAPON_CHOICES[weapon_idx]))
        return decoded

    def update(self, action: np.ndarray, drone_lookup: Dict[int, "Agent"]) -> EngagementOutcome:
        self.reward = 0.0
        self.tick_modules()
        ew_module = self.weapon_modules.get(WeaponType.EW)
        if ew_module and ew_module.active_targets:
            ew_module.active_targets = [
                drone_id
                for drone_id in ew_module.active_targets
                if drone_id in drone_lookup and drone_lookup[drone_id].ew_timer > 0
            ]

        if self.hp <= 0:
            self.obs = self.get_obs()
            return EngagementOutcome(None, False, weapon_type=WeaponType.IDLE, message="agent destroyed")

        decoded_preferences = self._decode_action(action)
        outcome = EngagementOutcome(None, False, weapon_type=WeaponType.IDLE)

        for order_slot, weapon_type in decoded_preferences:
            if weapon_type == WeaponType.IDLE:
                outcome = EngagementOutcome(None, False, weapon_type=WeaponType.IDLE, message="idle selection")
                break
            if order_slot < 0 or order_slot >= len(self._available_orders):
                continue
            order = self._available_orders[order_slot]
            if order.status not in (OrderStatus.PENDING, OrderStatus.ASSIGNED):
                continue
            if not self._weapon_available(weapon_type):
                continue
            order.status = OrderStatus.ASSIGNED
            order.assigned_to = self.agent_id
            outcome = self._engage(order, weapon_type, drone_lookup)
            self.attack_position = list(order.position)
            break

        self._apply_reward(outcome)
        self.obs = self.get_obs()
        return outcome

    def _apply_reward(self, outcome: EngagementOutcome):
        threat = outcome.threat
        reward = 0.0

        if outcome.weapon_type == WeaponType.IDLE:
            reward += -0.1 * len(self._available_orders)

        if outcome.success:
            reward += 12.0 * threat
            self.metrics["order_completion"] += 1.0
            if outcome.kill:
                reward += 8.0 * threat
                self.metrics["threat_neutralised"] += threat
                self.metrics["intercept_success"] += 1.0
        else:
            reward -= 6.0 * threat

        reward -= 0.05 * outcome.cost
        if outcome.weapon_type == WeaponType.SAM and threat < 0.4:
            reward -= 4.0
        if outcome.weapon_type == WeaponType.CIWS and threat > 0.7:
            reward -= 2.0

        self.metrics["resource_cost"] += outcome.cost
        self.reward = reward
        self._recent_outcome = reward

    def get_obs(self) -> List[float]:
        order_features: List[float] = []
        for order in self._available_orders:
            order_features.extend(order.encode_features())
        order_features = _pad_list(order_features, MAX_ORDERS_PER_AGENT * ORDER_FEATURES)

        module_features: List[float] = []
        for weapon_type in WEAPON_CHOICES:
            module = self.weapon_modules[weapon_type]
            cfg = WEAPON_CONFIG[weapon_type]
            availability = 1.0 if module.is_available() else 0.0
            ammo_capacity = cfg.get("ammo", 1.0)
            if weapon_type == WeaponType.LASER:
                ammo_capacity = cfg.get("duty_cycle", 1.0)
            module_features.extend(
                [
                    clamp(module.ammo / max(1.0, ammo_capacity), 0.0, 1.0),
                    clamp(module.cooldown_timer / max(1.0, cfg.get("cooldown", 1.0)), 0.0, 1.0),
                    clamp(module.reload_timer / max(1.0, cfg.get("reload_time", 300.0)), 0.0, 1.0),
                    availability,
                    clamp(len(module.active_targets) / max(1.0, cfg.get("max_targets", 1.0)), 0.0, 1.0),
                ]
            )

        aggregated = [
            len(self._available_orders) / float(MAX_ORDERS_PER_AGENT),
            self.metrics["intercept_success"] % 10 / 10.0,
            clamp(self._recent_outcome / 20.0, -1.0, 1.0),
        ]
        observation = order_features + module_features + aggregated
        return observation
