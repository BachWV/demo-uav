from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np

from envs.Agent import Agent
from envs.DefenceAgent import DefenceAgent, MAX_ORDERS_PER_AGENT
from envs.Radar import Radar
from envs.entities import (
    DRONE_TYPE_CONFIG,
    DroneStatus,
    DroneType,
    Order,
    OrderStatus,
    TIME_STEP,
    WeaponType,
    clamp,
    distance,
)
import reward_weight.weight01


class Swarm:
    """Order-driven confrontation scenario between red drone swarm and blue defence brigades."""

    def __init__(self, agent_num: int, args_all):
        self.args_all = args_all
        self.agent_num = agent_num
        self.current_time = 0.0
        self.step_count = 0
        self.random = np.random.RandomState(args_all.seed)
        self.high_value_targets = [
            np.array([0.0, 0.0]),
            np.array([0.0, 7500.0]),
            np.array([0.0, -7500.0]),
        ]
        self._build_blue_network()
        self.next_order_id = 0
        self.next_drone_id = 0
        self.order_pool: List[Order] = []
        self.order_map: Dict[int, Order] = {}
        self.agents: List[Agent] = []
        self.metrics = {
            "orders_completed": 0,
            "orders_failed": 0,
            "threat_neutralised": 0.0,
            "resource_cost": 0.0,
            "intercepts": 0,
            "leaks": 0,
        }
        self.trace_dir = None
        self.trace_file = None
        self.trace_buffer = {
            "drones": {},
            "orders": [],
            "defence": {},
            "metrics": [],
        }
        self.make_dir()
        self.reset_state()

    def _build_blue_network(self):
        self.radar_grid = [
            Radar("radar_central", -5000.0, 0.0),
            Radar("radar_north", -6000.0, 8000.0),
            Radar("radar_south", -6000.0, -8000.0),
        ]
        brigade_positions = [
            (-3000.0, 0.0),
            (-3000.0, 8000.0),
            (-3000.0, -8000.0),
        ]
        self.defence_agents = [
            DefenceAgent(agent_id=i, init_pos=brigade_positions[i], env=self, reward_weight={})
            for i in range(self.agent_num)
        ]

    def make_dir(self):
        base = Path(os.path.abspath("./")) / "trace" / self.args_all.scenario_name
        render_dir = base / ("render" if self.args_all.use_render else "train")
        render_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir = render_dir

    def reset_state(self):
        self.current_time = 0.0
        self.step_count = 0
        self.next_order_id = 0
        self.order_pool.clear()
        self.order_map.clear()
        self.metrics = {k: 0 for k in self.metrics}
        self.trace_buffer = {
            "drones": {},
            "orders": [],
            "defence": {},
            "metrics": [],
        }
        self._start_trace_file()
        for defence_agent in self.defence_agents:
            self.trace_buffer["defence"][defence_agent.agent_id] = {
                "reward": [],
                "attack_position": [],
                "available_orders": [],
            }
        self.agents = []
        self._spawn_wave()
        self._flush_trace()

    def _spawn_wave(self):
        wave_size = 30
        counts = {
            DroneType.ATTACK: int(wave_size * 0.8),
            DroneType.RECON: max(1, int(wave_size * 0.1)),
            DroneType.EW: max(1, wave_size - int(wave_size * 0.8) - int(wave_size * 0.1)),
        }
        spawn_radius = 28000.0
        for drone_type, count in counts.items():
            for _ in range(count):
                angle = self.random.uniform(-math.pi / 4, math.pi / 4)
                x = -spawn_radius
                y = self.random.uniform(-9000.0, 9000.0) + np.tan(angle) * 4000.0
                target = random.choice(self.high_value_targets)
                agent = Agent(
                    agent_id=self.next_drone_id,
                    init_pos=(x, y),
                    env=self,
                    drone_type=drone_type,
                    target=tuple(target),
                )
                agent.wave_id = 0
                self.agents.append(agent)
                self.trace_buffer["drones"][agent.agent_id] = {"x": [], "y": [], "hp": [], "status": []}
                self.next_drone_id += 1

    def pick_secondary_target(self, drone_type: DroneType, direction_vector: np.ndarray) -> np.ndarray:
        base_target = random.choice(self.high_value_targets)
        offset = direction_vector * self.random.uniform(1500.0, 4000.0)
        return base_target + offset

    def _sector_for(self, position: np.ndarray) -> int:
        if position[1] > 4000.0:
            return 1
        if position[1] < -4000.0:
            return 2
        return 0

    def _ensure_order(self, drone: Agent):
        if drone.agent_id in self.order_map:
            order = self.order_map[drone.agent_id]
            order.position = tuple(drone.position)
            order.heading = drone.heading_rad()
            order.velocity = drone.current_speed()
            order.estimated_time_to_target = drone.time_to_target()
            order.priority = self._compute_priority(drone, order.threat_level)
            order.target_sector = self._sector_for(drone.position)
            return

        threat = self._compute_threat(drone)
        priority = self._compute_priority(drone, threat)
        order = Order(
            order_id=self.next_order_id,
            drone_id=drone.agent_id,
            threat_level=threat,
            priority=priority,
            position=tuple(drone.position),
            heading=drone.heading_rad(),
            velocity=drone.current_speed(),
            time_created=self.current_time,
            estimated_time_to_target=drone.time_to_target(),
            target_sector=self._sector_for(drone.position),
            drone_type=drone.type,
        )
        self.order_pool.append(order)
        self.order_map[drone.agent_id] = order
        self.next_order_id += 1

    def _compute_priority(self, drone: Agent, threat: float) -> float:
        tti = clamp(drone.time_to_target() / 180.0, 0.0, 1.0)
        return clamp(0.6 * threat + 0.4 * (1.0 - tti), 0.0, 1.0)

    def _compute_threat(self, drone: Agent) -> float:
        role = DRONE_TYPE_CONFIG[drone.type]["role_value"]
        distance_factor = clamp(1.0 - distance(tuple(drone.position), tuple(random.choice(self.high_value_targets))) / 30000.0, 0.0, 1.0)
        time_factor = clamp(1.0 - drone.time_to_target() / 180.0, 0.0, 1.0)
        return clamp(0.5 * role + 0.3 * distance_factor + 0.2 * time_factor, 0.0, 1.0)

    def _cleanup_orders(self):
        active_orders = []
        lookup = self._drone_lookup()
        for order in self.order_pool:
            drone = lookup.get(order.drone_id)
            if drone is None or drone.is_destroyed():
                if drone is None or not getattr(drone, "leaked", False):
                    order.status = OrderStatus.COMPLETED
                    self.metrics["orders_completed"] += 1
                    self.metrics["threat_neutralised"] += order.threat_level
                else:
                    order.status = OrderStatus.FAILED
                    self.metrics["orders_failed"] += 1
                continue
            if getattr(drone, "leaked", False):
                order.status = OrderStatus.FAILED
                self.metrics["orders_failed"] += 1
                continue
            if self.current_time - order.time_created > 120.0:
                order.status = OrderStatus.FAILED
                self.metrics["orders_failed"] += 1
                continue
            active_orders.append(order)
        self.order_pool = active_orders
        self.order_map = {order.drone_id: order for order in self.order_pool}

    def _drone_lookup(self) -> Dict[int, Agent]:
        return {agent.agent_id: agent for agent in self.agents if not agent.is_destroyed()}

    def _assign_orders_to_defence(self):
        lookup = self._drone_lookup()
        orders_by_sector = {0: [], 1: [], 2: []}
        for order in self.order_pool:
            if order.status in (OrderStatus.PENDING, OrderStatus.ASSIGNED):
                orders_by_sector[order.target_sector].append(order)
        for agent in self.defence_agents:
            sector_idx = min(max(orders_by_sector.keys()), agent.agent_id)
            sector_orders = sorted(orders_by_sector[sector_idx], key=lambda o: o.priority, reverse=True)
            agent.set_available_orders(sector_orders)

    def _start_trace_file(self):
        if self.trace_dir is None:
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        base_name = f"run_{timestamp}"
        suffix = 0
        trace_path = self.trace_dir / f"{base_name}.json"
        while trace_path.exists():
            suffix += 1
            trace_path = self.trace_dir / f"{base_name}_{suffix}.json"
        self.trace_file = trace_path

    def _update_traces(self):
        for agent in self.agents:
            buf = self.trace_buffer["drones"].setdefault(
                agent.agent_id, {"x": [], "y": [], "hp": [], "status": []}
            )
            buf["x"].append(float(agent.position[0]))
            buf["y"].append(float(agent.position[1]))
            buf["hp"].append(float(agent.hp))
            buf["status"].append(agent.status.name)

        orders_snapshot = []
        for order in self.order_pool:
            orders_snapshot.append(
                {
                    "order_id": int(order.order_id),
                    "drone_id": int(order.drone_id),
                    "threat": float(order.threat_level),
                    "priority": float(order.priority),
                    "time_to_target": float(order.estimated_time_to_target),
                    "position": [float(order.position[0]), float(order.position[1])],
                    "status": order.status.name,
                    "weapon_assigned_to": order.assigned_to,
                }
            )
        self.trace_buffer["orders"].append(
            {
                "step": int(self.step_count),
                "time": float(self.current_time),
                "orders": orders_snapshot,
            }
        )

        for defence_agent in self.defence_agents:
            defence_buf = self.trace_buffer["defence"].setdefault(
                defence_agent.agent_id,
                {
                    "reward": [],
                    "attack_position": [],
                    "available_orders": [],
                },
            )
            defence_buf["reward"].append(float(defence_agent.reward))
            defence_buf["attack_position"].append(
                [float(defence_agent.attack_position[0]), float(defence_agent.attack_position[1])]
            )
            defence_buf["available_orders"].append(
                [order.order_id for order in getattr(defence_agent, "_available_orders", [])]
            )

        metric_entry = {
            "time": float(self.current_time),
            "step": int(self.step_count),
        }
        for key, value in self.metrics.items():
            if isinstance(value, (np.floating, float)):
                metric_entry[key] = float(value)
            else:
                metric_entry[key] = int(value)
        self.trace_buffer["metrics"].append(metric_entry)
        self._flush_trace()

    def _flush_trace(self):
        if self.trace_file is None:
            return
        payload = {
            "scenario": self.args_all.scenario_name,
            "seed": self.args_all.seed,
            "current_time": float(self.current_time),
            "step": int(self.step_count),
            "metrics": {
                key: float(val) if isinstance(val, (np.floating, float)) else int(val)
                for key, val in self.metrics.items()
            },
            "trace": self.trace_buffer,
        }
        with open(self.trace_file, "w", encoding="utf-8") as trace_file:
            json.dump(payload, trace_file, ensure_ascii=False, indent=2)

    def update(self, actions, defence_actions):
        self.current_time += TIME_STEP
        self.step_count += 1
        for agent in self.agents:
            agent.step()

        stable_tracks = []
        for radar in self.radar_grid:
            stable, _ = radar.scan(self.agents, self.current_time)
            stable_tracks.extend(stable)

        for drone in stable_tracks:
            if drone.is_active():
                self._ensure_order(drone)

        self._cleanup_orders()
        self._assign_orders_to_defence()
        drone_lookup = self._drone_lookup()

        outcomes = []
        for defence_agent, action in zip(self.defence_agents, defence_actions):
            outcome = defence_agent.update(action, drone_lookup)
            outcomes.append(outcome)
            if outcome.order_id is None:
                continue
            if outcome.weapon_type == WeaponType.EW:
                self.metrics["resource_cost"] += outcome.cost
                continue
            if outcome.success:
                self.metrics["orders_completed"] += 1
                if outcome.kill:
                    self.metrics["intercepts"] += 1
                    self.metrics["threat_neutralised"] += outcome.threat
                self.metrics["resource_cost"] += outcome.cost
            elif not outcome.success and outcome.order_id is not None:
                self.metrics["orders_failed"] += 1
                self.metrics["resource_cost"] += outcome.cost

        for agent in self.agents:
            if agent.status == DroneStatus.INFILTRATED and not agent.leaked:
                self.metrics["leaks"] += 1
                sector = self._sector_for(agent.position)
                for defence_agent in self.defence_agents:
                    if defence_agent.agent_id == sector:
                        defence_agent.reward -= 15.0
                agent.leaked = True
                agent.status = DroneStatus.DESTROYED

        self._cleanup_orders()
        self._update_traces()

    def summary(self):
        total_orders = self.metrics["orders_completed"] + self.metrics["orders_failed"]
        intercept_rate = 0.0
        if total_orders > 0:
            intercept_rate = self.metrics["intercepts"] / max(1, total_orders)
        return {
            "orders": total_orders,
            "completed": self.metrics["orders_completed"],
            "failed": self.metrics["orders_failed"],
            "intercepts": self.metrics["intercepts"],
            "leaks": self.metrics["leaks"],
            "threat_neutralised": self.metrics["threat_neutralised"],
            "resource_cost": self.metrics["resource_cost"],
            "intercept_rate": intercept_rate,
        }

    def reset(self):
        self.reset_state()


class MyEnv:
    """
    Gym-like wrapper exposing the swarm scenario to the training loop.
    """

    def __init__(self, args_all):
        args_all.reward_weight = reward_weight.weight01
        self.agent_num = args_all.num_agents
        self.swarm = Swarm(agent_num=self.agent_num, args_all=args_all)
        self.obs_dim = len(self.swarm.defence_agents[0].get_obs())
        self.action_dim = self.swarm.defence_agents[0].action_dim
        self.viewer = None

    def reset(self):
        self.swarm.reset()
        plane_obs = [agent.snapshot() for agent in self.swarm.agents]
        defence_obs = [np.array(agent.get_obs(), dtype=np.float32) for agent in self.swarm.defence_agents]
        return plane_obs, defence_obs

    def step(self, actions, defence_actions):
        self.swarm.update(actions, defence_actions)

        plane_obs = [agent.snapshot() for agent in self.swarm.agents]
        defence_obs = [np.array(agent.get_obs(), dtype=np.float32) for agent in self.swarm.defence_agents]
        rewards = [agent.reward for agent in self.swarm.defence_agents]
        done = [False for _ in self.swarm.defence_agents]
        info = [{"metrics": self.swarm.summary()} for _ in self.swarm.defence_agents]
        return [plane_obs, defence_obs, rewards, done, info]

    def render(self, mode="human"):
        return None
