from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from envs.entities import TIME_STEP, DroneType, WeaponType, clamp, distance


class Radar:
    """Simplified radar model producing stable tracks once target observed consistently."""

    def __init__(self, radar_id: str, x: float, y: float, detection_range: float = 14000.0):
        self.radar_id = radar_id
        self.position = np.array([x, y], dtype=float)
        self.detection_range = detection_range
        self.track_confirm_steps = 3
        self.trackers: Dict[int, int] = {}
        self.last_seen: Dict[int, float] = {}
        self.jammed_timer = 0.0

    def apply_jam(self, duration: float):
        self.jammed_timer = max(self.jammed_timer, duration)

    def _visibility(self, drone) -> float:
        base = 0.85 if drone.type == DroneType.ATTACK else 0.7
        if drone.type == DroneType.RECON:
            base = 0.6 * drone.rcs
        if drone.type == DroneType.EW:
            base = 0.55
        if self.jammed_timer > 0:
            base *= 0.5
        return clamp(base, 0.1, 0.95)

    def scan(self, drones: Iterable["Agent"], current_time: float) -> Tuple[List["Agent"], List["Agent"]]:
        if self.jammed_timer > 0:
            self.jammed_timer = max(0.0, self.jammed_timer - TIME_STEP)

        stable: List["Agent"] = []
        detected: List["Agent"] = []

        for drone in drones:
            drone_id = drone.agent_id
            if drone.is_destroyed():
                self.trackers.pop(drone_id, None)
                self.last_seen.pop(drone_id, None)
                continue

            dist = distance(tuple(self.position), tuple(drone.position))
            if dist > self.detection_range:
                # target temporarily lost
                if drone_id in self.trackers:
                    self.trackers[drone_id] = max(0, self.trackers[drone_id] - 1)
                continue

            visibility = self._visibility(drone)
            if np.random.random() < visibility:
                detected.append(drone)
                self.trackers[drone_id] = self.trackers.get(drone_id, 0) + 1
                self.last_seen[drone_id] = current_time
                if self.trackers[drone_id] >= self.track_confirm_steps:
                    stable.append(drone)
            else:
                if drone_id in self.trackers:
                    self.trackers[drone_id] = max(0, self.trackers[drone_id] - 1)

            if drone.type == DroneType.EW and dist < 9000.0:
                self.apply_jam(5.0)

        # drop stale tracks
        stale = [drone_id for drone_id, last in self.last_seen.items() if current_time - last > 20.0]
        for drone_id in stale:
            self.trackers.pop(drone_id, None)
            self.last_seen.pop(drone_id, None)

        return stable, detected

