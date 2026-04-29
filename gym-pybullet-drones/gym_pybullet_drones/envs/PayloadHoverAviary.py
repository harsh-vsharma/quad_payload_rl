"""Payload Hover Aviary — RL environment for quadrotor slung-load stabilisation.

A Crazyflie quadrotor carries a small payload attached via a rigid cable
(PyBullet point-to-point constraint). The RL agent controls the drone's
4 motor RPMs to suppress payload swing and maintain stable hover.

Supports progressive curriculum learning (V1–V8) through a config dict:
  V1: Baseline stabilisation from moderate swing
  V2: Wider initial conditions
  V3: Disturbance rejection (external impulses)
  V4: Wind and environmental noise
  V5: Stability + displacement control
  V6: Payload height control
  V7: Stability + efficiency (with height locked in)
  V8: Stability during transport (moving target)
"""

import numpy as np
import pybullet as p
import pkg_resources
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType,
)


class PayloadHoverAviary(BaseRLAviary):
    """Single-agent RL env: stabilise a slung payload while hovering."""

    # ----- default version config (V1 baseline) ----- #
    DEFAULT_CONFIG = dict(
        version=1,

        # Physics / timing
        payload_mass=0.01,       # kg
        cable_length=0.3,        # m
        episode_len_sec=10.0,
        ctrl_freq=30,            # Hz  (matches upstream HoverAviary)

        # Initial conditions
        init_swing_range_deg=15.0,   # max initial cable angle from vertical
        init_pos_noise=0.0,         # m, uniform noise on drone XY spawn

        # RPM dynamic range
        rpm_range=0.4,              # action ±1 → ±40 % of PAYLOAD_HOVER_RPM

        # Disturbances (V3+)
        impulse_enabled=False,
        impulse_interval_sec=(2.0, 5.0),
        impulse_magnitude=(0.01, 0.05),   # N

        # Wind (V4+)
        wind_enabled=False,
        wind_mean_speed=0.0,        # m/s
        wind_turbulence=0.0,        # OU sigma

        # Reward weights
        alive_bonus=1.0,
        swing_weight=4.0,
        vel_weight=0.3,
        tilt_weight=2.0,
        drift_weight=0.0,              # V5+
        payload_height_weight=0.0,     # V6+: reward for keeping payload at target z
        energy_weight=0.0,             # V7+
        target_weight=0.0,             # V8+: proximity reward at target
        progress_weight=0.0,           # V8+: potential-based shaping (distance closed/step)
        transit_relax=0.0,             # V8+: relax vel/tilt penalties when far from target
        arrival_bonus=0.0,             # V8+: one-time bonus for entering arrival zone
        arrival_radius=0.15,           # V8+: payload 3D distance = "arrived"

        # Go-to-target navigation (V8)
        random_target=False,           # sample new random target each episode
        target_xy_range=0.8,           # max XY distance from origin
        target_z_range=(0.8, 1.2),     # drone target height range
        moving_target=False,           # legacy: slowly drifting target
        target_speed=0.0,

        # Target override (for eval/demo; None → hover at spawn)
        fixed_target=None,
    )

    # ------------------------------------------------------------------ #
    #  Init                                                                #
    # ------------------------------------------------------------------ #

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 config: dict | None = None,
                 **config_overrides,
                 ):
        """
        Parameters
        ----------
        config : dict or None
            Full version config.  If None, uses DEFAULT_CONFIG.
        **config_overrides
            Individual keys to override in the config dict (convenience).
        """
        # Merge config
        self.cfg = dict(self.DEFAULT_CONFIG)
        if config is not None:
            self.cfg.update(config)
        self.cfg.update(config_overrides)

        # Unpack frequently-used fields
        self.PAYLOAD_MASS  = self.cfg["payload_mass"]
        self.CABLE_LENGTH  = self.cfg["cable_length"]
        self.RPM_RANGE     = self.cfg["rpm_range"]
        self.EPISODE_LEN_SEC = self.cfg["episode_len_sec"]

        ctrl_freq = self.cfg["ctrl_freq"]

        # Spawn drone at z = 1 m (payload will hang at z = 1 - cable_length)
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 1.0]])

        # Target pos (overwritten every reset unless fixed)
        self.TARGET_POS = np.array([0.0, 0.0, 1.0])

        # Disturbance bookkeeping
        self._next_impulse_time = np.inf
        self._wind_vec = np.zeros(3)

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        # Hover RPM accounting for payload weight
        combined_gravity = (self.M + self.PAYLOAD_MASS) * self.G
        self.PAYLOAD_HOVER_RPM = float(np.sqrt(combined_gravity / (4 * self.KF)))

    # ------------------------------------------------------------------ #
    #  Housekeeping / reset                                                #
    # ------------------------------------------------------------------ #

    def _housekeeping(self):
        """Set target, spawn payload, init disturbances."""
        # ----- target position -----
        fixed = self.cfg["fixed_target"]
        if fixed is not None:
            # Explicit target (used by demo_goto.py / eval)
            self.TARGET_POS = np.array(fixed, dtype=np.float64)
        elif self.cfg["random_target"]:
            # V8: sample a random fixed target each episode.
            # XY: uniform in a ring [0.1, target_xy_range] at any angle.
            # Z:  uniform within target_z_range.
            angle = np.random.uniform(0, 2 * np.pi)
            xy_max = self.cfg["target_xy_range"]
            dist = np.random.uniform(0.1, xy_max)
            z_lo, z_hi = self.cfg["target_z_range"]
            z = np.random.uniform(z_lo, z_hi)
            self.TARGET_POS = np.array([
                dist * np.cos(angle),
                dist * np.sin(angle),
                z,
            ])
        elif self.cfg["moving_target"]:
            # Legacy: slowly drifting target
            self.TARGET_POS = np.array([0.0, 0.0, 1.0])
            self._target_vel = self._sampleTargetVelocity()
        else:
            self.TARGET_POS = np.array([0.0, 0.0, 1.0])  # hover at spawn

        self.TARGET_PAYLOAD_POS = self.TARGET_POS - np.array([0.0, 0.0, self.CABLE_LENGTH])

        # Progress reward bookkeeping — payload-centric distance
        init_payload_pos = np.array([0.0, 0.0, 1.0 - self.CABLE_LENGTH])
        self._prev_target_dist = float(np.linalg.norm(
            self.TARGET_PAYLOAD_POS - init_payload_pos
        ))
        self._arrival_triggered = False

        # Parent housekeeping (loads drone, resets counters)
        super()._housekeeping()

        # Spawn payload with optional initial swing
        self._spawnPayload()

        # Init disturbances
        self._initDisturbances()

        # GUI markers
        if self.GUI:
            self._drawGoalMarker()

    # ------------------------------------------------------------------ #
    #  Payload spawning                                                    #
    # ------------------------------------------------------------------ #

    def _spawnPayload(self):
        """Load payload URDF and attach via point-to-point constraint.

        Optionally apply an initial angular offset to simulate a
        pre-existing swing (controlled by init_swing_range_deg).
        """
        drone_pos = self.INIT_XYZS[0].copy()

        # Add spawn noise to drone position
        noise = self.cfg["init_pos_noise"]
        if noise > 0:
            drone_pos[0] += np.random.uniform(-noise, noise)
            drone_pos[1] += np.random.uniform(-noise, noise)
            p.resetBasePositionAndOrientation(
                self.DRONE_IDS[0], drone_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT,
            )

        # Swing angle: sample magnitude in [swing_min, swing_max] to guarantee
        # nonzero offset every reset (no lucky 0° episodes).
        swing_min_rad = np.deg2rad(max(1.0, self.cfg["init_swing_range_deg"] * 0.1))
        swing_max_rad = np.deg2rad(self.cfg["init_swing_range_deg"])
        theta = np.random.uniform(swing_min_rad, swing_max_rad)
        phi   = np.random.uniform(0, 2 * np.pi)

        # Payload offset from drone (cable vector in drone frame)
        cable_dx = self.CABLE_LENGTH * np.sin(theta) * np.cos(phi)
        cable_dy = self.CABLE_LENGTH * np.sin(theta) * np.sin(phi)
        cable_dz = -self.CABLE_LENGTH * np.cos(theta)
        payload_start = drone_pos + np.array([cable_dx, cable_dy, cable_dz])

        self.PAYLOAD_ID = p.loadURDF(
            pkg_resources.resource_filename(
                "gym_pybullet_drones", "assets/payload.urdf"
            ),
            payload_start,
            p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.CLIENT,
        )

        # Rigid cable constraint (point-to-point)
        self.CABLE_CONSTRAINT = p.createConstraint(
            parentBodyUniqueId=self.DRONE_IDS[0],
            parentLinkIndex=-1,
            childBodyUniqueId=self.PAYLOAD_ID,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.0, 0.0, 0.0],
            childFramePosition=[0.0, 0.0, self.CABLE_LENGTH],
            physicsClientId=self.CLIENT,
        )
        p.changeConstraint(
            self.CABLE_CONSTRAINT, maxForce=500,
            physicsClientId=self.CLIENT,
        )

        # Disable drone-payload collision
        p.setCollisionFilterPair(
            self.DRONE_IDS[0], self.PAYLOAD_ID, -1, -1,
            enableCollision=0, physicsClientId=self.CLIENT,
        )

        if self.GUI:
            self._initCableLine()

    # ------------------------------------------------------------------ #
    #  Disturbance systems                                                 #
    # ------------------------------------------------------------------ #

    def _initDisturbances(self):
        """Reset wind vector and schedule first impulse."""
        self._wind_vec = np.zeros(3)
        if self.cfg["impulse_enabled"]:
            lo, hi = self.cfg["impulse_interval_sec"]
            self._next_impulse_time = np.random.uniform(lo, hi)
        else:
            self._next_impulse_time = np.inf

    def _applyDisturbances(self):
        """Called every control step. Apply impulses and wind as configured."""
        sim_time = self.step_counter / self.PYB_FREQ

        # ----- External impulses (V3+) -----
        if self.cfg["impulse_enabled"] and sim_time >= self._next_impulse_time:
            lo_mag, hi_mag = self.cfg["impulse_magnitude"]
            mag = np.random.uniform(lo_mag, hi_mag)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction) + 1e-8
            force = mag * direction

            p.applyExternalForce(
                self.PAYLOAD_ID, -1,
                forceObj=force.tolist(),
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )

            # Schedule next impulse
            lo_t, hi_t = self.cfg["impulse_interval_sec"]
            self._next_impulse_time = sim_time + np.random.uniform(lo_t, hi_t)

        # ----- Wind (V4+) — Ornstein-Uhlenbeck process -----
        if self.cfg["wind_enabled"]:
            dt = self.CTRL_TIMESTEP
            mean_speed = self.cfg["wind_mean_speed"]
            sigma = self.cfg["wind_turbulence"]
            theta_ou = 0.5  # reversion rate

            # Update wind vector (OU process)
            noise = sigma * np.sqrt(dt) * np.random.randn(3)
            self._wind_vec += theta_ou * (
                np.array([mean_speed, 0.0, 0.0]) - self._wind_vec
            ) * dt + noise

            # Apply to drone
            p.applyExternalForce(
                self.DRONE_IDS[0], -1,
                forceObj=self._wind_vec.tolist(),
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )
            # Apply to payload (scaled by cross-section ratio)
            p.applyExternalForce(
                self.PAYLOAD_ID, -1,
                forceObj=(0.3 * self._wind_vec).tolist(),
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )

    def _sampleTargetVelocity(self):
        """V7: sample a slow random horizontal velocity for the target."""
        speed = self.cfg["target_speed"]
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])

    # ------------------------------------------------------------------ #
    #  GUI helpers                                                         #
    # ------------------------------------------------------------------ #

    def _drawGoalMarker(self):
        """Draw a small green sphere at the target in GUI mode."""
        N = 32
        angles = np.linspace(0, 2 * np.pi, N)
        cx, cy, cz = self.TARGET_POS
        r = 0.06
        for i in range(N - 1):
            a0, a1 = angles[i], angles[i + 1]
            # XY ring
            p.addUserDebugLine(
                [cx + r * np.cos(a0), cy + r * np.sin(a0), cz],
                [cx + r * np.cos(a1), cy + r * np.sin(a1), cz],
                lineColorRGB=[0, 1, 0], lineWidth=2,
                physicsClientId=self.CLIENT,
            )
            # XZ ring
            p.addUserDebugLine(
                [cx + r * np.cos(a0), cy, cz + r * np.sin(a0)],
                [cx + r * np.cos(a1), cy, cz + r * np.sin(a1)],
                lineColorRGB=[0, 1, 0], lineWidth=2,
                physicsClientId=self.CLIENT,
            )

    def _initCableLine(self):
        drone_pos = self.pos[0]
        payload_pos, _ = p.getBasePositionAndOrientation(
            self.PAYLOAD_ID, physicsClientId=self.CLIENT,
        )
        self.CABLE_LINE_ID = p.addUserDebugLine(
            drone_pos.tolist(), list(payload_pos),
            lineColorRGB=[0.15, 0.15, 0.15], lineWidth=2,
            physicsClientId=self.CLIENT,
        )

    def _updateCableLine(self):
        if not hasattr(self, "CABLE_LINE_ID"):
            return
        drone_pos = self.pos[0]
        payload_pos, _ = p.getBasePositionAndOrientation(
            self.PAYLOAD_ID, physicsClientId=self.CLIENT,
        )
        self.CABLE_LINE_ID = p.addUserDebugLine(
            drone_pos.tolist(), list(payload_pos),
            lineColorRGB=[0.15, 0.15, 0.15], lineWidth=2,
            replaceItemUniqueId=int(self.CABLE_LINE_ID),
            physicsClientId=self.CLIENT,
        )

    # ------------------------------------------------------------------ #
    #  Action pre-processing                                               #
    # ------------------------------------------------------------------ #

    def _preprocessAction(self, action):
        """Normalised [-1,1]^4 → RPMs centred on PAYLOAD_HOVER_RPM."""
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            target = np.clip(action[k, :], -1.0, 1.0)
            rpm[k, :] = self.PAYLOAD_HOVER_RPM * (
                1.0 + self.RPM_RANGE * target
            )
            rpm[k, :] = np.clip(rpm[k, :], 0, self.MAX_RPM)
        return rpm

    # ------------------------------------------------------------------ #
    #  Observation                                                         #
    # ------------------------------------------------------------------ #

    OBS_DIM = 22  # fixed observation dimensionality

    def _getPayloadState(self):
        """Return (pos, vel) of the payload."""
        pos, _ = p.getBasePositionAndOrientation(
            self.PAYLOAD_ID, physicsClientId=self.CLIENT,
        )
        vel, _ = p.getBaseVelocity(
            self.PAYLOAD_ID, physicsClientId=self.CLIENT,
        )
        return np.array(pos), np.array(vel)

    def _observationSpace(self):
        """22-dim observation per drone.

        [0:3]   drone pos
        [3:6]   drone rpy
        [6:9]   drone vel
        [9:12]  drone ang_vel
        [12:15] cable_vec   (drone → payload, relative)
        [15:18] payload_vel
        [18:21] payload_target_disp (payload → payload_target, payload-centric)
        [21]    cable_angle (angle from vertical, rad)
        """
        lo = np.full((1, self.OBS_DIM), -np.inf, dtype=np.float32)
        hi = np.full((1, self.OBS_DIM),  np.inf, dtype=np.float32)
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _computeObs(self):
        s = self._getDroneStateVector(0)
        payload_pos, payload_vel = self._getPayloadState()

        drone_pos   = s[0:3]
        drone_rpy   = s[7:10]
        drone_vel   = s[10:13]
        drone_ang_v = s[13:16]

        # Cable vector: drone → payload (captures pendulum state)
        cable_vec = payload_pos - drone_pos

        # Cable angle from vertical (0 = hanging straight down)
        cable_horiz = np.linalg.norm(cable_vec[0:2])
        cable_vert  = -cable_vec[2]  # positive when payload is below
        cable_angle = np.arctan2(cable_horiz, cable_vert)

        # Payload-centric target displacement:
        # Directly encodes what the policy must reduce — the vector
        # from the payload to its delivery target.  For V1-V7 where
        # TARGET_PAYLOAD_POS ≈ payload_pos (hover), this is ~(0,0,0).
        payload_target_disp = self.TARGET_PAYLOAD_POS - payload_pos

        obs = np.concatenate([
            drone_pos,            # 0:3
            drone_rpy,            # 3:6
            drone_vel,            # 6:9
            drone_ang_v,          # 9:12
            cable_vec,            # 12:15
            payload_vel,          # 15:18
            payload_target_disp,  # 18:21  (payload-centric)
            [cable_angle],        # 21
        ]).reshape(1, self.OBS_DIM).astype(np.float32)

        return obs

    # ------------------------------------------------------------------ #
    #  Reward                                                              #
    # ------------------------------------------------------------------ #

    def _computeReward(self):
        """Version-aware shaped reward.

        Core idea: reward = alive_bonus
                            + stabilisation terms (always)
                            + optional drift / energy / target terms
        """
        s = self._getDroneStateVector(0)
        payload_pos, payload_vel = self._getPayloadState()

        drone_pos   = s[0:3]
        drone_rpy   = s[7:10]
        drone_vel   = s[10:13]

        cfg = self.cfg

        # ----- Cable swing angle -----
        cable_vec = payload_pos - drone_pos
        cable_horiz = np.linalg.norm(cable_vec[0:2])
        cable_vert  = -cable_vec[2]
        swing_angle = np.arctan2(cable_horiz, max(cable_vert, 1e-6))

        # ----- Stabilisation signal (exponential — smooth gradient) -----
        swing_reward = np.exp(-6.0 * swing_angle)  # ≈1 at 0°, ≈0.35 at 10°, ≈0.05 at 30°

        # ----- Velocity penalty -----
        vel_pen = np.linalg.norm(drone_vel) + np.linalg.norm(payload_vel)

        # ----- Tilt penalty -----
        tilt = abs(drone_rpy[0]) + abs(drone_rpy[1])

        # ----- Payload-to-target distance (the primary V8 metric) -----
        pld_target_dist = float(np.linalg.norm(
            self.TARGET_PAYLOAD_POS - payload_pos
        ))

        # ----- Phase detection (smooth transit → hold blend) -----
        # transit_blend: 1.0 = far from target (transit), 0.0 = at target (hold)
        # Smooth linear fade over a 0.1m band above the arrival radius.
        arrival_r = cfg["arrival_radius"]
        transit_blend = float(np.clip(
            (pld_target_dist - arrival_r) / 0.1, 0.0, 1.0
        ))

        # Transit-phase relaxation: vel/tilt penalties fade when far from target
        relax = cfg["transit_relax"] * transit_blend
        eff_vel_w  = cfg["vel_weight"]  * (1.0 - relax)
        eff_tilt_w = cfg["tilt_weight"] * (1.0 - relax)

        # ----- Assemble core reward -----
        reward = (
            + cfg["alive_bonus"]
            + cfg["swing_weight"] * swing_reward
            - eff_vel_w  * vel_pen
            - eff_tilt_w * tilt
        )

        # ----- V5+: drift penalty -----
        if cfg["drift_weight"] > 0:
            drift = np.linalg.norm(drone_pos[0:2] - self.TARGET_POS[0:2])
            reward -= cfg["drift_weight"] * drift

        # ----- V6+: payload height reward -----
        if cfg["payload_height_weight"] > 0:
            payload_height_err = abs(payload_pos[2] - self.TARGET_PAYLOAD_POS[2])
            reward += cfg["payload_height_weight"] * np.exp(-5.0 * payload_height_err)

        # ----- V7+: energy penalty (relaxed during transit) -----
        if cfg["energy_weight"] > 0 and len(self.action_buffer) >= 2:
            last_action = self.action_buffer[-1]
            prev_action = self.action_buffer[-2]
            action_diff = np.linalg.norm(last_action - prev_action)
            # Halve energy penalty during transit so agent can accelerate
            eff_energy_w = cfg["energy_weight"] * (1.0 - 0.5 * transit_blend)
            reward -= eff_energy_w * action_diff

        # ----- V8: target proximity (payload-centric, sharp kernel) -----
        if cfg["target_weight"] > 0:
            reward += cfg["target_weight"] * np.exp(-4.0 * pld_target_dist)

        # ----- V8: progress reward (fades near target to prevent jitter) -----
        if cfg["progress_weight"] > 0:
            progress = self._prev_target_dist - pld_target_dist
            reward += cfg["progress_weight"] * progress * transit_blend
            self._prev_target_dist = pld_target_dist

        # ----- V8: arrival bonus (one-time, discrete commitment signal) -----
        if cfg["arrival_bonus"] > 0:
            if not self._arrival_triggered and pld_target_dist < arrival_r:
                reward += cfg["arrival_bonus"]
                self._arrival_triggered = True

        return float(reward)

    # ------------------------------------------------------------------ #
    #  Termination / truncation                                            #
    # ------------------------------------------------------------------ #

    def _computeTerminated(self):
        s   = self._getDroneStateVector(0)
        pos = s[0:3]
        rpy = s[7:10]

        # Ground crash
        if pos[2] < 0.05:
            return True
        # Payload ground contact — prevents the "sink to save energy" exploit
        payload_pos, _ = self._getPayloadState()
        if payload_pos[2] < 0.02:
            return True
        # Ceiling
        if pos[2] > 3.0:
            return True
        # Flipped
        if abs(rpy[0]) > 1.2 or abs(rpy[1]) > 1.2:
            return True
        # Out of bounds (generous for hover)
        if abs(pos[0]) > 2.5 or abs(pos[1]) > 2.5:
            return True
        return False

    def _computeTruncated(self):
        return bool(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    def _computeInfo(self):
        s = self._getDroneStateVector(0)
        payload_pos, payload_vel = self._getPayloadState()

        cable_vec = payload_pos - s[0:3]
        cable_horiz = np.linalg.norm(cable_vec[0:2])
        cable_vert  = -cable_vec[2]
        swing_angle = np.arctan2(cable_horiz, max(cable_vert, 1e-6))

        return {
            "drone_pos":         s[0:3].tolist(),
            "drone_z":           float(s[2]),
            "payload_z":         float(payload_pos[2]),
            "payload_target_z":  float(self.TARGET_PAYLOAD_POS[2]),
            "payload_height_err":float(abs(payload_pos[2] - self.TARGET_PAYLOAD_POS[2])),
            "swing_angle_deg":   float(np.rad2deg(swing_angle)),
            "drone_pos_error":   float(np.linalg.norm(self.TARGET_POS - s[0:3])),
            "payload_pos_error": float(np.linalg.norm(
                self.TARGET_PAYLOAD_POS - payload_pos
            )),
            "drone_vel_norm":    float(np.linalg.norm(s[10:13])),
            "payload_vel_norm":  float(np.linalg.norm(payload_vel)),
            "tilt_rad":          float(abs(s[7]) + abs(s[8])),
            "target":            self.TARGET_POS.tolist(),
            "payload_hover_rpm": float(self.PAYLOAD_HOVER_RPM),
        }

    # ------------------------------------------------------------------ #
    #  Step override — disturbances + moving target                        #
    # ------------------------------------------------------------------ #

    def step(self, action):
        # Apply disturbances before physics
        self._applyDisturbances()

        # V7: move target
        if self.cfg["moving_target"] and hasattr(self, "_target_vel"):
            self.TARGET_POS += self._target_vel * self.CTRL_TIMESTEP
            self.TARGET_PAYLOAD_POS = (
                self.TARGET_POS - np.array([0.0, 0.0, self.CABLE_LENGTH])
            )

        obs, reward, terminated, truncated, info = super().step(action)

        if self.GUI:
            self._updateCableLine()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Obstacles (none)                                                    #
    # ------------------------------------------------------------------ #

    def _addObstacles(self):
        pass
