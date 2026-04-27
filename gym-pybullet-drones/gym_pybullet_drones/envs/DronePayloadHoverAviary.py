"""Drone Payload Hover Aviary — RL environment for slung-load hover with random goals.

A Crazyflie quadrotor lifts a small payload attached via a rigid cable
(PyBullet point-to-point constraint) and must learn to navigate to a
randomly assigned goal position within a 3m radius sphere and hover stably.

Key design decisions:
- Goal is randomized each episode: uniform in a 3m sphere, z in [cable_length+0.3, 3.0]
- Observation includes drone→goal displacement so the agent knows where to go
- Termination bounds widened to 3.5m horizontal to match the task space
- Reward shaped around goal proximity; payload swing penalized
"""

import numpy as np
import pybullet as p
import pkg_resources
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class DronePayloadHoverAviary(BaseRLAviary):
    """Single-agent RL env: fly to random goal and hover with slung payload."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 48,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 payload_mass: float = 0.01,
                 cable_length: float = 0.3,
                 fixed_target=None,
                 goal_radius: float = 3.0,
                 ):
        """
        Parameters
        ----------
        payload_mass  : float  — payload mass in kg
        cable_length  : float  — rigid cable length in metres
        fixed_target  : array or None — if set, always use this goal (for eval/debug)
        goal_radius   : float  — max XY+Z radius for random goal sampling (default 3 m)
        """
        self.PAYLOAD_MASS  = payload_mass
        self.CABLE_LENGTH  = cable_length
        self.GOAL_RADIUS   = goal_radius
        self._fixed_target = np.array(fixed_target, dtype=np.float64) if fixed_target is not None else None

        # Spawn drone at z=1; goal will be set at the start of each episode
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 1.0]])

        # Temporary target so super().__init__ can complete; overwritten at reset
        self.TARGET_POS         = np.array([0.0, 0.0, 1.0])
        self.TARGET_PAYLOAD_POS = self.TARGET_POS - np.array([0.0, 0.0, cable_length])

        self.EPISODE_LEN_SEC = 15   # slightly longer to allow travel time

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

        # Recompute hover RPM for combined drone+payload weight
        combined_gravity      = (self.M + self.PAYLOAD_MASS) * self.G
        self.PAYLOAD_HOVER_RPM = float(np.sqrt(combined_gravity / (4 * self.KF)))
        self.RPM_RANGE         = 0.25   # action ±1 → ±25 % of PAYLOAD_HOVER_RPM

    # ------------------------------------------------------------------ #
    #  Goal sampling                                                       #
    # ------------------------------------------------------------------ #

    def _sampleGoal(self):
        """Sample a random drone goal inside the 3 m sphere, above the ground.

        Strategy: rejection-sample in the sphere, enforce z bounds.
        Drone z must be high enough that the payload (cable_length below)
        stays at least 0.15 m above the ground plane.
        """
        z_min = self.CABLE_LENGTH + 0.15   # payload clears ground
        z_max = 3.0
        r     = self.GOAL_RADIUS

        for _ in range(200):   # rejection sampling; almost always succeeds in 1–3 tries
            # Uniform in cube, then accept if inside sphere
            xyz = np.random.uniform([-r, -r, z_min], [r, r, z_max])
            if np.linalg.norm(xyz) <= r:
                return xyz

        # Fallback: just go straight up (should never happen with these bounds)
        return np.array([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------ #
    #  Reset / housekeeping                                                #
    # ------------------------------------------------------------------ #

    def _housekeeping(self):
        """Extended housekeeping: assign new goal, spawn payload, create cable."""
        # Assign goal BEFORE super() so reward/obs during first step is valid
        if self._fixed_target is not None:
            self.TARGET_POS = self._fixed_target.copy()
        else:
            self.TARGET_POS = self._sampleGoal()

        self.TARGET_PAYLOAD_POS = self.TARGET_POS - np.array([0.0, 0.0, self.CABLE_LENGTH])

        super()._housekeeping()
        self._spawnPayload()

        if self.GUI:
            self._drawGoalMarker()

    # ------------------------------------------------------------------ #
    #  Payload spawning                                                    #
    # ------------------------------------------------------------------ #

    def _spawnPayload(self):
        """Load payload URDF and attach via point-to-point constraint."""
        drone_pos       = self.INIT_XYZS[0]
        payload_start   = drone_pos - np.array([0.0, 0.0, self.CABLE_LENGTH])

        self.PAYLOAD_ID = p.loadURDF(
            pkg_resources.resource_filename('gym_pybullet_drones', 'assets/payload.urdf'),
            payload_start,
            p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.CLIENT,
        )

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
        p.changeConstraint(self.CABLE_CONSTRAINT, maxForce=500, physicsClientId=self.CLIENT)
        p.setCollisionFilterPair(
            self.DRONE_IDS[0], self.PAYLOAD_ID, -1, -1,
            enableCollision=0, physicsClientId=self.CLIENT,
        )

        if self.GUI:
            self._initCableLine()

    # ------------------------------------------------------------------ #
    #  GUI helpers                                                         #
    # ------------------------------------------------------------------ #

    def _drawGoalMarker(self):
        """Draw a small coloured sphere at the goal in GUI mode."""
        # Three orthogonal circles approximate a sphere marker
        N = 32
        angles = np.linspace(0, 2 * np.pi, N)
        cx, cy, cz = self.TARGET_POS
        r = 0.08
        for i in range(N - 1):
            a0, a1 = angles[i], angles[i + 1]
            # XY ring
            p.addUserDebugLine(
                [cx + r*np.cos(a0), cy + r*np.sin(a0), cz],
                [cx + r*np.cos(a1), cy + r*np.sin(a1), cz],
                lineColorRGB=[0, 1, 0], lineWidth=2, physicsClientId=self.CLIENT,
            )
            # XZ ring
            p.addUserDebugLine(
                [cx + r*np.cos(a0), cy, cz + r*np.sin(a0)],
                [cx + r*np.cos(a1), cy, cz + r*np.sin(a1)],
                lineColorRGB=[0, 1, 0], lineWidth=2, physicsClientId=self.CLIENT,
            )

    def _initCableLine(self):
        drone_pos   = self.pos[0]
        payload_pos, _ = p.getBasePositionAndOrientation(self.PAYLOAD_ID, physicsClientId=self.CLIENT)
        self.CABLE_LINE_ID = p.addUserDebugLine(
            drone_pos.tolist(), list(payload_pos),
            lineColorRGB=[0.15, 0.15, 0.15], lineWidth=2, physicsClientId=self.CLIENT,
        )

    def _updateCableLine(self):
        if not hasattr(self, 'CABLE_LINE_ID'):
            return
        drone_pos   = self.pos[0]
        payload_pos, _ = p.getBasePositionAndOrientation(self.PAYLOAD_ID, physicsClientId=self.CLIENT)
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
            target   = np.clip(action[k, :], -1.0, 1.0)
            rpm[k, :] = self.PAYLOAD_HOVER_RPM * (1.0 + self.RPM_RANGE * target)
            rpm[k, :] = np.clip(rpm[k, :], 0, self.MAX_RPM)
        return rpm

    # ------------------------------------------------------------------ #
    #  Observation                                                         #
    # ------------------------------------------------------------------ #

    def _getPayloadStateVector(self):
        pos, _ = p.getBasePositionAndOrientation(self.PAYLOAD_ID, physicsClientId=self.CLIENT)
        vel, _ = p.getBaseVelocity(self.PAYLOAD_ID, physicsClientId=self.CLIENT)
        return np.array(list(pos) + list(vel), dtype=np.float64)

    def _observationSpace(self):
        """21-dim observation per drone:
            [0:3]   drone pos
            [3:6]   drone rpy
            [6:9]   drone vel
            [9:12]  drone ang_vel
            [12:15] payload pos
            [15:18] payload vel
            [18:21] drone → goal displacement  ← NEW
        Shape: (1, 21)
        """
        lo = np.full((1, 21), -np.inf, dtype=np.float32)
        hi = np.full((1, 21),  np.inf, dtype=np.float32)
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _computeObs(self):
        s  = self._getDroneStateVector(0)
        pl = self._getPayloadStateVector()

        drone_pos   = s[0:3]
        drone_rpy   = s[7:10]
        drone_vel   = s[10:13]
        drone_ang_v = s[13:16]
        payload_pos = pl[0:3]
        payload_vel = pl[3:6]

        goal_disp = self.TARGET_POS - drone_pos   # vector pointing to goal

        obs = np.hstack([
            drone_pos,    # 0:3
            drone_rpy,    # 3:6
            drone_vel,    # 6:9
            drone_ang_v,  # 9:12
            payload_pos,  # 12:15
            payload_vel,  # 15:18
            goal_disp,    # 18:21  ← goal-conditioned signal
        ]).reshape(1, 21).astype(np.float32)

        return obs

    # ------------------------------------------------------------------ #
    #  Reward                                                              #
    # ------------------------------------------------------------------ #

    def _computeReward(self):
        """Shaped reward for goal-conditioned navigation + stable hover.

        Phases naturally emerge:
          1. Approach  — exponential drone proximity bonus dominates
          2. Settle    — velocity + swing penalties drive smoothness
          3. Hold      — alive bonus accumulates while all errors stay small
        """
        s  = self._getDroneStateVector(0)
        pl = self._getPayloadStateVector()

        drone_pos   = s[0:3]
        drone_rpy   = s[7:10]
        drone_vel   = s[10:13]
        payload_pos = pl[0:3]
        payload_vel = pl[3:6]

        drone_err   = np.linalg.norm(self.TARGET_POS - drone_pos)
        payload_err = np.linalg.norm(self.TARGET_PAYLOAD_POS - payload_pos)

        # Exponential proximity — smooth gradient from anywhere in the sphere
        drone_prox   = np.exp(-2.0 * drone_err)     # scale -2 gives ~0.14 at 1 m
        payload_prox = np.exp(-2.0 * payload_err)

        # Velocity penalty — discourage oscillation / overshoot
        vel_pen    = np.linalg.norm(drone_vel) + np.linalg.norm(payload_vel)

        # Tilt penalty
        tilt       = abs(drone_rpy[0]) + abs(drone_rpy[1])

        # Swing: horizontal separation drone ↔ payload (ideally 0)
        swing      = np.linalg.norm(drone_pos[0:2] - payload_pos[0:2])

        # Alive bonus — agent is rewarded for surviving
        alive      = 0.5

        reward = (
            + 6.0 * drone_prox
            + 3.0 * payload_prox
            - 0.2 * vel_pen
            - 1.5 * tilt
            - 2.0 * swing
            + alive
        )

        # Hard crash penalty
        if self._computeTerminated():
            reward -= 50.0

        return float(reward)

    # ------------------------------------------------------------------ #
    #  Termination / truncation                                           #
    # ------------------------------------------------------------------ #

    def _computeTerminated(self):
        s   = self._getDroneStateVector(0)
        pos = s[0:3]
        rpy = s[7:10]

        # Out-of-bounds horizontally (slightly beyond goal_radius for tolerance)
        if np.linalg.norm(pos[0:2]) > self.GOAL_RADIUS + 0.5:
            return True
        # Ground crash or ceiling
        if pos[2] < 0.02 or pos[2] > 4.0:
            return True
        # Flipped
        if abs(rpy[0]) > 1.4 or abs(rpy[1]) > 1.4:
            return True
        return False

    def _computeTruncated(self):
        return bool(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    def _computeInfo(self):
        s  = self._getDroneStateVector(0)
        pl = self._getPayloadStateVector()
        return {
            "drone_pos_error":   float(np.linalg.norm(self.TARGET_POS - s[0:3])),
            "payload_pos_error": float(np.linalg.norm(self.TARGET_PAYLOAD_POS - pl[0:3])),
            "goal":              self.TARGET_POS.tolist(),
            "drone_z":           float(s[2]),
            "payload_z":         float(pl[2]),
            "tilt_rad":          float(abs(s[7]) + abs(s[8])),
            "payload_hover_rpm": float(self.PAYLOAD_HOVER_RPM),
        }

    def _addObstacles(self):
        pass

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.GUI:
            self._updateCableLine()
        return obs, reward, terminated, truncated, info