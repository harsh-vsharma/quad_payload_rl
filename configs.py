"""Configuration system for the PayloadHoverAviary environment.

Provides version presets (V1–V8) and YAML loading for experiment configs.
Each version builds on the previous one, adding progressive difficulty.

Curriculum ordering rationale:
  V1–V3: stabilisation, wider ICs, disturbance rejection
  V4:    wind + light positional reference (prevents passive drift)
  V5:    full position hold (drift penalty)
  V6:    payload height control (prevents "sink to save effort" in V7)
  V7:    energy efficiency (safe now that height is locked in)
  V8:    go-to-target navigation (capstone)
"""

import os
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional

# Try yaml; fall back to plain dict if not installed
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ------------------------------------------------------------------ #
#  Version presets                                                     #
# ------------------------------------------------------------------ #

# V1 is the base; other versions override specific fields.
V1_BASELINE = dict(
    version=1,
    payload_mass=0.01,
    cable_length=0.3,
    episode_len_sec=10.0,
    ctrl_freq=30,
    init_swing_range_deg=15.0,
    init_pos_noise=0.0,
    rpm_range=0.4,
    impulse_enabled=False,
    impulse_interval_sec=(2.0, 5.0),
    impulse_magnitude=(0.01, 0.05),
    wind_enabled=False,
    wind_mean_speed=0.0,
    wind_turbulence=0.0,
    alive_bonus=1.0,
    swing_weight=4.0,
    vel_weight=0.3,
    tilt_weight=2.0,
    drift_weight=0.0,
    payload_height_weight=0.0,
    energy_weight=0.0,
    target_weight=0.0,
    progress_weight=0.0,
    transit_relax=0.0,
    arrival_bonus=0.0,
    arrival_radius=0.15,
    random_target=False,
    target_xy_range=0.8,
    target_z_range=(0.8, 1.2),
    moving_target=False,
    target_speed=0.0,
    fixed_target=None,
)

V2_WIDER_IC = {**V1_BASELINE,
    "version": 2,
    "init_swing_range_deg": 30.0,
    "init_pos_noise": 0.1,
}

V3_DISTURBANCE = {**V2_WIDER_IC,
    "version": 3,
    "init_swing_range_deg": 15.0,
    "impulse_enabled": True,
    "impulse_interval_sec": (2.0, 5.0),
    "impulse_magnitude": (0.01, 0.05),
}

V4_WIND = {**V3_DISTURBANCE,
    "version": 4,

    # Environment
    "wind_enabled": True,
    "wind_mean_speed": 0.02,
    "wind_turbulence": 0.005,
    "impulse_enabled": False,

    # Core behaviour shaping
    "drift_weight": 0.3,          # <-- THIS is the key fix

    # Light anti-sink (optional but useful)
    "payload_height_weight": 0.3, # not full V6, just prevents collapse

    # Keep velocity penalty moderate (do NOT increase)
    "vel_weight": 0.3,

    # Keep swing dominant
    "swing_weight": 4.0,
}

V5_DISPLACEMENT = {**V4_WIND,
    "version": 5,
    "drift_weight": 1.0,
}

# V6: Payload height control — lock in altitude BEFORE adding efficiency.
# Without this, the energy penalty in V7 lets the drone sink to reduce effort.
V6_PAYLOAD_HEIGHT = {**V5_DISPLACEMENT,
    "version": 6,
    "payload_height_weight": 3.0,
}

# V7: Efficiency — safe now that payload height is rewarded.
V7_EFFICIENCY = {**V6_PAYLOAD_HEIGHT,
    "version": 7,
    "energy_weight": 0.5,
}

# ── V8: Go-to-target (staged sub-curriculum) ─────────────────────────
# V8 is a payload delivery task.  The drone must fly from origin to a
# random target and stabilise the payload there.  Trained in 3 stages:
#   V8a — near targets (0.3m)   learn to commit to the target
#   V8b — medium targets (0.6m) extend spatial range
#   V8c — full range (1.0m)     generalise to all reachable positions
#
# Reward: payload-centric proximity + progress shaping + arrival bonus
#         + phase-split transit relaxation.

# Shared V8 reward settings (all stages inherit these)
_V8_REWARD = dict(
    target_weight=3.0,            # proximity (exp(-4 * payload_dist))
    progress_weight=2.0,          # potential-based shaping (Ng 1999)
    transit_relax=0.5,            # relax vel/tilt penalties during transit
    arrival_bonus=5.0,            # one-time bonus for reaching target
    arrival_radius=0.15,          # payload 3D distance = "arrived"
    swing_weight=3.0,             # slightly lower to allow transit motion
    vel_weight=0.2,               # allow higher velocity during transit
    moving_target=False,
    random_target=True,
)

# Stage A — Near targets: first learn to commit and deliver
V8a_NEAR = {**V7_EFFICIENCY, **_V8_REWARD,
    "version": 8,
    "episode_len_sec": 12.0,
    "target_xy_range": 0.3,
    "target_z_range": (0.85, 1.15),
}

# Stage B — Medium range: extend spatial coverage
V8b_MED = {**V7_EFFICIENCY, **_V8_REWARD,
    "version": 8,
    "episode_len_sec": 15.0,
    "target_xy_range": 0.6,
    "target_z_range": (0.75, 1.25),
}

# Stage C — Full range: the complete go-to-target task
V8c_FULL = {**V7_EFFICIENCY, **_V8_REWARD,
    "version": 8,
    "episode_len_sec": 15.0,
    "target_xy_range": 1.0,
    "target_z_range": (0.7, 1.3),
}

V8_STAGES = {"a": V8a_NEAR, "b": V8b_MED, "c": V8c_FULL}

VERSION_PRESETS = {
    1: V1_BASELINE,
    2: V2_WIDER_IC,
    3: V3_DISTURBANCE,
    4: V4_WIND,
    5: V5_DISPLACEMENT,
    6: V6_PAYLOAD_HEIGHT,
    7: V7_EFFICIENCY,
    8: V8c_FULL,    # default V8 = full range
}


def get_version_config(version: int, stage: str = None, **overrides) -> dict:
    """Return a config dict for the given version, with optional overrides.

    For version 8, pass stage='a'/'b'/'c' to select the sub-curriculum stage.
    """
    if version == 8 and stage:
        if stage not in V8_STAGES:
            raise ValueError(f"Unknown V8 stage '{stage}'. Choose from {list(V8_STAGES.keys())}")
        cfg = copy.deepcopy(V8_STAGES[stage])
    elif version in VERSION_PRESETS:
        cfg = copy.deepcopy(VERSION_PRESETS[version])
    else:
        raise ValueError(f"Unknown version {version}. Choose from {list(VERSION_PRESETS.keys())}")
    cfg.update(overrides)
    return cfg


def load_config(path: str) -> dict:
    """Load a config from a YAML file, merging with the version preset."""
    if not HAS_YAML:
        raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}

    version = user_cfg.get("version", 1)
    base = copy.deepcopy(VERSION_PRESETS.get(version, V1_BASELINE))
    base.update(user_cfg)
    return base


def save_config(cfg: dict, path: str):
    """Save a config dict to YAML."""
    if not HAS_YAML:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
