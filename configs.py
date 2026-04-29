"""Configuration system for the PayloadHoverAviary environment.

Provides version presets (V1–V8) and YAML loading for experiment configs.
Each version builds on the previous one, adding progressive difficulty.

Curriculum ordering rationale:
  V1–V5: stabilisation, disturbances, position control
  V6:    payload height control (prevents "sink to save effort" in V7)
  V7:    energy efficiency (safe now that height is locked in)
  V8:    transport (moving target — capstone)
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
    "wind_enabled": True,
    "wind_mean_speed": 0.02,
    "wind_turbulence": 0.005,
    "impulse_enabled": False,
    "payload_height_weight": 0.0,
    "drift_weight": 0.2,
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

# V8: Go-to-target — capstone.  Random fixed target, all prior objectives.
# Each episode samples a different target at a random XY position + height.
# The drone must fly from origin to target and stabilise the payload there.
# Progress reward + transit relaxation give dense navigation gradient.
V8_GOTO = {**V7_EFFICIENCY,
    "version": 8,
    "episode_len_sec": 15.0,

    # Random fixed target per episode
    "random_target": True,
    "target_xy_range": 1.0,          # max XY distance from origin (m)
    "target_z_range": (0.7, 1.3),    # drone target height range (m)
    "moving_target": False,

    # Navigation reward
    "target_weight": 3.0,            # proximity reward (exp(-4 * dist))
    "progress_weight": 2.0,          # potential-based shaping (Ng 1999)
    "transit_relax": 0.5,            # relax vel/tilt penalties during transit
    "swing_weight": 3.0,             # slightly lower to allow transit motion
    "vel_weight": 0.2,               # allow higher velocity during transit
}

VERSION_PRESETS = {
    1: V1_BASELINE,
    2: V2_WIDER_IC,
    3: V3_DISTURBANCE,
    4: V4_WIND,
    5: V5_DISPLACEMENT,
    6: V6_PAYLOAD_HEIGHT,
    7: V7_EFFICIENCY,
    8: V8_GOTO,
}


def get_version_config(version: int, **overrides) -> dict:
    """Return a config dict for the given version, with optional overrides."""
    if version not in VERSION_PRESETS:
        raise ValueError(f"Unknown version {version}. Choose from {list(VERSION_PRESETS.keys())}")
    cfg = copy.deepcopy(VERSION_PRESETS[version])
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
