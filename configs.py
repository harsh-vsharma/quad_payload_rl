"""Configuration system for the PayloadHoverAviary environment.

Provides version presets (V1–V7) and YAML loading for experiment configs.
Each version builds on the previous one, adding progressive difficulty.
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
    energy_weight=0.0,
    target_weight=0.0,
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
}

V5_DISPLACEMENT = {**V4_WIND,
    "version": 5,
    "drift_weight": 1.0,
}

V6_EFFICIENCY = {**V5_DISPLACEMENT,
    "version": 6,
    "energy_weight": 0.5,
}

V7_TRANSPORT = {**V6_EFFICIENCY,
    "version": 7,
    "episode_len_sec": 15.0,
    "moving_target": True,
    "target_speed": 0.1,
    "target_weight": 3.0,
    "swing_weight": 3.0,
    "vel_weight": 0.2,
}

VERSION_PRESETS = {
    1: V1_BASELINE,
    2: V2_WIDER_IC,
    3: V3_DISTURBANCE,
    4: V4_WIND,
    5: V5_DISPLACEMENT,
    6: V6_EFFICIENCY,
    7: V7_TRANSPORT,
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
