"""Test / visualisation script for trained Drone Payload Stabilisation models.

Supports version-aware testing: each version highlights the metrics that
matter most for that curriculum stage.

Usage:
    # V1: baseline hover + swing damping
    python test.py --model results/v1_sac_*/best_model.zip --algo sac --gui --version 1

    # V3: impulse disturbance rejection
    python test.py --model results/v3_sac_*/best_model.zip --algo sac --gui --version 3 --impulse

    # V4: wind robustness (wind is automatic from version config)
    python test.py --model results/v4_sac_*/best_model.zip --algo sac --gui --version 4

    # V7: transport with moving target
    python test.py --model results/v7_sac_*/best_model.zip --algo sac --gui --version 7
"""

import os
import sys
import time
import argparse

import numpy as np
import pybullet as p
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.PayloadHoverAviary import PayloadHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import get_version_config, load_config


ALGO_MAP = {"sac": SAC, "ppo": PPO, "td3": TD3}

# Settle threshold: swing angle (deg) below which we consider the payload "settled"
SETTLE_THRESHOLD_DEG = 2.0


# ------------------------------------------------------------------ #
#  Impulse helper                                                      #
# ------------------------------------------------------------------ #

def _apply_impulse(env, magnitude, direction=None):
    """Apply a single impulse force to the payload."""
    if direction is None:
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    force = (magnitude * direction).tolist()

    p.applyExternalForce(
        env.PAYLOAD_ID, -1,
        forceObj=force,
        posObj=[0, 0, 0],
        flags=p.WORLD_FRAME,
        physicsClientId=env.CLIENT,
    )
    return direction, magnitude


# ------------------------------------------------------------------ #
#  Version-aware header/row formatting                                 #
# ------------------------------------------------------------------ #

def _get_header(version):
    """Return column header string appropriate for the version."""
    base = f"  {'t':>6}  {'swing':>7}  {'drift':>7}"
    if version >= 5:
        base += f"  {'target':>21}"
    if version >= 6:
        base += f"  {'jerk':>6}"
    if version >= 7:
        base += f"  {'trk_err':>7}"
    base += f"  {'reward':>7}  note"
    return base


def _format_row(t, info, reward, version, extra=None):
    """Format one status row appropriate for the version."""
    swing = info.get("swing_angle_deg", 0.0)
    drift = info.get("drone_pos_error", 0.0)
    row = f"  {t:6.1f}s  {swing:6.1f}°  {drift:6.3f}m"

    if version >= 5:
        target = info.get("target", [0, 0, 1])
        row += f"  ({target[0]:+.2f},{target[1]:+.2f},{target[2]:+.2f})"
    if version >= 6:
        jerk = extra.get("jerk", 0.0) if extra else 0.0
        row += f"  {jerk:5.3f}"
    if version >= 7:
        target = np.array(info.get("target", [0, 0, 1]))
        drone_pos = np.array(info.get("drone_pos", [0, 0, 1]))
        trk_err = np.linalg.norm(target - drone_pos)
        row += f"  {trk_err:6.3f}m"
    row += f"  {reward:+7.2f}"
    return row


# ------------------------------------------------------------------ #
#  Version description for display                                     #
# ------------------------------------------------------------------ #

VERSION_DESC = {
    1: "Baseline: hover + swing damping from 15° offset",
    2: "Wider ICs: 30° swing + spawn noise",
    3: "Disturbance rejection: external impulses on payload",
    4: "Wind robustness: continuous OU wind field",
    5: "Position hold: drift penalty active",
    6: "Efficiency: smooth, low-jerk motor control",
    7: "Transport: track slowly moving target",
}


# ------------------------------------------------------------------ #
#  Main test function                                                  #
# ------------------------------------------------------------------ #

def test(args):
    """Load a trained model and evaluate / visualise."""
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        return

    algo_cls = ALGO_MAP[args.algo.lower()]

    # ----- Config -----
    if args.config:
        env_config = load_config(args.config)
    else:
        env_config = get_version_config(args.version)

    version = env_config["version"]

    print(f"[INFO] Loading model: {args.model}")
    print(f"[INFO] Algorithm: {args.algo.upper()}")
    print(f"[INFO] Version {version}: {VERSION_DESC.get(version, '?')}")

    model = algo_cls.load(args.model)

    # ----- Headless evaluation -----
    print("\n[INFO] Evaluating (headless, 10 episodes)...")
    eval_env = PayloadHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False,
        config=env_config,
    )
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True,
    )
    print(f"[INFO] Mean reward: {mean_r:.2f} ± {std_r:.2f}")
    eval_env.close()

    if not args.gui:
        print("[INFO] Done (no GUI mode). Use --gui for visual demo.")
        return

    # ----- GUI demo -----
    impulse_mode = args.impulse
    if impulse_mode:
        print(f"\n[INFO] Impulse mode: {args.impulse_mag}N at t={args.impulse_time}s")

    if version >= 4 and env_config.get("wind_enabled"):
        print(f"[INFO] Wind active: mean={env_config['wind_mean_speed']} m/s, "
              f"turbulence σ={env_config['wind_turbulence']}")

    if version >= 7 and env_config.get("moving_target"):
        print(f"[INFO] Moving target: speed={env_config['target_speed']} m/s")

    print(f"\n[INFO] Running {args.episodes} visual episode(s) (GUI)...")

    test_env = PayloadHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=True,
        record=args.record,
        config=env_config,
    )

    for ep in range(args.episodes):
        obs, info = test_env.reset(seed=42 + ep)
        total_reward  = 0.0
        start         = time.time()
        prev_action   = None

        # ── Tracking variables ────────────────────────────────────────
        swing_history   = []
        drift_history   = []
        jerk_history    = []
        trk_err_history = []

        # Impulse recovery tracking
        impulse_fired     = False
        impulse_step      = None
        pre_impulse_swing = None
        peak_swing        = 0.0
        settled           = False
        settle_step       = None

        # Initial swing tracking (V1/V2)
        initial_swing = info.get("swing_angle_deg", 0.0)
        initial_settled = False
        initial_settle_step = None

        num_steps = int((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ)

        print(f"\n  Episode {ep+1}/{args.episodes}")
        print(f"  Initial swing: {initial_swing:.1f}°")
        print(f"  {'─'*72}")
        print(_get_header(version))
        print(f"  {'─'*72}")

        for step in range(num_steps):
            t = step / test_env.CTRL_FREQ

            # ── Fire manual impulse (V3+ test) ────────────────────────
            if (impulse_mode
                    and not impulse_fired
                    and t >= args.impulse_time):
                pre_impulse_swing = info.get("swing_angle_deg", 0.0)
                direction = None
                if args.impulse_dir:
                    try:
                        direction = np.array(
                            [float(v) for v in args.impulse_dir.split(",")]
                        )
                    except ValueError:
                        pass

                actual_dir, actual_mag = _apply_impulse(
                    test_env, args.impulse_mag, direction
                )
                impulse_fired = True
                impulse_step  = step
                print(f"\n  *** IMPULSE at t={t:.2f}s | "
                      f"dir=({actual_dir[0]:+.2f},{actual_dir[1]:+.2f},"
                      f"{actual_dir[2]:+.2f}) | "
                      f"F={actual_mag:.3f}N "
                      f"(pre-swing={pre_impulse_swing:.1f}°) ***\n")

            # ── Step ──────────────────────────────────────────────────
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward

            swing = info.get("swing_angle_deg", 0.0)
            drift = info.get("drone_pos_error", 0.0)
            swing_history.append(swing)
            drift_history.append(drift)

            # Action jerk (V6)
            extra = {}
            if prev_action is not None:
                jerk_val = float(np.linalg.norm(action - prev_action))
                jerk_history.append(jerk_val)
                extra["jerk"] = jerk_val
            prev_action = action.copy()

            # Target tracking error (V7)
            if version >= 7:
                target = np.array(info.get("target", [0, 0, 1]))
                drone_pos = np.array(info.get("drone_pos", [0, 0, 1]))
                trk_err_history.append(float(np.linalg.norm(target - drone_pos)))

            # Track initial settling (V1/V2)
            if not initial_settled and swing < SETTLE_THRESHOLD_DEG and step > 5:
                initial_settled = True
                initial_settle_step = step

            # Track impulse recovery
            if impulse_fired and not settled:
                if swing > peak_swing:
                    peak_swing = swing
                if swing < SETTLE_THRESHOLD_DEG and step > impulse_step + 5:
                    settled     = True
                    settle_step = step

            # ── Print at 1-second intervals ───────────────────────────
            if step % test_env.CTRL_FREQ == 0:
                note = ""
                if impulse_fired and impulse_step is not None and abs(step - impulse_step) < test_env.CTRL_FREQ:
                    note = "<-- IMPULSE"
                elif settled and settle_step is not None and abs(step - settle_step) < test_env.CTRL_FREQ:
                    dt = (settle_step - impulse_step) / test_env.CTRL_FREQ
                    note = f"<-- SETTLED ({dt:.2f}s)"
                elif initial_settled and initial_settle_step is not None and abs(step - initial_settle_step) < test_env.CTRL_FREQ:
                    dt = initial_settle_step / test_env.CTRL_FREQ
                    note = f"<-- SETTLED ({dt:.2f}s)"

                row = _format_row(t, info, reward, version, extra)
                if note:
                    row += f"  {note}"
                print(row)

            sync(step, start, test_env.CTRL_TIMESTEP)

            if terminated or truncated:
                reason = "TERMINATED" if terminated else "TRUNCATED (time limit)"
                print(f"\n  Episode ended: {reason}")
                break

        # ── Episode summary ───────────────────────────────────────────
        print(f"\n  {'─'*72}")
        print(f"  Episode {ep+1} Summary")
        print(f"    Total reward     : {total_reward:.2f}")
        print(f"    Mean swing       : {np.mean(swing_history):.2f}°")
        print(f"    Max swing        : {np.max(swing_history):.2f}°")
        print(f"    Final swing      : {swing_history[-1]:.2f}°")
        print(f"    Mean drift       : {np.mean(drift_history):.3f} m")

        if initial_settled:
            print(f"    Initial settle   : "
                  f"{initial_settle_step / test_env.CTRL_FREQ:.2f}s")
        else:
            print(f"    Initial settle   : did not settle below "
                  f"{SETTLE_THRESHOLD_DEG}°")

        if impulse_fired:
            print(f"    Pre-impulse swing: {pre_impulse_swing:.1f}°")
            print(f"    Peak after imp.  : {peak_swing:.1f}°")
            if settled:
                dt = (settle_step - impulse_step) / test_env.CTRL_FREQ
                print(f"    Recovery time    : {dt:.2f}s  ✓")
            else:
                print(f"    Recovery time    : "
                      f"did not settle below {SETTLE_THRESHOLD_DEG}°  ✗")

        if version >= 4 and env_config.get("wind_enabled"):
            print(f"    (Wind was active during this episode)")

        if version >= 6 and jerk_history:
            print(f"    Mean action jerk : {np.mean(jerk_history):.4f}")

        if version >= 7 and trk_err_history:
            print(f"    Mean tracking err: {np.mean(trk_err_history):.3f} m")
            print(f"    Final tracking   : {trk_err_history[-1]:.3f} m")

        print(f"  {'='*72}\n")

    test_env.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test / visualise a trained drone payload model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
─── Test commands by version ──────────────────────────────────────────

  V1 (hover + swing damp):
    python test.py --model results/v1_sac_*/best_model.zip --version 1 --gui

  V2 (wider initial conditions):
    python test.py --model results/v2_sac_*/best_model.zip --version 2 --gui

  V3 (disturbance rejection — with impulse):
    python test.py --model results/v3_sac_*/best_model.zip --version 3 --gui --impulse

  V3 (stronger impulse):
    python test.py --model results/v3_sac_*/best_model.zip --version 3 --gui \\
        --impulse --impulse_mag 0.08 --impulse_time 4.0

  V4 (wind — automatic from config):
    python test.py --model results/v4_sac_*/best_model.zip --version 4 --gui

  V5 (position hold):
    python test.py --model results/v5_sac_*/best_model.zip --version 5 --gui

  V6 (efficiency — watch for smooth control):
    python test.py --model results/v6_sac_*/best_model.zip --version 6 --gui

  V7 (transport — moving target):
    python test.py --model results/v7_sac_*/best_model.zip --version 7 --gui
        """,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model .zip file")
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac", "ppo", "td3"],
                        help="Algorithm used for training (default: sac)")
    parser.add_argument("--version", type=int, default=1,
                        help="Environment version 1-7 (default: 1)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run (default: 3)")
    parser.add_argument("--gui", action="store_true",
                        help="Show PyBullet GUI visualization")
    parser.add_argument("--record", action="store_true",
                        help="Record video of the simulation")

    # Impulse args (V3+ testing)
    parser.add_argument("--impulse", action="store_true",
                        help="Apply a single impulse to the payload")
    parser.add_argument("--impulse_time", type=float, default=3.0,
                        help="Time (s) to fire impulse (default: 3.0)")
    parser.add_argument("--impulse_mag", type=float, default=0.05,
                        help="Impulse force in Newtons (default: 0.05)")
    parser.add_argument("--impulse_dir", type=str, default=None,
                        help="Force direction as 'x,y,z' (default: random horizontal)")

    args = parser.parse_args()
    test(args)
