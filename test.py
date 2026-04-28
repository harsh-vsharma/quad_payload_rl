"""Test / visualisation script for trained Drone Payload Stabilisation models.

Supports version-aware testing: each version highlights the metrics that
matter most for that curriculum stage.

Usage:
    # V1: baseline hover + swing damping
    python test.py --model results/v1_sac_*/best_model.zip --algo sac --gui --version 1

    # V3: impulse disturbance rejection
    python test.py --model results/v3_sac_*/best_model.zip --algo sac --gui --version 3 --impulse

    # V6: payload height control
    python test.py --model results/v6_sac_*/best_model.zip --algo sac --gui --version 6

    # V8: transport with moving target
    python test.py --model results/v8_sac_*/best_model.zip --algo sac --gui --version 8
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
        forceObj=force, posObj=[0, 0, 0],
        flags=p.WORLD_FRAME, physicsClientId=env.CLIENT,
    )
    return direction, magnitude


# ------------------------------------------------------------------ #
#  Version descriptions                                                #
# ------------------------------------------------------------------ #

VERSION_DESC = {
    1: "Baseline: hover + swing damping from 15° offset",
    2: "Wider ICs: 30° swing + spawn noise",
    3: "Disturbance rejection: external impulses on payload",
    4: "Wind robustness: continuous OU wind field",
    5: "Position hold: drift penalty active",
    6: "Payload height: keep load at correct altitude",
    7: "Efficiency: smooth control with height locked in",
    8: "Go-to-target: fly to random fixed target with full stability",
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
        obs=ObservationType.KIN, act=ActionType.RPM,
        gui=False, config=env_config,
    )
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True,
    )
    print(f"[INFO] Mean reward: {mean_r:.2f} ± {std_r:.2f}")
    eval_env.close()

    if not args.gui:
        print("[INFO] Done (no GUI). Use --gui for visual demo.")
        return

    # ----- GUI demo -----
    impulse_mode = args.impulse

    if impulse_mode:
        print(f"\n[INFO] Impulse mode: {args.impulse_mag}N at t={args.impulse_time}s")
    if version >= 4 and env_config.get("wind_enabled"):
        print(f"[INFO] Wind active: mean={env_config['wind_mean_speed']} m/s")
    if version >= 8 and env_config.get("moving_target"):
        print(f"[INFO] Moving target: speed={env_config['target_speed']} m/s")

    print(f"\n[INFO] Running {args.episodes} visual episode(s)...\n")

    test_env = PayloadHoverAviary(
        obs=ObservationType.KIN, act=ActionType.RPM,
        gui=True, record=args.record, config=env_config,
    )

    for ep in range(args.episodes):
        obs, info = test_env.reset(seed=42 + ep)
        total_reward = 0.0
        start = time.time()
        prev_action = None

        # Tracking
        swing_hist = []
        drift_hist = []
        height_err_hist = []
        jerk_hist = []
        trk_err_hist = []

        # Impulse recovery
        impulse_fired = False
        impulse_step = None
        pre_impulse_swing = None
        peak_swing = 0.0
        settled = False
        settle_step = None

        # Initial settling
        initial_swing = info.get("swing_angle_deg", 0.0)
        initial_settled = False
        initial_settle_step = None

        num_steps = int((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ)

        # Header
        print(f"  Episode {ep+1}/{args.episodes}  |  initial swing: {initial_swing:.1f}°")
        hdr = f"  {'t':>5}  {'swing':>6}  {'drift':>6}  {'pld_z':>6}"
        if version >= 6:
            hdr += f"  {'h_err':>5}"
        if version >= 7:
            hdr += f"  {'jerk':>5}"
        if version >= 8:
            hdr += f"  {'trk':>6}"
        hdr += f"  {'rew':>6}  note"
        print(f"  {'─'*len(hdr)}")
        print(hdr)
        print(f"  {'─'*len(hdr)}")

        for step in range(num_steps):
            t = step / test_env.CTRL_FREQ

            # ── Impulse ───────────────────────────────────────────────
            if impulse_mode and not impulse_fired and t >= args.impulse_time:
                pre_impulse_swing = info.get("swing_angle_deg", 0.0)
                direction = None
                if args.impulse_dir:
                    try:
                        direction = np.array([float(v) for v in args.impulse_dir.split(",")])
                    except ValueError:
                        pass
                d, m = _apply_impulse(test_env, args.impulse_mag, direction)
                impulse_fired = True
                impulse_step = step
                print(f"\n  *** IMPULSE t={t:.2f}s  "
                      f"dir=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})  "
                      f"F={m:.3f}N  (pre-swing={pre_impulse_swing:.1f}°) ***\n")

            # ── Step ──────────────────────────────────────────────────
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward

            swing = info.get("swing_angle_deg", 0.0)
            drift = info.get("drone_pos_error", 0.0)
            pld_z = info.get("payload_z", 0.0)
            h_err = info.get("payload_height_err", 0.0)
            swing_hist.append(swing)
            drift_hist.append(drift)
            height_err_hist.append(h_err)

            # Jerk
            extra_jerk = 0.0
            if prev_action is not None:
                extra_jerk = float(np.linalg.norm(action - prev_action))
                jerk_hist.append(extra_jerk)
            prev_action = action.copy()

            # Target tracking (V8)
            if version >= 8:
                tgt = np.array(info.get("target", [0, 0, 1]))
                dpos = np.array(info.get("drone_pos", [0, 0, 1]))
                trk_err_hist.append(float(np.linalg.norm(tgt - dpos)))

            # Settling
            if not initial_settled and swing < SETTLE_THRESHOLD_DEG and step > 5:
                initial_settled = True
                initial_settle_step = step

            if impulse_fired and not settled:
                if swing > peak_swing:
                    peak_swing = swing
                if swing < SETTLE_THRESHOLD_DEG and step > impulse_step + 5:
                    settled = True
                    settle_step = step

            # ── Print every second ────────────────────────────────────
            if step % test_env.CTRL_FREQ == 0:
                note = ""
                if impulse_fired and impulse_step is not None and abs(step - impulse_step) < test_env.CTRL_FREQ:
                    note = "<-- IMPULSE"
                elif settled and settle_step is not None and abs(step - settle_step) < test_env.CTRL_FREQ:
                    note = f"<-- SETTLED ({(settle_step - impulse_step) / test_env.CTRL_FREQ:.2f}s)"
                elif initial_settled and initial_settle_step is not None and abs(step - initial_settle_step) < test_env.CTRL_FREQ:
                    note = f"<-- SETTLED ({initial_settle_step / test_env.CTRL_FREQ:.2f}s)"

                row = f"  {t:5.1f}s  {swing:5.1f}°  {drift:5.3f}  {pld_z:5.3f}"
                if version >= 6:
                    row += f"  {h_err:5.3f}"
                if version >= 7:
                    row += f"  {extra_jerk:5.3f}"
                if version >= 8 and trk_err_hist:
                    row += f"  {trk_err_hist[-1]:5.3f}"
                row += f"  {reward:+6.2f}"
                if note:
                    row += f"  {note}"
                print(row)

            sync(step, start, test_env.CTRL_TIMESTEP)

            if terminated or truncated:
                reason = "TERMINATED" if terminated else "TRUNCATED (time limit)"
                print(f"\n  Episode ended: {reason}")
                break

        # ── Summary ───────────────────────────────────────────────────
        print(f"\n  {'─'*60}")
        print(f"  Episode {ep+1} Summary")
        print(f"    Total reward      : {total_reward:.2f}")
        print(f"    Mean swing        : {np.mean(swing_hist):.2f}°")
        print(f"    Max swing         : {np.max(swing_hist):.2f}°")
        print(f"    Final swing       : {swing_hist[-1]:.2f}°")
        print(f"    Mean drift        : {np.mean(drift_hist):.3f} m")

        if initial_settled:
            print(f"    Initial settle    : {initial_settle_step / test_env.CTRL_FREQ:.2f}s")

        if impulse_fired:
            print(f"    Pre-impulse swing : {pre_impulse_swing:.1f}°")
            print(f"    Peak after impulse: {peak_swing:.1f}°")
            if settled:
                print(f"    Recovery time     : {(settle_step - impulse_step) / test_env.CTRL_FREQ:.2f}s  ✓")
            else:
                print(f"    Recovery          : did not settle  ✗")

        if version >= 6 and height_err_hist:
            print(f"    Mean payload h_err: {np.mean(height_err_hist):.4f} m")
            print(f"    Max  payload h_err: {np.max(height_err_hist):.4f} m")

        if version >= 7 and jerk_hist:
            print(f"    Mean action jerk  : {np.mean(jerk_hist):.4f}")

        if version >= 8 and trk_err_hist:
            print(f"    Mean tracking err : {np.mean(trk_err_hist):.3f} m")
            print(f"    Final tracking err: {trk_err_hist[-1]:.3f} m")

        if version >= 4 and env_config.get("wind_enabled"):
            print(f"    (Wind was active)")

        print(f"  {'='*60}\n")

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

  V4 (wind — automatic from config):
    python test.py --model results/v4_sac_*/best_model.zip --version 4 --gui

  V5 (position hold):
    python test.py --model results/v5_sac_*/best_model.zip --version 5 --gui

  V6 (payload height control):
    python test.py --model results/v6_sac_*/best_model.zip --version 6 --gui

  V7 (efficiency — watch for smooth control):
    python test.py --model results/v7_sac_*/best_model.zip --version 7 --gui

  V8 (transport — moving target):
    python test.py --model results/v8_sac_*/best_model.zip --version 8 --gui
        """,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model .zip file")
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac", "ppo", "td3"],
                        help="Algorithm (default: sac)")
    parser.add_argument("--version", type=int, default=1,
                        help="Environment version 1-8 (default: 1)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes (default: 3)")
    parser.add_argument("--gui", action="store_true",
                        help="Show PyBullet GUI")
    parser.add_argument("--record", action="store_true",
                        help="Record video")
    # Impulse
    parser.add_argument("--impulse", action="store_true",
                        help="Apply single impulse to payload")
    parser.add_argument("--impulse_time", type=float, default=3.0,
                        help="Impulse time in seconds (default: 3.0)")
    parser.add_argument("--impulse_mag", type=float, default=0.05,
                        help="Impulse force in N (default: 0.05)")
    parser.add_argument("--impulse_dir", type=str, default=None,
                        help="Force direction as 'x,y,z' (default: random horizontal)")

    args = parser.parse_args()
    test(args)
