"""Test / visualisation script for trained Drone Payload Stabilisation models.

Usage:
    # Headless evaluation only
    python test.py --model results/v1_sac_*/best_model.zip --algo sac

    # GUI visual demo
    python test.py --model results/v3_sac_*/best_model.zip --algo sac --gui

    # GUI with a single manual impulse applied at t=3s
    python test.py --model results/v3_sac_*/best_model.zip --algo sac --gui --impulse

    # Control impulse timing and strength
    python test.py --model results/v3_sac_*/best_model.zip --algo sac --gui \\
        --impulse --impulse_time 3.0 --impulse_mag 0.05
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


def _apply_impulse(env, magnitude: float, direction: np.ndarray | None = None):
    """Apply a single impulse force to the payload.

    Parameters
    ----------
    env       : PayloadHoverAviary instance (GUI mode)
    magnitude : Force magnitude in Newtons
    direction : Unit vector (optional). Defaults to a random horizontal kick.
    """
    if direction is None:
        # Random horizontal direction so each test looks a bit different
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    force = (magnitude * direction).tolist()

    p.applyExternalForce(
        env.PAYLOAD_ID,
        -1,
        forceObj=force,
        posObj=[0, 0, 0],
        flags=p.WORLD_FRAME,
        physicsClientId=env.CLIENT,
    )
    return direction, magnitude


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

    print(f"[INFO] Loading model: {args.model}")
    print(f"[INFO] Algorithm: {args.algo.upper()}, Version: {env_config['version']}")
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
        print(f"\n[INFO] Impulse mode ON — magnitude={args.impulse_mag}N "
              f"at t={args.impulse_time}s")

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
        total_reward = 0.0
        start = time.time()

        # Track swing recovery after impulse
        impulse_fired    = False
        impulse_step     = None
        pre_impulse_swing = None
        peak_swing       = 0.0
        settled          = False
        settle_step      = None
        SETTLE_THRESHOLD_DEG = 2.0   # consider settled when swing < 2°

        num_steps = int((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ)

        print(f"\n  Episode {ep+1}/{args.episodes}")
        if impulse_mode:
            print(f"  Impulse scheduled at t={args.impulse_time:.1f}s  "
                  f"(step {int(args.impulse_time * test_env.CTRL_FREQ)})")
        print(f"  {'─'*66}")
        print(f"  {'t':>6}  {'swing':>7}  {'drift':>7}  {'reward':>7}  note")
        print(f"  {'─'*66}")

        for step in range(num_steps):
            t = step / test_env.CTRL_FREQ

            # ── Fire impulse ──────────────────────────────────────────────
            if (impulse_mode
                    and not impulse_fired
                    and t >= args.impulse_time):
                pre_impulse_swing = info.get("swing_angle_deg", 0.0)
                direction = None  # random horizontal
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
                      f"(pre-impulse swing={pre_impulse_swing:.1f}°) ***\n")

            # ── Step the environment ──────────────────────────────────────
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward

            swing    = info.get("swing_angle_deg", 0.0)
            d_err    = info.get("drone_pos_error", 0.0)

            # Track peak swing after impulse
            if impulse_fired and not settled:
                if swing > peak_swing:
                    peak_swing = swing
                if swing < SETTLE_THRESHOLD_DEG and step > impulse_step + 5:
                    settled    = True
                    settle_step = step

            # ── Print every second ────────────────────────────────────────
            if step % test_env.CTRL_FREQ == 0:
                note = ""
                if impulse_fired and step == impulse_step:
                    note = "<-- IMPULSE"
                elif settled and step == settle_step:
                    note = f"<-- SETTLED ({(settle_step - impulse_step) / test_env.CTRL_FREQ:.2f}s after impulse)"

                print(f"  {t:6.1f}s  {swing:6.1f}°  {d_err:6.3f}m  "
                      f"{reward:+7.2f}  {note}")

            # Real-time sync for GUI
            sync(step, start, test_env.CTRL_TIMESTEP)

            if terminated or truncated:
                reason = "TERMINATED" if terminated else "TRUNCATED"
                print(f"\n  Episode ended early: {reason}")
                break

        # ── Episode summary ───────────────────────────────────────────────
        print(f"\n  {'─'*66}")
        print(f"  Episode {ep+1} Summary")
        print(f"    Total reward : {total_reward:.2f}")
        if impulse_fired:
            print(f"    Pre-impulse swing  : {pre_impulse_swing:.1f}°")
            print(f"    Peak swing after   : {peak_swing:.1f}°")
            if settled:
                recovery_sec = (settle_step - impulse_step) / test_env.CTRL_FREQ
                print(f"    Recovery time      : {recovery_sec:.2f}s  ✓")
            else:
                print(f"    Recovery time      : did not settle below "
                      f"{SETTLE_THRESHOLD_DEG}°  ✗")
        print(f"  {'='*66}\n")

    test_env.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test / visualise a trained drone payload model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Headless eval
  python test.py --model results/v3_sac_*/best_model.zip --algo sac

  # GUI demo (no impulse)
  python test.py --model results/v3_sac_*/best_model.zip --algo sac --gui --version 3

  # GUI with impulse at t=3s, magnitude 0.05N (default)
  python test.py --model results/v3_sac_*/best_model.zip --algo sac --gui --version 3 --impulse

  # Stronger impulse at t=4s in a fixed direction
  python test.py --model results/v3_sac_*/best_model.zip --algo sac --gui --version 3 \\
      --impulse --impulse_time 4.0 --impulse_mag 0.08 --impulse_dir "1,0,0"
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

    # ── Impulse args ─────────────────────────────────────────────────────
    parser.add_argument("--impulse", action="store_true",
                        help="Apply a single impulse to the payload during the episode")
    parser.add_argument("--impulse_time", type=float, default=3.0,
                        help="Time (seconds) at which to fire the impulse (default: 3.0)")
    parser.add_argument("--impulse_mag", type=float, default=0.05,
                        help="Impulse force magnitude in Newtons (default: 0.05)")
    parser.add_argument("--impulse_dir", type=str, default=None,
                        help="Force direction as 'x,y,z' (default: random horizontal)")

    args = parser.parse_args()
    test(args)
