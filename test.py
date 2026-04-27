"""Test / visualisation script for trained Drone Payload Stabilisation models.

Usage:
    python test.py --model results/v1_sac_*/best_model.zip --algo sac
    python test.py --model results/v1_sac_*/best_model.zip --algo sac --gui
    python test.py --model results/v1_sac_*/best_model.zip --algo sac --episodes 5 --version 3
"""

import os
import sys
import time
import argparse

import numpy as np
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.PayloadHoverAviary import PayloadHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import get_version_config, load_config


ALGO_MAP = {"sac": SAC, "ppo": PPO, "td3": TD3}


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
    print(f"\n[INFO] Running {args.episodes} visual episodes (GUI)...")
    test_env = PayloadHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=True,
        record=args.record,
        config=env_config,
    )

    for ep in range(args.episodes):
        obs, info = test_env.reset(seed=42 + ep)
        total_reward = 0
        start = time.time()

        num_steps = int(
            (test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ
        )

        for step in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward

            # Print status every second
            if step % test_env.CTRL_FREQ == 0:
                swing = info.get("swing_angle_deg", 0)
                drone_z = info.get("drone_z", 0)
                payload_z = info.get("payload_z", 0)
                d_err = info.get("drone_pos_error", 0)
                print(
                    f"  t={step/test_env.CTRL_FREQ:5.1f}s | "
                    f"swing={swing:5.1f}° | "
                    f"drone_z={drone_z:+.3f} | "
                    f"payload_z={payload_z:+.3f} | "
                    f"drone_err={d_err:.4f} | "
                    f"reward={reward:+.2f}"
                )

            # Real-time sync for GUI
            sync(step, start, test_env.CTRL_TIMESTEP)

            if terminated or truncated:
                reason = "TERMINATED" if terminated else "TRUNCATED"
                print(f"  Episode ended: {reason}")
                break

        print(f"\n  Episode {ep+1}/{args.episodes} — Total: {total_reward:.2f}")
        print(f"  {'='*50}\n")

    test_env.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test / visualise a trained drone payload model",
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

    args = parser.parse_args()
    test(args)
