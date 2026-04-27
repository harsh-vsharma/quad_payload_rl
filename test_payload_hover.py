"""Test/Demo script for Drone Payload Hover.
"
python test_payload_hover.py \
  --model results/payload_hover_04.26.2026_23.10.44/best_model.zip \
  --episodes 5 \
  --goal_radius 1.0
"
Usage:
    python test_payload_hover.py --model results/best_model.zip          # Run with GUI
    python test_payload_hover.py --model results/best_model.zip --record # Record video
    python test_payload_hover.py --model results/best_model.zip --episodes 5
"""

import os
import time
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.DronePayloadHoverAviary import DronePayloadHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync


def test(args):
    """Load a trained model and visualize the drone-payload hover."""

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        return

    print(f"[INFO] Loading model: {args.model}")
    model = PPO.load(args.model)

    # --- Evaluate without GUI first ---
    print("\n[INFO] Evaluating (headless)...")
    eval_env = DronePayloadHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False,
        payload_mass=args.payload_mass,
        cable_length=args.cable_length,
        goal_radius=args.goal_radius,
    )
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"[INFO] Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    eval_env.close()

    # --- Visual demo with GUI ---
    print(f"\n[INFO] Running {args.episodes} visual episodes (GUI)...")
    test_env = DronePayloadHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=True,
        record=args.record,
        payload_mass=args.payload_mass,
        cable_length=args.cable_length,
        goal_radius=args.goal_radius,
    )

    for ep in range(args.episodes):
        obs, info = test_env.reset(seed=42 + ep)
        total_reward = 0
        start = time.time()

        num_steps = (test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ
        for step in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward

            # Print status every second
            if step % test_env.CTRL_FREQ == 0:
                drone_z    = info.get("drone_z", 0)
                payload_z  = info.get("payload_z", 0)
                drone_err  = info.get("drone_pos_error", 0)
                payload_err = info.get("payload_pos_error", 0)
                goal       = info.get("goal", [0, 0, 0])
                print(
                    f"  t={step/test_env.CTRL_FREQ:5.1f}s | "
                    f"goal=[{goal[0]:+.2f},{goal[1]:+.2f},{goal[2]:+.2f}] | "
                    f"drone_z={drone_z:+.3f} | payload_z={payload_z:+.3f} | "
                    f"drone_err={drone_err:.4f} | payload_err={payload_err:.4f} | "
                    f"reward={reward:+.2f}"
                )

            # Real-time sync for GUI
            sync(step, start, test_env.CTRL_TIMESTEP)

            if terminated or truncated:
                reason = "TERMINATED" if terminated else "TRUNCATED"
                print(f"  Episode ended: {reason}")
                break

        print(f"\n  Episode {ep+1}/{args.episodes} — Total Reward: {total_reward:.2f}")
        print(f"  {'='*50}\n")

    test_env.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test drone payload hover model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model .zip file")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to visualize (default: 3)")
    parser.add_argument("--record", action="store_true",
                        help="Record video of the simulation")
    parser.add_argument("--payload_mass", type=float, default=0.01,
                        help="Payload mass in kg (default: 0.01)")
    parser.add_argument("--cable_length", type=float, default=0.3,
                        help="Cable length in m (default: 0.3)")
    parser.add_argument("--goal_radius", type=float, default=1.0,
                        help="Goal sampling radius in m (default: 1.0)")
    args = parser.parse_args()

    test(args)  