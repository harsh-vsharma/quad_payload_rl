"""Training script for Drone Payload Hover using PPO.

Usage:
    python train_payload_hover.py                     # Train with defaults
    python train_payload_hover.py --timesteps 1000000 # Train for 1M steps
    python train_payload_hover.py --resume results/best_model.zip  # Resume training
"""

import os
import time
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.DronePayloadHoverAviary import DronePayloadHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def make_env(**kwargs):
    """Factory function for creating the training environment."""
    return DronePayloadHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False,
        **kwargs,
    )


def train(args):
    """Train a PPO agent to hover with a slung payload."""

    # --- Output directory ---
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    output_dir = os.path.join(args.output_folder, f"payload_hover_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving to: {output_dir}")

    # --- Create environments ---
    train_env = make_vec_env(
        DronePayloadHoverAviary,
        env_kwargs=dict(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            gui=False,
            payload_mass=args.payload_mass,
            cable_length=args.cable_length,
            goal_radius=args.goal_radius,   # ← add
        ),
        n_envs=args.n_envs,                 # ← add
        seed=0,
    )

    eval_env = DronePayloadHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False,
        payload_mass=args.payload_mass,
        cable_length=args.cable_length,
    )

    print(f"[INFO] Action space: {train_env.action_space}")
    print(f"[INFO] Observation space: {train_env.observation_space}")
    print(f"[INFO] Payload mass: {args.payload_mass} kg")
    print(f"[INFO] Cable length: {args.cable_length} m")

    # --- Create or load model ---
    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=train_env)
        from stable_baselines3.common.utils import get_linear_fn
        model.clip_range    = get_linear_fn(args.clip_range, args.clip_range, 1.0)
        model.lr_schedule   = get_linear_fn(args.learning_rate, args.learning_rate, 1.0)
        model.learning_rate = args.learning_rate    
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            clip_range=args.clip_range,
            learning_rate=args.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=os.path.join(output_dir, "tb"),
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
        )

    # --- Callbacks ---
    # Stop early if we reach a good reward
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=args.target_reward,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        verbose=1,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="ppo_payload_hover",
    )

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"Starting training for {args.timesteps} steps...")
    print(f"{'='*60}\n")

    start_time = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10,
    )
    elapsed = time.time() - start_time
    print(f"\n[INFO] Training finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # --- Save final model ---
    final_path = os.path.join(output_dir, "final_model")
    model.save(final_path)
    print(f"[INFO] Final model saved to: {final_path}")

    # --- Evaluate ---
    print("\n[INFO] Evaluating best model...")
    best_model_path = os.path.join(output_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        best_model = PPO.load(best_model_path)
    else:
        best_model = model

    mean_reward, std_reward = evaluate_policy(
        best_model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"[INFO] Evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

    # --- Plot training curve ---
    eval_log = os.path.join(output_dir, "evaluations.npz")
    if os.path.exists(eval_log):
        data = np.load(eval_log)
        timesteps = data["timesteps"]
        results = data["results"].mean(axis=1)

        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, results, 'b-', linewidth=1, alpha=0.3, label="Raw")
        # Smoothed
        if len(results) > 5:
            smoothed = np.convolve(results, np.ones(5) / 5, mode="valid")
            plt.plot(timesteps[2:-2], smoothed, 'b-', linewidth=2, label="Smoothed")
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel("Training Steps")
        plt.ylabel("Mean Evaluation Reward")
        plt.title("Drone Payload Hover — Training Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150)
        print(f"[INFO] Training curve saved to: {output_dir}/training_curve.png")
        plt.show()

    # --- Cleanup ---
    train_env.close()
    eval_env.close()

    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for drone payload hover")
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--target_reward", type=float, default=250.0)
    parser.add_argument("--payload_mass", type=float, default=0.01)
    parser.add_argument("--cable_length", type=float, default=0.3)
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Parallel training environments (default: 8)")
    parser.add_argument("--goal_radius", type=float, default=3.0,
                        help="Max goal sampling radius in metres (default: 3.0)")
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    args = parser.parse_args()

    train(args)