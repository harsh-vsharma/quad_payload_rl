"""Unified training script for Drone Payload Stabilisation.

Supports SAC (default), PPO, and TD3.  Uses the PayloadHoverAviary
environment with version-configurable curriculum.

Usage examples:
    # V1 baseline with SAC (recommended first run)
    python train.py --version 1 --algo sac --timesteps 500000

    # V1 with PPO (baseline comparison)
    python train.py --version 1 --algo ppo --timesteps 1000000 --n_envs 8

    # V2 with SAC, resuming from V1 best model
    python train.py --version 2 --algo sac --resume results/v1_sac_*/best_model.zip

    # Load from YAML config
    python train.py --config configs/v3_disturbance.yaml --algo sac
"""

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CheckpointCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from gym_pybullet_drones.envs.PayloadHoverAviary import PayloadHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Add project root to path for configs module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import get_version_config, load_config, save_config


ALGO_MAP = {"sac": SAC, "ppo": PPO, "td3": TD3}


def make_env(env_config):
    """Factory function for creating the training environment."""
    return PayloadHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False,
        config=env_config,
    )


def build_model(algo_name, env, output_dir, args):
    """Create a fresh RL model with tuned hyperparameters."""
    algo_cls = ALGO_MAP[algo_name]

    common_kwargs = dict(
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb"),
        policy_kwargs=dict(
            net_arch=[256, 256],
        ),
    )

    if algo_name == "sac":
        model = algo_cls(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=300_000,
            learning_starts=1000,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            train_freq=1,
            gradient_steps=1,
            **common_kwargs,
        )

    elif algo_name == "ppo":
        model = algo_cls(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            **common_kwargs,
        )

    elif algo_name == "td3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        model = algo_cls(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            batch_size=256,
            buffer_size=300_000,
            learning_starts=1000,
            gamma=0.99,
            tau=0.005,
            action_noise=action_noise,
            **common_kwargs,
        )

    return model


def plot_training_curve(output_dir):
    """Plot evaluation reward curve from the evaluations.npz log."""
    eval_log = os.path.join(output_dir, "evaluations.npz")
    if not os.path.exists(eval_log):
        print("[WARN] No evaluations.npz found, skipping plot.")
        return

    data = np.load(eval_log)
    timesteps = data["timesteps"]
    results = data["results"].mean(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, results, "b-", linewidth=1, alpha=0.3, label="Raw")
    if len(results) > 5:
        smoothed = np.convolve(results, np.ones(5) / 5, mode="valid")
        plt.plot(
            timesteps[2:-2], smoothed, "b-", linewidth=2, label="Smoothed (5-ep)"
        )
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Evaluation Reward")
    plt.title("Payload Hover — Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curve saved to: {plot_path}")


def train(args):
    """Main training loop."""
    # ----- Config -----
    if args.config:
        env_config = load_config(args.config)
        version = env_config["version"]
    else:
        env_config = get_version_config(args.version)
        version = args.version

    # Apply CLI overrides
    if args.payload_mass is not None:
        env_config["payload_mass"] = args.payload_mass
    if args.cable_length is not None:
        env_config["cable_length"] = args.cable_length

    algo_name = args.algo.lower()
    if algo_name not in ALGO_MAP:
        print(f"[ERROR] Unknown algorithm: {algo_name}. Choose from: {list(ALGO_MAP.keys())}")
        return

    # ----- Output directory -----
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    run_name = f"v{version}_{algo_name}_{timestamp}"
    output_dir = os.path.join(args.output_folder, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    save_config(env_config, os.path.join(output_dir, "config.yaml"))
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Version: {version}, Algorithm: {algo_name.upper()}")
    print(f"[INFO] Config: {env_config}")

    # ----- Create environments -----
    if algo_name == "ppo":
        # PPO benefits from parallel envs
        train_env = make_vec_env(
            PayloadHoverAviary,
            env_kwargs=dict(
                obs=ObservationType.KIN,
                act=ActionType.RPM,
                gui=False,
                config=env_config,
            ),
            n_envs=args.n_envs,
            seed=0,
        )
    else:
        # SAC/TD3 use single env
        train_env = make_env(env_config)

    eval_env = make_env(env_config)

    print(f"[INFO] Action space:  {train_env.action_space}")
    print(f"[INFO] Obs space:     {eval_env.observation_space}")

    # ----- Create or resume model -----
    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        algo_cls = ALGO_MAP[algo_name]
        model = algo_cls.load(args.resume, env=train_env)
    else:
        model = build_model(algo_name, train_env, output_dir, args)

    # ----- Callbacks -----
    stop_cb = StopTrainingOnRewardThreshold(
        reward_threshold=args.target_reward, verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=stop_cb,
        verbose=1,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(args.timesteps // 100, 5000),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix=f"{algo_name}_v{version}",
    )

    # ----- Train -----
    print(f"\n{'='*60}")
    print(f"  Training {algo_name.upper()} on V{version} for {args.timesteps:,} steps")
    print(f"{'='*60}\n")

    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, ckpt_cb],
        log_interval=10,
    )
    elapsed = time.time() - t0
    print(f"\n[INFO] Training finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ----- Save final -----
    final_path = os.path.join(output_dir, "final_model")
    model.save(final_path)
    print(f"[INFO] Final model: {final_path}")

    # ----- Evaluate best -----
    best_path = os.path.join(output_dir, "best_model.zip")
    if os.path.exists(best_path):
        best_model = ALGO_MAP[algo_name].load(best_path)
    else:
        best_model = model

    mean_r, std_r = evaluate_policy(
        best_model, eval_env, n_eval_episodes=10, deterministic=True,
    )
    print(f"[INFO] Best model eval: {mean_r:.2f} ± {std_r:.2f}")

    # ----- Plot -----
    plot_training_curve(output_dir)

    # ----- Cleanup -----
    train_env.close()
    eval_env.close()

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL agent for drone payload stabilisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --version 1 --algo sac --timesteps 500000
  python train.py --version 1 --algo ppo --timesteps 1000000 --n_envs 8
  python train.py --config configs/v3_disturbance.yaml --algo sac
  python train.py --version 2 --algo sac --resume results/v1_sac_*/best_model.zip
        """,
    )

    parser.add_argument("--version", type=int, default=1,
                        help="Environment version 1-7 (default: 1)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides --version)")
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac", "ppo", "td3"],
                        help="RL algorithm (default: sac)")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--target_reward", type=float, default=400.0,
                        help="Early-stop reward threshold (default: 400)")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Parallel envs for PPO (default: 8)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model .zip to resume training from")
    parser.add_argument("--output_folder", type=str, default="results",
                        help="Output folder (default: results)")
    parser.add_argument("--payload_mass", type=float, default=None,
                        help="Override payload mass (kg)")
    parser.add_argument("--cable_length", type=float, default=None,
                        help="Override cable length (m)")

    args = parser.parse_args()
    train(args)
