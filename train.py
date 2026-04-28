"""Unified training script for Drone Payload Stabilisation.

Supports SAC (default), PPO, and TD3.  Uses the PayloadHoverAviary
environment with version-configurable curriculum.

Performance notes
-----------------
This script auto-detects CUDA GPUs and parallelises PyBullet environments
across CPU cores using SubprocVecEnv.  Key optimisations:
  - SubprocVecEnv: parallel physics across CPU cores
  - train_freq=4, gradient_steps=2: reduces CPU↔GPU switching overhead
  - batch_size=1024: better GPU utilisation
  - torch thread pinning: prevents CPU oversubscription with SubprocVecEnv
On a 32-thread server (EPYC 7313 + RTX A5000), expect ~800–1400 FPS.

Usage examples:
    # V1 baseline with SAC — auto-detect hardware
    python train.py --version 1 --algo sac --timesteps 500000

    # Explicit parallelism: 16 envs, CUDA device
    python train.py --version 1 --algo sac --timesteps 500000 --n_envs 16 --device cuda

    # V1 with PPO (baseline comparison)
    python train.py --version 1 --algo ppo --timesteps 1000000 --n_envs 16

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
from multiprocessing import cpu_count

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
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


# ------------------------------------------------------------------ #
#  Hardware detection                                                  #
# ------------------------------------------------------------------ #

def detect_device(requested: str = "auto") -> str:
    """Resolve device string to 'cuda' or 'cpu'."""
    if requested == "auto":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] CUDA detected: {gpu_name}")
            return "cuda"
        else:
            print("[INFO] No CUDA GPU detected, using CPU.")
            return "cpu"
    return requested


def auto_n_envs(requested: int, algo_name: str) -> int:
    """Choose sensible n_envs if the user didn't override.

    Default (0 = auto):
      - Clamp to half the available CPU threads (leave room for OS/training).
      - Minimum 4, maximum 16 for SAC/TD3 (diminishing returns beyond that).
      - PPO can benefit from more; cap at 24.
    """
    if requested > 0:
        return requested

    n_cores = cpu_count() or 4
    n_auto  = max(4, n_cores // 2)

    if algo_name in ("sac", "td3"):
        n_auto = min(n_auto, 16)
    else:
        n_auto = min(n_auto, 24)

    print(f"[INFO] Auto n_envs={n_auto}  (from {n_cores} CPU threads)")
    return n_auto


# ------------------------------------------------------------------ #
#  Environment factory                                                 #
# ------------------------------------------------------------------ #

def make_single_env(env_config):
    """Create a single (non-vectorised) environment."""
    return PayloadHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False,
        config=env_config,
    )


def make_parallel_envs(env_config, n_envs, seed=0):
    """Create a SubprocVecEnv with n_envs parallel PyBullet instances.

    Each PyBullet simulation runs in its own subprocess, using a
    separate CPU core.  This is the single biggest speedup for
    physics-heavy environments.
    """
    return make_vec_env(
        PayloadHoverAviary,
        env_kwargs=dict(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            gui=False,
            config=env_config,
        ),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )


# ------------------------------------------------------------------ #
#  Model builder                                                       #
# ------------------------------------------------------------------ #

def build_model(algo_name, env, output_dir, device, n_envs, args):
    """Create a fresh RL model with hardware-aware hyperparameters.

    Key performance tuning (SAC):
      - train_freq=(4, 'step'): collect 4 env steps before each training
        call.  Reduces the number of CPU↔GPU switches.
      - gradient_steps=2: only 2 SGD updates per training call (instead
        of matching n_envs).  This is the single biggest speed gain.
      - batch_size=1024: fills the GPU properly on A5000-class cards.
      - learning_starts=10_000: fill the replay buffer before training.
    """
    algo_cls = ALGO_MAP[algo_name]

    common_kwargs = dict(
        verbose=1,
        device=device,
        tensorboard_log=os.path.join(output_dir, "tb"),
    )

    if algo_name == "sac":
        model = algo_cls(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            batch_size=1024,                 # better GPU utilisation
            buffer_size=1_000_000,
            learning_starts=10_000,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            train_freq=(4, "step"),           # fewer update calls
            gradient_steps=2,                # MASSIVE speed improvement
            policy_kwargs=dict(
                net_arch=[256, 256],
            ),
            **common_kwargs,
        )

    elif algo_name == "ppo":
        model = algo_cls(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=max(256, n_envs * 32),   # scale batch with envs
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
        grad_steps = max(n_envs, 4)

        model = algo_cls(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            batch_size=512,
            buffer_size=1_000_000,
            learning_starts=max(10_000, n_envs * 100),
            gamma=0.99,
            tau=0.005,
            action_noise=action_noise,
            train_freq=1,
            gradient_steps=grad_steps,
            **common_kwargs,
        )

    return model


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Main training function                                              #
# ------------------------------------------------------------------ #

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
        print(f"[ERROR] Unknown algorithm: {algo_name}. "
              f"Choose from: {list(ALGO_MAP.keys())}")
        return

    # ----- Hardware -----
    device = detect_device(args.device)
    n_envs = auto_n_envs(args.n_envs, algo_name)

    # ----- Output directory -----
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    run_name = f"v{version}_{algo_name}_{timestamp}"
    output_dir = os.path.join(args.output_folder, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    save_config(env_config, os.path.join(output_dir, "config.yaml"))

    print(f"[INFO] Output:     {output_dir}")
    print(f"[INFO] Version:    {version}")
    print(f"[INFO] Algorithm:  {algo_name.upper()}")
    print(f"[INFO] Device:     {device}")
    print(f"[INFO] Envs:       {n_envs} parallel (SubprocVecEnv)")
    print(f"[INFO] Timesteps:  {args.timesteps:,}")
    print(f"[INFO] Config:     {env_config}")

    # ----- Create environments -----
    # ALL algorithms now use parallel envs for maximum throughput.
    train_env = make_parallel_envs(env_config, n_envs, seed=0)
    # Monitor wraps the eval env so SB3 can reliably track episode stats.
    eval_env  = Monitor(make_single_env(env_config))

    print(f"[INFO] Action space:  {train_env.action_space}")
    print(f"[INFO] Obs space:     {eval_env.observation_space}")

    # ----- Create or resume model -----
    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        algo_cls = ALGO_MAP[algo_name]
        model = algo_cls.load(args.resume, env=train_env, device=device)
    else:
        model = build_model(algo_name, train_env, output_dir, device,
                            n_envs, args)

    # ----- Callbacks -----
    # Early-stop: only enable if the user set an explicit threshold.
    # Default is disabled (-1) because the do-nothing hover baseline already
    # exceeds a naive threshold before any real learning has occurred.
    if args.target_reward > 0:
        stop_cb = StopTrainingOnRewardThreshold(
            reward_threshold=args.target_reward, verbose=1,
        )
        eval_cb_on_best = stop_cb
    else:
        eval_cb_on_best = None

    # SB3 callbacks fire based on env.step() *calls*, not total timesteps.
    # With SubprocVecEnv(n_envs), each call collects n_envs transitions.
    # We define intervals in real timesteps, then convert to step-calls.
    EVAL_EVERY_TIMESTEPS = 25_000    # eval every 25K real timesteps
    CKPT_EVERY_TIMESTEPS = 50_000    # checkpoint every 50K real timesteps

    eval_freq = max(1, EVAL_EVERY_TIMESTEPS // n_envs)
    ckpt_freq = max(1, CKPT_EVERY_TIMESTEPS // n_envs)

    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=eval_cb_on_best,
        verbose=1,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=3,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix=f"{algo_name}_v{version}",
    )

    # ----- Train -----
    print(f"\n{'='*65}")
    print(f"  Training {algo_name.upper()} on V{version}")
    print(f"  {args.timesteps:,} steps  |  {n_envs} envs  |  {device}")
    print(f"  eval every {EVAL_EVERY_TIMESTEPS:,} ts  |  ckpt every {CKPT_EVERY_TIMESTEPS:,} ts")
    print(f"{'='*65}\n")

    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, ckpt_cb],
        log_interval=100,
    )
    elapsed = time.time() - t0
    avg_fps = args.timesteps / elapsed
    print(f"\n[INFO] Training finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[INFO] Average throughput: {avg_fps:.0f} FPS")

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
Performance tips:
  On a multi-core server with GPU, use --n_envs 16 --device cuda.
  Default (--n_envs 0) auto-detects and uses half the CPU threads.

Examples:
  python train.py --version 1 --algo sac --timesteps 500000
  python train.py --version 1 --algo sac --timesteps 500000 --n_envs 16 --device cuda
  python train.py --version 1 --algo ppo --timesteps 1000000 --n_envs 16
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
    parser.add_argument("--target_reward", type=float, default=-1.0,
                        help="Early-stop reward threshold; -1 disables (default: -1)")
    parser.add_argument("--n_envs", type=int, default=0,
                        help="Parallel envs; 0=auto-detect (default: 0)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="PyTorch device (default: auto)")
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
