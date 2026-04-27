"""Evaluation and plotting for trained Drone Payload Stabilisation models.

Runs deterministic rollouts, collects per-step metrics, and generates
publication-quality plots for analysis and comparison.

Usage:
    python evaluate.py --model results/v1_sac_*/best_model.zip --algo sac
    python evaluate.py --model results/v1_sac_*/best_model.zip --algo sac --episodes 20
    python evaluate.py --compare results/v1_sac_*/best_model.zip results/v1_ppo_*/best_model.zip \\
                       --algos sac ppo --labels "SAC V1" "PPO V1"
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, PPO, TD3

from gym_pybullet_drones.envs.PayloadHoverAviary import PayloadHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import get_version_config, load_config


ALGO_MAP = {"sac": SAC, "ppo": PPO, "td3": TD3}


# ------------------------------------------------------------------ #
#  Rollout data collection                                             #
# ------------------------------------------------------------------ #

def collect_rollout(model, env_config, n_episodes=10, seed=42):
    """Run deterministic rollouts and collect per-step metrics.

    Returns
    -------
    list[dict]
        One dict per episode with keys:
        - swing_deg: list[float]
        - drone_z: list[float]
        - payload_z: list[float]
        - drone_vel: list[float]
        - payload_vel: list[float]
        - drone_pos_error: list[float]
        - rewards: list[float]
        - total_reward: float
        - n_steps: int
        - settling_time: float or None  (first time swing < 2°)
    """
    episodes = []

    for ep in range(n_episodes):
        env = PayloadHoverAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            gui=False,
            config=env_config,
        )
        obs, info = env.reset(seed=seed + ep)

        ep_data = {
            "swing_deg": [],
            "drone_z": [],
            "payload_z": [],
            "drone_vel": [],
            "payload_vel": [],
            "drone_pos_error": [],
            "tilt_rad": [],
            "rewards": [],
        }

        max_steps = int((env.EPISODE_LEN_SEC + 1) * env.CTRL_FREQ)
        settling_time = None

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_data["swing_deg"].append(info.get("swing_angle_deg", 0))
            ep_data["drone_z"].append(info.get("drone_z", 0))
            ep_data["payload_z"].append(info.get("payload_z", 0))
            ep_data["drone_vel"].append(info.get("drone_vel_norm", 0))
            ep_data["payload_vel"].append(info.get("payload_vel_norm", 0))
            ep_data["drone_pos_error"].append(info.get("drone_pos_error", 0))
            ep_data["tilt_rad"].append(info.get("tilt_rad", 0))
            ep_data["rewards"].append(reward)

            # Check settling: first time swing < 2° and stays < 2° for 1 second
            if settling_time is None and len(ep_data["swing_deg"]) > env.CTRL_FREQ:
                recent = ep_data["swing_deg"][-env.CTRL_FREQ:]
                if all(s < 2.0 for s in recent):
                    settling_time = (step - env.CTRL_FREQ) / env.CTRL_FREQ

            if terminated or truncated:
                break

        ep_data["total_reward"] = sum(ep_data["rewards"])
        ep_data["n_steps"] = len(ep_data["rewards"])
        ep_data["settling_time"] = settling_time

        episodes.append(ep_data)
        env.close()

    return episodes


# ------------------------------------------------------------------ #
#  Summary statistics                                                  #
# ------------------------------------------------------------------ #

def compute_summary(episodes):
    """Compute aggregate metrics across episodes."""
    total_rewards = [ep["total_reward"] for ep in episodes]
    settling_times = [ep["settling_time"] for ep in episodes if ep["settling_time"] is not None]
    final_swings = [ep["swing_deg"][-1] for ep in episodes]
    mean_swings = [np.mean(ep["swing_deg"]) for ep in episodes]
    mean_drifts = [np.mean(ep["drone_pos_error"]) for ep in episodes]

    summary = {
        "n_episodes": len(episodes),
        "reward_mean": np.mean(total_rewards),
        "reward_std": np.std(total_rewards),
        "settling_time_mean": np.mean(settling_times) if settling_times else None,
        "settling_time_std": np.std(settling_times) if settling_times else None,
        "settling_rate": len(settling_times) / len(episodes),
        "final_swing_mean": np.mean(final_swings),
        "final_swing_std": np.std(final_swings),
        "mean_swing": np.mean(mean_swings),
        "mean_drift": np.mean(mean_drifts),
    }
    return summary


def print_summary(summary, label=""):
    """Pretty-print summary stats."""
    if label:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")

    print(f"  Episodes:         {summary['n_episodes']}")
    print(f"  Reward:           {summary['reward_mean']:.1f} ± {summary['reward_std']:.1f}")
    print(f"  Settling rate:    {summary['settling_rate']*100:.0f}%")
    if summary["settling_time_mean"] is not None:
        print(f"  Settling time:    {summary['settling_time_mean']:.2f}s ± {summary['settling_time_std']:.2f}s")
    else:
        print(f"  Settling time:    N/A (never settled)")
    print(f"  Final swing:      {summary['final_swing_mean']:.1f}° ± {summary['final_swing_std']:.1f}°")
    print(f"  Mean swing:       {summary['mean_swing']:.1f}°")
    print(f"  Mean drift:       {summary['mean_drift']:.3f} m")


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

def plot_single_model(episodes, output_dir, label="Model"):
    """Generate per-model analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    ctrl_freq = 30  # approximate, for time axis

    # --- Swing angle over time (all episodes) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, ep in enumerate(episodes):
        t = np.arange(len(ep["swing_deg"])) / ctrl_freq
        ax.plot(t, ep["swing_deg"], alpha=0.4, linewidth=0.8)
    ax.axhline(y=2.0, color="r", linestyle="--", alpha=0.5, label="2° threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Swing Angle (°)")
    ax.set_title(f"{label} — Payload Swing Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "swing_angle.png"), dpi=150)
    plt.close(fig)

    # --- Drone Z altitude ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for ep in episodes:
        t = np.arange(len(ep["drone_z"])) / ctrl_freq
        ax.plot(t, ep["drone_z"], alpha=0.4, linewidth=0.8)
    ax.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="Target (1m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drone Z (m)")
    ax.set_title(f"{label} — Drone Altitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "drone_altitude.png"), dpi=150)
    plt.close(fig)

    # --- Reward per step ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for ep in episodes:
        t = np.arange(len(ep["rewards"])) / ctrl_freq
        ax.plot(t, ep["rewards"], alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title(f"{label} — Per-Step Reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rewards.png"), dpi=150)
    plt.close(fig)

    # --- Velocity norms ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ep in episodes:
        t = np.arange(len(ep["drone_vel"])) / ctrl_freq
        ax1.plot(t, ep["drone_vel"], alpha=0.3, linewidth=0.8)
        ax2.plot(t, ep["payload_vel"], alpha=0.3, linewidth=0.8)
    ax1.set_ylabel("Drone |vel| (m/s)")
    ax1.set_title(f"{label} — Velocities")
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Payload |vel| (m/s)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "velocities.png"), dpi=150)
    plt.close(fig)

    print(f"[INFO] Plots saved to: {output_dir}/")


def plot_comparison(all_summaries, labels, output_dir):
    """Generate side-by-side comparison bar charts."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(all_summaries)
    x = np.arange(n)

    # --- Reward comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    means = [s["reward_mean"] for s in all_summaries]
    stds = [s["reward_std"] for s in all_summaries]
    axes[0].bar(x, means, yerr=stds, capsize=5, color=plt.cm.Set2(x / n))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Total Reward")
    axes[0].grid(True, alpha=0.3, axis="y")

    # --- Settling time ---
    st_means = [s["settling_time_mean"] or 0 for s in all_summaries]
    st_stds = [s["settling_time_std"] or 0 for s in all_summaries]
    axes[1].bar(x, st_means, yerr=st_stds, capsize=5, color=plt.cm.Set2(x / n))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Settling Time")
    axes[1].grid(True, alpha=0.3, axis="y")

    # --- Final swing ---
    fs_means = [s["final_swing_mean"] for s in all_summaries]
    fs_stds = [s["final_swing_std"] for s in all_summaries]
    axes[2].bar(x, fs_means, yerr=fs_stds, capsize=5, color=plt.cm.Set2(x / n))
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].set_ylabel("Angle (°)")
    axes[2].set_title("Final Swing Angle")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Algorithm / Version Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
    plt.close(fig)
    print(f"[INFO] Comparison plot saved to: {output_dir}/comparison.png")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained drone payload models",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # --- Single model evaluation ---
    single_parser = sub.add_parser("single", help="Evaluate a single model")
    single_parser.add_argument("--model", type=str, required=True)
    single_parser.add_argument("--algo", type=str, default="sac",
                               choices=["sac", "ppo", "td3"])
    single_parser.add_argument("--version", type=int, default=1)
    single_parser.add_argument("--config", type=str, default=None)
    single_parser.add_argument("--episodes", type=int, default=10)
    single_parser.add_argument("--output", type=str, default="eval_results")

    # --- Multi-model comparison ---
    compare_parser = sub.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", nargs="+", required=True)
    compare_parser.add_argument("--algos", nargs="+", required=True)
    compare_parser.add_argument("--labels", nargs="+", default=None)
    compare_parser.add_argument("--version", type=int, default=1)
    compare_parser.add_argument("--config", type=str, default=None)
    compare_parser.add_argument("--episodes", type=int, default=10)
    compare_parser.add_argument("--output", type=str, default="eval_results")

    args = parser.parse_args()

    if args.command == "single":
        if args.config:
            env_config = load_config(args.config)
        else:
            env_config = get_version_config(args.version)

        algo_cls = ALGO_MAP[args.algo.lower()]
        model = algo_cls.load(args.model)

        label = f"{args.algo.upper()} V{env_config['version']}"
        print(f"[INFO] Evaluating: {label}")
        episodes = collect_rollout(model, env_config, args.episodes)
        summary = compute_summary(episodes)
        print_summary(summary, label)
        plot_single_model(episodes, args.output, label)

    elif args.command == "compare":
        if args.config:
            env_config = load_config(args.config)
        else:
            env_config = get_version_config(args.version)

        labels = args.labels or [
            f"{a.upper()} #{i}" for i, a in enumerate(args.algos)
        ]

        all_summaries = []
        for model_path, algo, label in zip(args.models, args.algos, labels):
            algo_cls = ALGO_MAP[algo.lower()]
            model = algo_cls.load(model_path)
            print(f"\n[INFO] Evaluating: {label}")
            episodes = collect_rollout(model, env_config, args.episodes)
            summary = compute_summary(episodes)
            print_summary(summary, label)
            plot_single_model(episodes, os.path.join(args.output, label.replace(" ", "_")), label)
            all_summaries.append(summary)

        plot_comparison(all_summaries, labels, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
