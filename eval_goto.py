"""Headless batch evaluation of the V8 point-to-point payload delivery model.

Runs a fixed set of delivery scenarios without any GUI and prints a
comprehensive performance report.  Designed for remote/server evaluation.

Usage:
    python eval_goto.py --model results/v8_sac_*/best_model.zip
    python eval_goto.py --model results/v8_sac_*/best_model.zip --algo sac
"""

import os
import sys
import time
import argparse

import numpy as np
from stable_baselines3 import SAC, PPO, TD3

from gym_pybullet_drones.envs.PayloadHoverAviary import PayloadHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import get_version_config


ALGO_MAP = {"sac": SAC, "ppo": PPO, "td3": TD3}
CABLE_LENGTH = 0.3   # matches environment default

# ------------------------------------------------------------------ #
#  Evaluation scenarios                                                #
# ------------------------------------------------------------------ #
# Each entry: (label, payload_target [x, y, z], transit_budget_sec)

SCENARIOS = [
    # --- Basic height control (no XY displacement) ---
    ("Origin — low  payload (z=0.5)",  [0.0,  0.0,  0.5],  6.0),
    ("Origin — nominal payload (z=0.7)",[0.0,  0.0,  0.7],  6.0),
    ("Origin — high payload (z=0.9)",  [0.0,  0.0,  0.9],  6.0),

    # --- Short XY displacement ---
    ("Near  — forward  (+0.4, 0, 0.7)",[0.4,  0.0,  0.7],  6.0),
    ("Near  — sideways (0, +0.4, 0.7)",[0.0,  0.4,  0.7],  6.0),
    ("Near  — diagonal (+0.3,+0.3,0.7)",[0.3, 0.3,  0.7],  6.0),

    # --- Long XY displacement ---
    ("Far   — forward  (+0.8, 0, 0.7)",[0.8,  0.0,  0.7],  8.0),
    ("Far   — diagonal (+0.8,+0.5,0.7)",[0.8, 0.5,  0.7],  8.0),
    ("Far   — negative (-0.6,-0.4,0.7)",[-0.6,-0.4, 0.7],  8.0),

    # --- Combined XY + height ---
    ("Combo — far + low  (0.6,0.3,0.5)",[0.6, 0.3,  0.5],  8.0),
    ("Combo — far + high (0.6,0.3,0.9)",[0.6, 0.3,  0.9],  8.0),
]

# Arrival threshold: drone within this distance = "arrived"
ARRIVAL_THRESH_M   = 0.15
# Hold-phase stability thresholds
HOLD_H_ERR_MAX     = 0.10   # m
HOLD_SWING_MAX     = 5.0    # degrees


# ------------------------------------------------------------------ #
#  Single-scenario runner                                              #
# ------------------------------------------------------------------ #

def run_scenario(model, label, payload_target, transit_budget, hold_sec=4.0):
    """Run one delivery scenario headlessly.  Returns a result dict."""
    payload_target = np.array(payload_target, dtype=np.float64)
    drone_target   = payload_target + np.array([0.0, 0.0, CABLE_LENGTH])

    env_config = get_version_config(8)
    env_config["moving_target"]   = False
    env_config["target_weight"]   = 3.0
    env_config["fixed_target"]    = drone_target.tolist()
    env_config["episode_len_sec"] = transit_budget + hold_sec + 1.0
    env_config["impulse_enabled"] = False
    env_config["wind_enabled"]    = False

    env = PayloadHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False,
        config=env_config,
    )
    obs, info = env.reset(seed=42)

    total_steps   = int(env_config["episode_len_sec"] * env.CTRL_FREQ)
    hold_start_step = int(transit_budget * env.CTRL_FREQ)

    pld_z_hist   = []
    h_err_hist   = []
    swing_hist   = []
    drift_hist   = []
    arrived      = False
    arrival_step = None
    episode_end  = total_steps

    for step in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        pld_z  = info.get("payload_z", 0.0)
        h_err  = info.get("payload_height_err", 0.0)
        swing  = info.get("swing_angle_deg", 0.0)

        drone_pos       = np.array(info.get("drone_pos", [0, 0, 1]))
        dist_to_target  = float(np.linalg.norm(drone_target - drone_pos))
        drift           = info.get("drone_pos_error", 0.0)

        pld_z_hist.append(pld_z)
        h_err_hist.append(h_err)
        swing_hist.append(swing)
        drift_hist.append(dist_to_target)

        if not arrived and dist_to_target < ARRIVAL_THRESH_M:
            arrived      = True
            arrival_step = step

        if terminated or truncated:
            episode_end = step
            break

    env.close()

    # Hold-phase slice
    hold_start = arrival_step if arrived else hold_start_step
    hold_sl    = slice(min(hold_start, len(pld_z_hist)), len(pld_z_hist))
    hold_h_err = np.mean(h_err_hist[hold_sl]) if len(h_err_hist[hold_sl]) > 0 else 999.0
    hold_swing = np.mean(swing_hist[hold_sl]) if len(swing_hist[hold_sl]) > 0 else 999.0

    transit_time = (arrival_step / env.CTRL_FREQ) if arrived else None
    hold_stable  = (hold_h_err < HOLD_H_ERR_MAX) and (hold_swing < HOLD_SWING_MAX)

    return dict(
        label        = label,
        target       = payload_target.tolist(),
        arrived      = arrived,
        transit_time = transit_time,
        # full-episode stats
        mean_h_err   = float(np.mean(h_err_hist)),
        max_h_err    = float(np.max(h_err_hist)),
        mean_swing   = float(np.mean(swing_hist)),
        max_swing    = float(np.max(swing_hist)),
        final_pld_z  = float(pld_z_hist[-1]) if pld_z_hist else 0.0,
        # hold-phase stats
        hold_h_err   = hold_h_err,
        hold_swing   = hold_swing,
        hold_stable  = hold_stable,
        episode_end  = episode_end,
    )


# ------------------------------------------------------------------ #
#  Report printing                                                     #
# ------------------------------------------------------------------ #

def print_report(results):
    W = 70
    print(f"\n{'='*W}")
    print(f"  V8 PAYLOAD DELIVERY — EVALUATION REPORT")
    print(f"{'='*W}")
    print(f"  {'Scenario':<40} {'Arrived':>7}  {'Transit':>7}  "
          f"{'H-err':>6}  {'Swing':>5}  {'Hold':>6}")
    print(f"  {'─'*W}")

    n_arrived = 0
    n_stable  = 0

    for r in results:
        arrived_sym  = "✓" if r["arrived"] else "✗"
        transit_str  = f"{r['transit_time']:.2f}s" if r["arrived"] else "  —   "
        hold_sym     = "✓ STABLE" if r["hold_stable"] else "✗ UNST."

        print(f"  {r['label']:<40} "
              f"{arrived_sym:>7}  "
              f"{transit_str:>7}  "
              f"{r['hold_h_err']:>5.3f}m  "
              f"{r['hold_swing']:>4.1f}°  "
              f"{hold_sym:>8}")

        if r["arrived"]:
            n_arrived += 1
        if r["hold_stable"]:
            n_stable += 1

    n = len(results)
    print(f"\n  {'─'*W}")
    print(f"  Arrival rate      : {n_arrived}/{n}  ({100*n_arrived/n:.0f}%)")
    print(f"  Hold-stable rate  : {n_stable}/{n}  ({100*n_stable/n:.0f}%)")

    arrived_results = [r for r in results if r["arrived"]]
    if arrived_results:
        avg_transit = np.mean([r["transit_time"] for r in arrived_results])
        avg_h_err   = np.mean([r["hold_h_err"]   for r in arrived_results])
        avg_swing   = np.mean([r["hold_swing"]    for r in arrived_results])
        print(f"  Avg transit time  : {avg_transit:.2f}s  (arrived scenarios only)")
        print(f"  Avg hold h_err    : {avg_h_err:.4f} m")
        print(f"  Avg hold swing    : {avg_swing:.2f}°")

    print(f"\n{'='*W}")
    print(f"  DETAILED METRICS")
    print(f"{'='*W}")
    for r in results:
        status = "✓" if (r["arrived"] and r["hold_stable"]) else "~" if r["arrived"] else "✗"
        print(f"\n  [{status}] {r['label']}")
        target = r["target"]
        print(f"      Payload target : ({target[0]:+.2f}, {target[1]:+.2f}, z={target[2]:.2f}m)")
        if r["arrived"]:
            print(f"      Transit time   : {r['transit_time']:.2f}s")
        else:
            print(f"      Transit        : did not reach within {ARRIVAL_THRESH_M}m")
        print(f"      Full ep swing  : mean {r['mean_swing']:.2f}°  max {r['max_swing']:.2f}°")
        print(f"      Full ep h_err  : mean {r['mean_h_err']:.4f}m  max {r['max_h_err']:.4f}m")
        print(f"      Hold h_err     : {r['hold_h_err']:.4f}m  {'✓' if r['hold_h_err'] < HOLD_H_ERR_MAX else '✗'}")
        print(f"      Hold swing     : {r['hold_swing']:.2f}°  {'✓' if r['hold_swing'] < HOLD_SWING_MAX else '✗'}")
        print(f"      Hold overall   : {'✓ STABLE' if r['hold_stable'] else '✗ UNSTABLE'}")

    # Overall verdict
    pass_rate = n_stable / n
    print(f"\n{'='*W}")
    if pass_rate >= 0.90:
        verdict = "EXCELLENT — model meets all delivery objectives"
    elif pass_rate >= 0.70:
        verdict = "GOOD — model is functional, some edge cases fail"
    elif pass_rate >= 0.50:
        verdict = "PARTIAL — core cases pass, distant/extreme targets struggle"
    else:
        verdict = "NEEDS WORK — significant instability in delivery scenarios"

    print(f"  VERDICT : {verdict}")
    print(f"  SCORE   : {n_stable}/{n} scenarios stable  ({100*pass_rate:.0f}%)")
    print(f"{'='*W}\n")


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main(args):
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    algo_cls = ALGO_MAP[args.algo.lower()]
    print(f"[INFO] Loading model : {args.model}")
    model = algo_cls.load(args.model)
    print(f"[INFO] Running {len(SCENARIOS)} delivery scenarios (headless)...\n")

    results = []
    for i, (label, target, transit) in enumerate(SCENARIOS):
        print(f"  [{i+1:2d}/{len(SCENARIOS)}]  {label} ...", end="", flush=True)
        t0 = time.time()
        r  = run_scenario(model, label, target, transit_budget=transit,
                          hold_sec=args.hold)
        elapsed = time.time() - t0
        status  = "✓" if (r["arrived"] and r["hold_stable"]) else "~" if r["arrived"] else "✗"
        print(f"  {status}  ({elapsed:.1f}s)")
        results.append(r)

    print_report(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Headless batch evaluation of V8 payload delivery model",
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained V8 model .zip")
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac", "ppo", "td3"],
                        help="Algorithm (default: sac)")
    parser.add_argument("--hold", type=float, default=4.0,
                        help="Hold duration at target in seconds (default: 4.0)")
    main(parser.parse_args())
