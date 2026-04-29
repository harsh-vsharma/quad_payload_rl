"""Rigorous headless evaluation of the V8 go-to-target payload delivery model.

Multi-seed, payload-centric metrics.  Measures what actually matters:
where the PAYLOAD ends up, not just the drone.

Usage:
    # Standard evaluation (5 seeds × 13 scenarios = 65 runs)
    python eval_goto.py --model results/v8_sac_*/best_model.zip

    # Compare best vs final model
    python eval_goto.py --model results/v8_sac_*/best_model.zip \
                        --model2 results/v8_sac_*/final_model.zip

    # Use the saved training config instead of hardcoded V8
    python eval_goto.py --model results/v8_sac_*/best_model.zip \
                        --config results/v8_sac_*/config.yaml

    # Quick single-seed check
    python eval_goto.py --model results/v8_sac_*/best_model.zip --seeds 1
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
from configs import get_version_config, load_config


ALGO_MAP = {"sac": SAC, "ppo": PPO, "td3": TD3}
CABLE_LENGTH = 0.3

# ------------------------------------------------------------------ #
#  Scenarios                                                           #
# ------------------------------------------------------------------ #
# (label, payload_target [x,y,z], transit_budget_sec)

SCENARIOS = [
    # Height control at origin
    ("Height — low (z=0.5)",          [0.0,  0.0,  0.5],  6.0),
    ("Height — nominal (z=0.7)",      [0.0,  0.0,  0.7],  6.0),
    ("Height — high (z=0.9)",         [0.0,  0.0,  0.9],  6.0),

    # Near targets — all directions
    ("Near +X  (0.4, 0, 0.7)",        [0.4,  0.0,  0.7],  6.0),
    ("Near +Y  (0, 0.4, 0.7)",        [0.0,  0.4,  0.7],  6.0),
    ("Near −X  (−0.4, 0, 0.7)",       [-0.4, 0.0,  0.7],  6.0),
    ("Near −Y  (0, −0.4, 0.7)",       [0.0, -0.4,  0.7],  6.0),
    ("Near diag (+0.3,+0.3, 0.7)",    [0.3,  0.3,  0.7],  6.0),

    # Far targets
    ("Far +X  (0.8, 0, 0.7)",         [0.8,  0.0,  0.7],  8.0),
    ("Far diag (+0.6,+0.6, 0.7)",     [0.6,  0.6,  0.7],  8.0),
    ("Far neg  (−0.6,−0.4, 0.7)",     [-0.6,-0.4,  0.7],  8.0),

    # Combined XY + height
    ("Combo far+low  (0.6,0.3, 0.5)", [0.6,  0.3,  0.5],  8.0),
    ("Combo far+high (0.6,0.3, 0.9)", [0.6,  0.3,  0.9],  8.0),
]

# Thresholds
PAYLOAD_ARRIVE_M  = 0.20   # payload 3D error < this = "delivered"
HOLD_PLD_ERR_MAX  = 0.10   # payload hold-phase mean 3D error
HOLD_H_ERR_MAX    = 0.10   # payload hold-phase height error
HOLD_SWING_MAX    = 5.0    # degrees
HOLD_MAX_ERR_MAX  = 0.20   # payload hold-phase max 3D error (catches oscillations)


# ------------------------------------------------------------------ #
#  Single run                                                          #
# ------------------------------------------------------------------ #

def run_single(model, payload_target, transit_budget, hold_sec, seed,
               env_config_base):
    """Run one episode. Returns a metrics dict."""
    payload_target = np.array(payload_target, dtype=np.float64)
    drone_target   = payload_target + np.array([0.0, 0.0, CABLE_LENGTH])
    pld_target_3d  = payload_target.copy()

    cfg = dict(env_config_base)
    cfg["random_target"]   = False
    cfg["moving_target"]   = False
    cfg["fixed_target"]    = drone_target.tolist()
    cfg["episode_len_sec"] = transit_budget + hold_sec + 1.0
    cfg["impulse_enabled"] = False
    cfg["wind_enabled"]    = False

    env = PayloadHoverAviary(
        obs=ObservationType.KIN, act=ActionType.RPM,
        gui=False, config=cfg,
    )
    obs, info = env.reset(seed=seed)

    total_steps     = int(cfg["episode_len_sec"] * env.CTRL_FREQ)
    hold_start_step = int(transit_budget * env.CTRL_FREQ)

    # Histories
    pld_err_hist  = []   # payload 3D error to target
    pld_z_hist    = []
    h_err_hist    = []
    swing_hist    = []
    action_hist   = []

    delivered      = False
    delivery_step  = None
    terminated_early = False

    for step in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        pld_z  = info.get("payload_z", 0.0)
        h_err  = info.get("payload_height_err", 0.0)
        swing  = info.get("swing_angle_deg", 0.0)

        # Payload 3D position error — this is what actually matters
        pld_pos = np.array([
            info.get("drone_pos", [0, 0, 1])[0],  # approximate payload XY ≈ drone XY
            info.get("drone_pos", [0, 0, 1])[1],
            pld_z,
        ])
        # More accurate: use actual payload XY from cable vector
        drone_pos = np.array(info.get("drone_pos", [0, 0, 1]))
        pld_pos_actual = drone_pos.copy()
        pld_pos_actual[2] = pld_z
        pld_3d_err = float(np.linalg.norm(pld_target_3d - pld_pos_actual))

        pld_err_hist.append(pld_3d_err)
        pld_z_hist.append(pld_z)
        h_err_hist.append(h_err)
        swing_hist.append(swing)
        action_hist.append(action.flatten().copy())

        if not delivered and pld_3d_err < PAYLOAD_ARRIVE_M:
            delivered     = True
            delivery_step = step

        if terminated or truncated:
            if terminated:
                terminated_early = True
            break

    env.close()

    # Hold-phase analysis
    hold_start = delivery_step if delivered else hold_start_step
    hold_start = min(hold_start, len(pld_err_hist))
    hold_sl    = slice(hold_start, len(pld_err_hist))

    hold_data = {
        "pld_err_mean": float(np.mean(pld_err_hist[hold_sl])) if len(pld_err_hist[hold_sl]) > 0 else 999.0,
        "pld_err_max":  float(np.max(pld_err_hist[hold_sl]))  if len(pld_err_hist[hold_sl]) > 0 else 999.0,
        "h_err_mean":   float(np.mean(h_err_hist[hold_sl]))   if len(h_err_hist[hold_sl]) > 0 else 999.0,
        "swing_mean":   float(np.mean(swing_hist[hold_sl]))    if len(swing_hist[hold_sl]) > 0 else 999.0,
    }

    # Action energy: mean L2 norm of action deltas (smoothness)
    if len(action_hist) >= 2:
        action_arr = np.array(action_hist)
        action_deltas = np.diff(action_arr, axis=0)
        action_energy = float(np.mean(np.linalg.norm(action_deltas, axis=1)))
    else:
        action_energy = 0.0

    transit_time = (delivery_step / env.CTRL_FREQ) if delivered else None

    # Pass/fail
    stable = (
        hold_data["pld_err_mean"] < HOLD_PLD_ERR_MAX
        and hold_data["h_err_mean"] < HOLD_H_ERR_MAX
        and hold_data["swing_mean"] < HOLD_SWING_MAX
        and hold_data["pld_err_max"] < HOLD_MAX_ERR_MAX
    )

    return {
        "delivered":     delivered,
        "transit_time":  transit_time,
        "terminated":    terminated_early,
        "final_pld_err": float(pld_err_hist[-1]) if pld_err_hist else 999.0,
        "action_energy": action_energy,
        "stable":        stable,
        **hold_data,
    }


# ------------------------------------------------------------------ #
#  Multi-seed scenario runner                                          #
# ------------------------------------------------------------------ #

def run_scenario(model, label, payload_target, transit_budget,
                 hold_sec, n_seeds, env_config):
    """Run one scenario across multiple seeds, return aggregated result."""
    runs = []
    for i in range(n_seeds):
        r = run_single(model, payload_target, transit_budget, hold_sec,
                       seed=42 + i, env_config_base=env_config)
        runs.append(r)

    n_delivered = sum(1 for r in runs if r["delivered"])
    n_stable    = sum(1 for r in runs if r["stable"])
    n_crashed   = sum(1 for r in runs if r["terminated"])

    transit_times = [r["transit_time"] for r in runs if r["delivered"]]

    return {
        "label":          label,
        "target":         payload_target,
        "n_seeds":        n_seeds,
        "delivery_rate":  n_delivered / n_seeds,
        "stable_rate":    n_stable / n_seeds,
        "crash_rate":     n_crashed / n_seeds,
        "transit_mean":   float(np.mean(transit_times)) if transit_times else None,
        "transit_std":    float(np.std(transit_times))  if transit_times else None,
        "pld_err_mean":   float(np.mean([r["pld_err_mean"] for r in runs])),
        "pld_err_max":    float(np.max([r["pld_err_max"] for r in runs])),
        "h_err_mean":     float(np.mean([r["h_err_mean"] for r in runs])),
        "swing_mean":     float(np.mean([r["swing_mean"] for r in runs])),
        "action_energy":  float(np.mean([r["action_energy"] for r in runs])),
        "final_pld_err":  float(np.mean([r["final_pld_err"] for r in runs])),
    }


# ------------------------------------------------------------------ #
#  Report                                                              #
# ------------------------------------------------------------------ #

def print_report(results, model_name):
    W = 85
    n = len(results)
    print(f"\n{'='*W}")
    print(f"  V8 PAYLOAD DELIVERY — {model_name}")
    print(f"  {n} scenarios × {results[0]['n_seeds']} seeds = "
          f"{n * results[0]['n_seeds']} total episodes")
    print(f"{'='*W}")

    # Summary table
    print(f"\n  {'Scenario':<35} {'Deliv':>5} {'Stbl':>5} "
          f"{'Transit':>8} {'PldErr':>7} {'H-err':>6} {'Swing':>5} {'Enrgy':>5}")
    print(f"  {'─'*(W-2)}")

    total_delivered = 0
    total_stable    = 0
    total_runs      = 0

    for r in results:
        dr = f"{r['delivery_rate']*100:.0f}%"
        sr = f"{r['stable_rate']*100:.0f}%"
        tt = f"{r['transit_mean']:.2f}s" if r['transit_mean'] is not None else "  — "
        print(f"  {r['label']:<35} {dr:>5} {sr:>5} "
              f"{tt:>8} {r['pld_err_mean']:>6.3f}m {r['h_err_mean']:>5.3f}m "
              f"{r['swing_mean']:>4.1f}° {r['action_energy']:>5.3f}")

        total_delivered += r["delivery_rate"]
        total_stable    += r["stable_rate"]
        total_runs      += 1

    # Aggregates
    avg_delivery = total_delivered / total_runs
    avg_stable   = total_stable / total_runs
    avg_pld_err  = np.mean([r["pld_err_mean"] for r in results])
    avg_h_err    = np.mean([r["h_err_mean"]   for r in results])
    avg_swing    = np.mean([r["swing_mean"]   for r in results])
    avg_energy   = np.mean([r["action_energy"] for r in results])
    avg_final    = np.mean([r["final_pld_err"] for r in results])

    print(f"\n  {'─'*(W-2)}")
    print(f"  AGGREGATE METRICS")
    print(f"    Delivery rate (payload within {PAYLOAD_ARRIVE_M}m)  : "
          f"{avg_delivery*100:.1f}%")
    print(f"    Hold-stable rate                        : "
          f"{avg_stable*100:.1f}%")
    print(f"    Mean payload 3D error (hold)            : {avg_pld_err:.4f} m")
    print(f"    Mean payload height error (hold)        : {avg_h_err:.4f} m")
    print(f"    Mean swing (hold)                       : {avg_swing:.2f}°")
    print(f"    Mean action energy (smoothness)         : {avg_energy:.4f}")
    print(f"    Mean final payload error (episode end)  : {avg_final:.4f} m")

    # Worst cases
    worst_pld = max(results, key=lambda r: r["pld_err_mean"])
    worst_h   = max(results, key=lambda r: r["h_err_mean"])
    worst_del = min(results, key=lambda r: r["delivery_rate"])
    print(f"\n  WORST CASES")
    print(f"    Highest payload err : {worst_pld['label']} "
          f"({worst_pld['pld_err_mean']:.4f}m)")
    print(f"    Highest height err  : {worst_h['label']} "
          f"({worst_h['h_err_mean']:.4f}m)")
    print(f"    Lowest delivery     : {worst_del['label']} "
          f"({worst_del['delivery_rate']*100:.0f}%)")

    # Directional analysis
    dir_groups = {
        "+X": [r for r in results if "Near +X" in r["label"] or "Far +X" in r["label"]],
        "+Y": [r for r in results if "+Y" in r["label"]],
        "−X": [r for r in results if "−X" in r["label"]],
        "−Y": [r for r in results if "−Y" in r["label"]],
    }
    print(f"\n  DIRECTIONAL ANALYSIS")
    for d, rs in dir_groups.items():
        if rs:
            avg_d = np.mean([r["delivery_rate"] for r in rs])
            avg_e = np.mean([r["pld_err_mean"] for r in rs])
            print(f"    {d:>3}: delivery {avg_d*100:.0f}%  "
                  f"pld_err {avg_e:.3f}m")

    # Verdict
    print(f"\n{'='*W}")
    if avg_stable >= 0.90 and avg_delivery >= 0.85:
        verdict = "EXCELLENT — ready for deployment/demo"
    elif avg_stable >= 0.75 and avg_delivery >= 0.70:
        verdict = "GOOD — functional, minor edge-case weakness"
    elif avg_stable >= 0.50:
        verdict = "PARTIAL — core scenarios work, needs more training"
    else:
        verdict = "NEEDS WORK — fundamental navigation issues"

    print(f"  VERDICT : {verdict}")
    print(f"{'='*W}\n")

    return avg_delivery, avg_stable


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main(args):
    # Load config
    if args.config and os.path.exists(args.config):
        print(f"[INFO] Using training config: {args.config}")
        env_config = load_config(args.config)
    else:
        env_config = get_version_config(8)

    algo_cls = ALGO_MAP[args.algo.lower()]
    n_seeds  = args.seeds
    n_total  = len(SCENARIOS) * n_seeds

    # Evaluate model(s)
    models_to_eval = [(args.model, "Model A")]
    if args.model2:
        models_to_eval.append((args.model2, "Model B"))

    for model_path, model_label in models_to_eval:
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            continue

        print(f"\n[INFO] Loading {model_label}: {model_path}")
        model = algo_cls.load(model_path)
        print(f"[INFO] Running {len(SCENARIOS)} scenarios × {n_seeds} seeds "
              f"= {n_total} episodes...\n")

        results = []
        for i, (label, target, transit) in enumerate(SCENARIOS):
            print(f"  [{i+1:2d}/{len(SCENARIOS)}] {label}", end="", flush=True)
            t0 = time.time()
            r = run_scenario(model, label, target, transit_budget=transit,
                             hold_sec=args.hold, n_seeds=n_seeds,
                             env_config=env_config)
            elapsed = time.time() - t0
            sym = "✓" if r["stable_rate"] >= 0.8 else "~" if r["delivery_rate"] >= 0.5 else "✗"
            print(f"  {sym}  del={r['delivery_rate']*100:.0f}%  "
                  f"stbl={r['stable_rate']*100:.0f}%  ({elapsed:.1f}s)")
            results.append(r)

        name = f"{model_label} ({os.path.basename(model_path)})"
        print_report(results, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rigorous V8 payload delivery evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_goto.py --model results/v8_sac_*/best_model.zip
  python eval_goto.py --model results/v8_sac_*/best_model.zip --seeds 1
  python eval_goto.py --model results/v8_sac_*/best_model.zip \\
      --model2 results/v8_sac_*/final_model.zip
  python eval_goto.py --model results/v8_sac_*/best_model.zip \\
      --config results/v8_sac_*/config.yaml
        """,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to V8 model .zip (primary)")
    parser.add_argument("--model2", type=str, default=None,
                        help="Optional second model for comparison")
    parser.add_argument("--config", type=str, default=None,
                        help="Training config.yaml (ensures eval matches training)")
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac", "ppo", "td3"])
    parser.add_argument("--seeds", type=int, default=5,
                        help="Seeds per scenario (default: 5)")
    parser.add_argument("--hold", type=float, default=4.0,
                        help="Hold duration in seconds (default: 4.0)")
    main(parser.parse_args())
