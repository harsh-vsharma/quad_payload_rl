"""Point-to-point payload delivery demo.

Flies the drone to a user-specified (x, y, payload_z) target using the
trained V8 policy.  The drone stabilises with the payload hanging at the
requested height, then holds position.

Usage:
    # Deliver payload to (0.5, 0.5, 0.7m height) and hold for 5 seconds
    python demo_goto.py --model results/v8_sac_*/best_model.zip --target 0.5 0.5 0.7

    # Custom hold time and starting swing
    python demo_goto.py --model results/v8_sac_*/best_model.zip \\
        --target 1.0 -0.5 0.6 --hold 8.0 --episodes 2

    # Use a different algorithm
    python demo_goto.py --model results/v8_sac_*/best_model.zip \\
        --algo sac --target 0.0 0.0 0.8

Notes:
    - The drone spawns at (0, 0, 1.0m) with the payload hanging at z=0.7m.
    - The target is the PAYLOAD position, not the drone position.
    - The drone automatically hovers cable_length (0.3m) above the payload target.
    - The episode ends automatically after transit + hold time.
"""

import os
import sys
import time
import argparse

import numpy as np
import pybullet as p
from stable_baselines3 import SAC, PPO, TD3

from gym_pybullet_drones.envs.PayloadHoverAviary import PayloadHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import get_version_config


ALGO_MAP = {"sac": SAC, "ppo": PPO, "td3": TD3}
CABLE_LENGTH = 0.3   # must match environment default


def _draw_target_sphere(client, pos, radius=0.04, colour=(0, 1, 0, 0.6)):
    """Draw a translucent sphere at the payload target location."""
    visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius,
        rgbaColor=colour, physicsClientId=client,
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=pos,
        physicsClientId=client,
    )


def run_demo(args):
    # ------------------------------------------------------------------ #
    #  Validate model                                                      #
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    algo_cls = ALGO_MAP[args.algo.lower()]
    print(f"[INFO] Loading model  : {args.model}")
    model = algo_cls.load(args.model)

    # ------------------------------------------------------------------ #
    #  Resolve target positions                                            #
    # ------------------------------------------------------------------ #
    payload_target = np.array(args.target, dtype=np.float64)
    drone_target   = payload_target + np.array([0.0, 0.0, CABLE_LENGTH])

    print(f"\n  Target payload pos : ({payload_target[0]:+.3f}, "
          f"{payload_target[1]:+.3f}, {payload_target[2]:+.3f})")
    print(f"  Target drone  pos  : ({drone_target[0]:+.3f}, "
          f"{drone_target[1]:+.3f}, {drone_target[2]:+.3f})")
    print(f"  Hold duration      : {args.hold:.1f}s after arrival")

    # Validate height — payload must be above ground
    if payload_target[2] < CABLE_LENGTH + 0.05:
        print(f"[WARN] Payload target z={payload_target[2]:.2f}m is very low. "
              f"Minimum safe: {CABLE_LENGTH + 0.05:.2f}m")

    # ------------------------------------------------------------------ #
    #  Build config: fixed_target = drone target, no moving target        #
    # ------------------------------------------------------------------ #
    env_config = get_version_config(8)          # V8 — full capability
    env_config["moving_target"]   = False       # override to static
    env_config["target_weight"]   = 3.0         # keep target tracking active
    env_config["fixed_target"]    = drone_target.tolist()
    env_config["episode_len_sec"] = args.transit_time + args.hold + 2.0
    env_config["impulse_enabled"] = False        # clean demo — no random kicks
    env_config["wind_enabled"]    = args.wind    # optional

    # ------------------------------------------------------------------ #
    #  Run episodes                                                        #
    # ------------------------------------------------------------------ #
    for ep in range(args.episodes):
        print(f"\n{'='*65}")
        print(f"  Episode {ep+1}/{args.episodes}")
        print(f"{'='*65}")

        env = PayloadHoverAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            gui=True,
            record=args.record,
            config=env_config,
        )

        obs, info = env.reset(seed=ep)

        # Draw a green sphere at the payload target for visual reference
        _draw_target_sphere(env.CLIENT, payload_target.tolist())

        start       = time.time()
        total_steps = int(env_config["episode_len_sec"] * env.CTRL_FREQ)
        hold_start_step = int(args.transit_time * env.CTRL_FREQ)

        # Metrics
        payload_z_hist    = []
        swing_hist        = []
        drift_hist        = []
        h_err_hist        = []
        arrived           = False
        arrival_step      = None
        ARRIVAL_THRESH    = 0.15   # m — drone within 15cm of target = "arrived"

        print(f"\n  {'t':>5}  {'payload_z':>9}  {'tgt_z':>6}  "
              f"{'h_err':>6}  {'swing':>6}  {'dist':>6}  phase")
        print(f"  {'─'*65}")

        for step in range(total_steps):
            t = step / env.CTRL_FREQ

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            pld_z   = info.get("payload_z", 0.0)
            h_err   = info.get("payload_height_err", 0.0)
            swing   = info.get("swing_angle_deg", 0.0)
            drift   = info.get("drone_pos_error", 0.0)

            payload_z_hist.append(pld_z)
            swing_hist.append(swing)
            drift_hist.append(drift)
            h_err_hist.append(h_err)

            # Arrival detection
            drone_pos = np.array(info.get("drone_pos", [0, 0, 1]))
            dist_to_target = float(np.linalg.norm(drone_target - drone_pos))
            if not arrived and dist_to_target < ARRIVAL_THRESH:
                arrived = True
                arrival_step = step
                print(f"\n  >>> ARRIVED at target at t={t:.2f}s  "
                      f"(payload_z={pld_z:.3f}, target_z={payload_target[2]:.3f}) <<<\n")

            phase = "HOLD" if (arrived or step >= hold_start_step) else "TRANSIT"

            # Print every second
            if step % env.CTRL_FREQ == 0:
                print(f"  {t:5.1f}s  {pld_z:9.4f}  "
                      f"{payload_target[2]:6.3f}  {h_err:6.4f}  "
                      f"{swing:5.1f}°  {dist_to_target:5.3f}m  {phase}")

            sync(step, start, env.CTRL_TIMESTEP)

            if terminated or truncated:
                reason = "TERMINATED" if terminated else "TIME LIMIT"
                print(f"\n  Episode ended: {reason}")
                break

        # ── Summary ───────────────────────────────────────────────────
        hold_start = arrival_step if arrived else hold_start_step
        hold_slice = slice(hold_start, len(payload_z_hist))

        print(f"\n  {'─'*65}")
        print(f"  Demo Summary")
        print(f"    Payload target z    : {payload_target[2]:.3f} m")
        print(f"    Drone  target       : ({drone_target[0]:+.3f}, "
              f"{drone_target[1]:+.3f}, {drone_target[2]:+.3f})")
        if arrived:
            print(f"    Transit time        : {arrival_step / env.CTRL_FREQ:.2f}s  ✓")
        else:
            print(f"    Transit time        : did not reach within {ARRIVAL_THRESH}m  ✗")
        print(f"    — Full episode —")
        print(f"    Mean swing          : {np.mean(swing_hist):.2f}°")
        print(f"    Max  swing          : {np.max(swing_hist):.2f}°")
        print(f"    Mean payload h_err  : {np.mean(h_err_hist):.4f} m")
        if len(payload_z_hist[hold_slice]) > 0:
            print(f"    — Hold phase —")
            print(f"    Hold payload z      : {np.mean(payload_z_hist[hold_slice]):.4f} m  "
                  f"(target {payload_target[2]:.3f})")
            print(f"    Hold h_err (mean)   : {np.mean(h_err_hist[hold_slice]):.4f} m")
            print(f"    Hold swing (mean)   : {np.mean(swing_hist[hold_slice]):.2f}°")
            hold_ok = (
                np.mean(h_err_hist[hold_slice]) < 0.10
                and np.mean(swing_hist[hold_slice]) < 5.0
            )
            print(f"    Hold quality        : {'✓ STABLE' if hold_ok else '✗ UNSTABLE'}")

        print(f"  {'='*65}\n")
        env.close()

    print("[INFO] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Point-to-point payload delivery demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deliver payload to (0.5, 0.5) at 0.7m height
  python demo_goto.py --model results/v8_sac_*/best_model.zip --target 0.5 0.5 0.7

  # Far target with longer transit budget
  python demo_goto.py --model results/v8_sac_*/best_model.zip \\
      --target 1.0 -0.8 0.65 --transit_time 6.0 --hold 6.0

  # Origin hover at a specific height (demo the height control)
  python demo_goto.py --model results/v8_sac_*/best_model.zip --target 0.0 0.0 0.5

  # With wind disturbance
  python demo_goto.py --model results/v8_sac_*/best_model.zip \\
      --target 0.8 0.3 0.7 --wind
        """,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained V8 model .zip",
    )
    parser.add_argument(
        "--algo", type=str, default="sac", choices=["sac", "ppo", "td3"],
        help="Algorithm (default: sac)",
    )
    parser.add_argument(
        "--target", type=float, nargs=3,
        metavar=("X", "Y", "PAYLOAD_Z"),
        default=[0.5, 0.5, 0.7],
        help="Target (x, y, payload_z) in metres (default: 0.5 0.5 0.7)",
    )
    parser.add_argument(
        "--transit_time", type=float, default=5.0,
        help="Budget (seconds) to reach the target before hold phase (default: 5.0)",
    )
    parser.add_argument(
        "--hold", type=float, default=5.0,
        help="Seconds to hold at target after arrival (default: 5.0)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of demo runs (default: 1)",
    )
    parser.add_argument(
        "--wind", action="store_true",
        help="Enable wind disturbance during the demo",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record PyBullet video",
    )

    args = parser.parse_args()
    run_demo(args)
