"""Quick analysis of all training runs in results/.
Run from the project root: python3 analyse_results.py
"""
import numpy as np
import os

results_dir = "results"

print("=" * 65)
print("  TRAINING RUN ANALYSIS")
print("=" * 65)

for run in sorted(os.listdir(results_dir)):
    npz_path = os.path.join(results_dir, run, "evaluations.npz")
    cfg_path  = os.path.join(results_dir, run, "config.yaml")

    if not os.path.exists(npz_path):
        print(f"\n{run}: no evaluations.npz (likely failed or aborted)")
        continue

    data  = np.load(npz_path)
    ts    = data["timesteps"]                    # (n_evals,)
    R     = data["results"]                      # (n_evals, n_ep)
    lens  = data.get("ep_lengths", None)         # (n_evals, n_ep) or None

    means    = R.mean(axis=1)
    stds     = R.std(axis=1)
    best_idx = int(means.argmax())

    ep_lens_mean = lens.mean(axis=1) if lens is not None else [None] * len(ts)
    max_ep_steps = 300  # 10 s × 30 Hz

    print(f"\n{'─'*65}")
    print(f"  Run : {run}")
    print(f"  Evals: {len(ts)}   Steps: {ts[0]:,} → {ts[-1]:,}")
    print(f"  First eval reward : {means[0]:+.1f} ± {stds[0]:.1f}")
    print(f"  Final eval reward : {means[-1]:+.1f} ± {stds[-1]:.1f}")
    print(f"  Best  eval reward : {means[best_idx]:+.1f} ± {stds[best_idx]:.1f}  @ step {ts[best_idx]:,}")
    if ep_lens_mean[0] is not None:
        print(f"  Ep length first/best/last: "
              f"{ep_lens_mean[0]:.0f} / {ep_lens_mean[best_idx]:.0f} / {ep_lens_mean[-1]:.0f}"
              f"  (max={max_ep_steps})")

    print()
    print(f"  {'Step':>8}  {'Mean R':>8}  {'Std R':>6}  {'Ep len':>6}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}")
    for i, (t, m, s) in enumerate(zip(ts, means, stds)):
        el = f"{ep_lens_mean[i]:.0f}" if ep_lens_mean[i] is not None else "  -  "
        tag = " <-- BEST" if i == best_idx else ""
        print(f"  {t:>8,}  {m:>+8.1f}  {s:>6.1f}  {el:>6}{tag}")

# Verdict
print("\n" + "=" * 65)
print("  VERDICT")
print("=" * 65)
print("""
Key questions to answer:
  1. Is the mean reward rising over training?           (should be YES)
  2. Is the best model at or near the end?             (should be near end)
  3. Is episode length hitting the max (300)?           (should be YES if stable)
  4. Is std low relative to mean?                      (low = consistent policy)
""")
