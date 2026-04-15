"""Demo: SimulationTotalResult — full feature showcase.

Runs a small simulation grid (2 models × 1 voter count × 2 candidate counts)
then exercises every public method of SimulationTotalResult:

    1. Build via simulation_series_from_config  (or manual add_series)
    2. Accessors & repr
    3. Filtering
    4. Summary DataFrame
    5. Metric matrix (pivot)
    6. Rule-pair DataFrame
    7. Metric heatmap plot
    8. Rule-pair heatmap plot
    9. Comparison grid plot
   10. Persistence: save_to_dir / load_from_dir / delete_dir
"""

from __future__ import annotations

import os

from vote_simulation.models.results.total_result import SeriesKey, SimulationTotalResult
from vote_simulation.simulation.simulation import simulation_instance

# ── Configuration ────────────────────────────────────────────────────────────
# We use a small set of rules to keep it fast; adjust as needed.
RULES = ["PLU1", "PLU2", "BORD", "RV", "COPE", "HARE", "MJ"]
MODELS = ["UNI", "IC"]
VOTERS = [101]
CANDIDATES = [3, 14]
N_ITER = 50  # low for demo speed; use 1000+ in production
SEED = 42
BASE_PATH = "data"
DEMO_SAVE_DIR = "demo/results/total_demo"

SEP = "=" * 60


def build_total_manually() -> SimulationTotalResult:
    """Build a SimulationTotalResult by calling simulation_instance in a loop."""
    total = SimulationTotalResult()
    for model in MODELS:
        for n_v in VOTERS:
            for n_c in CANDIDATES:
                series = simulation_instance(
                    gen_code=model,
                    n_v=n_v,
                    n_c=n_c,
                    rule_codes=RULES,
                    n_iteration=N_ITER,
                    seed=SEED,
                    base_path=BASE_PATH,
                )
                total.add_series(series)
    return total


if __name__ == "__main__":
    # ── 1. Build ─────────────────────────────────────────────────────────────
    print(SEP)
    print("1) Building SimulationTotalResult (manual loop)")
    print(SEP)
    total = build_total_manually()

    # ── 2. Accessors & repr ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("2) Accessors & repr")
    print(SEP)
    print(f"  repr:             {total!r}")
    print(f"  series_count:     {total.series_count}")
    print(f"  len:              {len(total)}")
    print(f"  gen_models:       {total.gen_models}")
    print(f"  voter_counts:     {total.voter_counts}")
    print(f"  candidate_counts: {total.candidate_counts}")
    print(f"  keys:             {total.keys}")

    # __contains__
    print(f"  ('UNI',101,3) in: {SeriesKey('UNI', 101, 3) in total}")
    print(f"  ('XX',1,1) in:    {SeriesKey('XX', 1, 1) in total}")

    # get_series
    s = total.get_series("UNI", 101, 3)
    print(f"  get_series UNI/101/3: {s.step_count} steps, mean_dist={s.mean_distance:.2f}%")

    # __iter__
    print("  Iteration order:")
    for key, series in total:
        print(f"    {key}  {series.step_count} steps")

    # ── 3. Filtering ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("3) Filtering")
    print(SEP)
    uni = total.filter(gen_model="UNI")
    print(f"  filter(gen_model='UNI'):       {uni.series_count} series — {uni.keys}")

    c14 = total.filter(n_candidates=14)
    print(f"  filter(n_candidates=14):       {c14.series_count} series — {c14.keys}")

    one = total.filter(gen_model="IC", n_candidates=3)
    print(f"  filter(IC, n_candidates=3):    {one.series_count} series — {one.keys}")

    empty = total.filter(gen_model="MISSING")
    print(f"  filter(gen_model='MISSING'):   {empty.series_count} series")

    # 4. Summary DataFrame
    print(f"\n{SEP}")
    print("4) summary_frame()")
    print(SEP)
    df = total.summary_frame()
    print(df.to_string(index=False))

    # 5. Metric matrix (pivot)
    print(f"\n{SEP}")
    print("5) metric_matrix() — mean_distance pivoted by model × candidates")
    print(SEP)

    # Fix voters (only one value) → pivot models vs candidates
    pivot, desc = total.metric_matrix(
        row_param="gen_model",
        col_param="n_candidates",
        metric="mean_distance",
    )
    print(f"  Fixed: {desc}")
    print(pivot.to_string())

    # 6. Rule-pair DataFrame
    print(f"\n{SEP}")
    print("6) rule_pair_frame('PLU1', 'BORD')")
    print(SEP)
    rpf = total.rule_pair_frame("PLU1", "BORD")
    print(rpf.to_string(index=False))

    # 7. Metric heatmap plot
    print(f"\n{SEP}")
    print("7) plot_metric_heatmap()")
    print(SEP)
    save_hm = os.path.join(DEMO_SAVE_DIR, "metric_heatmap.png")
    total.plot_metric_heatmap(
        row_param="gen_model",
        col_param="n_candidates",
        metric="mean_distance",
        show=False,
        save_path=save_hm,
    )
    print(f"  Saved to {save_hm}")

    # 8. Rule-pair heatmap plot
    print(f"\n{SEP}")
    print("8) plot_rule_pair_heatmap('PLU1', 'BORD')")
    print(SEP)
    save_rph = os.path.join(DEMO_SAVE_DIR, "rule_pair_heatmap.png")
    total.plot_rule_pair_heatmap(
        "PLU1",
        "BORD",
        row_param="gen_model",
        col_param="n_candidates",
        show=False,
        save_path=save_rph,
    )
    print(f"  Saved to {save_rph}")

    # 9. Comparison grid plot
    print(f"\n{SEP}")
    print("9) plot_comparison_grid() — UNI, vary n_candidates")
    print(SEP)
    save_grid = os.path.join(DEMO_SAVE_DIR, "comparison_grid.png")
    uni_fixed = total.filter(gen_model="UNI", n_voters=101)
    uni_fixed.plot_comparison_grid(
        "n_candidates",
        show=False,
        save_path=save_grid,
    )
    print(f"  Saved to {save_grid}")

    # 10. Persistence
    print(f"\n{SEP}")
    print("10) Persistence: save_to_dir / load_from_dir / delete_dir")
    print(SEP)
    persist_dir = os.path.join(DEMO_SAVE_DIR, "parquet")

    # Save
    total.save_to_dir(persist_dir)
    files = os.listdir(persist_dir)
    print(f"  Saved {len(files)} parquet files to {persist_dir}/")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(persist_dir, f)) / 1024
        print(f"    {f}  ({size:.1f} KB)")

    # Load
    loaded = SimulationTotalResult.load_from_dir(persist_dir)
    print(f"\n  Loaded back: {loaded!r}")
    assert loaded.series_count == total.series_count

    # Verify metrics match
    orig_df = total.summary_frame()
    load_df = loaded.summary_frame()
    import numpy as np

    np.testing.assert_array_almost_equal(
        orig_df["mean_distance"].values,
        load_df["mean_distance"].values,
    )
    print("  Round-trip metrics match ✓")

    # Delete
    deleted = SimulationTotalResult.delete_dir(persist_dir)
    print(f"  Deleted {persist_dir}: {deleted}")
    print(f"  Exists after delete: {os.path.exists(persist_dir)}")

    # ── Done ─────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("Demo complete.")
    print(SEP)
