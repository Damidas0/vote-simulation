"""Utilities to inspect one generated profile and visualize key metrics.

This module centralizes the former demo code that lived in
``vote_simulation/simulation/simulation.py`` and adds plotting helpers.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.manifold import MDS

from vote_simulation.models.data_generation.generator_registry import get_generator_builder, list_generator_codes

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def generate_and_visualize_profiles():
    """Generate profiles for all registered generators and visualize them."""

    codes = list_generator_codes()
    print("Registered generators:", codes)

    n_v, n_c = 1000, 3

    for gen_code in codes:
        if gen_code == "IANC":
            print(f"  → {gen_code} ... skipped (slow)", flush=True)
            continue
        print(f"  → {gen_code} ...", end=" ", flush=True)
        builder = get_generator_builder(gen_code)
        profile = builder(n_v=n_v, n_c=n_c, seed=42)

        # - Plot3 (svvamp 3-D utility scatter) ---
        profile.plot3()
        plot3_path = IMG_DIR / f"{gen_code.lower()}Plot3.png"
        plt.savefig(plot3_path, dpi=150, bbox_inches="tight")
        plt.close()

        # - MDS 2-D projection ---
        mds = MDS(n_components=2, random_state=42, normalized_stress="auto")
        coords_2d = mds.fit_transform(profile.preferences_ut)

        plt.figure(figsize=(5, 5))
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.7, edgecolors="k", linewidths=0.3)
        plt.title(f"{gen_code} (n_v={n_v}, n_c={n_c}) — MDS 2D projection")
        plt.tight_layout()
        mds_path = IMG_DIR / f"{gen_code.lower()}MDS.png"
        plt.savefig(mds_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"saved {plot3_path.name}, {mds_path.name}")

    print(f"\nAll plots saved to {IMG_DIR}")


if __name__ == "__main__":
    generate_and_visualize_profiles()
