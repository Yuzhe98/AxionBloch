"""Plot the packet-count convergence data written by the convergence study.

Run the study first, then run this plotting example from the repository root::

    python examples/convergency/inhomogeneity_numpt_convergence.py
    python examples/plot_convergency.py

The plot distinguishes packet-grid quadrature error from RK4 integration
error, using the pulse-aware prediction saved by the study.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path(__file__).parent / "convergency" / "inhomogeneity_numpt_convergence.csv"
PLOT_PATH = Path(__file__).parent / "convergency" / "inhomogeneity_numpt_summary.png"


def load_data(path: Path = DATA_PATH) -> dict[str, np.ndarray]:
    """Load the convergence study's CSV columns as NumPy arrays."""
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No convergence rows found in {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
    }


def main() -> None:
    data = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].loglog(data["numPt"], data["quadrature_Mx"], "o-", label=r"$M_x$")
    axes[0].loglog(data["numPt"], data["quadrature_My"], "s-", label=r"$M_y$")
    axes[0].loglog(data["numPt"], data["quadrature_total"], "^-", label="combined")
    axes[0].set(
        xlabel="Inhomogeneity packets (numPt)",
        ylabel="Normalized L2 error",
        title="Quadrature error vs. 3201-point theory",
    )
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].loglog(data["numPt"], data["integrator_Mx"], "o-", label=r"$M_x$")
    axes[1].loglog(data["numPt"], data["integrator_My"], "s-", label=r"$M_y$")
    axes[1].loglog(data["numPt"], data["integrator_total"], "^-", label="combined")
    axes[1].set(
        xlabel="Inhomogeneity packets (numPt)",
        ylabel="Normalized L2 error",
        title="RK4 error vs. pulse-aware theory",
    )
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Pulse-aware convergence at 4000 Hz, FWHM=0.1 ppm")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    print(f"Saved plot to {PLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
