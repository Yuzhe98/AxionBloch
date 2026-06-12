from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import (Baltimore, BuenosAires, CapeTown, Geneva,
                                Mainz, Mumbai, Sanya, Sydney, Tokyo)
from axionbloch.utils import check

stations = [
    Mainz,
    Geneva,
    Baltimore,
    Tokyo,
    Mumbai,
    Sanya,
    Sydney,
    CapeTown,
    BuenosAires,
]
# stations = [
#     CapeTown,
#     Mumbai,
# ]
# station = Baltimore
rhoE_DM = 0.3 * unit.GeV / unit.cm**3

states_to_check = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d"]
# states_to_check = ["2p", "3p"]

halo = EarthBoundAxionHalo(
    nu_a=1.348 * unit.MHz,  # axion Compton frequency in Hz
    N=int(2**12),  # number of grid points
    extent=128.0 * unit.R_earth,  # spatial extent of the grid in units of earth radius
    g_aNN=1e-9 * unit.GeV**-1,  # axion-nucleon coupling
    verbose=True,
)

halo.showValueAndUnits()


halo.solve_TISE_3D(
    l_vals=[0, 1, 2],  # angular momentum quantum number
    max_n_r=64,  # maximum principal quantum number to plot
    verbose=False,
)

# gradient obtain by Equation \ref{{eq:a0_Psi_real}} in the Yuzhe's thesis
N_a = halo.totalMassEnclosed / halo.m_a
gradient_factor = (
    const.c * halo.g_aNN * (2 * N_a * const.hbar**3 * const.c / halo.m_a) ** 0.5
)
gradients_dict = {}
for stateName in states_to_check:
    for station in stations:
        r, R_r, r_line, grad_r_line, grad_theta_line, grad_phi_line = (
            halo.findGradientsAtDirection(
                stateNames=[stateName],
                station=station,
                truncRadius=2 * unit.earthRad,
                showPlot=False,
                verbose=False,
            )
        )
        earthRad_idx = np.argmin(np.abs(r_line - 1 * unit.earthRad))
        gradients_dict[(stateName, station.name)] = {
            "grad_r": (gradient_factor * grad_r_line[earthRad_idx]).to(
                unit.Hz, equivalencies=unit.dimensionless_angles()
            ),  # ignore unit.radian for gradient value since it's just a scaling factor and doesn't affect the relative comparison between stations
            "grad_theta": (gradient_factor * grad_theta_line[earthRad_idx]).to(
                unit.Hz, equivalencies=unit.dimensionless_angles()
            ),
            "grad_phi": (gradient_factor * grad_phi_line[earthRad_idx]).to(
                unit.Hz,
                equivalencies=unit.dimensionless_angles(),  # TODO: check this. does it affect the magnitude?
            ),
        }

# scatter plot gradients for all stations and states
# x-axis: station name, y-axis: gradient value, different colors for different states

# repeat the plotting to set consistent limits for better comparison
bottom_min, top_max = 0, 0
for stateName in states_to_check:
    fig = plt.figure(figsize=(17 / 2.54, 4.0 / 2.54), dpi=300)  # initialize a figure
    gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
    ax = fig.add_subplot(gs[0, 0])
    for station in stations:
        grad_r_scatter = ax.scatter(
            [station.name],
            [gradients_dict[(stateName, station.name)]["grad_r"].value],
            color="blue",
            marker="o",
        )
        grad_theta_scatter = ax.scatter(
            [station.name],
            [gradients_dict[(stateName, station.name)]["grad_theta"].value],
            color="green",
            marker="s",
        )
        grad_phi_scatter = ax.scatter(
            [station.name],
            [gradients_dict[(stateName, station.name)]["grad_phi"].value],
            color="red",
            marker="^",
        )
    bottom, top = ax.get_ylim()
    bottom_min = min(bottom_min, bottom)
    top_max = max(top_max, top)
    plt.close()

# repeat the plotting to set consistent limits for better comparison
max_abs = max(abs(bottom_min), abs(top_max))
bottom_min, top_max = -max_abs, max_abs
for stateName in states_to_check:
    yUnit = gradients_dict[(stateName, station.name)]["grad_r"].unit
    check(yUnit.to_string("unicode"))
    plt.rcParams["lines.markersize"] = 2.5  # size of markers
    fig = plt.figure(figsize=(12 / 2.54, 4.0 / 2.54), dpi=300)  # initialize a figure
    gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
    ax = fig.add_subplot(gs[0, 0])
    for station in stations:
        grad_r_scatter = ax.scatter(
            [station.name],
            [gradients_dict[(stateName, station.name)]["grad_r"].value],
            color="blue",
            marker="o",
        )
        grad_theta_scatter = ax.scatter(
            [station.name],
            [gradients_dict[(stateName, station.name)]["grad_theta"].value],
            color="green",
            marker="s",
        )
        grad_phi_scatter = ax.scatter(
            [station.name],
            [gradients_dict[(stateName, station.name)]["grad_phi"].value],
            color="red",
            marker="^",
        )
    ax.tick_params(axis="x", labelrotation=20)
    # ax.set_xlabel("Station")
    ax.set_ylabel(f"Rabi frequency ({yUnit.to_string('latex_inline')})")
    ax.set_ylim(bottom_min, top_max)
    ax.legend(
        handles=[grad_r_scatter, grad_theta_scatter, grad_phi_scatter],
        labels=[
            "$\\partial_r\\phi$",
            "$\\frac{1}{r}\\partial_\\theta\\phi$",
            "$\\frac{1}{r\\sin\\theta}\\partial_\\varphi\\phi$",
        ],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )

    fig.suptitle("eigenstate: " + stateName, wrap=True)
    plt.tight_layout()
    plt.savefig(f"gradients_comparison_{stateName}.png", dpi=300, bbox_inches="tight")
    plt.show()
