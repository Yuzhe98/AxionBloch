import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate as correlate

from scipy.interpolate import interp1d

from astropy import units as unit
from astropy.constants import codata2018 as const
from astropy.units import Quantity

from axionbloch.enphylope import PhysicalQuantity as PQ
from axionbloch.constants import AtomicUnits as AU


from axionbloch.utils import check
from axionbloch.GravBoundAxionHalo import GravBoundAxionHalo


# def loadPEMdata():
#     # Load the data
#     data = pd.read_csv(
#         "data/Earth_Models/PEM_Parametric_Earth_Models_data.txt",
#         comment="#",
#         sep=r"\s+",
#         header=None,
#         names=[
#             "radius_m",
#             "density_kg_m3",
#             "vpv",
#             "vsv",
#             "Q_kappa",
#             "Q_miu",
#             "vph",
#             "vsh",
#             "eta",
#         ],
#     )

#     # Sort just in case
#     data = data.sort_values(by="radius_m")

#     return data


# def loadPEMdata(filepath="axionbloch/data/Earth_Models/PEM_Parametric_Earth_Models_data.txt"):
#     """
#     Load PEM data file without pandas.
#     Returns a list of dictionaries (like rows).
#     """

#     columns = [
#         "radius_m",
#         "density_kg_m3",
#         "vpv",
#         "vsv",
#         "Q_kappa",
#         "Q_miu",
#         "vph",
#         "vsh",
#         "eta",
#     ]

#     data = []

#     with open(filepath, "r") as f:
#         for line in f:
#             line = line.strip()

#             # skip comments and empty lines
#             if not line or line.startswith("#"):
#                 continue

#             parts = line.split()

#             # convert to float
#             values = [float(x) for x in parts]

#             # map to dict
#             row = dict(zip(columns, values))
#             data.append(row)

#     # sort by radius_m
#     data.sort(key=lambda x: x["radius_m"])

#     return data


def loadPEMdata(
    filepath="axionbloch/data/Earth_Models/PEM_Parametric_Earth_Models_data.txt",
):
    data = {
        "radius_m": [],
        "density_kg_m3": [],
        "vpv": [],
        "vsv": [],
        "Q_kappa": [],
        "Q_miu": [],
        "vph": [],
        "vsh": [],
        "eta": [],
    }

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            values = [float(x) for x in line.split()]

            for key, val in zip(data.keys(), values):
                data[key].append(val)

    # sort by radius
    order = sorted(range(len(data["radius_m"])), key=lambda i: data["radius_m"][i])

    for key in data:
        data[key] = [data[key][i] for i in order]
    for key in data:
        data[key] = np.asanyarray(data[key])
    return data


def PREM_density(radius_km):
    # Use the coefficients of the polynomials describing the Preliminary Reference Earth Model (PREM) to find out density.
    radius_km = np.abs(radius_km)

    # Earth's radius in km (converted to meters in DataFrame)
    EARTH_RADIUS_KM = 6371

    # PREM density functions for each region (x = r/6371)
    def density_inner_core(x):
        return 13.0885 - 8.8381 * x**2

    def density_outer_core(x):
        return 12.5815 - 1.2638 * x - 3.6426 * x**2 - 5.5281 * x**3

    def density_lower_mantle(x):
        return 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3

    def density_transition_zone(x):
        if x <= 5771 / EARTH_RADIUS_KM:
            return 5.3197 - 1.4836 * x
        elif x <= 5971 / EARTH_RADIUS_KM:
            return 11.2494 - 8.0298 * x
        else:
            return 7.1089 - 3.8045 * x

    def density_lvz_lid(x):
        return 2.6910 + 0.6924 * x

    def density_crust(x):
        if x <= 6356 / EARTH_RADIUS_KM:
            return 2.900
        else:
            return 2.600

    def density_ocean(x):
        return 1.020

    def density_space(x):
        return 0.0

    # # Generate radius values (in km) at 10 km intervals
    # radius_km = np.arange(0, EARTH_RADIUS_KM., 10)

    # Calculate density for each radius
    # density = []
    x = radius_km / EARTH_RADIUS_KM
    # radius_km = radius_km
    if radius_km <= 1221.5:
        return density_inner_core(x)
    elif radius_km <= 3480.0:
        return density_outer_core(x)
    elif radius_km <= 5701.0:
        return density_lower_mantle(x)
    elif radius_km <= 6151.0:
        return density_transition_zone(x)
    elif radius_km <= 6346.6:
        return density_lvz_lid(x)
    elif radius_km <= 6368.0:
        return density_crust(x)
    elif radius_km <= 6371.0:
        return density_ocean(x)
    else:
        return density_space(x)


def earth_grav_potential_infty():
    """
    Returns a function Phi(r[m]) [J/kg], valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    # Load the data (assumed to return a DataFrame with 'radius_m' and 'density_kg_m3')
    data = loadPEMdata()

    # Extract radius and density
    r = data["radius_m"]
    rho = data["density_kg_m3"]

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    M_total = M_r[-1]

    # Gravitational constant
    G = 6.67430e-11  # m³/kg/s²

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    r_outside = np.linspace(
        r_max, 1000 * r_max, 800
    )  # from surface to 1000 Earth radii
    Phi_outside = -G * M_total / r_outside

    # Compute gravitational potential inside Earth
    Phi_inside = np.zeros_like(r)
    Phi_inside[1:] = 2 * Phi_outside[0] + G * M_r[1:] / r[1:]
    Phi_inside[0] = 2 * Phi_outside[0] + G * M_r[1] / r[1]  # avoid zero-division

    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])
    Phi_full = np.concatenate([Phi_inside, Phi_outside])
    # Phi_full -= np.amin(Phi_full)

    # Enforce symmetry: add negative r values
    r_sym = np.concatenate([-r_full[::-1], r_full])  # mirror and append
    Phi_sym = np.concatenate([Phi_full[::-1], Phi_full])  # symmetric values

    # Optional: sort to ensure increasing r (for interp1d)
    sorted_indices = np.argsort(r_sym)
    r_sym_sorted = r_sym[sorted_indices]
    Phi_sym_sorted = Phi_sym[sorted_indices]

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    Phi_func = interp1d(
        r_sym_sorted,
        Phi_sym_sorted,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )

    return Phi_func


def earth_grav_potential_earth_center_au():
    """
    Returns a function Phi(r[a.u.]) [a.u.], valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    # Load the data (assumed to return a DataFrame with 'radius_m' and 'density_kg_m3')
    data = loadPEMdata()

    # Extract radius and density
    r = data["radius_m"]
    rho = data["density_kg_m3"]

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    M_total = M_r[-1]

    # Gravitational constant
    G = 6.67430e-11  # m³/kg/s²

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    r_outside = np.linspace(
        r_max, 1000 * r_max, 100000
    )  # from surface to 1000 Earth radii
    Phi_outside = -G * M_total / r_outside

    # Compute gravitational potential inside Earth
    Phi_inside = np.zeros_like(r)
    Phi_inside[1:] = 2 * Phi_outside[0] + G * M_r[1:] / r[1:]
    Phi_inside[0] = 2 * Phi_outside[0] + G * M_r[1] / r[1]  # avoid zero-division

    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])
    Phi_full = np.concatenate([Phi_inside, Phi_outside])
    # Phi_full -= np.amin(Phi_full)

    # Enforce symmetry: add negative r values
    r_sym = np.concatenate([-r_full[::-1], r_full])  # mirror and append
    Phi_sym = np.concatenate([Phi_full[::-1], Phi_full])  # symmetric values

    # Optional: sort to ensure increasing r (for interp1d)
    sorted_indices = np.argsort(r_sym)
    r_sym_sorted = r_sym[sorted_indices]
    Phi_sym_sorted = Phi_sym[sorted_indices]
    Phi_sym_sorted -= np.amin(Phi_sym_sorted)

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    # TODO make this more modular
    Phi_func = interp1d(
        r_sym_sorted * AU.meter,
        Phi_sym_sorted * (AU.J / AU.kg),
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    # check(np.amax(r_sym_sorted) / m)
    return Phi_func


def solid_sphere_grav_potential_earth_center_au():
    """
    Returns a function Phi(r[a.u.]) [a.u.], valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    # Load the data (assumed to return a DataFrame with 'radius_m' and 'density_kg_m3')
    data = loadPEMdata()

    # Extract radius
    r = data["radius_m"]
    R_earth = r[-1]  # [m]

    M_total = 5.974e24  # kg
    rho_mean = M_total / (4 / 3.0 * np.pi * R_earth**3)  # kg/m3

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho_mean * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    M_total = M_r[-1]

    # Gravitational constant
    G = 6.67430e-11  # m³/kg/s²

    # Extend to radii beyond Earth's surface

    r_outside = np.linspace(
        R_earth, 1000 * R_earth, 800
    )  # from surface to 1000 Earth radii
    Phi_outside = -G * M_total / r_outside

    # Compute gravitational potential inside Earth
    Phi_inside = np.zeros_like(r)
    Phi_inside[1:] = 2 * Phi_outside[0] + G * M_r[1:] / r[1:]
    Phi_inside[0] = 2 * Phi_outside[0] + G * M_r[1] / r[1]  # avoid zero-division

    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])
    Phi_full = np.concatenate([Phi_inside, Phi_outside])
    # Phi_full -= np.amin(Phi_full)

    # Enforce symmetry: add negative r values
    r_sym = np.concatenate([-r_full[::-1], r_full])  # mirror and append
    Phi_sym = np.concatenate([Phi_full[::-1], Phi_full])  # symmetric values

    # Optional: sort to ensure increasing r (for interp1d)
    sorted_indices = np.argsort(r_sym)
    r_sym_sorted = r_sym[sorted_indices]
    Phi_sym_sorted = Phi_sym[sorted_indices]
    Phi_sym_sorted -= np.amin(Phi_sym_sorted)

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    Phi_func = interp1d(
        r_sym_sorted * AU.meter,
        Phi_sym_sorted * (AU.J / AU.kg),
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    # check(np.amax(r_sym_sorted) / m)
    return Phi_func


def earth_grav_potential_earth_center_():
    """
    Returns a function Phi(r), valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    # Load the data (assumed to return a DataFrame with 'radius_m' and 'density_kg_m3')
    data = loadPEMdata()

    # Extract radius and density
    r = data["radius_m"]
    rho = data["density_kg_m3"]

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    M_total = M_r[-1]

    # Gravitational constant
    G = const.G.value  # m³/kg/s²

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    r_outside = np.linspace(
        r_max, 1000 * r_max, 100000
    )  # from surface to 1000 Earth radii
    Phi_outside = -G * M_total / r_outside

    # Compute gravitational potential inside Earth
    Phi_inside = np.zeros_like(r)
    Phi_inside[1:] = 2 * Phi_outside[0] + G * M_r[1:] / r[1:]
    Phi_inside[0] = 2 * Phi_outside[0] + G * M_r[1] / r[1]  # avoid zero-division

    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])
    Phi_full = np.concatenate([Phi_inside, Phi_outside])
    # Phi_full -= np.amin(Phi_full)

    # Enforce symmetry: add negative r values
    r_sym = np.concatenate([-r_full[::-1], r_full])  # mirror and append
    Phi_sym = np.concatenate([Phi_full[::-1], Phi_full])  # symmetric values

    # Optional: sort to ensure increasing r (for interp1d)
    sorted_indices = np.argsort(r_sym)
    r_sym_sorted = r_sym[sorted_indices]
    Phi_sym_sorted = Phi_sym[sorted_indices]
    Phi_sym_sorted -= np.amin(Phi_sym_sorted)

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    # TODO make this more modular
    Phi_func = interp1d(
        r_sym_sorted,
        Phi_sym_sorted,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    r_unit = PQ(1, "meter")
    Phi_unit = (
        PQ(1, "m**3/kg/s**2") * PQ(1, "kg") / PQ(1, "meter")
    )  #  = PQ(1, "joule / kilogram")
    # print(Phi_unit.to("joule / kilogram"))
    return Phi_func, r_unit, Phi_unit


def get_CumulativeMass():
    """
    Returns the cumulative mass as a function of radius.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    # Load the data (assumed to return a DataFrame with 'radius_m' and 'density_kg_m3')
    data = loadPEMdata()

    # Extract radius and density
    r = data["radius_m"] * unit.meter
    rho = data["density_kg_m3"] * (unit.kg / unit.meter**3)

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    return r, M_r


def earth_grav_potential_earth_center():
    """
    Returns a function Phi(r), valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    r, M_r = get_CumulativeMass()
    M_total = M_r[-1]

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    r_outside = np.linspace(
        r_max, 1000 * r_max, 100000
    )  # from surface to 1000 Earth radii
    Phi_outside: Quantity = -const.G * M_total / r_outside  # G = 6.67430e-11 m³/kg/s²

    # Compute gravitational potential inside Earth
    Phi_inside = np.zeros_like(r) / r.unit * Phi_outside[0].unit
    Phi_inside[1:] = 2 * Phi_outside[0] + const.G * M_r[1:] / r[1:]
    Phi_inside[0] = 2 * Phi_outside[0] + const.G * M_r[1] / r[1]  # avoid zero-division

    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])
    Phi_full = np.concatenate([Phi_inside, Phi_outside])
    # Phi_full -= np.amin(Phi_full)

    # Enforce symmetry: add negative r values
    r_sym = np.concatenate([-r_full[::-1], r_full])  # mirror and append
    Phi_sym = np.concatenate([Phi_full[::-1], Phi_full])  # symmetric values

    # Optional: sort to ensure increasing r (for interp1d)
    sorted_indices = np.argsort(r_sym)
    r_sym_sorted = r_sym[sorted_indices]
    Phi_sym_sorted = Phi_sym[sorted_indices]
    Phi_sym_sorted -= np.amin(Phi_sym_sorted)

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    r_unit = unit.R_earth
    Phi_unit = unit.joule / unit.kilogram
    
    Phi_func = interp1d(
        r_sym_sorted.to_value(r_unit),
        Phi_sym_sorted.to_value(Phi_unit),
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    return Phi_func, r_unit, Phi_unit

def plot_earth_grav_potential():

    # Load the data
    data = loadPEMdata()

    # Compute shell volumes and mass in each shell
    r = data["radius_m"] * unit.meter
    # outside earth
    r_outside = np.linspace(r[-1], 10 * r[-1], 400)  # from surface to 10 Earth radii
    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])

    rho = data["density_kg_m3"] * (unit.kg / unit.meter**3)

    r, M_r = get_CumulativeMass()
    Phi_func, r_unit, Phi_unit = earth_grav_potential_earth_center()
    r_extended = np.linspace(0, 3 * r[-1], 1000)

    print("Earth radius = ", r[-1] / 1e3, "km")
    # Compute shell thicknesses (dr)
    dr = np.gradient(r)

    # Volume of each spherical shell = 4πr² dr
    dV = 4 * np.pi * r**2 * dr

    # Mass in each shell = ρ * dV
    dm = rho * dV

    # Total mass
    M_total = np.sum(dm)  # in kg

    # Cumulative mass as a function of radius
    M_r = np.cumsum(dm)  # in kg

    # Constants
    G = 6.67430e-11  # gravitational constant [m³/kg/s²]

    Phi_func = earth_grav_potential_infty()

    # Gravitational potential in [J/kg]
    Phi = np.zeros_like(r)
    Phi[1:] = G * M_r[1:] / r[1:]
    Phi[0] = 0  #
    Phi += Phi_func(0)  # use this line if assuming the potential at infinity is zero

    # Print result
    print(f"Total Earth mass from PEM profile: {M_total:.3e} kg")

    # Plot

    # plot style
    plt.rc("font", size=6)  # font size for all figures
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Make math text match Times New Roman
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["mathtext.rm"] = "Times New Roman"

    # plt.style.use('seaborn-dark')  # to specify different styles
    # print(plt.style.available)  # if you want to know available styles

    cm = 1 / 2.56  # convert cm to inch

    fig = plt.figure(figsize=(8.5 * cm, 8.5 * cm), dpi=300)  # initialize a figure

    gs = gridspec.GridSpec(nrows=2, ncols=2)  # create grid for multiple figures

    # # fix the margins
    # left=
    # bottom=
    # right=
    # top=
    # wspace=
    # hspace=
    # fig.subplots_adjust(left=left, top=top, right=right,
    #                     bottom=bottom, wspace=wspace, hspace=hspace)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0:])
    # ax11 = fig.add_subplot(gs[1, 1])

    # earth density
    ax00.plot(
        r / 1000,
        data["density_kg_m3"] / 1000,
        label="Density Profile",
        color="darkblue",
    )
    ax00.set_ylim(0, 15)
    ax00.set_xlabel("Radius (km)")
    ax00.set_ylabel("Density (g/cm³)")
    # ax00.set_title("Earth Density Profile (PEM Data)")
    # ax00.grid(True)
    # ax00.legend()
    # ax00.gca().invert_xaxis()  # Optional: so Earth's center is on the left

    # earth mass
    ax01.plot(r / 1000, M_r / 1e24, label="Mass Profile", color="darkgreen")
    # ax00.set_ylim(0, 15)
    ax01.set_xlabel("Radius (km)")
    ax01.set_ylabel("Enclosed Mass ($10^{24}\\,$kg)")
    # ax01.set_title("Earth Density Profile (PEM Data)")
    # ax01.grid(True)
    # ax01.legend()
    # ax01.gca().invert_xaxis()  # Optional: so Earth's center is on the left
    ax01.ticklabel_format(useOffset=False)

    # earth gravitational potential given by Phi_r
    # ax10.plot(r / 1000, Phi / 1e6, label="Obtained by earth density", color="k")
    # ax10.plot(
    #     r / 1000,
    #     Phi_func(r) / 1e6,
    #     label="Grav. Pot. by Phi_func",
    #     color="darkorange",
    #     linestyle="dashed",
    # )

    # Obtained by earth density and calculation
    ax10.plot(
        r_full / 1000,
        Phi_func(r_full) / 1e6,
        label="Earth grav. pot.",
        color="darkorange",
        # linestyle="dashed",
    )
    ax10.axvline(
        x=r[-1] / 1e3,
        color="k",
        linestyle="dotted",
        linewidth=1,
        alpha=0.8,
        label="Earth radius",
    )

    # ax10.axhline(
    #     y=r[-1] / 1e3,
    #     color="k",
    #     linestyle="dotted",
    #     linewidth=1,
    #     alpha=0.8,
    #     label="Earth radius",
    # )

    ax10.axvline(
        x=5 * r[-1] / 1e3,
        color="green",
        linestyle="dashed",
        linewidth=1,
        alpha=0.8,
        label="$5\\times\\,$Earth radius",
    )

    # ax10.axvline(
    #     x=384400 ,
    #     color="tab:orange",
    #     linestyle="dotted",
    #     linewidth=1,
    #     alpha=0.8,
    #     label="Moon orbital radius",
    # )

    ax10.set_xlim(-0.1 * r[-1] / 1e3, 5.5 * r[-1] / 1e3)
    ax10.set_ylim(-130, 5)
    ax10.set_xlabel("Radius (km)")
    ax10.set_ylabel("Grav. Pot. (MJ/kg) ref. to $\\infty$")
    # ax10.set_title("Earth Density Profile (PEM Data)")
    # ax10.grid(True)
    ax10.legend()
    # ax10.gca().invert_xaxis()  # Optional: so Earth's center is on the left

    fig.suptitle("Earth Profiles (from PEM Data)")
    fig.tight_layout()
    plt.savefig("figures/Earth-Profiles-(PEM-Data).png", transparent=False)
    plt.show()


class EarthBoundAxionHalo(GravBoundAxionHalo):
    # Create the "axion stream" (axion field) object
    # you can get properties of the axion field, computed based on the input information
    def __init__(
        self,
        name="Earth-Bound Axion Halo",
        nu_a: Quantity = None,  # axion Compton frequency
        N: int = int(2**12),
        extent: Quantity = 128.0 * unit.R_earth,
        Phi_func=earth_grav_potential_earth_center(),
        verbose: bool = False,
    ):
        super().__init__(
            name=name,
            nu_a=nu_a,
            N=N,
            extent=extent,
            Phi_func=Phi_func,
            verbose=verbose,
        )

    def getBfield(
        self,
        rate_Hz: float,
        timeLen: int,
        rand_seed: int,
        numFields: int = 1,
        verbose: bool = False,
    ):
        # 1 MHz axion
        eigenEnergies_eV = [
            8.185620266405553e-19,
            2.230884526115819e-18,
            3.17699654698912e-18,
            3.3025178098192375e-18,
            3.302688446107381e-18,
            3.8484585622919455e-18,
            4.1004249795459864e-18,
            4.1004251503701215e-18,
            4.341652193125044e-18,
            4.407871816010306e-18,
            4.408009935951216e-18,
            4.649824763113033e-18,
            4.793034098628449e-18,
            4.7930342283222715e-18,
            4.8831463914209464e-18,
            4.919259056702106e-18,
            4.919342580168352e-18,
            5.042907628381992e-18,
            5.124371776790224e-18,
            5.124371863603223e-18,
            5.1684784792208225e-18,
            5.1897014793057166e-18,
            5.189752636536869e-18,
            5.260382049853425e-18,
            5.310111086559351e-18,
            5.310111144839848e-18,
            5.335147509172691e-18,
            5.348511363425495e-18,
            5.348544297168566e-18,
            5.3924738060527425e-18,
            5.424802508063358e-18,
            5.4248025483118064e-18,
            5.440435917665003e-18,
            5.449341478496205e-18,
            5.449363729093159e-18,
            5.478480245818686e-18,
            5.500595123173712e-18,
            5.5005951518592556e-18,
            5.511033438087251e-18,
            5.517245349489185e-18,
            5.517261015401705e-18,
            5.537528824223334e-18,
            5.553289963907005e-18,
            5.553289984958548e-18,
            5.560615718245721e-18,
            5.565112331351988e-18,
            5.565123747711691e-18,
            5.579789494642934e-18,
            5.591403811323711e-18,
            5.591403827177387e-18,
        ]
        self.rate_Hz: float = rate_Hz
        self.timeLen: int = timeLen
        self.duration_s: float = self.timeLen / self.rate_Hz
        eigenEnergies_eV = np.asarray(eigenEnergies_eV)
        eigenFreqs_Hz: np.ndarray = eigenEnergies_eV  # / h_Planck.value_in("eV * s")
        if verbose:
            print(eigenFreqs_Hz.mean())
        B_rms_T = 1e-15
        self.Ba = np.zeros(timeLen)
        timeStamp = np.arange(timeLen) / rate_Hz
        rng = (
            np.random.default_rng(seed=rand_seed)
            if rand_seed is not None
            else np.random.default_rng()
        )

        # phases
        phases = (
            2 * np.pi * rng.random((len(eigenFreqs_Hz), numFields))
        )  # shape: (numFreqs, numFields)

        phase_time: np.ndarray = (
            2 * np.pi * eigenFreqs_Hz[:, None] * timeStamp[None, :]
        )  # shape: (numFreqs, timeLen)

        total_phase = phase_time[:, :, None] + phases[:, None, :]
        # shape: (numFreqs, timeLen, numFields)

        self.Ba = np.exp(1j * total_phase).sum(axis=0)  # shape: (timeLen, numFields)
        self.Ba *= B_rms_T
        if verbose:
            print("phases.shape", phases.shape)
            print("phase_time.shape", phase_time.shape)
            print("total_phase.shape", total_phase.shape)
            print("self.Ba.shape", self.Ba.shape)
        return self.Ba

    def coh_time_g1(self):
        """
        x : complex-valued time series
        dt: sampling interval
        method: "1e" or "integral"
        """
        x = self.Ba[:, 0] - np.mean(self.Ba[:, 0])
        dt = 1 / self.rate_Hz
        E = np.array(x)  # complex field

        N = len(E)

        # tic = time.time()
        corr = correlate(E, E.conj(), mode="full")
        # toc = time.time()
        # print(f"Time taken for correlation: {toc - tic:.3f} seconds")
        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(np.abs(corr), label="")
        ax00.set_xlabel("")
        ax00.set_ylabel("corr (arb. units)")
        ax00.legend()
        fig.suptitle("", wrap=True)
        plt.tight_layout()
        plt.show()

        check(len(corr) // 2)
        corr = corr[len(corr) // 2 :]
        g1 = corr / corr[0]

        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(g1.real, label="real part")
        ax00.plot(g1.imag, label="imaginary part")
        ax00.set_xlabel("time (s)")
        ax00.set_ylabel("g1 (arb. units)")
        ax00.legend()
        fig.suptitle("", wrap=True)
        plt.tight_layout()
        plt.show()

        tau = 2 * np.sum(np.abs(g1)) * dt
        print("tau =", tau)
        print("duration =", self.duration_s)
        if tau > self.duration_s:
            print("WARNING: tau > self.duration_s")
        return tau


# rate_Hz = 3e-2
# duration_s = 1e6
# timeLen = int(rate_Hz * duration_s)
# axion = EarthBoundAxionHalo()
# axion.getBfield(rate_Hz=rate_Hz, timeLen=timeLen, rand_seed=1, numFields=1, verbose=False)
# axion.coh_time_g1()
