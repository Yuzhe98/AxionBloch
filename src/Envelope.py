##########################################
# for unit management and physical calculation
##########################################

from pint import UnitRegistry
from typing import Optional
import numpy as np

# Create a Unit Registry for managing units
ureg = UnitRegistry()
# Define units
ureg.define(
    "parsec = 30856775814913673 * meter = pc"
)  # Parsec. The built-in parsec is not precise
ureg.define("solar_mass = 1.98847e30 * kilogram = M_sun")  # Solar Mass
ureg.define("ppb = 1e-9")  # parts per billion


class PhysicalQuantity:
    """
    1. Base Units
        Length:
            meter (m), centimeter (cm), kilometer (km), millimeter (mm),
            micrometer (µm), nanometer (nm), etc.
        Mass:
            kilogram (kg), gram (g), milligram (mg), microgram (µg),
            tonne (t), etc.
        Time:
            second (s), minute (min), hour (h), day (d).
        Electric Current:
            ampere (A).
        Temperature:
            kelvin (K), degree Celsius (°C).
        Amount of Substance:
            mole (mol).
        Luminous Intensity:
            candela (cd).
    2. Derived Units
        Area:
            square meter (m²), square kilometer (km²), hectare (ha), acre.
        Volume:
            cubic meter (m³), liter (L), milliliter (mL).
        Speed/Velocity:
            meter per second (m/s), kilometer per hour (km/h), miles per hour (mph).
        Acceleration:
            meter per second squared (m/s²).
        Force:
            newton (N).
        Energy:
            joule (J), kilojoule (kJ), calorie (cal), electronvolt (eV), kilowatt-hour (kWh).
        Power:
            watt (W), kilowatt (kW), horsepower (hp).
        Pressure:
            pascal (Pa), atmosphere (atm), bar, millibar (mbar), torr.
        Charge:
            coulomb (C).
        Voltage:
            volt (V).
        Resistance:
            ohm (Ω).
        Capacitance:
            farad (F).
        Inductance:
            henry (H).
        Magnetic Field Strength:
            tesla (T), gauss (G).
        Magnetic Flux:
            weber (Wb).
        Frequency:
            hertz (Hz).
        Angular Velocity:
            radian per second (rad/s).
    3. Atomic and Particle Physics Units
        Atomic Mass Unit:
            unified atomic mass unit (u or amu).
        Electronvolt:
            electronvolt (eV), megaelectronvolt (MeV), gigaelectronvolt (GeV).
        Planck Units:
            Planck length, Planck time, Planck mass, Planck charge, Planck temperature.
        Photon Energy:
            joule (J), electronvolt (eV).
        Magnetic Moment:
            joule per tesla (J/T), ampere meter squared (A·m²).
    4. Thermodynamics
        Entropy:
            joule per kelvin (J/K).
        Specific Heat Capacity:
            joule per kilogram kelvin (J/(kg·K)).
        Thermal Conductivity:
            watt per meter kelvin (W/(m·K)).
    5. Other Common Units in Physics
        Luminance:
            candela per square meter (cd/m²).
        Radiance:
            watt per steradian square meter (W/(sr·m²)).
        Irradiance/Flux:
            watt per square meter (W/m²).
        Solar Mass:
            mass of the sun (M☉).
        Astronomical Unit:
            distance from Earth to the Sun (au).
        Light-Year:
            distance light travels in one year.
        Parsec:
            3.26 light-years.
        Barn:
            area used in nuclear physics (10⁻²⁸ m²).
    """

    def __init__(self, value: float, unit: str):
        """
        Initialize a physical quantity using pint's Quantity.
        :param value: Numerical value of the quantity
        :param unit: String representing the unit
        """
        self.quantity = value * ureg(unit)

    @property
    def value(self):
        return self.quantity.magnitude

    @property
    def unit(self):
        return str(self.quantity.units)

    def __repr__(self):
        """
        String representation of the physical quantity.
        """
        return f"{self.quantity}"

    def __add__(self, other):
        """
        Add two physical quantities.
        """
        assert isinstance(other, PhysicalQuantity)
        result = self.quantity + other.quantity
        return PhysicalQuantity(result.magnitude, str(result.units))

    def __sub__(self, other):
        """
        Subtract two physical quantities.
        """
        assert isinstance(other, PhysicalQuantity)
        result = self.quantity - other.quantity
        return PhysicalQuantity(result.magnitude, str(result.units))

    def __mul__(self, other):
        """
        Multiply two physical quantities or multiply two physical quantity by a scalar on the right.
        """
        if isinstance(other, PhysicalQuantity):
            result = self.quantity * other.quantity
        elif isinstance(other, (int, float)):  #
            result = PhysicalQuantity(other * self.value, str(self.unit)).quantity
        else:
            TypeError(f"Unsupported division with type {type(other)}")
        return PhysicalQuantity(result.magnitude, str(result.units))

    def __rmul__(self, other):
        """
        Multiply two physical quantities or multiply two physical quantity by a scalar on the left.
        """
        if isinstance(other, PhysicalQuantity):
            result = self.quantity * other.quantity
        elif isinstance(other, (int, float)):  #
            result = PhysicalQuantity(self.value * other, str(self.unit)).quantity
        else:
            TypeError(f"Unsupported division with type {type(other)}")
        return PhysicalQuantity(result.magnitude, str(result.units))

    def __truediv__(self, other):
        """
        Divide two physical quantities or by a scalar.
        """
        if isinstance(other, PhysicalQuantity):
            result = self.quantity / other.quantity
        elif isinstance(other, (int, float)):  # Division by scalar (right-hand side)
            result = PhysicalQuantity(self.value / other, str(self.unit)).quantity
        else:
            TypeError(f"Unsupported right-hand division with type {type(other)}")
        return PhysicalQuantity(result.magnitude, str(result.units))

    def __rtruediv__(self, other):
        if isinstance(other, PhysicalQuantity):
            result = self.quantity / other.quantity
        elif isinstance(other, (int, float)):  # Division by scalar (left-hand side)
            result = PhysicalQuantity(
                other / self.value,
                "(" + str(self.unit) + ")**-1",
            ).quantity
        else:
            raise TypeError(f"Unsupported left-hand division with type {type(other)}")
        return PhysicalQuantity(result.magnitude, str(result.units))

    def __pow__(self, power):
        """
        Raise a physical quantity to a power.
        """
        result = self.quantity**power
        return PhysicalQuantity(result.magnitude, str(result.units))

    def __abs__(self):
        """
        Return the absolute value of the physical quantity.
        Units are preserved.

        Example
        -------
        q1 = PhysicalQuantity(-3.5, "T")
        print(abs(q1))  # PhysicalQuantity(3.5, 'T')
        """
        return PhysicalQuantity(abs(self.value), self.unit)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercept NumPy ufunc calls including
        np.exp, np.tanh, np.sin, np.cos, np.sinh, np.cosh, np.add, np.subtract, np.multiply, np.divide
        """
        # Only handle the default '__call__' method (the actual function call)
        if method != "__call__":
            return NotImplemented

        # Extract magnitudes and check units
        magnitudes = []
        units = set()
        for x in inputs:
            if isinstance(x, PhysicalQuantity):
                magnitudes.append(x.value)
                units.add(x.unit)
            else:
                magnitudes.append(x)
                units.add("")

        # Handle specific ufuncs
        if ufunc in (np.exp, np.tanh, np.sin, np.cos, np.sinh, np.cosh):
            # Convert all inputs to dimensionless before computation
            dimless_magnitudes = [
                x.convert_to("").value if isinstance(x, PhysicalQuantity) else x
                for x in inputs
            ]
            result = ufunc(*dimless_magnitudes)
            return PhysicalQuantity(result, "")  # Result is dimensionless

        elif ufunc in (np.add, np.subtract):
            # For add/subtract, units must match
            if len(units) > 1:
                raise ValueError("Cannot add/subtract quantities with different units.")
            result = ufunc(*magnitudes)
            return PhysicalQuantity(result, units.pop())

        elif ufunc in (np.multiply, np.divide):
            # Combine units roughly
            if ufunc is np.multiply:
                combined_unit = "*".join(u for u in units if u)
            else:
                # crude example for division
                combined_unit = "/".join(u for u in units if u)
            result = ufunc(*magnitudes)
            return PhysicalQuantity(result, combined_unit)

        # If not handled, let NumPy fall back
        return NotImplemented

    def convert_to(self, unit: str)-> "PhysicalQuantity":
        """
        Convert the quantity to a new unit.
        :param unit: The new unit as a string
        """
        converted_quantity = self.quantity.to(unit)
        return PhysicalQuantity(
            converted_quantity.magnitude, str(converted_quantity.units)
        )

    def value_in(self, unit):
        """
        Convert the quantity to a new unit and return the value of the quantity.
        :param unit: The new unit as a string
        """
        converted_quantity = self.quantity.to(unit)
        return converted_quantity.magnitude


def _safe_convert(quantity: Optional[PhysicalQuantity], unit: str) -> Optional[PhysicalQuantity]:
    """
    convert quantities to a new unit
    return None if the quantity is None
    """
    return quantity.convert_to(unit) if quantity is not None else None


# Electron charge
e = PhysicalQuantity(-1.602176634e-19, "coulomb")

# mol to number by Avogadro's number
mol_to_num = PhysicalQuantity(6e23, "mol**(-1)")

# Constants

# Light speed  speed_of_light
# c = PhysicalQuantity(299792458, "m / s")
c = PhysicalQuantity(1, "c")

# Atomic mass unit
u = PhysicalQuantity(931.49410242, "MeV / c**2")

# Boltzmann constant in eV K^-1
kB = PhysicalQuantity(8.617333262145e-5, "eV / kelvin")
# Boltzmann constant in J K^-1
# kB = PhysicalQuantity(1.380649e-23, "joule / kelvin")

# Planck constant
h_Planck = PhysicalQuantity(4.135667696e-15, "eV * s")

# Reduced Planck constant
hbar = PhysicalQuantity(1, "hbar")

# Masses of electron, proton, and neutron
me = PhysicalQuantity(0.51099895000, "MeV / c**2")
mp = PhysicalQuantity(938.27208816, "MeV / c**2")
mn = PhysicalQuantity(939.56542052, "MeV / c**2")

# Bohr magneton
mu_B = PhysicalQuantity(5.7883818012e-5, "eV / tesla")

# Magnetic permeability of free space
mu_0 = PhysicalQuantity(1.25663706212e-6, "henry / m")


# Nuclear magneton
def mu_N(m):
    return (-1.0 * e * hbar) / (2 * m)  # * c **2


# Magnetic dipole moment of proton
gp = PhysicalQuantity(5.585694713, "")
I_p = PhysicalQuantity(1 / 2, "") * hbar
mu_p = gp * mu_N(mp) * I_p / hbar

# Gyromagnetic ratio of proton
gamma_p = PhysicalQuantity(2.6752218708e8, "hertz / tesla")

# Magnetic dipole moment of Xe nucleus
mu_Xe129 = PhysicalQuantity(-0.777969, "dimensionless") * mu_N(mp)

# Gyromagnetic ratio of Xe129
gamma_Xe129 = PhysicalQuantity(-7.441e7, "hertz / tesla")


if __name__ == "__main__":
    # solar_mass = PhysicalQuantity(1, 'solar_mass').convert_to('kg').value
    # parsec = PhysicalQuantity(1, 'parsec').convert_to('m').value  # 30856775814913673 m
    # # 30856775814913673
    # # 30856775814671916.000000
    # au = PhysicalQuantity(1, 'au').convert_to('m').value  # 149597870700
    # parsecFromAU = au * 648000. / (np.pi)
    # oneppm = PhysicalQuantity(1, 'ppm').convert_to('')
    # a = 1 + oneppm
    # # print(f'{parsec:22f}')
    # # print(f'{parsecFromAU:22f}')
    # print(f'{oneppm.value}')
    # cfreq = PhysicalQuantity(1.348570, "MHz")
    # print((cfreq / (gamma_p / (2 * pi))).convert_to("T"))
    # print(PhysicalQuantity(1.0, "ppb").convert_to(""))

    # Dimensionless examples
    x = PhysicalQuantity(1.0, "cm**3 m**(-3)")
    print("np.exp(x)   =", np.exp(x))
    print("np.tanh(x)  =", np.tanh(x))
    print("np.sin(x)   =", np.sin(x))
    print("np.cos(x)   =", np.cos(x))
    print("np.sinh(x)  =", np.sinh(x))
    print("np.cosh(x)  =", np.cosh(x))

    # Arithmetic examples with units
    a = PhysicalQuantity(3.0, "m")
    b = PhysicalQuantity(2.0, "m")
    c = PhysicalQuantity(5.0, "s")

    print("\nnp.add(a, b)      =", np.add(a, b))  # m
    print("np.subtract(a, b) =", np.subtract(a, b))  # m
    print("np.multiply(a, b) =", np.multiply(a, b))  # m*m
    print("np.divide(a, c)   =", np.divide(a, c))  # m/s

    print(mu_Xe129.convert_to("J/T").unit)
    induc = PhysicalQuantity(100, "henry")
    print(induc.convert_to("nanohenry"))
    print(induc.convert_to("nH"))

    res = PhysicalQuantity(100, "ohm")
    print(res.convert_to("kiloohm"))
    # print(induc.convert_to("nH"))


"""
all units

meter
m
metre
second
s
sec
ampere
A
amp
candela
cd
candle
gram
g
mole
mol
kelvin
K
degK
°K
degree_Kelvin
degreeK
radian
rad
bit
count
pi
π
tansec
ln10
wien_x
wien_u
eulers_number
speed_of_light
c
c_0
planck_constant
ℎ
elementary_charge
e
avogadro_number
boltzmann_constant
k
k_B
standard_gravity
g_0
g0
g_n
gravity
standard_atmosphere
atm
atmosphere
conventional_josephson_constant
K_J90
conventional_von_klitzing_constant
R_K90
zeta
ζ
dirac_constant
ħ
hbar
atomic_unit_of_action
a_u_action
avogadro_constant
N_A
molar_gas_constant
R
faraday_constant
conductance_quantum
G_0
magnetic_flux_quantum
Φ_0
Phi_0
josephson_constant
K_J
von_klitzing_constant
R_K
stefan_boltzmann_constant
σ
sigma
first_radiation_constant
c_1
second_radiation_constant
c_2
wien_wavelength_displacement_law_constant
wien_frequency_displacement_law_constant
newtonian_constant_of_gravitation
gravitational_constant
rydberg_constant
R_∞
R_inf
electron_g_factor
g_e
atomic_mass_constant
m_u
electron_mass
m_e
atomic_unit_of_mass
a_u_mass
proton_mass
m_p
neutron_mass
m_n
lattice_spacing_of_Si
d_220
K_alpha_Cu_d_220
K_alpha_Mo_d_220
K_alpha_W_d_220
fine_structure_constant
α
alpha
vacuum_permeability
µ_0
mu_0
mu0
magnetic_constant
vacuum_permittivity
ε_0
epsilon_0
eps_0
eps0
electric_constant
impedance_of_free_space
Z_0
characteristic_impedance_of_vacuum
coulomb_constant
k_C
classical_electron_radius
r_e
thomson_cross_section
σ_e
sigma_e
turn
revolution
cycle
circle
degree
deg
arcdeg
arcdegree
angular_degree
arcminute
arcmin
arc_minute
angular_minute
arcsecond
arcsec
arc_second
angular_second
milliarcsecond
mas
grade
grad
gon
mil
steradian
sr
square_degree
sq_deg
sqdeg
baud
Bd
bps
byte
B
octet
percent
%
permille
‰
ppm
angstrom
Å
ångström
Å
micron
µ
μ
fermi
fm
light_year
ly
lightyear
astronomical_unit
au
parsec
pc
nautical_mile
nmi
bohr
a_0
a0
bohr_radius
atomic_unit_of_length
a_u_length
x_unit_Cu
Xu_Cu
x_unit_Mo
Xu_Mo
angstrom_star
Å_star
planck_length
metric_ton
t
tonne
unified_atomic_mass_unit
u
amu
dalton
Da
grain
gr
gamma_mass
carat
ct
karat
planck_mass
minute
min
hour
h
hr
day
d
week
fortnight
year
a
yr
julian_year
month
century
centuries
millennium
millennia
eon
shake
svedberg
atomic_unit_of_time
a_u_time
gregorian_year
sidereal_year
tropical_year
common_year
leap_year
sidereal_day
sidereal_month
tropical_month
synodic_month
lunar_month
planck_time
degree_Celsius
°C
celsius
degC
degreeC
delta_degree_Celsius
Δ°C
Δcelsius
ΔdegC
ΔdegreeC
delta_celsius
delta_degC
delta_degreeC
degree_Rankine
°R
rankine
degR
degreeR
degree_Fahrenheit
°F
fahrenheit
degF
degreeF
delta_degree_Fahrenheit
Δ°F
Δfahrenheit
ΔdegF
ΔdegreeF
delta_fahrenheit
delta_degF
delta_degreeF
degree_Reaumur
°Re
reaumur
degRe
degreeRe
degree_Réaumur
réaumur
delta_degree_Reaumur
Δ°Re
Δreaumur
ΔdegRe
ΔdegreeRe
Δdegree_Réaumur
Δréaumur
delta_reaumur
delta_degRe
delta_degreeRe
delta_degree_Réaumur
delta_réaumur
atomic_unit_of_temperature
a_u_temp
planck_temperature
are
barn
b
darcy
hectare
ha
liter
l
L
ℓ
litre
cubic_centimeter
cc
lambda
λ
stere
hertz
Hz
revolutions_per_minute
rpm
revolutions_per_second
rps
counts_per_second
cps
reciprocal_centimeter
cm_1
kayser
knot
kt
knot_international
international_knot
mile_per_hour
mph
MPH
kilometer_per_hour
kph
KPH
kilometer_per_second
kps
meter_per_second
mps
foot_per_second
fps
sverdrup
sv
galileo
Gal
newton
N
dyne
dyn
force_kilogram
kgf
kilogram_force
pond
force_gram
gf
gram_force
force_metric_ton
tf
metric_ton_force
force_t
t_force
atomic_unit_of_force
a_u_force
joule
J
erg
watt_hour
Wh
watthour
electron_volt
eV
rydberg
Ry
hartree
E_h
Eh
hartree_energy
atomic_unit_of_energy
a_u_energy
calorie
cal
thermochemical_calorie
cal_th
international_calorie
cal_it
international_steam_table_calorie
fifteen_degree_calorie
cal_15
british_thermal_unit
Btu
BTU
Btu_iso
international_british_thermal_unit
Btu_it
thermochemical_british_thermal_unit
Btu_th
quadrillion_Btu
quad
therm
thm
EC_therm
US_therm
ton_TNT
tTNT
tonne_of_oil_equivalent
toe
atmosphere_liter
atm_l
watt
W
volt_ampere
VA
horsepower
hp
UK_horsepower
hydraulic_horsepower
boiler_horsepower
metric_horsepower
electrical_horsepower
refrigeration_ton
ton_of_refrigeration
cooling_tower_ton
standard_liter_per_minute
slpm
slm
conventional_watt_90
W_90
mercury
Hg
Hg_0C
Hg_32F
conventional_mercury
water
H2O
conventional_water
mercury_60F
Hg_60F
water_39F
water_4C
water_60F
pascal
Pa
barye
Ba
barie
barad
barrie
baryd
bar
technical_atmosphere
at
torr
pound_force_per_square_inch
psi
kip_per_square_inch
ksi
millimeter_Hg
mmHg
mm_Hg
millimeter_Hg_0C
centimeter_Hg
cmHg
cm_Hg
centimeter_Hg_0C
inch_Hg
inHg
in_Hg
inch_Hg_32F
inch_Hg_60F
inch_H2O_39F
inch_H2O_60F
foot_H2O
ftH2O
feet_H2O
centimeter_H2O
cmH2O
cm_H2O
sound_pressure_level
SPL
foot_pound
ft_lb
footpound
poise
P
reyn
stokes
St
rhe
particle
molec
molecule
molar
M
katal
kat
enzyme_unit
U
enzymeunit
clausius
Cl
entropy_unit
eu
becquerel
Bq
curie
Ci
rutherford
Rd
gray
Gy
sievert
Sv
rads
rem
roentgen
röntgen
peak_sun_hour
PSH
langley
Ly
nit
stilb
lambert
lumen
lm
lux
lx
atomic_unit_of_intensity
a_u_intensity
biot
Bi
abampere
abA
atomic_unit_of_current
a_u_current
mean_international_ampere
A_it
US_international_ampere
A_US
conventional_ampere_90
A_90
planck_current
coulomb
C
abcoulomb
abC
faraday
conventional_coulomb_90
C_90
ampere_hour
Ah
volt
V
abvolt
abV
mean_international_volt
V_it
US_international_volt
V_US
conventional_volt_90
V_90
atomic_unit_of_electric_field
a_u_electric_field
townsend
Td
ohm
Ω
abohm
abΩ
mean_international_ohm
Ω_it
ohm_it
US_international_ohm
Ω_US
ohm_US
conventional_ohm_90
Ω_90
ohm_90
siemens
S
mho
absiemens
abS
abmho
farad
F
abfarad
abF
conventional_farad_90
F_90
weber
Wb
unit_pole
henry
H
abhenry
abH
conventional_henry_90
H_90
tesla
T
gamma
γ
ampere_turn
At
biot_turn
gilbert
Gb
debye
D
buckingham
bohr_magneton
µ_B
mu_B
nuclear_magneton
µ_N
mu_N
refractive_index_unit
RIU
decibelwatt
dBW
decibelmilliwatt
dBm
decibelmicrowatt
dBu
decibel
dB
decade
octave
oct
neper
Np
thou
th
mil_length
inch
in
international_inch
inches
international_inches
hand
foot
ft
international_foot
feet
international_feet
yard
yd
international_yard
mile
mi
international_mile
circular_mil
cmil
square_inch
sq_in
square_inches
square_foot
sq_ft
square_feet
square_yard
sq_yd
square_mile
sq_mi
cubic_inch
cu_in
cubic_foot
cu_ft
cubic_feet
cubic_yard
cu_yd
link
li
survey_link
survey_foot
sft
fathom
rod
rd
pole
perch
chain
furlong
fur
cables_length
survey_mile
smi
us_statute_mile
league
square_rod
sq_rod
sq_pole
sq_perch
acre
square_survey_mile
section
square_league
acre_foot
acre_feet
dry_pint
dpi
US_dry_pint
dry_quart
dqt
US_dry_quart
dry_gallon
dgal
US_dry_gallon
peck
pk
bushel
bu
dry_barrel
US_dry_barrel
board_foot
FBM
board_feet
BF
BDFT
super_foot
superficial_foot
super_feet
superficial_feet
minim
fluid_dram
fldr
fluidram
US_fluid_dram
US_liquid_dram
fluid_ounce
floz
US_fluid_ounce
US_liquid_ounce
gill
gi
liquid_gill
US_liquid_gill
pint
pt
liquid_pint
US_pint
fifth
US_liquid_fifth
quart
qt
liquid_quart
US_liquid_quart
gallon
gal
liquid_gallon
US_liquid_gallon
teaspoon
tsp
tablespoon
tbsp
shot
jig
US_shot
cup
cp
liquid_cup
US_liquid_cup
barrel
bbl
oil_barrel
oil_bbl
beer_barrel
beer_bbl
hogshead
dram
dr
avoirdupois_dram
avdp_dram
drachm
ounce
oz
avoirdupois_ounce
avdp_ounce
pound
lb
avoirdupois_pound
avdp_pound
stone
quarter
bag
hundredweight
cwt
short_hundredweight
long_hundredweight
ton
short_ton
long_ton
slug
slinch
blob
slugette
force_ounce
ozf
ounce_force
force_pound
lbf
pound_force
force_ton
ton_force
force_short_ton
short_ton_force
force_long_ton
long_ton_force
kip
poundal
pdl
UK_hundredweight
UK_cwt
UK_ton
UK_force_ton
UK_ton_force
US_hundredweight
US_cwt
US_ton
US_force_ton
US_ton_force
pennyweight
dwt
troy_ounce
toz
ozt
troy_pound
tlb
lbt
scruple
apothecary_dram
ap_dr
apothecary_ounce
ap_oz
apothecary_pound
ap_lb
imperial_minim
imperial_fluid_scruple
imperial_fluid_drachm
imperial_fldr
imperial_fluid_dram
imperial_fluid_ounce
imperial_floz
UK_fluid_ounce
imperial_gill
imperial_gi
UK_gill
imperial_cup
imperial_cp
UK_cup
imperial_pint
imperial_pt
UK_pint
imperial_quart
imperial_qt
UK_quart
imperial_gallon
imperial_gal
UK_gallon
imperial_peck
imperial_pk
UK_pk
imperial_bushel
imperial_bu
UK_bushel
imperial_barrel
imperial_bbl
UK_bbl
pica
printers_pica
point
pp
printers_point
big_point
bp
didot
cicero
tex_point
tex_pica
tex_didot
tex_cicero
scaled_point
css_pixel
px
pixel
dot
pel
picture_element
pixels_per_centimeter
PPCM
pixels_per_inch
dots_per_inch
PPI
ppi
DPI
printers_dpi
bits_per_pixel
bpp
tex
Tt
dtex
denier
den
jute
Tj
aberdeen
Ta
RKM
number_english
Ne
NeC
ECC
number_meter
Nm
franklin
Fr
statcoulomb
statC
esu
statvolt
statV
statampere
statA
gauss
G
maxwell
Mx
oersted
Oe
ørsted
statohm
statΩ
statfarad
statF
statmho
statweber
statWb
stattesla
statT
stathenry
statH
kilogram
centimeter
milligram
decimeter
kilometer
decitex
micrometer
microliter
femtometer
microgram
micromole
millimeter
centipoise
"""
