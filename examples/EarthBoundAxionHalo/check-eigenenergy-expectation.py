"""Check eigenvalue and Hamiltonian-expectation consistency.

For an eigenstate of the discretized Hamiltonian,

    eigenE == <T> + <V_eff>,

up to numerical errors from the finite grid, half-domain integration, and
finite-difference evaluation of the second derivative.
"""

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


states_to_check = ["1s", "2s", "2p"]
relative_tolerance = 5e-4

halo = EarthBoundAxionHalo(
    nu_a=1.348585 * unit.MHz,
    N=2**12,
    extent=128 * unit.R_earth,
    verbose=True,
)
halo.solve_TISE_3D(
    l_vals=[0, 1],
    max_n_r=4,
    verbose=False,
)

header = (
    f"{'state':<7}"
    f"{'eigenE (aeV)':>18}"
    f"{'<T+V_eff> (aeV)':>22}"
    f"{'difference (aeV)':>20}"
    f"{'relative':>14}"
    f"{'consistent':>14}"
)
print(header)
print("-" * len(header))

failed_states = []
for state_name in states_to_check:
    state = halo.states[state_name]
    eigenE = state["eigenE"].to(unit.attoelectronvolt)
    eigenE_expect = state["eigenE_expect"].to(unit.attoelectronvolt)
    difference = eigenE_expect - eigenE
    relative_difference = np.abs(difference / eigenE).to_value(unit.one)
    consistent = relative_difference <= relative_tolerance

    print(
        f"{state_name:<7}"
        f"{eigenE.value:18.9e}"
        f"{eigenE_expect.value:22.9e}"
        f"{difference.value:20.9e}"
        f"{relative_difference:14.3e}"
        f"{str(consistent):>14}"
    )

    if not consistent:
        failed_states.append(state_name)

if failed_states:
    raise AssertionError(
        "Energy expectation mismatch above "
        f"rtol={relative_tolerance:g} for states: {failed_states}"
    )

print(
    "\nAll selected states satisfy "
    f"|<H> - eigenE| / |eigenE| <= {relative_tolerance:g}."
)
