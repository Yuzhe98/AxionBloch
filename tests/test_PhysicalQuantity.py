# TODO: add more tests for PhysicalQuantity, especially for unit conversion and arithmetic operations

from axionbloch.enphylope import PhysicalQuantity


def test_SI2AtomicUnit():
    hbar_AU = 1.0
    m_e_AU = 1.0
    a_0_AU = 1.0
    e_AU = 1.0
    hartree_AU = 1.0
    Eh_AU = hartree_AU

    nm_AU = 1.8897261246257702e1
    Å_AU = 1.8897261246257702
    eV_AU = 0.03674932217565499
    ps_AU = 4.134137333518212e4
    picoseconds_AU = 4.134137333518212e4

    fs_AU = 4.134137333518212 * 10.0
    femtoseconds_AU = 4.134137333518212 * 10.0

    V_AU = 0.03674932217565499
    V_m_AU = 1.9446903811488876e-12
    T_AU = 4.254382157326325e-06
    m_AU = 1.8897261246257702e10
    C_AU = 6.241509074460763e18
    s_AU = 4.134137333518173e16
    Hz_AU = 2.4188843265857225e-17
    kg_AU = 1.0977691057577634e30
    J_AU = 2.293712278396328e17
    A_AU = 150.974884744557

    # some physical constants, expressed in atomic units

    k_AU = 0.5  # hbar**2 / (2*m_e)
    m_p_AU = 1836.1526734400013
    𝜇0_AU = 0.0006691762566207213
    ε0_AU = 0.0795774715459477
    c_AU = 137.035999083818
    α_AU = 0.0072973525693

    # earth profiles
    earth_radius_au_AU = 6371.0e3 * m_AU  #

    km_AU = 1.8897261246257702e13

    # Oneh = PhysicalQuantity(1, "planck_constant")
    # Oneh = Oneh.to_base_units()
    # print(Oneh.value_in("J*s"))
    # print(Oneh.to("J*s"))
    # q = PhysicalQuantity(1, "kg* m* s  *A*  K  *mol*  cd")
    # print(q.to_base_units())
    q = PhysicalQuantity(1, "eV")
    q_AU = q.to_atomic_units()
    print(q_AU)
    q = PhysicalQuantity(1, "kg m* s  *A")
    q_AU = q.to_atomic_units()
    print(q_AU)


if __name__ == "__main__":
    test_SI2AtomicUnit()
