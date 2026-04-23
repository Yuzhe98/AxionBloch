from axionbloch.constants import *
from axionbloch.utils import check
from astropy import units as u

def _print_constants():
    check(g_p)
    check(I_p)
    check(mu_p)
    check(gamma_p)
    check(mu_Xe129)
    check(gamma_Xe129)
    check(earth_radius)
    a=u.Quantity(3.1, "g / cm**3 ")
    check(a.si)
    check(a.to("kg/m**3"))


if __name__ == "__main__":
    _print_constants()
