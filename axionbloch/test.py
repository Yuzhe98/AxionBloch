from astropy.constants import codata2018 as const
import astropy.units as u

value = 1.0 * u.K * u.rad
print(value.to(u.K))
