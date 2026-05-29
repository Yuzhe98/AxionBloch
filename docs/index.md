# axionbloch

**Bloch-equation-based simulations for axion-induced spin dynamics.**

The interaction of ultralight bosonic dark matter with nuclear spins can be
interpreted as a pseudomagnetic field acting on ordinary matter.
**axionbloch** models such interactions — together with the usual magnetic
interactions — using the spin-evolution (Bloch) equations, providing a
numerical tool for deriving axion signal signatures that are crucial for
designing experimental searches and data analysis.

The package interface is implemented in Python for simplicity and
readability; the underlying RK4 numerical integration is implemented in C++
(exposed via **pybind11**) for computational efficiency.  All inputs and
outputs use SI units via **astropy**, making results directly comparable to
experimental parameters and observables.

axionbloch supports the following axion field configurations:

- **Milky Way axion halo** — stochastic axion wind from the Standard Halo
  Model, with analytical lineshapes and Monte-Carlo amplitude spectra.
- **Earth-bound axion halo** — axions gravitationally trapped by the Earth,
  solved as bound states of the radial Schrödinger equation (TISE).
- **Fine-grained axion streams** — single coherent streams with well-defined
  velocity and phase.

The package is also useful for general NMR simulations: pulse sequences,
free decay, spin echoes, and CW excitation can all be configured and
visualised, making axionbloch suitable for educational purposes as well.

The package is documented here and available at
[github.com/Yuzhe98/AxionBloch](https://github.com/Yuzhe98/AxionBloch).

```{toctree}
:maxdepth: 16
:caption: Contents

installation
theory
usage
api
```
