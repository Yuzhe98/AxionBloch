# Theory

## Axion dark matter

The QCD axion was originally proposed to resolve the strong-{math}`CP` problem
in quantum chromodynamics.  More general axion-like particles (ALPs) arise
in various extensions of the Standard Model.  Axions and ALPs are
well-motivated dark-matter candidates; for brevity this document uses
"axion" for both.

In the Standard Halo Model (SHM) the local axion dark-matter field behaves
as a classical, coherently oscillating pseudoscalar:

```{math}
a(\mathbf{r}, t)
= a_0 \cos\!\bigl(2\pi\nu_a t - \mathbf{p}_a \cdot \mathbf{r}/\hbar + \phi\bigr),
```

where {math}`\phi` is a random phase, {math}`\mathbf{p}_a` is the axion
momentum, and the field amplitude is set by the local dark-matter energy density
{math}`\rho_\mathrm{DM}`:

```{math}
a_0 = \sqrt{\frac{2\hbar^3 \rho_\mathrm{DM}}{m_a^2 c}}.
```

The axion Compton frequency {math}`\nu_a = m_a c^2 / h` is the fundamental
frequency of oscillation.  The field is coherent on a timescale

```{math}
\tau_a \sim \frac{Q_a}{\pi\,\nu_a},
\qquad
Q_a = \left(\frac{c}{v_\mathrm{lab}}\right)^2 \approx 10^6,
```

where {math}`v_\mathrm{lab} \approx 233\,\mathrm{km\,s^{-1}}` is the speed of
the laboratory in the galactic rest frame.

### Laboratory-frame gradients

#### Frames and Lorentz factor

The axion field {math}`a` is a Lorentz scalar, so
{math}`a_\mathrm{lab}(x_\mathrm{lab})=a_\mathrm{halo}(x_\mathrm{halo})`.
Its time and spatial derivatives nevertheless mix under a boost.

Let the laboratory move with velocity {math}`\mathbf{v}_\mathrm{rel}` relative
to the halo rest frame. With this sign convention,

```{math}
\beta=\frac{|\mathbf{v}_\mathrm{rel}|}{c},
\qquad
\gamma=\frac{1}{\sqrt{1-\beta^2}}.
```

The implementation uses
`gamma = 1.0 / np.sqrt(1.0 - beta**2)` and rejects
{math}`\beta\geq1`. Reversing the definition of
{math}`\mathbf{v}_\mathrm{rel}` reverses the signs of the
velocity-dependent terms.

The derivative transformations are

```{math}
\partial_{t_\mathrm{lab}}a
=\gamma\left(
\partial_{t_\mathrm{halo}}a
+\mathbf{v}_\mathrm{rel}\cdot\nabla_\mathrm{halo}a
\right)
```

and

```{math}
\nabla_\mathrm{lab}a
=\nabla_\mathrm{halo}a
+(\gamma-1)\hat{\mathbf v}
 \left(\hat{\mathbf v}\cdot\nabla_\mathrm{halo}a\right)
+\gamma\frac{\mathbf v_\mathrm{rel}}{c^2}\,
 \partial_{t_\mathrm{halo}}a,
```

where
{math}`\hat{\mathbf v}=\mathbf{v}_\mathrm{rel}/|\mathbf{v}_\mathrm{rel}|`.
The term proportional to {math}`\gamma-1` changes the gradient component
parallel to the boost; perpendicular components are unchanged by this part.

#### Complex axion-field amplitude

`GravBoundAxionHalo` represents the real oscillating field using a complex
spatial amplitude:

```{math}
a(\mathbf{x},t)
\propto
\operatorname{Re}\!\left[
\Psi(\mathbf{x})e^{-i\omega_a t}
\right],
\qquad
\omega_a=2\pi\nu_a.
```

Consequently,

```{math}
\partial_t a\longleftrightarrow-i\omega_a\Psi,
\qquad
\nabla a\longleftrightarrow
\mathbf{G}_\mathrm{halo}:=\nabla\Psi.
```

The complex laboratory-frame gradient amplitude implemented by
`findGradientsAtDirection` is therefore

```{math}
\mathbf{G}_\mathrm{lab}
=\mathbf{G}_\mathrm{halo}
+(\gamma-1)\hat{\mathbf v}
 \left(\hat{\mathbf v}\cdot\mathbf{G}_\mathrm{halo}\right)
-i\gamma\frac{\omega_a}{c^2}\mathbf{v}_\mathrm{rel}\Psi.
```

The first term is the intrinsic halo-profile gradient. The last term is the
motion-induced gradient. For a real intrinsic gradient it is
{math}`90^\circ` out of phase, which explains the factor {math}`-i`.
For nonrelativistic laboratory speeds,
{math}`\gamma\simeq1+\beta^2/2`, and the expression reduces to

```{math}
\mathbf{G}_\mathrm{lab}
\simeq\nabla\Psi
-i\frac{\omega_a}{c^2}\mathbf{v}_\mathrm{rel}\Psi.
```

The implementation retains both {math}`\gamma` and the
{math}`\gamma-1` longitudinal correction. It uses the common Compton
frequency {math}`\omega_a=2\pi\nu_a`; small state-dependent binding-energy
corrections to this frequency are not currently included.

#### Implementation in spherical components

The solver first calculates

```{math}
G_r=\partial_r\Psi,\qquad
G_\theta=\frac{1}{r}\partial_\theta\Psi,\qquad
G_\phi=\frac{1}{r\sin\theta}\partial_\phi\Psi.
```

It then projects the Cartesian relative velocity
{math}`(v_x,v_y,v_z)` onto the local spherical basis:

```{math}
v_r
=v_x\sin\theta\cos\phi
+v_y\sin\theta\sin\phi
+v_z\cos\theta,
```

```{math}
v_\theta
=v_x\cos\theta\cos\phi
+v_y\cos\theta\sin\phi
-v_z\sin\theta,
```

```{math}
v_\phi=-v_x\sin\phi+v_y\cos\phi.
```

Defining

```{math}
D:=\mathbf{v}_\mathrm{rel}\cdot\mathbf{G}_\mathrm{halo}
=v_rG_r+v_\theta G_\theta+v_\phi G_\phi,
```

the code applies the boost component by component:

```{math}
G_{q,\mathrm{lab}}
=G_q
+(\gamma-1)\frac{v_qD}{|\mathbf{v}_\mathrm{rel}|^2}
-i\gamma\frac{\omega_a}{c^2}v_q\Psi,
\qquad q\in\{r,\theta,\phi\}.
```

When {math}`|\mathbf{v}_\mathrm{rel}|=0`, the longitudinal correction is set
to zero explicitly, avoiding division by zero.

#### Unit consistency

In a convention-independent form,

```{math}
[\nabla\Psi]=\frac{[\Psi]}{L}.
```

The boost term has the same units:

```{math}
\left[
i\gamma\frac{\omega_a}{c^2}\mathbf{v}_\mathrm{rel}\Psi
\right]
=
\frac{T^{-1}}{L^2T^{-2}}\,
\frac{L}{T}\,[\Psi]
=\frac{[\Psi]}{L},
```

because {math}`i`, {math}`\gamma`, and {math}`\beta` are dimensionless,
{math}`[\omega_a]=T^{-1}`, {math}`[c]=LT^{-1}`, and
{math}`[\mathbf{v}_\mathrm{rel}]=LT^{-1}`. Therefore,

```{math}
[\mathbf{G}_\mathrm{lab}]
=[\nabla\Psi]
=\left[
i\gamma\frac{\omega_a}{c^2}\mathbf{v}_\mathrm{rel}\Psi
\right]
=\frac{[\Psi]}{L}.
```

The normalized three-dimensional wavefunctions used by
`GravBoundAxionHalo` satisfy {math}`[\Psi]=L^{-3/2}`. Thus,

```{math}
[\mathbf{G}_\mathrm{lab}]
=[\nabla\Psi]
=\left[
i\gamma\frac{\omega_a}{c^2}\mathbf{v}_\mathrm{rel}\Psi
\right]
=L^{-5/2}=\mathrm{m}^{-5/2}.
```

#### Interpreting the complex result

The returned gradient components are complex phasors, not instantaneous real
fields. At a chosen carrier phase, the physical gradient is proportional to

```{math}
\nabla a(\mathbf{x},t)
\propto
\operatorname{Re}\!\left[
\mathbf{G}_\mathrm{lab}(\mathbf{x})e^{-i\omega_a t}
\right].
```

Thus, `gradient.real` contains only the in-phase quadrature and can omit a
purely boost-induced contribution. The oscillation amplitude is
{math}`|\mathbf{G}_\mathrm{lab}|`; accordingly, the
`findOmega_aOverTime` helper uses the absolute value of each complex gradient
component. Multiplication by the halo field-normalization factor converts the
normalized-wavefunction gradient into the physical axion-field gradient.

#### Relative velocity used by the code

The velocity must use the same solar-Z Cartesian frame as the wavefunction
grid:

- {math}`\hat{\mathbf z}` points from Earth toward the Sun.
- {math}`\hat{\mathbf x}` follows Earth's heliocentric velocity after its
  component parallel to {math}`\hat{\mathbf z}` is removed.
- {math}`\hat{\mathbf y}=\hat{\mathbf z}\times\hat{\mathbf x}`.

If `relative_velocity=None`, Astropy calculates the station velocity caused
by Earth's rotation in GCRS, and `Station` projects it into the solar-Z frame.
This default assumes a nonrotating geocentric Earth halo. The speed is about
{math}`0.30\,\mathrm{km\,s^{-1}}` at Mainz, for which
{math}`\beta\sim10^{-6}` and {math}`\gamma-1\sim5\times10^{-13}`.

Earth's approximately {math}`240\,\mathrm{km\,s^{-1}}` Galactic bulk velocity
is not used for an Earth-bound halo because Earth and its bound halo share
that motion. A different model can supply an explicit three-component
`relative_velocity`. A zero vector describes a perfectly corotating halo,
while `include_lorentz_boost=False` returns only the intrinsic gradient.

## Axion–spin interaction

Axion field gradients couple to fermionic spins through a pseudoscalar
interaction.  The Hamiltonian is

```{math}
\mathcal{H}_\mathrm{int} = -\,g_{aNN}\,\nabla a \cdot \mathbf{S},
```

where {math}`g_{aNN}` is the axion–nucleon coupling strength (GeV{math}`^{-1}`)
and {math}`\mathbf{S}` is the spin operator.  This has the same form as the
Zeeman interaction

```{math}
\mathcal{H} = -\hbar\gamma\,\mathbf{B} \cdot \mathbf{S},
```

so the axion field gradient acts as an effective *pseudomagnetic field*

```{math}
\mathbf{B}_\mathrm{axion}
= \frac{g_{aNN}}{\gamma}\,\nabla a(\mathbf{r}, t).
```

This field drives resonant transitions when {math}`\nu_a` equals the nuclear
Larmor frequency {math}`\nu_L = \gamma B_0 / (2\pi)`.

The rms Rabi frequency in the perpendicular-gradient coupling geometry
(``grad_perp``) is

```{math}
\Omega_\mathrm{rms}
= \frac{1}{2}\,g_{aNN}\sqrt{2\hbar c\,\rho_\mathrm{DM}}\;v_\mathrm{lab}.
```

## Bloch equations

The nuclear spin magnetization {math}`\mathbf{M}` evolves under the total
effective field {math}`\mathbf{B}_\mathrm{eff}` (bias field plus pseudomagnetic
field) according to

```{math}
\frac{dM_x}{dt} = \gamma\left(\mathbf{M}\times\mathbf{B}_\mathrm{eff}\right)_x
                  - \frac{M_x}{T_2},
```

```{math}
\frac{dM_y}{dt} = \gamma\left(\mathbf{M}\times\mathbf{B}_\mathrm{eff}\right)_y
                  - \frac{M_y}{T_2},
```

```{math}
\frac{dM_z}{dt} = \gamma\left(\mathbf{M}\times\mathbf{B}_\mathrm{eff}\right)_z
                  - \frac{M_z - M_0}{T_1},
```

where {math}`T_1` is the longitudinal (spin–lattice) relaxation time,
{math}`T_2` is the intrinsic transverse relaxation time, and {math}`M_0`
is the equilibrium magnetization.

## Rotating coordinate frame

The bias field {math}`B_0` is typically much stronger than the axion
pseudomagnetic field.  To remove the fast Larmor precession at
{math}`\nu_L = \gamma B_0 / (2\pi)` and enable efficient numerical integration
of the slower axion-induced dynamics, axionbloch transforms to the
**rotating coordinate frame** (RCF) that co-rotates with {math}`B_0`.

## Field inhomogeneity and {math}`T_2^*`

In a real magnet the bias field is not perfectly uniform.  The ensemble of
nuclear spins in a sample experiences a spread of Larmor frequencies,
causing additional dephasing characterised by {math}`T_2^* < T_2`.
axionbloch models this by sampling {math}`N_\mathrm{pt}` discrete values from
a Lorentzian field distribution with a user-specified fractional FWHM (e.g.
2 ppm), solving the Bloch equations independently for each spin packet, and
averaging the resulting signals.

## Numerical integration (RK4)

The Bloch equations are integrated with the fourth-order Runge–Kutta (RK4)
method.  Given a time step {math}`\Delta t` and current magnetization
{math}`\mathbf{M}^n`:

```{math}
\mathbf{M}^{n+1}
= \mathbf{M}^n
  + \frac{\Delta t}{6}
    \left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right),
```

where

```{math}
\mathbf{k}_1 = \left.\frac{d\mathbf{M}}{dt}\right|_{\mathbf{M}^n},
\qquad
\mathbf{k}_2 = \left.\frac{d\mathbf{M}}{dt}\right|_{\mathbf{M}^n+\frac{1}{2}\mathbf{k}_1\Delta t},
```

```{math}
\mathbf{k}_3 = \left.\frac{d\mathbf{M}}{dt}\right|_{\mathbf{M}^n+\frac{1}{2}\mathbf{k}_2\Delta t},
\qquad
\mathbf{k}_4 = \left.\frac{d\mathbf{M}}{dt}\right|_{\mathbf{M}^n+\mathbf{k}_3\Delta t}.
```

The local truncation error is {math}`\mathcal{O}(\Delta t^5)` and the
accumulated error over {math}`N = 1/\Delta t` steps is
{math}`\mathcal{O}(\Delta t^4)`.  The time step is chosen to be at least one
order of magnitude smaller than all characteristic timescales of the system
(relaxation times, axion period).  The RK4 kernel is implemented in C++ and
exposed to Python via **pybind11** for computational efficiency.

## Axion power spectral density lineshapes

The analytical PSD of the SHM axion field (Gramolin *et al.*) is supported
for three coupling geometries:

| Case | Description |
|------|-------------|
| ``non-grad`` | Non-gradient coupling |
| ``grad_par`` | Gradient coupling, sensitive axis ∥ {math}`\mathbf{v}_\mathrm{lab}` |
| ``grad_perp`` | Gradient coupling, sensitive axis ⊥ {math}`\mathbf{v}_\mathrm{lab}` |

All lineshapes are one-sided (zero below {math}`\nu_a`) and normalized so
that {math}`\int S(\nu)\,d\nu = 1`.

## Gravitationally bound axion halo

When axions are gravitationally bound to a compact body (e.g. Earth), they
occupy discrete quantum states described by the time-independent Schrödinger
equation (TISE):

```{math}
\hat{H}\,\psi = E\,\psi,
\qquad
\hat{H} = -\frac{\hbar^2}{2m_a}\nabla^2 + m_a\,\Phi(\mathbf{r}),
```

where {math}`\Phi(\mathbf{r})` is the gravitational potential.  Separating
radial and angular parts via {math}`\psi_{nlm} = R_{nl}(r)\,Y_l^m(\theta,\phi)`,
the radial equation for {math}`u(r) = rR_{nl}(r)` becomes

```{math}
-\frac{\hbar^2}{2m_a}u'' + V_\mathrm{eff}(r)\,u = E\,u,
\qquad
V_\mathrm{eff}(r) = m_a\,\Phi(r) + \frac{\hbar^2 l(l+1)}{2m_a r^2}.
```

The package discretises the Hamiltonian on a uniform 1-D radial grid with
{math}`N` points spanning {math}`\pm L/2` using a three-point
finite-difference stencil and diagonalizes the resulting dense matrix with
`scipy.linalg.eigh`.  Eigenstates are labelled by the spectroscopic convention
{math}`n = n_r + l + 1` (1s, 2s, 2p, 3s, …).

### Earth gravitational potential

`EarthBoundAxionHalo` pre-configures the solver with Earth's gravitational
potential from the Preliminary Earth Model (PEM) density profile.  The
interior potential is obtained by integrating the spherical-shell mass
distribution; outside the surface it reduces to the point-mass approximation
{math}`\Phi(r) = -GM_\oplus/r`.  The reference point is the Earth's centre
({math}`\Phi = 0` at {math}`r = 0`).

### Axion field gradient at a ground station

For a superposition of eigenstates the total wavefunction is

```{math}
\Psi(\mathbf{r})
= \sum_{n_r, l} c_{n_r l}\,R_{n_r l}(r)\,Y_l^0(\theta,\phi),
```

and its spherical-coordinate gradient is

```{math}
\nabla\Psi
= \partial_r\Psi\;\hat{r}
  + \frac{1}{r}\partial_\theta\Psi\;\hat{\theta}
  + \frac{1}{r\sin\theta}\partial_\phi\Psi\;\hat{\phi}.
```

The package evaluates each component on a 3-D {math}`(r,\theta,\phi)` mesh,
then interpolates along the radial line pointing toward the experimental
station to obtain {math}`\nabla\Psi(r_\mathrm{station})`.
