"""axionbloch — Bloch-equation simulation toolkit for axion dark-matter NMR searches.

Modules
-------
constants
    Physical constants and atomic-unit conversion factors.
Sample
    NMR sample model (:class:`~axionbloch.Sample.Sample`).
Apparatus
    Static magnet model (:class:`~axionbloch.Apparatus.Magnet`).
Station
    Geographic lab-location model (:class:`~axionbloch.Station.Station`).
MilkyWayAxionHalo
    Milky Way SHM axion-wind model.
GravBoundAxionHalo
    Gravitationally-bound axion halo solver (TISE).
EarthBoundAxionHalo
    Earth-specific subclass of :class:`~axionbloch.GravBoundAxionHalo.GravBoundAxionHalo`.
FineGrainedAxionStream
    Single fine-grained axion-stream model.
SimuTypes
    Shared type definitions (:class:`~axionbloch.SimuTypes.SimuParams`,
    :class:`~axionbloch.SimuTypes.SimuEntry`).
SimuTools
    Bloch-equation simulation engine and magnetic-field builders.
DataAnalysis
    NMR signal processing (:class:`~axionbloch.DataAnalysis.Signal`).
utils
    General-purpose helpers (curve fitting, debugging, base classes).
"""

try:
    from .blochsimulation import *
except ImportError:
    pass
