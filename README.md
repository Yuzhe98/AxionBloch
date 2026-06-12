# AxionBloch
A python package for simulating **axion**-induced spin dynamics based on **Bloch equations**, implemented in **Python**  and **C++**. 

Yuzhe Zhang - Uni Mainz - yuhzhang@uni-mainz.de

📖 Documentation: [axionbloch.readthedocs.io](https://axionbloch.readthedocs.io/en/latest/)

## Features

In AxionBloch, the axion (including axionlike-particle or ALP) field acts like a **pseudomagnetic field** coupling to (nuclear) spins.

The package provides a **numerical platform** to study these subtle effects using Bloch-equation-based simulations efficiently.

## Requirements

- Python >= 3.10

## Installation

Development versions are published on [TestPyPI](https://test.pypi.org/project/axionbloch/). Since TestPyPI does not host the dependencies, point pip at both indexes:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ axionbloch
```

The source code is available on [GitHub](https://github.com/Yuzhe98/AxionBloch). To build and install from source (requires a C++17 compiler for the simulation extension):
```bash
git clone https://github.com/Yuzhe98/AxionBloch.git
cd AxionBloch
pip install .
```

## Quick Start
Examples can be found in the examples/.

## License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)