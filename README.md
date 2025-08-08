# farmingpy: a Python package for developing Digital Twins of precision farming systems

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://twinyields.github.io)

[farmingpy](https://github.com/TwinYields/farmingpy) is a Python library developed to support the development of digital twins for supporting precision farming. It is developed in the Digital Technologies in Agriculture group at at [Natural Resources Institute Finland (Luke)](https://luke.fi/en).

farmingpy provides the following functionality:
- Interfacing APSIM simulation model for setting up high resolution spatial simulations, running model ensembles for assimilating EO and sensor data with the APSIM simulation models, reading the simulation outputs and optimizing model parameters.
- Reading planned and implemented ISOBUS task data.
- Unified interface to USDA Rosetta and EUPTF2 pedotransfer function models to estimate soil water holding capacity based on soil data available from farms.

## Publication

> Hartikainen, A., Backman, J., & Pastell, M. (2025). Farmingpy: Python package for developing digital twins and precision farming data processing. In Precision agriculture’25 (pp. 1287–1293). Wageningen Academic. https://brill.com/edcollchap-oa/book/9789004725232/BP000168.xml

## Installation

The library can be installed with pip. Dotnet installation is required to interface APSIM and for reading ISOBUS data: https://dotnet.microsoft.com/en-us/download/dotnet/. See the [documentation](https://twinyields.github.io/install.html) for instructions on how install [APSIM]() to be used with farmingpy.

```
pip install git+https://github.com/TwinYields/farmingpy.git
```
