# FarmingPy

Python library for Smart Farming data and modeling. Enables reading of ISOBUS task files, EO data from SentinelHub and interfacing APSIM simulation models.

## Installation

Dotnet installation is required to run the code https://dotnet.microsoft.com/en-us/download/dotnet/6.0. A C# library is used to read in timelog data (trough PythonNet).

```
pip install git+https://github.com/TwinYields/farmingpy.git
```

## APSIM interface

In order to use APSIM interface from `farmingpy.apsim` you need to install APSIM and add the directory containing the Models executable to path (to find the right .dll files). On Windows you can install the APSIM using the installer or from source.

On Linux you can use the following to build APSIM:

```bash
git clone --depth 1 https://github.com/APSIMInitiative/ApsimX.git
dotnet build -o ~/.local/lib/apsimx -c Release ApsimX/Models/Models.csproj
```

And add the build location to Pythonpath:

```bash
export PYTHONPATH=~/.local/lib/apsimx
```


## Usage:

```python
import farmingpy as fp
tl = fp.TimeLogData("TASKDATA_20210603_0159/TASKDATA.XML")
# List measurement info
tl.headers
# DataFrame containing all logged data
tl.data()
# GeoDataFrame containing only "Actual Mass Per Area Application Rate" (DD entity 7)
# and "Actual Work State" columns
tl.rates()
```



