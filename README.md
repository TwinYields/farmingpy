# FarmingPy

Python library for reading ISOBUS task files and processing data.

### Installation

Dotnet installation is required to run the code https://dotnet.microsoft.com/en-us/download/dotnet/6.0. A C# library is used to read in timelog data (trough PythonNet).

```
pip install git+https://github.com/TwinYields/farmingpy.git
```

### Usage:

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

