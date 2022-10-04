
# FarmingPy

Python library for reading ISOBUS task files and processing data.

Usage:

```python
import farmingpy as fp
tl = fp.TimeLogData("TASKDATA_20210603_0159/TASKDATA.XML")
tl.headers # List measurement info 
tl.data # GeoDataFrame containing all logged data
```