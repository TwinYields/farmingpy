import re
import pandas as pd

def read_met(weatherfile):
    """Read APSIM .met file to dataframe

    Args:
        weatherfile (str): Filename

    Returns:
        DataFrame: Weather data
    """
    # Header can be of varying length
    # find header line
    with open(weatherfile) as wf:
        lines = wf.readlines()
        row = 1
        for line in lines:
            if line.strip().startswith("year"):
                break
            row += 1

    hdr = re.split("\s+", line.strip())
    wdf = pd.read_fwf(weatherfile, skiprows=row+1,
                names=hdr)
    return wdf