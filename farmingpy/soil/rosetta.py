#%%
from rosetta import rosetta, SoilData
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from functools import singledispatch
import os

# %%

def rosetta_params(soildata):
    """
    Get soil hydrological parameters using USDA Rosetta model.

    soildata: DataFrame with columns "sand", "silt", "clay"

    Uses: https://github.com/usda-ars-ussl/rosetta-soil
    """
    sdata = soildata.rename(str.lower, axis=1)
    sarray = sdata[["sand", "silt", "clay"]].to_numpy()
    if sarray.max() <= 1.0:
        sarray *= 100.0
    r_soildata = SoilData.from_array(sarray)
    mean, stdev, codes = rosetta(3, r_soildata)
    p_names = ["theta_r", "theta_s", "log10_alpha", "log10_n", "log10_ksat" ]
    mu_params =  pd.DataFrame(mean,
                       columns=p_names)
    sd_params = pd.DataFrame(stdev,
                       columns= [f"{p}_std" for p in p_names])
    rcode = pd.DataFrame(codes, columns=["rosetta_model_code"])
    return pd.concat([mu_params, sd_params, rcode], axis=1)


def rosetta_water_retention(phi, rosettaparams):
    """
    Function that computes volumetric water content from  soil
    matric potential using the van Genuchten (1980) model.
    Used with Rosetta model predictions.

    phi: Matric potential kPa
    rosetta_params: DataFrame from `rosetta_params`
    """
    # Rosetta units for alpha and npar are 1/cm and [-]
    # convert kPa to cm of H20
    phi = np.array(phi) * 10.19716
    alpha = 10**rosettaparams["log10_alpha"]
    n = 10**rosettaparams["log10_n"]
    theta_r = rosettaparams["theta_r"]
    theta_s = rosettaparams["theta_s"]

    theta = theta_r + (theta_s-theta_r)*(1+(alpha*phi)**n)**-(1-1/n)
    return theta

#%%
class RosettaData:
    rosetta_avg = None

    @classmethod
    def rosetta_averages(cls):
        """Return average soil data for USDA Rosetta model.
        Source: https://www.ars.usda.gov/pacific-west-area/riverside-ca/agricultural-water-efficiency-and-salinity-research-unit/docs/model/rosetta-class-average-hydraulic-parameters/
        """
        if cls.rosetta_avg is None:
            cls.rosetta_avg = pd.read_csv(os.path.dirname(__file__) + "/../data/rosetta_average_parameters.csv")
        return cls.rosetta_avg

    @classmethod
    def find_params(cls, soilname):
        cls.rosetta_averages()
        rparams = cls.rosetta_avg[cls.rosetta_avg.texture_name.str.replace(" ", "") == soilname.lower().replace(" ", "")]
        return rparams.reset_index()

#%%
def rosetta_soil_properties(soil, fc = 33, pwp=1500, sat=0.1):
    """
    soildata: Soil full name as string or DataFrame
    with columns "sand", "silt", "clay"
    """
    if type(soil) == str:
        rparams = RosettaData.find_params(soil)
        soildata = pd.DataFrame(dict(texture = [soil]))
    else:
        soildata = soil
        rparams = rosetta_params(soildata)
    phi_vec = [("sat", sat), ("fc", fc), ("pwp", pwp)]
    data = pd.DataFrame()
    for pname, phi in phi_vec:
        data[pname] = rosetta_water_retention(phi, rparams)
    data["awc"] = data["fc"] - data["pwp"]
    return pd.concat([soildata, data], axis=1)

#%%
def plot_soil_water(rosettaparams, fc = 33, pwp=1500, sat=0.1):
    wpts = np.array([sat, fc, pwp]) # SAT, FC, PWP in kPAs
    phis = np.logspace(-3.5,6,2000)
    curve = rosetta_water_retention(phis, rosettaparams)
    pts = rosetta_water_retention(wpts, rosettaparams)
    _ = plt.semilogy(curve, phis)
    _ = plt.plot(pts, wpts, linestyle="None", marker="o")
    #plt.hlines(wpts.T, xmin=0, xmax=np.max(curve), color="k")
    plt.xlabel("Volumetric Water Content")
    plt.ylabel("Suction (kPA)")
