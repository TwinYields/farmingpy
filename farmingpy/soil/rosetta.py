#%%
from rosetta import rosetta, SoilData
import pandas as pd
import geopandas as gpd
import numpy as np
from functools import singledispatch
import os
try:
    import matplotlib.pyplot as plt
except:
    pass


# %%

class Rosetta(object):
    """
    Interface to USDA Rosetta pedotransfer function model using https://pypi.org/project/rosetta-soil/.
    Used to estimate soil hydraulic properties.
    """

    rosetta_avg = None

    def __init__(self, soildata):
        """
        soildata: DataFrame with columns "sand", "silt", "clay"
        """
        if type(soildata) in [dict, pd.core.series.Series]:
            soildata = pd.DataFrame([soildata])
        elif type(soildata) is str:
            soildata = [soildata]
        elif type(soildata) is list and type(soildata[0]) is str:
            rparams = [self.find_params(soil) for soil in soildata]
            self.rosettaparams = pd.concat(rparams).reset_index(drop=True)
            self.soildata = pd.DataFrame(dict(name = soildata))
            return None

        self.soildata = soildata
        self.rosettaparams = self.vg_params()

    @classmethod
    def average_soils(cls):
        """Return average soil data for USDA Rosetta model.
        Source: https://www.ars.usda.gov/pacific-west-area/riverside-ca/agricultural-water-efficiency-and-salinity-research-unit/docs/model/rosetta-class-average-hydraulic-parameters/
        """
        if cls.rosetta_avg is None:
            cls.rosetta_avg = pd.read_csv(os.path.dirname(__file__) + "/../data/rosetta_average_parameters.csv")
        return cls.rosetta_avg

    @classmethod
    def find_params(cls, soilname):
        cls.average_soils()
        rparams = cls.rosetta_avg[cls.rosetta_avg.texture_name.str.replace(" ", "") == soilname.lower().replace(" ", "")]
        return rparams.reset_index(drop=True)

    def vg_params(self):
        """
        Get soil hydrological parameters using USDA Rosetta model.

        Uses: https://github.com/usda-ars-ussl/rosetta-soil
        """
        sdata = self.soildata.rename(str.lower, axis=1)
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

    def water_retention(self, phi, rosettaparams = None):
        """
        Function that computes volumetric water content from  soil
        matric potential using the van Genuchten (1980) model.
        Used with Rosetta model predictions.

        phi: Matric potential kPa
        """
        if rosettaparams is None:
            rosettaparams = self.rosettaparams
        # Rosetta units for alpha and npar are 1/cm and [-]
        # convert kPa to cm of H20
        phi = np.array(phi) * 10.19716
        alpha = 10**rosettaparams["log10_alpha"]
        n = 10**rosettaparams["log10_n"]
        theta_r = rosettaparams["theta_r"]
        theta_s = rosettaparams["theta_s"]

        theta = theta_r + (theta_s-theta_r)*(1+(alpha*phi)**n)**-(1-1/n)
        return theta

    def water_capacity(self):
        """
        Get soil saturation capacity, field capacity at 10 and 33 kPA,
        wilting point (1500kPA) and available water capacity.
        """
        #    rparams = self.find_params(soil)
        #    soildata = pd.DataFrame(dict(texture = [soil]))
        #else:
        #    soildata = soil
        #    rparams = self.rosettaparams
        phi_vec = [("sat", 0.1), ("fc_10", 10), ("fc_33", 33), ("wp", 1500)]
        data = pd.DataFrame()
        for pname, phi in phi_vec:
            data[pname] = self.water_retention(phi)
        data["awc_10"] = data["fc_10"] - data["wp"]
        data["awc_33"] = data["fc_33"] - data["wp"]
        data["ksat"] = 10**self.rosettaparams["log10_ksat"]
        #return pd.concat([soildata, data], axis=1)
        return data

    def plot_water_retention(self, fc = 10, wp=1500, sat=0.1, legend=True):
        """Plot water retention curve"""

        wpts = np.array([sat, fc, wp]) # SAT, FC, wp in kPAs

        phis = np.logspace(-3.5,6,2000)
        curves = [self.water_retention(phis, p) for i,p in self.rosettaparams.iterrows()]
        pts = [self.water_retention(wpts, p) for i,p in self.rosettaparams.iterrows()]
        N = len(curves)
        if "name" in self.soildata.columns:
            labels = self.soildata["name"].to_list()
        else:
            labels = [f"{i}" for i in range(N)]

        for i in range(N):
            _ = plt.semilogy(curves[i].T, phis, label = labels[i])
            _ = plt.plot(pts[i].T, wpts, linestyle="None", marker="o")
        plt.xlabel("Volumetric Water Content")
        plt.ylabel("Suction (kPA)")
        if legend:
            plt.legend(loc=1)

# %%
