#%%
import pandas as pd
import geopandas as gpd
import numpy as np
import io
from contextlib import redirect_stdout
# Requires
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    euptf =  importr("euptf2")
except:
    euptf = None
try:
    import matplotlib.pyplot as plt
except:
    pass


#%%
# Van genuchten and soil hydraulic properties
#sdata_vg =  euptf.euptfFun(ptf = "PTF01", predictor = sdata_r,
#                           target = "MVG", query = "predictions")
#r_to_pd(sdata_vg).head()

def pd_to_r(df):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(df)

def r_to_pd(rdf):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(rdf)

class EUPTF2(object):
    """
    Interface to eutptfv2 pedotransfer functions using https://github.com/tkdweber/euptf2
    via rpy2. These need to be installed by the user.

    Used to estimate soil hydraulic properties.
    """

    def __init__(self, soildata):
        """
        soildata: DataFrame with columns "sand", "silt", "clay".
        Optionally also "om" (organic matter) and "depth_m" (sampling depth) can be given.
        """

        if euptf is None:
            #print("Using euptft2 requires rpy2 and euptf2 package installed in R https://github.com/tkdweber/euptf2")
            raise UserWarning("""

euptf2 R-package not found
--------------------------

Using EUPTF2 class requires rpy2 in python https://rpy2.github.io/
and euptf2 package installed in R https://github.com/tkdweber/euptf2
""")
        self.soildata = soildata
        self.soildata_ptf = self._eupt_fy(self.soildata)
        self.soildata_r = pd_to_r(self.soildata_ptf)
        self.funs = self.which_ptf()

    def which_ptf(self):
        """
        Find the right prediction model based on input data.
        """

        with io.StringIO() as buf, redirect_stdout(buf):
            funs = euptf.which_PTF(predictor= self.soildata_r, target =
                                   ro.StrVector(["THS", "FC", "FC_2", "WP", "KS", "VG", "AWC", "AWC_2"]))
        funs = r_to_pd(funs).to_dict(orient="records")[0]
        return funs

    def water_capacity(self, quantiles = False):
        """
        Get soil saturation capacity, field capacity at 10 and 33 kPA,
        wilting point (1500kPA) and available water capacity.
        """

        targets = ["THS", "FC_2", "FC", "WP", "AWC", "AWC_2"]
        query = "quantiles" if quantiles else "predictions"

        sdata_r = self.soildata_r
        for t in targets:
            sdata_r =  euptf.euptfFun(ptf = self.funs[t], predictor = sdata_r,
                                    target = t, query=query)
        sdata = r_to_pd(sdata_r)
        names = self._pred_names(targets, self.funs, quantiles)
        rdata = sdata.rename(names, axis=1)[list(names.values())]
        for v in ["awc_33", "awc_10", "wp", "fc_33", "fc_10", "sat"]: #Reorder columns
            rdata.insert(0, v, rdata.pop(v))
        return rdata

    def _pred_names(self, targets, funs, quantiles):
        names = dict()
        new_names = {"THS": "sat", "WP" : "wp",
                     "FC": "fc_33", "FC_2": "fc_10",
                    "AWC_2": "awc_10", "AWC" : "awc_33"}
        for t in targets:
            tnew = new_names[t]
            if quantiles:
                names[f"{t}_{funs[t]}_quantile= 0.05"] = f"{tnew}_05"
                names[f"{t}_{funs[t]}_quantile= 0.25"] = f"{tnew}_25"
                names[f"{t}_{funs[t]}_quantile= 0.5"] =  f"{tnew}"
                names[f"{t}_{funs[t]}_quantile= 0.75"] =  f"{tnew}_75"
                names[f"{t}_{funs[t]}_quantile= 0.95"] =  f"{tnew}_95"
            else:
                names[f"{t}_{funs[t]}"] = f"{tnew}"

        return names

    def _eupt_fy(self, soildata):
        """rename columns and mandatory depth"""
        sdata = soildata.rename(str.lower, axis=1)
        sdata = sdata.rename({"clay" : "USCLAY", "silt" : "USSILT",
                            "sand" : "USSAND"}, axis=1)
        if "om" in sdata.columns:
            sdata = sdata.rename({"om" : "OC"}, axis=1)
        if not "depth_m" in sdata.columns:
            sdata["DEPTH_M"] = 10
        return sdata
