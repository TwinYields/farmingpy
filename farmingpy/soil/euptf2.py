#%%
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
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

def _quantile_names(targets, funs):
    names = dict()
    for t in targets:
        if t == "THS":
            tnew = "sat"
        elif t == "WP":
            tnew = "pwp"
        else:
            tnew  = t.lower()

        names[f"{t}_{funs[t]}_quantile= 0.05"] = f"{tnew}_05"
        names[f"{t}_{funs[t]}_quantile= 0.25"] = f"{tnew}_25"
        names[f"{t}_{funs[t]}_quantile= 0.5"] =  f"{tnew}"
        names[f"{t}_{funs[t]}_quantile= 0.75"] =  f"{tnew}_75"
        names[f"{t}_{funs[t]}_quantile= 0.95"] =  f"{tnew}_95"
    return names

def _pp_names(targets, funs):
    names = dict()
    for t in targets:
        if t == "THS":
            tnew = "sat"
        elif t == "WP":
            tnew = "pwp"
        else:
            tnew  = t.lower()
        names[f"{t}_{funs[t]}"] = f"{tnew}"
    return names

def _eupt_fy(soildata):
    """rename columns and mandatory depth"""
    sdata = soildata.rename(str.lower, axis=1)
    sdata = sdata.rename({"clay" : "USCLAY", "silt" : "USSILT",
                        "sand" : "USSAND"}, axis=1)
    if not "depth_m" in sdata.columns:
        sdata["DEPTH_M"] = 10
    return sdata

def euptf2_which_ptf(sdata):
    sdata = _eupt_fy(sdata)
    sdata_r = pd_to_r(sdata)
    return _euptf2_which_ptf_r(sdata_r)

def _euptf2_which_ptf_r(sdata_r):
    #Find best function for a parameter, redirect redundant print from R
    with io.StringIO() as buf, redirect_stdout(buf):
        funs = euptf.which_PTF(predictor= sdata_r, target = ro.StrVector(["THS", "FC", "WP", "KS", "VG", "AWC"]))
    funs = r_to_pd(funs).to_dict(orient="records")[0]
    return funs
#%%
def euptf2_soil_properties(soildata, quantiles = False):
    sdata = _eupt_fy(soildata)
    targets = ["THS", "FC", "WP", "AWC"]
    query = "quantiles" if quantiles else "predictions"
    sdata_r = pd_to_r(sdata)
    funs = _euptf2_which_ptf_r(sdata_r)

    for t in targets:
        sdata_r =  euptf.euptfFun(ptf = funs[t], predictor = sdata_r,
                                target = t, query=query)
    sdata = r_to_pd(sdata_r)
    names = _quantile_names(targets, funs) if quantiles else _pp_names(targets, funs)
    rdata = sdata.rename(names, axis=1)[list(names.values())]
    for v in ["awc", "pwp", "fc", "sat"]: #Reorder columns
        rdata.insert(0, v, rdata.pop(v))
    return rdata

