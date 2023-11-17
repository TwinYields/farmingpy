import xarray as xr
import datetime
import numpy as np
import math
import os

class BioPhysS2tbx(object):

    def __init__(self, product="LAI", resolution=20, stbx_version="3.0"):
        """Implementation of Sentinel 2 Toolbox Neural Network for predicting
        biophysical parameters from Sentinel 2 data. Uses weights from
        s2tbx-biophysical. The model does input and output range validation,
        but omits convexity checks.

        Based on:
        Weiss, M., Baret, F., Jay, S., 2020.
        S2ToolBox level 2 products LAI, FAPAR, FCOVER. EMMAH-CAPTE, INRAe Avignon.
        http://step.esa.int/docs/extra/ATBD_S2ToolBox_V2.0.pdf

        Args:
            product (str, optional): Output product, "LAI", "FAPAR", "FCOVER",
                "LAI_Cab" or "LAI_cw". Defaults to "LAI".
            resolution (int, optional): Model resolution to use 10 or 20. Defaults to 20.
            stbx_version (str, optional): Toolbox version for weights 2.1 or 3.0. Defaults to "3.0".
        """

        # Load weights and parameters from SNAP files
        if stbx_version == "3.0":
            if resolution == 10:
                snap_path = os.path.dirname(__file__) + "/../data/s2tbx_biophysical_auxdata/3_0/S2A_10m"
            else:
                snap_path = os.path.dirname(__file__) + "/../data/s2tbx_biophysical_auxdata/3_0/S2A"
        elif stbx_version == "2.1":
            snap_path = os.path.dirname(__file__) + "/../data/s2tbx_biophysical_auxdata/2_1/"

        self.stbx_version = stbx_version
        self.resolution = resolution
        self.product = product

        self.extreme_cases = np.loadtxt(f"{snap_path}/{product}/{product}_ExtremeCases", delimiter=",")

        # 2.1 has negative tolerance, 3.0 positive
        if stbx_version == "2.1":
            self.extreme_cases[0] = -self.extreme_cases[0]

        self.normalize_minmax = np.loadtxt(f"{snap_path}/{product}/{product}_Normalisation", delimiter=",")
        self.denormalize_minmax = np.loadtxt(f"{snap_path}/{product}/{product}_Denormalisation", delimiter=",")
        self.minmax_domain = np.loadtxt(f"{snap_path}/{product}/{product}_DefinitionDomain_MinMax", delimiter=",")
        b1 = np.loadtxt(f"{snap_path}/{product}/{product}_Weights_Layer1_Bias", delimiter=",").T
        w1 = np.loadtxt(f"{snap_path}/{product}/{product}_Weights_Layer1_Neurons", delimiter=",").T
        self.wts = np.vstack([b1, w1])
        self.b2 = np.loadtxt(f"{snap_path}/{product}/{product}_Weights_Layer2_Bias", delimiter=",")
        self.wts2 = np.loadtxt(f"{snap_path}/{product}/{product}_Weights_Layer2_Neurons", delimiter=",")


    def __call__(self, ds, *args, **kwargs):
        """Run the model on dataset.

        Args:
            ds xarray.DataArray: Sentinel 2 data. The format
                needs to match the data retrieved using `twinyields.eo.S2SentinelHub`
                class.

        Returns:
            xarray.DataArray: Vegetation index
        """

        ds = ds.transpose("y", "x", "band")
        ds = self.clean_input(ds)
        nm = self.normalize_minmax
        degToRad = math.pi/ 180

        intercept = ds.isel(band=0)
        intercept.coords["band"] = "I"
        intercept = intercept.where(np.isnan, 1)

        # With 10m resolution fewer bands are used
        # http://step.esa.int/docs/extra/ATBD_S2ToolBox_V2.0.pdf
        if self.resolution == 10:
            bands = ds.sel(band=["B03", "B04", "B08"])
            for i in range(3):
                bands[:,:,i] = self.normalize(bands[:,:,i], *nm[i,:])

            viewZen_norm = self.normalize(np.cos(ds.sel(band="viewZenithMean") * degToRad), *nm[3,:])
            sunZen_norm  = self.normalize(np.cos(ds.sel(band="sunZenithAngles") * degToRad), *nm[4,:])
            relAzim_norm = self.normalize(np.cos((ds.sel(band="sunAzimuthAngles") - ds.sel(band="viewAzimuthMean")) * degToRad), *nm[5,:])
            relAzim_norm.coords["band"] = "relAzim_norm"
            X = xr.concat([intercept, bands, viewZen_norm,sunZen_norm,relAzim_norm],
                      dim="band")
        else:
            b03_norm = self.normalize(ds.sel(band="B03"), *nm[0,:])
            b04_norm = self.normalize(ds.sel(band="B04"), *nm[1,:])
            b05_norm = self.normalize(ds.sel(band="B05"), *nm[2,:])
            b06_norm = self.normalize(ds.sel(band="B06"), *nm[3,:])
            b07_norm = self.normalize(ds.sel(band="B07"), *nm[4,:])
            b8a_norm = self.normalize(ds.sel(band="B8A"), *nm[5,:])
            b11_norm = self.normalize(ds.sel(band="B11"), *nm[6,:])
            b12_norm = self.normalize(ds.sel(band="B12"), *nm[7,:])
            viewZen_norm = self.normalize(np.cos(ds.sel(band="viewZenithMean") * degToRad), *nm[8,:])
            sunZen_norm  = self.normalize(np.cos(ds.sel(band="sunZenithAngles") * degToRad), *nm[9,:])
            relAzim_norm = self.normalize(np.cos((ds.sel(band="sunAzimuthAngles") - ds.sel(band="viewAzimuthMean")) * degToRad), *nm[10,:])
            relAzim_norm.coords["band"] = "relAzim_norm"
            X = xr.concat([intercept, b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm,sunZen_norm,relAzim_norm],
                      dim="band")

        X = X.transpose("y", "x", "band").to_numpy()
        l1 = np.tanh(np.dot(X, self.wts))
        l2 = np.dot(l1, self.wts2) + self.b2
        lai = self.denormalize(l2, *self.denormalize_minmax)
        lds = ds.isel(band=0)
        lds.values = lai
        lds.coords["band"] = "LAI"

        # Remove extreme values from output
        tolerance = self.extreme_cases[0]
        lai_min = self.extreme_cases[1]
        lai_max = self.extreme_cases[2]
        l_copy = lds.copy()
        lds = lds.where(l_copy >= -tolerance, np.nan) #Everything less than tolerance is nan
        lds = lds.where(l_copy >= lai_min, lai_min) #Everything below lai_min = lai_min
        lds = lds.where(l_copy <= lai_max, lai_max) #Everything above lai_max to lai_max
        lds = lds.where(l_copy <= (lai_max + tolerance), np.nan) #Everything above lai_max - tolerance to NaN
        return lds

    def normalize(self, unnormalized, min, max):
        return 2 * (unnormalized - min) / (max - min) - 1

    def denormalize(self, normalized, min, max):
        return 0.5 * (normalized + 1) * (max - min) + min

    # Remove values outside snap accepted range
    def clean_input(self, ds):
        if self.resolution == 10:
            s2_bands = ["B03", "B04", "B08"]
        else:
            s2_bands = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
        lai_min = self.minmax_domain[0,:]
        lai_max = self.minmax_domain[1,:]
        X = ds.sel(band=s2_bands)
        X = X.where(X > lai_min, np.nan).where(X < lai_max, np.nan)
        ds.loc[:,:,s2_bands] = X.loc[:,:,s2_bands]
        return ds
