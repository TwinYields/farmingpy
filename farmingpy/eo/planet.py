import rioxarray
import os
import xarray as xr
import glob
import json
import numpy as np

def read_planet(img_file, apply_mask=True, confidence=60, clip=None):
    """Read planetscope AnalyticMS_SR_8b .tif and apply mask.

    Args:
        img_file (str): Image file path.
        apply_mask (bool, optional): Apply udm1 mask and remove points with low mask
            confidence if True. Defaults to True.
        confidence (int, optional): Mask confidence threshold. Defaults to 60.
        clip (GeoDataFrame, optional): GeoDataFrame used to clip the image.

    Returns:
        xarray.DataArray: Planetscope image
        xarray.DataArray: Mask
    """

    dir, file = os.path.split(img_file)
    t = file.split("_3B")[0]
    mask_file = glob.glob(f"{dir}/*{t}*udm2*.tif")[0]
    mask = rioxarray.open_rasterio(mask_file)
    pl_img = rioxarray.open_rasterio(img_file, masked=True)
    crs = pl_img.spatial_ref.attrs["crs_wkt"]


    if apply_mask:
        pl_img = pl_img.where(mask.sel(band=8) == 0, np.nan)
        pl_img = pl_img.where(mask.sel(band=7) >= confidence, np.nan)

    if clip is not None:
        clip = clip.to_crs(crs)
        pl_img = pl_img.rio.clip(clip.geometry.values, drop=True)
        mask = mask.rio.clip(clip.geometry.values, drop=True)

    pl_img["band"] = np.array(pl_img.attrs["long_name"])
    mask["band"] = np.array(mask.attrs["long_name"])
    pl_img = xr.concat([pl_img, mask], dim="band")
    pl_img.attrs["long_name"] = list(pl_img.band.to_numpy())

    pl_img = pl_img.rio.write_crs(crs)
    # Filter out udm1 mask
    pl_img  = pl_img.transpose('band', 'y', 'x')
    mask  = mask.transpose('band', 'y', 'x')
    time = np.datetime64(pl_img.attrs["TIFFTAG_DATETIME"].replace(":", "-", 2))
    attrs = pl_img.attrs.copy()
    pl_img = pl_img / 1e4
    pl_img.attrs.update(attrs)
    pl_img["time"] = time.astype('datetime64[ns]')
    return pl_img

def planet_to_S2_dataset(ds):
    """Convert PlanetScope 8 band image to fake Sentinel2
    dataset to be used with `twinyields.eo.BioPhysStbx` models.

    Args:
        ds (xarray.DataArray): Planet image loaded using `read_planet`.

    Returns:
        xarray.DataArray: Dataset with bands that can be used with
        `twinyields.eo.BioPhysStbx` 10m models.
    """

    pl_img = ds
    pl_s2 = pl_img.sel(band=["green", "red", "nir"])
    pl_s2["band"] = np.array(["B03", "B04", "B08"])
    info = json.loads(pl_img.attrs['TIFFTAG_IMAGEDESCRIPTION'])["atmospheric_correction"]
    vz = pl_s2.isel(band=0).copy()
    vz["band"] = "viewZenithMean"
    vz = vz.where(np.isnan, info["satellite_zenith_angle"])

    sz = pl_s2.isel(band=0).copy()
    sz["band"] = "sunZenithAngles"
    sz = sz.where(np.isnan, info["solar_zenith_angle"])

    va = pl_s2.isel(band=0).copy()
    va["band"] = "viewAzimuthMean"
    va = va.where(np.isnan, info["satellite_azimuth_angle"])

    sa = pl_s2.isel(band=0).copy()
    sa["band"] = "sunAzimuthAngles"
    sa = sa.where(np.isnan, info["solar_azimuth_angle"])

    ds_pl = xr.concat([pl_s2, sz, sa, vz, va], dim="band")
    return ds_pl