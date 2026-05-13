import xarray as xr
import pandas as pd
import rioxarray as rio
import geopandas as gpd
import numpy as np
import pystac
import rasterio
import os
import pystac_client
from tqdm.autonotebook import trange
import configparser
from pathlib import Path
import rasterio
from .stac import items_to_df

#xr.set_options(use_new_combine_kwarg_defaults=True)
class S2CDSE(object):
    """Access Sentinel-2 L2A imagery from the Copernicus Data Space Ecosystem."""

    def __init__(self, geodf, access_key = None, secret_key = None,
                 query= {"eo:cloud_cover": {"lt": 60}}
                 ):
        """Initialize a Copernicus Data Space Sentinel-2 client.

        Args:
            geodf (geopandas.GeoDataFrame): Area of interest used for item search,
                quality filtering, and clipping downloaded imagery.
            access_key (str): Copernicus Data Space S3 access key.
            secret_key (str): Copernicus Data Space S3 secret key.
            query (dict, optional): STAC item-search query. Defaults to filtering
                scenes with less than 60 percent cloud cover.
        """
        if access_key == None:
            s3cfg = configparser.ConfigParser()
            s3cfg.read(Path.home() / ".s3cfg")
            if s3cfg.has_section("cdse"):
                access_key = s3cfg["cdse"]["access_key"]
                secret_key = s3cfg["cdse"]["access_token"]
            else:
                raise Exception("No credidentials found for CSDE. Either pass keys as arguments or add to .s2cfg section [cdse].")
            

        self.query = query
        self.clipdf = geodf
        self.location = gpd.GeoDataFrame(geometry=[self.clipdf.union_all().centroid], 
                                         crs=self.clipdf.crs).to_crs("epsg:4326")["geometry"].iloc[0]
        
        # Configuration from
        # https://dataspace.copernicus.eu/news/2025-4-15-exploring-cdse-stac-catalogue-powerful-metadata-discovery-and-abstraction-tool
        os.environ['GDAL_HTTP_TCP_KEEPALIVE'] = "YES"
        os.environ['AWS_S3_ENDPOINT'] = "eodata.dataspace.copernicus.eu"
        os.environ['AWS_ACCESS_KEY_ID'] = access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
        os.environ['AWS_HTTPS'] = "YES"
        os.environ['AWS_VIRTUAL_HOSTING'] = "FALSE"
        os.environ['GDAL_HTTP_UNSAFESSL'] = "YES"

        URL = "https://stac.dataspace.copernicus.eu/v1"
        self.catalog = pystac_client.Client.open(URL)
        self.catalog.add_conforms_to("ITEM_SEARCH")

        self.data = None
        self.downloaded_files = None

    def download_data(self, startdate, enddate, qi_filter=0.1,
                       grid_code = None,
                       path_prefix="", verbose=True):
        """Download quality-filtered Sentinel-2 scenes to NetCDF files.

        This is a convenience wrapper around :meth:`get_data` with
        ``to_files=True``. File paths for accepted scenes are stored in
        ``self.downloaded_files``.

        Args:
            startdate (str): Start date for the STAC datetime range.
            enddate (str): End date for the STAC datetime range.
            qi_filter (float, optional): Maximum allowed fraction of poor-quality
                pixels in the area of interest, based on SCL classes. Defaults to 0.1.
            path_prefix (str or pathlib.Path, optional): Output path prefix used
                when naming downloaded NetCDF files. If empty, files are written
                under ``s2data`` with the prefix ``s2data``.
            verbose (bool, optional): Whether to print file save and skip messages.
                Defaults to True.
        """
        self.get_data(startdate, enddate, qi_filter = qi_filter, grid_code=grid_code,
                      to_files=True, path_prefix=path_prefix, verbose=verbose)

    def get_data(self, startdate, enddate, qi_filter=0.1, 
                    grid_code=None, to_files = False, 
                    path_prefix = "", verbose=True):
        """Search, quality-filter, download, and concatenate Sentinel-2 scenes.

        Args:
            startdate (str): Start date for the STAC datetime range.
            enddate (str): End date for the STAC datetime range.
            qi_filter (float, optional): Maximum allowed fraction of poor-quality
                pixels in the area of interest, based on SCL classes. Defaults to 0.1.
            to_files (bool, optional): If True, write each accepted scene to a
                NetCDF file instead of keeping it in memory. Defaults to False.
            path_prefix (str or pathlib.Path, optional): Output path prefix used
                when ``to_files`` is True. If empty, files are written under
                ``s2data`` with the prefix ``s2data``.
            verbose (bool, optional): Whether to print file save and skip messages
                when ``to_files`` is True. Defaults to True.

        Notes:
            When ``to_files`` is False, downloaded data are stored in ``self.data``
            as an ``xarray.Dataset`` with a ``time`` dimension. If no scenes pass
            the filter, ``self.data`` is set to an empty list. When ``to_files`` is
            True, accepted output paths are stored in ``self.downloaded_files``.
        """

        items = self.search_items(startdate, enddate, grid_code=grid_code)
        data = []
        N = len(items)

        path = Path(path_prefix)
        if path.name == "":
            dir = path
            if path.as_posix() == ".":
                dir = Path("s2data")
            name_prefix = "s2data"
        else:
            dir = path.parent
            name_prefix = path.name
        
        dir.mkdir(exist_ok=True, parents=True)
        fnames = []

        #for item in items:
        for i in trange(N):
            item = items[i]
            q, qdf = item_quality(item, self.clipdf)
            if q <= qi_filter:
                try:
                    ds = download_s2_item(item, self.clipdf)
                except Exception as e:  #rasterio.RasterioIOError as e:
                    print(f"Failed to read from {item}")
                    print(e)
                    continue
                if to_files:
                    fname = dir /  (f"{name_prefix}_{ds.time.values}".split(".")[0].replace(":", "") + ".nc")
                    if fname.exists():
                        if verbose:
                            print(f"File {fname} already existis, skipping download")
                    else:
                        ds.to_netcdf(fname)
                    if verbose:
                        print(f"Saved {fname}")
                    fnames.append(fname)
                else:
                    data.append(ds)

        if data:
            self.data = xr.concat(data, dim="time", coords="different",
                                   join="outer",
                                   compat="equals")
        elif to_files:
            self.downloaded_files = fnames
        else:
            self.data = []

        
    def search_items(self, startdate, enddate, unique=True, grid_code=None):
        """Search CDSE STAC for Sentinel-2 L2A items intersecting the AOI centroid.

        Args:
            startdate (str): Start date for the STAC datetime range.
            enddate (str): End date for the STAC datetime range.

        Returns:
            pystac.ItemCollection: Unique Sentinel-2 L2A items matching the date
            range and the instance query.
        """

        dates = f"{startdate}/{enddate}"

        query = self.query.copy()
        if grid_code is not None:
            query.update({"grid:code" : {"eq" : grid_code}})

        items = self.catalog.search(
            intersects=dict(type="Point", coordinates=[self.location.x, self.location.y]),
            collections=["sentinel-2-l2a"],
            datetime= dates,
            sortby="properties.datetime",
            query= query,
        ).item_collection()
        if unique:
            return unique_items(items, grid_code=grid_code)
        else:
            return items
        

SCL_NODATA = 255

"""
def items_to_df(items):
    item_data = []
    for item in items:
        props = dict(time = item.properties["datetime"],
                gridcode = item.properties["grid:code"],
                id = item.id
                )
        props.update(item.properties["statistics"])
        item_data.append(props)

    data = pd.DataFrame(item_data)
    data["good"] = data[["vegetation", "not_vegetated", "water"]].sum(axis=1)
    # Drop duplicated dates S2A and S2B can have the same acquisition date
    data.insert(0, "date", pd.to_datetime(data["time"]).dt.date)
    return data
"""
    
def unique_items(items, grid_code = None, source="cdse"):
    """Filter items to return all data from the same tile"""
    
    data = items_to_df(items, source)
    # Remove duplicate dates
    data = data.sort_values(["date", "good"], 
                            ascending=True).drop_duplicates("date", ignore_index=True)
    uids = data.id.to_list()
    
    if grid_code is None:
        # Select only one tile
        best_tile = data.groupby("gridcode", as_index=False).agg({"nodata" : "mean", 
                                                            "good" : "mean"}).sort_values("good", ascending=False).iloc[0]["gridcode"]
    else:
        best_tile = grid_code

    items = [item for item in items if item.properties["grid:code"] == best_tile]
    seen_ids = set()
    u_items = []
    # Filter duplicated items by id
    for item in items:
        if item.id in uids and not item.id in seen_ids:
            u_items.append(item)
            seen_ids.add(item.id)

    return pystac.ItemCollection(u_items)




def item_quality(item, clipdf):
    i = item
    path = i.assets["SCL_20m"].href
    clipdf = clipdf.to_crs(i.assets["SCL_20m"].extra_fields["proj:code"])
    scl = rio.open_rasterio(path, masked=True, cache=False, 
                            lock=False).rio.clip(clipdf.geometry.values[:1],  
                                                 drop=True, from_disk=True)
    scl = scl.rio.write_nodata(SCL_NODATA).rio.clip(clipdf.geometry.values[:1],  
                                                    drop=True, from_disk=False)
    class_df = pd.DataFrame(i.assets["SCL_20m"].extra_fields["classification:classes"])
    class_names = class_df.name.to_list()
    aoi_pixels = np.sum(scl != SCL_NODATA)

    cls_data = {}
    for idx, cls in enumerate(class_names):
        qi = float(np.sum(scl == idx)/aoi_pixels)
        cls_data[cls] = qi

    labels = ["no_data", "saturated_or_defective", "dark_area_pixels",	"cloud_shadows", "unclassified", "cloud_medium_probability",	
              "cloud_high_probability",	"thin_cirrus", "snow"]
    qdf = pd.DataFrame(cls_data, index=[0])

    return qdf[labels].sum(axis=1).iloc[0], qdf


def download_s2_item(item, clipdf):
    i = item
    
    props = [("view:azimuth", "view_azimuth" ), 
            ("view:incidence_angle", "view_zenith"),
            ("view:sun_azimuth", "sun_azimuth"), 
            ("view:sun_elevation", "sun_zenith"),
            ("platform", "platform"),
            ("grid:code", "grid_code")
            ]
    
    crs = i.assets["SCL_20m"].extra_fields["proj:code"]
    clipdf = clipdf.to_crs(crs)

    bdata = []
    for band in ["B02_10m", "B03_10m", "B04_10m", "B05_20m", "B06_20m", "B07_20m", "B08_10m", "B8A_20m", "B11_20m", "B12_20m", "SCL_20m"]:
        path = i.assets[band].href
        
        data = rio.open_rasterio(path, 
                                 cache=False, 
                                 lock=False).rio.clip(clipdf.geometry.values,
                                                               drop=True, from_disk=True)
        
        if not "SCL" in band:
            scale = i.assets[band].extra_fields["raster:scale"]
            offset = i.assets[band].extra_fields["raster:offset"]
            data = (data*scale) + offset
        
        if "20m" in band:
            data = data.rio.reproject_match(bdata[0], resampling=rasterio.enums.Resampling.bilinear)
        data.coords["band_name"] = band.split("_")[0]

        if not "SCL" in band:
            data = data.rio.write_nodata(np.nan).rio.clip(clipdf.geometry.values[:1])
        else:
            data = data.rio.write_nodata(SCL_NODATA).rio.clip(clipdf.geometry.values[:1])

        bdata.append(data)
        
    da = xr.concat(bdata, dim="band", 
                   coords="different", compat="equals")
    ds = da.to_dataset(name="data")
    
    
    for p in props:
        ds[p[1]] = i.properties[p[0]]
    ds["time"] = pd.to_datetime(i.properties["datetime"]).to_datetime64()
    ds = ds.set_coords("time")
    ds.coords["band"] = ds.coords["band_name"]

    # Create a mask of valid pixels
    # 4=vegetation, 5=not_vegetated, 6=water, 11=snow
    SCL = ds.sel(band="SCL")
    mask = SCL.where((SCL == 4) | (SCL == 5) | (SCL == 6) | (SCL == 11)) > 0.0
    mask["band"] = "mask"
    ds["mask"] = mask["data"]
    
    del ds.coords["band_name"]
    return ds
