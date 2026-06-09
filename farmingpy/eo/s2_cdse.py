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
from .stac import item_quality, unique_items, download_s2_item

#xr.set_options(use_new_combine_kwarg_defaults=True)
class S2CDSE(object):
    """Access Sentinel-2 L2A imagery from the Copernicus Data Space Ecosystem."""

    def __init__(self, geodf=None, access_key = None, secret_key = None,
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
        if geodf is not None:
            self.location = gpd.GeoDataFrame(geometry=[self.clipdf.union_all().centroid], 
                                         crs=self.clipdf.crs).to_crs("epsg:4326")["geometry"].iloc[0]
        else:
            self.location = None

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
        self.source = "cdse"

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
        if N == 0:
            print("Nothing to download")
            return
        

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
            q, qdf = item_quality(item, self.clipdf, source = self.source)
            if q <= qi_filter:
                try:
                    ds = download_s2_item(item, self.clipdf, source = self.source)
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

        if len(items) == 0:
            return items

        if unique:
            return unique_items(items, grid_code=grid_code, source=self.source)
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
    
