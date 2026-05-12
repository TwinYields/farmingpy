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

    def get_data(self, startdate, enddate, qi_filter=0.1):
        """Search, quality-filter, download, and concatenate Sentinel-2 scenes.

        Args:
            startdate (str): Start date for the STAC datetime range.
            enddate (str): End date for the STAC datetime range.
            qi_filter (float, optional): Maximum allowed fraction of poor-quality
                pixels in the area of interest, based on SCL classes. Defaults to 0.1.

        Notes:
            Downloaded data are stored in ``self.data`` as an ``xarray.Dataset`` with
            a ``time`` dimension. If no scenes pass the filter, ``self.data`` is set
            to an empty list.
        """

        items = self.search_items(startdate, enddate)
        data = []
        N = len(items)
        #for item in items:
        for i in trange(N):
            item = items[i]
            q, qdf = item_quality(item, self.clipdf)
            if q <= qi_filter:
                ds = download_s2_item(item, self.clipdf)
                data.append(ds)
        if data:
            self.data = xr.concat(data, dim="time", coords="different", compat="equals")
        else:
            self.data = []
        
    def search_items(self, startdate, enddate):
        """Search CDSE STAC for Sentinel-2 L2A items intersecting the AOI centroid.

        Args:
            startdate (str): Start date for the STAC datetime range.
            enddate (str): End date for the STAC datetime range.

        Returns:
            pystac.ItemCollection: Unique Sentinel-2 L2A items matching the date
            range and the instance query.
        """

        dates = f"{startdate}/{enddate}"

        items = self.catalog.search(
            intersects=dict(type="Point", coordinates=[self.location.x, self.location.y]),
            collections=["sentinel-2-l2a"],
            datetime= dates,
            sortby="properties.datetime",
            query= self.query,
        ).item_collection()
        return unique_items(items)
        

SCL_NODATA = 255

def unique_items(items):
    seen_times = set()
    u_items = []
    for item in items:
        time = item.properties["datetime"]
        if not time  in seen_times:
            u_items.append(item)
            seen_times.add(time)
    return pystac.ItemCollection(u_items)


def item_quality(item, clipdf):
    i = item
    path = i.assets["SCL_20m"].href
    clipdf = clipdf.to_crs(i.assets["SCL_20m"].extra_fields["proj:code"])
    scl = rio.open_rasterio(path, masked=True).rio.clip(clipdf.geometry.values[:1],  drop=True, from_disk=True)
    scl = scl.rio.write_nodata(SCL_NODATA).rio.clip(clipdf.geometry.values[:1],  drop=True, from_disk=False)
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
            ("platform", "platform")
            ]
    
    crs = i.assets["SCL_20m"].extra_fields["proj:code"]
    clipdf = clipdf.to_crs(crs)

    bdata = []
    for band in ["B02_10m", "B03_10m", "B04_10m", "B05_20m", "B06_20m", "B07_20m", "B08_10m", "B8A_20m", "B11_20m", "B12_20m", "SCL_20m"]:
        path = i.assets[band].href
        
        data = rio.open_rasterio(path, lock=False).rio.clip(clipdf.geometry.values,  drop=True, from_disk=True)
        
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
