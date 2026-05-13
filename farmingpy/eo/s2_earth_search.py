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
#from .stac import item_quality, items_to_df
from .s2_cdse import S2CDSE
from .stac import item_quality, unique_items, download_s2_item

class S2EarthSearch(S2CDSE):
   
    def __init__(self, geodf, 
                query= {"eo:cloud_cover": {"lt": 60}}
                ):
        """Initialize a Earth Search Sentinel-2 client.

        Args:
            geodf (geopandas.GeoDataFrame): Area of interest used for item search,
                quality filtering, and clipping downloaded imagery.
            query (dict, optional): STAC item-search query. Defaults to filtering
                scenes with less than 60 percent cloud cover.
        """

        self.query = query
        self.clipdf = geodf
        self.location = gpd.GeoDataFrame(geometry=[self.clipdf.union_all().centroid], 
                                            crs=self.clipdf.crs).to_crs("epsg:4326")["geometry"].iloc[0]
        
        URL = "https://earth-search.aws.element84.com/v1"
        self.catalog = pystac_client.Client.open(URL)
        self.catalog.add_conforms_to("ITEM_SEARCH")

        self.data = None
        self.downloaded_files = None
        self.source = "es"