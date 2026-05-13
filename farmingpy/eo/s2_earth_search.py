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

class S2EarthSearch(object):

   
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
        #if unique:
        #    return unique_items(items, grid_code=grid_code)
        #else:
        #    return items
        return items
    
    @staticmethod    
    def items_to_df(items):
        item_data = []
        for item in items:
            props = dict(time = item.properties["datetime"],
                    gridcode = item.properties["grid:code"],
                    id = item.id,
                    vegetation =  item.properties["s2:vegetation_percentage"],
                    not_vegetated = item.properties['s2:not_vegetated_percentage'], 
                    water =  item.properties['s2:water_percentage']
                    )
            item_data.append(props)

        data = pd.DataFrame(item_data)
        data["good"] = data[["vegetation", "not_vegetated", "water"]].sum(axis=1)
        # Drop duplicated dates S2A and S2B can have the same acquisition date
        data.insert(0, "date", pd.to_datetime(data["time"]).dt.date)
        return data