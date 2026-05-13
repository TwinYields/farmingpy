import pandas as pd
import rioxarray as rio
import pystac
import rasterio
import xarray as xr

SCL_NODATA = 255
import numpy as np

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

def items_to_df(items, source="cdse"):
    item_data = []
    if source == "cdse":
        for item in items:
            props = dict(time = item.properties["datetime"],
                gridcode = item.properties["grid:code"],
                id = item.id
                )
            props.update(item.properties["statistics"])
            item_data.append(props)
    elif source == "es":
        for item in items:
            props = dict(time = item.properties["datetime"],
                gridcode = item.properties["grid:code"],
                id = item.id,
                vegetation =  item.properties["s2:vegetation_percentage"],
                not_vegetated = item.properties['s2:not_vegetated_percentage'], 
                water =  item.properties['s2:water_percentage'],
                nodata = item.properties['s2:nodata_pixel_percentage']
                )
            item_data.append(props)
    
    data = pd.DataFrame(item_data)
    data["good"] = data[["vegetation", "not_vegetated", "water"]].sum(axis=1)
    data.insert(0, "date", pd.to_datetime(data["time"]).dt.date)
    return data


def item_quality(item, clipdf, source="cdse"):
    i = item
    if source == "cdse":
        scl_band = "SCL_20m"
        crs = i.assets["SCL_20m"].extra_fields["proj:code"]
    elif source == "es":
        scl_band = "scl"
        crs = i.properties["proj:code"]

    path = i.assets[scl_band].href

    clipdf = clipdf.to_crs(crs)
    scl = rio.open_rasterio(path, masked=True, cache=False, 
                            lock=False).rio.clip(clipdf.geometry.values[:1],  
                                                 drop=True, from_disk=True)
    scl = scl.rio.write_nodata(SCL_NODATA).rio.clip(clipdf.geometry.values[:1],  
                                                    drop=True, from_disk=False)
    
    #class_df = pd.DataFrame(i.assets["SCL_20m"].extra_fields["classification:classes"])
    #class_names = class_df.name.to_list()
    #print(class_names)
    class_names = ['no_data', 'saturated_or_defective', 
                   'dark_area_pixels', 'cloud_shadows', 
                   'vegetation', 'not_vegetated', 'water', 
                   'unclassified', 'cloud_medium_probability', 
                   'cloud_high_probability', 'thin_cirrus', 'snow']

    
    aoi_pixels = np.sum(scl != SCL_NODATA)

    cls_data = {}
    for idx, cls in enumerate(class_names):
        qi = float(np.sum(scl == idx)/aoi_pixels)
        cls_data[cls] = qi

    labels = ["no_data", "saturated_or_defective", "dark_area_pixels",	"cloud_shadows", "unclassified", "cloud_medium_probability",	
              "cloud_high_probability",	"thin_cirrus", "snow"]
    qdf = pd.DataFrame(cls_data, index=[0])

    return qdf[labels].sum(axis=1).iloc[0], qdf



def download_s2_item(item, clipdf, source = "cdse"):
    i = item
    
    props = [("view:azimuth", "view_azimuth" ), 
            ("view:incidence_angle", "view_zenith"),
            ("view:sun_azimuth", "sun_azimuth"), 
            ("view:sun_elevation", "sun_zenith"),
            ("platform", "platform"),
            ("grid:code", "grid_code")
            ]
    
    cdse_bands = ["B02_10m", "B03_10m", "B04_10m", "B05_20m", 
                  "B06_20m", "B07_20m", "B08_10m", "B8A_20m", "B11_20m", "B12_20m", "SCL_20m"]
    
    if source == "cdse":
        scl_band = "SCL_20m"
        crs = i.assets["SCL_20m"].extra_fields["proj:code"]
        bands = cdse_bands
    elif source == "es":
        scl_band = "scl"
        bands = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", 
                 "swir22", "scl"]
        crs = i.properties["proj:code"]
    clipdf = clipdf.to_crs(crs)

    bdata = []
    for bidx, band in enumerate(bands):
        path = i.assets[band].href
        
        data = rio.open_rasterio(path, 
                                 cache=False, 
                                 lock=False).rio.clip(clipdf.geometry.values,
                                                               drop=True, from_disk=True)
        
        if not scl_band in band:
            if source == "cdse":
                scale = i.assets[band].extra_fields["raster:scale"]
                offset = i.assets[band].extra_fields["raster:offset"]
            elif source == "es":
                scale = i.assets[band].extra_fields["raster:bands"][0]["scale"]
                #offset = i.assets[band].extra_fields["raster:bands"][0]["offset"]
                # ES data doesn't really seem to have an offset
                # This matches the data from CDSE
                offset = 0.0

            data = (data*scale) + offset

        if i.assets[band].extra_fields["gsd"] == 20:
            data = data.rio.reproject_match(bdata[0], resampling=rasterio.enums.Resampling.bilinear)
        
        # Use the same band names
        if source == "cdse":
            data.coords["band_name"] = band.split("_")[0]
        elif source == "es":
            data.coords["band_name"] = cdse_bands[bidx].split("_")[0]

        if not scl_band in band:
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
