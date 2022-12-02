import verde as vd
import pyproj
import numpy as np
import pandas as pd
import xarray
import rasterio.features
from shapely.geometry import shape
import geopandas as gpd

def rasterize(df, cols, spacing=1, maxdist=5, crs="epsg:2393"):
        transformer = pyproj.Transformer.from_crs("epsg:4326", crs, always_xy=True)
        proj_coords = transformer.transform(df.longitude, df.latitude)
        reducer = vd.BlockReduce(np.median, spacing=spacing)
        data = tuple(df[col].to_numpy() for col in cols)
        filter_coords, filter_data = reducer.filter(proj_coords, data)
        x,y=proj_coords
        region = (np.floor(x.min()), np.ceil(x.max()), np.floor(y.min()), np.ceil(y.max()))
        grid_coords = vd.grid_coordinates(region = region, spacing=spacing, adjust="region")
        for i in range(len(cols)):
            colname = cols[i]
            if type(filter_data) == tuple:
                gridder = vd.ScipyGridder(method="nearest").fit(filter_coords, filter_data[i])
            else:
                gridder = vd.ScipyGridder(method="nearest").fit(filter_coords, filter_data)
            grid = gridder.grid(coordinates=grid_coords, data_names=colname)
            grid = vd.distance_mask(proj_coords, maxdist=maxdist, grid=grid)
            if i == 0:
                idata = grid.copy()
            else:
                idata[colname] = grid[colname]
        idata = idata.rename({"easting" : "x", "northing" : "y"})
        idata.rio.write_crs(crs, inplace=True)
        return idata

def unique_zones(data, connectivity = 8, min_area=500):
    T = type(data)
    if T == xarray.DataArray:
        data = data.to_dataset()

    cols = list(data.data_vars.keys())
    #Round and convert to numpy
    idata = {}
    for col in cols:
        idata[col] = np.around(data[col].values.flatten(), -1)

    ## Get unique combinations of col measurements
    df = pd.DataFrame(idata).dropna()
    grps = df.groupby(cols)
    grid = data.copy()
    grid["data"] = data[cols[0]].copy()
    grp_idx = 1
    zones = []
    for g,v in grps:
        uidx = np.unravel_index(v.index, grid.data.shape)
        grid.data.values[uidx] = grp_idx
        vdf = {"level" : grp_idx, "grp" : g}
        grp_idx +=1
        for i in range(len(cols)):
            vdf[cols[i]] = v[cols[i]].iloc[0]
        zones.append(vdf)
    zone_df = pd.DataFrame(zones)

    #print(zone_df)
    #print(grid)
    gdf = shapes(grid.data, connectivity)
    #print(gdf)
    gdf = gdf.merge(zone_df)
    gdf = gdf[gdf.area > min_area]
    gdf = gdf.iloc[np.argsort(-gdf.area)]
    gdf["zone"] = np.array(range(gdf.shape[0]))+1
    gdf = gdf.reset_index(drop=True)
    return gdf


def shapes(data, connectivity=8):
    T = type(data)
    if T == xarray.DataArray:
        M = data.values.astype("float32")
    else:
        col = list(data.data_vars.keys())[0]
        M = data[col].values.astype("float32")
    shapes = rasterio.features.shapes(M,
                                  mask = M > 0, connectivity = connectivity,
                                  transform=data.rio.transform())
    levels = []
    geometry = []
    zones = []
    idx = 0
    for shapedict, value in shapes:
        levels.append(value)
        geometry.append(shape(shapedict))
        zones.append(idx)
        idx += 1
    gdf = gpd.GeoDataFrame({'zone' : zones, 'level': levels, 'geometry': geometry},
        crs=data.rio.crs)
    return gdf