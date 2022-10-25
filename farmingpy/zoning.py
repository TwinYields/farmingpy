import verde as vd
import pyproj
import numpy as np
import pandas as pd
from rasterio.transform import Affine
import rasterio.features
from shapely.geometry import shape
import geopandas as gpd

def rasterize(df, cols, spacing=1, crs="epsg:2393"):
        transformer = pyproj.Transformer.from_crs("epsg:4326", crs, always_xy=True)
        proj_coords = transformer.transform(df.longitude, df.latitude)
        reducer = vd.BlockReduce(np.median, spacing=1)
        data = tuple(df[col].to_numpy() for col in cols)
        filter_coords, filter_data = reducer.filter(proj_coords, data)
        x,y=proj_coords
        region = (np.floor(x.min()), np.ceil(x.max()), np.floor(y.min()), np.ceil(y.max()))
        grid_coords = vd.grid_coordinates(region = region, spacing=spacing, adjust="region")
        for i in range(len(cols)):
            colname = cols[i]
            gridder = vd.ScipyGridder(method="nearest").fit(filter_coords, filter_data[i])
            grid = gridder.grid(coordinates=grid_coords, data_names=colname)
            grid = vd.distance_mask(proj_coords, maxdist=5, grid=grid)
            if i == 0:
                idata = grid.copy()
            else:
                idata[colname] = grid[colname]
        idata.attrs["crs"] = crs
        return idata

def unique_zones(data, min_area=500):
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
            vdf[cols[i]] = g[i]
        zones.append(vdf)
    zone_df = pd.DataFrame(zones)

    ## Convert to polygons
    x = grid.easting.values
    y = grid.northing.values
    Z = grid.data.values

    xres = (x[-1] - x[0]) / len(x)
    yres = (y[-1] - y[0]) / len(y)
    transform = Affine.translation(x[0] - xres / 2, y[0] - yres / 2) * Affine.scale(xres, yres)

    shapes = rasterio.features.shapes(grid.data.values.astype("float32"),
                                  mask=grid.data.values>0, connectivity=4, transform=transform)
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
        crs=data.crs)

    gdf = gdf.merge(zone_df)
    gdf = gdf[gdf.area > min_area]
    gdf = gdf.iloc[np.argsort(-gdf.area)]
    gdf["zone"] = np.array(range(gdf.shape[0]))+1
    gdf = gdf.reset_index(drop=True)
    return gdf