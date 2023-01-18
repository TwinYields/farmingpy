import geopandas as gpd
import pandas as pd
import shapely
import h3
import h3.api.numpy_int as h3int

def h3grid(gdf, resolution=11, clip=False, buffer=True, on = None):
    orig_crs = gdf.crs

    # TODO test that buffering works
    if buffer:
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(gdf.estimate_utm_crs())
        gdf.geometry = gdf.buffer(h3.average_hexagon_edge_length(resolution, unit="m"))

    gdf  = gdf.to_crs("epsg:4326")
    dfs = []
    for idx in range(gdf.shape[0]):
        f = gdf.iloc[[idx]] # Double index returns geodataframe
        g = f.geometry.iloc[0]
        if type(g) == shapely.MultiPolygon:
            g = f.geometry.unary_union
        outer = [(c[1], c[0]) for c in shapely.get_coordinates(g)]
        holes = [[(c[1], c[0]) for c in shapely.get_coordinates(i)] for i in g.interiors]
        poly = h3.Polygon(outer, *holes)
        hexids = list(h3int.polygon_to_cells(poly, resolution))
        geoms = [shapely.Polygon(h3int.cell_to_boundary(h, geo_json=True)) for h in hexids]
        df = gpd.GeoDataFrame({"H3cell" : hexids}, geometry=geoms, crs="epsg:4326")
        if on is not None:
            for col in on:
                df[col] = f[col].iloc[0]
        if clip:
            df = df.clip(g)
        dfs.append(df)
    res = pd.concat(dfs).to_crs(orig_crs)
    return res