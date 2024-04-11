import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import matplotlib.pyplot as plt

class FinnishSoil(object):

    def __init__(self):
        self.soil_texture = self.make_soil_df()
        self.soil_regions = self.soil_polygons()
        self.average_soil = self._average_soil()

    def make_soil_df(self):
        """Get finnish soil triangle polygon corners as clay, silt and sand"""
        # Soil trianle polygon corners in ternary
        soil = dict()
        soil["Ht"] = [[0, 0, 100], [30, 0, 70], [30, 20, 50], [0,50,50]]
        soil["He"] = [[0,50,50], [30,20,50], [30,50,20]]
        soil["HeS"] = [[30,20,50], [60, 20, 20], [30,50,20]]
        soil["HtS"] = [[30, 0, 70], [60, 0, 40], [60,20,20], [30, 20, 50]]
        soil["AS"] = [[60,0,40], [100, 0, 0], [60,40,0]]
        soil["HsS"] = [[30,50,20], [60,20,20], [60,40,0], [30,70,0]]
        soil["Hs"] = [[0,50,50], [30,50,20], [30,70,0], [0,100,0]]
        s = []

        for i, name in enumerate(soil.keys(),1):
            soils = pd.DataFrame(np.array(soil[name]), columns=["clay", "silt", "sand"])
            soils["soilclass"] = i
            soils["name"] = name
            s.append(soils)
        soils = pd.concat(s).reset_index(drop=True)
        return soils

    def soil_polygons(self):
        soils = self.soil_texture
        x,y = tri2cart(soils["clay"].to_numpy(), soils["silt"].to_numpy(), soils["sand"].to_numpy())
        out = []
        NS = len(soils["name"].unique())
        for sc in range(1,NS+1):
            idx = soils["soilclass"] == sc
            xs = x[idx]
            ys = y[idx]
            geom = shapely.Polygon([shapely.Point(p) for p in zip(xs,ys)])
            nidx = int(np.where(idx)[0][0])
            out.append({"geometry" : geom, "soilclass" : sc, "name": soils["name"][nidx]})

        soil_regions = gpd.GeoDataFrame(out)
        return soil_regions

    def get_om(self, om_class, range=False):
        """Get organic matter content range from soil sample Multavuus
        attribute

        Based on https://cdnmedia.eurofins.com/european-east/media/1818630/viljavuustutkimuksentulkinta2017teroprint.pdf
        """
        if om_class.endswith("t"):
            om = (40, 60) # Peatlands
        else:
            om_dict = {"vm" : (0,3),
                   "m" : (3, 5.9),
                   "rm" : (6, 11.9),
                   "erm" : (12,19.9),
                   "Mm" : (20, 39.9),
                    }
            om = om_dict[om_class]
        if range:
            return om
        else:
            return (om[0]+om[1])/2

    def get_texture_and_om(self, soil_class, om_class):
        soil = self.average_soil[self.average_soil["name"].str.lower() == soil_class.lower()]
        soil = soil.reset_index(drop=True)
        soil["om"] = self.get_om(om_class)
        return soil

    def _average_soil(self):
        """Return soil texture from soil triangle region centroids"""
        soil_regions = self.soil_regions
        x = soil_regions.centroid.x
        y = soil_regions.centroid.y
        clay,silt,sand = cart2tri(x,y)
        tx = pd.DataFrame(dict(clay = clay, silt=silt, sand = sand))
        tx *= 100.0
        return pd.concat([self.soil_regions[["soilclass", "name"]], tx], axis=1)

    def plot_soil_triangle(self):
        ax = plot_ternary()
        soil_regions = self.soil_regions
        soil_regions.boundary.plot(color="k",zorder=10, ax=ax)
        for i in range(soil_regions.shape[0]):
            xy = list(soil_regions.centroid.iloc[i].coords)[0]
            ax.annotate(soil_regions.name.iloc[i], xy, ha="center", va="center",
                 bbox=dict(facecolor='white', edgecolor = "none", alpha=1),
                 fontsize=12)
        return ax


def plot_ternary():
    ax = plt.axes()
    ax.axis("off")
    for i in range(0, 110, 10):
        io = -(i-100)
        # Parallel
        x1,y1 = tri2cart(i, 0, 100-i)
        ax.annotate(i, (x1-.04,y1), ha="center", va="center") #left
        x2,y2 = tri2cart(i, io , 0)
        ax.plot([x1,x2], [y1,y2],c="gray", linestyle="--")
        x1,y1 = tri2cart(i, 0, 100-i)
        x2,y2 = tri2cart(0, i, io)
        ax.plot([x1,x2], [y1,y2], c="gray", linestyle="--")
        x1,y1 = tri2cart(io, i, 0)
        ax.annotate(i, (x1+.04,y1), ha="center", va="center") #right
        x2,y2 = tri2cart(0, i, io)
        ax.annotate(io, (x2,y2-.04), ha="center", va="center") #bottom
        ax.plot([x1,x2], [y1,y2], c="gray", linestyle="--")
    return ax


def tri2cart(upper_apex, right_apex, left_apex):
    """Converts Ternary to cartesian coordinates"""
    total = upper_apex + right_apex + left_apex
    upper_apex = upper_apex / total
    right_apex = right_apex / total
    left_apex = left_apex / total

    x = 0.5 * (upper_apex + 2 * right_apex) / (upper_apex + right_apex + left_apex)
    y = (3**0.5 / 2) * upper_apex / (upper_apex + right_apex + left_apex)
    return x,y

def cart2tri(x,y):
    """Converts Cartesian x,y to Ternary coordinates"""
    a = y/np.sqrt(0.75)
    b = x - np.sqrt(a**2-y**2)
    c = 1- a - b
    return a,b,c