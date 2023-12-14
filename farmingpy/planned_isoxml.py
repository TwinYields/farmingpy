import xml.etree.ElementTree as ET
import os
import numpy as np
import zipfile
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import Polygon


class TaskReader(object):

    def __init__(self, taskfile):
        """
        Class to read planned ISOBUS task files in grid format.

        Parameters
        ----------
        taskfile
            Path to TASKDATA.XML or zipped task file.

        Based on: ISO 11783-10:2015.
        """

        if os.path.isdir(taskfile):
            taskfile = os.path.join(taskfile, "TASKDATA.XML")

        ext = os.path.splitext(taskfile)[1].lower()

        if ext == ".xml":
            self.task = ET.parse(taskfile)
            self.path = os.path.split(taskfile)[0]
            self.root = self.task.getroot()
        elif ext == ".zip":
            self.zf = zipfile.ZipFile(taskfile)
            tf = [f for f in self.zf.namelist() if "TASKDATA.XML" in f][0]
            self.root = ET.fromstring(self.zf.read(tf))

        self.ext = ext
        self.field = Polygon([(float(pnt.attrib["D"]), float(pnt.attrib["C"])) for pnt in self.root.findall("PFD/PLN/LSG/PNT")])
        self.read_grid()

    def read_grid(self, grid_idx = 0):
        grids = self.root.findall('./TSK/GRD')
        g = grids[grid_idx]
        grid_info = g.attrib
        self.grid_info = grid_info
        grd_file = grid_info["G"] + ".bin"
        grid_type = int(self.grid_info["I"])
        assert grid_type == 2, "Unsupported task file, grid type 2 is required (GRD tag, I attribute)"

        pdv = self.root.find("./TSK/TZN").findall("PDV")
        pdt = self.root.findall("./PDT")
        vpns = [p.attrib["E"] for p in pdv]

        self.products = [p.attrib["B"] for p in pdt]
        self.units = [self.root.find(f"./VPN[@A='{v}']").attrib["E"] for v in vpns]
        self.scales = [self.root.find(f"./VPN[@A='{v}']").attrib["C"] for v in vpns]

        if self.ext == ".xml":
            grid = np.fromfile(os.path.join(self.path,  grd_file), dtype=np.int32)
        elif self.ext == ".zip":
            gf = [f for f in self.zf.namelist() if grd_file in f][0]
            grid = np.frombuffer(self.zf.read(gf), dtype=np.int32)


        # Grid Minimum East and North positions
        y0, x0 = np.asarray([grid_info["A"], grid_info["B"]]).astype("float")
        # Grid cell sizes
        sy, sx = np.asarray([grid_info["C"], grid_info["D"]]).astype("float")

        nr = int(grid_info["F"])
        nc = int(grid_info["E"])
        grid = grid.reshape((nr, nc, -1))
        x = np.array([x0+c*sx for c in range(nc)])
        y = np.array([y0+r*sy for r in range(nr)])

        gx = xr.DataArray(np.float64(grid), dims=["y", "x", "product"],
                  coords={"x" : x, "y" : y})

        gx.attrs["products"] = self.products
        gx.attrs["units"] = self.units

        # Scale data according to VPN attrib C
        for c in range(len(self.scales)):
            gx[:,:,c] *= float(self.scales[c])

        self.grid = gx.rio.write_crs("epsg:4326")
        self.grid = self.grid.rio.write_nodata(np.nan)

        # Try to clip based on field polygon
        # sometimes the field polygon doesn't match the grid and this fails
        # -> just returns the grid
        try:
            self.grid = self.grid.rio.clip([self.field])
        except:
            pass


    @property
    def points(self):
        df = self.grid.to_dataframe("rate").reset_index()
        gdf = gpd.GeoDataFrame(df[["product", "rate"]],
                         geometry=gpd.points_from_xy(df["x"], df["y"]), crs="epsg:4326" )
        return gdf.dropna()
