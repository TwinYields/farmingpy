import pythonnet
if pythonnet.get_runtime_info() is None:
    pythonnet.load("coreclr")
import clr
import sys
import os
from .ddi import read_ddis
import pandas as pd
import geopandas as gpd
import numpy as np

sys.path.append(os.path.dirname(__file__) + "/data")
clr.AddReference("ISOXML")
import ISOXML

DDIs = read_ddis(os.path.dirname(__file__) + "/data/ddi_export_20221004.txt")

class TimeLogData(object):

    def __init__(self, taskfile):
        self.load(taskfile)
        self.farm = self._timelog_data[0].farm
        self.field = self._timelog_data[0].field
        self.taskname = self._timelog_data[0].taskname
        self.devices = list(self._timelog_data[0].devices)
        self.products = dict(self._timelog_data[0].products)
        self._rates = None
        self._data = None

    def load(self, taskfile):
        # Call C# library to read data
        self._timelog_data = ISOXML.TimeLogReader.ReadTaskFile(taskfile)

    @property
    def headers(self):
        det = []
        detno = []
        dpd = []
        dvc = []
        ddi = []
        ddi_desc = []
        tlg = self._timelog_data[0]
        for tdata in tlg.datalogdata:
            det.append(tdata.DETdesignator)
            detno.append(tdata.DETno)
            dpd.append(tdata.DPDdesignator)
            dvc.append(tdata.DVCdesignator)
            ddi.append(tdata.DDI)
            ddi_desc.append(DDIs.get(tdata.DDI, {"description": ""})["description"])
        return pd.DataFrame({"Device": det, "DET": detno, "Description": dpd, "DVC": dvc,
                      "DDI": ddi, "DDI_description": ddi_desc})

    @property
    def data(self):
        if self._data is not None:
            return self._data
        df = pd.concat([self._tlg_to_dataframe(tlg) for tlg in self._timelog_data])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.PositionEast / 1e7, df.PositionNorth / 1e7),
                           crs="epsg:4326")
        del gdf["PositionEast"]
        del gdf["PositionNorth"]
        self._data = gdf
        return gdf

    def _tlg_to_dataframe(self, tlg):
        df = pd.DataFrame()
        for hdata in tlg.datalogheader:
            df[hdata.name.replace(" ", "_")] = np.array(hdata.values)
        for tdata in tlg.datalogdata:
            df[tdata.name.replace(" ", "_").lower()] = np.array(tdata.values)
        df.insert(0, "time", pd.DatetimeIndex(df.TimeStartDATE + "T" + df.TimeStartTOFD))
        del df["TimeStartDATE"]
        del df["TimeStartTOFD"]
        #df.insert(1, "gpstime", pd.DatetimeIndex(df.GpsUtcDate + "T" + df.GpsUtcTime))
        return df

    """Read only application rate columns"""
    @property
    def rates(self):
        if self._rates is not None:
            return self._rates

        df = pd.concat([self._rates_to_dataframe(tlg) for tlg in self._timelog_data])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.PositionEast / 1e7, df.PositionNorth / 1e7),
                               crs="epsg:4326")
        del gdf["PositionEast"]
        del gdf["PositionNorth"]
        self._rates = gdf
        return gdf

    def _rates_to_dataframe(self, tlg):
        ddis = np.array([d.DDI for d in tlg.datalogdata])
        rate_idx = np.where(ddis == 7)[0]
        working_state_idx = np.where(ddis == 141)[0]
        rate_dets = []
        df = pd.DataFrame()
        for hdata in tlg.datalogheader:
            if hdata.name in ["TimeStartDATE", "TimeStartTOFD", "PositionEast", "PositionNorth"]:
                df[hdata.name.replace(" ", "_")] = np.array(hdata.values)
        #Get columns with rate measurement
        for ridx in rate_idx:
            tdata = tlg.datalogdata[int(ridx)]
            rate_dets.append(tdata.DETno)
            df[tdata.name.replace(" ", "_").lower() + "_" + tdata.DETno] = np.array(tdata.values)/100
        #Get working states for DETs with rate
        for widx in working_state_idx:
            tdata = tlg.datalogdata[int(widx)]
            if tdata.DETno in rate_dets:
                df[tdata.name.replace(" ", "_").lower() + "_" + tdata.DETno] = np.array(tdata.values)
        df.insert(0, "time", pd.DatetimeIndex(df.TimeStartDATE + "T" + df.TimeStartTOFD))
        del df["TimeStartDATE"]
        del df["TimeStartTOFD"]
        return df

    def _repr_html_(self):
        products = list(self.products.values())
        len(self._timelog_data)

        return f"""
        <strong>Farm:</strong> {self.farm} <br/>
        <strong>Field:</strong> {self.field} <br/>
        <strong>Task name: </strong> {self.taskname} <br/>
        <strong>Products: </strong> {products}
        """
