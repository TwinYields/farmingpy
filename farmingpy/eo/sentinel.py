from sentinelhub import CRS, BBox, Geometry, DataCollection, SHConfig
from eolearn.core import EOWorkflow, FeatureType, LoadTask, OutputTask, SaveTask, linearly_connect_tasks
from eolearn.io import SentinelHubDemTask, SentinelHubEvalscriptTask, SentinelHubInputTask, ExportToTiffTask
import xarray as xr
import rioxarray
import rasterio
import datetime
import numpy as np
import math

class S2SentinelHub(object):

    def __init__(self, *, maxcc = 0.5, resolution=10, config=SHConfig()):

        self.bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        self.config = config
        self.bbox = None
        self._result = None
        self.maxcc = maxcc
        self.resolution = resolution
        self.time_difference = datetime.timedelta(hours=1)

    def get_data(self, geom, startdate, enddate):
        self.geom = geom
        time_interval = (startdate, enddate)
        bbox = BBox(geom.envelope, crs= CRS.UTM_35N)
        self._run_tasks(bbox, time_interval)

    def _run_tasks(self, bbox, time_interval):
        s2task = SentinelHubInputTask(
            data_collection=DataCollection.SENTINEL2_L2A,
            bands=self.bands,
            bands_feature=(FeatureType.DATA, "L2A_data"),
            additional_data=[(FeatureType.MASK, "CLD"),
                            (FeatureType.MASK, "CLP"),
                            (FeatureType.DATA, "sunZenithAngles"),
                            (FeatureType.DATA, "sunAzimuthAngles"),
                            (FeatureType.DATA, "viewZenithMean"),
                            (FeatureType.DATA, "viewAzimuthMean")],
            resolution= self.resolution,
            maxcc= self.maxcc,
            time_difference= self.time_difference,
            config = self.config,
            max_threads=8,
        )
        output_task = OutputTask("s2data")
        workflow_nodes = linearly_connect_tasks(s2task, output_task)
        workflow = EOWorkflow(workflow_nodes)
        result = workflow.execute({
            workflow_nodes[0]: {"bbox": bbox, "time_interval": time_interval},
        })
        self.s2task = s2task
        self._result = result
        self._orig_ds = self._eopatch_to_dataset(result.outputs["s2data"])
        self.ds = self._orig_ds.copy()

    def _eopatch_to_dataset(self, ep):
        bb = ep["bbox"]
        x = np.concatenate((ep.data["L2A_data"],
                            ep.data["sunZenithAngles"],
                            ep.data["sunAzimuthAngles"],
                            ep.data["viewZenithMean"],
                            ep.data["viewAzimuthMean"],
                            ep.mask["CLD"]/100,
                            ep.mask["CLP"]/255),
                           axis=3)
        tn,yn,xn,bn = x.shape
        tf = rasterio.transform.from_bounds(*bb, width=xn, height=yn) # Matches GeoTiff task in eo-learn
        xc = np.arange(tf.c, tf.c + (xn*tf.a), step=tf.a)[:xn] + tf.a/2
        yc = np.arange(tf.f, tf.f + (yn*tf.e), step=tf.e)[:yn] + tf.e/2
        cl_bands = self.bands + ["sunZenithAngles", "sunAzimuthAngles", "viewZenithMean",
                            "viewAzimuthMean", "CLD", "CLP"]
        ds = xr.DataArray(x, dims=["time", "y", "x", "band"],
            coords= {
                "time" : ep.timestamp,
                "y" : yc,
                "x" : xc,
                "band" : cl_bands
            })
        ds.rio.set_crs(bb.crs.pyproj_crs(), inplace=True)
        ds = ds.rio.clip([self.geom])
        return ds

    @property
    def ndvi(self):
        b4 = self.ds.sel(band="B04")
        b8 = self.ds.sel(band="B8A")
        return (b8 - b4) / (b8 + b4)

    def mask_clouds(self, p_cloud = .1, p_missing = .5):
        cp = self._orig_ds.sel(band="CLP")
        m_ds = self._orig_ds.where(cp < p_cloud)
        N = self._orig_ds.count(["x", "y", "band"]).max()
        tidx = m_ds.count(["x", "y", "band"]) > N * p_missing
        self.ds = m_ds.sel(time = tidx)
