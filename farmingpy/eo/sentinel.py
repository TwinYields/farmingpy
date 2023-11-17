from sentinelhub import CRS, BBox, Geometry, DataCollection, SHConfig, Band, Unit
from eolearn.core import EOWorkflow, FeatureType, LoadTask, OutputTask, SaveTask, linearly_connect_tasks
from eolearn.io import SentinelHubDemTask, SentinelHubEvalscriptTask, SentinelHubInputTask, ExportToTiffTask
import xarray as xr
import rioxarray
import rasterio
import datetime
import numpy as np
import math

from dataclasses import dataclass, field
from enum import Enum

@dataclass
class BandConfig:
    collection : DataCollection
    bands : list
    metabands : list
    resolution : int
    aux_request_args : dict = field(default_factory=dict)

# Add collection with HSL bands available from both constellations
# https://docs.sentinel-hub.com/api/latest/data/hls/
# based on
# https://forum.sentinel-hub.com/t/harmonized-landsat-sentinel-available-on-sentinel-hub/6258/4
DataCollection.define(
    name="HLS_BOTH",
    api_id="hls",
    catalog_id="hls",
    collection_type="HLS",
    service_url="https://services-uswest2.sentinel-hub.com",
    bands= tuple(Band(name, (Unit.REFLECTANCE, Unit.DN), (np.float32, np.int16))
            for name in ["CoastalAerosol", "Blue", "Green", "Red", "NIR_Narrow", "SWIR1", "SWIR2","Cirrus"]),
    metabands = tuple(Band(name, (Unit.DEGREES, Unit.DN), (np.uint16, np.uint16))
                    for name in ["VAA", "VZA", "SAA", "SZA"])  + tuple([Band("QA", (Unit.DN,), (np.uint8,))]),
    has_cloud_coverage=True)

class Bands(Enum):
    S2 = BandConfig(collection = DataCollection.SENTINEL2_L2A,
                            bands = None,
                            metabands= ["sunZenithAngles", "sunAzimuthAngles", "viewZenithMean",
                            "viewAzimuthMean", "CLD", "CLP", "CLM"], resolution=10)

    S2_optical = BandConfig(collection=DataCollection.SENTINEL2_L2A,
                    bands = None,
                    metabands= ["CLD"], resolution=10)

    S2_all = BandConfig(collection=DataCollection.SENTINEL2_L2A,
                    bands = None,
                    metabands= None, resolution=10)

    HLS = BandConfig(collection=DataCollection.HLS_BOTH,
                    bands = None,
                    metabands= None, resolution=30)


    S1_card4l = BandConfig(collection=DataCollection.SENTINEL1_IW_ASC,
                    bands=None, metabands=["localIncidenceAngle", "shadowMask"],
                    resolution=10,
                    aux_request_args= {"processing" : {"backCoeff": "GAMMA0_TERRAIN",
                                        "orthorectify": True,
                                        "demInstance": "COPERNICUS",
                                        "downsampling": "BILINEAR",
                                        "upsampling": "BILINEAR",
                                        "speckleFilter": {
                                            "type": "LEE",
                                            "windowSizeX": 5,
                                            "windowSizeY": 5}
                                        }})
    @property
    def collection(self):
        return self.value.collection

    @property
    def bands(self):
        return self.value.bands

    @property
    def metabands(self):
        return self.value.metabands

    @property
    def resolution(self):
        return self.value.resolution

    @property
    def aux_request_args(self):
        return self.value.aux_request_args
class SentinelHubClient(object):

    def __init__(self, config : SHConfig = None, max_threads = 8, crs = CRS.UTM_35N,
                time_difference = datetime.timedelta(hours=1)):
        if config is None:
            self.config = SHConfig()
        self.max_threads = max_threads
        self.crs = crs
        self._last_task = None
        self.time_difference = time_difference

    def get(self, bandconfig, geom, time_interval, maxcc = 0.5):
        return self.get_from_collection(bandconfig.collection, geom, time_interval,
                    bandconfig.resolution, maxcc, bandconfig.aux_request_args,
                    bandconfig.bands, bandconfig.metabands
        )

    def get_from_collection(self, data_collection, geom, time_interval, resolution,
                    maxcc=0.5, aux_request_args = None, bands=None, metabands=None):
        if bands is None:
            bands = [b.name for b in data_collection.bands]
        if metabands is None:
            metabands = [mb.name for mb in data_collection.metabands]

        # Get metaband featuretypes
        additional_data = []
        mb_data = [] # Keep names for converting to DataArray
        mb_mask = []
        for mb in data_collection.metabands:
            if mb.name in metabands:
                if mb.output_types[0] == np.float32:
                    ftype = FeatureType.DATA
                    mb_data.append(mb.name)
                else:
                    ftype = FeatureType.MASK
                    mb_mask.append(mb.name)
                additional_data.append((ftype, mb.name))

        task = SentinelHubInputTask(data_collection= data_collection,
            bands = bands,
            bands_feature = (FeatureType.DATA, "bands"),
            additional_data = additional_data,
            resolution = resolution,
            maxcc = maxcc,
            config = self.config,
            max_threads = self.max_threads,
            time_difference = self.time_difference,
            aux_request_args = aux_request_args)
        self._last_task = task

        bbox = BBox(geom, self.crs)
        ep = task.execute(bbox = bbox, time_interval = time_interval)

        # TODO remove in release
        self._ep = ep
        self._bands = bands
        self._mb_data = mb_data
        self._mb_mask = mb_mask

        return self._eopatch_to_dataset(ep, geom, bands, mb_data, mb_mask, data_collection)

    def _eopatch_to_dataset(self, ep, geom, bands, mb_data, mb_mask, data_collection):
        x = ep.data["bands"]
        if len(mb_data) > 0:
            x = np.concatenate([x,
                    np.concatenate([ep.data[b] for b in mb_data], axis=3)],
                    axis = 3)
        if len(mb_mask) > 0:
            x = np.concatenate([x,
                    np.concatenate([ep.mask[b] for b in mb_mask], axis=3)],
                    axis = 3)
        tn,yn,xn,bn = x.shape

        bb = ep["bbox"]
        tf = rasterio.transform.from_bounds(*bb, width=xn, height=yn) # From GeoTiff task in eo-learn
        xc = np.arange(tf.c, tf.c + (xn*tf.a), step=tf.a)[:xn] + tf.a/2
        yc = np.arange(tf.f, tf.f + (yn*tf.e), step=tf.e)[:yn] + tf.e/2

        ds = xr.DataArray(x, dims=["time", "y", "x", "band"],
            coords= {
                "time" : ep.timestamp,
                "y" : yc,
                "x" : xc,
                "band" : bands + mb_data + mb_mask
            })
        ds.rio.set_crs(bb.crs.pyproj_crs(), inplace=True)
        ds = ds.rio.clip([geom])
        ds.attrs["source"] = "SentinelHub"
        ds.attrs["datacollection"] = data_collection.name
        return ds


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
        bbox = BBox(geom.envelope.bounds, crs= CRS.UTM_35N)
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
                "time" : ep.timestamps,
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

