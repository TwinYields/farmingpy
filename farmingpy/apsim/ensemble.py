from . import apsim
import numpy as np
import pandas as pd
import joblib
from typing import Union
import Models
from Models.TwinYields import ModelEnsemble, SimulationEnsemble
from Models.Core import IModel
from dataclasses import dataclass

@dataclass
class Fertilizer:
    no3 : float = 0.0
    nh4: float = 0.0


class Report(object):
    """Reporting class for APSIM model ensembles"""

    def __init__(self, ensemble):
        """
        Parameters
        ----------

        ensemble
            Model ensemble used for reporting
        """
        self.en = ensemble
        self.report_PMF = False
        self.report_RG = False

        if self.en.Plants is not None:
            self.Plants = self.en.Plants
            self.Grains = self.en.Grains
            self.Leaves = self.en.Leaves
            self.Stems = self.en.Stems
            self.Spikes = self.en.Spikes
            self.report_PMF = True

        if self.en.AGPRyegrass is not None:
            self.AGPRyegrass = self.en.AGPRyegrass
            self.report_RG = True

        self.data = []

    @property
    def today(self):
        """
        Returns
        -------
        np.datetime64
            Current simulation date
        """
        return self.en.today

    @property
    def dataframe(self):
        """
        Returns
        -------
        DataFrame
            Simulated outputs
        """
        df = pd.concat([pd.DataFrame.from_dict(d) for d in self.data])
        return df.reset_index(drop=True)

    def grain_protein(self, Grain):
        """Calculate grain protein %, equation from APSIM"""

        gn = Grain.Live.N + Grain.Dead.N
        gwt = Grain.Live.Wt + Grain.Dead.Wt
        if gwt > 0:
            return (gn/gwt) * 100 * 5.71
        else:
            return 0

    def report(self):
        """
        Called on each timestep in the ensemble, used to store
        simulated values.
        """
        if self.report_PMF:
            self.data.append({
                "idx" : [idx for idx in range(self.en.N)],
                "date" : [self.today for _ in range(self.en.N)],
                "LAI" : [p.LAI for p in self.Plants],
                "Yield" : [g.Wt*10 for g in self.Grains],
                "Biomass" : [p.AboveGround.Wt for p in self.Plants],
                "Grain_protein" : [self.grain_protein(g) for g in self.Grains],
                "Crop_AboveGroundNKgha" : [p.AboveGround.N*10 for p in self.Plants],
                "N_stress" : [l.Fn for l in self.Leaves],
                "W_stress" : [l.Fw for l in self.Leaves],
                "N_TotalPlantSupply" : [p.Arbitrator.N.TotalPlantSupply for p in self.Plants],
                "N_TotalPlantDemand" : [p.Arbitrator.N.TotalPlantDemand for p in self.Plants],
                "DM_NutrientLimitation" : [p.Arbitrator.DM.NutrientLimitation for p in self.Plants]
            })
        if self.report_RG:
            self.data.append({
                "idx" : [idx for idx in range(self.en.N)],
                "date" : [self.today for _ in range(self.en.N)],
                "LAI" : [p.LAI for p in self.AGPRyegrass],
                "Biomass" : [p.AboveGround.Wt for p in self.AGPRyegrass],
            })

class APSIMXEnsemble(object):
    """Ensemble of APSIM simulation models.
    Requires custom build of APSIM from:
    https://github.com/mpastell/ApsimX/tree/clock_management
    """

    def __init__(self, model : Union[str, IModel], N = 50, Ncores = -1):
        """
        Parameters
        ----------

        model
            Union[str, IModel] Base model for the ensemble.
            Path to .apsimx file or C# Models.Core.Simulations object
            Should contain single simulation with single field and
            needs to use TwinClock.
        N
            Number of ensemble members
        Ncores
            Number of cores to use, if -1 all physical cores are used.
        """

        if type(model) == str:
            model = apsim.APSIMX(model).Model
        if Ncores == -1:
            Ncores = joblib.cpu_count(only_physical_cores=False)

        self.en = ModelEnsemble(model, N, Ncores)
        self.en.Prepare()

        # Find objects
        self.Models = self.en.Models
        self.Simulations = self.en.Simulations
        self.N = self.en.N

        self.Fertilisers = [sim.FindDescendant[Models.Fertiliser]() for sim in self.en.Simulations]
        self.WaterBalances = [sim.FindDescendant[Models.WaterModel.WaterBalance]() for sim in self.en.Simulations]

        self.Plants = [sim.FindDescendant[Models.PMF.Plant]() for sim in self.en.Simulations]
        if self.Plants[0] is None:
            self.Plants = None
        else:
            self.Leaves = [plant.FindChild[Models.PMF.Organs.Leaf]() for plant in self.Plants]
            self.Grains = [plant.FindChild[Models.PMF.Organs.ReproductiveOrgan]("Grain") for plant in self.Plants]
            self.Stems = [plant.FindDescendant[Models.PMF.Organs.GenericOrgan]("Stem") for plant in self.Plants]
            self.Spikes = [plant.FindDescendant[Models.PMF.Organs.GenericOrgan]("Spike") for plant in self.Plants]

        ps = [s.FindDescendant[Models.AgPasture.PastureSpecies]() for s in self.en.Simulations]
        if ps[0] is not None:
            self.AGPRyegrass = [rg for rg in ps if rg.Name=='AGPRyegrass']
            if len(self.AGPRyegrass) == 0:
                self.AGPRyegrass = None
        else:
            self.AGPRyegrass = None

        self.models = [apsim.APSIMX(m) for m in self.Models]
        self.report = None
        self.fertilize_events = {}
        self.irrigate_events = {}

    def commence(self):
        """Commence the simulation"""
        self.en.Commence()

    def step(self):
        """Proceed with one day"""
        self.en.Step()

    def done(self):
        """Call at the end of simulation"""
        self.en.Done()

    @property
    def today(self):
        """
        Returns
        -------
        np.datetime64
            Current simulation date
        """
        return np.datetime64(self.en.Today.ToString("yyy-MM-dd"))

    @property
    def enddate(self):
        """
        Returns
        -------
        np.datetime64
            Simulation end date
        """
        return np.datetime64(self.en.EndDate.ToString("yyy-MM-dd"))

    def fertilize_on(self, date, fertilizer):
        self.fertilize_events[str(date)] =  fertilizer

    def apply_fertilizer(self, fertilizer):
        if fertilizer.no3 > 0:
            [f.Apply(fertilizer.no3, Models.Fertiliser.Types.NO3N) for f in self.Fertilisers]
        if fertilizer.nh4 > 0:
            [f.Apply(fertilizer.no3, Models.Fertiliser.Types.NH4N) for f in self.Fertilisers]

    def irrigate_on(self, date, amount):
        self.irrigate_events[str(date)] =  amount

    def irrigate(self, amount):
        for wb in self.WaterBalances:
            wb.Water[0] = wb.Water[0] + amount

    def run(self, reportclass = Report):
        """
        Run the simulation, custom class can be used to
        control reporting.

        Parameters
        ----------
        reportclass
            Class to handle reporting, see `farmingpy.apsim.Report`
            for the default implementation

        Returns
        -------
        DataFrame
            Simulated outputs
        """

        reporter = reportclass(self)
        self.commence()
        while self.today <= self.enddate:
            today_str = str(self.today)
            if today_str in self.fertilize_events:
                self.apply_fertilizer(self.fertilize_events[today_str])
            if today_str in self.irrigate_events:
                self.irrigate(self.irrigate_events[today_str])

            self.step()


            reporter.report()
        self.done()
        self.report = reporter
        return self.report.dataframe