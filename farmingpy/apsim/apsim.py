"""
Interface to APSIM simulation models using Python.NET.
"""

import pythonnet
if pythonnet.get_runtime_info() is None:
    pythonnet.load("coreclr")
import clr
import sys
import numpy as np
import pandas as pd
import shutil
import os
import pathlib
import shutil
import datetime
import warnings


apsim_path = shutil.which("Models")
if apsim_path is not None:
    apsim_path = os.path.split(os.path.realpath(apsim_path))[0]
    sys.path.append(apsim_path)
clr.AddReference("Models")
clr.AddReference("System")

# C# imports
import Models
import Models.Core
import Models.Core.ApsimFile
import Models.Core.Run;
import Models.PMF
import System.IO
import System.Linq
from System.Collections.Generic import *
from System import *

#from Models.Core import Zone, Simulations
from Models.PMF import Cultivar
from Models.Core.ApsimFile import FileFormat
from Models.Climate import Weather
from Models.Soils import Soil, Physical, SoilCrop

class APSIMX():
    """Modify and run Apsim next generation simulation models."""

    def __init__(self, apsimx_file, copy=True, out_path=None):
        """
        Parameters
        ----------
        apsimx_file
            Path to .apsimx file
        copy, optional
            If `True` a copy of original simulation will be created on init, by default True.
        out_path, optional
            Path of modified simulation, if `None` will be set automatically.
        """

        name, ext = os.path.splitext(apsimx_file)
        if copy:
            if out_path is None:
                copy_path = f"{name}_py{ext}"
            else:
                copy_path = out_path
            shutil.copy(apsimx_file, copy_path)
            pathlib.Path(f"{name}.db").unlink(missing_ok=True)
            pathlib.Path(f"{name}.db-shm").unlink(missing_ok=True)
            pathlib.Path(f"{name}.db-wal").unlink(missing_ok=True)
            self.path = copy_path
        else:
            self.path = apsimx_file

        self.results = None #: Simulation results as dataframe
        self._Simulation = None #TopLevel Simulations object
        self.simulations = None # List of Simulation object
        self.py_simulations = None
        self.datastore = None
        self.harvest_date = None

        self._load(self.path)

        plant = self._Simulation.FindDescendant[Models.Core.Zone]().Plants[0]
        cultivar = plant.FindChild[Cultivar]()

        # TODO fix this to work with the sown cultivar and
        # accept simulation name as argument
        try:
            self.cultivar_command = self._cultivar_params(cultivar)
        except:
            pass

    def _load(self, path):
        self._Simulation = FileFormat.ReadFromFile[Models.Core.Simulations](path, None, True)
        # This is needed for APSIM ~5/2023, hacky attempt to also support old version
        # TODO catch errors etc.
        try:
            self._Simulation = self._Simulation.get_NewModel()
        except:
            pass
        self.simulations = list(self._Simulation.FindAllChildren[Models.Core.Simulation]())
        self.py_simulations = [Simulation(s) for s in self.simulations]
        self.datastore = self._Simulation.FindChild[Models.Storage.DataStore]().FileName
        self._DataStore = self._Simulation.FindChild[Models.Storage.DataStore]()

    def save(self, out_path=None):
        """Save the model

        Parameters
        ----------
        out_path, optional
            Path of output .apsimx file, by default `None`
        """
        if out_path is None:
            out_path = self.path
        json = Models.Core.ApsimFile.FileFormat.WriteToString(self._Simulation)
        with open(out_path, "w") as f:
            f.write(json)

    def run(self, simulations=None, clean=True, multithread=True):
        """Run simulations

        Parameters
        ----------
        simulations, optional
            List of simulation names to run, if `None` runs all simulations, by default `None`.
        clean, optional
            If `True` remove existing database for the file before running, by default `True`
        multithread, optional
            If `True` APSIM uses multiple threads, by default `True`
        """
        if multithread:
            runtype = Models.Core.Run.Runner.RunTypeEnum.MultiThreaded
        else:
            runtype = Models.Core.Run.Runner.RunTypeEnum.SingleThreaded

        # Clear old data before running
        self.results=None
        if clean:
            self._DataStore.Dispose()
            pathlib.Path(self._DataStore.FileName).unlink(missing_ok=True)
            self._DataStore.Open()
        if simulations is None:
            r = Models.Core.Run.Runner(self._Simulation, True, False, False, None, runtype)
        else:
            sims = self._find_simulations(simulations)
            # Runner needs C# list
            cs_sims = List[Models.Core.Simulation]()
            for s in sims:
                cs_sims.Add(s)
            r = Models.Core.Run.Runner(cs_sims, True, False, False, None, runtype)
        e = r.Run()
        if (len(e) > 0):
            print(e[0].ToString())
        self.results = self._read_results()

        try:
            self.harvest_date = self.results.loc[self.results.WheatPhenologyCurrentStageName  == 'HarvestRipe',
                                    ["Zone", "ClockToday"]]
        except:
            self.harvest_date = None

    def _read_results(self):
        #df = pd.read_sql_table("Report", "sqlite:///" + self.datastore) # errors with datetime since 5/2023
        df = pd.read_sql_query("select * from Report", "sqlite:///" + self.datastore)
        df = df.rename(mapper=lambda x: x.replace(".", ""), axis=1)
        try:
            # ClockToday has . delimiters on Mac
            df["ClockToday"] = [datetime.datetime.strptime(t.replace(".", ":"), "%Y-%m-%d %H:%M:%S") for t in df.ClockToday]
        except:
            warnings.warn("Unable to parse time format, 'ClockToday' column is still a string")
        return df

    """Convert cultivar command to dict"""
    def _cultivar_params(self, cultivar):
        cmd = cultivar.Command
        params = {}
        for c in cmd:
            if c:
                p, v = c.split("=")
                params[p.strip()] = v.strip()
        return params

    def update_cultivar(self, parameters, simulations=None, clear=False):
        """Update cultivar parameters

        Parameters
        ----------
        parameters
            Parameter = value dictionary of cultivar paramaters to update.
        simulations, optional
            List of simulation names to update, if `None` update all simulations.
        clear, optional
            If `True` remove all existing parameters, by default `False`.
        """
        for sim in self._find_simulations(simulations):
            zone = sim.FindChild[Models.Core.Zone]()
            cultivar = zone.Plants[0].FindChild[Models.PMF.Cultivar]()
            if clear:
                params = parameters
            else:
                params = self._cultivar_params(cultivar)
                params.update(parameters)
            cultivar.Command = [f"{k}={v}" for k,v in params.items()]
            self.cultivar_command = params

    def show_management(self, simulations=None):
        """Show management

        Parameters
        ----------
        simulations, optional
            List of simulation names to update, if `None` show all simulations.
        """
        for sim in self._find_simulations(simulations):
            zone = sim.FindChild[Models.Core.Zone]()
            print("Zone:", zone.Name)
            for action in zone.FindAllChildren[Models.Manager]():
                print("\t", action.Name, ":")
                for param in action.Parameters:
                    print("\t\t", param.Key,":", param.Value)

    def update_management(self, management, simulations=None, reload=True):
        """Update management

        Parameters
        ----------
        management
            Parameter = value dictionary of management paramaters to update. Call
            `show_management` to see current values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations.
        reload, optional
            _description_, by default True
        """
        for sim in self._find_simulations(simulations):
            zone = sim.FindChild[Models.Core.Zone]()
            for action in zone.FindAllChildren[Models.Manager]():
                if action.Name in management:
                    #print("Updating", action.Name)
                    values = management[action.Name]
                    for i in range(len(action.Parameters)):
                        param = action.Parameters[i].Key
                        if param in values:
                            action.Parameters[i] = KeyValuePair[String, String](param, f"{values[param]}")
        # Saved and restored the model to recompile the scripts
        # haven't figured out another way to make it work
        if reload:
            self.save()
            self._load(self.path)

    # Convert CS KeyValuePair to dictionary
    def _kvtodict(self, kv):
        return {kv[i].Key : kv[i].Value for i in range(kv.Count)}

    def get_management(self):
        """Get management of all simulations as dataframe"""
        res = []
        for sim in self.simulations:
            actions = sim.FindAllDescendants[Models.Manager]()
            out = {}
            out["simulation"] = sim.Name
            for action in actions:
                params = self._kvtodict(action.Parameters)
                if "FertiliserType" in params:
                    out[params["FertiliserType"]] = float(params["Amount"])
                if "CultivarName" in params:
                    out["crop"] = params["Crop"]
                    out["cultivar"] = params["CultivarName"]
                    out["plant_population"] = params["Population"]

            if len(out) > 1:
                res.append(out)
        return pd.DataFrame(res)

    def set_dates(self, start_date=None, end_date=None, simulations = None):
        """Set simulation dates

        Parameters
        ----------
        start_date, optional
            Start date as string, by default `None`
        end_date, optional
            End date as string, by default `None`
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """
        for sim in self._find_simulations(simulations):
            clock = sim.FindChild[Models.Clock]()
            if start_date is not None:
                #clock.End = DateTime(start_time.year, start_time.month, start_time.day, 0, 0, 0)
                clock.Start = DateTime.Parse(start_date)
            if end_date is not None:
                #clock.End = DateTime(end_time.year, end_time.month, end_time.day, 0, 0, 0)
                clock.End = DateTime.Parse(end_date)

    def get_dates(self, simulations = None):
        """Get simulation dates
        Parameters
        ----------
        simulations, optional
            List of simulation names to get, if `None` get all simulations
        Returns
        -------
            Dictionary of simulation names with dates
        """
        dates =  {}
        for sim in self._find_simulations(simulations):
            clock = sim.FindChild[Models.Clock]()
            st = clock.Start
            et = clock.End
            dates[sim.Name] = {}
            dates[sim.Name]["start"] = datetime.date(st.Year, st.Month, st.Day)
            dates[sim.Name]["end"] = datetime.date(et.Year, et.Month, et.Day)
        return dates

    def set_weather(self, weather_file, simulations = None):
        """Set simulation weather file

        Parameters
        ----------
        weather_file
            Weather file name, path should be relative to simulation or absolute.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """
        for sim in self._find_simulations(simulations):
            weather = sim.FindChild[Weather]()
            weather.FileName = weather_file

    def show_weather(self):
        """Show weather file for all simulations"""
        for weather in self._Simulation.FindAllDescendants[Weather]():
            print(weather.FileName)

    def set_report(self, report, simulations = None):
        """Set APSIM report

        Parameters
        ----------
        report
            New report string.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """
        simulations = self._find_simulations(simulations)
        for sim in simulations:
            r = sim.FindDescendant[Models.Report]()
            r.set_VariableNames(report.strip().splitlines())

    def get_report(self, simulation = None):
        """Get current report string

        Parameters
        ----------
        simulation, optional
            Simulation name, if `None` use the first simulation.
        Returns
        -------
            List of report lines.
        """
        sim = self._find_simulation(simulation)
        report = list(sim.FindAllDescendants[Models.Report]())[0]
        return list(report.get_VariableNames())

    def _find_physical_soil(self, simulation = None):
        sim = self._find_simulation(simulation)
        soil = sim.FindDescendant[Soil]()
        psoil = soil.FindDescendant[Physical]()
        return psoil

    # Find a list of simulations by name
    def _find_simulations(self, simulations = None):
        if simulations is None:
            return self.simulations
        if type(simulations) == str:
            simulations = [simulations]
        sims = []
        for s in self.simulations:
            if s.Name in simulations:
                sims.append(s)
        if len(sims) == 0:
            print("Not found!")
        else:
            return sims

    # Find a single simulation by name
    def _find_simulation(self, simulation = None):
        if simulation is None:
            return self.simulations[0]
        sim = None
        for s in self.simulations:
            if s.Name == simulation:
                sim = s
                break
        if sim is None:
            print("Not found!")
        else:
            return sim

    def get_dul(self, simulation=None):
        """Get soil dry upper limit (DUL)

        Parameters
        ----------
        simulation, optional
            Simulation name.
        Returns
        -------
            Array of DUL values
        """
        psoil = self._find_physical_soil(simulation)
        return np.array(psoil.DUL)

    def set_dul(self, dul, simulations=None):
        """Set soil dry upper limit (DUL)

        Parameters
        ----------
        dul
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """
        for sim in self._find_simulations(simulations):
            psoil = sim.FindDescendant[Physical]()
            psoil.DUL = dul
            self._fix_crop_ll(sim.Name)

    # Make sure that crop ll is below DUL in all layers
    def _fix_crop_ll(self, simulation):
        tmp_cll = self.get_crop_ll()
        dul = self.get_dul(simulation)
        for j in range(len(tmp_cll)):
            if tmp_cll[j] > dul[j]:
                tmp_cll[j] = dul[j] - 0.01
        self.set_crop_ll(tmp_cll, simulation)

    def set_sat(self, sat, simulations=None):
        """Set soil saturated water content (SAT)

        Parameters
        ----------
        sat
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """

        for sim in self._find_simulations(simulations):
            psoil = sim.FindDescendant[Physical]()
            psoil.SAT = sat
            psoil.SW = psoil.DUL

    def get_sat(self, simulation=None):
        """Get soil saturated water content (SAT)

        Parameters
        ----------
        simulation, optional
            Simulation name.
        Returns
        -------
            Array of SAT values
        """

        psoil = self._find_physical_soil(simulation)
        return np.array(psoil.SAT)

    def get_ll15(self, simulation=None):
        """Get soil water content lower limit (LL15)

        Parameters
        ----------
        simulation, optional
            Simulation name.
        Returns
        -------
            Array of LL15 values
        """
        psoil = self._find_physical_soil(simulation)
        return np.array(psoil.LL15)

    def set_ll15(self, ll15, simulations=None):
        """Set soil water content lower limit (LL15)

        Parameters
        ----------
        ll15
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """
        for sim in self._find_simulations(simulations):
            psoil = sim.FindDescendant[Physical]()
            psoil.LL15 = ll15

    def get_crop_ll(self, simulation=None):
        """Get crop lower limit

        Parameters
        ----------
        simulation, optional
            Simulation name.
        Returns
        -------
            Array of values
        """

        psoil = self._find_physical_soil(simulation)
        sc = psoil.FindChild[SoilCrop]()
        return np.array(sc.LL)

    def set_crop_ll(self, ll, simulations=None):
        """Set crop lower limit

        Parameters
        ----------
        ll
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """

        for sim in self._find_simulations(simulations):
            psoil = sim.FindDescendant[Physical]()
            sc = psoil.FindChild[SoilCrop]()
            sc.LL = ll

    def get_soil(self, simulation=None):
        """Get soil definition as dataframe

        Parameters
        ----------
        simulation, optional
            Simulation name.
        Returns
        -------
            Dataframe with soil definition
        """
        sat = self.get_sat(simulation)
        dul = self.get_dul(simulation)
        ll15 = self.get_ll15(simulation)
        cll = self.get_crop_ll(simulation)
        psoil = self._find_physical_soil(simulation)
        depth = psoil.Depth
        return pd.DataFrame({"Depth" : depth, "LL15" : ll15, "DUL" : dul, "SAT" : sat, "Crop LL" : cll,
                    "Initial NO3" : self.get_initial_no3(),
                    "Initial NH4" : self.get_initial_nh4()})

    def _find_solute(self, solute, simulation=None):
        sim = self._find_simulation(simulation)
        solutes = sim.FindAllDescendants[Models.Soils.Solute]()
        return [s for s in solutes if s.Name == solute][0]

    def _get_initial_values(self, name, simulation):
        s = self._find_solute(name, simulation)
        return np.array(s.InitialValues)

    def _set_initial_values(self, name, values, simulations):
        sims = self._find_simulations(simulations)
        for sim in sims:
            s = self._find_solute(name, sim.Name)
            s.InitialValues = values

    def get_initial_no3(self, simulation=None):
        """Get soil initial NO3 content"""
        return self._get_initial_values("NO3", simulation)

    def set_initial_no3(self, values, simulations=None):
        """Set soil initial NO3 content

        Parameters
        ----------
        values
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """
        self._set_initial_values("NO3", values, simulations)

    def get_initial_nh4(self, simulation=None):
        """Get soil initial NH4 content"""
        return self._get_initial_values("NH4", simulation)

    def set_initial_nh4(self, values, simulations=None):
        """Set soil initial NH4 content

        Parameters
        ----------
        values
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """

        self._set_initial_values("NH4", values, simulations)

    def get_initial_urea(self, simulation=None):
        """Get soil initial urea content"""
        return self._get_initial_values("Urea", simulation)

    def set_initial_urea(self, values, simulations=None):
        """Set soil initial urea content

        Parameters
        ----------
        values
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """


        self._set_initial_values("Urea", values, simulations)


class Simulation(object):

    def __init__(self, simulation):
        self.simulation = simulation
        self.zones = [Zone(z) for z in simulation.FindAllChildren[Models.Core.Zone]()]

    def find_physical_soil(self):
        soil = self.simulation.FindDescendant[Soil]()
        psoil = soil.FindDescendant[Physical]()
        return psoil

    # TODO should these be linked to zones instead?
    def get_dul(self):
        psoil = self.find_physical_soil()
        return np.array(psoil.DUL)

    def set_dul(self, dul):
        psoil = self.find_physical_soil()
        psoil.DUL = dul

class Zone(object):

    def __init__(self, zone):
        self.zone = zone
        self.name = zone.Name
        self.soil = self.zone.FindDescendant[Soil]()
        self.physical_soil = self.soil.FindDescendant[Physical]()

    # TODO should these be linked to zones instead?
    @property
    def dul(self):
        return np.array(self.physical_soil.DUL)

    @dul.setter
    def dul(self, dul):
        self.physical_soil.DUL = dul
















