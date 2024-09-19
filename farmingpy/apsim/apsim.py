"""
Interface to APSIM simulation models using Python.NET.
"""

from typing import Union
import pythonnet

# Prefer dotnet
try:
    if pythonnet.get_runtime_info() is None:
        pythonnet.load("coreclr")
except:
    print("dotnet not found loading alternate runtime")
    print("Using: pythonnet.get_runtime_info()")
    pythonnet.load()

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

# Try to load from pythonpath and only then look for Model.exe
try:
    clr.AddReference("Models")
except:
    print("Looking for APSIM")
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
from Models.AgPasture import PastureSpecies
from Models.Core import Simulations

class APSIMX():
    """Modify and run Apsim next generation simulation models."""

    def __init__(self, model : Union[str, Simulations], copy=True, out_path=None):
        """
        Parameters
        ----------

        model
            Path to .apsimx file or C# Models.Core.Simulations object
        copy, optional
            If `True` a copy of original simulation will be created on init, by default True.
        out_path, optional
            Path of modified simulation, if `None` will be set automatically.
        """

        self.results = None #: Simulation results as dataframe
        self.Model = None #TopLevel Simulations object
        #self.simulations = None # List of Simulation object
        #self.py_simulations = None
        self.datastore = None
        self.harvest_date = None


        if type(model) == str:
            apsimx_file = model
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

            self._load(self.path)

        elif type(model) == Simulations:
            self.Model = model
            self.datastore = self.Model.FindChild[Models.Storage.DataStore]().FileName
            self._DataStore = self.Model.FindChild[Models.Storage.DataStore]()

        plant = self.Model.FindDescendant[Models.Core.Zone]().Plants[0]
        cultivar = plant.FindChild[Cultivar]()

        # TODO fix this to work with the sown cultivar and
        # accept simulation name as argument
        try:
            self.cultivar_command = self._cultivar_params(cultivar)
        except:
            pass

    @property
    def simulations(self):
        return list(self.Model.FindAllChildren[Models.Core.Simulation]())

    def _load(self, path):
        # When the last argument (init in another thread) is False,
        # models with errors fail to load. More elegant solution would be handle
        # errors like the GUI does.
        # If loading fails the the model has errors -> Use ApsimNG user interface to debug
        self.Model = FileFormat.ReadFromFile[Models.Core.Simulations](path, None, False)
        # This is needed for APSIM ~5/2023, hacky attempt to also support old version
        # TODO catch errors etc.
        try:
            self.Model = self.Model.get_NewModel()
        except:
            pass
        #self.simulations = list(self._Simulation.FindAllChildren[Models.Core.Simulation]())
        #self.py_simulations = [Simulation(s) for s in self.simulations]
        self.datastore = self.Model.FindChild[Models.Storage.DataStore]().FileName
        self._DataStore = self.Model.FindChild[Models.Storage.DataStore]()

    def _reload(self):
        self.save()
        self._load(self.path)

    def save(self, out_path=None):
        """Save the model

        Parameters
        ----------
        out_path, optional
            Path of output .apsimx file, by default `None`
        """
        if out_path is None:
            out_path = self.path
        json = Models.Core.ApsimFile.FileFormat.WriteToString(self.Model)
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
            r = Models.Core.Run.Runner(self.Model, True, False, False, None, runtype)
        else:
            sims = self.find_simulations(simulations)
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

    def clone_simulation(self, target, simulation=None):
        """Clone a simulation and add it to Model

        Parameters
        ----------
        target
            target simulation name
        simulation, optional
            Simulation name to be cloned, of None clone the first simulation in model
        """

        sim = self._find_simulation(simulation)

        clone_sim = Models.Core.Apsim.Clone(sim)
        clone_sim.Name = target
        #clone_zone = clone_sim.FindChild[Models.Core.Zone]()
        #clone_zone.Name = target

        self.Model.Children.Add(clone_sim)
        self._reload()

    def remove_simulation(self, simulation):
        """Remove a simulation from the model

            Parameters
            ----------
            simulation
                The name of the simulation to remove
        """

        sim = self._find_simulation(simulation)
        self.Model.Children.Remove(sim)
        self.save()
        self._load(self.path)

    def clone_zone(self, target, zone, simulation=None):
        """Clone a zone and add it to Model

            Parameters
            ----------
            target
                target simulation name
            zone
                Name of the zone to clone
            simulation, optional
                Simulation name to be cloned, of None clone the first simulation in model
        """

        sim = self._find_simulation(simulation)
        zone = sim.FindChild[Models.Core.Zone](zone)
        clone_zone = Models.Core.Apsim.Clone(zone)
        clone_zone.Name = target
        sim.Children.Add(clone_zone)
        self.save()
        self._load(self.path)

    def find_zones(self, simulation):
        """Find zones from a simulation

            Parameters
            ----------
            simulation
                simulation name

            Returns
            -------
                list of zones as APSIM Models.Core.Zone objects
        """

        sim = self._find_simulation(simulation)
        zones = sim.FindAllDescendants[Models.Core.Zone]()
        return list(zones)

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
        for sim in self.find_simulations(simulations):
            zone = sim.FindChild[Models.Core.Zone]()
            cultivar = zone.Plants[0].FindChild[Models.PMF.Cultivar]()
            if clear:
                params = parameters
            else:
                params = self._cultivar_params(cultivar)
                params.update(parameters)
            cultivar.Command = [f"{k}={v}" for k,v in params.items()]
            self.cultivar_command = params

    def print_cultivar(self, simulation=None):
        """Print current cultivar paramaters, can be copied to APSIM user interface

        Parameters
        ----------
        simulation, optional
                Simulation name to be cloned, of None clone the first simulation in model
        """
        sim = self._find_simulation(simulation)
        zone = sim.FindChild[Models.Core.Zone]()
        cultivar = zone.Plants[0].FindChild[Models.PMF.Cultivar]()
        print('\n'.join(list(cultivar.Command)))

    def get_default_phenological_parameters(self, simulation=None):
        """
        Return all default parameters for a PMF crop in the simulation

        Parameters
        ----------
        simulation, optional
                Simulation name to be cloned, of None clone the first simulation in model

        Returns
        -------
            dictionary of parameters with default values
        """

        sim = self._find_simulation(simulation)
        phenology = sim.FindDescendant[Models.PMF.Phen.Phenology]()
        targets = {}
        for ch in phenology.FindAllDescendants[Models.Functions.Constant]():
            pth = ch.FullPath.split("Phenology.")[1]
            targets[f"[Phenology].{pth}"] = ch.Value()
        return targets

    def show_management(self, simulations=None):
        """Show management

        Parameters
        ----------
        simulations, optional
            List of simulation names to update, if `None` show all simulations.
        """
        for sim in self.find_simulations(simulations):
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
        for sim in self.find_simulations(simulations):
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

    def get_agpasture_crops(self, simulations = None):
        """Get AgPasture crops from simulations.

        Parameters
        ----------
        start_date, optional
            Start date as string, by default `None`
        end_date, optional
            End date as string, by default `None`
        simulations, optional
            List of simulation names to update, if `None` get from all simulations

        Returns
        ----
            List of PastureSpecies (C# class exposed trough pythonnet)
        """
        species = []
        for sim in self.find_simulations(simulations):
            species += sim.FindAllDescendants[PastureSpecies]()
        return species

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
        for sim in self.find_simulations(simulations):
            clock = sim.FindChild[Models.IClock]()
            if start_date is not None:
                #clock.End = DateTime(start_time.year, start_time.month, start_time.day, 0, 0, 0)
                clock.StartDate = DateTime.Parse(start_date)
            if end_date is not None:
                #clock.End = DateTime(end_time.year, end_time.month, end_time.day, 0, 0, 0)
                clock.EndDate = DateTime.Parse(end_date)

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
        for sim in self.find_simulations(simulations):
            clock = sim.FindChild[Models.IClock]()
            st = clock.StartDate
            et = clock.EndDate
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
        for sim in self.find_simulations(simulations):
            weathers = sim.FindAllDescendants[Weather]()
            for weather in weathers:
                weather.FileName = weather_file

    def show_weather(self):
        """Show weather file for all simulations"""
        for weather in self.Model.FindAllDescendants[Weather]():
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
        simulations = self.find_simulations(simulations)
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

    def find_physical_soil(self, simulation = None):
        """Find physical soil

        Parameters
        ----------
        simulation, optional
            Simulation name, if `None` use the first simulation.
        Returns
        -------
            APSIM Models.Soils.Physical object
        """

        sim = self._find_simulation(simulation)
        soil = sim.FindDescendant[Soil]()
        psoil = soil.FindDescendant[Physical]()
        return psoil

    # Find a list of simulations by name
    def find_simulations(self, simulations = None):
        """Find simulations by name

        Parameters
        ----------
        simulations, optional
            List of simulation names to find, if `None` return all simulations
        Returns
        -------
            list of APSIM Models.Core.Simulation objects
        """

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
        psoil = self.find_physical_soil(simulation)
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
        for sim in self.find_simulations(simulations):
            psoil = sim.FindDescendant[Physical]()
            psoil.DUL = dul
            self._fix_crop_ll(sim.Name)

    # Set crop LL to LL15 and make sure it's below DUL in all layers
    def _fix_crop_ll(self, simulation):
        tmp_cll = self.get_crop_ll(simulation)
        dul = self.get_dul(simulation)
        ll15 = self.get_ll15(simulation)
        for j in range(len(tmp_cll)):
            if tmp_cll[j] > dul[j]:
                tmp_cll[j] = dul[j] - 0.02
        for j in range(len(tmp_cll)):
            tmp_cll[j] = ll15[j]

        self.set_crop_ll(tmp_cll, simulation)


    def _fill_layer(self, p, N_layers):
        ns = len(p)
        if ns == N_layers:
            return p
        else:
            pfill = np.repeat(p[-1], N_layers -ns)
            return np.concatenate([p, pfill])

    def set_soil(self, soildf, simulations=None):
        """Set soil properties using a DataFrame

        Parameters
        ----------
        soildf
            DataFrame with column names matching the parameter to be set. Soil
            will be filled to have the same depth as current soil in the model.
            cf. `get_soil`.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """
        csoil = self.get_soil(simulations)
        N_layers = csoil.shape[0]
        for column in soildf:
            col = column.lower()
            p = soildf[column].to_numpy()
            new = self._fill_layer(p, N_layers)
            if col == "sat":
                self.set_sat(new, simulations)
            if col in ["fc_10", "fc", "dul"]:
                self.set_dul(new, simulations)
            if col in ["wp", "pwp", "ll15"]:
                self.set_ll15(new, simulations)
            if col in ["nh4", "initial nh4"]:
                self.set_initial_nh4(new, simulations)
            if col in ["no3", "initial no3"]:
                self.set_initial_no3(new, simulations)
            if col in ["bd", "bulk density"]:
                self.set_bd(new, simulations)
            if col in ["swcon"]:
                self.set_swcon(new, simulations)
            if col in["ksat", "ksat_mm"]:
                self.set_ksat(new, simulations)
            if col in["sw"]:
                self.set_sw(new, simulations)

        #SW can't exceed SAT
        csoil = self.get_soil(simulations)
        self.set_sw(np.min(csoil[["SAT", "SW"]], axis=1), simulations)

    def set_sat(self, sat, simulations=None):
        """Set soil saturated water content (SAT)

        Parameters
        ----------
        sat
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """

        for sim in self.find_simulations(simulations):
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

        psoil = self.find_physical_soil(simulation)
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
        psoil = self.find_physical_soil(simulation)
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
        for sim in self.find_simulations(simulations):
            psoil = sim.FindDescendant[Physical]()
            psoil.LL15 = ll15
            psoil.AirDry = ll15
            self._fix_crop_ll(sim.Name)

    def set_bd(self, bd, simulations=None):
        """Set soil bulk density

        Parameters
        ----------
        bd
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """

        for sim in self.find_simulations(simulations):
            psoil = sim.FindDescendant[Physical]()
            psoil.BD = bd

    def get_bd(self, simulation=None):
        """Get soil bulk density

        Parameters
        ----------
        simulation, optional
            Simulation name.
        Returns
        -------
            Array of BD values
        """
        psoil = self.find_physical_soil(simulation)
        return np.array(psoil.BD)

    def set_ksat(self, ksat, simulations=None):
        """Set saturated hydraulic conductivity of soil mm/day

        Parameters
        ----------
        bd
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """

        for sim in self.find_simulations(simulations):
            psoil = sim.FindDescendant[Physical]()
            psoil.KS = ksat

    def get_ksat(self, simulation=None):
        """Get saturated hydraulic conductivity of soil mm/day

        Parameters
        ----------
        simulation, optional
            Simulation name.
        Returns
        -------
            Array of BD values
        """
        psoil = self.find_physical_soil(simulation)
        return np.array(psoil.KS)

    def set_sw(self, sw, simulations=None):
        """Set soil water content

        Parameters
        ----------
        bd
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """

        for sim in self.find_simulations(simulations):
            psoil = sim.FindDescendant[Physical]()
            psoil.SW = sw

    def get_sw(self, simulation=None):
        """Get soil water content

        Parameters
        ----------
        simulation, optional
            Simulation name.
        Returns
        -------
            Array of BD values
        """
        psoil = self.find_physical_soil(simulation)
        return np.array(psoil.SW)


    def set_swcon(self, swcon, simulations=None):
        """Set soil water conductivity (SWCON) constant for each soil layer.

        Parameters
        ----------
        swcon
            Collection of values, has to be the same length as existing values.
        simulations, optional
            List of simulation names to update, if `None` update all simulations
        """

        for sim in self.find_simulations(simulations):
            wb = sim.FindDescendant[Models.WaterModel.WaterBalance]()
            wb.SWCON = swcon

    def get_swcon(self, simulation=None):
        """Get soil water conductivity (SWCON) constant for each soil layer.

        Parameters
        ----------
        simulation, optional
            Simulation name.
        Returns
        -------
            Array of SWCON values
        """
        sim = self._find_simulation(simulation)
        wb = sim.FindDescendant[Models.WaterModel.WaterBalance]()
        return np.array(wb.SWCON)

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

        psoil = self.find_physical_soil(simulation)
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

        for sim in self.find_simulations(simulations):
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
        psoil = self.find_physical_soil(simulation)
        depth = psoil.Depth


        return pd.DataFrame({"Depth" : depth, "LL15" : ll15, "DUL" : dul,
                    "SAT" : sat, "Crop LL" : cll,
                    "Bulk density": self.get_bd(simulation),
                    "Ksat" : self.get_ksat(simulation),
                    "SW" : self.get_sw(simulation),
                    "SWCON" : self.get_swcon(simulation),
                    "Initial NO3" : self.get_initial_no3(simulation),
                    "Initial NH4" : self.get_initial_nh4(simulation)})

    def _find_solute(self, solute, simulation=None):
        sim = self._find_simulation(simulation)
        solutes = sim.FindAllDescendants[Models.Soils.Solute]()
        return [s for s in solutes if s.Name == solute][0]

    def _get_initial_values(self, name, simulation):
        s = self._find_solute(name, simulation)
        return np.array(s.InitialValues)

    def _set_initial_values(self, name, values, simulations):
        sims = self.find_simulations(simulations)
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
















