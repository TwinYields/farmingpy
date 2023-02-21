"""
Optimizers for APSIM simulation models.
"""

import numpy as np
import pandas as pd
import nlopt

class OptimizerBase(object):

    def __init__(self, model, params, obs_yield, obs_lai, harvest_date, zone_names = None, multithread = True):
       
        obs_yield.rename(str.lower, axis=1, inplace=True)
        obs_lai.rename(str.lower, axis=1, inplace=True)

        obs_lai.columns = obs_lai.columns.str.replace(r".*lai.*", "obs_lai", regex=True)
        obs_yield.columns = obs_yield.columns.str.replace(r".*yield.*", "obs_yield", regex=True)

        if zone_names is not None:
            if type(zone_names) == str:
                zone_names = [zone_names]
            obs_yield = obs_yield[obs_yield.zone.apply(lambda x : x in zone_names)]
            obs_lai = obs_lai[obs_lai.zone.apply(lambda x : x in zone_names)]

        self.zone_names = zone_names
        self.model = model
        self.obs_lai = obs_lai
        self.obs_yield = obs_yield
        self.harvest_date = harvest_date
        self.multithread = multithread

        self.optvars = sorted(params.keys())
        self.params = params.copy()
        self.N = len(self.optvars)

        self._iter = 0
        self.print_interval = 10
        self.log_level = 1

    def _print(self, clai, cyield, charvest, cost):
        if (self._iter % self.print_interval) == 0:
            print(f"\tIteration {self._iter}, cost: LAI {clai:.3f} Yield {cyield:.3f}, Harvest {charvest:.3f}, Total {cost:.3f}")

    def _merge_sim_obs(self):
        df = self.model.results
        df.rename(str.lower, axis=1, inplace=True)
        df["sim_lai"] = df["lai"]
        sim_lai = df[["clocktoday", "zone", "sim_lai"]]
        sim_lai = self.obs_lai.merge(sim_lai, how="left")

        sim_yield = df[["zone", "yield", "lai"]].groupby("zone", as_index=False).agg(np.max)
        sim_yield.rename({"yield" : "sim_yield", "lai" : "sim_lai"}, axis=1, inplace=True)
        sim_yield =  sim_yield.merge(self.obs_yield)

        htime = self.model.harvest_date
        #print(htime)
        if htime is None or htime.shape[0] == 0:
            print("Out of range")
            harvest_time = df["clocktoday"].max()
        else:
            harvest_time = htime.iloc[0]["ClockToday"]
        return sim_yield, sim_lai, harvest_time

    def _calculate_cost(self, p, grad = []):
        self._step(p, grad)
        self._run()
        self._iter += 1
        sim_yield, lai_df, harvest_time = self._merge_sim_obs()
        # Calculate errors
        e_lai = (lai_df.sim_lai - lai_df.obs_lai)
        e_yield = (sim_yield.sim_yield - sim_yield.obs_yield)
        e_harvest = (harvest_time - self.harvest_date).total_seconds() /(24*60*60)
        # Apply cost function
        return self._scaled_lsq(e_lai, e_yield, e_harvest)

    def _scaled_lsq(self, e_lai, e_yield, e_harvest):
        cost_lai = np.mean((e_lai)**2)/np.max(self.obs_lai["obs_lai"])
        cost_yield = np.mean(np.abs(e_yield))/np.max(self.obs_yield["obs_yield"].max())
        cost_harvest = abs(e_harvest) / 7 # Error or 1 week in same scale with other variables
        total_cost =  cost_lai + cost_yield + cost_harvest
        if self.log_level > 0:
            self._print(cost_lai, cost_yield, cost_harvest, total_cost)
        return total_cost

    def _rand_start(self, r):
        l, h = r
        vs = np.linspace(l,h, 10)
        idx = np.random.randint(0, len(vs))
        return vs[idx]

    @property
    def optimized_data(self):
        return self._merge_sim_obs()

    @property
    def optimized_parameters(self):
        return pd.DataFrame({self.optvars[i] :self.opt_values[i] for i in range(self.N)}, index=[0])


    def _print_optimized(self):
        pass

    def _step(self, p, grad=[]):
        pass

    def _run(self):
        self.model.run(self.zone_names, clean=False, multithread = self.multithread)

    def optimize(self, alg = nlopt.GN_DIRECT_L, maxeval = 5):
        """Run the optimizer

        Parameters
        ----------
        alg, optional
            nlopt algorithm to use, by default nlopt.GN_DIRECT_L
        maxeval, optional
            Maximum number of iterations, by default 5
        """
        if self.log_level > 0:
            print(f"Optimizing {self.N} parameters, max {maxeval} iterations")
        opt = nlopt.opt(alg, self.N)
        opt.set_min_objective(self._calculate_cost)

        opt.set_lower_bounds([self.params[v][0] for v in self.optvars])
        opt.set_upper_bounds([self.params[v][1] for v in self.optvars])

        opt.set_maxeval(maxeval)
        init = [self._rand_start(self.params[v]) for v in self.optvars]

        self.opt = opt
        self.opt_values = opt.optimize(init)

        self._step(self.opt_values)
        if self.log_level > 0:
            print(f"Done after {self._iter} iterations")
            self._print_optimized()


class SoilOptimizer(OptimizerBase):
        """Optimize soil paramters."""

        def __init__(self, model, params, obs_yield, obs_lai, harvest_date, zone_names=None, multithread=True):
            
            """Optimize model parameters

            Parameters
            ----------
            model
                APSIMX object
            params
                Dictionary of soil parameters to optimize with allowed ranges. 
                The range indicates difference to original values. 
                e.g. ``{"ll15" : (-0.01, 0.05), "dul" : (0.0, -0.05)}``. Supported parameters are: 
                ``ll15, dul, no3, nh4, urea``. The same change is applied to all soil layers.
            obs_yield
                Dataframe with observed yield for each zone.
            obs_lai
                Dataframe with observed LAI for each zone.
            harvest_date
                Harvest date as datetime
            zone_names, optional
                Name of zones to optimize.
            multithread, optional
                Allow APSIM to use multiple threads, by default True
            """

            super(SoilOptimizer, self).__init__(model, params, obs_yield, obs_lai, harvest_date, zone_names, multithread)
            

            #Save starting values
            self.ll15 = self.model.get_ll15().copy()
            self.dul = self.model.get_dul().copy()
            self.sat = self.model.get_sat().copy()
            self.cll = self.model.get_crop_ll().copy()
            self.initial_no3 = self.model.get_initial_no3().copy()
            self.initial_nh4 = self.model.get_initial_nh4().copy()
            self.initial_urea = self.model.get_initial_urea().copy()

        def _print_optimized(self):
            print(str.join(", ", [f"{self.optvars[i]} = {self.opt_values[i]:.2f}" for i in range(self.N)]))

        def _step(self, p, grad=[]):
            #for zone in self.zone_names:
            zones = self.zone_names
            for i in range(self.N):
                v = self.optvars[i]
                if v == "ll15":
                    self.model.set_ll15(self.ll15 + p[i], zones)
                if v == "dul":
                    new_dul = self.dul + p[i]
                    self.model.set_dul(new_dul, zones)
                    self.model.set_sat(self.sat + p[i], zones)
                if v.lower() == "no3":
                    self.model.set_initial_no3(self.initial_no3 + p[i])
                if v.lower() == "nh4":
                    self.model.set_initial_nh4(self.initial_nh4 + p[i])
                if v.lower() == "urea":
                    self.model.set_initial_urea(self.initial_urea + p[i])


class PhenologyOptimizer(OptimizerBase):
    """Optimize cultivar parameters."""
    
    def __init__(self, model, params, obs_yield, obs_lai, harvest_date, zone_names=None, multithread=True):
        """Optimize model parameters

        Parameters
        ----------
        model
            APSIMX object
        params
            Dictionary of cultivar parameters to optimize with allowed ranges. 
        obs_yield
            Dataframe with observed yield for each zone.
        obs_lai
            Dataframe with observed LAI for each zone.
        harvest_date
            Harvest date as datetime
        zone_names, optional
            Name of zones to optimize.
        multithread, optional
            Allow APSIM to use multiple threads, by default True
        """
        super(PhenologyOptimizer, self).__init__(model, params, obs_yield, obs_lai, harvest_date, zone_names, multithread)

    def _step(self, p, grad=[]):
        c = {self.optvars[i] : p[i] for i in range(self.N)}
        self.model.update_cultivar(c, self.zone_names)

    def _print_optimized(self):
        print(str.join("\n", [f"{self.optvars[i]} = {self.opt_values[i]:.2f}" for i in range(self.N)]))
    

