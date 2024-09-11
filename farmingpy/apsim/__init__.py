try:
    from .apsim import APSIMX
except:
    print("Failed to load APSIMX, see https://github.com/TwinYields/farmingpy?tab=readme-ov-file#apsim-interface")

from .metfiles import read_met

# This will fail without a custom build of APSIM
# TODO catch only C# import errors
try:
    from .optimizer import SoilOptimizer, PhenologyOptimizer, OptimizerBase
    from .ensemble import APSIMXEnsemble, Report, Fertilizer
except:
    pass