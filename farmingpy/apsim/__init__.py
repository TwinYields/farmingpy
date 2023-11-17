from .apsim import APSIMX

# This will fail without a custom build of APSIM
# TODO catch only C# import errors
try:
    from .optimizer import SoilOptimizer, PhenologyOptimizer, OptimizerBase
    from .ensemble import APSIMXEnsemble, Report
except:
    pass