from .apsim import APSIMX
from .optimizer import SoilOptimizer, PhenologyOptimizer, OptimizerBase
# This will fail without a custom build of APSIM
# TODO catch only C# import errors
try:
    from .ensemble import APSIMXEnsemble, Report
except:
    pass