"""
Fed-Batch MPC Optimizer - Main Package

The world's first publicly available Model Predictive Control simulator
for therapeutic antibody production.
"""

__version__ = "0.1.0"
__author__ = "Kemal Yaylali"
__license__ = "MIT"

from src.models.cho_kinetics import CHOCellModel
from src.models.bioreactor import Bioreactor
from src.models.parameters import CHOKineticParameters, get_scale_parameters

from src.control.fixed_recipe import FixedFeedingStrategy, FixedRecipeLibrary
from src.control.exponential_feeding import ExponentialFeedingStrategy, ExponentialRecipeLibrary
from src.control.mpc_controller import MPCController, SimplifiedMPCController

__all__ = [
    'CHOCellModel',
    'Bioreactor',
    'CHOKineticParameters',
    'get_scale_parameters',
    'FixedFeedingStrategy',
    'FixedRecipeLibrary',
    'ExponentialFeedingStrategy',
    'ExponentialRecipeLibrary',
    'MPCController',
    'SimplifiedMPCController',
]
