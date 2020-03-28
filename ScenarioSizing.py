"""
ScenarioSizing.py

This Python class contains methods and attributes vital for completing the scenario analysis.
"""

__author__ = 'Halley Nathwani, Evan Giarta, Thien Nygen'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version


import storagevet

from TechnologiesDER.BatterySizing import BatterySizing
from TechnologiesDER.CAESSizing import CAESSizing
from TechnologiesDER.CurtailPVSizing import CurtailPVSizing
from TechnologiesDER.ICESizing import ICESizing

from storagevet.Scenario import Scenario

from cbaDER import CostBenefitAnalysis

import logging

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')

# constant names of available value streams
CLASS_MAP = {
    'battery': BatterySizing,
    'caes': CAESSizing,
    'ice': ICESizing,
    'pv': CurtailPVSizing
}

class ScenarioSizing(Scenario):
    """ A scenario is one simulation run in the model_parameters file.

    """

    def __init__(self, input_tree):
        """ Initialize a scenario with sizing technology and paramsDER

        Args:
            input_tree (Dict): Dict of input attributes such as time_series, params, and monthly_data

        """
        Scenario.__init__(self, input_tree)

        sizing_tech_map = {}
        for name, tech_object in self.active_technology_inputs_map.items():
            sizing_tech_map.update({name: CLASS_MAP[name]})

        Scenario.init_POI(self, input_tree, sizing_tech_map)
        Scenario.activate_controller(self)

        u_logger.info("ScenarioSizing initialized ...")

    def init_financials(self, finance_inputs):
        """ Initializes the financial class with a copy of all the price data from timeseries, the tariff data, and any
         system variables required for post optimization analysis.

         Args:
             finance_inputs (Dict): Financial inputs

        """

        self.financials = CostBenefitAnalysis(finance_inputs)
        u_logger.info("Finished adding Financials...")

    def optimize_problem_loop(self, annuity_scalar=1):
        """This function selects on opt_agg of data in time_series and calls optimization_problem on it. We determine if the
        optimization will be sizing and calculate a lifetime project NPV multiplier to pass into the optimization problem

        Args:
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                the entire project lifetime (only to be set iff sizing)

        """
        if self.poi.sizing_optimization:
            annuity_scalar = self.financials.annuity_scalar(self.start_year, self.end_year, self.opt_years)

        super().optimize_problem_loop(annuity_scalar)
