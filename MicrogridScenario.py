"""
MicrogridScenario.py

This Python class contains methods and attributes vital for completing the scenario analysis.
"""

__author__ = 'Halley Nathwani, Evan Giarta, Thien Nygen'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

import storagevet.ValueStreams as ValueStreams
from MicrogridValueStreams.Reliability import Reliability
from MicrogridDER.BatterySizing import BatterySizing
from MicrogridDER.CAESSizing import CAESSizing
from MicrogridDER.PVSizing import PVSizing
from MicrogridDER.ICESizing import ICESizing
from MicrogridDER.LoadControllable import ControllableLoad
from storagevet.Scenario import Scenario
from CBA import CostBenefitAnalysis
from MicrogridPOI import MicrogridPOI
from MicrogridServiceAggregator import MicrogridServiceAggregator

import logging

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class MicrogridScenario(Scenario):
    """ A scenario is one simulation run in the model_parameters file.

    """

    def __init__(self, input_tree):
        """ Initialize a scenario with sizing technology and paramsDER

        Args:
            input_tree (Dict): Dict of input attributes such as time_series, params, and monthly_data

        """
        Scenario.__init__(self, input_tree)

        self.value_stream_input_map.update({'Reliability': input_tree.Reliability})

        u_logger.info("ScenarioSizing initialized ...")

    def set_up_poi_and_service_aggregator(self):
        """ Initialize the POI and service aggregator with DERs and valuestreams to be evaluated.

        """
        technology_class_map = {
            'CAES': CAESSizing,
            'Battery': BatterySizing,
            'PV': PVSizing,
            'ICE': ICESizing,
            'Load': ControllableLoad
        }

        value_stream_class_map = {
            'Deferral': ValueStreams.Deferral,
            'DR': ValueStreams.DemandResponse,
            'RA': ValueStreams.ResourceAdequacy,
            'Backup': ValueStreams.Backup,
            'Volt': ValueStreams.VoltVar,
            'User': ValueStreams.UserConstraints,
            'DA': ValueStreams.DAEnergyTimeShift,
            'FR': ValueStreams.FrequencyRegulation,
            'LF': ValueStreams.LoadFollowing,
            'SR': ValueStreams.SpinningReserve,
            'NSR': ValueStreams.NonspinningReserve,
            'DCM': ValueStreams.DemandChargeReduction,
            'retailTimeShift': ValueStreams.EnergyTimeShift,
            'Reliability': Reliability
        }
        # these need to be initialized after opt_agg is created
        self.poi = MicrogridPOI(self.poi_inputs, self.technology_inputs_map, technology_class_map)
        self.service_agg = MicrogridServiceAggregator(self.value_stream_input_map, value_stream_class_map)

    def optimize_problem_loop(self):
        """This function selects on opt_agg of data in time_series and calls optimization_problem on it. We determine if the
        optimization will be sizing and calculate a lifetime project NPV multiplier to pass into the optimization problem

        """
        if self.poi.is_sizing_optimization:
            alpha = CostBenefitAnalysis.annuity_scalar(**self.finance_inputs)
        else:
            alpha = 1

        super().optimize_problem_loop(annuity_scalar=alpha, ignore_der_costs=self.service_agg.post_facto_reliability_only())
