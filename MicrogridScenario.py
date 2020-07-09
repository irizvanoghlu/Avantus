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

from MicrogridValueStreams.Reliability import Reliability
from MicrogridDER.Battery import Battery
from MicrogridDER.CAES import CAES
from MicrogridDER.PV import PV
from MicrogridDER.ICE import ICE
from MicrogridDER.LoadControllable import ControllableLoad
from storagevet.ValueStreams.DAEnergyTimeShift import DAEnergyTimeShift
from storagevet.ValueStreams.FrequencyRegulation import FrequencyRegulation
from storagevet.ValueStreams.NonspinningReserve import NonspinningReserve
from storagevet.ValueStreams.DemandChargeReduction import DemandChargeReduction
from storagevet.ValueStreams.EnergyTimeShift import EnergyTimeShift
from storagevet.ValueStreams.SpinningReserve import SpinningReserve
from storagevet.ValueStreams.Backup import Backup
from storagevet.ValueStreams.Deferral import Deferral
from storagevet.ValueStreams.DemandResponse import DemandResponse
from storagevet.ValueStreams.ResourceAdequacy import ResourceAdequacy
from storagevet.ValueStreams.UserConstraints import UserConstraints
from storagevet.ValueStreams.VoltVar import VoltVar
from storagevet.ValueStreams.LoadFollowing import LoadFollowing
from storagevet.Scenario import Scenario
from CBA import CostBenefitAnalysis
from MicrogridPOI import MicrogridPOI
from MicrogridServiceAggregator import MicrogridServiceAggregator
import time
import pandas as pd

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
            'CAES': CAES,
            'Battery': Battery,
            'PV': PV,
            'ICE': ICE,
            'Load': ControllableLoad
        }

        value_stream_class_map = {
            'Deferral': Deferral,
            'DR': DemandResponse,
            'RA': ResourceAdequacy,
            'Backup': Backup,
            'Volt': VoltVar,
            'User': UserConstraints,
            'DA': DAEnergyTimeShift,
            'FR': FrequencyRegulation,
            'LF': LoadFollowing,
            'SR': SpinningReserve,
            'NSR': NonspinningReserve,
            'DCM': DemandChargeReduction,
            'retailTimeShift': EnergyTimeShift,
            'Reliability': Reliability
        }
        # these need to be initialized after opt_agg is created
        self.poi = MicrogridPOI(self.poi_inputs, self.technology_inputs_map, technology_class_map)
        self.service_agg = MicrogridServiceAggregator(self.value_stream_input_map, value_stream_class_map)

    def optimize_problem_loop(self, **kwargs):
        """ This function selects on opt_agg of data in time_series and calls optimization_problem on it.

        Args:
            **kwargs: allows child classes to pass in additional arguments to set_up_optimization

        """
        alpha = 1
        if self.poi.is_sizing_optimization:
            self.check_sizing_conditions()
            # calculate the annuity scalar that will convert any yearly costs into a present value
            alpha = CostBenefitAnalysis.annuity_scalar(**self.finance_inputs)

        if self.service_agg.is_deferral_only() or self.service_agg.post_facto_reliability_only():
            u_logger.info("Only active Value Stream is Deferral or post facto only, so not optimizations will run...")
            return True

        # calculate and check that system requirement set by value streams can be met
        system_requirements = self.check_system_requirements()

        u_logger.info("Starting optimization loop")
        for opt_period in self.optimization_levels.predictive.unique():

            # used to select rows from time_series relevant to this optimization window
            mask = self.optimization_levels.predictive == opt_period

            # apply past degradation in ESS objects (NOTE: if no degredation module applies to specific ESS tech, then nothing happens)
            for der in self.poi.der_list:
                if der.technology_type == "Energy Storage System":
                    der.apply_past_degredation(opt_period)

            if self.verbose:
                print(f"{time.strftime('%H:%M:%S')} Running Optimization Problem starting at {self.optimization_levels.loc[mask].index[0]} hb")

            # setup + run optimization then return optimal objective costs
            functions, constraints = self.set_up_optimization(mask, system_requirements,
                                                              annuity_scalar=alpha,
                                                              ignore_der_costs=self.service_agg.post_facto_reliability_only())
            objective_values = self.run_optimization(functions, constraints, opt_period)

            # calculate degradation in ESS objects (NOTE: if no degredation module applies to specific ESS tech, then nothing happens)
            sub_index = self.optimization_levels.loc[mask].index
            for der in self.poi.der_list:
                if der.technology_type == "Energy Storage System":
                    der.calc_degradation(opt_period, sub_index[0], sub_index[-1])

            # then add objective expressions to financial obj_val
            self.objective_values = pd.concat([self.objective_values, objective_values])

            # record the solution of the variables and run again
            for der in self.poi.der_list:
                der.save_variable_results(sub_index)
            for vs in self.service_agg.value_streams.values():
                vs.save_variable_results(sub_index)
        return True

    def check_sizing_conditions(self):
        """ Throws an error if any DER is being sized under assumptions that will not
        result in a solution within a reasonable amount of time.

        """
        error = False
        if self.n == 'year':
            e_logger.error('Trying to size without setting the optimization window to \'year\'')
            error = error and True
        if self.service_agg.is_whole_sale_market():
            # whole sale markets
            e_logger.error('Params Error: trying to size the power of the battery to maximize profits in wholesale markets')
            error = error and True
        if self.service_agg.post_facto_reliability_only():
            # whole sale markets
            e_logger.error('Params Error: trying to size and preform post facto calculations only')
            error = error and True
        if self.poi.is_dcp_error(self.incl_binary):
            e_logger.error('Params Error: trying to size power and use binary formulation results in nonlinear models')
            error = error and True
        # add validation step here to check on compatibility of the tech size constraints and timeseries service constraints
        # NOTE: change this conditional if there is multiple Storage technologies (POI will resolve this)
        error = error and self.error_checks_on_sizing_with_ts_service_constraints()
        if error:
            raise ArithmeticError("Further calculations requires that economic dispatch is solved, but "
                                  + "no optimization was built or solved. Please check log files for more information. ")

    def error_checks_on_sizing_with_ts_service_constraints(self):
        """ perform error checks on DERs that are being sized with ts_user_constraints
        collect errors and raise if any were found"""
        errors_found = False
        for der in self.poi.der_list:
            try:
                solve_for_size = der.being_sized()
                # only check for BatterySizing instances
                # TODO add capability for checking other technology sizing ? --AE
                if not isinstance(der, Battery):
                    continue
                max_power_size_constraint = der.user_ch_rated_max + der.user_dis_rated_max
                for service_name, service in self.service_agg.value_streams.items():
                    try:
                        if service.error_checks_on_sizing_with_ts_service_constraints(max_power_size_constraint):
                            u_logger.info(f"Finished error checks on sizing {der.name} with timeseries {service_name} service constraints...")
                        else:
                            errors_found = True
                    except AttributeError:
                        pass
            except AttributeError:
                pass
        return errors_found
