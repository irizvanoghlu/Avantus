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
import numpy as np

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
            if self.service_agg.is_whole_sale_market():
                # whole sale markets
                e_logger.error('Params Error: trying to size the power of the battery to maximize profits in wholesale markets')
                return False
            if self.service_agg.post_facto_reliability_only():
                # whole sale markets
                e_logger.error('Params Error: trying to size and preform post facto calculations only')
                return False
            if self.poi.is_dcp_error(self.incl_binary):
                e_logger.error('Params Error: trying to size power and use binary formulation results in nonlinear models')
                return False
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

    def Reliability_based_sizing_module(self):


        #check if you have to size for reliability
        #if yes
        for opt_period in self.optimization_levels.predictive.unique():

            # used to select rows from time_series relevant to this optimization window
            mask = self.optimization_levels.predictive == opt_period
            der_list=self.poi.der_list
            top_n_outages = 1
            tech_specs = {}
            soc = None
            ess_properties = {
                'charge max': 0,
                'discharge max': 0,
                'rte list': [],
                'operation SOE min': 0,
                'operation SOE max': 0,
                'energy rating': 0
            }


            total_pv_max = np.zeros(len(self.service_agg.value_streams['Reliability'].critical_load))
            total_dg_max = 0

            for der_inst in der_list:
                if der_inst.technology_type == 'Intermittent Resource':
                    total_pv_max += der_inst.maximum_generation(None)
                if der_inst.technology_type == 'Generator':
                    total_dg_max += der_inst.discharge_capacity()
                if der_inst.technology_type == 'Energy Storage System':
                    ess_properties['rte list'].append(der_inst.rte)
                    # only if not sizing
                    # ess_properties['operation SOE min'] += der_inst.operational_min_energy()
                    # ess_properties['operation SOE max'] += der_inst.operational_max_energy()
                    # ess_properties['discharge max'] += der_inst.discharge_capacity(solution=False)
                    # ess_properties['charge max'] += der_inst.charge_capacity(solution=False)
                    # ess_properties['energy rating'] += der_inst.energy_capacity(solution=False)
                    cost_funcs = der_inst.get_capex()
            if self.service_agg.value_streams['Reliability'].n_2:
                total_dg_max -= self.service_agg.value_streams['Reliability'].ice_rating
            generation = np.repeat(total_dg_max, len(self.service_agg.value_streams['Reliability'].critical_load))

            # The maximum load demand that is unserved
            max_load_demand_unserved = np.around(self.service_agg.value_streams['Reliability'].critical_load.values - generation - total_pv_max, decimals=5)

            # Sort the outages by max demand that is unserved
            indices = np.argsort(-1 * max_load_demand_unserved)

            # Find the top n analysis indices that we are going to size our DER mix for.
            analysis_indices = indices[0:top_n_outages]
            # calculate and check that system requirement set by value streams can be met
            system_requirements = self.check_system_requirements()
            consts = []

            for outage_ind in (analysis_indices):
                Outage_mask = mask
                Outage_mask[:] = False
                Outage_mask[outage_ind: (outage_ind + int(self.service_agg.value_streams['Reliability'].outage_duration))] = True

                opt_var_size = int(np.sum(mask))
                # set up variables
                self.poi.initialize_optimization_variables(opt_var_size)
                self.service_agg.initialize_optimization_variables(opt_var_size)

                # grab values from the POI that is required to know calculate objective functions and constraints
                load_sum, critical_load_sum, var_gen_sum, gen_sum, tot_net_ess, total_soe, agg_p_in, agg_p_out = self.poi.get_state_of_system(Outage_mask)



                consts += [cvx.Zero(tot_net_ess+gen_sum+var_gen_sum+critical_load_sum)]
                for der_inst in der_list:


                    consts += der_inst.constraints(Outage_mask)
            obj = cvx.Minimize(cost_funcs)
            prob = cvx.Problem(obj, consts)
            obj=prob.solve(solver=cvx.GLPK_MI)  # ,'gp=Ture')

            IsReliable = 'No'
            Total_failures = []

            while IsReliable == 'No':
                der_list = self.sizing_optimization(mask, analysis_indices, der_list, self.soc_init,
                                                    self.outage_duration)

            new_der_list=self.service_agg.value_streams['Reliability'].size_for_Reliability(mask,der_list,time_series_data=None, technology_summary=None, sizing_df=None)

            self.poi.der_list=new_der_list

        #call
        return None