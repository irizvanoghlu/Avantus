"""
Scenario.py

This Python class contains methods and attributes vital for completing the scenario analysis.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

# from ValueStreams import DemandChargeReduction, EnergyTimeShift
from ValueStreamsDER import Reliability
from TechnologiesDER import CurtailPV, BatterySizing
from storagevet.Scenario import Scenario
from storagevet.ValueStreams import DemandChargeReduction, EnergyTimeShift
import sys
import copy
import numpy as np
import pandas as pd
from cbaDER import CostBenDER
import cvxpy as cvx
import svet_helper as sh
import time
import os
from pathlib import Path
import logging
from prettytable import PrettyTable

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')


class Sizing(Scenario):
    """ A scenario is one simulation run in the model_parameters file.

    """

    def __init__(self, input_tree):
        """ Initialize a scenario.

        Args:
            #TODO correct this comment! - YY
            input_tree (Dict): Dict of input attributes such as time_series, params, and monthly_data

        TODO: remove self.input
        """
        Scenario.__init__(self, input_tree)

        self.predispatch_service_inputs_map.update({'Reliability': input_tree.Reliability})

    def init_financials(self, finance_inputs):
        """ Initializes the financial class with a copy of all the price data from timeseries, the tariff data, and any
         system variables required for post optimization analysis.

         Args:
             finance_inputs (Dict): Financial inputs

        """

        self.financials = CostBenDER(finance_inputs)
        dLogger.info("Finished adding Financials...")

    def add_technology(self):
        """ Reads params and adds technology. Each technology gets initialized and their physical constraints are found.

        TODO: perhaps add any data relating to anything that could be a technology here -- HN**

        """
        Scenario.add_technology(self)
        generator_action_map = {
            'PV': CurtailPV.CurtailPV,

        }

        active_generator = self.active_objects['generator']
        storage_inputs = self.technology_inputs_map['Storage']
        for gen in active_generator:
            inputs = self.technology_inputs_map[gen]
            tech_func = generator_action_map[gen]
            self.technologies[gen] = tech_func(gen, self.financials, inputs, storage_inputs, self.cycle_life)
            dLogger.info("Finished adding generators...")

    def add_services(self):
        """ Reads through params to determine which services are turned on or off. Then creates the corresponding
        service object and adds it to the list of services.

        Notes:
            This method needs to be applied after the technology has been initialized.
            ALL SERVICES ARE CONNECTED TO THE TECH

        TODO [mulit-tech] need dynamic mapping of services to tech in RIVET
        """

        print("Adding Predispatch Services...") if self.verbose else None

        if self.input.Reliability is not None:
            self.input.Reliability.update()
            print('Reliability') if self.verbose else None
            new_service = Reliability.Reliability(self.input.Reliability, self.technologies['Storage'], self.opt_results,
                                                  self.financials.fin_inputs)
            self.predispatch_services['Reliability'] = new_service

        print("Adding Dispatch Services...") if self.verbose else None
        service_active_map = {
            'DCM': self.input.DCM,
            'retailTimeShift': self.input.retailTimeShift
        }
        service_action_map = {
            'DCM': DemandChargeReduction.DemandChargeReduction,
            'retailTimeShift': EnergyTimeShift.EnergyTimeShift
        }
        for service in service_action_map.keys():
            if service_active_map[service] is not None:
                print(service) if self.verbose else None
                new_service = service_action_map[service](service_active_map[service], self.financials, self.technologies['Storage'], self.dt)
                self.services[service] = new_service

    def add_control_constraints(self, deferral_check=False):
        """ Creates time series control constraints for each technology based on predispatch services.
        Must be run after predispatch services are attached to case.

        Args:
            deferral_check (bool): flag to return non feasible timestamps if running deferral feasbility analysis

        """
        tech = self.technologies['Storage']
        for service in self.predispatch_services.values():
            tech.add_service(service, predispatch=True)
        feasible_check = tech.calculate_control_constraints(self.opt_results.index)  # should pass any user inputted constraints here

        if (feasible_check is not None) & (not deferral_check):
            # if not running deferral failure analysis and infeasible scenario then stop and tell user
            print('Predispatch and Technology inputs results in infeasible scenario')
            quit()
        elif deferral_check:
            # return failure dttm to deferral failure analysis
            return feasible_check
        else:
            print("Control Constraints Successfully Created...") if self.verbose else None

    def optimize_problem_loop(self):
        """ This function selects on opt_agg of data in self.time_series and calls optimization_problem on it.

        """

        print("Preparing Optimization Problem...") if self.verbose else None

        # list of all optimization windows
        periods = pd.Series(copy.deepcopy(self.opt_results.opt_agg.unique()))
        periods.sort_values()

        for ind, opt_period in enumerate(periods):
            # ind = 0
            # opt_period = self.opt_results.opt_agg[ind]

            print(time.strftime('%H:%M:%S'), ": Running Optimization Problem for", opt_period, "...", end=" ") if self.verbose else None

            # used to select rows from time_series relevant to this optimization window
            mask = self.opt_results.loc[:, 'opt_agg'] == opt_period

            if self.incl_cycle_degrade:
                # apply past degradation
                degrade_perc = 0
                for tech in self.technologies.values():
                    if tech.degrade_data is not None:
                        # if time period since end of last optimization period is greater than dt need to estimate missing degradation
                        time_diff = (self.opt_results[mask].index[0]-self.opt_results[self.opt_results.opt_agg == periods[max(ind - 1, 0)]].index[-1])
                        if time_diff > pd.Timedelta(self.dt, unit='h'):
                            avg_degrade = np.mean(np.diff(tech.degrade_data.degrade_perc[0:ind]))
                            days_opt_agg = 30 if self.n == 'month' else int(self.n)
                            degrade_perc = avg_degrade/days_opt_agg*time_diff.days  # this could be a better estimation
                        else:
                            # else look back to previous degrade rate
                            degrade_perc = tech.degrade_data.iloc[max(ind - 1, 0)].loc['degrade_perc']

                        # apply degradation to technology (affects physical_constraints['ene_max_rated'] and control constraints)
                        tech.degrade_data.loc[opt_period, 'eff_e_cap'] = tech.apply_degradation(degrade_perc, self.opt_results.index)

            # run optimization and return optimal variable and objective expressions
            results, obj_exp = self.optimization_problem(mask)

            # Add past degrade rate with degrade from calculated period
            if self.incl_cycle_degrade:
                for tech in self.technologies.values():
                    if tech.degrade_data is not None:
                        tech.degrade_data.loc[opt_period, 'degrade_perc'] = tech.calc_degradation(results.index[0], results.index[-1], results['ene']) + degrade_perc

            # add optimization variable results to opt_results
            if not results.empty:
                self.opt_results = sh.update_df(self.opt_results, results)

            # add objective expressions to financial obj_val
            if not obj_exp.empty:
                obj_exp.index = [opt_period]
                self.financials.obj_val = sh.update_df(self.financials.obj_val, obj_exp)

    def optimization_problem(self, mask):
        """ Sets up and runs optimization on a subset of data.

        Args:
            mask (DataFrame): DataFrame of booleans used, the same length as self.time_series. The value is true if the
                        corresponding column in self.time_series is included in the data to be optimized.

        Returns:
            variable_values (DataFrame): Optimal dispatch variables for each timestep in optimization period.
            obj_values (Data Frame): Objective expressions representing the financials of each service for the optimization period.

        """

        opt_var_size = int(np.sum(mask))

        # subset of input data relevant to this optimization period
        subs = self.opt_results.loc[mask, :]

        obj_const = []  # list of constraint expressions (make this a dict for continuity??)
        variable_dic = {}  # Dict of optimization variables
        obj_expression = {}  # Dict of objective expressions

        # add optimization variables for each technology
        # TODO [multi-tech] need to handle naming multiple optimization variables (i.e ch_1)
        for tech in self.technologies.values():
            variable_dic.update(tech.add_vars(opt_var_size, self.input.Scenario['binary'],
                                              self.input.Scenario['slack'], self.input.Scenario['startup']))
        if self.pv is None:
            variable_dic.update({'pv_out': cvx.Parameter(shape=opt_var_size, name='pv_out', value=np.zeros(opt_var_size))})
            obj_const += [cvx.NonPos(variable_dic['dis'] - variable_dic['ch'] + variable_dic['pv_out'] - subs['load'])]

        # default power and energy reservations (these could be filled in with optimization variables or expressions below)
        power_reservations = np.array([0, 0, 0, 0])  # [c_max, c_min, d_max, d_min]
        energy_reservations = [cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='zero'),
                               cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='zero'),
                               cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='zero')]  # [e_upper, e, e_lower]

        for service in self.services.values():
            variable_dic.update(service.add_vars(opt_var_size))  # add optimization variables associated with each service
            obj_expression.update(service.objective_function(variable_dic, subs))  # add objective expression associated with each service
            obj_const += service.build_constraints(variable_dic, subs)   # add constraints associated with each service
            temp_power, temp_energy = service.power_ene_reservations(variable_dic)
            power_reservations = power_reservations + np.array(temp_power)   # add power reservations associated with each service
            energy_reservations = energy_reservations + np.array(temp_energy)   # add energy reservations associated with each service

        reservations = {'C_max': power_reservations[0],
                        'C_min': power_reservations[1],
                        'D_max': power_reservations[2],
                        'D_min': power_reservations[3],
                        'E': energy_reservations[1],
                        'E_upper': energy_reservations[0],
                        'E_lower': energy_reservations[2]}

        # add any objective expressions from tech and the main physical constraints
        # TODO: make slack, binary class attributes
        for tech in self.technologies.values():

            obj_expression.update(tech.objective_function(variable_dic, mask, self.dt, self.input.Scenario['slack'],
                                                          self.input.Scenario['startup']))
            obj_const += tech.build_master_constraints(variable_dic, self.dt, mask, reservations,
                                                       self.input.Scenario['binary'],
                                                       self.input.Scenario['slack'],
                                                       self.input.Scenario['startup'])

        obj = cvx.Minimize(sum(obj_expression.values()))
        prob = cvx.Problem(obj, obj_const)

        try:
            if prob.is_mixed_integer():
                # result = prob.solve(verbose=self.verbose_opt, solver=cvx.ECOS_BB,
                #                     mi_abs_eps=1, mi_rel_eps=1e-2, mi_max_iters=1000)
                result = prob.solve(verbose=self.verbose_opt, solver=cvx.GLPK_MI)
            else:
                # ECOS is default sovler and seems to work fine here
                result = prob.solve(verbose=self.verbose_opt, solver=cvx.GLPK_MI)
                # result = prob.solve(verbose=self.verbose_opt)
        except cvx.error.SolverError as err:
            sys.exit(err)

        # TODO: not sure if we want to stop the simulation if a optimization is suboptimal or just alert the user
        print(prob.status) if self.verbose else None
        # assert prob.status == 'optimal', 'Optimization problem not solved to optimality'

        # save solver used
        self.solvers = self.solvers.union(prob.solver_stats.solver_name)

        # evaluate optimal objective expression
        cvx_types = (cvx.expressions.cvxtypes.expression(), cvx.expressions.cvxtypes.constant())
        obj_values = pd.DataFrame({name: [obj_expression[name].value if isinstance(obj_expression[name], cvx_types)
                                          else obj_expression[name]] for name in list(obj_expression)})
        # collect optimal dispatch variables
        variable_values = pd.DataFrame({name: variable_dic[name].value for name in list(variable_dic)}, index=subs.index)

        # check for non zero slack
        if np.any(abs(obj_values.filter(regex="_*slack$")) >= 1):
            print('WARNING! non-zero slack variables found in optimization solution')
            # sys.exit()

        # check for charging and discharging in same time step
        eps = 1e-3
        if any(((abs(variable_values['ch']) >= eps) & (abs(variable_values['dis']) >= eps))):
            print('WARNING! non-zero charge and discharge powers found in optimization solution. Try binary formulation')

        # collect actual energy contributions from services
        # TODO: switch order of loops -- no need to loop through each serv if the simulation is customer-sided  --HN
        for serv in self.services.values():
            if "DCM" not in self.services.keys() and "retailTimeShift" not in self.services.keys():
                sub_list = serv.e[-1].value.flatten('F')
                temp_ene_df = pd.DataFrame({'ene': sub_list}, index=subs.index)
                serv.ene_results.update(temp_ene_df)

        self.sizing_results = pd.DataFrame({'Energy Capacity (kWh)': self.technologies['Storage'].ene_max_rated.value,
                                            'Power Capacity (kW)': self.technologies['Storage'].ch_max_rated.value,
                                            'Duration (hours)': self.technologies['Storage'].ene_max_rated.value/self.technologies['Storage'].ch_max_rated.value,
                                            'Capital Cost ($)': self.technologies['Storage'].ccost,
                                            'Capital Cost ($/kW)': self.technologies['Storage'].ccost_kw,
                                            'Capital Cost ($/kWh)': self.technologies['Storage'].ccost_kwh},
                                           index=pd.Index(['Battery']))  # TODO: replace with name of technology sized
        print('ene_rated: ', self.technologies['Storage'].ene_max_rated.value)
        print('ch_rated: ', self.technologies['Storage'].ch_max_rated.value)

        return variable_values, obj_values
