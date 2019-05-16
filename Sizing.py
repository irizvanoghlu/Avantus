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
        storage_inputs = self.technologies['Storage']

        predispatch_service_action_map = {
            'Reliability': Reliability
        }

        for service in self.active_objects['pre-dispatch']:
            dLogger.info("Using: " + str(service))
            inputs = self.predispatch_service_inputs_map[service]
            service_func = predispatch_service_action_map[service]
            new_service = service_func(inputs, storage_inputs, self.power_kw, self.dt)
            new_service.estimate_year_data(self.opt_years, self.frequency)
            self.predispatch_services[service] = new_service

        dLogger.info("Finished adding Predispatch Services for Value Stream")

        service_action_map = {
            'DCM': DemandChargeReduction,
            'retailTimeShift': EnergyTimeShift
        }

        for service in self.active_objects['service']:
            dLogger.info("Using: " + str(service))
            inputs = self.service_input_map[service]
            service_func = service_action_map[service]
            new_service = service_func(inputs, storage_inputs, self.dt)
            new_service.estimate_year_data(self.opt_years, self.frequency)
            self.services[service] = new_service

        dLogger.info("Finished adding Services for Value Stream")

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
            start = time.time()
            prob.solve(verbose=self.verbose_opt, solver=cvx.GLPK_MI)
            end = time.time()
            print("Time it takes for Scenario solver to finish: " + str(end - start))
        except cvx.error.SolverError as err:
            uLogger.error("Solver Error...")
            dLogger.error("Solver Error...")
            sys.exit(err)

        uLogger.info(prob.status)

        # save solver used
        self.solvers = self.solvers.union(prob.solver_stats.solver_name)

        cvx_types = (cvx.expressions.cvxtypes.expression(), cvx.expressions.cvxtypes.constant())
        # evaluate optimal objective expression
        obj_values = pd.DataFrame(
            {name: [obj_expression[name].value if isinstance(obj_expression[name], cvx_types) else obj_expression[name]] for name in
             list(obj_expression)})
        # collect optimal dispatch variables
        variable_values = pd.DataFrame({name: variable_dic[name].value for name in list(variable_dic)}, index=subs.index)

        # check for non zero slack
        if np.any(abs(obj_values.filter(regex="_*slack$")) >= 1):
            uLogger.info('WARNING! non-zero slack variables found in optimization solution')

        # check for charging and discharging in same time step
        eps = 1e-4
        if any(((abs(variable_values['ch']) >= eps) & (abs(variable_values['dis']) >= eps)) & ('CAES' not in self.active_objects['storage'])):
            uLogger.info('WARNING! non-zero charge and discharge powers found in optimization solution. Try binary formulation')

        # collect actual energy contributions from services
        for serv in self.services.values():
            if self.customer_sided:
                temp_ene_df = pd.DataFrame({'ene': np.zeros(len(subs.index))}, index=subs.index)
            else:
                sub_list = serv.e[-1].value.flatten('F')
                temp_ene_df = pd.DataFrame({'ene': sub_list}, index=subs.index)
            serv.ene_results = pd.concat([serv.ene_results, temp_ene_df], sort=True)

        return variable_values, obj_values
