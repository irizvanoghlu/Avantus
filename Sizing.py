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
from ValueStreamsDER import Reliability, DemandChargeReduction, EnergyTimeShift
from TechnologiesDER import CurtailPV, BatterySizing
import sys
import copy
import numpy as np
import pandas as pd
import cbaDER as Fin
import cvxpy as cvx
import svet_helper as sh
import time
import os
import matplotlib.pyplot as plt
import pickle
import plotly as py
from pathlib import Path
import logging
from prettytable import PrettyTable


class Scenario(object):
    """ A scenario is one simulation run in the model_parameters file.

    """

    def __init__(self, input_tree):
        """ Initialize a scenario.

        Args:
            #TODO correct this comment! - YY
            input_tree (Dict): Dict of input attributes such as time_series, params, and monthly_data

        TODO: remove self.input
        """

        self.verbose = input_tree.Scenario['verbose']
        print("Creating Scenario...") if self.verbose else None

        # this line needs to be removed, no more input_tree
        self.input = input_tree

        #
        self.start_time = time.time()
        self.start_time_frmt = time.strftime('%Y%m%d%H%M%S')
        self.end_time = 0
        self.user_cols = []

        # add general case params
        self.dt = input_tree.Scenario['dt']
        self.n = input_tree.Scenario['n']
        self.n_control = input_tree.Scenario['n_control']
        self.start_year = input_tree.Scenario['start_year']
        self.end_year = input_tree.Scenario['end_year']
        self.opt_years = input_tree.Scenario['opt_years']
        self.incl_site_load = input_tree.Scenario['incl_site_load']
        self.incl_aux_load = input_tree.Scenario['incl_aux_load']
        self.incl_cycle_degrade = input_tree.Technology['incl_cycle_degrade']
        self.active_objects = input_tree.active_service

        # save loaded data
        self.time_series = input_tree.Scenario['time_series']
        self.customer_tariff = input_tree.Scenario['customer_tariff']
        self.cycle_life = input_tree.Scenario['cycle_life']
        self.monthly_data = input_tree.Scenario['monthly_data']

        # tech
        self.pv = input_tree.PV
        if self.pv is not None:
            self.pv_loc = input_tree.PV['pv_loc']
        self.Battery = input_tree.Battery
        self.Diesel = input_tree.Diesel
        self.tech = input_tree.Technology
        self.Reliability = input_tree.Reliability

        self.verbose_opt = input_tree.Scenario['verbose_opt']
        self.solvers = set()

        # internal attributes to Case
        self.services = {}
        self.predispatch_services = {}
        self.technologies = {}
        self.financials = None
        self.results_path = './Results/' + '_' + self.start_time_frmt
        self.growth_rates = {'default': input_tree.Scenario['def_growth']}

        # TODO: this monthy_cols no longer works, new diffrent solution is needed - YY

        self.rt_market = 0  # TODO: Change this accordingly --HN** (set when adding services)

        self.init_financials(input_tree)

        # outputs
        self.opt_results = pd.DataFrame()
        self.prep_opt_results(self.opt_years)
        self.results = pd.DataFrame()
        self.sizing_results = pd.DataFrame()
        self.reliability_df = pd.DataFrame()
        self.dispatch_map = pd.DataFrame()
        self.peak_day_load = pd.DataFrame()
        self.energyp_map = pd.DataFrame()

        print("Scenario Created Successfully...") if self.verbose else None

    def prep_opt_results(self, opt_years):
        """
        Create standard dataframe of power data from user time_series inputs

         Args:
            opt_years (list): years need data for

        """
        # create basic data table with dttm columns
        self.opt_results = sh.create_outputs_df(self.time_series.index)

        # which columns are needed from time_series [input validation logic]
        needed_cols = []
        if self.incl_site_load:
            needed_cols += ['Site_Load (kW)']
        if self.pv:
            needed_cols += ['PV_Gen (kW)']
        # user inputed constraints
        self.user_cols = list(np.intersect1d(sh.bound_names, list(self.time_series)))
        # intersect1d() just finds the intersection of the 2 arrays. -HN

        # join in time_series data
        self.opt_results = self.opt_results.join(self.time_series[needed_cols+self.user_cols])

        # # join in monthly data
        # if not self.monthly_data.empty:
        #     self.opt_results = self.opt_results.reset_index().merge(self.monthly_data.reset_index(), on='yr_mo', how='left').set_index(self.opt_results.index.names)

        if self.dt != (self.time_series.index[1]-self.time_series.index[0]).seconds/3600:
            print('Time Series not same granularity as dt parameter. Aggregating...')
            if self.dt != 1:
                print('agg not set up for anything other than an hour')
                quit()
            # self.opt_results['dt'] = self.opt_results['hour']/self.dt

            # interval beginning
            new_index = self.opt_results.reset_index().groupby(['year', 'month', 'day', 'he']).nth(0).Datetime.values
            # interval ending
            # new_index = self.opt_results.reset_index().groupby(['year', 'month', 'he']).nth(-1).Datetime.values
            self.opt_results = self.opt_results.groupby(['year', 'month', 'day', 'he']).mean().reset_index()
            self.opt_results.index = new_index

        # calculate data for simulation of future years using growth rate
        self.opt_results = self.add_growth_data(self.opt_results, opt_years, self.dt, self.verbose)

        # create opt_agg (has to happen after adding future years)
        self.opt_results = sh.create_opt_agg(self.opt_results, self.input.Scenario['n'], self.dt)

        # add individual generation data to general generation columns
        # this will have to be more dynamic in RIVET

        # AC versus DC handling
        self.opt_results['generation'] = 0
        if self.pv:
            self.opt_results['generation'] += self.opt_results['PV_Gen (kW)']

        # logic to exclude site or aux load from load used
        self.opt_results['load'] = 0
        if self.incl_site_load:
            self.opt_results['load'] += self.opt_results['Site_Load (kW)']

        self.opt_results = self.opt_results.sort_index()

    # methods used in case
    def init_financials(self, input_tree):
        """ Initializes the financial class with a copy of all the price data from timeseries, the tariff data, and any
         system variables required for post optimization analysis.

        """
        print("Adding Financials...") if self.verbose else None
        # adding data to Financial input dictionary ... there seems to be duplicates, couple XML parameters?
        input_tree.Finance['n'] = self.n
        input_tree.Finance['start_year'] = self.start_year
        input_tree.Finance['end_year'] = self.end_year
        input_tree.Finance['services'] = input_tree.active_service["services"]
        input_tree.Finance['predispatch'] = input_tree.active_service["pre-dispatch"]
        input_tree.Finance['def_growth'] = input_tree.Scenario['def_growth']

        self.financials = Fin.Financial(input_tree.Finance, self.time_series, self.customer_tariff,
                                        self.monthly_data, self.opt_years, self.dt)
        self.financials.calc_retail_energy_price()

    def add_technology(self):
        """ Reads params and adds technology. Each technology gets initialized and their physical constraints are found.

        TODO: perhaps add any data relating to anything that could be a technology here -- HN**

        """

        print("Adding Technology...") if self.verbose else None

        # add battery and find constraints CHANGE THIS LATER
        # make this dynamic in RIVET
        print('Battery')
        self.add_storage('Storage')
        if self.pv is not None:
            print('PV')
            self.technologies['PV'] = CurtailPV.CurtailPV('PV', self.financials, self.input.Battery, self.tech, self.time_series)

    def add_storage(self, name):
        """ Add a battery to the model.

        Note:
            This creates a general technology, not a specific one.

        Args:
            name (str): A name/description of the service provided.

        ToDo: add checks to make sure no duplicates are being added
        """
        # TODO: move self.cycle_life into self.tech
        # TODO: remove self.input.Battery (it is empty)
        self.tech.update({'binary': self.input.Scenario['binary'],
                          'dt': self.dt})
        if self.Battery is not None:
            self.technologies[name] = BatterySizing.BatterySizing('Battery', self.financials, self.input.Battery,  self.tech, self.cycle_life)

    def add_services(self):
        """ Reads through params to determine which services are turned on or off. Then creates the corresponding
        service object and adds it to the list of services.

        Notes:
            This method needs to be applied after the technology has been initialized.
            ALL SERVICES ARE CONNECTED TO THE TECH

        TODO [mulit-tech] need dynamic mapping of services to tech in RIVET
        """

        print("Adding Predispatch Services...") if self.verbose else None
        # predispatch_service_active_map = {
        #     'Reliability': self.input.Reliability
        # }
        # predispatch_service_action_map = {
        #     'Reliability': Reliability.Reliability
        # }
        # for service in predispatch_service_action_map.keys():
        #     if predispatch_service_active_map[service] is not None:
        #         print(service) if self.verbose else None
        #         new_service = predispatch_service_action_map[service](predispatch_service_active_map[service],
        #                                                               self.technologies['Storage'], self.opt_results,
        #                                                               self.financials.fin_inputs)
        #         self.predispatch_services[service] = new_service

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
        # if self.input.retailTimeShift is not None:
        #     self.input.retailTimeShift.update({'price': self.financials.fin_inputs.loc[:, 'p_energy'],
        #                                        'tariff': self.financials.tariff.loc[:, self.financials.tariff.columns != 'Demand_rate']})
        #     new_service = EnergyTimeShift.EnergyTimeShift(self.input.retailTimeShift, self.technologies['Storage'], self.dt)
        #     self.services['retailTimeShift'] = new_service
        #
        # if self.input.DCM is not None:
        #     self.input.DCM.update({'tariff': self.financials.tariff.loc[:, self.financials.tariff.columns != 'Energy_price'],
        #                            'billing_period': self.financials.fin_inputs.loc[:, 'billing_period']})
        #     new_service = DemandChargeReduction.DemandChargeReduction(self.input.DCM, self.technologies['Storage'], self.dt)
        #     self.services['DCM'] = new_service

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

    def post_optimization_analysis(self):
        """ Wrapper for Post Optimization Analysis. Depending on what the user wants and what services were being
        provided, analysis on the optimization solutions are completed here.

        TODO: [multi-tech] a lot of this logic will have to change with multiple technologies
        """
        self.end_time = time.time()
        print("Performing Post Optimization Analysis...") if self.verbose else None

        # add MONTHLY ENERGY BILL if customer sided
        # TODO change this check to look if customer sided
        if "DCM" in self.services.keys() or "retailTimeShift" in self.services.keys():
            self.financials.calc_energy_bill(self.opt_results)

        # add other helpful information to a RESULTS DATAFRAME (important to keep this separate from opt_results)
        self.results = pd.DataFrame(index=self.opt_results.index)
        if self.pv is not None:
            self.results['PV Maximum (kW)'] = self.opt_results['PV_gen']
            self.results['PV Power (kW)'] = self.opt_results['pv_out']
        self.results['Load (kW)'] = self.opt_results['load']
        self.results['Discharge (kW)'] = self.opt_results['dis']
        self.results['Charge (kW)'] = self.opt_results['ch']
        self.results['Battery Power (kW)'] = self.opt_results['dis'] - self.opt_results['ch']
        self.results['State of Energy (kWh)'] = self.opt_results['ene']
        self.results['SOC (%)'] = self.opt_results['ene'] / self.technologies['Storage'].ene_max_rated.value
        self.results['Net Load (kW)'] = self.opt_results['load'] - self.opt_results['dis'] + self.opt_results['ch'] - self.opt_results['pv_out']
        self.results['Billing Period'] = self.financials.fin_inputs['billing_period']
        self.results['Energy Price ($)'] = self.financials.fin_inputs['p_energy']

        # calculate FINANCIAL SUMMARY
        self.financials.yearly_financials(self.technologies, self.services, self.opt_results)

        if self.Reliability:
            reliability_requirement = self.predispatch_services['Reliability'].reliability_requirement
            self.results['SOC Constraints (%)'] = reliability_requirement / self.technologies['Storage'].ene_max_rated.value
            # calculate RELIABILITY SUMMARY
            outage_energy = self.predispatch_services['Reliability'].reliability_requirement
            self.results['Total Outage Requirement (kWh)'] = outage_energy
            outage_requirement = outage_energy.sum()
            coverage_timestep = self.predispatch_services['Reliability'].coverage_timesteps

            reliability = {}
            if self.pv:
                reverse = self.results['PV Power (kW)'].iloc[::-1]  # reverse the time series to use rolling function
                reverse = reverse.rolling(coverage_timestep, min_periods=1).sum() * self.dt  # rolling function looks back, so reversing looks forward
                pv_outage = reverse.iloc[::-1].values  # set it back the right way
                if self.technologies['PV'].no_export:
                    pv_outage = pv_outage.clip(min=0)
                pv_contribution = np.sum(pv_outage)/outage_requirement
                reliability.update({'PV': pv_contribution})
                battery_outage_ene = outage_energy.values - pv_outage

                over_generation = battery_outage_ene.clip(max=0)
                pv_outage_energy = pv_outage + over_generation
                battery_outage_ene = battery_outage_ene.clip(min=0)
            else:
                pv_outage_energy = np.zeros(len(self.results.index))
                pv_contribution = 0
                battery_outage_ene = outage_energy
            battery_contribution = 1 - pv_contribution
            self.results['PV Outage Contribution (kWh)'] = pv_outage_energy
            self.results['Battery Outage Contribution (kWh)'] = battery_outage_ene

            reliability.update({'Battery': battery_contribution})
            # TODO: go through each technology/DER (each contribution should sum to 1)
            self.reliability_df = pd.DataFrame(reliability, index=pd.Index(['Reliability contribution'])).T

        # create DISPATCH MAP
        if self.Battery is not None:
            dispatch = self.results.loc[:, 'Battery Power (kW)'].to_frame()
            dispatch['date'] = self.opt_results['date']
            dispatch['he'] = self.opt_results['he']
            dispatch = dispatch.reset_index(drop=True)

            energy_price = self.results.loc[:, 'Energy Price ($)'].to_frame()
            energy_price['date'] = self.opt_results['date']
            energy_price['he'] = self.opt_results['he']
            energy_price = energy_price.reset_index(drop=True)

            self.dispatch_map = dispatch.pivot_table(values='Battery Power (kW)', index='he', columns='date')
            self.energyp_map = energy_price.pivot_table(values='Energy Price ($)', index='he', columns='date')

        # DESIGN PLOT (peak load day)
        max_day = self.opt_results.loc[self.opt_results['site_load'].idxmax(), :]['date']
        max_day_data = self.opt_results[self.opt_results['date'] == max_day]
        time_step = pd.Index(np.arange(0, 24, self.dt), name='Timestep Beginning')
        max_day_net_load = max_day_data['load'] - max_day_data['dis'] + max_day_data['ch'] - max_day_data['pv_out']
        self.peak_day_load = pd.DataFrame({'Date': max_day_data['date'].values,
                                           'Load (kW)': max_day_data['site_load'].values,
                                           'Net Load (kW)': max_day_net_load.values}, index=time_step)
        if self.sizing_results['Duration (hours)'].values[0] > 24:
            print('The duration of the Energy Storage System is greater than 24 hours!')

    def add_growth_data(self, df, opt_years, dt, verbose=False):
        """ Helper function: Adds rows to df where missing opt_years

        Args:
            df (DataFrame): given data
            opt_years (List): List of Period years where we need data for
            dt (float): optimization time step
            verbose (bool):

        Returns:
            df (DataFrame):

        Todo: remove some of the function inputs (can be pulled from class attributes)
        """

        data_year = df['year'].unique()  # which years was data given for
        no_data_year = {pd.Period(year) for year in opt_years} - {pd.Period(year) for year in data_year}  # which years do we not have data for

        # if there is a year we dont have data for
        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = max(data_year)  # which year to to apply growth rate to (is this the logic we want??)

                # create new dataframe for missing year
                new_index = pd.DatetimeIndex(start='01/01/' + str(yr), end='01/01/' + str(yr + 1), freq=pd.Timedelta(self.dt, unit='h'), closed='right')
                new_data = sh.create_outputs_df(new_index)

                source_data = df[df['year'] == source_year]  # use source year data

                def_rate = self.growth_rates['default']
                growth_cols = set(list(df)) - set(self.user_cols) - {'year', 'yr_mo', 'date', 'weekday', 'he'}

                # for each column in growth column
                for col in growth_cols:
                    # look for specific growth rate in params, else use default growth rate
                    name = col.split(sep='_')[0]
                    col_type = col.split(sep='_')[1].lower()
                    if col_type == 'load':
                        if name in self.growth_rates.keys():
                            rate = self.growth_rates[name]
                        else:
                            print((name, ' growth not in params. Using default growth rate:', def_rate)) if verbose else None
                            rate = def_rate
                        new_data[col] = sh.apply_growth(source_data[col], rate, source_year, yr, dt)  # apply growth rate to column
                    elif col_type == 'price':
                        if name in self.financials.growth_rates.keys():
                            rate = self.financials.growth_rates[name]
                        else:
                            print((name, ' growth not in params. Using default growth rate:', def_rate)) if verbose else None
                            rate = def_rate
                        new_data[col] = sh.apply_growth(source_data[col], rate, source_year, yr, dt)  # apply growth rate to column
                    else:
                        new_data[col] = sh.apply_growth(source_data[col], 0, source_year, yr, dt)

                # add new year to original data frame
                df = pd.concat([df, new_data], sort=True)

        return df

    def __eq__(self, other, compare_init=False):
        """ Determines whether case object equals another case object. Compare_init = True will do an initial comparison
        ignoring any attributes that are changed in the course of running a case.

        Args:
            other (Case): Case object to compare
            compare_init (bool): Flag to ignore attributes that change after initialization

        Returns:
            bool: True if objects are close to equal, False if not equal.
        """
        return sh.compare_class(self, other, compare_init)

    def save_results_csv(self, savepath=None):
        """ Save useful DataFrames to disk in csv files in the user specified path for analysis.

        """
        if savepath is None:
            savepath = self.results_path
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        self.results.to_csv(path_or_buf=Path(savepath, 'timeseries_results.csv'))
        if "DCM" in self.services.keys() or "retailTimeShift" in self.services.keys():
            self.financials.adv_monthly_bill.to_csv(path_or_buf=Path(savepath, 'adv_monthly_bill.csv'))
            self.financials.sim_monthly_bill.to_csv(path_or_buf=Path(savepath, 'simple_monthly_bill.csv'))
        if self.Reliability:
            self.reliability_df.to_csv(path_or_buf=Path(savepath, 'reliability_summary.csv'))
        self.peak_day_load.to_csv(path_or_buf=Path(savepath, 'peak_day_load.csv'))
        self.sizing_results.to_csv(path_or_buf=Path(savepath, 'size.csv'))
        if self.Battery is not None:
            self.dispatch_map.to_csv(path_or_buf=Path(savepath, 'dispatch_map.csv'))
            self.energyp_map.to_csv(path_or_buf=Path(savepath, 'energyp_map.csv'))

        self.financials.pro_forma.to_csv(path_or_buf=Path(savepath, 'pro_forma.csv'))
        self.financials.npv.to_csv(path_or_buf=Path(savepath, 'npv.csv'))
        self.financials.cost_benefit.to_csv(path_or_buf=Path(savepath, 'cost_benefit.csv'))

    @staticmethod
    def search_schema_type(root, attribute_name):
        for child in root:
            attributes = child.attrib
            if attributes.get('name') == attribute_name:
                if attributes.get('type') == None:
                    return "other"
                else:
                    return attributes.get('type')

    # TODO: this summary is for a specific scenario - TN
    def instance_summary(self):
        tree = self.input.xmlTree
        treeRoot = tree.getroot()
        schema = self.input.schema_tree

        logging.info("Printing summary table for each scenario...")
        table = PrettyTable()
        table.field_names = ["Category", "Element", "Active?", "Property", "Analysis?",
                             "Value", "Value Type", "Sensitivity"]
        for element in treeRoot:
            schemaType = self.search_schema_type(schema.getroot(), element.tag)
            activeness = element.attrib.get('active')
            for property in element:
                table.add_row([schemaType, element.tag, activeness, property.tag, property.attrib.get('analysis'),
                               property.find('value').text, property.find('type').text, property.find('sensitivity').text])

        print(table)
        logging.info('\n' + str(table))
        logging.info("Successfully printed summary table for class Scenario in log file")

        return 0
