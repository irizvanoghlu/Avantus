"""
Reliability.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Suma Jothibasu, Halley Nathwani and Miles Evans'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = '0.1.1'


from storagevet.SystemRequirement import Requirement
import storagevet.Library as Lib
from storagevet.ValueStreams.ValueStream import ValueStream
from MicrogridDER.Sizing import Sizing
import numpy as np
import cvxpy as cvx
import pandas as pd
import time
import logging
import random

u_logger = logging.getLogger('User')
DEBUG = False


class Reliability(ValueStream):
    """ Reliability Service. Each service will be daughters of the PreDispService class.
    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
        """

        # generate the generic predispatch service object
        super().__init__('Reliability', params)
        self.outage_duration = int(params['target'])  # must be in hours
        self.dt = params['dt']
        self.post_facto_only = params['post_facto_only']
        self.soc_init = params['post_facto_initial_soc'] / 100
        self.nu = params['nu'] / 100
        self.gamma = params['gamma'] / 100
        self.max_outage_duration = params['max_outage_duration']
        self.n_2 = params['n-2']
        self.critical_load = params['critical load']

        # determines how many time_series timestamps relates to the reliability target hours to cover
        self.coverage_timesteps = int(np.round(self.outage_duration / self.dt))  # integral type for indexing

        self.reliability_requirement = self.rolling_sum(self.critical_load.loc[:], self.coverage_timesteps) * self.dt
        self.contribution_perc_df = None
        self.outage_contribution_df = None
        self.ice_rating = 0  # this is the rating of all DERs (expect for the intermittent resources)
        self.min_soc_df=None
        self.use_soc_init=False

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        self.critical_load = Lib.fill_extra_data(self.critical_load, years, load_growth, frequency)
        self.critical_load = Lib.drop_extra_data(self.critical_load, years)

    def calculate_system_requirements(self, der_lst):
        """ Calculate the system requirements that must be meet regardless of what other value streams are active
        However these requirements do depend on the technology that are active in our analysis

        Args:
            der_lst (list): list of the initialized DERs in our scenario

        """
        for der_instance in der_lst:

            if der_instance.technology_type == 'Energy Storage System' and not self.post_facto_only and self.min_soc_df is not None:

                # add the power and energy constraints to ensure enough energy and power in the ESS for the next x hours
                # there will be 2 constraints: one for power, one for energy
                soe_array = self.min_soc_df['soe']
                self.system_requirements = [Requirement('energy', 'min', self.name, soe_array)]
                # this should be the constraint that makes sure the next x hours have enough energy

    @staticmethod
    def rolling_sum(data, window):
        """ calculate a rolling sum of the date

        Args:
            data (DataFrame, Series): data of integers that can be added
            window (int): number of indexes to add

        Returns:

        """
        # reverse the time series to use rolling function
        reverse = data.iloc[::-1]
        # rolling function looks back, so reversing looks forward
        reverse = reverse.rolling(window, min_periods=1).sum()
        # set it back the right way
        data = reverse.iloc[::-1]
        return data

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.critical_load.index)
        if not self.post_facto_only:
            report.loc[:, 'Total Critical Load (kWh)'] = self.reliability_requirement

        report.loc[:, 'Critical Load (kW)'] = self.critical_load
        if self.min_soc_df is not None:
            report.loc[:, 'Reliability min State of Energy (kWh)'] = self.min_soc_df['soe']
            report.loc[:, 'Reliability min SOC (%)'] = self.min_soc_df['soc']

        return report

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, sizing_df=None, der_list=None):
        """ Calculates any service related dataframe that is reported to the user.

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with

        """
        df_dict = {}
        u_logger.info('Starting load coverage calculation. This may take a while.')
        df_dict['load_coverage_prob'] = self.load_coverage_probability(der_list,time_series_data, technology_summary)
        u_logger.info('Finished load coverage calculation.')
        # calculate RELIABILITY SUMMARY
        if not self.post_facto_only:
            self.contribution_summary(technology_summary, time_series_data)
            df_dict['outage_energy_contributions'] = self.outage_contribution_df
            df_dict['reliability_summary'] = self.contribution_perc_df
        return df_dict

    def contribution_summary(self, technology_summary_df, results):
        """ Determines that contribution from each DER type in the event of an outage.
        Call IFF attribute POST_FACTO_ONLY is False

        Args:
            technology_summary_df (DataFrame): list of active technologies
            results (DataFrame): dataframe that holds all the results of the optimzation

        Returns: dataframe of der's outage contribution

        """
        outage_energy = self.reliability_requirement
        sum_outage_requirement = outage_energy.sum()  # sum of energy required to provide x hours of energy if outage occurred at every timestep

        percent_usage = {}
        contribution_arrays = {}

        pv_names = technology_summary_df.loc[technology_summary_df['Type'] == 'Intermittent Resource']
        if len(pv_names):
            agg_pv_max = np.zeros(len(results))
            for name in pv_names['Name']:

                agg_pv_max += results.loc[:, f'PV: {name} Maximum (kW)'].values
            agg_pv_max = pd.Series(agg_pv_max, index=results.index)
            # rolling sum of energy within a coverage_timestep window
            pv_outage_e = self.rolling_sum(agg_pv_max, self.coverage_timesteps) * self.dt
            # try to cover as much of the outage that can be with PV energy
            net_outage_energy = outage_energy - pv_outage_e
            # pv generation might have more energy than in the outage, so dont let energy go negative
            outage_energy = net_outage_energy.clip(lower=0)

            # remove any extra energy from PV contribution
            # over_gen = -net_outage_energy.clip(upper=0)
            # pv_outage_e = pv_outage_e - over_gen
            pv_outage_e += net_outage_energy.clip(upper=0)

            # record contribution
            percent_usage.update({'PV': np.sum(pv_outage_e) / sum_outage_requirement})
            contribution_arrays.update({'PV Outage Contribution (kWh)': pv_outage_e})

        ess_names = technology_summary_df.loc[technology_summary_df['Type'] == 'Energy Storage System']
        if len(ess_names):
            ess_outage = results.loc[:, 'Aggregated State of Energy (kWh)']
            # try to cover as much of the outage that can be with the ES
            net_outage_energy = outage_energy - ess_outage
            # ESS might have more energy than in the outage, so dont let energy go negative
            outage_energy = net_outage_energy.clip(lower=0)

            # remove any extra energy from ESS contribution
            ess_outage = ess_outage + net_outage_energy.clip(upper=0)

            # record contribution
            percent_usage.update({'Storage': np.sum(ess_outage) / sum_outage_requirement})
            contribution_arrays.update({'Storage Outage Contribution (kWh)': ess_outage.values})

        ice_names = technology_summary_df.loc[technology_summary_df['Type'] == 'ICE']
        if len(ice_names):
            # supplies what every energy that cannot be by pv and diesel
            # diesel_contribution is what ever is left
            percent_usage.update({'ICE': 1 - sum(percent_usage.keys())})
            contribution_arrays.update({'ICE Outage Contribution (kWh)': outage_energy.values})

        self.contribution_perc_df = pd.DataFrame(percent_usage, index=pd.Index(['Reliability contribution'])).T

        self.outage_contribution_df = pd.DataFrame(contribution_arrays, index=self.critical_load.index)

    def load_coverage_probability(self, der_list, results_df, technology_summary_df):
        """ Creates and returns a data frame with that reports the load coverage probability of outages that last from 0 to
        OUTAGE_LENGTH hours with the DER mix described in TECHNOLOGIES

        Args:

            results_df (DataFrame): the dataframe that consoidates all results
            technology_summary_df(DataFrame): maps DER type to user inputted name that indexes the size df
            der_list (list): list of ders

        Returns: DataFrame with 2 columns - 'Outage Length (hrs)' and 'Load Coverage Probability (%)'

        """
        start = time.time()

        # 1) collect information required to call simulate_outage
        tech_specs = {}
        soc = None
        generation, total_pv_max, ess_properties, demand_left, reliability_check = self.get_der_limits(der_list)
        if 'Energy Storage System' in technology_summary_df['Type'].values:
            tech_specs['ess_properties'] = ess_properties
            # save the state of charge
            if not self.use_soc_init:
                soc = results_df.loc[:, 'Aggregated State of Energy (kWh)']
            else:
                soc = np.repeat(self.soc_init, len(self.critical_load)) * ess_properties['energy rating']

        end = time.time()
        u_logger.info(f'Critical Load Coverage Curve overhead time: {end - start}')

        # 2) simulate outage starting on every timestep
        start = time.time()
        # initialize a list to track the frequency of the results of the simulate_outage method
        frequency_simulate_outage = np.zeros(int(self.max_outage_duration / self.dt) + 1)
        outage_init = 0
        while outage_init < (len(self.critical_load)):
            if soc is not None:
                tech_specs['init_soe'] = soc[outage_init]
            outage_soc_profile = self.simulate_outage(reliability_check[outage_init:], demand_left[outage_init:], self.max_outage_duration, **tech_specs)
            # record value of foo in frequency count
            longest_outage = len(outage_soc_profile)
            frequency_simulate_outage[int(longest_outage)] += 1
            # start outage on next timestep
            outage_init += 1

        # 3) calculate probabilities
        load_coverage_prob = []
        length = self.dt
        while length <= self.max_outage_duration:
            scenarios_covered = frequency_simulate_outage[int(length / self.dt):].sum()
            total_possible_scenarios = len(self.critical_load) - (length / self.dt) + 1
            percentage = scenarios_covered / total_possible_scenarios
            load_coverage_prob.append(percentage)
            length += self.dt

        # 3) build DataFrame to return
        outage_coverage = {'Outage Length (hrs)': np.arange(self.dt, self.max_outage_duration + self.dt, self.dt),
                           'Load Coverage Probability (%)': load_coverage_prob}
        end = time.time()
        u_logger.info(f'Critical Load Coverage Curve calculation time: {end - start}')
        lcpc_df = pd.DataFrame(outage_coverage)
        lcpc_df.set_index('Outage Length (hrs)')
        return lcpc_df

    def get_der_limits(self, der_list, sizing=False):
        # collect information required to call simulate_outage
        # TODO change handling of multiple ESS
        ess_properties = {
            'charge max': 0,
            'discharge max': 0,
            'rte list': [],
            'operation SOE min': 0,
            'operation SOE max': 0,
            'energy rating': 0,
            'pv present': False
        }

        total_pv_max = np.zeros(len(self.critical_load))
        total_dg_max = 0
        solution = not sizing
        for der_inst in der_list:
            if der_inst.technology_type == 'Intermittent Resource' and (not der_inst.being_sized() or not sizing):
                total_pv_max += der_inst.maximum_generation(None)
                ess_properties['pv present'] = True
            if der_inst.technology_type == 'Generator' and (not der_inst.being_sized() or not sizing):
                total_dg_max += der_inst.max_power_out()
            if der_inst.technology_type == 'Energy Storage System':
                ess_properties['rte list'].append(der_inst.rte)
                ess_properties['operation SOE min'] += der_inst.operational_min_energy(solution=solution)
                ess_properties['operation SOE max'] += der_inst.operational_max_energy(solution=solution)
                ess_properties['discharge max'] += der_inst.discharge_capacity(solution=solution)
                ess_properties['charge max'] += der_inst.charge_capacity(solution=solution)
                ess_properties['energy rating'] += der_inst.energy_capacity(solution=solution)
        # takes care of N-2 case
        if self.n_2:
            total_dg_max -= self.ice_rating
        generation = np.repeat(total_dg_max, len(self.critical_load))
        demand_left = np.around(self.critical_load.values - generation - total_pv_max, decimals=5)
        reliability_check = np.around(self.critical_load.values - generation - (self.nu * total_pv_max), decimals=5)

        return generation, total_pv_max, ess_properties, demand_left, reliability_check

    def simulate_outage(self, reliability_check, demand_left, outage_left, ess_properties=None, init_soe=None):
        """ Simulate an outage that starts with lasting only1 hour and will either last as long as MAX_OUTAGE_LENGTH
        or the iteration loop hits the end of any of the array arguments.
        Updates and tracks the SOC throughout the outage

        Args:
            reliability_check (np.ndarray): the amount of load minus fuel generation and a percentage of PV generation
            demand_left (np.ndarray): the amount of load minus fuel generation and all of PV generation
            init_soe (float, None): the soc of the ESS (if included in analysis) at the beginning of time t
            outage_left (int): the length of outage yet to be simulated
            ess_properties (dict): dictionary that describes the physical properties of the ess in the analysis
                includes 'charge max', 'discharge max, 'operation SOE min', 'operation SOE max', 'rte'

        Returns: an 1 x M dimensional list where M is the SOC at each index in time,

        TODO return an N x M dimensional list where N is the number of ESS present and M is the SOC at each index in time

        """
        # base case: when to terminate recursion
        if outage_left == 0 or not len(reliability_check):
            return []
        current_reliability_check = reliability_check[0]
        current_demand_left = demand_left[0]
        if 0 >= current_reliability_check:
            # check to see if there is space to storage energy in the ESS to save extra generation
            if ess_properties is not None and ess_properties['operation SOE max'] >= init_soe:
                # the amount we can charge based on its current SOC
                random_rte = random.choice(ess_properties['rte list'])
                charge_possible = (ess_properties['operation SOE max'] - init_soe) / (random_rte * self.dt)
                charge = min(charge_possible, -current_demand_left, ess_properties['charge max'])
                # update the state of charge of the ESS
                next_soe = init_soe + (charge * random_rte * self.dt)
            else:
                # there is no space to save the extra generation, so the ess will not do anything
                next_soe = init_soe
        # can reliably meet the outage in that timestep: jump to SIMULATE OUTAGE IN NEXT TIMESTEP
        else:
            # check that there is enough SOC in the ESS to satisfy worst case
            if ess_properties is not None:
                # if there is pv present, then buffer energy require based on pv variability
                if ess_properties['pv present']:
                    energy_check = (current_reliability_check * self.gamma) - init_soe
                else:
                    energy_check = current_reliability_check - init_soe
                if 0 >= energy_check:
                    # so discharge to meet the load offset by all generation
                    discharge_possible = (init_soe - ess_properties['operation SOE min']) / self.dt
                    discharge = min(discharge_possible, current_demand_left, ess_properties['discharge max'])
                    if discharge < current_demand_left:
                        # can't discharge enough to meet demand
                        return []
                    # update the state of charge of the ESS
                    next_soe = init_soe - (discharge * self.dt)
                    # we can reliably meet the outage in that timestep: jump to SIMULATE OUTAGE IN NEXT TIMESTEP
                else:
                    # there is not enough energy in the ESS to cover the load reliabily
                    return []
            else:
                # there is no more that can be discharged to meet the load requirement
                return []
        # SIMULATE OUTAGE IN NEXT TIMESTEP
        return [next_soe] + self.simulate_outage(reliability_check[1:], demand_left[1:], outage_left - 1, ess_properties, next_soe)

    def reliability_sizing_for_analy_indices(self, mask, analysis_indices, der_list):
        """ Sets up sizing optimization.

        Args:
            mask (Series): index should match the index of the timeseries data being passed around
            der_list (list): list of initialized DERs from the POI class
            analysis_indices

        Returns: modified DER list

        """

        # have to include n-2 case --SJ
        # if self.n_2:
        #     combined_rating -= self.ice_rating
        consts = []
        cost_funcs = 0
        for der_instance in der_list:
            cost_funcs += der_instance.get_capex()

        for outage_ind in analysis_indices:
            mask[outage_ind: (outage_ind + self.outage_duration)] = True
            # set up variables
            var_gen_sum = cvx.Parameter(value=np.zeros(self.outage_duration), shape=self.outage_duration,
                                        name='POI-Zero')  # at POI
            gen_sum = cvx.Parameter(value=np.zeros(self.outage_duration), shape=self.outage_duration, name='POI-Zero')
            tot_net_ess = cvx.Parameter(value=np.zeros(self.outage_duration), shape=self.outage_duration, name='POI-Zero')

            for der_instance in der_list:
                # initialize variables
                der_instance.initialize_variables(self.outage_duration)
                consts += der_instance.constraints(mask, sizing_for_rel=True, find_min_soe=False)
                if der_instance.technology_type == 'Energy Storage System':
                    tot_net_ess += der_instance.get_net_power(mask)
                if der_instance.technology_type == 'Generator':
                    gen_sum += der_instance.get_discharge(mask)
                if der_instance.technology_type == 'Intermittent Resource':
                    var_gen_sum += der_instance.get_discharge(mask)

            critical_load_arr = cvx.Parameter(value=self.critical_load.loc[mask].values, shape=self.outage_duration)
            consts += [cvx.Zero(tot_net_ess + (-1) * gen_sum + (-1) * var_gen_sum + critical_load_arr)]

        obj = cvx.Minimize(cost_funcs)
        prob = cvx.Problem(obj, consts)
        prob.solve(solver=cvx.GLPK_MI)  # ,'gp=Ture')

        return der_list

    def find_first_uncovered(self, start_indx, reliability_check, demand_left, ess_properties=None, soe=None):
        """ THis function will return the first outage that is not covered with the given DERs

        Args:
            reliability_check (np.ndarray): the amount of load minus fuel generation and a percentage of PV generation
            demand_left (np.ndarray): the amount of load minus fuel generation and all of PV generation
            soe (list, None):
            start_indx (int): start index, idetifies the index of the start of the outage we are going to simulate

            ess_properties (dict): dictionary that describes the physical properties of the ess in the analysis
                includes 'charge max', 'discharge max, 'operation SOE min', 'operation SOE max', 'rte'

        Returns: index of the first outage that cannot be covered by the DER sizes

        """
        # base case 1: outage_init is beyond range of critical load
        if start_indx < (len(self.critical_load)):
            return 1

        # find longest possible outage
        soe_profile = self.simulate_outage(reliability_check[start_indx:], demand_left[start_indx:], self.outage_duration, ess_properties, soe[start_indx])
        longest_outage = len(soe_profile)
        # base case 2: longest outage is less than the outage duration target
        if longest_outage < self.outage_duration:
            if longest_outage < (len(self.critical_load) - start_indx):
                return 0
        # else, go on to test the next outage_init (increase index returned
        return 1 + self.find_first_uncovered(start_indx+1, reliability_check, demand_left, ess_properties, soe)

    def reliability_min_soe(self, mask, der_list):
        """ Calculates min SOE at every time step for the given DER size

           Args:
               mask
               der_list

           Returns:
               functions (dict): functions or objectives of the optimization
               constraints (list): constraints that define behaviors, constrain variables, etc. that the optimization must meet

        """
        min_soc = np.zeros(len(mask))

        cost_funcs = 0
        month_min_soc = {}
        # for der_instance in der_list:
        #     cost_funcs += der_instance.get_capex()

        #spl_indices=np.ceil((len(mask)/1000))
        for month in mask.index.month.unique():

            Outage_mask = month==mask.index.month
            consts = []

            min_soc = {}
            ana_ind= [a for a in range(len(mask)) if Outage_mask[a]==True]

            for outage_ind in ana_ind:

                Outage_mask = np.copy(mask)
                Outage_mask[outage_ind: (outage_ind + (self.outage_duration))] = True
                # set up variables
                var_gen_sum = cvx.Parameter(value=np.zeros(self.outage_duration), shape=self.outage_duration,
                                            name='POI-Zero')  # at POI
                gen_sum = cvx.Parameter(value=np.zeros(self.outage_duration), shape=self.outage_duration,
                                        name='POI-Zero')
                tot_net_ess = cvx.Parameter(value=np.zeros(self.outage_duration), shape=self.outage_duration,
                                            name='POI-Zero')

                for der_instance in der_list:
                    # initialize variables
                    der_instance.initialize_variables(self.outage_duration)

                    if der_instance.technology_type == 'Energy Storage System':
                        tot_net_ess += der_instance.get_net_power(Outage_mask)
                        der_instance.soc_target = cvx.Variable(shape=1, name=der_instance.name +str(outage_ind)+ '-min_soc' )
                        consts += [cvx.NonPos(der_instance.soc_target - 1)]
                        consts += [cvx.NonPos(-der_instance.soc_target)]

                        min_soc[outage_ind] = der_instance.soc_target

                    if der_instance.technology_type == 'Generator':
                        gen_sum += der_instance.get_discharge(Outage_mask)
                    if der_instance.technology_type == 'Intermittent Resource':
                        var_gen_sum += der_instance.get_discharge(Outage_mask)

                    consts += der_instance.constraints(Outage_mask, sizing_for_rel=True, find_min_soe=True)

                if outage_ind+self.outage_duration > len(mask):
                    remaining_out_duration=len(mask)-outage_ind
                    crit_load=np.zeros(self.outage_duration)
                    crit_load[0:remaining_out_duration]=self.critical_load.loc[Outage_mask].values
                    critical_load_arr = cvx.Parameter(value=crit_load,shape=self.outage_duration)

                else:
                    critical_load_arr = cvx.Parameter(value=self.critical_load.loc[Outage_mask].values,
                                                      shape=self.outage_duration)
                consts += [cvx.Zero(tot_net_ess + (-1) * gen_sum + (-1) * var_gen_sum + critical_load_arr)]

            # sizing_df=Sizing.solve_and_save(cost_funcs,consts)
            cost_funcs=sum(min_soc.values())
            obj = cvx.Minimize(cost_funcs)
            prob = cvx.Problem(obj, consts)
            start = time.time()
            prob.solve(solver=cvx.GLPK_MI)  # ,'gp=Ture')
            end = time.time()
            print(end - start)

            month_min_soc[month]=min_soc

        #soc_dict = {}

        for der_instance in der_list:

            if der_instance.technology_type == 'Energy Storage System':
                # TODO multi ESS
                # Get energy rating
                try:
                    energy_rating = der_instance.ene_max_rated.value
                except AttributeError:
                    energy_rating = der_instance.ene_max_rated

                # Collecting soe array for all ES
                month_min_soc_array = []
                outage_ind = 0
                for month in month_min_soc.keys():  # make sure this is in order
                    for hours in range(len(month_min_soc[month])):

                        month_min_soc_array.append(month_min_soc[month][outage_ind].value[0])
                        outage_ind += 1
                month_min_soe_array = (month_min_soc_array * energy_rating)

        #month_min_soe_array=np.repeat(energy_rating,len(mask))
        #month_min_soc_array=np.repeat(1,len(mask))
        zippedList=list(zip(month_min_soe_array,month_min_soc_array))
        self.min_soc_df=pd.DataFrame(zippedList ,index=mask.index, columns=['soe', 'soc'])

    def reliability_min_soe_iterative(self, mask, der_list):
        """ Calculates min SOE at every time step for the given DER size

           Args:
               mask
               der_list

        Returns: der_list -- ESSs will have an SOE min if they were sized for reliability

        """

        for der_instance in der_list:

            if der_instance.technology_type == 'Energy Storage System':
                # TODO multi ESS
                # Get energy rating
                try:
                    energy_rating = der_instance.ene_max_rated.value
                except AttributeError:
                    energy_rating = der_instance.ene_max_rated

                #Check if ES is sized for Reliability:
                if energy_rating>0:

                    generation, total_pv_max, ess_properties, demand_left, reliability_check = self.get_der_limits(der_list)

                    outage_init = 0
                    min_soe_array = []
                    soc = np.repeat(self.soc_init, len(self.critical_load)) * ess_properties['energy rating']
                    # while outage_init < (len(mask)):
                    #
                    #     soe_profile = self.simulate_outage(reliability_check[outage_init:], demand_left[outage_init:], self.outage_duration, ess_properties,soc[outage_init])
                    #
                    #     min_soe_array.append(self.soe_used(soe_profile))
                    #     outage_init+=1
                    min_soe_array = [self.soe_used(self.simulate_outage(reliability_check[outage_init:],
                                                                        demand_left[outage_init:],
                                                                        self.outage_duration,
                                                                        ess_properties,
                                                                        soc[outage_init])) for outage_init in range(len(mask))]
                    self.min_soc_df = pd.DataFrame(min_soe_array, index=mask.index, columns=['soe'])  # eventually going to give this to ESS to apply on itself
        return der_list

    @staticmethod
    def soe_used(soe_profile):
        """ this is the range that the battery system as to be able to acheive during the corresponding outage in order
        for the outage to be reliabily covered

        Args:
            soe_profile (list): the SOE profile of an ESS system during a simulated outage

        Returns (float) : Maximum SOE of profile - Minimum SOE of profile

        """
        min_soe = np.min(soe_profile)
        max_soe = np.max(soe_profile)
        effective_soe = max_soe - min_soe
        return effective_soe
