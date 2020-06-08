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
        self.outage_duration = params['target']  # must be in hours
        self.dt = params['dt']
        self.post_facto_only = params['post_facto_only']
        #if self.post_facto_only:
        self.soc_init = params['post_facto_initial_soc'] / 100
        self.nu = params['nu'] / 100
        self.gamma = params['gamma'] / 100
        self.max_outage_duration = params['max_outage_duration']
        self.n_2 = params['n-2']
        # self.n_2 = 0

        # determines how many time_series timestamps relates to the reliability target hours to cover
        self.coverage_timesteps = int(np.round(self.outage_duration / self.dt))  # integral type for indexing
        self.critical_load = params['critical load']

        self.reliability_requirement = None
        self.contribution_perc_df = None
        self.outage_contribution_df = None
        self.ice_rating = 0  # this is the rating of all DERs (expect for the intermittent resources)

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
        for der in der_lst:
            if der.tag == 'ICE':
                self.ice_rating = der.rated_power  # save ONE random ICE's rated power in case we n-2 is true

        self.reliability_requirement = self.rolling_sum(self.critical_load.loc[:], self.coverage_timesteps) * self.dt

        if not self.post_facto_only:
            # add the power and energy constraints to ensure enough energy and power in the ESS for the next x hours
            # there will be 2 constraints: one for power, one for energy
            self.system_requirements = [Requirement('energy', 'min', self.name, self.reliability_requirement)]
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

    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, combined_rating):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional generation within the system
            net_ess_power (list, Expression): the sum of the net power of all the ESS in the system. flow out into the grid is negative
            combined_rating (Dictionary): the combined rating of each DER class type

        Returns:
            A list of constraints
        """
        if not self.post_facto_only:
            if self.n_2:
                combined_rating -= self.ice_rating

            # We want the minimum power capability of our DER mix in the discharge direction to be the maximum net load (load - solar)
            # to ensure that our DER mix can cover peak net load during any outage in the year
            return [cvx.NonPos(cvx.max(self.critical_load.loc[mask].values - tot_variable_gen*self.nu) - combined_rating)]
        else:
            return super().constraints(mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, combined_rating)

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.critical_load.index)
        if not self.post_facto_only:
            report.loc[:, 'Total Outage Requirement (kWh)'] = self.reliability_requirement
        report.loc[:, 'Critical Load (kW)'] = self.critical_load
        return report

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, sizing_df=None, der_list=None):
        """ Calculates any service related dataframe that is reported to the user.

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with

        """
        df_dict = {}
        u_logger.info('Starting load coverage calculation. This may take a while.')
        df_dict['load_coverage_prob'] = self.load_coverage_probability(time_series_data, technology_summary, der_list)
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

    def load_coverage_probability(self, results_df, technology_summary_df, der_list):
        """ Creates and returns a data frame with that reports the load coverage probability of outages that last from 0 to
        OUTAGE_LENGTH hours with the DER mix described in TECHNOLOGIES

        Args:
            results_df (DataFrame): the dataframe that consoidates all results
            technology_summary_df(DataFrame): maps DER type to user inputted name that indexes the size df
            der_list (list): list of ders

        Returns: DataFrame with 2 columns - 'Outage Length (hrs)' and 'Load Coverage Probability (%)'

        """
        start = time.time()

        # initialize a list to track the frequency of the results of the simulate_outage method
        frequency_simulate_outage = np.zeros(int(self.max_outage_duration / self.dt) + 1)

        # 1) simulate an outage that starts at every timestep

        # collect information required to call simulate_outage
        tech_specs = {}
        soc = None
        generation, total_pv_max, ess_properties = self.get_der_limits(der_list)

        demand_left = np.around(self.critical_load.values - generation - total_pv_max, decimals=5)
        reliability_check = np.around(self.critical_load.values - generation - (self.nu * total_pv_max), decimals=5)

        if 'Energy Storage System' in technology_summary_df['Type'].values:
            tech_specs['ess_properties'] = ess_properties
            # save the state of charge
            try:
                soc = results_df.loc[:, 'Aggregated State of Energy (kWh)']
            except KeyError:
                soc = np.repeat(self.soc_init, len(self.critical_load)) * ess_properties['energy rating']

        end = time.time()
        u_logger.info(f'Critical Load Coverage Curve overhead time: {end - start}')
        # simulate outage starting on every timestep
        start = time.time()
        outage_init = 0
        while outage_init < len(self.critical_load):
            if soc is not None:
                tech_specs['init_soe'] = soc[outage_init]
            longest_outage = self.simulate_outage(reliability_check[outage_init:], demand_left[outage_init:], self.max_outage_duration, **tech_specs)
            if not self.post_facto_only and longest_outage<self.outage_duration:
                break
            # record value of foo in frequency count
            frequency_simulate_outage[int(longest_outage / self.dt)] += 1
            # start outage on next timestep
            outage_init += 1
        # 2) calculate probabilities
        load_coverage_prob = []
        length = self.dt
        while length <= self.max_outage_duration:
            if not self.post_facto_only and length==self.outage_duration:
                break
            scenarios_covered = frequency_simulate_outage[int(length / self.dt):].sum()
            total_possible_scenarios = len(self.critical_load) - (length / self.dt) + 1
            percentage = scenarios_covered / total_possible_scenarios
            load_coverage_prob.append(percentage)
            length += self.dt

        # 3) build DataFrame to return
        outage_coverage = {'Outage Length (hrs)': np.arange(self.dt, self.max_outage_duration + self.dt, self.dt),
                           # '# of simulations where the outage lasts up to and including': frequency_simulate_outage,
                           'Load Coverage Probability (%)': load_coverage_prob}  # first index is prob of covering outage of 0 hours (P=100%)
        end = time.time()
        u_logger.info(f'Critical Load Coverage Curve calculation time: {end - start}')
        lcpc_df = pd.DataFrame(outage_coverage)
        lcpc_df.set_index('Outage Length (hrs)')
        return lcpc_df

    def get_der_limits(self, der_list, sizing=False):
        # collect information required to call simulate_outage
        ess_properties = {
            'charge max': 0,
            'discharge max': 0,
            'rte list': [],
            'operation SOE min': 0,
            'operation SOE max': 0,
            'energy rating': 0
        }

        total_pv_max = np.zeros(len(self.critical_load))
        total_dg_max = 0
        solution = not sizing
        for der_inst in der_list:
            if der_inst.technology_type == 'Intermittent Resource':
                total_pv_max += der_inst.maximum_generation(None)
            if der_inst.technology_type == 'Generator':
                total_dg_max += der_inst.discharge_capacity()
            if der_inst.technology_type == 'Energy Storage System':
                ess_properties['rte list'].append(der_inst.rte)
                ess_properties['operation SOE min'] += der_inst.operational_min_energy()
                ess_properties['operation SOE max'] += der_inst.operational_max_energy()
                ess_properties['discharge max'] += der_inst.discharge_capacity(solution=solution)
                ess_properties['charge max'] += der_inst.charge_capacity(solution=solution)
                ess_properties['energy rating'] += der_inst.energy_capacity(solution=solution)
        if self.n_2:
            total_dg_max -= self.ice_rating
        generation = np.repeat(total_dg_max, len(self.critical_load))
        return generation, total_pv_max, ess_properties

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

        Returns: the length of the outage that starts at the beginning of the array that can be reliably covered

        """
        # base case: when to terminate recursion
        if outage_left == 0 or not len(reliability_check):
            return 0
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
            if ess_properties is not None and 0 >= (current_reliability_check * self.gamma) - init_soe:
                # so discharge to meet the load offset by all generation
                discharge_possible = (init_soe - ess_properties['operation SOE min']) / self.dt
                discharge = min(discharge_possible, current_demand_left, ess_properties['discharge max'])
                if discharge < current_demand_left:
                    # can't discharge enough to meet demand
                    return 0
                # update the state of charge of the ESS
                next_soe = init_soe - (discharge * self.dt)
                # we can reliably meet the outage in that timestep: jump to SIMULATE OUTAGE IN NEXT TIMESTEP
            else:
                # an outage cannot be reliably covered at this timestep, nor will it be covered beyond
                return 0
        # SIMULATE OUTAGE IN NEXT TIMESTEP
        return self.dt + self.simulate_outage(reliability_check[1:], demand_left[1:], outage_left - 1, ess_properties, next_soe)

    def size_for_Reliability(self,mask,poi):
        """
        This part of the code finds the minimum DER size for a given top n number of outages or 'top_n_outages'.
        The default value for 'top_n_outages' is 10 but it can be changed.

        Returns:

        """
        top_n_outages=10
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
        total_pv_max = np.zeros(len(self.critical_load))
        total_dg_max = 0

        for der_inst in der_list:
            if der_inst.technology_type == 'Intermittent Resource':
                total_pv_max += der_inst.maximum_generation(None)
            if der_inst.technology_type == 'Generator':
                total_dg_max += der_inst.discharge_capacity()
            if der_inst.technology_type == 'Energy Storage System':
                ess_properties['rte list'].append(der_inst.rte)
                #only if not sizing
                #ess_properties['operation SOE min'] += der_inst.operational_min_energy()
                #ess_properties['operation SOE max'] += der_inst.operational_max_energy()
                #ess_properties['discharge max'] += der_inst.discharge_capacity(solution=False)
                #ess_properties['charge max'] += der_inst.charge_capacity(solution=False)
                #ess_properties['energy rating'] += der_inst.energy_capacity(solution=False)
        if self.n_2:
            total_dg_max -= self.ice_rating
        generation = np.repeat(total_dg_max, len(self.critical_load))

        #The maximum load demand that is unserved
        max_load_demand_unserved = np.around(self.critical_load.values - generation - total_pv_max, decimals=5)

        #Sort the outages by max demand that is unserved
        indices = np.argsort(-1 * max_load_demand_unserved)

        #Find the top n analysis indices that we are going to size our DER mix for.
        analysis_indices = indices[0:top_n_outages]


        startTime = time.time()
        IsReliable = 'No'
        Total_failures = []

        while IsReliable == 'No':
            der_list = self.sizing_optimization(mask,analysis_indices, der_list, self.soc_init, self.outage_duration)

##############
            First_failure = self.load_coverage_probability(time_series_data, technology_summary, der_list)

            #Total_failures.append(failures)
            print(IsReliable)
            if IsReliable == 'No':
                print(First_failure[0])
                Analysis_indices = np.append(Analysis_indices, First_failure[0])
                print(Analysis_indices)

        endTime = time.time()
        elapsedTime = endTime - startTime
        print('Elapsed time (s)=%s' % elapsedTime)
        print('Check for all outages')

    def sizing_optimization(self, mask,analysis_indices, der_list, initial_soc, outage_duration):
        """ Sets up sizing optimization.

        Args:
            datetimes (list): list of indices that need to be checked (correspond to datetimes of the analysis year)
            mask
            der_list
            initial_soc
            verbose

        Returns:
            functions (dict): functions or objectives of the optimization
            constraints (list): constraints that define behaviors, constrain variables, etc. that the optimization must meet

        """

        SOC_start = np.ones(len(analysis_indices)) * initial_soc

        consts = []
        cost_funcs = 0
        for der_inst in der_list:
            cost_funcs += der_inst.get_capex()
        for outage_ind in (analysis_indices):

            Outage_mask=mask
            Outage_mask[:]=False


            Outage_mask[outage_ind: (outage_ind+int(outage_duration))]=True

            for der_inst in der_list:
                # initialize optimization variables

                # collect capital costs of each active

                # add size_constraints
                #consts += der_inst.size_constraints
                # add constraints that define dispatch of each DER
                der_inst.initialize_variables(int(outage_duration))
                consts += der_inst.constraints(Outage_mask)

        # add constraints that define dispatch of DERs
        # total_cases = self.outage_duration * total_outages
        # for j in range(total_outages):
        #     if (j % 1000) == 0 and verbose:
        #         print(j)
        #     k = datetimes[j]
        #     PV_irr = total_pv_max[k:k + self.outage_duration]
        #     Load = self.critical_load[k:k + self.outage_duration]
        #
        #     lhs = PV[(j * self.outage_duration):(j * self.outage_duration) + self.outage_duration] + dch[(j * self.outage_duration):(j * self.outage_duration) + self.outage_duration] - ch[(j * self.outage_duration):(j * self.outage_duration) + self.outage_duration]
        #
        #     for DG_index in range(DG_type_no):
        #
        #         lhs += DG[((j * self.outage_duration) + (DG_index * total_cases)):(((j * self.outage_duration) + self.outage_duration) + (DG_index * total_cases))]
        #     consts.append(lhs == Load)



        obj = cvx.Minimize(cost_funcs)
        prob = cvx.Problem(obj, consts)
        prob.solve( solver=cvx.GLPK_MI)#,'gp=Ture')

        return der_list #cost_funcs, consts


    def calc_pv(self, der_list, initial_soc):
        """ Requires DERs to have a min size.

        Returns:

        """
        num_scenarios = len(self.critical_load) - self.outage_duration

        # collect DER data for reliability
        total_pv_max = np.zeros(len(self.critical_load))
        total_dg_max = 0
        SOE_max = 0
        SOE_min = 0
        rte_list = []
        ess_dch_rating = 0
        ess_ch_rating = 0
        for der_inst in der_list:
            if der_inst.technology_type == 'IntermittentResource':
                total_pv_max += der_inst.maximum_generation().values
            if der_inst.technology_type == 'Generator':
                total_dg_max += der_inst.discharge_capacity()
            if der_inst.technology_type == 'ESS':
                SOE_max += der_inst.operational_max_energy()
                SOE_min += der_inst.operational_min_energy()
                rte_list.append(der_inst.rte)
                ess_dch_rating += der_inst.discharge_capacity()
                ess_ch_rating += der_inst.charge_capacity()
        generation = np.ones([self.outage_duration, num_scenarios]) * total_dg_max

        cl_case = np.empty([self.outage_duration, num_scenarios])
        pv_case = np.zeros([self.outage_duration, num_scenarios])
        for i in range(0, num_scenarios):
            cl_case[:, i] = self.critical_load[i:i + self.outage_duration]
            pv_case[:, i] = total_pv_max[i:i + self.outage_duration]

        netgen = np.around(generation + pv_case - cl_case, decimals=5)
        netload_rel = np.around(cl_case - generation - pv_case, decimals=5)
        SOCminreq = netload_rel / SOE_max
        SOCminreq_total = -netgen / SOE_max

        # SOC start
        SOC_start = np.repeat(initial_soc, [1, num_scenarios])

        reliability = np.zeros([self.outage_duration, num_scenarios])
        for scenario in range(num_scenarios):
            SOC = np.ones([self.outage_duration, 1])
            SOC[0] = SOC_prev = SOC_start[0, scenario]

            for hr in range(self.outage_duration):  # Delt is 1 hour

                # Check if I have enough capactiy
                if netload_rel[hr, scenario] > 0:
                    if SOC_prev >= SOCminreq[hr, scenario] and ess_dch_rating >= netload_rel[hr, scenario] and SOC_prev >= SOCminreq_total[hr, scenario]:
                        reliability[hr, scenario] = 1
                    else:
                        break
                else:
                    reliability[hr, scenario] = 1

                # SOC evolution
                if reliability[hr, scenario]:
                    P_ch = 0
                    P_dch = 0
                    ES_E_prev_4ch = (1 - SOC_prev) * (SOE_max / random.choice(rte_list))
                    ES_E_prev_4dch = SOC_prev * SOE_max
                    if netgen[hr, scenario] > 0:
                        # calculate the max that we can charge
                        P_ch = min([netgen[hr, scenario], ess_ch_rating, ES_E_prev_4ch])

                    elif netgen[hr, scenario] < 0:
                        # calculate the max that we can discharge
                        P_dch = min([-netgen[hr, scenario], ess_dch_rating, ES_E_prev_4dch])
                    SOC_prev = SOC[hr] = SOC_prev + ((random.choice(rte_list) * P_ch) - P_dch) / SOE_max

        r_index = (np.argwhere(reliability[self.outage_duration - 1, :] == 0))

        Reliability_cum = np.cumprod(reliability, axis=0)
        R = Reliability_cum.sum(axis=1)
        total_failures = num_scenarios - R[self.outage_duration - 1]

        if len(r_index):
            is_reliable = 'No'
            first_failure = r_index[0]
        else:
            is_reliable = 'Yes'
            first_failure = 0
        return is_reliable, first_failure, total_failures

    def calculate_min_soe(self):
        # Variables
        # ch = cvx.Variable(Outage_length)
        # dch = cvx.Variable(Outage_length)
        # SOE = cvx.Variable(Outage_length)
        #
        # PV = cvx.Variable(Outage_length)
        # DG = cvx.Variable(Outage_length)
        # on_DG = cvx.Variable(Outage_length, boolean=True)
        SOC_start = cvx.Variable(1)

        obj = min(SOC_start)

        PV_irr = PV_profile[j:j + Outage_length]
        Load = Load_Profile[j:j + Outage_length]

        const1 = [SOE[0] == SOC_start - dch[0] + Batt_eff * ch[0]]
        for i in range(1, Outage_length):
            const1.append(SOE[i] == SOE[i - 1] - dch[i] + Batt_eff * ch[i])

        const1.append(PV <= cvx.multiply(PV_irr, PV_size))
        const1.append(PV + dch - ch + DG == Load)

        const1.append(dch <= ES_P)
        const1.append(ch <= ES_P)
        const1.append(SOE <= ES_E)

        const1.append(PV >= 0)

        const1.append(DG >= (on_DG * DG_params['DG_p_min']))
        const1.append(DG <= (DG_no * DG_params['DG_rating']) * on_DG)

        const1.append(ch >= 0)
        const1.append(dch >= 0)
        const1.append(SOE >= 0)

        prob = cvx.Problem(cvx.Minimize(obj), const1)
        prob.solve(solver=cvx.GLPK_MI)

        if any(on_DG.value == 0):
            print(j)

        return np.ceil(SOC_start.value[0])
