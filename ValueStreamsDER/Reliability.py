"""
Reliability.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import storagevet.Constraint as Const
import numpy as np
import storagevet
import cvxpy as cvx
import pandas as pd


class Reliability(storagevet.ValueStream):
    """ Reliability Service. Each service will be daughters of the PreDispService class.
    """

    def __init__(self, params, techs, load_data, dt):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
            techs (Dict): technology objects after initialization, as saved in a dictionary
            load_data (DataFrame): table of time series load data
            dt (float): optimization timestep (hours)
        """

        # generate the generic predispatch service object
        storagevet.ValueStream.__init__(self, techs['Storage'], 'Reliability', dt)
        self.outage_duration_coverage = params['target']  # must be in hours
        self.dt = params['dt']
        self.post_facto_only = params['post_facto_only']

        if 'Diesel' in techs:
            self.ice_rated_power = techs['Diesel'].rated_power
        # else:
        #     self.ice_rated_power = 0

        # determines how many time_series timestamps relates to the reliability target hours to cover
        self.coverage_timesteps = int(np.round(self.outage_duration_coverage / self.dt))  # integral type for indexing

        self.reliability_requirement = params['load'].dropna()  # band aid, though dropna cause it to be a deep copy
        # TODO: atm this load is only the site load, should consider aux load if included by user  --HN

        # set frequency gap between time data, thought this might not be necessary
        self.reliability_requirement.index.freq = self.reliability_requirement.index[1] - self.reliability_requirement.index[0]

        reverse = self.reliability_requirement.iloc[::-1]  # reverse the time series to use rolling function
        reverse = reverse.rolling(self.coverage_timesteps, min_periods=1).sum()*self.dt  # rolling function looks back, so reversing looks forward
        self.reliability_requirement = reverse.iloc[::-1]  # set it back the right way

        if not self.post_facto_only:
            ####self.reliability_pwr_requirement =
            # add the power and energy constraints to ensure enough energy and power in the ESS for the next x hours
            # there will be 2 constraints: one for power, one for energy
            ene_min_add = Const.Constraint('ene_min_add', self.name, self.reliability_requirement)
            ###dis_min = Const.Constraint('dis_min',self.name,)

            self.constraints = {'ene_min_add': ene_min_add}  # this should be the constraint that makes sure the next x hours have enough energy

    def objective_constraints(self, variables, subs, generation, reservations=None):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            variables (Dict): dictionary of variables being optimized
            subs (DataFrame): Subset of time_series data that is being optimized
            generation (list, Expression): the sum of generation within the system for the subset of time
                being optimized
            reservations (Dict): power reservations from dispatch services

        Returns:
            An empty list
        """

        try:
            pv_generation = variables['pv_out']  # time series curtailed pv optimization variable
        except KeyError:
            pv_generation = np.zeros(subs.shape[0])

        try:
            ice_rated_power = variables['n']*self.ice_rated_power  # ICE generator max rated power
        except (KeyError, AttributeError):
            ice_rated_power = 0

        try:
            battery_dis_size = variables['dis_max_rated']  # discharge size parameter for batteries
        except KeyError:
            battery_dis_size = 0

        # We want the minimum power capability of our DER mix in the discharge direction to be the maximum net load (load - solar)
        # to ensure that our DER mix can cover peak net load during any outage in the year
        return [cvx.NonPos(cvx.max(subs.loc[:, "load"].values - pv_generation) - battery_dis_size - ice_rated_power)]

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        try:
            storage_energy_rating = self.storage.ene_max_rated.value
        except AttributeError:
            storage_energy_rating = self.storage.ene_max_rated
        report = pd.DataFrame(index=self.reliability_requirement.index)
        report.loc[:, 'SOC Constraints (%)'] = self.reliability_requirement / storage_energy_rating
        report.loc[:, 'Total Outage Requirement (kWh)'] = self.reliability_requirement

        return report

    def load_coverage_probability(self, max_outage, critical_load, technologies, dt):
        """ Creates and returns a data frame with that reports the load coverage probability of outages that last from 0 to
        OUTAGE_LENGTH hours with the DER mix described in TECHNOLOGIES

        Args:
            max_outage (int): the max outage we want to cover
            critical_load (DataFrame): the load that must be covered to be considered reliable
            technologies (dict): dictionary of technologies (from Scenario)
            dt (float): delta time of the timeseries

        Returns: DataFrame with 2 columns - 'Outage Length (hrs)' and 'Load Coverage Probability (%)'

        Notes: This function assumes dt=1 (TODO)
                This function assumes only 1 storage (TODO)
        """
        # initialize a list to track the frequency of the results of the simulate_outage method
        frequency_simulate_outage = np.zeros(int(max_outage/dt)+1)
        # 1) simulate an outage that starts at every timestep
        outage_init = 0
        # collect technology specs required to call simulate_outage
        tech_specs = {}
        soc = None
        if 'Storage' in technologies:
            storage = technologies['Storage']
            tech_specs['ess_properties'] = storage.physical_properties()
            # save the state of charge
            soc = storage.variables.loc[:, 'ene']/storage.ene_max_rated
        if 'PV' in technologies:
            pv = technologies['PV']
            tech_specs['pv_generation'] = pv.max_generation()
        if 'ICE' in technologies:
            ice = technologies['ICE']
            tech_specs['fuel_generation'] = ice.max_power_out()
        while outage_init < len(critical_load):
            if soc is not None:
                tech_specs['init_soc'] = soc.iloc[outage_init]
            longest_covered_outage = self.simulate_outage(critical_load[outage_init:], dt, max_outage, **tech_specs)
            # record value of foo in frequency count
            frequency_simulate_outage[int(longest_covered_outage/dt)] += 1
            # start outage on next timestep
            outage_init += 1

        # 2) calculate probabilities
        outage_lengths = np.arange(1, max_outage+1, dt)
        outage_coverage = {'Outage Length (hrs)': outage_lengths,
                           'Load Coverage Probability (%)': []}
        for length in outage_lengths:
            scenarios_covered = frequency_simulate_outage[int(length/dt):].sum()
            total_possible_scenarios = len(critical_load) - (length/dt) + 1
            percentage = scenarios_covered/total_possible_scenarios
            outage_coverage['Load Coverage Probability (%)'].append(percentage)
        return pd.DataFrame(outage_coverage)

    def simulate_outage(self, critical_load, dt, outage_left, fuel_generation=0, pv_generation=None, ess_properties=None, init_soc=None):
        """ Simulate an outage that starts with lasting only1 hour and will either last as long as MAX_OUTAGE_LENGTH
        or the iteration loop hits the end of any of the array arguments.
        Updates and tracks the SOC throughout the outage

        Args:
            fuel_generation (int): the maximum fuel generation possible
            pv_generation (DataFrame): the maximum pv generation possible
            critical_load (DataFrame): the load profile that must be met during the outage at time t
            dt (float): the delta time
            init_soc (float, None): the soc of the ESS (if included in analysis) at the beginning of time t
            outage_left (int): the length of outage yet to be simulated
            ess_properties (dict): dictionary that describes the physical properties of the ess in the analysis
                includes 'charge max', 'discharge max, 'operation soc min', 'operation soc max', 'rte', 'energy cap'

        Returns: the length of the outage that starts at the begining of the array that can be reliabily covered

        """
        # base case: when to terminate recursion
        if outage_left == 0 or critical_load is None:
            return 0
        # check to see if there is enough fuel generation to meet the load as offset by the amount of PV
        # generation you are confident will be delivered (usually 20% of PV forecast)
        reliability_check1 = critical_load.iloc[0]
        demand_left = critical_load.iloc[0]
        if pv_generation is not None:
            reliability_check1 -= 0.2 * pv_generation.iloc[0]
            demand_left -= pv_generation
        if fuel_generation:
            reliability_check1 -= fuel_generation
            demand_left -= fuel_generation
        extra_generation = -demand_left
        if 0 >= reliability_check1:
            # check to see if there is space to storage energy in the ESS to save extra generation
            if ess_properties is not None and ess_properties['operation soc max'] >= init_soc:
                # the amount we can charge based on its current SOC
                soc_charge = (ess_properties['operation soc max'] - init_soc) * ess_properties['energy cap'] / (ess_properties['rte'] * dt)
                charge = min(soc_charge, extra_generation, ess_properties['charge max'])
                # update the state of charge of the ESS
                next_soc = init_soc + charge*ess_properties['rte']*dt
            else:
                # there is no space to save the extra generation, so the ess will not do anything
                next_soc = init_soc
            # can reliably meet the outage in that timestep: CHECK NEXT TIMESTEP
        else:
            # check that there is enough SOC in the ESS to satisfy worst case
            if ess_properties is not None and 0 >= (reliability_check1*0.43/ess_properties['energy cap']) - init_soc:
                # so discharge to meet the load offset by all generation
                soc_discharge = (init_soc - ess_properties['operation soc min']) * ess_properties['energy cap'] / dt
                discharge = min(soc_discharge, demand_left, ess_properties['discharge max'])
                # update the state of charge of the ESS
                next_soc = init_soc - discharge * dt
                # we can reliably meet the outage in that timestep: CHECK NEXT TIMESTEP
            else:
                # an outage cannot be reliably covered at this timestep, nor will it be covered beyond
                return 0
        # CHECK NEXT TIMESTEP
        # drop the first index of each array (so we can check the next timestep)
        new_pv = pv_generation.iloc[1:]
        new_cl = critical_load.iloc[1:]
        return dt + self.simulate_outage(new_cl, dt, outage_left - 1, fuel_generation, new_pv, ess_properties, next_soc)
