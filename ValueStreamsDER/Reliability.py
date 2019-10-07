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
        except KeyError:
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
