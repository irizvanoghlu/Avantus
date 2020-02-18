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

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
        """

        # generate the generic predispatch service object
        storagevet.ValueStream.__init__(self, 'Reliability', params)
        self.outage_duration_coverage = params['target']  # must be in hours
        self.reliability_requirement = params['load'].dropna()  # band aid, though dropna cause it to be a deep copy
        # TODO: atm this load is only the site load, should consider aux load if included by user  --HN

        self.coverage_timesteps = 0

    def calculate_system_requirements(self, der_dict):
        """ Calculate the system requirements that must be meet regardless of what other value streams are active
        However these requirements do depend on the technology that are active in our analysis

        Args:
            der_dict (Dict): dictionary of the initialized DERs in our scenario

        """
        # determines how many time_series timestamps relates to the reliability target hours to cover
        self.coverage_timesteps = int(np.round(self.outage_duration_coverage / self.dt))  # integral type for indexing

        # set frequency gap between time data, thought this might not be necessary
        self.reliability_requirement.index.freq = self.reliability_requirement.index[1] - self.reliability_requirement.index[0]

        reverse = self.reliability_requirement.iloc[::-1]  # reverse the time series to use rolling function
        reverse = reverse.rolling(self.coverage_timesteps, min_periods=1).sum()*self.dt  # rolling function looks back, so reversing looks forward
        self.reliability_requirement = reverse.iloc[::-1]  # set it back the right way

        # add the power and energy constraints to ensure enough energy and power in the ESS for the next x hours
        # there will be 2 constraints: one for power, one for energy
        ene_min_add = Const.Constraint('ene_min', self.name, self.reliability_requirement)

        self.constraints = {'ene_min': ene_min_add}  # this should be the constraint that makes sure the next x hours have enough energy

    def objective_constraints(self, mask, load, net_power, combined_rating, critical_load):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set
            load (DataFrame): Subset of time_series load data that is being optimized
            net_power (Expression): the sum of all power flows in the system. flow out into the grid is negative
            combined_rating (Dictionary): the combined rating of each DER class type
            critical_load (pd.Expression): the load that must be met in an outage (usually load - pv generation)

        Returns: power constraint
        """

        # We want the minimum power capability of our DER mix in the discharge direction to be the maximum net load (load - solar)
        # to ensure that our DER mix can cover peak net load during any outage in the year
        return [cvx.NonPos(cvx.max(critical_load) - combined_rating)]

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.reliability_requirement.index)
        report.loc[:, 'Total Outage Requirement (kWh)'] = self.reliability_requirement
        return report
