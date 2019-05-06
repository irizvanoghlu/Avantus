"""
DemandChargeReduction.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

from ValueStreams.ValueStream import ValueStream
import numpy as np
import pandas as pd
import cvxpy as cvx
import sys


class DemandChargeReduction(ValueStream):
    """ Retail demand charge reduction. A behind the meter service.

    """

    def __init__(self, params, financials, tech, dt):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
            financials (Financial): Financial object
            tech (Technology): Storage technology object
            dt (float): optimization timestep (hours)
        """
        financials_df = financials.fin_inputs
        ValueStream.__init__(self, tech, 'DCM', dt)
        self.demand_rate = financials.tariff.loc[:, 'Demand_rate']
        self.billing_period = financials_df.loc[:, 'billing_period']

    def objective_function(self, variables, subs):
        """ Generates the full objective function, including the optimization variables.

        Args:
            variables (Dict): Dictionary of optimization variables
            subs (DataFrame): Subset of time_series data that is being optimized
            mask (DataFrame): DataFrame of booleans used, the same length as self.time_series. The value is true if the
                        corresponding column in self.time_series is included in the data to be optimized.
        Returns:
            An Expression--The portion of the objective function that it affects. This can be passed into the cvxpy solver.

        """
        # pandas converts the billing period lists to ints if there are only one
        # per time step. This checks for that and handles appropriately
        billing_period = self.billing_period.loc[subs.index]
        if isinstance(billing_period.iloc[0], list):
            pset = [item for sublist in billing_period for item in sublist]
            pset = set(pset)
        elif isinstance(billing_period.iloc[0], int):
            pset = set(billing_period)
        else:
            print('Billing period neither list nor int')
            sys.exit()
        dcterm = 0
        # if mask contains more than a month, then add dcterm monthly
        yr_mo = pd.DataFrame(index=subs.index)
        yr_mo['yr_mo'] = (subs.index - pd.Timedelta('1s')).to_period('M')
        for month in yr_mo['yr_mo'].unique():
            month_mask = (yr_mo.loc[:, 'yr_mo'] == month)
            # generation_array = np.array(subs.loc[:, "dc_gen"]) + np.array(subs.loc[:, "ac_gen"])
            load_array = np.array(subs.loc[:, "load"])
            net_load = load_array - variables['dis'] + variables['ch'] - variables['pv_out']
            for per in pset:  # Add demand charge calculation for each applicable billing period
                mask1 = [(int(per) in billing_period.iloc[i]) and month_mask[i] for i in range(subs.shape[0])]
                if sum(mask1):
                    dcterm += self.demand_rate.loc[int(per)] * cvx.sum_largest(net_load[mask1], 1)
        self.costs.append(dcterm)
        return {self.name: dcterm}

    def update_data(self, time_series, fin_inputs):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            time_series (DataFrame): the mutated time_series data with extrapolated load data
            fin_inputs (DataFrame): the mutated time_series data with extrapolated price data

        """
        self.billing_period = fin_inputs.loc[:, 'billing_period']
