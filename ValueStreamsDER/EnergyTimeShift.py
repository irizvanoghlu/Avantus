"""
EnergyTimeShift.py

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
import cvxpy as cvx


class EnergyTimeShift(ValueStream):
    """ Retail energy time shift. A behind the meter service.

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
        ValueStream.__init__(self, tech, 'retailETS', dt)
        self.p_energy = financials_df.loc[:, 'p_energy']

    def objective_function(self, variables, subs):
        """ Generates the full objective function, including the optimization variables.

        Args:
            variables (Dict): Dictionary of optimization variables
            subs (DataFrame): Subset of time_series data that is being optimized
            mask (DataFrame): DataFrame of booleans used, the same length as self.time_series. The value is true if the
                        corresponding column in self.time_series is included in the data to be optimized.

        Returns:
            The expression of the objective function that it affects. This can be passed into the cvxpy solver.

        """
        size = subs.index.size
        load = cvx.Parameter(size, value=np.array(subs.loc[:, "load"]), name='load')
        # gen = cvx.Parameter(size, value=np.array(subs.loc[:, "dc_gen"]) + np.array(subs.loc[:, "ac_gen"]), name='gen')
        gen = variables['pv_out']
        p_energy = cvx.Parameter(size, value=self.p_energy.loc[subs.index].values, name='energy_price')
        self.costs.append(cvx.sum(p_energy*load*self.dt - p_energy*variables['dis']*self.dt + p_energy*variables['ch']*self.dt - p_energy*gen*self.dt))
        return {self.name: cvx.sum(p_energy*load*self.dt - p_energy*variables['dis']*self.dt + p_energy*variables['ch']*self.dt - p_energy*gen*self.dt)}

    def update_data(self, time_series, fin_inputs):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            time_series (DataFrame): the mutated time_series data with extrapolated load data
            fin_inputs (DataFrame): the mutated time_series data with extrapolated price data

        """
        self.p_energy = fin_inputs.loc[:, 'p_energy']
