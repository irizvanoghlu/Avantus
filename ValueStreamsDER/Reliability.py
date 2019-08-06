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


class Reliability(storagevet.ValueStream):
    """ Reliability Service. Each service will be daughters of the PreDispService class.
    """

    def __init__(self, params, tech, load_data, dt):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
            tech (Technology): Storage technology object
            load_data (DataFrame): table of time series load data
            dt (float): optimization timestep (hours)
        """

        # generate the generic predispatch service object
        storagevet.ValueStream.__init__(self, tech, 'Reliability', dt)
        self.outage_duration_coverage = params['target']  # must be in hours
        self.dt = params['dt']

        # determines how many time_series timestamps relates to the reliability target hours to cover
        self.coverage_timesteps = int(np.round(self.outage_duration_coverage / self.dt))  # integral type for indexing

        self.reliability_requirement = params['load'].dropna()  # band aid, though dropna cause it to be a deep copy
        # TODO: atm this load is only the site load, should consider aux load if included by user  --HN

        # set frequency gap between time data, thought this might not be necessary
        self.reliability_requirement.index.freq = self.reliability_requirement.index[1] - self.reliability_requirement.index[0]

        reverse = self.reliability_requirement.iloc[::-1]  # reverse the time series to use rolling function
        reverse = reverse.rolling(self.coverage_timesteps, min_periods=1).sum()*self.dt  # rolling function looks back, so reversing looks forward
        self.reliability_requirement = reverse.iloc[::-1]  # set it back the right way

        # add the power and energy constraints to ensure enough energy and power in the ESS for the next x hours
        # there will be 2 constraints: one for power, one for energy
        ene_min_add = Const.Constraint('ene_min_add', self.name, self.reliability_requirement)

        self.constraints = {'ene_min_add': ene_min_add}  # this should be the constraint that makes sure the next x hours have enough energy


