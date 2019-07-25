"""
BatteryTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

from storagevet.Technology.BatteryTech import BatteryTech
import logging
import cvxpy as cvx
import pandas as pd

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class BatterySizing(BatteryTech):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, name,  opt_agg, params, cycle_life):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            opt_agg (DataFrame): Initalized Financial Class
            params (dict): params dictionary from dataframe for one case
            cycle_life (DataFrame): Cycle life information
        """

        # create generic storage object
        BatteryTech.__init__(self, name,  opt_agg, params, cycle_life)

        # if the user inputted the energy rating as 0, then size for duration
        if not self.ene_max_rated:
            self.ene_max_rated = cvx.Variable(name='Energy_cap')

        # if both the discharge and charge ratings are 0, then size for both and set them equal to each other
        if not self.ch_max_rated and not self.dis_max_rated:
            self.ch_max_rated = cvx.Variable(name='power_cap')
            self.dis_max_rated = self.ch_max_rated
        elif not self.ch_max_rated:  # if the user inputted the discharge rating as 0, then size discharge rating
            self.ch_max_rated = cvx.Variable(name='charge_power_cap')
        elif not self.dis_max_rated:  # if the user inputted the charge rating as 0, then size for charge
            self.dis_max_rated = cvx.Variable(name='discharge_power_cap')

    def sizing_summary(self):
        """

        Returns: A datafram indexed by the terms that describe this DER's size and captial costs.

        """
        # obtain the size of the battery, these may or may not be optimization variable
        # therefore we check to see if it is by trying to get its value attribute in a try-except statement.
        # If there is an error, then we know that it was user inputted and we just take that value instead.
        try:
            energy_rated = self.ene_max_rated.value
        except AttributeError:
            energy_rated = self.ene_max_rated

        try:
            ch_max_rated = self.ch_max_rated.value
        except AttributeError:
            ch_max_rated = self.ch_max_rated

        try:
            dis_max_rated = self.dis_max_rated.value
        except AttributeError:
            dis_max_rated = self.dis_max_rated

        sizing_data = [energy_rated,
                       ch_max_rated,
                       dis_max_rated,
                       energy_rated / dis_max_rated,
                       self.ccost,
                       self.ccost_kw,
                       self.ccost_kwh]
        index = pd.Index(['Energy Rating (kWh)',
                          'Charge Rating (kW)',
                          'Discharge Rating (kW)',
                          'Duration (hours)',
                          'Capital Cost ($)',
                          'Capital Cost ($/kW)',
                          'Capital Cost ($/kWh)'], name='Size and Costs')
        sizing_results = pd.DataFrame({self.name: sizing_data}, index=index)
        return sizing_results
