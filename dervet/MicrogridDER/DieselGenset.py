"""
DieselGenset Sizing class

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Andrew Etringer'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

import cvxpy as cvx
from .ICE import ICE
import numpy as np
from storagevet.Technology.DistributedEnergyResource import DER


class DieselGenset(ICE):
    """ DieselGenset Technology, with sizing optimization

    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """
        # base class is ICE
        super().__init__(params)
        self.tag = 'DieselGenset'
        self.can_participate_in_market_services = False

    def get_discharge_up_schedule(self, mask):
        """ the amount of discharge power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        # inherit from DER class (all zeros) since no market participation
        return super(DER, self).get_discharge_up_schedule(mask)

    def get_discharge_down_schedule(self, mask):
        """ the amount of discharging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        # inherit from DER class (all zeros) since no market participation
        return super(DER, self).get_discharge_down_schedule(mask)

    def get_uenergy_decrease(self, mask):
        """ the amount of energy in a timestep that is taken from the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        # inherit from DER class (all zeros) since no market participation
        return super(DER, self).get_uenergy_decrease(mask)
