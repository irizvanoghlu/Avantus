"""
BatteryTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

import logging
import cvxpy as cvx
from MicrogridDER.Sizing import Sizing
from storagevet.Technology import BatteryTech
from MicrogridDER.DERExtension import DERExtension
from MicrogridDER.ESSSizing import ESSSizing


u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')
DEBUG = False


class Battery(BatteryTech.Battery, ESSSizing):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, params):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            params (dict): params dictionary from dataframe for one case
        """

        # create generic storage object
        BatteryTech.Battery.__init__(self, params)
        ESSSizing.__init__(self, self.technology_type, params)

        self.user_duration = params['duration_max']

        if self.user_duration:
            self.size_constraints += [cvx.NonPos((self.ene_max_rated / self.dis_max_rated) - self.user_duration)]

