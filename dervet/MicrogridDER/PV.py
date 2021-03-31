"""
PV.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
import cvxpy as cvx
from dervet.MicrogridDER.IntermittentResourceSizing import IntermittentResourceSizing
from storagevet.ErrorHandling import *
import numpy as np


class PV(IntermittentResourceSizing):
    """ Assumes perfect foresight. Ability to curtail generation

    """

    def __init__(self, params):
        """ Initializes an intermittent resource class where perfect foresight of generation is assumed.
        It inherits from the technology class. Additionally, it sets the type and physical constraints of the
        technology.

        Args:
            params (dict): Dict of parameters
        """
        TellUser.debug(f"Initializing {__name__}")
        super().__init__(params)

        self.tag = 'PV'
