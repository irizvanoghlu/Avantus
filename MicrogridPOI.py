"""
MicrogridPOI.py

"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']


import logging
import numpy as np
import cvxpy as cvx
import pandas as pd
from storagevet.POI import POI

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')


class MicrogridPOI(POI):
    """
        This class holds the load data for the case described by the user defined model parameter. It will also
        impose any constraints that should be opposed at the microgrid's POI.
    """

    def __init__(self, params, technology_inputs_map, technology_class_map):
        super().__init__(params, technology_inputs_map, technology_class_map)
        self.is_sizing_optimization = self.check_if_sizing_ders()

    def check_if_sizing_ders(self):
        """ This method will iterate through the initialized DER instances and return a logical OR of all of their
        'being_sized' methods.

        Returns: True if ANY DER is getting sized

        """
        for der_instance in self.der_list:
            try:
                solve_for_size = der_instance.being_sized()
            except AttributeError:
                solve_for_size = False
            if solve_for_size:
                return True
        return False

    def sizing_summary(self):
        rows = list(map(lambda der: der.sizing_summary(), self.der_list))
        sizing_df = pd.DataFrame(rows)
        sizing_df.set_index('DER')
        return sizing_df
