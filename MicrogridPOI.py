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

    def sizing_summary(self):
        sizing_df = pd.DataFrame()
        for der_category in self.der_list.values():
            for der_instance in der_category.values():
                # sizing_summary for CAES is currently similar to it for Battery
                sizing_df = der_instance.sizing_summary()
                sizing_df = pd.concat([sizing_df, sizing_df], axis=0, sort=False)
        return sizing_df
