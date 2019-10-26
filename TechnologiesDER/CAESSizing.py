"""
CAESSizing.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Thien Nguyen', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import storagevet
import cvxpy as cvx
import logging
import pandas as pd
import numpy as np
import storagevet.Constraint as Const

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class CAESSizing(storagevet.CAESTech):
    """ CAES class that inherits from Storage.

    """

    def __init__(self, name,  opt_agg, params, cycle_life):
        """ Initializes CAES class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            opt_agg (DataFrame): Initialized Financial Class
            params (dict): params dictionary from dataframe for one case
            cycle_life (DataFrame): Cycle life information
        """

        # create generic storage object
        storagevet.CAESTech.__init__(self, name,  opt_agg, params, cycle_life)

        self.size_constraints = []

        self.optimization_variables = {}


    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            ene (Variable): A cvxpy variable for Energy at the end of the time step
            dis (Variable): A cvxpy variable for Discharge Power, kW during the previous time step
            ch (Variable): A cvxpy variable for Charge Power, kW during the previous time step
            ene_max_slack (Variable): A cvxpy variable for energy max slack
            ene_min_slack (Variable): A cvxpy variable for energy min slack
            ch_max_slack (Variable): A cvxpy variable for charging max slack
            ch_min_slack (Variable): A cvxpy variable for charging min slack
            dis_max_slack (Variable): A cvxpy variable for discharging max slack
            dis_min_slack (Variable): A cvxpy variable for discharging min slack

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """

        variables = {'ene': cvx.Variable(shape=size, name='caes_ene'),
                     'dis': cvx.Variable(shape=size, name='caes_dis'),
                     'ch': cvx.Variable(shape=size, name='caes_ch'),
                     'ene_max_slack': cvx.Parameter(shape=size, name='caes_ene_max_slack', value=np.zeros(size)),
                     'ene_min_slack': cvx.Parameter(shape=size, name='caes_ene_min_slack', value=np.zeros(size)),
                     'dis_max_slack': cvx.Parameter(shape=size, name='caes_dis_max_slack', value=np.zeros(size)),
                     'dis_min_slack': cvx.Parameter(shape=size, name='caes_dis_min_slack', value=np.zeros(size)),
                     'ch_max_slack': cvx.Parameter(shape=size, name='caes_ch_max_slack', value=np.zeros(size)),
                     'ch_min_slack': cvx.Parameter(shape=size, name='caes_ch_min_slack', value=np.zeros(size)),
                     'on_c': cvx.Parameter(shape=size, name='caes_on_c', value=np.ones(size)),
                     'on_d': cvx.Parameter(shape=size, name='caes_on_d', value=np.ones(size)),
                     }

        if self.incl_slack:
            self.variable_names.update(['caes_ene_max_slack', 'caes_ene_min_slack', 'caes_dis_max_slack', 'caes_dis_min_slack', 'caes_ch_max_slack', 'caes_ch_min_slack'])
            variables.update({'ene_max_slack': cvx.Variable(shape=size, name='caes_ene_max_slack'),
                              'ene_min_slack': cvx.Variable(shape=size, name='caes_ene_min_slack'),
                              'dis_max_slack': cvx.Variable(shape=size, name='caes_dis_max_slack'),
                              'dis_min_slack': cvx.Variable(shape=size, name='caes_dis_min_slack'),
                              'ch_max_slack': cvx.Variable(shape=size, name='caes_ch_max_slack'),
                              'ch_min_slack': cvx.Variable(shape=size, name='caes_ch_min_slack')})
        if self.incl_binary:
            self.variable_names.update(['on_c', 'on_d'])
            variables.update({'on_c': cvx.Variable(shape=size, boolean=True, name='caes_on_c'),
                              'on_d': cvx.Variable(shape=size, boolean=True, name='caes_on_d')})
            if self.incl_startup:
                self.variable_names.update(['bat_start_c', 'bat_start_d'])
                variables.update({'start_c': cvx.Variable(shape=size, name='caes_start_c'),
                                  'start_d': cvx.Variable(shape=size, name='caes_start_d')})

        variables.update(self.optimization_variables)




