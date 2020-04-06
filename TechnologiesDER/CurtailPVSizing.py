"""
CurtailPVPV.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

import cvxpy as cvx
import pandas as pd
import storagevet


class CurtailPVSizing(storagevet.CurtailPV):
    """ Pre_IEEE 1547 2018 standards. Assumes perfect foresight. Ability to curtail PV generation, unlike ChildPV.

    """

    def __init__(self, params):
        """ Initializes a PV class where perfect foresight of generation is assumed.
        It inherits from the technology class. Additionally, it sets the type and physical constraints of the
        technology.

        Args:
            name (str): A unique string name for the technology being added, also works as category.
            params (dict): Dict of parameters
        """
        # create generic technology object
        super().__init__(params)

    def add_vars(self, size):
        """
        Args:
            size (Int): Length of optimization variables to create
        """
        tech_id = self.unique_tech_id()
        super().add_vars(size)
        if not self.rated_capacity:
            self.rated_capacity = cvx.Variable(shape=size, name='PV rating', integer=True)
            self.variables_dict.update({tech_id + 'rated_capacity': self.rated_capacity})

    def sizing_summary(self):
        """

        Returns: A dataframe indexed by the terms that describe this DER's size and captial costs.

        """
        # obtain the size of the battery, these may or may not be optimization variable
        # therefore we check to see if it is by trying to get its value attribute in a try-except statement.
        # If there is an error, then we know that it was user inputted and we just take that value instead.
        try:
            rated_capacity = self.rated_capacity.value
        except AttributeError:
            rated_capacity = self.rated_capacity

        index = pd.Index([self.name], name='DER')
        sizing_results = pd.DataFrame({'Power Capacity (kW)': rated_capacity,
                                       'Capital Cost ($/kW)': self.capital_costs['/kW']}, index=index)
        return sizing_results

    def objective_constraints(self, mask, mpc_ene=None, sizing=True):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            mpc_ene (float): value of energy at end of last opt step (for mpc opt)
            sizing (bool): flag that tells indicates whether the technology is being sized

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraints = super().objective_constraints(mask, mpc_ene, sizing)
        if not self.rated_capacity:
            constraints += [cvx.NonPos(-self.rated_capacity)]
        return constraints

    def objective_function(self, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                    the entire project lifetime (only to be set iff sizing, else alpha should not affect the aobject function)

        Returns:
            self.costs (Dict): Dict of objective costs
        """
        tech_id = self.unique_tech_id()
        super().objective_function(mask, annuity_scalar)

        if not self.rated_capacity:
            self.capex = self.capital_costs['/kW'] * self.rated_capacity
            self.costs.update({tech_id + 'capex': self.capex * annuity_scalar})

        return self.costs

    def proforma_report(self, opt_years, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            opt_years (list): list of years the optimization problem ran for
            results (DataFrame): DataFrame with all the optimization variable solutions

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

            Creates a dataframe with only the years that we have data for. Since we do not label the column,
            it defaults to number the columns with a RangeIndex (starting at 0) therefore, the following
            DataFrame has only one column, labeled by the int 0

        """
        # recalculate capex before reporting proforma
        self.capex = self.capital_costs['/kW'] * self.rated_capacity
        proforma = super().proforma_report(opt_years, results)
        return proforma

    def max_generation(self):
        """

        Returns: the maximum generation that the pv can produce

        """
        try:
            max_gen = self.get_generation().value
        except AttributeError:
            max_gen = self.get_generation()
        return max_gen

    def being_sized(self):
        """ checks itself to see if this instance is being sized

        Returns: true if being sized, false if not being sized

        """
        return True if not self.rated_capacity else False

