"""
Sizing Module

"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = 'beta'

import numpy as np
import pandas as pd
import cvxpy as cvx


class Sizing:
    """ This class is to be inherited by DER classes that want to also define the ability
    to optimal size of itself

    """

    def __init__(self):
        self.size_constraints = []

    def being_sized(self):
        """ checks itself to see if this instance is being sized

        Returns: true if being sized, false if not being sized

        """
        return bool(len(self.size_constraints))

    def sizing_summary(self):
        """ Creates the template for sizing df that each DER must fill to report their size.

        Returns: A dictionary describe this DER's size and captial costs.

        """
        # sizing_dict = {
        #     'DER': np.nan,
        #     'Energy Rating (kWh)': np.nan,
        #     'Charge Rating (kW)': np.nan,
        #     'Discharge Rating (kW)': np.nan,
        #     'Round Trip Efficiency (%)': np.nan,
        #     'Lower Limit on SOC (%)': np.nan,
        #     'Upper Limit on SOC (%)': np.nan,
        #     'Duration (hours)': np.nan,
        #     'Capital Cost ($)': np.nan,
        #     'Capital Cost ($/kW)': np.nan,
        #     'Capital Cost ($/kWh)': np.nan,
        #     'Power Capacity (kW)': np.nan,
        #     'Quantity': 1,
        # }
        # return sizing_dict

    def sizing_optimization(self, datetimes, der_list, initial_soc, verbose=True):
        """ Sets up sizing optimization.

        Args:
            datetimes (list): list of indices that need to be checked (correspond to datetimes of the analysis year)
            mask
            der_list
            initial_soc
            verbose

        Returns:
            functions (dict): functions or objectives of the optimization
            constraints (list): constraints that define behaviors, constrain variables, etc. that the optimization must meet

        """
        total_outages = len(datetimes)
        SOC_start = np.ones(total_outages) * initial_soc

        consts = []
        cost_funcs = 0

        for der_inst in der_list:
            # initialize optimization variables

            # collect capital costs of each active
            cost_funcs += der_inst.get_capex()
            # add size_constraints
            consts += der_inst.size_constraints
            # add constraints that define dispatch of each DER
            consts += der_inst.constraints(mask)

        # add constraints that define dispatch of DERs
        total_cases = self.outage_duration * total_outages
        for j in range(total_outages):
            if (j % 1000) == 0 and verbose:
                print(j)
            k = datetimes[j]
            PV_irr = total_pv_max[k:k + self.outage_duration]
            Load = self.critical_load[k:k + self.outage_duration]

            lhs = PV[(j * self.outage_duration):(j * self.outage_duration) + self.outage_duration] + dch[(j * self.outage_duration):(j * self.outage_duration) + self.outage_duration] - ch[(j * self.outage_duration):(j * self.outage_duration) + self.outage_duration]

            for DG_index in range(DG_type_no):

                lhs += DG[((j * self.outage_duration) + (DG_index * total_cases)):(((j * self.outage_duration) + self.outage_duration) + (DG_index * total_cases))]
            consts.append(lhs == Load)

        return cost_funcs, consts

    def solve_and_save(self, funcs, consts):
        """

        Args:
            funcs (cvx.Expression): sum of each DER's cost function
            consts: constraints that define that operation of each DER

        Returns:

        """
        prob = cvx.Problem(cvx.Minimize(funcs), consts)
        prob.solve(solver=cvx.GLPK_MI)

        rows = list(map(lambda der: der.sizing_summary(), self.der_list))
        sizing_df = pd.DataFrame(rows)
        sizing_df.set_index('DER')
        return sizing_df
