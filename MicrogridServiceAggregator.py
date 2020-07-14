__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Thien Nguyen', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = "x.x.x"

from storagevet.ServiceAggregator import ServiceAggregator
from ErrorHandelling import *
import cvxpy as cvx


class MicrogridServiceAggregator(ServiceAggregator):
    """ The entity that tracks the value streams and bids the Microgrid's capabilities
    into energy markets

    """
    def post_facto_reliability_only(self):
        """

        Returns: A boolean that is true if Reliability is doing post facto calculations only

        """
        return len(self.value_streams.keys()) == 1 and 'Reliability' in self.value_streams.keys() and self.value_streams['Reliability'].post_facto_only

    def is_whole_sale_market(self):
        """

        Returns: boolean, interect btw list of market services

        """
        return {'SR', 'NSR', 'FR', 'LF'} & set(self.value_streams.keys())

    def set_size(self, der_lst, start_year):
        """ iterates over a list of DER+DERExtension objects and sets their minimum size
        based on the P and E requirements set by MIN_YEAR objective.

        Args:
            der_lst:

        Returns: der_list with size minimums

        """
        deferral = self.value_streams['Deferral']
        min_year = deferral.min_years
        last_year_to_defer = start_year.year + min_year-1
        p_e_req = deferral.deferral_df.loc[last_year_to_defer, :]
        min_power = p_e_req.loc['Power Capacity Requirement (kW)']
        min_energy = p_e_req.loc['Energy Capacity Requirement (kWh)']
        der_lst[0].ch_max_rated = min_power
        der_lst[0].dis_max_rated = min_power
        der_lst[0].ene_max_rated = min_energy
        # der_lst[0].size_constraints += [cvx.NonPos(min_energy - der_lst[0].ene_max_rated)]
        # der_lst[0].size_constraints += [cvx.NonPos(min_power - der_lst[0].ch_max_rated)]
        # der_lst[0].size_constraints += [cvx.NonPos(min_power - der_lst[0].dis_max_rated)]
        return der_lst

    def does_wholesale_markets_have_max_defined(self):
        """

        Returns:

        """
        error = False
        for vs_name in {'LF', 'SR', 'NSR', 'FR'}:
            vs = self.value_streams.get(vs_name, False)
            if vs and not vs.u_ts_constraints and not vs.d_ts_constraints:
                TellUser.error('Trying to size the power of the system to maximize profits ' +
                               f'in wholesale markets, but {vs_name} time-series constraints is not applied.')
                error = True
        return error

    def any_max_participation_constraints_not_included(self):
        return bool(sum([1 if not vs.max_participation_is_defined() else 0 for vs in self.value_streams]))
