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


import pandas as pd
from storagevet.POI import POI
import cvxpy as cvx
from ErrorHandelling import *
import numpy as np


class MicrogridPOI(POI):
    """
        This class holds the load data for the case described by the user defined model parameter. It will also
        impose any constraints that should be opposed at the microgrid's POI.
    """

    def __init__(self, params, technology_inputs_map, technology_class_map):
        super().__init__(params, technology_inputs_map, technology_class_map)
        self.is_sizing_optimization = self.check_if_sizing_ders()
        if self.is_sizing_optimization:
            self.error_checks_on_sizing()

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

    def grab_active_ders(self, indx):
        """ drops DER that are not considered active in the optimization window's horizon

        """
        year = indx.year[0]
        active_ders = [der_instance for der_instance in self.der_list if der_instance.operational(year)]
        self.active_ders = active_ders

    def error_checks_on_sizing(self):
        # perform error checks on DERs that are being sized
        # collect errors and raise if any were found
        errors_found = [1 if der.sizing_error() else 0 for der in self.der_list]
        if sum(errors_found):
            raise ParameterError(f'Sizing of DERs has an error. Please check error log.')

    def is_any_sizable_der_missing_power_max(self):
        return bool(sum([1 if not der_inst.max_power_defined else 0 for der_inst in self.der_list]))

    def is_dcp_error(self, is_binary_formulation):
        """ If trying to sizing power of batteries (or other DERs) AND using the binary formulation (of ESS)
        our linear model will not be linear anymore

        Args:
            is_binary_formulation (bool):

        Returns: a boolean

        """
        solve_for_size = False
        for der_instance in self.der_list:
            if der_instance.tag == 'Battery':
                solve_for_size = solve_for_size or (der_instance.is_power_sizing() and is_binary_formulation)
        return solve_for_size

    def sizing_summary(self):
        rows = list(map(lambda der: der.sizing_summary(), self.der_list))
        sizing_df = pd.DataFrame(rows)
        sizing_df.set_index('DER')
        return sizing_df

    def get_state_of_system(self, mask):
        """ POI method to measure the state of POI depending on available types of DERs. used in SET_UP_OPTIMIZATION
        Builds extends StorageVET's method to take into account types of technologies added by DERVET

        Args:
            mask (DataFrame): DataFrame of booleans used, the same length as time_series. The value is true if the
                        corresponding column in time_series is included in the data to be optimized.

        Returns:
            aggregation of loads
            aggregation of generation from variable resources
            aggregation of generation from other sources
            total net power from ESSs
            total state of energy stored in the system
            aggregation of all the power flows into the POI
            aggregation of all the power flows out if the POI
        """
        opt_var_size = sum(mask)
        load_sum = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        var_gen_sum = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        gen_sum = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        tot_net_ess = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        total_soe = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        agg_power_flows_in = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        agg_power_flows_out = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI

        for der_instance in self.active_ders:
            # add the state of the der's power over time & stored energy over time to system's
            agg_power_flows_in += der_instance.get_charge(mask)
            agg_power_flows_out += der_instance.get_discharge(mask)

            if der_instance.technology_type == 'Load':
                load_sum += der_instance.get_charge(mask)
            if der_instance.technology_type == 'Electric Vehicle':
                load_sum += der_instance.get_charge(mask)
                total_soe += der_instance.get_state_of_energy(mask)

            if der_instance.technology_type == 'Energy Storage System':
                total_soe += der_instance.get_state_of_energy(mask)
                tot_net_ess += der_instance.get_net_power(mask)
            if der_instance.technology_type == 'Generator':
                gen_sum += der_instance.get_discharge(mask)
            if der_instance.technology_type == 'Intermittent Resource':
                var_gen_sum += der_instance.get_discharge(mask)
        return load_sum, var_gen_sum, gen_sum, tot_net_ess, total_soe, agg_power_flows_in, agg_power_flows_out

    def merge_reports(self, index):
        """ Collects and merges the optimization results for all DERs into
        Builds extends StorageVET's method to take into account types of technologies added by DERVET

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = pd.DataFrame(index=index)
        monthly_data = pd.DataFrame()

        # initialize all the data columns that will ALWAYS be present in our results
        results.loc[:, 'Total Original Load (kW)'] = 0
        results.loc[:, 'Total Load (kW)'] = 0
        results.loc[:, 'Total Generation (kW)'] = 0
        results.loc[:, 'Total Storage Power (kW)'] = 0
        results.loc[:, 'Aggregated State of Energy (kWh)'] = 0

        for der_instance in self.der_list:
            report_df = der_instance.timeseries_report()
            results = pd.concat([report_df, results], axis=1)
            if der_instance.technology_type in ['Generator', 'Intermittent Resource']:
                results.loc[:, 'Total Generation (kW)'] += results[f'{der_instance.unique_tech_id()} Generation (kW)']
            if der_instance.technology_type == 'Energy Storage System':
                results.loc[:, 'Total Storage Power (kW)'] += results[f'{der_instance.unique_tech_id()} Power (kW)']
                results.loc[:, 'Aggregated State of Energy (kWh)'] += results[f'{der_instance.unique_tech_id()} State of Energy (kWh)']
            if der_instance.technology_type == 'Load':
                results.loc[:, 'Total Original Load (kW)'] += results[f'{der_instance.unique_tech_id()} Original Load (kW)']
                if der_instance.tag == "ControllableLoad":
                    results.loc[:, 'Total Load (kW)'] += results[f'{der_instance.unique_tech_id()} Load (kW)']
                else:
                    results.loc[:, 'Total Load (kW)'] += results[f'{der_instance.unique_tech_id()} Original Load (kW)']
            if der_instance.technology_type == 'Electric Vehicle':
                results.loc[:, 'Total Load (kW)'] += results[f'{der_instance.unique_tech_id()} Charge (kW)']
                if der_instance.tag == 'ElectricVehicle1':
                    results.loc[:, 'Aggregated State of Energy (kWh)'] += results[f'{der_instance.unique_tech_id()} State of Energy (kWh)']
            report = der_instance.monthly_report()
            monthly_data = pd.concat([monthly_data, report], axis=1, sort=False)
            # assumes the orginal net load only does not contain the Storage system

            # net load is the load see at the POI
            results.loc[:, 'Net Load (kW)'] = results.loc[:, 'Total Load (kW)'] - results.loc[:, 'Total Generation (kW)'] - results.loc[:, 'Total Storage Power (kW)']
        return results, monthly_data
