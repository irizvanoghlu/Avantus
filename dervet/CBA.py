"""
Finances.py

This Python class contains methods and attributes vital for completing financial analysis given optimal dispathc.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version

from storagevet.Finances import Financial
import numpy as np
import copy
import pandas as pd
from storagevet.ErrorHandelling import *

SATURDAY = 5


class CostBenefitAnalysis(Financial):
    """ This Cost Benefit Analysis Module

    """

    def __init__(self, financial_params, start_year, end_year):
        """ Initialize CBA model and edit any attributes that the user denoted a separate value
        to evaluate the CBA with

        Args:
            financial_params (dict): parameter dictionary as the Params class created
        """
        super().__init__(financial_params, start_year, end_year)
        self.horizon_mode = financial_params['analysis_horizon_mode']
        self.location = financial_params['location']
        self.ownership = financial_params['ownership']
        self.state_tax_rate = financial_params['state_tax_rate']/100
        self.federal_tax_rate = financial_params['federal_tax_rate']/100
        self.property_tax_rate = financial_params['property_tax_rate']/100
        self.ecc_mode = financial_params['ecc_mode']
        self.ecc_df = pd.DataFrame()
        self.equipment_lifetime_report = pd.DataFrame()

        self.Scenario = financial_params['CBA']['Scenario']
        self.Finance = financial_params['CBA']['Finance']
        self.valuestream_values = financial_params['CBA']['valuestream_values']
        self.ders_values = financial_params['CBA']['ders_values']
        if 'Battery' in self.ders_values.keys():
            self.ders_values['Battery'] = self.ders_values.pop('Battery')
        if 'CAES' in self.ders_values.keys():
            self.ders_values['CAES'] = self.ders_values.pop('CAES')

        self.value_streams = {}
        self.ders = []

        self.macrs_depreciation = {
            3: [33.33, 44.45, 14.81, 7.41],
            5: [20, 32, 19.2, 11.52, 11.52, 5.76],
            7: [14.29, 24.49, 17.49, 12.49, 8.93, 8.92, 8.93, 4.46],
            10: [10, 18, 14.4, 11.52, 9.22, 7.37, 6.55, 6.55, 6.56, 6.55,
                 3.28],
            15: [5, 9.5, 8.55, 7.7, 6.83, 6.23, 5.9, 5.9, 5.91, 5.9,
                 5.91, 5.9, 5.91, 5.9, 5.91, 2.95],
            20: [3.75, 7.219, 6.677, 6.177, 5.713, 5.285, 4.888, 4.522, 4.462, 4.461,
                 4.462, 4.461, 4.462, 4.461, 4.462, 4.461, 4.462, 4.461, 4.462, 4.461,
                 2.231]
        }

    def find_end_year(self, der_list):
        """ This method looks at the analysis horizon mode and sets up the CBA class
        for the indicated mode

        Args:
            der_list (list): list of DERs initialized with user values

        Returns: pandas Period representation of the year that DERVET will end CBA analysis

        """
        project_start_year = self.start_year
        user_given_end_year = self.end_year
        # (1) User-defined (this should continue to be default)
        if self.horizon_mode == 1:
            self.end_year = user_given_end_year
        # (2) Auto-calculate based on shortest equipment lifetime. (No size optimization)
        if self.horizon_mode == 2:
            shortest_lifetime = 1000  # no technology should last 1000 years -- so this is safe to hardcode
            for der_instance in der_list:
                shortest_lifetime = min(der_instance.expected_lifetime, shortest_lifetime)
                if der_instance.being_sized():
                    TellUser.error("Analysis horizon mode == 'Auto-calculate based on shortest equipment lifetime', DER-VET will not size any DERs " +
                                   f"when this horizon mode is selected. {der_instance.name} is being sized. Please resolve and rerun.")
                    self.end_year = pd.Period(year=0, freq='y')  # cannot preform size optimization with mode==2
            self.end_year = project_start_year + shortest_lifetime-1
        # (3) Auto-calculate based on longest equipment lifetime. (No size optimization)
        if self.horizon_mode == 3:
            longest_lifetime = 0
            for der_instance in der_list:
                longest_lifetime = max(der_instance.expected_lifetime, longest_lifetime)
                if der_instance.being_sized():
                    TellUser.error("Analysis horizon mode == 'Auto-calculate based on longest equipment lifetime', DER-VET will not size any DERs " +
                                   f"when this horizon mode is selected. {der_instance.name} is being sized. Please resolve and rerun.")
                    self.end_year = pd.Period(year=0, freq='y')  # cannot preform size optimization with mode==3
            self.end_year = project_start_year + longest_lifetime-1
        return self.end_year

    def ecc_checks(self, der_list, service_dict):
        """

        Args:
            der_list: list of ders
            service_dict: dictionary of services

        Returns:

        """
        # require that ownership model is Utility TODO

        # check that a service in this set: {Reliability, Deferral} - the union of the 2 sets should not be 0
        if not len(set(service_dict.keys()) & {'Reliability', 'Deferral'}):
            TellUser.error(f"An ecc analysis does not make sense for the case you selected. A reliability or asset deferral case" +
                           "would be better suited for economic carrying cost analysis")
            raise ModelParameterError("The combination of services does not work with the rest of your case settings. " +
                                      "Please see log file for more information.")
        # require that e < d
        for der_inst in der_list:
            conflict_occured = False
            if der_inst.escalation_rate >= self.npv_discount_rate:
                conflict_occured = True
                TellUser.error(f"The technology escalation rate ({der_inst.escalation_rate}) cannot be greater " +
                               f"than the project discount rate ({self.npv_discount_rate}). Please edit the 'ter' value for {der_inst.name}.")
            if conflict_occured:
                raise ModelParameterError("TER and discount rates conflict. Please see log file for more information.")

    @staticmethod
    def get_years_after_failures(end_year, der_list):
        """ The optimization should be re-run for every year an 'unreplacable' piece of equipment fails before the
        lifetime of the longest-lived equipment. No need to re-run the optimization if equipment fails in some
        year and is replaced.

        Args:
            end_year (pd.Period): the last year the project is operational
            der_list (list): list of DERs initialized with user values

        Returns: list of the year(s) after an 'unreplacable' DER fails/reaches its end of life

        """
        rerun_opt_on = []
        for der_instance in der_list:
            last_operation_year = None
            if der_instance.tag == 'Battery' and der_instance.incl_cycle_degrade:
                # ignore battery's failure years as defined by user if user wants to include degradation in their analysis
                # instead set it to be the project's last year+1
                last_operation_year = end_year.year
            yrs_failed = der_instance.set_failure_years(end_year, last_operation_year)
            if not der_instance.replaceable:
                # if the DER is not replaceable then add the following year to the set of analysis years
                rerun_opt_on += yrs_failed
        # increase the year by 1 (this will be the years that the operational DER mix will change)
        diff_der_mix_yrs = [year+1 for year in rerun_opt_on if year < end_year.year]
        return list(set(diff_der_mix_yrs))  # get rid of any duplicates

    def annuity_scalar(self, opt_years):
        """Calculates an annuity scalar, used for sizing, to convert yearly costs/benefits
        this method is sometimes called before the class is initialized (hence it has to be
        static)

        Args:
            opt_years (list): List of years that the user wants to optimize--should be length=1

        Returns: the NPV multiplier

        """
        n = self.end_year.year - self.start_year.year
        dollar_per_year = np.ones(n)
        base_year = min(opt_years)
        yr_index = base_year - self.start_year.year
        while yr_index < n - 1:
            dollar_per_year[yr_index + 1] = dollar_per_year[yr_index] * (1 + self.inflation_rate)
            yr_index += 1
        yr_index = base_year - self.start_year.year
        while yr_index > 0:
            dollar_per_year[yr_index - 1] = dollar_per_year[yr_index] * (1 / (1 + self.inflation_rate))
            yr_index -= 1
        lifetime_npv_alpha = np.npv(self.npv_discount_rate, [0] + dollar_per_year)
        return lifetime_npv_alpha

    def calculate(self, technologies, value_streams, results, opt_years):
        """ this function calculates the proforma, cost-benefit, npv, and payback using the optimization variable results
        saved in results and the set of technology and service instances that have (if any) values that the user indicated
        they wanted to use when evaluating the CBA.

        Instead of using the technologies and services as they are passed in from the call in the Results class, we will pass
        the technologies and services with the values the user denoted to be used for evaluating the CBA.

        Args:
            technologies (list): all active technologies (provided access to ESS, generators, renewables to get capital and om costs)
            value_streams (Dict): Dict of all services to calculate cost avoided or profit
            results (DataFrame): DataFrame of all the concatenated timseries_report() method results from each DER
                and ValueStream
            opt_years (list)

        """
        self.initiate_cost_benefit_analysis(technologies, value_streams)
        super().calculate(self.ders, self.value_streams, results, opt_years)
        self.create_equipment_lifetime_report(self.ders)

    def initiate_cost_benefit_analysis(self, technologies, valuestreams):
        """ Prepares all the attributes in this instance of cbaDER with all the evaluation values.
        This function should be called before any finacial methods so that the user defined evaluation
        values are used

        Args:
            technologies (list): the management point of all active technology to access (needed to get capital and om costs)
            valuestreams (Dict): Dict of all services to calculate cost avoided or profit

        """
        # we deep copy because we do not want to change the original ValueStream objects
        self.value_streams = copy.deepcopy(valuestreams)
        self.ders = copy.deepcopy(technologies)

        self.place_evaluation_data()

    def place_evaluation_data(self):
        """ Place the data specified in the evaluation column into the correct places. This means all the monthly data,
        timeseries data, and single values are saved in their corresponding attributes within whatever ValueStream and DER
        that is active and has different values specified to evaluate the CBA with.

        """
        monthly_data = self.Scenario.get('monthly_data')
        time_series = self.Scenario.get('time_series')

        if time_series is not None or monthly_data is not None:
            for value_stream in self.value_streams.values():
                value_stream.update_price_signals(monthly_data, time_series)

        if 'customer_tariff' in self.Finance:
            self.tariff = self.Finance['customer_tariff']

        if 'User' in self.value_streams.keys():
            self.update_with_evaluation(self.value_streams['User'], self.valuestream_values['User'], self.verbose)

        for der_inst in self.ders:
            der_tag = der_inst.tag
            der_id = der_inst.id
            evaluation_inputs = self.ders_values.get(der_tag, {}).get(der_id)
            if evaluation_inputs is not None:
                der_inst.update_for_evaluation(evaluation_inputs)

    @staticmethod
    def update_with_evaluation(param_object, evaluation_dict, verbose):
        """Searches through the class variables (which are dictionaries of the parameters with values to be used in the CBA)
        and saves that value

        Args:
            param_object (DER, ValueStream): the actual object that we want to edit
            evaluation_dict (dict, None): keys are the string representation of the attribute where value is saved, and values
                are what the attribute value should be
            verbose (bool): true or fla

        Returns: the param_object with attributes set to the evaluation values instead of the optimization values

        """
        if evaluation_dict:  # evaluates true if dict is not empty and the value is not None
            for key, value in evaluation_dict.items():
                try:
                    setattr(param_object, key, value)
                    TellUser.debug('attribute (' + param_object.name + ': ' + key + ') set: ' + str(value))
                except KeyError:
                    TellUser.debug('No attribute ' + param_object.name + ': ' + key)

    def proforma_report(self, technologies, valuestreams, results, opt_years):
        """ Calculates and returns the proforma

        Args:
            technologies (list): list of technologies (needed to get capital and om costs)
            valuestreams (Dict): Dict of all services to calculate cost avoided or profit
            results (DataFrame): DataFrame of all the concatenated timseries_report() method results from each DER
                and ValueStream
            opt_years (list)

        Returns: dataframe proforma
        """
        proforma = super().proforma_report(technologies, valuestreams, results, opt_years)
        proforma_wo_yr_net = proforma.drop('Yearly Net Value', axis=1)
        proforma = self.replacement_costs(proforma_wo_yr_net, technologies)
        proforma = self.zero_out_dead_der_costs(proforma, technologies)
        proforma = self.update_capital_cost_construction_year(proforma, technologies)
        # add EOL costs to proforma
        der_eol = self.calculate_end_of_life_value(proforma, technologies, self.inflation_rate)
        proforma = proforma.join(der_eol)

        if self.ecc_mode:
            for der_inst in technologies:
                if der_inst.tag == "Load":
                    continue
                # replace capital cost columns with economic_carrying cost
                der_ecc_df, total_ecc = der_inst.economic_carrying_cost_report(self.inflation_rate, self.end_year, self.apply_escalation)
                # drop original Capital Cost
                proforma.drop(columns=[der_inst.zero_column_name()], inplace=True)
                # drop any replacement costs
                if f"{der_inst.unique_tech_id()} Replacement Costs" in proforma.columns:
                    proforma.drop(columns=[f"{der_inst.unique_tech_id()} Replacement Costs"], inplace=True)
                # add the ECC to the proforma
                proforma = proforma.join(total_ecc)
                # add ECC costs broken out by when initial cost occurs to complete DF
                self.ecc_df = pd.concat([self.ecc_df, der_ecc_df], axis=1)
        else:
            proforma = self.calculate_taxes(proforma, technologies)
        # check if there are are costs on CAPEX YEAR -- if there arent, then remove it from proforma
        if not proforma.loc['CAPEX Year', :].any():
            proforma.drop('CAPEX Year', inplace=True)
        # sort alphabetically
        proforma.sort_index(axis=1, inplace=True)
        proforma.fillna(value=0, inplace=True)
        # recalculate the net (sum of the row's columns)
        proforma['Yearly Net Value'] = proforma.sum(axis=1)
        return proforma

    def replacement_costs(self, proforma, technologies):
        """ takes the proforma and adds cash flow columns that represent any tax that was received or paid
        as a result

        Args:
            proforma (DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active
            technologies (list): Dict of technologies (needed to get capital and om costs)

        """
        replacement_df = pd.DataFrame()
        for der_inst in technologies:
            temp = der_inst.replacement_report(self.end_year, self.apply_escalation)
            if temp is not None and not temp.empty:
                replacement_df = pd.concat([replacement_df, temp], axis=1)
        proforma = proforma.join(replacement_df)
        proforma = proforma.fillna(value=0)
        return proforma

    def zero_out_dead_der_costs(self, proforma, technologies):
        """ Determines years of the project that a DER is past its expected lifetime, then
        zeros out the costs for those years (for each DER in the project)

        Args:
            proforma:
            technologies:

        Returns: updated proforma

        """
        no_more_der_yr = 0
        for der_isnt in technologies:
            last_operating_year = der_isnt.last_operation_year
            if der_isnt.tag != 'Load':
                no_more_der_yr = max(no_more_der_yr, last_operating_year.year)
            if not der_isnt.replaceable and self.end_year > last_operating_year:
                column_mask = proforma.columns.str.contains(der_isnt.unique_tech_id(), regex=False)
                proforma.loc[last_operating_year + 1:, column_mask] = 0

        # zero out all costs and benefits after the last equipement piece fails
        if self.end_year.year >= no_more_der_yr + 1 >= self.start_year.year:
            proforma.loc[pd.Period(no_more_der_yr + 1, freq='y'):, ] = 0
        return proforma

    @staticmethod
    def update_capital_cost_construction_year(proforma, technologies):
        """ Determines years of the project that a DER is past its expected lifetime, then
        zeros out the costs for those years (for each DER in the project)

        Args:
            proforma:
            technologies:

        Returns: updated proforma

        """
        for der_isnt in technologies:
            capex_df = der_isnt.put_capital_cost_on_construction_year(proforma.index)
            proforma.update(capex_df)
        return proforma

    def calculate_end_of_life_value(self, proforma, technologies, inflation_rate):
        """ takes the proforma and adds cash flow columns that represent any tax that was received or paid
        as a result

        Args:
            proforma (DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active
            technologies (list): Dict of technologies (needed to get capital and om costs)

        """
        end_of_life_costs = pd.DataFrame(index=proforma.index)
        for der_inst in technologies:
            temp = pd.DataFrame(index=proforma.index)
            # collect the decommissioning costs at the technology's end of life
            decommission_pd = der_inst.decommissioning_report(self.end_year)
            if decommission_pd is not None and not decommission_pd.empty:
                # apply inflation rate from operation year
                decommission_pd = Financial.apply_escalation(decommission_pd, inflation_rate, self.start_year.year)
                temp = temp.join(decommission_pd)

            salvage_pd = der_inst.salvage_value_report(self.end_year)
            if salvage_pd is not None and not salvage_pd.empty:
                # apply technology escalation rate from operation year
                salvage_pd = Financial.apply_escalation(salvage_pd, der_inst.escalation_rate, der_inst.operation_year.year)
                temp = temp.join(salvage_pd)
            end_of_life_costs = end_of_life_costs.join(temp)
        end_of_life_costs = end_of_life_costs.fillna(value=0)

        return end_of_life_costs

    def calculate_taxes(self, proforma, technologies):
        """ takes the proforma and adds cash flow columns that represent any tax that was received or paid
        as a result, then recalculates the Yearly Net Value column

        Args:
            proforma (DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active
            technologies (list): Dict of technologies (needed to get capital and om costs)

        Returns: proforma considering the 'Overall Tax Burden'

        """
        proj_years = len(proforma) - 1
        yearly_net = proforma.drop('CAPEX Year').sum(axis=1).values

        # 1) Redistribute capital cost columns according to the DER's MACRS value
        capital_costs = np.zeros(proj_years)
        for der_inst in technologies:
            tax_contribution = der_inst.tax_contribution(self.macrs_depreciation, proj_years)
            if tax_contribution is not None:
                capital_costs += tax_contribution
        yearly_net += capital_costs

        # 2) Calculate State tax based on the net cash flows in each year
        state_tax = yearly_net * self.state_tax_rate

        # 3) Calculate Federal tax based on the net cash flow in each year minus State taxes from that year
        yearly_net_post_state_tax = yearly_net - state_tax
        federal_tax = yearly_net_post_state_tax * self.federal_tax_rate

        # 4) Add the overall tax burden (= state tax + federal tax) to proforma, make sure columns are ordered s.t. yrly net is last
        overall_tax_burden = state_tax + federal_tax

        proforma['State Tax Burden'] = np.insert(state_tax, 0, 0)
        proforma['Federal Tax Burden'] = np.insert(federal_tax, 0, 0)
        proforma['Overall Tax Burden'] = np.insert(overall_tax_burden, 0, 0)
        return proforma

    def payback_report(self, techologies, proforma, opt_years):
        """ calculates and saves the payback period and discounted payback period in a dataframe

        Args:
            techologies (list)
            proforma (DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active
            opt_years (list)

        """
        super().payback_report(techologies, proforma, opt_years)
        npv_df = pd.DataFrame({'Lifetime Net Present Value':  self.npv['Lifetime Present Value'].values},
                              index=pd.Index(['$'], name="Unit"))
        other_metrics = pd.DataFrame({'Internal Rate of Return': self.internal_rate_of_return(proforma),
                                     'Benefit-Cost Ratio': self.benefit_cost_ratio(self.cost_benefit)},
                                     index=pd.Index(['-'], name='Unit'))
        self.payback = pd.merge(self.payback, npv_df, how='outer', on='Unit')
        self.payback = pd.merge(self.payback, other_metrics, how='outer', on='Unit')

    @staticmethod
    def internal_rate_of_return(proforma):
        """ calculates the discount rate that would return lifetime NPV= 0

        Args:
            proforma (DataFrame): Pro-forma DataFrame that was created from each ValueStream or DER active

        Returns: internal rate of return

        """
        return np.irr(proforma['Yearly Net Value'].values)

    @staticmethod
    def benefit_cost_ratio(cost_benefit):
        """ calculate the cost-benefit ratio

        Args:
            cost_benefit (DataFrame):

        Returns: discounted cost/discounted benefit

        """
        lifetime_discounted_cost = cost_benefit.loc['Lifetime Present Value', 'Cost ($)']
        lifetime_discounted_benefit = cost_benefit.loc['Lifetime Present Value', 'Benefit ($)']
        if np.isclose(lifetime_discounted_cost, 0):
            return np.nan
        return lifetime_discounted_benefit/lifetime_discounted_cost

    def create_equipment_lifetime_report(self, der_lst):
        """

        Args:
            der_lst:

        """
        data = {
            der_inst.unique_tech_id(): [der_inst.construction_year, der_inst.operation_year, der_inst.last_operation_year, der_inst.expected_lifetime]
            for der_inst in der_lst
        }
        self.equipment_lifetime_report = pd.DataFrame(data, index=['Beginning of Life', 'Operation Begins', 'End of Life', 'Expected Lifetime'])
