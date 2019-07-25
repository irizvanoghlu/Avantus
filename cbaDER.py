"""
Finances.py

This Python class contains methods and attributes vital for completing financial analysis given optimal dispathc.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import logging
from storagevet.Finances import Financial


SATURDAY = 5

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')


class CostBenDER(Financial):

    def __init__(self, params):
        """ Initialized Financial object for case

         Args:
            params (Dict): input parameters
        """
        Financial.__init__(self, params)

    # def yearly_financials(self, technologies, services, results, use_inflation=True):
    #     """ Aggregates optimization period financials to yearly and interpolated between optimization years
    #
    #     Args:
    #         technologies (Dict): Dict of technologies (needed to get capital and om costs)
    #         services (Dict): Dict of services to calculate cost avoided or profit
    #         results (DataFrame): DataFrame with all the optimization variable solutions
    #         use_inflation (bool): Flag to determine if using inflation rate to determine financials for extrapolation. If false, use extrapolation
    #
    #     """
    #     # create yearly data table
    #     yr_index = pd.period_range(start=self.start_year, end=self.end_year, freq='y')
    #     project_index = np.insert(yr_index.values, 0, 'CAPEX Year')
    #     pro_forma = pd.DataFrame(index=project_index)
    #
    #     # list of all the financials that are constant
    #     const_col = []
    #     # list of financials that are zero unless already specified
    #     zero_col = []
    #
    #     for tech in technologies.values():
    #         name = tech.name
    #
    #         # add capital costs
    #         col_name = name + ' Capital Cost'
    #         zero_col += [col_name]
    #         pro_forma.loc['CAPEX Year', col_name] = -tech.capex
    #
    #         # add fixed o&m costs
    #         col_name = name + ' Fixed O&M Cost'
    #         const_col += [col_name]
    #         pro_forma.loc[self.start_year:, col_name] = np.repeat(-tech.fixed_om, len(pro_forma.index) - 1)
    #
    #         # add variable o&m costs
    #         vari_om_costs = tech.variable_costs(self.opt_years)
    #         if list(vari_om_costs.columns)[0]:
    #             # checks to see if default (empty) df was returned (see DER documentation)
    #             pro_forma = pro_forma.join(vari_om_costs)
    #
    #         # add fuel costs
    #         fuel_costs = tech.fuel_costs(self.opt_years)
    #         if list(fuel_costs.columns)[0]:
    #             # checks to see if default (empty) df was returned (see DER documentation)
    #             pro_forma = pro_forma.join(fuel_costs)
    #
    #         # add startup costs
    #         startup_costs = tech.startup_costs(self.opt_years)
    #         if list(startup_costs.columns)[0]:
    #             # checks to see if default (empty) df was returned (see DER documentation)
    #             pro_forma = pro_forma.join(startup_costs)
    #
    #     # calculate benefit for each service
    #     if "DCM" in services.keys():
    #         for year in yr_index.year.unique():
    #             billing_pd_charges = services['DCM'].billing_period_bill
    #             year_monthly = billing_pd_charges[billing_pd_charges.index.year == year]
    #
    #             new_demand_charge = year_monthly['Demand Charge ($)'].values.sum()
    #             orig_demand_charge = year_monthly['Original Demand Charge ($)'].values.sum()
    #             pro_forma.loc[pd.Period(year=year, freq='y'), 'Avoided Demand Charge'] = orig_demand_charge - new_demand_charge
    #
    #     if "retailTimeShift" in services.keys():
    #         for year in yr_index.year.unique():
    #             monthly_energy_bill = services['retailTimeShift'].monthly_bill
    #             year_monthly = monthly_energy_bill[monthly_energy_bill.index.year == year]
    #
    #             new_energy_charge = year_monthly['Energy Charge ($)'].values.sum()
    #             orig_energy_charge = year_monthly['Original Energy Charge ($)'].values.sum()
    #             pro_forma.loc[pd.Period(year=year, freq='y'), 'Avoided Energy Charge'] = orig_energy_charge - new_energy_charge
    #
    #     if "DA" in services.keys():
    #         energy_cost = self.dt * np.multiply(results.loc[:, 'Net Load (kW)'], results.loc[:, 'DA Price Signal ($/kWh)'])
    #         for year in yr_index.year.unique():
    #             year_monthly = energy_cost[energy_cost.index.year == year]
    #             pro_forma.loc[pd.Period(year=year, freq='y'), 'DA ETS'] = year_monthly.sum()
    #
    #     if "FR" in services.keys():
    #         reg_up = results.loc[:, 'Regulation Up Bid (Charging) (kW)'] + results.loc[:, 'Regulation Up Bid (Discharging) (kW)']
    #         reg_down = results.loc[:, 'Regulation Down Bid (Charging) (kW)'] + results.loc[:, 'Regulation Down Bid (Discharging) (kW)']
    #
    #         energy_through_prof = np.multiply(results.loc[:, "FR Energy Throughput (kWh)"], results.loc[:, "FR Energy Settlement Price Signal ($/kWh)"])
    #         regulation_up_prof = np.multiply(reg_up, results.loc[:, "Regulation Up Price Signal ($/kW)"])
    #         regulation_down_prof = np.multiply(reg_down, results.loc[:, "Regulation Down Price Signal ($/kW)"])
    #         # combine all potential value streams into one df for faster splicing into years
    #         fr_results = pd.DataFrame({'Energy': energy_through_prof,
    #                                    'Reg up': regulation_up_prof,
    #                                    'Reg down': regulation_down_prof}, index=results.index)
    #         for year in yr_index.year.unique():
    #             year_monthly = fr_results[fr_results.index.year == year]
    #             pro_forma.loc[pd.Period(year=year, freq='y'), 'FR Energy Throughput'] = year_monthly['Energy'].sum()
    #             pro_forma.loc[pd.Period(year=year, freq='y'), 'Regulation Up'] = year_monthly['Reg up'].sum()
    #             pro_forma.loc[pd.Period(year=year, freq='y'), 'Regulation Down'] = year_monthly['Reg down'].sum()
    #
    #     if "SR" in services.keys():
    #         spin_bid = results.loc[:, 'Spinning Reserve Bid (Charging) (kW)'] + results.loc[:, 'Spinning Reserve Bid (Discharging) (kW)']
    #         spinning_prof = np.multiply(spin_bid, results.loc[:, 'SR Price Signal ($/kW)']) * self.dt
    #         for year in yr_index.year.unique():
    #             year_monthly = spinning_prof[spinning_prof.index.year == year]
    #             pro_forma.loc[pd.Period(year=year, freq='y'), 'Spinning Reserves'] = year_monthly.sum()
    #
    #     if "NSR" in services.keys():
    #         nonspin_bid = results.loc[:, 'Non-spinning Reserve Bid (Charging) (kW)'] + results.loc[:, 'Non-spinning Reserve Bid (Discharging) (kW)']
    #         nonspinning_prof = np.multiply(nonspin_bid, results.loc[:, 'NSR Price Signal ($/kW)']) * self.dt
    #         for year in yr_index.year.unique():
    #             year_monthly = nonspinning_prof[nonspinning_prof.index.year == year]
    #             pro_forma.loc[pd.Period(year=year, freq='y'), 'Non-Spinning Reserves'] = year_monthly.sum()
    #
    #     # the rest of columns should grow year over year
    #     growth_col = list(set(list(pro_forma)) - set(const_col) - set(zero_col))
    #
    #     # set the 'CAPEX Year' row to all zeros
    #     pro_forma.loc['CAPEX Year', growth_col + const_col] = np.zeros(len(growth_col + const_col))
    #     # use linear interpolation for growth in between optimization years
    #     pro_forma[growth_col] = pro_forma[growth_col].apply(lambda x: x.interpolate(method='linear', limit_area='inside'), axis=0)
    #
    #     if use_inflation:
    #         # TODO: fill in between analysis years --HN
    #         # forward fill growth columns with inflation
    #         last_sim = max(self.opt_years)
    #         for yr in pd.period_range(start=last_sim + 1, end=self.end_year, freq='y'):
    #             pro_forma.loc[yr, growth_col] = pro_forma.loc[yr - 1, growth_col] * (1 + self.inflation_rate / 100)
    #         # backfill growth columns (needed for year 0)
    #         pro_forma[growth_col] = pro_forma[growth_col].fillna(value=0)
    #         # fill in constant columns
    #         pro_forma[const_col] = pro_forma[const_col].fillna(method='ffill')
    #         # fill in zero columns
    #         pro_forma[zero_col] = pro_forma[zero_col].fillna(value=0)
    #
    #     else:
    #         # extrapolate TODO Test this
    #         pro_forma[growth_col] = pro_forma[growth_col].apply(lambda x: x.interpolate(method='polynomial', order=3, limit_area='outside'),
    #                                                             axis=0)  # is this what we want???
    #         pro_forma = pro_forma.interpolate(method='linear', limit_area='inside')  # is this what we want???
    #         pro_forma = pro_forma.interpolate(method='polynomial', order=3, limit_area='outside')  # is this what we want???
    #         # fill in fixed O&M columns
    #         pro_forma[const_col] = pro_forma[const_col].fillna(method='ffill')
    #         # fill in zero columns
    #         pro_forma[zero_col] = pro_forma[zero_col].fillna(value=0)
    #
    #     # prepare for cost benefit (we dont want to include net values, so we do this first)
    #     cost_df = pd.DataFrame(pro_forma.values.clip(max=0))
    #     cost_df.columns = pro_forma.columns
    #     benefit_df = pd.DataFrame(pro_forma.values.clip(min=0))
    #     benefit_df.columns = pro_forma.columns
    #
    #     # calculate the net (sum of the row's columns)
    #     pro_forma['Yearly Net Value'] = pro_forma.sum(axis=1)
    #     self.pro_forma = pro_forma
    #
    #     # CALCULATING NET PRESENT VALUES
    #     # use discount rate to calculate NPV for net
    #     npv_dict = {}
    #     # NPV for growth_cols
    #     for col in pro_forma.columns:
    #         npv_dict.update({col: [np.npv(self.npv_discount_rate / 100, pro_forma[col].values)]})
    #     self.npv = pd.DataFrame(npv_dict, index=pd.Index(['NPV']))
    #
    #     # CALCULATING COST-BENEFIT TABLE
    #     cost_pv = 0  # cost present value (discounted cost)
    #     benefit_pv = 0  # benefit present value (discounted benefit)
    #     self.cost_benefit = pd.DataFrame({'Lifetime Present Value': [0, 0]}, index=pd.Index(['Cost ($)', 'Benefit ($)']))
    #     for col in cost_df.columns:
    #         present_cost = np.npv(self.npv_discount_rate / 100, cost_df[col].values)
    #         present_benefit = np.npv(self.npv_discount_rate / 100, benefit_df[col].values)
    #
    #         self.cost_benefit[col] = [np.abs(present_cost), present_benefit]
    #
    #         cost_pv += present_cost
    #         benefit_pv += present_benefit
    #     self.cost_benefit['Lifetime Present Value'] = [np.abs(cost_pv), benefit_pv]
