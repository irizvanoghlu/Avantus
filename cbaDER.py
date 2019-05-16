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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import svet_helper as sh
import copy
from storagevet.Finances import Financial
import re

SATURDAY = 5


class CostBenDER(Financial):

    def __init__(self, params):
        """ Initialized Financial object for case

         Args:
            params (Dict): input parameters
        """
        Financial.__init__(self, params)

    def yearly_financials(self, technologies, services, results, use_inflation=True):
        """ Aggregates optimization period financials to yearly and interpolated between optimization years

        Args:
            technologies (Dict): Dict of technologies (needed to get capital and om costs)
            services (Dict): Dict of services to calculate cost avoided or profit
            results (DataFrame): DataFrame with all the optimization variable solutions
            use_inflation (bool): Flag to determine if using inflation rate to determine financials for extrapolation. If false, use extrapolation

        """
        # create yearly data table
        yr_index = pd.period_range(start=self.start_year, end=self.end_year, freq='y')
        yr_index = np.insert(yr_index.values, 0, 'CAPEX Year')
        pro_forma = pd.DataFrame(index=yr_index)

        # add capital costs for each technology (only the battery at this time)
        # for tech in technologies.values():
        tech = technologies['Storage']
        ccost_0 = tech.ccost
        ccost_0 += tech.ccost_kw * tech.dis_max_rated.value
        ccost_0 += tech.ccost_kwh * tech.ene_max_rated.value
        pro_forma.loc['CAPEX Year', 'Battery Capital Cost'] = -ccost_0

        fixed_om = tech.fixedOM * tech.dis_max_rated.value
        pro_forma.loc[self.start_year:, 'Battery Fixed O&M'] = np.repeat(fixed_om, len(pro_forma.index)-1)

        # calculate benefit for each service
        if "DCM" in services.keys() or "retailTimeShift" in services.keys():  # AKA if customer-sided
            for year in self.adv_monthly_bill.index.year.unique():
                year_monthly = self.adv_monthly_bill[self.adv_monthly_bill.index.year == year]

                new_demand_charge = year_monthly['Demand Charge ($)'].values.sum()
                orig_demand_charge = year_monthly['Original Demand Charge ($)'].values.sum()
                pro_forma.loc[pd.Period(year=year, freq='y'), 'Avoided Demand Charge'] = orig_demand_charge - new_demand_charge

                new_energy_charge = year_monthly['Energy Charge ($)'].values.sum()
                orig_energy_charge = year_monthly['Original Energy Charge ($)'].values.sum()
                pro_forma.loc[pd.Period(year=year, freq='y'), 'Avoided Energy Charge'] = orig_energy_charge - new_energy_charge

                mask = results['year'] == year  # select data for one year at a time
                tot_energy_delievered_es = results[mask]['dis'].values.sum() * len(results.index) * self.dt
                pro_forma.loc[pd.Period(year=year, freq='y'), 'Battery Variable O&M'] = tot_energy_delievered_es*tech.OMexpenses/1000
        # list of all the financials that are constant
        const_col = ['Battery Fixed O&M']
        # list of financials that are zero unless already specified
        zero_col = ['Battery Capital Cost']
        # the rest of columns should grow year over year
        growth_col = list(set(list(pro_forma)) - set(const_col) - set(zero_col))

        # set the 'CAPEX Year' row to all zeros
        pro_forma.loc['CAPEX Year', growth_col+const_col] = np.zeros(len(growth_col+const_col))
        # use linear interpolation for growth in between optimization years
        pro_forma[growth_col] = pro_forma[growth_col].apply(lambda x: x.interpolate(method='linear', limit_area='inside'), axis=0)

        if use_inflation:
            # forward fill growth columns with inflation
            last_sim = max(self.opt_years)
            for yr in pd.period_range(start=last_sim+1, end=self.end_year, freq='y'):
                pro_forma.loc[yr, growth_col] = pro_forma.loc[yr-1, growth_col]*(1+self.inflation_rate/100)
            # backfill growth columns (needed for year 0)
            pro_forma[growth_col] = pro_forma[growth_col].fillna(value=0)
            # fill in constant columns
            pro_forma[const_col] = pro_forma[const_col].fillna(method='ffill')
            # fill in zero columns
            pro_forma[zero_col] = pro_forma[zero_col].fillna(value=0)

        else:
            # extrapolate TODO Test this
            pro_forma[growth_col] = pro_forma[growth_col].apply(lambda x: x.interpolate(method='polynomial', order=3, limit_area='outside'), axis=0)  # is this what we want???
            pro_forma = pro_forma.interpolate(method='linear', limit_area='inside')  # is this what we want???
            pro_forma = pro_forma.interpolate(method='polynomial', order=3, limit_area='outside')  # is this what we want???
            # fill in fixed O&M columns
            pro_forma[const_col] = pro_forma[const_col].fillna(method='ffill')
            # fill in zero columns
            pro_forma[zero_col] = pro_forma[zero_col].fillna(value=0)

        # prepare for cost benefit (we dont want to include net values, so we do this first)
        cost_df = pd.DataFrame(pro_forma.values.clip(max=0))
        cost_df.columns = pro_forma.columns
        benefit_df = pd.DataFrame(pro_forma.values.clip(min=0))
        benefit_df.columns = pro_forma.columns

        # calculate the net (sum of the row's columns)
        pro_forma['Yearly Net Value'] = pro_forma.sum(axis=1)
        self.pro_forma = pro_forma

        # CALCULATING NET PRESENT VALUES
        # use discount rate to calculate NPV for net
        discount_rate = self.npv_discount_rate/100
        npv_dict = {}
        # NPV for growth_cols
        for col in pro_forma.columns:
            npv_dict.update({col: [np.npv(discount_rate / 100, pro_forma[col].values)]})
        self.npv = pd.DataFrame(npv_dict, index=pd.Index(['NPV']))

        # CALCULATING COST-BENEFIT TABLE
        cost_pv = 0  # cost present value (discounted cost)
        benefit_pv = 0  # benefit present value (discounted benefit)
        self.cost_benefit = pd.DataFrame({'Lifetime Present Value': [0,0]}, index=pd.Index(['Cost ($)', 'Benefit ($)']))
        for col in cost_df.columns:
            present_cost = np.npv(discount_rate, cost_df[col].values)
            present_benefit = np.npv(discount_rate, benefit_df[col].values)

            self.cost_benefit[col] = [np.abs(present_cost), present_benefit]

            cost_pv += present_cost
            benefit_pv += present_benefit
        self.cost_benefit['Lifetime Present Value'] = [np.abs(cost_pv), benefit_pv]
        # self.cost_benefit = pd.DataFrame({'Cost ($)': [np.abs(cost_pv)],
        #                                   'Benefit ($)': [benefit_pv]}, index=pd.Index(['Lifetime Present Value']))
