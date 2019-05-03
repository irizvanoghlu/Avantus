"""
cbaDER.py

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
import storagevet.Library as Lib
import storagevet.Finances as Finances
import copy

SATURDAY = 5


class cbaDER(Finances):

    def __init__(self, params, time_series, tariff, monthly_data, opt_years, dt):
        """ Initialized cbaDER object for case

         Args:
            params (Dict): input parameters
            time_series (DataFrame): inputted time series data
            tariff (DataFrame): inputted retail tarrif data
            monthly_data (DataFrame): inputted monthly data
            opt_years (list): years to run optimization over. need this separate from params since deferral feasibility check can change this
            dt (float): optimization timestep
        """

        # assign important financial attributes
        self.tariff = tariff
        self.dt = dt
        self.n = params['n']
        self.start_year = params['start_year']
        self.end_year = params['end_year']
        self.opt_years = opt_years
        self.inflation_rate = params['inflation_rate']
        self.npv_discount_rate = params['npv_discount_rate']
        self.state_tax_rate = params['state_tax_rate']
        self.property_tax_rate = params['property_tax_rate']
        self.federal_tax_rate = params['federal_tax_rate']
        self.service_list = params['services']
        self.predispatch_list = params['predispatch']
        self.growth_rates = {'default': params['def_growth']}
        # which columns are needed from monthly_data [input validation logic]
        self.required_price_cols = []
        if 'Backup' in self.predispatch_list:
            self.required_price_cols += ['backup_price']
        if 'RA' in self.predispatch_list:
            self.required_price_cols += ['ra_price']
        if 'DR' in self.predispatch_list:
            self.required_price_cols += ['dr_cap_price', 'dr_ene_price']
        self.monthly_prices = monthly_data[self.required_price_cols]

        # create financial inputs data table
        self.fin_inputs = None
        self.prep_fin_inputs(time_series, params)

        # prep outputs
        self.obj_val = pd.DataFrame(index=np.sort(self.fin_inputs['opt_agg'].unique()))
        self.pro_forma = pd.DataFrame()
        self.npv = pd.DataFrame()
        self.cost_benefit = pd.DataFrame()
        self.adv_monthly_bill = None
        self.sim_monthly_bill = None

    def prep_fin_inputs(self, time_series):
        """
        Create standard dataframe of prices from user time_series input

        Args:
            time_series (DataFrame): ParamsDER of time series prices

        """
        # create basic dataframe with dttm columns
        outputs_df = time_series

        outputs_df['year'] = time_series.index.to_period('Y')
        outputs_df['yr_mo'] = time_series.index.to_period('M')
        outputs_df['date'] = time_series.index.to_period('D')
        outputs_df['weekday'] = (outputs_df.index.weekday < SATURDAY).astype('int64')
        outputs_df['he'] = (time_series.index + pd.Timedelta('1s')).hour + 1

        # only join needed price columns from time_series [input validation logic]
        for col in list(time_series):
            outputs_df[col] = time_series[col].values

        # merge in monthly prices
        if not self.monthly_prices.empty:
            outputs_df = outputs_df.reset_index().merge(self.monthly_prices.reset_index(), on='yr_mo', how='left').set_index(outputs_df.index.names)

        # calculate data for simulation of future years using growth rate
        outputs_df = self.add_growth_data(outputs_df, self.opt_years, self.verbose)

        # create opt_agg column (has to happen after adding future years)
        if self.mpc:
            outputs_df = Lib.create_opt_agg(outputs_df, "mpc", self.dt)
        else:
            outputs_df = Lib.create_opt_agg(outputs_df, self.n, self.dt)

        return outputs_df

    def calc_retail_energy_price(self):
        """ Calculates retail energy rates with given data and adds the retail prices into the time_series data frame.

        ToDo: this is a messy function and needs to be cleaned up

        """
        size = len(self.fin_inputs.index)
        # Build Energy Price Vector
        self.fin_inputs.loc[:, 'p_energy'] = [0 for _ in range(size)]

        billing_period = [[] for _ in range(size)]

        for p in range(len(self.tariff)):
            # edit the pricedf energy price and period values for all of the periods defined
            # in the tariff input file
            bill = self.tariff.iloc[p, :]
            month_mask = (bill["Start_month"] <= self.fin_inputs['yr_mo'].apply((lambda x: x.month))) & \
                         (self.fin_inputs['yr_mo'].apply((lambda x: x.month)) <= bill["End_month"])
            time_mask = (bill['Start_time'] <= self.fin_inputs['he']) & (self.fin_inputs['he'] <= bill['End_time'])
            weekday_mask = True
            exclud_mask = False
            if not bill['weekday'] == 2:
                weekday_mask = bill['weekday'] == self.fin_inputs['weekday']
            if not np.isnan(bill['Excluding_start']) and np.isnan(bill['Excluding_end']):
                exclud_mask = (bill['Excluding_start'] <= self.fin_inputs['he']) & (self.fin_inputs['he'] <= bill['Excluding_end'])
            mask = np.array(month_mask & time_mask & np.logical_not(exclud_mask) & weekday_mask)
            # Add energy prices
            self.fin_inputs.loc[mask, 'p_energy'] = self.tariff.loc[p+1, 'Energy_price']  # CODE BREAKING HERE
            mask = np.where(mask)[0]  # COME BACK
            for j in range(len(mask)):
                billing_period[mask[j]].append(p + 1)
        billing_period = pd.DataFrame({'billing_period': billing_period}, dtype='object')
        self.fin_inputs.loc[:, 'billing_period'] = billing_period.values

        # ADD CHECK TO MAKE SURE ENERGY PRICES ARE THE SAME FOR EACH OVERLAPPING BILLING PERIOD
        # Check to see that each timestep has a period assigned to it
        if not billing_period.apply(len).all():
            print('The billing periods in the input file do not partition the year')
            print('please check the tariff input file')
            while True:
                yn = eval(input('Do you wish to exit? (y/n)\n'))
                if yn is 'y':
                    print('exiting...')
                    sys.exit()
                elif yn is 'n':
                    print('Alright... Pressing on...')
                    print('but, FYI, this is what we\'re working with')
                    plt.figure()
                    plt.plot(self.fin_inputs['p_energy'])
                    plt.title('Energy Prices')
                    plt.xlabel('Time Step')
                    plt.ylabel('Energy Price ($/kWh)')
                    plt.draw()
                    plt.show()
                    break
                else:
                    print('please enter y or n')

    def calc_energy_bill(self, results):
        """ Calculates the retail energy bill for the optimal dispatch by billing period and by month.

        Args:
            results (DataFrame): Dataframe with all the optimization solutions
        """

        results = copy.deepcopy(results)
        # results.loc[:, 'net_power'] = results.loc[:, 'dis'] - results.loc[:, 'ch'] - results.loc[:, 'load'] + results.loc[:, 'ac_gen'] + results.loc[:, 'dc_gen']
        results.loc[:, 'net_power'] = results.loc[:, 'dis'] - results.loc[:, 'ch'] - results.loc[:, 'load'] + results.loc[:, 'pv_out']
        results.loc[:, 'original_net_power'] = - results.loc[:, 'load'] + results.loc[:, 'ac_gen'] + results.loc[:, 'dc_gen']
        # calculate energy cost every time step
        results.loc[:, 'energy_cost'] = -self.dt * np.multiply(results.loc[:, 'net_power'], self.fin_inputs.loc[:, 'p_energy'])
        results.loc[:, 'original_energy_cost'] = -self.dt * np.multiply(results.loc[:, 'original_net_power'], self.fin_inputs.loc[:, 'p_energy'])
        # Calculate Demand Charge per month
        monthly_bill = pd.DataFrame()
        for item in range(len(self.tariff)):
            bill = self.tariff.iloc[item, :]
            month_mask = (bill["Start_month"] <= results['yr_mo'].apply((lambda x: x.month))) & \
                         (results['yr_mo'].apply((lambda x: x.month)) <= bill["End_month"])
            time_mask = (bill['Start_time'] <= results['he']) & (results['he'] <= bill['End_time'])
            weekday_mask = True
            exclud_mask = False
            if not bill['weekday'] == 2:
                weekday_mask = bill['weekday'] == self.fin_inputs['weekday']
            if bill['Excluding_start'] and bill['Excluding_end']:
                exclud_mask = (bill['Excluding_start'] <= results['he']) & (results['he'] <= bill['Excluding_end'])
            demand_rate = bill['Demand_rate']

            temp_df = results[month_mask & time_mask & np.logical_not(exclud_mask) & weekday_mask]
            demand = -temp_df.groupby(by=['yr_mo'])[['net_power', 'original_net_power']].min() * demand_rate
            demand.columns = pd.Index(['Demand Charge ($)', 'Original Demand Charge ($)'])

            retail = temp_df.groupby(by=['yr_mo'])[['energy_cost', 'original_energy_cost']].sum()
            retail.columns = pd.Index(['Energy Charge ($)', 'Original Energy Charge ($)'])

            billing_pd = pd.Series(np.repeat(bill.name, len(demand)), name='Billing Period', index=retail.index)

            temp_bill = pd.concat([demand, retail, billing_pd], sort=False, axis=1)
            monthly_bill = monthly_bill.append(temp_bill)
        # monthly_bill.columns = pd.Index(['demand_charge', 'original_demand_charge', 'energy_charge', 'original_energy_charge'])
        monthly_bill = monthly_bill.sort_index(axis=0)
        self.adv_monthly_bill = monthly_bill
        self.adv_monthly_bill.index.name = 'Month-Year'
        self.sim_monthly_bill = monthly_bill.groupby(monthly_bill.index.name).sum()
        for month_yr_index in monthly_bill.index.unique():
            mo_yr_data = monthly_bill.loc[month_yr_index, :]
            if mo_yr_data.ndim > 1:
                billing_periods = ', '.join(str(int(pd)) for pd in mo_yr_data['Billing Period'].values)
            else:
                billing_periods = str(int(mo_yr_data['Billing Period']))
            self.sim_monthly_bill.loc[month_yr_index, 'Billing Period'] = '[' + billing_periods + ']'
        self.sim_monthly_bill.index.name = 'Month-Year'

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

    def add_growth_data(self, df, opt_years, dt, verbose=False):
        """ Helper function: Adds rows to df where missing opt_years

        Args:
            df (DataFrame): given data
            opt_years (List): List of Period years where we need data for
            dt (float): optimization time step
            verbose (bool):

        Returns:
            df (Data Frame)

        Todo: remove some of the function inputs (can be pulled from class attributes)
        """

        data_year = df.year.unique()  # which years was data given for
        # data_year = [item.year for item in data_year]
        no_data_year = {pd.Period(year) for year in opt_years} - {pd.Period(year) for year in data_year}  # which years do we not have data for

        # if there is a year we dont have data for
        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = pd.Period(max(data_year))  # which year to to apply growth rate to (is this the logic we want??)

                # create new dataframe for missing year
                new_index = pd.DatetimeIndex(start='01/01/' + str(yr), end='01/01/' + str(yr + 1), freq=pd.Timedelta(self.dt, unit='h'), closed='right')
                new_data = Lib.create_outputs_df(new_index)

                source_data = df[df['year'] == source_year]  # use source year data

                def_rate = self.growth_rates['default']
                growth_cols = self.required_price_cols

                # for each column in growth column
                # TODO: not sure if this can handle FR growth come back to check
                for col in growth_cols:
                    name = col.split(sep='_')[0].upper()
                    if name in self.growth_rates.keys():
                        rate = self.growth_rates[name]
                    else:
                        print((name, ' rate not in params. Using default growth rate:', def_rate)) if verbose else None
                        rate = def_rate
                    new_data[col] = Lib.apply_growth(source_data[col], rate, source_year, yr, dt)  # apply growth rate to column

                # add new year to original data frame
                df = pd.concat([df, new_data], sort=True)

        return df

    def add_price_growth(self, rate):
        """ Updates the growth_rates attribute with price related, which will be used within the add_growth function.

        Args:
            rate (Dict): key is the name of item of which the rate applies to its price (ie. DA or FR), and the value is the rate value

        """
        self.growth_rates.update(rate)

    def __eq__(self, other, compare_init=False):
        """ Determines whether Analysis object equals another Analysis object. Compare_init = True will do an initial
        comparison ignoring any attributes that are changed in the course of running a case.

        Args:
            other (Analysis): Analysis object to compare
            compare_init (bool): Flag to ignore attributes that change after initialization

        Returns:
            bool: True if objects are close to equal, False if not equal.
        """
        return Lib.compare_class(self, other, compare_init)
