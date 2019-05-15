"""
Result.py

"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']


import pandas as pd
import logging
import copy
import numpy as np
from pathlib import Path
import os

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')


class Result:
    """

    """

    @classmethod
    def initialize(cls):
        """
            Initialize the class variable of the Result class that will be used to create Result objects for analyses.
            Specifically, it will preload the needed CSV and/or time series data, and identify sensitivity variables.

        """
        cls.instances = dict()
        # cls.template = cls()

    def __init__(self, scenario):
        """ Initialize all Result objects, given a Scenario object with the following attributes.

            Args:
                scenario (Scenario.Scenario): scenario object after optimization has run to completion
        """
        self.opt_results = scenario.power_kw
        self.active_objects = scenario.active_objects
        self.customer_sided = scenario.customer_sided
        self.frequency = scenario.frequency
        self.dt = scenario.dt
        self.verbose_opt = scenario.verbose_opt
        self.n = scenario.n
        self.n_control = scenario.n_control
        self.mpc = scenario.mpc

        self.start_year = scenario.start_year
        self.end_year = scenario.end_year
        self.opt_years = scenario.opt_years
        self.incl_site_load = scenario.incl_site_load
        self.incl_aux_load = scenario.incl_aux_load
        self.incl_binary = scenario.incl_binary
        self.incl_slack = scenario.incl_slack
        self.power_growth_rates = scenario.growth_rates
        self.technologies = scenario.technologies
        self.services = scenario.services
        self.predispatch_services = scenario.predispatch_services
        self.financials = scenario.financials
        self.verbose = scenario.verbose
        # self.pv = scenario.pv
        self.opt_results = scenario.power_kw
        self.results_path = scenario.results_path

        # outputted DataFrames
        self.dispatch_map = pd.DataFrame()
        self.peak_day_load = pd.DataFrame()
        self.results = pd.DataFrame()
        self.energyp_map = pd.DataFrame()

    def post_analysis(self):
        """ Wrapper for Post Optimization Analysis. Depending on what the user wants and what services were being
        provided, analysis on the optimization solutions are completed here.

        TODO: [multi-tech] a lot of this logic will have to change with multiple technologies
        """

        print("Performing Post Optimization Analysis...") if self.verbose else None

        # add MONTHLY ENERGY BILL if customer sided
        if self.customer_sided:
            self.financials.calc_energy_bill(self.opt_results, self.services['retailTimeShift'].p_energy)

        # add other helpful information to a RESULTS DATAFRAME
        self.results = copy.deepcopy(self.opt_results)

        if 'PV' in self.active_objects['generator']:
            self.results['PV Generation (kW)'] = self.opt_results['PV_Gen (kW)']
            # self.results['PV Curtailed (kW)'] = self.opt_results['pv_out']
        self.results['Load (kW)'] = self.opt_results['load']

        self.results['Discharge (kW)'] = self.opt_results['dis']
        self.results['Charge (kW)'] = self.opt_results['ch']
        self.results['Battery Power (kW)'] = self.opt_results['dis'] - self.opt_results['ch']
        self.results['State of Energy (kWh)'] = self.opt_results['ene']
        # self.results['Billing Period'] = self.financials.fin_inputs['billing_period']
        # self.results['Energy Price ($)'] = self.financials.fin_inputs['p_energy']

        self.results['SOC (%)'] = self.opt_results['ene'] / self.technologies['Storage'].ene_max_rated

        # ac_power should include total power flow out of technology
        ac_power = copy.deepcopy(self.results['Battery Power (kW)'])

        for name, serv in self.services.items():
            temp_serv_p = serv.ene_results['ene'] / self.dt
            self.results[name + " Energy"] = serv.ene_results['ene']
            ac_power = ac_power + temp_serv_p
        self.results['ac_power'] = ac_power  # RENAME: this is a bad label for this column

        for name, tech in self.technologies.items():
            for constraint_name, constraint in tech.control_constraints.items():
                temp_constraint_values = constraint.value
                self.results[name + ' ' + constraint_name] = temp_constraint_values

        # these try to capture the import power to the site pre and post storage technology
        # will have to be made more dynamic with RIVET
        self.results['pre_import_power'] = self.results['load'] - (self.results['generation'])
        self.results['net_import_power'] = self.results['pre_import_power'] + self.results['ac_power']  # at the POC
        if 'Deferral' in self.active_objects['pre-dispatch']:
            self.results['pre_deferral_import_power'] = self.predispatch_services['Deferral'].load + self.results['load'] - self.results['generation']
            self.results['net_deferral_import_power'] = self.results['pre_deferral_import_power'] + self.results['ac_power']  # at the POC

        # calculate FINANCIAL SUMMARY
        self.financials.yearly_financials(self.technologies, self.services, self.opt_results)

        # # calculate RELIABILITY SUMMARY
        # if self.Reliability:
        #     outage_requirement = self.predispatch_services['Reliability'].reliability_requirement.sum()
        #     coverage_timestep = self.predispatch_services['Reliability'].coverage_timesteps
        #
        #     reliability = {}
        #     if self.pv:
        #         reverse = self.results['PV Curtailed (kW)'].iloc[::-1]  # reverse the time series to use rolling function
        #         reverse = reverse.rolling(coverage_timestep, min_periods=1).sum() * self.dt  # rolling function looks back, so reversing looks forward
        #         pv_outage = reverse.iloc[::-1]  # set it back the right way
        #         pv_contribution = pv_outage.sum()/outage_requirement
        #         reliability.update({'PV': pv_contribution})
        #     else:
        #         pv_contribution = 0
        #     battery_contribution = 1 - pv_contribution
        #     reliability.update({'Battery': battery_contribution})
        #     # TODO: go through each technology/DER (each contribution should sum to 1)
        #     self.reliability_df = pd.DataFrame(reliability, index=pd.Index(['Reliability contribution']))

        # create DISPATCH MAP
        if 'Battery' in self.active_objects['storage']:
            dispatch = self.results.loc[:, 'Battery Power (kW)'].to_frame()
            dispatch['date'] = self.opt_results.index.date
            dispatch['hour'] = (self.opt_results.index + pd.Timedelta('1s')).hour + 1
            dispatch = dispatch.reset_index(drop=True)

            # energy_price = self.results.loc[:, 'Energy Price ($)'].to_frame()
            # energy_price['date'] = self.opt_results['date']
            # energy_price['he'] = self.opt_results['he']
            # energy_price = energy_price.reset_index(drop=True)

            self.dispatch_map = dispatch.pivot_table(values='Battery Power (kW)', index='hour', columns='date')
            # self.energyp_map = energy_price.pivot_table(values='Energy Price ($)', index='he', columns='date')

        # DESIGN PLOT (peak load day)
        max_day = self.opt_results['load'].idxmax().date()
        max_day_data = self.opt_results[self.opt_results.index.date == max_day]
        day_index = pd.Index(np.arange(0, 24, self.dt), name='Timestep Beginning')
        self.peak_day_load = pd.DataFrame({'Load Power (kW)': max_day_data['load'].values}, index=day_index)
        # if self.sizing_results['Duration (hours)'].values[0] > 24:
        #     print('The duration of the Energy Storage System is greater than 24 hours!')
        dLogger.debug("Finished post optimization analysis")

    def save_results_csv(self, savepath=None):
        """ Save useful DataFrames to disk in csv files in the user specified path for analysis.

        """
        if savepath is None:
            savepath = self.results_path
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        self.results.to_csv(path_or_buf=Path(savepath, 'timeseries_results.csv'))
        if "DCM" in self.services.keys() or "retailTimeShift" in self.services.keys():
            self.financials.adv_monthly_bill.to_csv(path_or_buf=Path(savepath, 'adv_monthly_bill.csv'))
            self.financials.sim_monthly_bill.to_csv(path_or_buf=Path(savepath, 'simple_monthly_bill.csv'))
        # if self.Reliability:
        #     self.reliability_df.to_csv(path_or_buf=Path(savepath, 'reliability_summary.csv'))
        self.peak_day_load.to_csv(path_or_buf=Path(savepath, 'peak_day_load.csv'))
        # self.sizing_results.to_csv(path_or_buf=Path(savepath, 'size.csv'))
        if 'Battery' in self.active_objects['storage']:
            self.dispatch_map.to_csv(path_or_buf=Path(savepath, 'dispatch_map.csv'))
            # self.energyp_map.to_csv(path_or_buf=Path(savepath, 'energyp_map.csv'))

        self.financials.pro_forma.to_csv(path_or_buf=Path(savepath, 'pro_forma.csv'))
        self.financials.npv.to_csv(path_or_buf=Path(savepath, 'npv.csv'))
        self.financials.cost_benefit.to_csv(path_or_buf=Path(savepath, 'cost_benefit.csv'))

    @classmethod
    def add_instance(cls, dict):
        """

            Args:

            Returns:

        """
        cls.instances.update(dict)

    # # TODO Taken from svet_outputs. ploting functions haven't been used - EG + YY
    #
    # def plotly_stacked(self, p1_y1_bar=None, p1_y2=None, p2_y1=None, price_col=None, sep_ene_plot=True, filename=None,  year=None, start=None, end=None):
    #
    #     deferral = self.inputs['params']['Deferral']
    #
    #     if p1_y1_bar is None:
    #         p1_y1_bar = ['ch', 'dis']
    #     if p1_y2 is None:
    #         p1_y2 = ['ene']
    #     if p2_y1 is None:
    #         p2_y1 = []
    #         if deferral:
    #             p2_y1_line = ['net_deferral_import_power', 'pretech_deferral_import_power']
    #         else:
    #             p2_y1_line = ['net_import_power', 'pretech_import_power']
    #
    #         p2_y1_load = ['load']
    #         if deferral:
    #             p2_y1_load += ['deferral_load']
    #
    #         p2_y1_gen = []
    #         if all(self.results['ac_gen'] != 0):
    #             p2_y1_gen += ['ac_gen']
    #         if all(self.results['dc_gen'] != 0):
    #             p2_y1_gen += ['dc_gen']
    #         if deferral:
    #             p2_y1_gen = ['deferral_gen']
    #         p2_y1 += p2_y1_line + p2_y1_load + p2_y1_gen
    #
    #     # get price columns
    #     if price_col is None:
    #         price_col = []
    #
    #     # TODO do this smarter
    #     p1_y1_arrow = []
    #     price_col_kwh = []
    #     price_col_kw = []
    #     if self.inputs['params']['SR']:
    #         p1_y1_arrow += ['sr_d', 'sr_c']
    #         price_col_kw += ['p_sr']
    #     if self.inputs['params']['FR']:
    #         p1_y1_arrow += ['regu_d', 'regu_c', 'regd_d', 'regd_c']
    #         price_col_kw += ['p_regu', 'p_regd']
    #     if self.inputs['params']['DA']:
    #         price_col_kwh += ['p_da']
    #     if self.inputs['params']['retailTimeShift']:
    #         price_col_kwh += ['p_energy']
    #     price_col += price_col_kwh + price_col_kw
    #     price_plot_kw = len(price_col_kw) > 0  # flag for subplot logic
    #     price_plot_kwh = len(price_col_kwh) > 0  # flag for subplot logic
    #
    #
    #     # convert $/kW to $/MW
    #     price_data = self.financials.fin_inputs[price_col] * 1000
    #
    #     # merge price data with load data
    #     col_names = ['year'] + p1_y1_bar+p1_y1_arrow + p1_y2 + p2_y1
    #     plot_results = pd.merge(self.results[col_names], price_data, left_index=True, right_index=True, how='left')
    #
    #     # represent charging as negative
    #     neg_cols = ['ch']
    #
    #     # combine FR columns to reg up and reg down
    #     if self.inputs['params']['FR']:
    #         plot_results['reg_up'] = plot_results['regu_d'] + plot_results['regu_c']
    #         plot_results['reg_down'] = plot_results['regd_d'] + plot_results['regd_c']
    #         neg_cols += ['reg_down']
    #         for col in ['regu_d', 'regu_c', 'regd_d', 'regd_c']:
    #             p1_y1_arrow.remove(col)
    #         p1_y1_arrow += ['reg_up', 'reg_down']
    #     plot_results[neg_cols] = -plot_results[neg_cols]
    #
    #     # subset plot_results based on parameters
    #     if year is not None:
    #         plot_results = plot_results[plot_results.year == year]
    #     if start is not None:
    #         plot_results = plot_results[plot_results.index > start]
    #     if end is not None:
    #         plot_results = plot_results[plot_results.index <= end]
    #
    #     # round small numbers to zero for easier viewing
    #     plot_results = plot_results.round(decimals=6)
    #
    #     # add missing rows to avoid interpolation
    #     plot_results = lib.fill_gaps(plot_results)
    #
    #     # create figure
    #     fig = py.tools.make_subplots(rows=2 + sep_ene_plot + price_plot_kwh + price_plot_kw, cols=1, shared_xaxes=True, print_grid=False)
    #
    #     fig['layout'].update(barmode='relative')  # allows negative values to have negative bars
    #     # fig['layout'].update(barmode='overlay')
    #
    #     # add ch and discharge
    #     for col in p1_y1_bar:
    #         trace = py.graph_objs.Bar(x=plot_results.index, y=plot_results[col], name=col, offset=pd.Timedelta(self.dt/2, unit='h'))
    #         fig.append_trace(trace, 1, 1)
    #     battery_power = plot_results['dis'] + plot_results['ch']
    #
    #     # add capacity commitments such as reg up and reg down as error bars
    #     colors = py.colors.DEFAULT_PLOTLY_COLORS
    #     for i, col in enumerate(p1_y1_arrow):
    #         y0_txt = [str(y0) for y0 in plot_results[col]]
    #         trace = py.graph_objs.Bar(x=plot_results.index, y=battery_power*0, base=battery_power, name=col, offset=pd.Timedelta(self.dt/2, unit='h'), text=y0_txt,
    #                                   hoverinfo='x+text+name', marker=dict(color='rgba(0, 0, 0, 0)'), hoverlabel=dict(bgcolor=colors[i+len(p1_y1_bar)]),
    #                                   showlegend=False,  # TODO hiding until figure out how to include error bars in legend
    #                                   error_y=dict(visible=True, symmetric=False, array=plot_results[col], type='data', color=colors[i+len(p1_y1_bar)]))
    #         fig.append_trace(trace, 1, 1)
    #
    #     # other methods I tried instead of using error bars for capacity commitments
    #
    #     # for i, col in enumerate(p1_y1_arrow):
    #     #     y0_txt = [str(y0) for y0 in plot_results[col]]
    #     #     trace = ff.create_quiver(x=plot_results.index, y=battery_power.values, u=plot_results.index, v=plot_results[col].values)
    #     #     fig.append_trace(trace, 1, 1)
    #
    #     # colors = py.colors.DEFAULT_PLOTLY_COLORS
    #     # for i, col in enumerate(p1_y1_arrow):
    #     #     y0_txt = [str(y0) for y0 in plot_results[col]]
    #     #     trace = py.graph_objs.Bar(x=plot_results.index, y=plot_results[col], base=battery_power, name=col, offset=-55*60*case.dt*1e3/2, width=10000,
    #     #                               marker=dict(color='rgba(0, 0, 0, 0)', line=dict(width=2, color=colors[i+2])), text=y0_txt, hoverinfo='x+text+name')
    #     #     fig.append_trace(trace, 1, 1)
    #
    #     # for col in p1_y1_arrow:
    #     #     battery_power = plot_results['dis'] - plot_results['ch']
    #     #     trace = py.graph_objs.Scatter(x=plot_results.index-pd.Timedelta(-case.dt/2, unit='h'), y=battery_power+plot_results[col], name=col, mode='markers', line=dict(shape='vh'))
    #     #     fig.append_trace(trace, 1, 1)
    #
    #     if sep_ene_plot:
    #         # add separate energy plot
    #         for col in p1_y2:
    #             trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], line=dict(shape='linear'), name=col, mode='lines+markers')
    #             fig.append_trace(trace, 2, 1)
    #     else:
    #         # add energy on second y axis to charge and discharge
    #         for col in p1_y2:
    #             trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], line=dict(shape='linear'), name=col, yaxis='y5', mode='lines+markers')
    #             fig.append_trace(trace, 1, 1)
    #             names = []
    #             for i in fig.data:
    #                 names += [i['name']]
    #             fig.data[names.index('ene')].update(yaxis='y5')
    #
    #     # system power plot
    #     p2_y1_load_val = np.zeros(len(plot_results))
    #     for col in p2_y1_load:
    #         p2_y1_load_val += plot_results[col]
    #         y0_txt = [str(y0) for y0 in plot_results[col]]
    #         trace = py.graph_objs.Scatter(x=plot_results.index, y=copy.deepcopy(p2_y1_load_val), line=dict(shape='vh'), name=col, fill='tonexty',mode='none',
    #                                       text=y0_txt, hoverinfo='x+text+name')
    #         fig.append_trace(trace, 2 + sep_ene_plot, 1)
    #     p2_y1_gen_val = np.zeros(len(plot_results))
    #     for col in p2_y1_gen:
    #         p2_y1_gen_val -= plot_results[col]
    #         y0_txt = [str(y0) for y0 in plot_results[col]]
    #         trace = py.graph_objs.Scatter(x=plot_results.index, y=p2_y1_gen_val, line=dict(shape='vh'), name=col, fill='tonexty', mode='none',
    #                                       text=y0_txt, hoverinfo='x+text+name')
    #         fig.append_trace(trace, 2 + sep_ene_plot, 1)
    #     for col in p2_y1_line:
    #         trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], line=dict(shape='vh'), name=col, mode='lines+markers')
    #         fig.append_trace(trace, 2 + sep_ene_plot, 1)
    #
    #     # price plot
    #     if price_plot_kw:
    #         for col in price_col_kw:
    #             trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], line=dict(shape='vh'), name=col)
    #             fig.append_trace(trace, 3 + sep_ene_plot, 1)
    #     if price_plot_kwh:
    #         for col in price_col_kwh:
    #             trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], line=dict(shape='vh'), name=col, yaxis='y6')
    #             fig.append_trace(trace, 3 + price_plot_kw + sep_ene_plot, 1)
    #
    #     # axis labels
    #     fig['layout']['xaxis1'].update(title='Time')
    #     fig['layout']['yaxis1'].update(title='Power (kW)')
    #     if not sep_ene_plot:
    #         fig['layout']['yaxis2'].update(title='Power (kW)')
    #         if price_plot_kw:
    #             fig['layout']['yaxis3'].update(title='Price ($/MW)')
    #         if price_plot_kwh:
    #             fig['layout']['yaxis4'].update(title='Price ($/MWh)')
    #         fig['layout']['yaxis5'] = dict(overlaying='y1', anchor='x1', side='right', title='Energy (kWh)')
    #     else:
    #         fig['layout']['yaxis2'].update(title='Energy (kWh)')
    #         fig['layout']['yaxis3'].update(title='Power (kW)')
    #         if price_plot_kw:
    #             fig['layout']['yaxis4'].update(title='Price ($/MW)')
    #         if price_plot_kwh:
    #             fig['layout']['yaxis5'].update(title='Price ($/MWh)')
    #
    #     # move legend to middle (plotly does not support separate legends for subplots as of now)
    #     fig['layout']['legend'] = dict(y=0.5, traceorder='normal')
    #
    #     if filename is None:
    #         filename = self.name + '_plot.html'
    #
    #     py.offline.plot(fig, filename=filename)
    #
    # def plotly_groupby(self, group='he'):
    #     """ Plot Results averaged to a certain group column
    #
    #     Args:
    #         case (Case): case object
    #         group (string, list): columns in case.results to group on
    #
    #     """
    #     yrs = self.opt_years
    #
    #     fig = py.tools.make_subplots(rows=len(yrs), cols=1, shared_xaxes=True, print_grid=False, subplot_titles=[yr.year for yr in yrs])
    #
    #     colors = py.colors.DEFAULT_PLOTLY_COLORS
    #
    #     for i, yr in enumerate(yrs):
    #         plot_results = self.power_kw[self.power_kw.year == yr].groupby(group).mean()
    #         neg_cols = ['ch']
    #         if self.inputs['params']['FR']:
    #             plot_results['reg_up'] = plot_results['regu_d'] + plot_results['regu_c']
    #             plot_results['reg_down'] = plot_results['regd_d'] + plot_results['regd_c']
    #             neg_cols += ['reg_down']
    #
    #         plot_results[neg_cols] = -plot_results[neg_cols]
    #
    #         plot_cols = ['ch', 'dis', 'reg_up', 'reg_down', 'load']
    #         for ii, col in enumerate(plot_cols):
    #             trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], name=col, mode='lines+markers',
    #                                           line=dict(color=colors[ii]))
    #             fig.append_trace(trace, i+1, 1)
    #     py.offline.plot(fig)
    #
    # def save_case(self, savepath=None, savename=None):
    #     """ Helper function: Saves case as a pickle object based on result path and name
    #
    #     Args:
    #         case (Case): case object
    #         savepath (str): path to save case to
    #         savename (str): name to call save file (must end in .pickle)
    #
    #     """
    #
    #     if savepath is None:
    #         savepath = self.results_path  # use default results path if not provided
    #
    #     # create directory if does not exist
    #     if not os.path.exists(savepath):
    #         os.makedirs(savepath)
    #
    #     # name of save file
    #     if savename is None:
    #         savename = self.name + "_" + self.start_time_frmt + '.pickle'
    #
    #     # save pickle file
    #     with open(Path(savepath, savename), 'wb') as f:
    #         pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    #
    # # older plotting function
    # def plotly_case(self, y1_col=None, y2_col=None, binary=False, filename=None, start=None, end=None):
    #     """ Generic plotly function for case object (depreciated by plotly_stacked)
    #
    #     Args:
    #         case (Case): case object
    #         y1_col (list, optional): column names in case.result to plot on y1 axis
    #         y2_col (list, optional): column names in case.result to plot on y2 axis
    #         binary (bool, optional): Flag to add binary variables
    #         filename (str, optional): where to save file (must end in .html)
    #         start (date-like, optional): start timestamp to subset data (exclusive)
    #         end (date-like, optional): end timestamp to subset data (inclusive)
    #
    #     """
    #     if y1_col is None:
    #         y1_col = ['ch', 'dis', 'net_import_power', 'pretech_import_power']
    #     if y2_col is None:
    #         y2_col = ['ene']
    #
    #     limit_cols = []
    #
    #     plot_results = copy.deepcopy(self.results)
    #     predispatch = list(self.predispatch_services)
    #
    #     if 'Deferral' in predispatch:
    #         plot_results['deferral_max_import'] = self.predispatch_services['Deferral'].deferral_max_import
    #         plot_results['deferral_max_export'] = self.predispatch_services['Deferral'].deferral_max_export
    #         y1_col += ['deferral_max_import', 'deferral_max_export']
    #         limit_cols += ['deferral_max_import', 'deferral_max_export']
    #     if 'Volt' in predispatch:
    #         plot_results['inv_max'] = self.inputs['params']['inv_max']
    #         y1_col += ['vars_load', 'inv_max']
    #         limit_cols += ['inv_max']
    #     if 'Backup' in predispatch:
    #         y2_col += ['backup_energy']
    #
    #     # TODO do this smarter
    #     if self.inputs['params']['SR']:
    #         y1_col += ['sr_d', 'sr_c']
    #     if self.inputs['params']['FR']:
    #         y1_col += ['regu_d', 'regu_c', 'regd_d', 'regd_c']
    #
    #     if binary:  # or (binary is None and case.inputs['params']['binary']):
    #         y1_col += ['on_c', 'on_d']
    #
    #     if start is not None:
    #         plot_results = plot_results[plot_results.index > start]
    #     if end is not None:
    #         plot_results = plot_results[plot_results.index <= end]
    #
    #     # round small numbers to zero for easier viewing
    #     plot_results = plot_results.round(decimals=6)
    #
    #     # add missing rows to avoid interpolation
    #     plot_results = lib.fill_gaps(plot_results)
    #
    #     fig = plot_results[y1_col+y2_col].iplot(kind='line', mode='lines', interpolation='vh', asFigure=True, yTitle='Power (kW)', secondary_y=y2_col)
    #
    #     names = []
    #     for i in fig.data:
    #         names += [i['name']]
    #
    #     if 'ene' in y2_col:
    #         fig.layout.yaxis2.title = 'Energy (kWh)'
    #
    #         fig.data[names.index('ene')].line.shape = 'linear'
    #
    #     for n in limit_cols:
    #         fig.data[names.index(n)].line.dash = 'dot'
    #         fig.data[names.index(n)].line.width = 1
    #
    #     if filename is None:
    #         filename = self.name + '_plot.html'
    #
    #     py.offline.plot(fig, filename=filename)
    #
    # def plotly_prices(self, y1_col=None, price_col=None, filename=None, start=None, end=None):
    #     """ Generic plotly price function for case object (depreciated by plotly_stacked)
    #
    #     Args:
    #         case (Case): case object
    #         y1_col (list, optional): column names in case.result to plot on y1 axis
    #         price_col (list, optional): price column names in case.financials.fin_inputs to plot on y2 axis
    #         filename (str, optional): where to save file (must end in .html)
    #         start (date-like, optional): start timestamp to subset data (exclusive)
    #         end (date-like, optional): end timestamp to subset data (inclusive)
    #
    #     """
    #     if y1_col is None:
    #         y1_col = ['ch', 'dis', 'pretech_import_power', 'net_import_power']
    #     if price_col is None:
    #         price_col = []
    #
    #     # TODO do this smarter
    #     if self.inputs['params']['SR']:
    #         y1_col += ['sr_d', 'sr_c']
    #         price_col += ['p_sr']
    #     if self.inputs['params']['FR']:
    #         y1_col += ['regu_d', 'regu_c', 'regd_d', 'regd_c']
    #         price_col += ['p_regu', 'p_regd']
    #     if self.inputs['params']['DA']:
    #         price_col += ['p_da']
    #     if self.inputs['params']['retailTimeShift']:
    #         price_col += ['p_energy']
    #
    #     price_data = self.financials.fin_inputs[price_col]*1000
    #
    #     plot_results = pd.merge(self.results[y1_col], price_data, left_index=True, right_index=True, how='left')
    #
    #     if start is not None:
    #         plot_results = plot_results[plot_results.index > start]
    #     if end is not None:
    #         plot_results = plot_results[plot_results.index <= end]
    #
    #     # round small numbers to zero for easier viewing
    #     plot_results = plot_results.round(decimals=6)
    #
    #     # add missing rows to avoid interpolation
    #     plot_results = lib.fill_gaps(plot_results)
    #
    #     fig = plot_results[y1_col+price_col].iplot(kind='line', mode='lines', interpolation='vh', asFigure=True, yTitle='Power (kW)', secondary_y=list(plot_results[price_col]))
    #     fig.layout.yaxis2.title = 'Price ($/MWh)'
    #
    #     if filename is None:
    #         filename = self.name + '_plot.html'
    #
    #     py.offline.plot(fig, filename=filename)
    #
    # def plot_results(self, start_data=0, end_data=None, save=False):
    #     """ Plot the energy and demand charges before and after the storage acts and validates results by checking if
    #     the change in SOC is accounted for by AC power. Additionally saves plots in the results folder created when
    #     initialised.
    #
    #     """
    #     if not end_data:
    #         end_data = self.inputs['time_series'].index.size
    #     charging_opt_var = ['reg']
    #     results = self.financials.obj_val
    #     os.makedirs(self.results_path)
    #     plt.rcParams['figure.dpi'] = 300
    #     plt.rcParams['figure.figsize'] = [12, 6.75]
    #     plt.figure()
    #
    #     load = self.power_kw['site_load']
    #     bulk_power = self.power_kw['dis'] - self.power_kw['ch']
    #     pv = self.power_kw['PV_gen']
    #     net_power = load - bulk_power - pv  # at the POC
    #     soc = self.power_kw['ene'] / self.technologies['Storage'].ene_max_rated
    #     soc_diff = soc.diff()
    #     ac_power = copy.deepcopy(bulk_power)
    #     for serv in self.services.values():
    #         temp_serv_p = serv.ene_results['ene'] / self.dt
    #         ac_power = ac_power + temp_serv_p
    #
    #     plt.plot(load.index[start_data:end_data], load[start_data:end_data])
    #     plt.plot(net_power.index[start_data:end_data], net_power[start_data:end_data])
    #     plt.plot(pv.index[start_data:end_data], pv[start_data:end_data])
    #     plt.legend(['Site Load', 'Net Power', 'PV'])
    #     plt.ylabel('Power (kW)')
    #     if save:
    #         plt.savefig(self.results_path + 'net_power.png')
    #     plt.close()
    #
    #     plt.plot(bulk_power.index[start_data:end_data], bulk_power[start_data:end_data])
    #     plt.title('Storage Power')
    #     plt.ylabel('Power (kW)')
    #     if save:
    #         plt.savefig(self.results_path + 'sto_power.png')
    #     plt.close()
    #
    #     plt.plot(soc.index[start_data:end_data], soc[start_data:end_data])
    #     plt.title('Storage State of Charge')
    #     plt.ylabel('Power (kW)')
    #     if save:
    #         plt.savefig(self.results_path + 'state_of_charge.png')
    #     plt.close()
    #
    #     # plot the energy and demand charges before and after the storage acts
    #     # width = .2
    #     # plt.figure()
    #     # plt.bar(list(list(zip(*case.monthly_bill.index.values))[0]) + 2 * width, case.monthly_bill.loc[:, 'energy_charge'], width)
    #     # plt.bar(list(list(zip(*case.monthly_bill.index.values))[0]) + width, case.monthly_bill.loc[:, 'original_energy_charge'], width)
    #     # plt.bar(list(list(zip(*case.monthly_bill.index.values))[0]) - 2 * width, case.monthly_bill.loc[:, 'demand_charge'], width)
    #     # plt.bar(list(list(zip(*case.monthly_bill.index.values))[0]) - width, case.monthly_bill.loc[:, 'original_demand_charge'], width)
    #     # plt.title('Monthly energy and demand charges')
    #     # plt.legend(['Energy Charges', 'Original Energy Charges', 'Demand Charges', 'Original Demand Charges'])
    #     # plt.xlabel('Month')
    #     # plt.ylabel('$')
    #     # plt.draw()
    #     # plt.savefig(case.results_path + 'charges.png')
    #
    #     # Validate Results by checking if the change in SOC is accounted for by AC power
    #     plt.scatter(soc_diff, ac_power)
    #     plt.xlabel('delta SOC')
    #     plt.ylabel('AC Storage Power')
    #     plt.draw()
    #     if save:
    #         plt.savefig(self.results_path + 'SOCvkW.png')
