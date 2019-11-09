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
import storagevet
from storagevet.Result import Result

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class ResultDER(Result):
    """

    """

    def __init__(self, scenario):
        """ Initialize all Result objects, given a Scenario object with the following attributes.

            Args:
                scenario (Scenario.Scenario): scenario object after optimization has run to completion
        """
        Result.__init__(self, scenario)
        self.reliability_df = pd.DataFrame()
        self.sizing_df = pd.DataFrame()

    def post_analysis(self):
        """ Wrapper for Post Optimization Analysis. Depending on what the user wants and what services were being
        provided, analysis on the optimization solutions are completed here.

        TODO: [multi-tech] a lot of this logic will have to change with multiple technologies
        """
        Result.post_analysis(self)
        for name, tech in self.technologies.items():
            # sizing_summary for CAES is currently similar to it for Battery
            sizing_df = tech.sizing_summary()
            self.sizing_df = pd.concat([self.sizing_df, sizing_df], axis=0, sort=False)
        if (self.sizing_df['Duration (hours)'] > 24).any():
            print('The duration of an Energy Storage System is greater than 24 hours!')

        # DESIGN PLOT (peak load day)
        max_day = self.results['Original Net Load (kW)'].idxmax().date()
        max_day_data = self.results[self.results.index.date == max_day]
        time_step = pd.Index(np.arange(0, 24, self.dt), name='Timestep Beginning')
        self.peak_day_load = pd.DataFrame({'Date': max_day_data.index.date,
                                           'Load (kW)': max_day_data['Original Net Load (kW)'].values,
                                           'Net Load (kW)': max_day_data['Net Load (kW)'].values}, index=time_step)

        if 'Reliability' in self.predispatch_services.keys():  # TODO: possibly make an method of Reliability --HN
            # TODO: make this more dynamic
            # calculate RELIABILITY SUMMARY
            outage_energy = self.predispatch_services['Reliability'].reliability_requirement
            sum_outage_requirement = outage_energy.sum()  # sum of energy required to provide x hours of energy if outage occurred at every timestep
            coverage_timestep = self.predispatch_services['Reliability'].coverage_timesteps  # guaranteed power for this many hours in outage

            reliability = {}
            if 'PV' in self.technologies.keys():
                # reverse the time series to use rolling function
                # rolling function looks back, so reversing looks forward
                reverse_pv_out = self.results['PV Generation (kW)'].iloc[::-1]
                reverse_pv_rolsum = reverse_pv_out.rolling(coverage_timestep, min_periods=1).sum() * self.dt
                # rolling sum of energy within a coverage_timestep window
                pv_rolsum = reverse_pv_rolsum.iloc[::-1].values
                # remove any over generation within each x hour long outage
                over_generation = pv_rolsum - outage_energy
                pv_outage_energy = pv_rolsum - over_generation
                pv_contribution = np.sum(pv_outage_energy)/sum_outage_requirement
                reliability.update({'PV': pv_contribution})

                # the energy a battery will provide in an outage is whatever that is not being provided by pv
                remaining_outage_ene = outage_energy.values - pv_rolsum
                remaining_outage_ene = remaining_outage_ene.clip(min=0)
            else:
                remaining_outage_ene = outage_energy.values
                pv_outage_energy = np.zeros(len(self.results.index))
                pv_contribution = 0

            if 'Battery' in self.technologies.keys():
                battery_energy = self.opt_results['ene'].values
                extra_energy = (battery_energy - remaining_outage_ene).clip(min=0)
                battery_outage_ene = battery_energy - extra_energy
                remaining_outage_ene -= battery_outage_ene
                battery_contribution = np.sum(battery_outage_ene) / sum_outage_requirement
                reliability.update({'Battery': battery_contribution})
            else:
                battery_outage_ene = np.zeros(len(self.results.index))
                battery_contribution = 0

            if 'CAES' in self.technologies.keys():
                print('What is CAES output behavior when there is Reliability?')
                # pending status - TN

            if 'Diesel' in self.technologies.keys():
                # supplies what every energy that cannot be by pv and diesel
                reverse_diesel_gen = self.results['Diesel Generation (kW)'].iloc[::-1]
                reverse_diesel_rolsum = reverse_diesel_gen.rolling(coverage_timestep, min_periods=1).sum() * self.dt
                diesel_rolsum = reverse_diesel_rolsum.iloc[::-1].values  # set it back the right way

                extra_energy = (diesel_rolsum - remaining_outage_ene).clip(min=0)
                diesel_outage_ene = diesel_rolsum - extra_energy
                diesel_contribution = np.sum(diesel_outage_ene) / sum_outage_requirement
                # diesel_contribution = 1 - pv_contribution - battery_contribution
                reliability.update({'Diesel': diesel_contribution})

                # we additionally subtract energy provided by the generator from the energy the battery will have to provide
                battery_outage_ene = battery_outage_ene - diesel_contribution
            else:
                diesel_outage_ene = np.zeros(len(self.results.index))

            self.results.loc[:, 'PV Outage Contribution (kWh)'] = pv_outage_energy
            self.results.loc[:, 'Battery Outage Contribution (kWh)'] = battery_outage_ene
            self.results.loc[:, 'Generator Outage Contribution (kWh)'] = diesel_outage_ene
            # does CAES have outage contribution? This depends on if CAES contributes during Reliability
            # self.results.loc[:, 'CAES Outage Contribution (kWh)'] = caes_outage_ene

            # TODO: go through each technology/DER (each contribution should sum to 1)
            self.reliability_df = pd.DataFrame(reliability, index=pd.Index(['Reliability contribution'])).T

    def save_as_csv(self, instance_key, sensitivity=False):
        """ Save useful DataFrames to disk in csv files in the user specified path for analysis.

        Args:
            instance_key (int): string of the instance value that corresponds to the Params instance that was used for
                this simulation.
            sensitivity (boolean): logic if sensitivity analysis is active. If yes, save_path should create additional
                subdirectory

        Prints where the results have been saved when completed.
        """
        Result.save_as_csv(self, instance_key, sensitivity)
        if sensitivity:
            savepath = self.dir_abs_path + "\\" + str(instance_key)
        else:
            savepath = self.dir_abs_path
        if 'Reliability' in self.predispatch_services.keys():
            self.reliability_df.to_csv(path_or_buf=Path(savepath, 'reliability_summary' + self.csv_label + '.csv'))
        self.sizing_df.to_csv(path_or_buf=Path(savepath, 'size' + self.csv_label + '.csv'))
        print('DER results have been saved to: ' + self.dir_abs_path)


