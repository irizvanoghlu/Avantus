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
from storagevet.Result import Result

dLogger = logging.getLogger('Developer')
uLogger = logging.getLogger('User')


class ResultDER(Result):
    """

    """

    def __init__(self, scenario, results_inputs):
        """ Initialize all Result objects, given a Scenario object with the following attributes.

            Args:
                scenario (Scenario.Scenario): scenario object after optimization has run to completion
                results_inputs (Dict): user-defined inputs from the model parameter inputs
        """
        Result.__init__(self, scenario, results_inputs)
        self.reliability_df = pd.DataFrame()
        self.sizing_df = pd.DataFrame(index=pd.Index(['Power Capacity (kW)',
                                                      'Capital Cost ($)'], name='Size and Costs'))

    def post_analysis(self):
        """ Wrapper for Post Optimization Analysis. Depending on what the user wants and what services were being
        provided, analysis on the optimization solutions are completed here.

        TODO: [multi-tech] a lot of this logic will have to change with multiple technologies
        """
        Result.post_analysis(self)
        for name, tech in self.technologies.items():
            sizing_df = tech.sizing_summary()
            self.sizing_df = pd.merge(self.sizing_df, sizing_df, how='outer', on='Size and Costs')
        if 'Reliability' in self.predispatch_services.keys():  # TODO: possibly make an method of Reliability --HN
            # TODO: make this more dynamic
            reliability_requirement = self.predispatch_services['Reliability'].reliability_requirement
            self.results.loc[:, 'SOC Constraints (%)'] = reliability_requirement / self.technologies['Storage'].ene_max_rated.value
            # calculate RELIABILITY SUMMARY
            outage_energy = self.predispatch_services['Reliability'].reliability_requirement
            self.results.loc[:, 'Total Outage Requirement (kWh)'] = outage_energy
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

            # TODO: go through each technology/DER (each contribution should sum to 1)
            self.reliability_df = pd.DataFrame(reliability, index=pd.Index(['Reliability contribution'])).T

    def save_results_csv(self):
        """ Save useful DataFrames to disk in csv files in the user specified path for analysis.

        """
        Result.save_results_csv(self)
        savepath = self.results_path
        if 'Reliability' in self.predispatch_services.keys():
            self.reliability_df.to_csv(path_or_buf=Path(savepath, 'reliability_summary' + self.csv_label + '.csv'))
        self.sizing_df.to_csv(path_or_buf=Path(savepath, 'size' + self.csv_label + '.csv'))
        print('DER results have been saved to: ' + self.results_path)
