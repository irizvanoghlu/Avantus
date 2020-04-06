"""
Result.py

"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',"Thien Nguyen"]
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = 'beta'  # beta version


import pandas as pd
import logging
import copy
import numpy as np
from pathlib import Path
import os
import storagevet
from storagevet.Result import Result
from cbaDER import CostBenefitAnalysis


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
        super().__init__(scenario, CostBenefitAnalysis)
        self.reliability_df = pd.DataFrame()
        self.sizing_df = pd.DataFrame()
        self.load_coverage_prob = pd.DataFrame()

    def post_analysis(self):
        """ Wrapper for Post Optimization Analysis. Depending on what the user wants and what services were being
        provided, analysis on the optimization solutions are completed here.

        """
        super().post_analysis()
        for name, tech in self.technologies.items():  # TODO
            # sizing_summary for CAES is currently similar to it for Battery
            sizing_df = tech.sizing_summary()
            self.sizing_df = pd.concat([self.sizing_df, sizing_df], axis=0, sort=False)

        # DESIGN PLOT (peak load day)
        max_day = self.results['Total Load (kW)'].idxmax().date()
        max_day_data = self.results[self.results.index.date == max_day]
        time_step = pd.Index(np.arange(0, 24, self.dt), name='Timestep Beginning')
        self.peak_day_load = pd.DataFrame({'Date': max_day_data.index.date,
                                           'Load (kW)': max_day_data['Total Load (kW)'].values,
                                           'Net Load (kW)': max_day_data['Net Load (kW)'].values}, index=time_step)

        if 'Reliability' in self.predispatch_services.keys():
            reliability = self.predispatch_services['Reliability']
            # save/calculate load coverage
            u_logger.info('Starting load coverage calculation. This may take a while.')
            self.load_coverage_prob = reliability.load_coverage_probability(self.results, self.sizing_df, self.technology_summary)
            u_logger.info('Finished load coverage calculation.')
            # calculate RELIABILITY SUMMARY
            der_contributions, self.reliability_df = reliability.contribution_summary(self.technologies.keys(), self.results)
            self.results = pd.concat([self.results, der_contributions], axis=1)

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
            self.load_coverage_prob.to_csv(path_or_buf=Path(savepath, 'load_coverage_probability' + self.csv_label + '.csv'), index=False)
        self.sizing_df.to_csv(path_or_buf=Path(savepath, 'size' + self.csv_label + '.csv'))
        print('DER results have been saved to: ' + self.dir_abs_path)
        u_logger.info('DER results have been saved to: ' + self.dir_abs_path)

