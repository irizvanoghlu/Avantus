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
from CBA import CostBenefitAnalysis


u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class MicrogridResult(Result):
    """

    """

    def __init__(self, scenario):
        """ Initialize all Result objects, given a Scenario object with the following attributes.

            Args:
                scenario (Scenario.Scenario): scenario object after optimization has run to completion
        """
        super().__init__(scenario)
        self.sizing_df = None

    def collect_results(self):
        """ Collects any optimization variable solutions or user inputs that will be used for drill down
        plots, as well as reported to the user. No matter what value stream or DER is being evaluated, these
        dataFrames should always be made and reported to the user

        Three attributes are edited in this method: TIME_SERIES_DATA, MONTHLY_DATA, TECHNOLOGY_SUMMARY
        """
        super().collect_results()
        for name, tech in self.technologies.items():
            # sizing_summary for CAES is currently similar to it for Battery
            sizing_df = tech.sizing_summary()
            self.sizing_df = pd.concat([self.sizing_df, sizing_df], axis=0, sort=False)

    def save_as_csv(self, instance_key, sensitivity=False):
        """ Save useful DataFrames to disk in csv files in the user specified path for analysis.

        Args:
            instance_key (int): string of the instance value that corresponds to the Params instance that was used for
                this simulation.
            sensitivity (boolean): logic if sensitivity analysis is active. If yes, save_path should create additional
                subdirectory

        Prints where the results have been saved when completed.
        """
        super().save_as_csv(instance_key, sensitivity)
        if sensitivity:
            savepath = self.dir_abs_path + "\\" + str(instance_key)
        else:
            savepath = self.dir_abs_path
        # self.peak_day_load.to_csv(path_or_buf=Path(savepath, f'peak_day_load{self.csv_label}.csv'))

        # if 'Reliability' in self.controller.value_streams.keys():
        #     self.reliability_df.to_csv(path_or_buf=Path(savepath, 'reliability_summary' + self.csv_label + '.csv'))
        #     self.load_coverage_prob.to_csv(path_or_buf=Path(savepath, 'load_coverage_probability' + self.csv_label + '.csv'), index=False)
        self.sizing_df.to_csv(path_or_buf=Path(savepath, 'size' + self.csv_label + '.csv'))
        print('DER results have been saved to: ' + self.dir_abs_path)
        u_logger.info('DER results have been saved to: ' + self.dir_abs_path)

