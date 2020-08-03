# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:46:23 2020

@author: paco011
"""

import pandas as pd

path = r'C:\Users\paco011\Documents\Results\_18_07_38_06_02_2020'

path_input = r'C:\Users\paco011\Documents\dervet\storagevet\Data'

file_results = r'timeseries_results.csv'
file_input = r'hourly_timeseries.csv'

input_data = pd.read_csv(path_input + '/' + file_input)

results = pd.read_csv(path + '/' + file_results)

(input_data['Deferral Load (kW)'] - (results['BATTERY: 2MW-5hr Discharge (kW)']-results['BATTERY: 2MW-5hr Charge (kW)'])-14000).plot()

(input_data['Deferral Load (kW)']-11000).plot()

results['Aggregated State of Energy (kWh)'].plot()