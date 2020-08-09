#this program removes any row from a csv that has at least 1 #na & creates a new, clean csv in folder below
#suggest better to include this logic in main AI program code to minimise number of maintained csvs

import pandas as pd

# per Luke's code, including feature names - unsure if required
feature_names = [ 'Date', 'XNDX', 'ASA51', 'SPXT', 'CACR', 'DAX', 'NKYTR', 'HSI1', 'KSP2TR']

voting_data = pd.read_csv( 'raw.csv', names = feature_names ) # read file
voting_data.head() 
voting_data.dropna( inplace = True ) # this removes incomplete rows... interesting
voting_data.describe()

export_csv = voting_data.to_csv (r'C:\Users\brian\A3D\Index_strat\rawnona.csv', index=None, header=True) #export to csv in file location, left

print(voting_data)
