# -*- coding: utf-8 -*-
"""
Created Aug 11, 2020

author: Mark Panas
"""

def OpenAirBeam2(filename):
    
    import numpy as np
    import pandas as pd

    with open(filename) as fp:
        out = fp.readlines()

    #print(out[0].rstrip().split(','))

    if out[0].rstrip().split(',')[0] != "":
        
        #print("Data format = 1")

        bad_rows = []
        element_names = []


        for i in range(len(out)):

            try:

                float(out[i].rstrip().split(',')[3])

            except(ValueError):

                #print("Line %i:" % (i),out[i].rstrip().split(','))

                if out[i].rstrip().split(',')[0] == "sensor:model":
                    bad_rows.append(i)

                if out[i].rstrip().split(',')[0].split('-')[0] == 'AirBeam2':
                    element_names.append(out[i].rstrip().split(',')[0].split('-')[1])


        #print(element_names)
        d_pm = {}

        col_names = out[2].rstrip().split(',')

        for i in range(len(bad_rows)):

            if i == 0:

                skip_rows_start = np.asarray([bad_rows[i],bad_rows[i]+1, bad_rows[i]+2])
                skip_rows_rest = np.arange(bad_rows[i+1],len(out))

                skip_rows_all = np.concatenate((skip_rows_start, skip_rows_rest))

                d_pm[element_names[i]] = pd.read_csv(filename, header=None, names=col_names, skiprows=skip_rows_all)


            elif i != len(bad_rows)-1:

                skip_rows_start = np.arange(0,bad_rows[i]+1)
                skip_rows_mid = np.asarray([bad_rows[i],bad_rows[i]+1, bad_rows[i]+2])
                skip_rows_rest = np.arange(bad_rows[i+1],len(out))

                skip_rows_all = np.concatenate((skip_rows_start, skip_rows_mid, skip_rows_rest))
                d_pm[element_names[i]] = pd.read_csv(filename, header=None, names=col_names, skiprows=skip_rows_all)

            else:
                d_pm[element_names[i]] = pd.read_csv(filename, header=None, names=col_names, skiprows=np.arange(0,bad_rows[i]+3))
              
    
        data_format = 1
        col_names = element_names
    
    
    else:
        col_names = ['F', 'PM1', 'PM10', 'PM2.5', 'RH']
        all_col_names = ['Timestamp', 'Latitude', 'Longitude', 'F', 'PM1', 'PM10', 'PM2.5', 'RH']
        d_pm = pd.read_csv(filename, names=all_col_names, skiprows=9, usecols=range(2,10))
                
        data_format = 2

    
    # Arrays of different values may be different lengths
    # Find the smallest length
    
    column_lengths = []

    for i in range(len(col_names)):
        
        if data_format == 1: column_lengths.append(d_pm[col_names[i]]["Value"].shape)
        if data_format == 2: column_lengths.append(d_pm[col_names[i]].dropna().shape)

    min_length = min(column_lengths)[0]

    
    # Consolidate the lat long data into one average array
    
    lats = np.empty((min_length,5))
    longs = np.empty((min_length,5))

    for i in range(len(col_names)):
        
        if data_format == 1:
            lats[:,i] = d_pm[col_names[i]]['geo:lat'][0:min_length]
            longs[:,i] = d_pm[col_names[i]]['geo:long'][0:min_length]
            
        if data_format == 2:
            lats[:,i] = d_pm['Latitude'][d_pm[col_names[i]].dropna()[0:min_length].index]
            longs[:,i] = d_pm['Longitude'][d_pm[col_names[i]].dropna()[0:min_length].index]

    lats = np.mean(lats, axis=1)
    longs = np.mean(longs, axis=1)
    
    # Generate arrays for absolute time and relative time
    if data_format == 1:
        
        d_pm['datetime'] = pd.DataFrame()
        
        for i in range(len(col_names)):
            
            d_pm['datetime'][col_names[i]] = pd.to_datetime(d_pm[col_names[i]]['Timestamp'],format="%Y-%m-%dT%H:%M:%S.%f-0400")
            
            if i == 0:
                min_time = np.min(d_pm['datetime'][col_names[i]])
                max_time = np.min(d_pm['datetime'][col_names[i]])
            else:
                if d_pm['datetime'][col_names[i]].min() < min_time:
                    min_time = np.min(d_pm['datetime'][col_names[i]])
                if d_pm['datetime'][col_names[i]].max() > max_time:
                    max_time = np.max(d_pm['datetime'][col_names[i]])

        
    if data_format == 2:
        
        d_pm['datetime'] = pd.to_datetime(d_pm['Timestamp'],format="%Y-%m-%dT%H:%M:%S.%f")
        
        min_time = np.min(d_pm['datetime'])
        max_time = np.max(d_pm['datetime'])
    
    
    datetimes = np.asarray(pd.date_range(min_time, max_time, min_length).to_series(), dtype=np.datetime64)

    t_end = float((max_time - min_time) // pd.Timedelta('1ms'))/1000
    rel_time = np.linspace(0,t_end, min_length)
    

    # Copy the measurement values into numpy arrays
    if data_format == 1:
        temp = np.asarray(d_pm["F"]["Value"][:min_length])
        pm1 = np.asarray(d_pm["PM1"]["Value"][:min_length])
        pm10 = np.asarray(d_pm["PM10"]["Value"][:min_length])
        pm2 = np.asarray(d_pm["PM2.5"]["Value"][:min_length])
        rh = np.asarray(d_pm["RH"]["Value"][:min_length])

        
    if data_format == 2:
        temp = np.asarray(d_pm["F"].dropna()[:min_length])
        pm1 = np.asarray(d_pm["PM1"].dropna()[:min_length])
        pm10 = np.asarray(d_pm["PM10"].dropna()[:min_length])
        pm2 = np.asarray(d_pm["PM2.5"].dropna()[:min_length])
        rh = np.asarray(d_pm["RH"].dropna()[:min_length])

    
    return datetimes, rel_time, temp, pm1, pm10, pm2, rh, lats, longs


def OpenAeroqual(filename):
    
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(filename, header=0, skipinitialspace=True)

    df['datetime'] = pd.to_datetime(df['Date Time'],format="%d %b %Y %H:%M")
    
    td = (df['datetime'] - df['datetime'][0])// pd.Timedelta('1ms')/1000

    abs_time = np.asarray(df['datetime'], dtype=np.datetime64)
    rel_time = np.asarray(td)
    
    if any(df.columns == 'CO2(ppm)'):
        vmr = np.asarray(df['CO2(ppm)'])
    else:
        vmr = np.asarray(df['O3(ppm)'])
    
    return abs_time, rel_time, vmr


def PointLabels(x, y, n, plot_index=False):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    xy_locs = list(zip(x[::n], y[::n]))
    
    if plot_index == True:
        
        x = np.arange(0, x.shape[0])
        
        xy_labels = list(zip(x[::n], y[::n]))
    
    else:
        
        xy_labels = xy_locs
    
    
    for i in range(len(xy_locs)):
        plt.annotate('(%s, %s)' % xy_labels[i], xy=xy_locs[i], textcoords='data')
        
        
def factorization(n):

    from math import gcd
    
    factors = []

    def get_factor(n):
        x_fixed = 2
        cycle_size = 2
        x = 2
        factor = 1

        while factor == 1:
            for count in range(cycle_size):
                if factor > 1: break
                x = (x * x + 1) % n
                factor = gcd(x - x_fixed, n)

            cycle_size *= 2
            x_fixed = x

        return factor

    while n > 1:
        next = get_factor(n)
        factors.append(next)
        n //= next

    return factors


def SaveAirbeam2(filename, pm_datetimes, pm_rel_time, pm1, pm2, pm10, pm_temp, pm_rh):
    
    import pandas as pd
    
    d = {"datetimes":pm_datetimes,"rel_time":pm_rel_time, "pm1":pm1, "pm2.5":pm2, "pm10":pm10, "pm_temp":pm_temp, "pm_rh":pm_rh}
    
    pd.DataFrame(d).to_csv(filename)


def SaveAeroqual(filename, datetimes, rel_time, vmr):
    
    import pandas as pd
    
    d = {"datetimes":datetimes,"rel_time":rel_time, "vmr":vmr}
    
    pd.DataFrame(d).to_csv(filename)