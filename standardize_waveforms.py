# Versión para probar el flujo de trbajo en GAIA
from obspy import read
from obspy.core import UTCDateTime
import pandas as pd
import os
import glob
import numpy as np
import datetime as dt

# General path 
#path = "/home/calo/compartido/phd/phasenet/tmp_data/"
path = "/home/calo/compartido/AAA-master/BALANCE_CLASES/data/raw/test_data/"

# Crear lista de archivos
file_list = glob.glob(path + '*.sac.sr')


for i in file_list:
    st = read(i)
    year = i.split('/')[-1].split('.')[3]
    jday = i.split('/')[-1].split('.')[4]
    date = year + '-' + jday
    date = dt.datetime.strptime(date, "%Y-%j")
    time = dt.datetime.strptime("00:00:00", "%H:%M:%S").time()
    date = dt.datetime.combine(date,time)
    date = dt.datetime.strftime(date, '%Y-%m-%dT%H:%M:%S')
    start = UTCDateTime(date) 
    end = start + 86400
    st = st.trim(start, end, pad=True, fill_value=0)
    tr = st[0]
    tr.stats.sac.b = 0.0
    tr.stats.sac.e = 86400.00
    tr.stats.sac.nzjday = int(jday)
    tr.stats.sac.nzhour = 0
    tr.stats.sac.nzmin = 0
    tr.stats.sac.nzsec = 0
    tr.stats.sac.nzmsec = 0

    # Save standardized file
    tr.write(i,format="SAC")
    print(st[0].stats.station, st[0].stats.channel, st[0].stats.sampling_rate)

