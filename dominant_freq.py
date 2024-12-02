# --------------------------------------------------------------------
# Author: Karina Bernal Manzanilla
# Returns csv with class, date, time, length and dominant frequency
# of signals in the catalogue
# --------------------------------------------------------------------

import numpy as np
import pandas as pd
from obspy import read
import pickle
import datetime
from tools import butter_bandpass_filter, bestFFTlength
import time

pathToCatalogue = '/home/calo/compartido/AAA-master/UC1_PRUEBAS/data/shaped/catalogueGAIA_45hz_sr.pd'

# Start clock
tStart = time.time()

# Load catalogue
cat = pickle.load(open(pathToCatalogue, 'rb'))

# Number of events in the catalogue
nData = len(cat.index)

# Array of zeros for dominant frequencies
dom_freq = np.zeros((nData,), dtype=object)

# Array of zeros for maximum amplitude
max_amp = np.zeros((nData,), dtype=object)

# Run over all events:
for i in range(nData):
    secondFloat = cat.iloc[i]['second']
    tStartSignature = datetime.datetime(int(cat.iloc[i]['year']),
                                        int(cat.iloc[i]['month']),
                                        int(cat.iloc[i]['day']),
                                        int(cat.iloc[i]['hour']),
                                        int(cat.iloc[i]['minute']),
                                        int(secondFloat),
                                        int((secondFloat - int(secondFloat)) * 1000000))  # microseconds
    duration = cat.iloc[i]['length']

    # Get path to read signal
    path = cat['path'][i]

    # Get  day long signal
    st = read(path)
    trace = st[0]
    signal_day = trace.data

    # Trim to get data of event
    d0 = trace.stats.starttime
    trace_start = datetime.datetime(d0.year, d0.month, d0.day, d0.hour, d0.minute, d0.second)
    fs = trace.stats.sampling_rate
    tStartSignatureInRecording = (tStartSignature - trace_start).total_seconds()
    nStartSignatureInRecording = int(tStartSignatureInRecording * fs)
    nEndSignatureInRecording = nStartSignatureInRecording + int(duration * fs)
    signal = signal_day[nStartSignatureInRecording:nEndSignatureInRecording]  # event

    # Filtering
    f_min = cat['f0'][i]
    f_max = cat['f1'][i]
    butter_order = 4
    signal = butter_bandpass_filter(signal, f_min, f_max, fs, order=butter_order)

    # Get max amp:
    max_amp[i] = np.max(np.absolute(signal))

    # Fourier transform
    sig_freq = np.absolute(np.fft.fft(signal, bestFFTlength(len(signal))))

    # Frequencies of FT
    freqs = np.fft.fftfreq(bestFFTlength(len(signal)), d=1 / fs)
    positive_frequencies = freqs[np.where(freqs >= 0)]
    magnitudes = abs(sig_freq[np.where(freqs >= 0)])
    peak_frequency = np.argmax(magnitudes)
    idx_domf = positive_frequencies[peak_frequency]  # Index of dom freq

    # Add dom freq of signal to array
    dom_freq[i] = positive_frequencies[peak_frequency]

# Dictionary with info for csv:
data = {
    'class': cat['class'],
    'year': cat['year'],
    'month': cat['month'],
    'day': cat['day'],
    'hour': cat['hour'],
    'minute': cat['minute'],
    'second': cat['second'],
    'length': cat['length'],
    'amp': max_amp,
    'dom_f': dom_freq
}

# Data frame:
df = pd.DataFrame(data)

# Save dataframe as csv file
file = 'cat_dom_freq.csv'
df.to_csv(file, index=False)

tEnd = time.time()
print('=================')
print()
print('Tag file has been saved as: {}'.format(file))
print()
print('Total time of computation: {} s'.format(tEnd-tStart))
print()
print('=================')
