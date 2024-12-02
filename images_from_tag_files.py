# **** Este código se corre en el entorno base ****

from obspy import read
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm
import numpy as np
import pandas as pd
from datetime import datetime
from obspy.core.utcdatetime import UTCDateTime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os import mkdir
from os.path import isdir

# NEEDED VALUES
year = '2020'
jday = '038'
station = 'PPJU'
comps = ['HHZ', 'HHN', 'HHE']

# --------- FUNCTIONS -------------
# Function: Go to signal path
def signal_path(year, jday, station, comp):
    if station.startswith('P'):
        pre = "/home/calo/LEONARDA_POPO/"
        archivo = (pre + year + '/sacSR/' + jday + '/'
                   + 'CN' + '.' + station + '.' + comp
                   + '.' + year + '.' + jday
                   + '.' + '000000.sac.sr')
    else:
        pre = "/home/calo/compartido/bases_de_datos/sr2_sac/"
        archivo = (pre + year + '/' + + jday + '/'
                   + 'MC' + '.' + station + '.' + comp
                   + '.' + year + '.' + jday
                   + '.' + '000000.SAC.sr2')
    return archivo


def filter_with_obspy(event, fmin=0.66, fmax=45.0):
    event = event.detrend("linear")
    event = event.taper(max_percentage=0.05, type="hann")
    event = event.filter("bandpass", freqmin=fmin, freqmax=fmax,
                         corners=4, zerophase=True)
    return event


def get_trace(year, jday, station, comp):
    signal = signal_path(year=year, jday=jday, station=station, comp=comp)
    signal = read(signal)
    tr = signal[0].copy()
    tr = filter_with_obspy(tr)
    return tr


# DEAL WITH DATES
def deal_with_dates(i, tr):
    # Beginning
    date = datetime.strftime(tr.stats.starttime.date, '%y/%m/%d')
    df.at[i, 'beginning'] = date + ' ' + df.at[i, 'beginning']
    start = datetime.strptime(df.at[i, 'beginning'], '%y/%m/%d %H:%M:%S')
    start = UTCDateTime(start)
    # End
    df.at[i, 'end'] = date + ' ' + df.at[i, 'end']
    end = datetime.strptime(df.at[i, 'end'], '%y/%m/%d %H:%M:%S')
    end = UTCDateTime(end)
    return start, end


# DEAL WITH DATES FROM CENAPRED
def deal_with_dates_cenapred(i, tr):
    # Beginning
    df_cn.at[i, 'beginning'] = datetime.strptime(df_cn.at[i, 'beginning'], '%H:%M:%S').time()
    df_cn.at[i, 'beginning'] = datetime.combine(
        tr.stats.starttime.date, df_cn.at[i, 'beginning'])
    # End
    df_cn.at[i, 'end'] = datetime.strptime(df_cn.at[i, 'end'], '%H:%M:%S').time()    
    df_cn.at[i, 'end'] = datetime.combine(
        tr.stats.starttime.date, df_cn.at[i, 'end'])
    return None


def spectrogram(event, integration_len=250):
    tr = event
    w = signal.kaiser(integration_len, 18)
    w = w * integration_len / sum([pow(j, 2) for j in w])
    f, t, Sxx = signal.spectrogram(event.data, tr.stats.sampling_rate, nperseg=integration_len,
                                   nfft=1.5 * integration_len,
                                   noverlap=0.9 * integration_len, window=w)
    return f, t, Sxx


# Time axes:
def time_axes(comp, ax, label):
    # Array for times in %H:%M:%S
    ti = pd.date_range(start=comp.stats.starttime.datetime, end=comp.stats.endtime.datetime,
                       periods=comp.stats.npts)
    ax.plot(ti, comp.data, 'k', lw=0.5, label=label)
    ax.set_xlim(left=ti[0], right=ti[-1]);
    ax.set_ylabel('Velocidad \n(m/s)')
    ax.legend(loc=1)
    # ax.axvline(start.datetime)
    # ax.axvline(end.datetime)
    return None


def sectrogram_axes(comp, ax):
    f, t, Sxx = spectrogram(comp)
    ti2 = pd.date_range(start=comp.stats.starttime.datetime, end=comp.stats.endtime.datetime,
                        periods=len(t))
    spec_db = 10 * np.log10(Sxx)
    im = ax.pcolormesh(ti2, f, spec_db, cmap=cm.jet, shading='nearest', vmin=-185, vmax=-100)
    ax.set_ylim(bottom=0, top=45);
    ax.set_ylabel('Frecuencia (Hz)')
    return im


def aaa_win_vlines(start, end, axs):
    # HHZ:
    axs[0].axvline(start.datetime)
    axs[0].axvline(end.datetime)
    # HHN:
    axs[2].axvline(start.datetime)
    axs[2].axvline(end.datetime)
    # HHE:
    axs[4].axvline(start.datetime)
    axs[4].axvline(end.datetime)
    return None


def cn_win_vlines(df_cn, axs):
    # CENAPRED
    for i in range(len(df_cn['end'])):
        if df_cn.at[i, 'beginning'] > start_pad and df_cn.at[i, 'beginning'] < end_pad:
            label = df_cn.at[i, 'Evento'] + '_i'
            axs[0].axvline(df_cn.at[i, 'beginning'], label=label, color='orange')
            axs[2].axvline(df_cn.at[i, 'beginning'], label=label, color='orange')
            axs[4].axvline(df_cn.at[i, 'beginning'], label=label, color='orange')
            axs[0].legend(loc=1)
            axs[2].legend(loc=1)
            axs[4].legend(loc=1)

        if df_cn.at[i, 'end'] > start_pad and df_cn.at[i, 'end'] < end_pad:
            label = df_cn.at[i, 'Evento'] + '_f'
            axs[0].axvline(df_cn.at[i, 'end'], label=label, color='orange')
            axs[2].axvline(df_cn.at[i, 'end'], label=label, color='orange')
            axs[4].axvline(df_cn.at[i, 'end'], label=label, color='orange')
            axs[0].legend(loc=1)
            axs[2].legend(loc=1)
            axs[4].legend(loc=1)
    return None


# Plot func
def plot_win(i):
    fig, axs = plt.subplots(6, 1, figsize=(13, 12), gridspec_kw={'width_ratios': [1.2]})

    # Title
    title = (str(i).zfill(3) + '\n' + df['event'][i] + '\n'
             + start.datetime.strftime('%Y-%m-%d   %H:%M:%S'))
    axs[0].set_title(title)

    # HHZ
    divider_z = make_axes_locatable(axs[1])
    cax_z = divider_z.append_axes('right', size='1%', pad=0.05)
    # Time
    hhz_t = time_axes(comp=event_z, ax=axs[0], label='HHZ')
    # Spectrogram
    im = sectrogram_axes(comp=event_z, ax=axs[1])
    # Colorbar
    cbar_z = fig.colorbar(im, cax=cax_z, orientation='vertical')
    cbar_z.set_label('Intensidad dB')

    # HHN
    divider_n = make_axes_locatable(axs[3])
    cax_n = divider_n.append_axes('right', size='1%', pad=0.05)
    # Time
    hhn_t = time_axes(comp=event_n, ax=axs[2], label='HHN')
    # Spectrogram
    im = sectrogram_axes(comp=event_n, ax=axs[3])
    # Colorbar
    cbar_n = fig.colorbar(im, cax=cax_n, orientation='vertical')
    cbar_n.set_label('Intensidad dB')

    # HHE
    divider_e = make_axes_locatable(axs[5])
    cax_e = divider_e.append_axes('right', size='1%', pad=0.05)
    # Time
    hhe_t = time_axes(comp=event_e, ax=axs[4], label='HHE')
    # Spectrogram
    im = sectrogram_axes(comp=event_e, ax=axs[5])
    # Colorbar
    cbar_e = fig.colorbar(im, cax=cax_e, orientation='vertical')
    cbar_e.set_label('Intensidad dB')

    # AAA win vlines
    aaa_wins = aaa_win_vlines(start=start, end=end, axs=axs)

    # CENAPRED vlines:
    cn_vlines = cn_win_vlines(df_cn=df_cn, axs=axs)

    # Save figure
    name_fig = (grl_sv_path + '/' + str(i).zfill(3) + '_' + 
                year + '_' + jday + '_' +
                start.datetime.strftime("%H%M%S") + '_' +
                end.datetime.strftime("%H%M%S") + '_' +
                df['event'][i] + '.png')
    fig.savefig(name_fig, dpi=110, facecolor='w')
    return fig, axs


# Load tag file
tag_file = ('/home/calo/compartido/AAA-master/automatic_processing/' +
            'output_files/tag_files/' +
            year + '/' + jday + '/' +
            year + '_' + jday + '_' + station + '_all_066hz_AAA.csv')
df = pd.read_csv(tag_file)

# Load CENAPRED catalogue
cn_cat = ('/home/calo/compartido/AAA-master/automatic_processing/' +
          'output_files/comparacion_AAA_CENAPRED/'
          + year + '_' + jday + '.csv')
df_cn = pd.read_csv(cn_cat)

# Rename columns
df_cn = df_cn.rename(columns={"Inicio": "beginning", "Fin": "end"})

# Saving path for figures:
grl_sv_path = ('/home/calo/compartido/AAA-master/automatic_processing/'
               + 'output_files/tag_files/' +
               year + '/' + jday + '/figures_from_tag_files')
if not isdir(grl_sv_path):
    mkdir(grl_sv_path)

# Get FILTERED traces:
tr_z = get_trace(year=year, jday=jday, station=station, comp=comps[0])
tr_n = get_trace(year=year, jday=jday, station=station, comp=comps[1])
tr_e = get_trace(year=year, jday=jday, station=station, comp=comps[2])

# Change cenapred date format
for i in range(len(df_cn['Evento'])):
    deal_with_dates_cenapred(i, tr_z)

for i in range(len(df['end'])):
    start, end = deal_with_dates(i, tr_z)
    # Extended window (+/- 1 min)
    delta = 120  # (given in seconds)
    start_pad = start - delta
    end_pad = end + delta

    # Constrain to daily signal
    if start_pad < tr_z.stats.starttime:
        start_pad = tr_z.stats.starttime

    if end_pad > tr_z.stats.endtime:
        end_pad = tr_z.stats.endtime

    # Get windows for 3 components:
    event_z = tr_z.slice(start_pad, end_pad)
    event_n = tr_n.slice(start_pad, end_pad)
    event_e = tr_e.slice(start_pad, end_pad)

    # Create figure
    fig, axs = plot_win(i)
    plt.close(fig=fig)
    if i % 50 == 0:
        print(i)
