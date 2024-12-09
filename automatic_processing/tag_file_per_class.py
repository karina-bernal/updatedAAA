import numpy as np
import glob
import pandas as pd
import pickle
import time
import sys
import os
from os.path import isdir

# Time control
tStartGlobal = time.time()

# Check arguments
if len(sys.argv) != 5:
    print('The way to run this script is: \n"python3 tag_file_per_class.py path_to_recording_files '
          'year initial_day final_day", \n,'
          'please modify and run again.')
    print('\nUse julian day for initial and final days')
    sys.exit()

# Load analyzer
path_analyzer = sys.argv[1] + '/analyzer'
analyzer = pickle.load(open(path_analyzer, 'rb'))  # analyzer object

# General path to save tag_files:
path_tag_files = '/home/calo/compartido/AAA-master/automatic_processing/output_files/training_LPTRNOOT/tag_files/'

# Path to save csv with number of events per class, per day:
path_csv_counts = path_tag_files + '/number_of_events_summaries/'

# Pass year, initial and final days
year_files = sys.argv[2]
ini = sys.argv[3]
fin = sys.argv[4]

# Name of summary file:
name_summary_file = path_csv_counts + 'summary_' + year_files + '_' + ini + '_' + fin + '.csv'

# =================================================
# NEEDED FUNCTIONS:

# Function to convert seconds to HH:MM:SS
def convert_time(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%02d:%02d:%02d" % (hour, min, sec)

# Function: Map standard label with real label:
def numbers_to_class(argument):
    switcher = {
        0: classes[0],
        1: classes[1],
        2: classes[2],
        3: classes[3],
        4: classes[4],
        #-1: classes[3],
       # 5: classes[5],
       # -1: classes[6]
    }
    return switcher.get(argument, "nothing")


# Function: Dictionary --> data frame for class X:
def class_dataframe(x_beg, x_end, x):
    class_dict = {
        'event': x,
        'beginning': x_beg,
        'end': x_end
    }
    # Data frame:
    x_df = pd.DataFrame(class_dict)
    return x_df

# Function: Merge all adjoining windows with the same label
def merge_adjoining(x_df):
    i = 0
    j = len(x_df['event'])
    while i < j:
        if i == (len(x_df['event']) - 1):
            break
        elif x_df['beginning'][i+1] <= x_df['end'][i]:
            x_df['end'][i] = x_df['end'][i+1]
            x_df = x_df.drop([i + 1])
            x_df = x_df.reset_index(drop=True)
        else:
            i += 1
    return x_df


# Funtion: Merge all data frames
# I removed re_df from arguments (KBM 2021/11/21)
def merge_df(lp_df, tr_df, no_df, un_df, ot_df):
#def merge_df(ex_df, lp_df, tr_df, vt_df, no_df, un_df):
    beg, end, event = [], [], []
    #for i in range(len(ex_df['event'])):
        #beg.append(ex_df['beginning'][i])
        #end.append(ex_df['end'][i])
        #event.append(ex_df['event'][i])
    for i in range(len(lp_df['event'])):
        beg.append(lp_df['beginning'][i])
        end.append(lp_df['end'][i])
        event.append(lp_df['event'][i])
    #for i in range(len(re_df['event'])):
        #beg.append(re_df['beginning'][i])
        #end.append(re_df['end'][i])
        #event.append(re_df['event'][i])
    for i in range(len(tr_df['event'])):
        beg.append(tr_df['beginning'][i])
        end.append(tr_df['end'][i])
        event.append(tr_df['event'][i])
    for i in range(len(ot_df['event'])):
        beg.append(ot_df['beginning'][i])
        end.append(ot_df['end'][i])
        event.append(ot_df['event'][i])
    for i in range(len(no_df['event'])):
        beg.append(no_df['beginning'][i])
        end.append(no_df['end'][i])
        event.append(no_df['event'][i])
    for i in range(len(un_df['event'])):
        beg.append(un_df['beginning'][i])
        end.append(un_df['end'][i])
        event.append(un_df['event'][i])
    # Make dict:
    all_dict = {
        'event': event,
        'beginning': beg,
        'end': end
    }
    # Make df:
    all_df = pd.DataFrame(all_dict)
    # Sort:
    all_df = all_df.sort_values(by=['beginning'])
    return all_df


# Function: New dataframe with format needed in swarm tag file for each class:
def mk_tagfile(x_df, events):
    if len(x_df['event']) == 0:
        return print('No data for:', events + ' class')

    df2 = pd.DataFrame(np.repeat(x_df.values, 2, axis=0))
    df2.rename(columns={0: 'start',
                        1: 'end',
                        2: 'event'}, inplace=True)
    # End of event/window needs to be a new tag:
    for i in range(1, len(df2['event']), 2):
        df2['event'][i] += '-end'
        df2['start'][i] = df2['end'][i - 1]
    # Second column of tag file has info of station, component and net
    # First column has date and time:
    for i in range(len(df2['end'])):
        df2['end'][i] = station + ' ' + component + ' ' + net
        df2['start'][i] = date + ' ' + df2['start'][i]
    # Save dataframe as csv file
    file = dir + '/' + year + '_' + jday + '_' + events + '_' + station + '_066hz_AAA.csv'
    df2.to_csv(file, index=False, header=False)
    # print('Created file: '.format(file))
    return

#====================================================================

# Genaral path (for analyzer and recording objects):
gen_path = sys.argv[1]

# Empty DataFrame to store summary of number of events per class, per day
# I deleted 'RE' from column_names (KBM, 2022/11/21)
#column_names = ['DAY', 'EX', 'LP', 'TR', 'VT', 'NO', 'UN']
column_names = ['DAY', 'LP', 'TR', 'NO', 'OT', 'UN']
summ_df = pd.DataFrame(columns=column_names)

# Loop over recording files:
for i in range(int(ini), int(fin)+1):
    wild_card = '*.' + str(i).zfill(3) + '.*'
    path_rec = os.path.join(gen_path, wild_card)
    print(path_rec)
    file_rec = glob.glob(path_rec)[0]
    print('---------------------------------')
    print('Loading file: {}\n'.format(file_rec))

    # Load rec file
    rec = pickle.load(open(file_rec, 'rb'))

    # Load some important arrays

    # Array with predicted probabilities:
    predP = rec['predictedProbas']
    # Array with decided clasess
    decClasses = rec['decidedClasses']
    # Array with associated probabilities:
    assoP = rec['associatedProbas']
    # Array with initial and final times for sliding window:
    w_starts = rec['w_starts'] / rec['fs']  # Convert to seconds
    w_ends = rec['w_ends'] / rec['fs']  # Convert to seconds
    # Beginning of signal:
    t_start_signal = rec['t_start']
    # Covevert initial time to seconds:
    t_start_s = (t_start_signal.hour * 60 + t_start_signal.minute) * 60 + t_start_signal.second

    n_windows = w_starts.shape[1]  # Number of windows (steps of sliding window)
    # Reshapes:
    w_starts = np.reshape(w_starts, (n_windows,))
    w_ends = np.reshape(w_ends, (n_windows,))
    decClasses = np.reshape(decClasses, (n_windows,))

    # Arrays of time with right format HH:MM:SS
    w_starts_t = np.zeros((n_windows,), dtype=object)
    w_ends_t = np.zeros((n_windows,), dtype=object)

    for i in range(len(w_starts)):
        w_starts_t[i] = convert_time(w_starts[i] + t_start_s)
        w_ends_t[i] = convert_time(w_ends[i] + t_start_s)

    # Needed values:
    station = rec['path'].split('/')[-1].split('.')[1]
    component = rec['path'].split('/')[-1].split('.')[2]
    net = rec['path'].split('/')[-1].split('.')[0]
    date = rec['t_start'].strftime("%Y-%m-%d")
    year = rec['path'].split('/')[-1].split('.')[3]
    jday = rec['path'].split('/')[-1].split('.')[4]

    # Load label encoder to know labels
    labelEncoder = analyzer['labelEncoder']
    # Get classes:
    classes = labelEncoder.classes_
    # Add 'unknown' (not in original data):
    classes = np.append(classes, ['unknown'])

    # Array for real classes:
    decClasses_real = np.zeros((n_windows,), dtype=object)
    for i in range(len(decClasses)):
        decClasses_real[i] = numbers_to_class(decClasses[i])  # Map

    # Dictionary with prediction data to create data frame:
    predictions_data = {
        'event': decClasses_real,
        'beginning': w_starts_t,
        'end': w_ends_t
    }
    # Data frame:
    df_pred = pd.DataFrame(predictions_data)

    # Lists per class:
    ex_beg, ex_end, ex = [], [], []
    lp_beg, lp_end, lp = [], [], []
    #re_beg, re_end, re = [], [], []
    tr_beg, tr_end, tr = [], [], []
    ot_beg, ot_end, ot = [], [], []
    no_beg, no_end, no = [], [], []
    un_beg, un_end, un = [], [], []

    # Sort row for each event into their lists
    for i in range(len(df_pred['event'])):
        #if df_pred['event'][i] == 'Explosion':
        #    ex_beg.append(df_pred['beginning'][i])
        #    ex_end.append(df_pred['end'][i])
        #    ex.append('Explosion')
        if df_pred['event'][i] == 'LP':
            lp_beg.append(df_pred['beginning'][i])
            lp_end.append(df_pred['end'][i])
            lp.append('LP')
        #elif df_pred['event'][i] == 'Regional':
            #re_beg.append(df_pred['beginning'][i])
            #re_end.append(df_pred['end'][i])
            #re.append('Regional')
        elif df_pred['event'][i] == 'Tremor':
            tr_beg.append(df_pred['beginning'][i])
            tr_end.append(df_pred['end'][i])
            tr.append('Tremor')
        elif df_pred['event'][i] == 'Other':
            ot_beg.append(df_pred['beginning'][i])
            ot_end.append(df_pred['end'][i])
            ot.append('OT')
        elif df_pred['event'][i] == 'noise':
            no_beg.append(df_pred['beginning'][i])
            no_end.append(df_pred['end'][i])
            no.append('noise')
        elif df_pred['event'][i] == 'unknown':
            un_beg.append(df_pred['beginning'][i])
            un_end.append(df_pred['end'][i])
            un.append('unknown')

    #print('Number of windows:', len(df_pred['event']))
    # I deleted + len(re) from print (KBM, 2022/11/21)
    #print('Sum of events', len(lp) + len(ex) + len(tr)
          #+ len(vt) + len(no) + len(un))

    # Make data frames for each class
    #ex_df = class_dataframe(ex_beg, ex_end, ex)
    lp_df = class_dataframe(lp_beg, lp_end, lp)
    #re_df = class_dataframe(re_beg, re_end, re)
    tr_df = class_dataframe(tr_beg, tr_end, tr)
    ot_df = class_dataframe(ot_beg, ot_end, ot)
    no_df = class_dataframe(no_beg, no_end, no)
    un_df = class_dataframe(un_beg, un_end, un)

    # Merge adjoining windows of the same class
    #ex_df = merge_adjoining(ex_df)
    lp_df = merge_adjoining(lp_df)
    #re_df = merge_adjoining(re_df)
    tr_df = merge_adjoining(tr_df)
    ot_df = merge_adjoining(ot_df)
    no_df = merge_adjoining(no_df)
    un_df = merge_adjoining(un_df)

    print('Number of events for each class:')
    #print('Ex:', len(ex_df))
    print('LP:', len(lp_df))
    #print('Re:', len(re_df))
    print('Tr:', len(tr_df))
    print('OT:', len(ot_df))
    print('No:', len(no_df))
    print('Un:', len(un_df))

    # Store number of events per class, per day on summary DataFrame
    # I deleted 'RE': len(re_df) from dic (KBM, 2022/11/21)
    #dic = {'DAY': year + '/' + jday, 'EX': len(ex_df),
    #       'LP': len(lp_df),
    #       'TR': len(tr_df), 'VT': len(vt_df),
    #       'NO': len(no_df), 'UN': len(un_df)}
    dic = {'DAY': year + '/' + jday, 
           'LP': len(lp_df),
           'TR': len(tr_df), 'OT': len(ot_df),
           'NO': len(no_df), 'UN': len(un_df)}
    
    summ_df = summ_df.append(dic, ignore_index=True)

    # I deleted re_df from all_df (KBM, 2022/11/21)
    #all_df = merge_df(ex_df, lp_df, tr_df, vt_df, no_df, un_df)
    all_df = merge_df(lp_df, tr_df, ot_df, no_df, un_df)

    # Save all_df as csv
    dir = path_tag_files + '/' + year + '/' + jday
    if not isdir(dir):
        os.makedirs(dir)
    output_file = dir + '/' + year + '_' + jday + '_' + station + '_all_066hz_AAA.csv'
    all_df.to_csv(output_file, index=False, header=True)

    #mk_tagfile(ex_df, 'ex')
    mk_tagfile(lp_df, 'lp')
    #mk_tagfile(re_df, 're')
    mk_tagfile(tr_df, 'tr')
    mk_tagfile(ot_df, 'ot')
    mk_tagfile(no_df, 'no')
    mk_tagfile(un_df, 'un')

# Save summary file
summ_df.to_csv(name_summary_file, index=False, header=True)

tEndGlobal = time.time()

print('=================')
print('Total time of computation: {}'.format(convert_time(tEndGlobal-tStartGlobal)))
print()
print('=================')
