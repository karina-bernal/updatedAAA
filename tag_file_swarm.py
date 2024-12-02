import numpy as np
import glob
import pandas as pd
import pickle
import time
import sys
import os

# Time control
tStartGlobal = time.time()

# Check arguments
if len(sys.argv) != 4:
    print('The way to run this script is: \n"python3 tag_file_swarm.py path_to_recording_files '
          'initial_day final_day", \n,'
          'please modify and run again.')
    print('\nUse julian day for initial and final days')
    sys.exit()

# Load analyzer
path_analyzer = sys.argv[1] + '/analyzer'
analyzer = pickle.load(open(path_analyzer, 'rb'))  # analyzer object

# Pass initial and final days
ini = sys.argv[2]
fin = sys.argv[3]

# Function to convert seconds to HH:MM:SS
def convert_time(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%02d:%02d:%02d" % (hour, min, sec)

# Genaral path
gen_path = sys.argv[1]

# Loop over recording files:
for i in range(int(ini), int(fin)+1):
    wild_card = '*.' + str(i).zfill(3) + '.*'
    path_rec = os.path.join(gen_path, wild_card)
    print(path_rec)
    file_rec = glob.glob(path_rec)[0]
    print('Loading file: {}'.format(file_rec))

    # Load rec file
    rec = pickle.load(open(file_rec, 'rb'))

    # Array with predicted probabilities:
    predP = rec['predictedProbas']
    # Array with decided clasess
    decClasses = rec['decidedClasses']
    # Array with associated probabilities:
    assoP = rec['associatedProbas']
    # Array with initial and final times for sliding window:
    w_starts = rec['w_starts'] / rec['fs']  # Convert to seconds
    w_ends = rec['w_ends'] / rec['fs']  # Convert to seconds
    # Biggining of signal:
    t_start_signal = rec['t_start']
    t_start_s = (t_start_signal.hour * 60 + t_start_signal.minute) * 60 + t_start_signal.second  # Convert to seconds

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


    # Map std label with real label:
    def numbers_to_class(argument):
        switcher = {
            0: classes[0],
            1: classes[1],
            2: classes[2],
            3: classes[3],
            4: classes[4],
            5: classes[5],
            -1: classes[6]
        }
        return switcher.get(argument, "nothing")


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
    df_pred.head()
    # Get a copy:
    df = df_pred.copy()

    # Merge all adjoining windows with the same label
    i = 0
    j = len(df['event'])
    while i < j:
        if i == (len(df['event']) - 1):
            break
        elif df['event'][i] == df['event'][i + 1]:
            df['end'][i] = df['end'][i + 1]
            df = df.drop([i + 1])
            df = df.reset_index(drop=True)
        else:
            i += 1

    # New dataframe with format needed in swarm tag file:
    df2 = pd.DataFrame(np.repeat(df.values, 2, axis=0))
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
    file = year + '_' + jday + '_AAA.csv'
    df2.to_csv(file, index=False, header=False)

    print('Tag file created: {}'.format(file))

tEndGlobal = time.time()

print('=================')
print('Total time of computation: {}'.format(convert_time(tEndGlobal-tStartGlobal)))
print()
print('=================')


