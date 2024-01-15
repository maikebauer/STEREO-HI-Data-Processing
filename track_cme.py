import matplotlib.pyplot as plt
import matplotlib
from matplotlib import image
import pickle
import matplotlib.dates as mdates
import datetime
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import os
import glob
import sys

matplotlib.use("TkAgg")

register_matplotlib_converters()

line = 0

if os.path.isfile('config.txt'):
	config_path = 'config.txt'
else:
	config_path = 'sample_config.txt'
        
file = open(config_path, 'r')
config = file.readlines()

path = config[0].splitlines()[0]
save_path = config[1].splitlines()[0]
datpath = config[2].splitlines()[0]
ftpsc = config[3].splitlines()[0].split(',')
instrument = config[4].splitlines()[0]
bflag = config[5].splitlines()[0].split(',')
start = config[6].splitlines()[0].split(',')
mode = config[7].splitlines()[0]
task = config[8].splitlines()[0]
save_img = config[9].splitlines()[0]
silent = config[10].splitlines()[0]

list_len = len(ftpsc)

if any(len(lst) != list_len for lst in [ftpsc, bflag, start]):
    print('Number of specified spacecraft, dates, and/or science/beacon arguments does not match. Exiting...')
    sys.exit()


for num in range(list_len):

    savepath = path + 'jplot/' + ftpsc[num] + '/' + bflag[num] + '/'

    param_fil = glob.glob(savepath + 'hi1hi2/' + start[num][0:4] + '/params/' + 'jplot_hi1hi2_' + start[num] + '*' + bflag[num][0] + '_params.pkl')

    param_fil = param_fil[0]

    with open(param_fil, 'rb') as f:
        t_h1_beg, t_h1_end, t_h2_beg, t_h2_end, e_h1_beg, e_h1_end, e_h2_beg, e_h2_end = pickle.load(f)

    try:
        file_h1 = glob.glob(savepath + 'hi_1/' + start[num][0:4] + '/jplot_hi1_' + start[num] + '_*_' + ftpsc[num] + '_' + bflag[num][0] + '.pkl')[0]

    except IndexError:
        print('No file found under ' + savepath + 'hi_1/' + start[num][0:4] + '/jplot_hi1_' + start[num] + '_*_' + ftpsc[num] + '_' + bflag[num][0] + '.pkl. Exiting...')
        sys.exit()

    try:
        file_h2 = glob.glob(savepath + 'hi_2/' + start[num][0:4] + '/jplot_hi2_' + start[num] + '_*_' + ftpsc[num] + '_' + bflag[num][0] + '.pkl')[0]

    except IndexError:
        print('No file found under ' + savepath + 'hi_2/' + start[num][0:4] + '/jplot_hi2_' + start[num] + '_*_' + ftpsc[num] + '_' + bflag[num][0] + '.pkl. Exiting...')
        sys.exit()
        
    with open(file_h1, 'rb') as f:
        img_rescale_h1, orig_h1 = pickle.load(f)

    with open(file_h2, 'rb') as f:
        img_rescale_h2, orig_h2 = pickle.load(f)
    
    vmin_h1 = np.nanmedian(img_rescale_h1) - 2 * np.nanstd(img_rescale_h1)
    vmax_h1 = np.nanmedian(img_rescale_h1) + 2 * np.nanstd(img_rescale_h1)

    vmin_h2 = np.nanmedian(img_rescale_h2) - 2 * np.nanstd(img_rescale_h2)
    vmax_h2 = np.nanmedian(img_rescale_h2) + 2 * np.nanstd(img_rescale_h2)
    
    if bflag[num] == 'beacon':
        cadence_h1 = 120.0
        cadence_h2 = 120.0

    if bflag[num] == 'science':
        cadence_h1 = 40.0
        cadence_h2 = 120.0    
        
    dt_h1 = (cadence_h1/60)/24
    dt_h2 = (cadence_h2/60)/24
    
    dy_h1 = np.abs((e_h1_end-e_h1_beg)/(np.shape(img_rescale_h1)[0]-1))
    dy_h2 = np.abs((e_h2_end-e_h2_beg)/(np.shape(img_rescale_h2)[0]-1))
    
    fig, ax = plt.subplots(figsize=(10, 5))

    plt.ylim(4, 80)

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24)))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    ax.xaxis_date()

    plt.xlabel('Date (d/m/y)')
    plt.ylabel('Elongation (°)')

    img_rescale_h1 = img_rescale_h1.astype(float)
    img_rescale_h2 = img_rescale_h2.astype(float)

    ax.imshow(img_rescale_h1, cmap='gray', aspect='auto', vmin=vmin_h1, vmax=vmax_h1, interpolation='none', origin=orig_h1, extent=[mdates.date2num(t_h1_beg), mdates.date2num(t_h1_end), e_h1_beg, e_h1_end])
    ax.imshow(img_rescale_h2, cmap='gray', aspect='auto', vmin=vmin_h2, vmax=vmax_h2, interpolation='none', origin=orig_h2, extent=[mdates.date2num(t_h2_beg), mdates.date2num(t_h2_end), e_h2_beg, e_h2_end])
    ax.set_title(start[num] + ' STEREO-' + ftpsc[num])
    
    data = []
    inp = fig.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=2, show_clicks=True)
    data.append(inp)
    elon = [data[0][i][1] for i in range(len(data[0]))]
    date_time_obj = [mdates.num2date(data[0][i][0]) for i in range(len(data[0]))]
    date_time_obj.sort()
    date = [datetime.datetime.strftime(x, '%Y-%b-%d %H:%M:%S.%f') for x in date_time_obj]

    elon_stdd = np.zeros(len(data[0]))
    SC = [ftpsc for x in range(len(data[0]))]

    pd_data = {'TRACK_DATE': date, 'ELON': elon, 'ELON_STDD': elon_stdd, 'SC': SC}

    csv_path = savepath + '/hi1hi2/' + start[num][0:4] + '/Tracks/' + start[num] + '/'

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    prev_files = glob.glob(csv_path + start[num] + '_' + bflag[num][0] + '_' + ftpsc[num] + '_track' + '_*.csv')
    num_files = len(prev_files) + 1

    df = pd.DataFrame(pd_data, columns=['TRACK_DATE', 'ELON', 'SC', 'ELON_STDD'])
    df.to_csv(csv_path + start[num] + '_' + bflag[num][0] + '_' + ftpsc[num] + '_track' + '_' + str(num_files) + '.csv', index=False, date_format='%Y-%m-%dT%H:%M:%S')
    plt.close()
