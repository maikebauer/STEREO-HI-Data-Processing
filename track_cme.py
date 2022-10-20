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

matplotlib.use("TkAgg")
register_matplotlib_converters()

line = 0

file = open('config.txt', 'r')
config = file.readlines()
path = config[0].splitlines()[0]
ftpsc = config[3].splitlines()[0]
instrument = config[4].splitlines()[0]
bflag = config[5].splitlines()[0]
start = config[6 + line].splitlines()[0]

savepath = path + 'jplot/' + ftpsc + '/' + bflag + '/'

param_fil = glob.glob(savepath + 'hi1hi2/' + start[0:4] + '/params/' + 'jplot_hi1hi2_' + start + '*' + bflag[0] + '_params.pkl')
param_fil = param_fil[0]

with open(param_fil, 'rb') as f:
    time_beg1, time_end1, time_beg2, time_end2, e1_beg, e1_end, e2_beg, e2_end, dy1, dy2 = pickle.load(f)

file_h1 = glob.glob(savepath + 'hi_1/' + start[0:4] + '/jplot_hi1_' + start + '*' + '.png')[0]
file_h2 = glob.glob(savepath + 'hi_2/' + start[0:4] + '/jplot_hi2_' + start + '*' + '.png')[0]

jplot1 = image.imread(file_h1)
jplot2 = image.imread(file_h2)

fig, ax = plt.subplots(figsize=(10, 5))

plt.ylim(4, 30)

plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24)))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
ax.xaxis_date()

plt.xlabel('Date (d/m/y)')
plt.ylabel('Elongation (°)')

if bflag == 'science':
    dx1 = 40/(60*24)
    dx2 = 120/(60*24)

if bflag == 'beacon':
    dx1 = 120/(60*24)
    dx2 = 120/(60*24)

ax.imshow(jplot1, cmap='gray', extent=[time_beg1, time_end1+dx1, e1_beg, e1_end+dy1], aspect='auto')
ax.imshow(jplot2, cmap='gray', extent=[time_beg2, time_end2+dx2, e2_beg, e2_end+dy2], aspect='auto')

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

csv_path = savepath + '/hi1hi2/' + start[0:4] + '/Tracks/' + start + '/'

if not os.path.exists(csv_path):
    os.makedirs(csv_path)

prev_files = glob.glob(csv_path + start + '_' + bflag[0] + '_' + ftpsc + '_track' + '_*.csv')
num = len(prev_files) + 1

df = pd.DataFrame(pd_data, columns=['TRACK_DATE', 'ELON', 'SC', 'ELON_STDD'])
df.to_csv(csv_path + start + '_' + bflag[0] + '_' + ftpsc + '_track' + '_' + str(num) + '.csv', index=False, date_format='%Y-%m-%dT%H:M:S')