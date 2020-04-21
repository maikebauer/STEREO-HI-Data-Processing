import matplotlib.pyplot as plt
from matplotlib import image
import pickle
import matplotlib.dates as d
import datetime
import numpy as np
import pandas as pd
import time
from pandas.plotting import register_matplotlib_converters
import os

register_matplotlib_converters()

file = open('config.txt', 'r')
config = file.readlines()
path = config[0].splitlines()[0]
calpath = config[1].splitlines()[0]
start = config[2].splitlines()[0]
ftpsc = config[3].splitlines()[0]
instrument = config[4].splitlines()[0]
bflag = config[5].splitlines()[0]
high_contr = config[6].splitlines()[0]

savepath = path + start + '_' + ftpsc + '_' + bflag + '_red/'

if not os.path.exists(savepath + 'Tracks'):
        try:
            os.mkdir(savepath + 'Tracks')

        except OSError:
            print('Creation of the directory %s failed' % path)
        else:
            print('Directory successfuly created!\n')

with open(savepath + 'objs.pkl', 'rb') as f:
    time_beg, time_end, e1, e2 = pickle.load(f)

jplot = image.imread(savepath + start + '_jplot.png')

fig, ax = plt.subplots(figsize=(10, 5))

plt.ylim(3, 50)

plt.gca().xaxis.set_major_locator(d.HourLocator(byhour=range(0, 24, 24)))
plt.gca().xaxis.set_major_formatter(d.DateFormatter('%d/%m/%y'))
ax.xaxis_date()

plt.xlabel('Date (d/m/y)')
plt.ylabel('Elongation (°)')

ax.imshow(jplot, cmap='gray', extent=[time_beg, time_end, e1, e2], aspect='auto')

plt.savefig(savepath + start + '.png', bbox_inches = 0)

data = []
inp = fig.ginput(n=-1, timeout=-1, mouse_add=3, mouse_pop=2, mouse_stop=1)
data.append(inp)

elon = [data[0][i][1] for i in range(len(data[0]))]
date_time_obj = [d.num2date(data[0][i][0]) for i in range(len(data[0]))]
date_time_obj.sort()
date = [datetime.datetime.strftime(x, '%Y-%b-%d %H:%M:%S.%f') for x in date_time_obj]

elon_stdd = np.zeros(len(data[0]))
SC = [ftpsc for x in range(len(data[0]))]

pd_data = {'TRACK_DATE': date, 'ELON': elon, 'ELON_STDD': elon_stdd, 'SC': SC}

df = pd.DataFrame(pd_data, columns=['TRACK_DATE', 'ELON', 'SC', 'ELON_STDD'])
df.to_csv(savepath + 'Tracks/' + 'track.csv', index=False, date_format='%Y-%m-%dT%H:M:S')
