import matplotlib.pyplot as plt
from matplotlib import image
import pickle
import matplotlib.dates as mdates
import datetime
import numpy as np
import pandas as pd
import time
from pandas.plotting import register_matplotlib_converters
import os
from functions import read_sav
import sys
import cv2
import glob
register_matplotlib_converters()

line = 0

file = open('config.txt', 'r')
config = file.readlines()
path = config[0].splitlines()[0]
ftpsc = config[2].splitlines()[0]
instrument = config[3].splitlines()[0]
bflag = config[4].splitlines()[0]
start = config[5+line].splitlines()[0]

savepath = path + 'jplot/' + bflag + '/'

param_fil = glob.glob(savepath+'hi1hi2/' + start[0:4] + '/params/'+'jplot_hi1hi2_'+start+'*'+bflag[0]+'_params.pkl')

param_fil = param_fil[0]

with open(param_fil, 'rb') as f:
    time_beg1, time_end1, time_beg2, time_end2, e1_beg, e1_end, e2_beg, e2_end = pickle.load(f)

file_h1 = glob.glob(savepath+'hi_1/'+ start[0:4] + '/jplot_hi1_'+start+'_'+'*'+'UT_'+bflag[0]+'.png')[0]
file_h2 = glob.glob(savepath+'hi_2/'+ start[0:4] + '/jplot_hi2_'+start+'_'+'*'+'UT_'+bflag[0]+'.png')[0]

jplot1 = image.imread(file_h1)
jplot2 = image.imread(file_h2)

fig, ax = plt.subplots(figsize=(10, 5))

plt.ylim(4, 80)

plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24)))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
ax.xaxis_date()

plt.xlabel('Date (d/m/y)')
plt.ylabel('Elongation (°)')

#try:
  #date_seconds, elon_sav, dates = read_sav('/nas/helio/data/STEREO/HItracks/SATPLOT/' + start + '_' + ftpsc + '/satplot_track.sav')
  #ndates_sav = mdates.date2num(dates)

  #sav = 1

#except OSError:
  #print('No matching .sav file found.')

  #sav = 0

try:
  date_seconds, elon_sav, dates = read_sav('/nas/helio/data/STEREO/HItracks/' + start + '.sav')
  ndates_sav = mdates.date2num(dates)

  sav = 1

except OSError:
  print('No matching .sav file found.')

  sav = 0

try:
  f = open('/nas/helio/data/STEREO/HItracks/' + start + '.txt', 'r')
  strlist = f.readlines()[15:]

  elon_txt = [float(strlist[i].split()[0]) for i in range(len(strlist))]
  datestr = [strlist[i].split()[1] for i in range(len(strlist))]

  dates = [datetime.datetime.strptime(datestr[i], '%Y-%m-%dT%H:%M:%S') for i in range(len(datestr))]
  ndates_txt = [mdates.date2num(dates[i]) for i in range(len(dates))]

  txt = 1

except OSError:
  print('No matching .txt file found.')

  txt = 0

if txt and sav:

  ax.imshow(jplot1, cmap='gray', extent=[time_beg1, time_end1, e1_beg, e1_end], aspect='auto')
  ax.imshow(jplot2, cmap='gray', extent=[time_beg2, time_end2, e2_beg, e2_end], aspect='auto')

  plt.scatter(np.array(ndates_sav), np.array(elon_sav[0]), marker='+')
  plt.scatter(np.array(ndates_txt), np.array(elon_txt), marker='+')

  plt.show()

elif txt and not sav:

  ax.imshow(jplot1, cmap='gray', extent=[time_beg1, time_end1, e1_beg, e1_end], aspect='auto')
  ax.imshow(jplot2, cmap='gray', extent=[time_beg2, time_end2, e2_beg, e2_end], aspect='auto')

  plt.scatter(np.array(ndates_txt), np.array(elon_txt), marker='+')

  plt.show()

elif not txt and sav:
  ax.imshow(jplot1, cmap='gray', extent=[time_beg1, time_end1, e1_beg, e1_end], aspect='auto')
  ax.imshow(jplot2, cmap='gray', extent=[time_beg2, time_end2, e2_beg, e2_end], aspect='auto')

  #plt.scatter(np.array(ndates_sav), np.array(elon_sav[0]), marker='+')
  plt.scatter(np.array(ndates_sav), np.array(elon_sav), marker='+')
  plt.show()

else:
  ax.imshow(jplot1, cmap='gray', extent=[time_beg1, time_end1, e1_beg, e1_end], aspect='auto')
  ax.imshow(jplot2, cmap='gray', extent=[time_beg2, time_end2, e2_beg, e2_end], aspect='auto')

  plt.show()
