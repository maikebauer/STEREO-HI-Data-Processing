import matplotlib.pyplot as plt
import matplotlib.dates as d
from functions import rej_out, get_earth_pos, bin_elong, hi_img
import datetime
import numpy as np
import pandas as pd

bflag = str(input('Choose science or beacon data (science/beacon):'))
ftpsc = str(input('Enter the spacecraft (A/B):'))
start = str(input('Enter the start date (YYYYMMDD/today):'))


if start == 'today':
    start = datetime.datetime.today() - datetime.timedelta(days=7)
    start = start.strftime('%Y%m%d')

x_lims_hi1, dt_hi1, delon_hi1, y_hi1, imsh_im_hi1, den_hi1, el_hi1, path = hi_img(start, ftpsc, 'h1', 20, bflag)
x_lims_hi2, dt_hi2, delon_hi2, y_hi2, imsh_im_hi2, den_hi2, _, _ = hi_img(start, ftpsc, 'h2', el_hi1, bflag)

imsh_im_hi1 = np.nan_to_num(imsh_im_hi1)
imsh_im_hi2 = np.nan_to_num(imsh_im_hi2)


fig, ax = plt.subplots(figsize=(10, 7))

ax.imshow(imsh_im_hi1, origin='lower', cmap='gray',
          extent=[x_lims_hi1[1]-dt_hi1/2, x_lims_hi1[-1]+dt_hi1/2, y_hi1[0]-delon_hi1/2, y_hi1[-1]+delon_hi1/2], aspect='auto')

ax.imshow(imsh_im_hi2, origin='lower', cmap='gray',
          extent=[x_lims_hi2[1]-dt_hi2/2, x_lims_hi2[-1]+dt_hi2/2, y_hi2[0]-delon_hi2/2, y_hi2[-1]+delon_hi2/2], aspect='auto')

plt.gca().xaxis.set_major_locator(d.HourLocator(byhour=range(0, 24, 24)))
plt.gca().xaxis.set_major_formatter(d.DateFormatter('%d/%m/%y'))
ax.xaxis_date()

plt.xlabel('Date (d/m/y)')
plt.ylabel('Elongation (°)')

ax.set_ylim(0, 60)

data = []
inp = plt.ginput(0, 0)
data.append(inp)

plt.savefig(path+start+'_'+bflag+'.png')
plt.close()

elon = [d.num2date(data[0][i][1]) for i in range(len(data[0]))]
date_time_obj = [d.num2date(data[0][i][0]) for i in range(len(data[0]))]
date = [datetime.datetime.strftime(x, '%Y-%b-%d %H:%M:%S.%f') for x in date_time_obj]

elon_stdd = np.zeros(len(data[0]))
SC = [ftpsc for x in range(len(data[0]))]

pd_data = {'TRACK_DATE': date, 'ELON': elon, 'ELON_STDD': elon_stdd, 'SC': SC}

df = pd.DataFrame(pd_data, columns=['TRACK_DATE', 'ELON', 'SC', 'ELON_STDD'])
df.to_csv(path + start + '_' + ftpsc + '_' + bflag + '_red/' + 'track.csv', index=False, date_format='%Y-%m-%dT%H:M:S')
