import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
from functions import rej_out, get_earth_pos, bin_elong, hi_img
import datetime

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
import datetime

ftpsc = str(input('Enter the spacecraft (A/B):'))
start = str(input('Enter the start date (YYYYMMDD/today):'))

if start == 'today':
    start = datetime.datetime.today() - datetime.timedelta(days=7)
    start = start.strftime('%Y%m%d')

x_lims_hi1, dt_hi1, delon_hi1, y_hi1, imsh_im_hi1, den_hi1, el_hi1, path = hi_img(start, ftpsc, 'hi_1', 20)
x_lims_hi2, dt_hi2, delon_hi2, y_hi2, imsh_im_hi2, den_hi2, _, _ = hi_img(start, ftpsc, 'hi_2', el_hi1)

fig, ax = plt.subplots(figsize=(10,7))

ax.imshow(imsh_im_hi1, origin='lower', cmap='gray',
          extent=[x_lims_hi1[1], x_lims_hi1[-1], y_hi1[0]-delon_hi1/2, y_hi1[-1]+delon_hi1/2], aspect='auto')

ax.imshow(imsh_im_hi2, origin='lower', cmap='gray',
          extent=[x_lims_hi2[1], x_lims_hi2[-1], y_hi2[0]-delon_hi2/2, y_hi2[-1]+delon_hi2/2], aspect='auto')

plt.gca().xaxis.set_major_locator(HourLocator(byhour=range(0, 24, 24)))
plt.gca().xaxis.set_major_formatter(DateFormatter('%d/%m/%y'))
ax.xaxis_date()

plt.xlabel('Date (d/m/y)')
plt.ylabel('Elongation (°)')

ax.set_ylim(0, 60)
plt.savefig('/Users/student/Downloads/'+start+'.png')