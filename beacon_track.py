import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import sunpy.io
import sunpy.map
import pandas as pd
import pyreadstat
import datetime

ftpsc = str(input('Enter the spacecraft (A/B):'))
instrument = str(input('Enter the instrument (hi_1/hi_2):'))
start = str(input('Enter the start date (YYYYMMDD):'))

file = open('config.txt', 'r')
path = file.readlines()
path = path[0].splitlines()[0]

red_path = path + '/' + start + '_' + ftpsc + '_red/'
files = []

for file in sorted(glob.glob(red_path + '*' + instrument + '*.fts')):
    files.append(file)

data = [fits.getdata(files[i]) for i in range(len(files))]
header = [fits.getheader(files[i]) for i in range(len(files))]

derot_map = [sunpy.map.Map(data[k], header[k]) for k in range(len(files))]

w = []

# coordinates of derotated images are put into a list

print('Converting coordinates...')

for i in range(len(files)):
    w.append(derot_map[i].wcs)

px = np.arange(0, np.shape(derot_map[0].data)[0], 1)
py = np.arange(0, np.shape(derot_map[0].data)[1], 1)

tx = np.zeros((len(files), len(px)))
ty = np.zeros((len(files), len(py)))

# pixel coordinates are converted to world coordinates

for i in range(len(files)):
    tx[i, :], ty[i, :] = w[i].wcs_pix2world(px, py, 1)

# convert to radians

tx = tx * np.pi / 180
ty = ty * np.pi / 180

hrln = np.zeros(shape=(len(files), len(px), len(py)))  # psi
hrlt = np.zeros(shape=(len(files), len(px), len(py)))  # elongation

# convert from helioprojective cartesian to helioprojective radial coordinates

for k in range(len(files)):
    for i in range(len(px)):
        for j in range(len(py)):
            hrln[k, i, j] = math.atan2(-np.cos(tx[k, i]) * np.sin(ty[k, j]), np.sin(tx[k, i]))
            hrlt[k, i, j] = math.atan2(np.sqrt(math.pow(np.cos(tx[k, i]), 2) * math.pow(np.sin(ty[k, j]), 2)
                                               + math.pow(np.sin(tx[k, i]), 2)), np.cos(tx[k, i]) * np.cos(ty[k, j]))

# converto to degrees

hrln = hrln * 180 / np.pi
hrlt = hrlt * 180 / np.pi

# arrays are necessary for plotting later

maxlt = []
minlt = []
maxln = []
minln = []

for i in range(len(files)):
    maxlt.append(np.max(hrlt[i]))
    minlt.append(np.min(hrlt[i]))

    maxln.append(np.max(hrln[i]))
    minln.append(np.min(hrln[i]))

dlt = []
dln = []

for i in range(len(derot_map)):
    dlt.append((maxlt[i] - minlt[i]) / (derot_map[i].data.shape[0] - 1))
    dln.append((maxln[i] - minln[i]) / (derot_map[i].data.shape[1] - 1))

# dates are inserted into list

date = []

for i in range(len(derot_map)):
    date.append(derot_map[i].date.value)

# line used to mark ecliptic in plots

ecliptic = np.zeros((2, 2, len(files)))

for i in range(len(files)):
    ecliptic[:, :, i] = np.array([[maxlt[i], minlt[i]], [0, 0]])


# following block of code is only necessary if science tracks are supposed to be overplotted on beacon images

# comp_sec, comp_elon, _ = read_sav(path+datelist_int[0]+'.sav')

# date_dif = []

# for i in range(len(header)-1):
#    date_dif.append(datetime.datetime.strptime(date[i+1], '%Y-%m-%dT%H:%M:%S.%f') - datetime.datetime(1970, 1, 1))

# date_sec = [date_dif[i].total_seconds() for i in range(len(date_dif))]

# interp_elon = np.interp(date_sec, comp_sec, comp_elon[0])

# plot final images
# axes must be aligned with image data
# x and y values must be located in the middle of a pixel, not the edges
# axes must be correctly oriented for ahead and behind data

def handle(event):
    if event.key == 'escape':
        plt.close(fig)

data = []

for i in range(len(derot_map)):
    fig, ax = plt.subplots()
    if ftpsc == 'B':
        im = derot_map[i].plot(
            extent=[maxlt[i] + dlt[i] / 2, minlt[i] - dlt[i] / 2, minln[i] - dln[i] / 2, maxln[i] + dln[i] / 2], aspect='auto')

    if ftpsc == 'A':
        im = derot_map[i].plot(
            extent=[minlt[i] - dlt[i] / 2, maxlt[i] + dlt[i] / 2, minln[i] - dln[i] / 2, maxln[i] + dln[i] / 2], aspect='auto')

    ax.plot([maxlt, minlt], [0, 0], color='k', linestyle='--', linewidth=1)
    fig.canvas.mpl_connect('key_press_event', handle)
    # ax.plot(interp_elon[i], 0, 'g*')
    ax.set_xlabel('Elongation (deg)')

    # points that are clicked on are put into list
    inp = plt.ginput(0, 0)
    data.append(inp)

    if inp == []:
        date[i] = []

    plt.show()

date = [x for x in date if x != []]
date = [x.replace('T', ' ') for x in date]
date_time_obj = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') for x in date]
date = [datetime.datetime.strftime(x, '%Y-%b-%d %H:%M:%S.%f') for x in date_time_obj]

data = [x for x in data if x != []]
data = [x[0][0] for x in data]

# list of elongations and dates is saved to .sav file
#print(data[0])
#print(data[1])
#print(data[:][:][0])
#print(data[0][0][0])
#print(np.shape(data))

elon_stdd = np.zeros(len(data))
SC = [ftpsc for x in range(len(data))]

pd_data = {'TRACK_DATE': date, 'ELON': data, 'ELON_STDD': elon_stdd, 'SC': [ftpsc for x in range(len(data))]}

df = pd.DataFrame(pd_data, columns=['TRACK_DATE', 'ELON', 'SC', 'ELON_STDD'])

df.to_csv(red_path+'track.csv', index=False, date_format='%Y-%m-%dT%H:M:S')
