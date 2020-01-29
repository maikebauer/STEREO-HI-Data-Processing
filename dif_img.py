import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import glob
import sunpy.io
import sunpy.map
import datetime
from skimage.exposure import equalize_adapthist
from datetime import datetime
from scipy import ndimage
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, HourLocator
from functions import rej_out, get_earth_pos, bin_elong

ftpsc = str(input('Enter the spacecraft (A/B):'))
instrument = str(input('Enter the instrument (hi_1/hi_2):'))
start = str(input('Enter the start date (YYYYMMDD):'))

cadence = 120.0
maxgap = -3.5

file = open('config.txt', 'r')
path = file.readlines()
path = path[0].splitlines()[0]

red_path = path + start + '_' + ftpsc + '_red/'
files = []

for file in sorted(glob.glob(red_path + '*' + instrument + '*' + '*.fts')):
    files.append(file)

data = np.array([fits.getdata(files[i]) for i in range(len(files))])
header = [fits.getheader(files[i]) for i in range(len(files))]

missing = np.array([header[i]['NMISSING'] for i in range(len(header))])
mis = np.array(np.where(missing > 0))

if np.size(mis) > 0:
    for i in mis:
        data[i, :, :] = np.nan

time = [header[i]['DATE-OBS'] for i in range(len(header))]
time_obj = [Time(time[i], format='isot', scale='utc') for i in range(len(time))]

nandata = np.full_like(data[0], np.nan)

r_dif = [nandata]

for i in range(1, len(time)):
    tdiff = (time_obj[i] - time_obj[i - 1]).sec / 60
    if (tdiff <= -maxgap * cadence) & (tdiff > 0.5 * cadence):
        r_dif.append(data[i] - data[i - 1])
    else:
        r_dif.append(nandata)

dif_map = [sunpy.map.Map(r_dif[k], header[k]) for k in range(len(files))]

wc = []

for i in range(len(dif_map)):
    wc.append(dif_map[i].wcs)

w = np.shape(dif_map[0].data)[0]
h = np.shape(dif_map[0].data)[1]

p = [[y, x] for x in range(w) for y in range(h)]

coords = np.zeros((len(dif_map), len(p), 2))

# pixel coordinates are converted to world coordinates

for i in range(len(dif_map)):
    coords[i, :, :] = wc[i].all_pix2world(p, 0)  # ??????? NOT SURE 0/1 (0 at first)

# convert to radians

coords = coords * np.pi / 180

hpclat = coords[:, :, 1]
hpclon = coords[:, :, 0]

elongs = np.arccos(np.cos(hpclat) * np.cos(hpclon))

pas = np.sin(hpclat) / np.sin(elongs)
elongs = elongs * 180 / np.pi

if ftpsc == 'A':
    pas = np.arccos(pas) * 180 / np.pi

if ftpsc == 'B':
    pas = (2 * np.pi - np.arccos(pas)) * 180 / np.pi

paearth = get_earth_pos(time, ftpsc)

instrument = [instrument]

for ins in instrument:

    if ins == 'hi_1':
        noelongs = 360

    if ins == 'hi_2':
        noelongs = 180

    elongint = 90.0 / noelongs
    elongst = np.arange(noelongs) * elongint
    elongen = elongst + elongint

    tmpdata = np.arange(noelongs) * np.nan
    patol = 3.

    result = [np.where((pas[i] >= paearth[i] - patol) & (pas[i] <= paearth[i] + patol)) for i in range(len(pas))]

    tmpelongs = [elongs[i][result[i][:]] for i in range(len(elongs))]
    re_dif = np.reshape(r_dif, (len(r_dif), 256 * 256))
    tmpdiff = [re_dif[i][result[i][:]] for i in range(len(re_dif))]

    result2 = [[np.where((tmpelongs[i] >= elongst[j] - elongint) & (tmpelongs[i] < elongen[j] + elongint)
                         & np.isfinite(tmpdiff[i])) for j in range(noelongs)] for i in range(len(tmpelongs))]

    result2 = np.array(result2)

    t_data = np.zeros((len(tmpdiff), noelongs))

    for i in range(len(tmpdiff)):
        t_data[i][:] = bin_elong(noelongs, result2[i], tmpdiff[i], tmpdata)

    zrange = [-9000, 9000]

    img = np.empty((len(t_data), noelongs,))
    img[:] = np.nan

    for i in range(len(t_data)):
        img[i][:] = (t_data[i] - zrange[0]) / (zrange[1] - zrange[0])

    nn = equalize_adapthist(img, kernel_size=len(t_data) / 2)
    # nn = np.nan_to_num(img)

    nimg = np.where(nn >= 0, nn, 0)
    nimg = np.where(nimg <= 1, nimg, 0)
    nimg = np.where(nimg == 0, np.nan, nimg)

    y1 = elongst
    y2 = elongen
    den = ndimage.uniform_filter(nimg)

    nanp = np.where(np.isnan(nimg))
    den[nanp] = np.nan

    imsh_im = nimg.T
    den_im = den.T

    time_t = [datetime.strptime(day, '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y/%m/%d %H:%M:%S.%f') for day in time]

    x_lims = mdates.datestr2num(time_t)

    dt = (np.max(x_lims) - np.min(x_lims)) / (imsh_im.shape[1] - 1)
    delon = (np.max(y2) - np.min(y1)) / (noelongs - 1)

    fig, ax = plt.subplots()

    ax.imshow(imsh_im, origin='lower', cmap='gray',
              extent=[x_lims[0]-dt/2, x_lims[-1]+dt/2, y1[0]-delon/2, y2[-1]+delon/2], aspect='auto')

    plt.gca().xaxis.set_major_locator(HourLocator(byhour=range(0, 24, 24)))
    plt.gca().xaxis.set_major_formatter(DateFormatter('%d/%m/%y'))
    ax.xaxis_date()
    plt.savefig(red_path+start+'_'+ins+'_jplot.png')
