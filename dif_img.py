import matplotlib.pyplot as plt
import matplotlib.dates as d
from functions import rej_out, get_earth_pos, bin_elong, hi_img
import datetime
import numpy as np
import pandas as pd
from skimage.exposure import equalize_adapthist, equalize_hist, adjust_gamma, rescale_intensity, histogram
#bflag = str(input('Choose science or beacon data (science/beacon):'))
#ftpsc = str(input('Enter the spacecraft (A/B):'))
#start = str(input('Enter the start date (YYYYMMDD/today):'))

file = open('config.txt', 'r')
config = file.readlines()
path = config[0].splitlines()[0]
calpath = config[1].splitlines()[0]
start = config[2].splitlines()[0]
ftpsc = config[3].splitlines()[0]
instrument = config[4].splitlines()[0]
bflag = config[5].splitlines()[0]
high_contr = config[6].splitlines()[0]

# high_contr decides if histogram equalization is performed or not

if high_contr == 'high contrast':
  high_contr = True

else:
  high_contr = False


if start == 'today':
    start = datetime.datetime.today() - datetime.timedelta(days=7)
    start = start.strftime('%Y%m%d')

savepath = path + start + '_' + ftpsc + '_' + bflag + '_red'

if bflag == 'science':

  img1, img2, e1, e2, time1, time2, tflag = hi_img(start, ftpsc, bflag, path, calpath, high_contr)

  if tflag:
    orig = 'lower'

  if not tflag:
    orig = 'upper'

  fig, ax = plt.subplots(figsize=(15, 7))

  #img_1 = ndimage.gaussian_filter(img1, sigma=(1, 1), order=0)
  #img_2 = ndimage.gaussian_filter(r2, sigma=(1, 1), order=0)

  # images are plotted via imshow, one above the other
  # extent keyword sepcifies beginning and end values of x- and y-axis for h1 and h2 image

  if high_contr:

    vmin1 = np.nanmin(img1)
    vmax1 = np.nanmax(img1)
    vmin2 = np.nanmin(img2)
    vmax2 = np.nanmax(img2)

    img1_center = img1
    img2_center = img2

    plt.hist(img1_center, bins=100)
    plt.savefig(savepath+'/'+start+'_hist.png')
  # specific boundaries must be chosen for images on which no histogram equalization was performed
  # since they are not normalized to [0, 1]

  if not high_contr:

    vmin1 = -1
    vmax1 = 1
    vmin2 = -1
    vmax2 = 1

    #img1 = rej_out(img1, 4)
    #img2 = rej_out(img2, 4)

    #img1_norm = 2*(img1_center - np.nanmin(img1_center)) / (np.nanmax(img1_center) - np.nanmin(img1_center)) - 1
    #img2_norm = 2*(img2_center - np.nanmin(img2_center)) / (np.nanmax(img2_center) - np.nanmin(img2_center)) - 1

    #img1_eqadapt = equalize_adapthist(img1_norm)
    #img2_eqadapt = equalize_adapthist(img2_norm)
    img1_center = img1
    img2_center = img2

    plt.hist(img1_center, bins=100)

    plt.savefig(savepath+'/'+start+'_hist.png')
  print(d.num2date(time1[0]))
  print(d.num2date(time2[0]))

  print(d.num2date(time1[1]))
  print(d.num2date(time2[1]))

  print(d.num2date(time1[-1]))
  print(d.num2date(time2[-1]))

  print(np.shape(img1_center))
  print(np.shape(img2_center))
  dt2 = (time2[-1]-time2[0])/len(time2)

  ax.imshow(img1_center, cmap='gray', extent=[time1[0], time1[-1], e1[0], e1[-1]], aspect='auto', vmin = vmin1, vmax = vmax1, origin = orig)

  ax.imshow(img2_center, cmap='gray', extent=[time2[0]+dt2, time2[-1]+dt2, e2[0], e2[-1]], aspect='auto', vmin = vmin2, vmax = vmax2, origin = orig)

  plt.ylim(3, 50)

if bflag == 'beacon':

  img1, img2, e1, e2, time_h1, time_h2, tflag = hi_img(start, ftpsc, bflag, path, calpath, high_contr)

  if tflag:
    orig = 'lower'

  if not tflag:
    orig = 'upper'

  #img_1 = ndimage.gaussian_filter(img1, sigma=(1, 1), order=0)
  #img_2 = ndimage.gaussian_filter(r2, sigma=(1, 1), order=0)

  # images are plotted via imshow, one above the other
  # extent keyword sepcifies beginning and end values of x- and y-axis for h1 and h2 image

  img1_center = (img1 - np.nanmean(img1)) / np.std(img1)
  img2_center = (img2 - np.nanmean(img2)) / np.std(img2)

  #img1_norm = 2*(img1_center - np.nanmin(img1_center)) / (np.nanmax(img1_center) - np.nanmin(img1_center)) - 1
  #img2_norm = 2*(img2_center - np.nanmin(img2_center)) / (np.nanmax(img2_center) - np.nanmin(img2_center)) - 1

  #img1_eqadapt = equalize_adapthist(img1_norm)
  #img2_eqadapt = equalize_adapthist(img2_norm)

  plt.hist(img1_center, bins=100)

  plt.savefig(savepath+'/'+start+'_hist.png')

  fig, ax = plt.subplots(figsize=(15, 7))

  ax.imshow(img1_center, cmap='gray', extent=[time_h1[0], time_h1[-1], e1[0], e1[-1]], aspect='auto', vmin = -1, vmax = 1, origin = orig)

  ax.imshow(img2_center, cmap='gray', extent=[time_h2[0], time_h2[-1], e2[0], e2[-1]], aspect='auto', vmin = -1, vmax = 1, origin = orig)

  plt.ylim(5, 50)

plt.gca().xaxis.set_major_locator(d.HourLocator(byhour=range(0, 24, 24)))
plt.gca().xaxis.set_major_formatter(d.DateFormatter('%d/%m/%y'))
ax.xaxis_date()

plt.xlabel('Date (d/m/y)')
plt.ylabel('Elongation (°)')

#data = []
#inp = plt.ginput(0, 0)
#data.append(inp)

plt.savefig(savepath+'/'+start+'.png')
plt.close()

# uncomment all lines below to enable tracking in JPlots

#elon = [d.num2date(data[0][i][1]) for i in range(len(data[0]))]
#date_time_obj = [d.num2date(data[0][i][0]) for i in range(len(data[0]))]
#date = [datetime.datetime.strftime(x, '%Y-%b-%d %H:%M:%S.%f') for x in date_time_obj]

#elon_stdd = np.zeros(len(data[0]))
#SC = [ftpsc for x in range(len(data[0]))]

#pd_data = {'TRACK_DATE': date, 'ELON': elon, 'ELON_STDD': elon_stdd, 'SC': SC}

#df = pd.DataFrame(pd_data, columns=['TRACK_DATE', 'ELON', 'SC', 'ELON_STDD'])
#df.to_csv(path + start + '_' + ftpsc + '_' + bflag + '_red/' + 'track.csv', index=False, date_format='%Y-%m-%dT%H:M:S')
