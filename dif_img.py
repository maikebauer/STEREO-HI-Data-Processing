import matplotlib.pyplot as plt
import matplotlib.dates as d
from functions import rej_out, get_earth_pos, bin_elong, hi_img
import datetime
import numpy as np
import pandas as pd
from skimage.exposure import equalize_adapthist, equalize_hist, adjust_gamma, rescale_intensity, histogram
from pandas.plotting import register_matplotlib_converters
import pickle
register_matplotlib_converters()

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

  img1, img2, e1, e2, time_h1, time_h2, tflag, img_comb = hi_img(start, ftpsc, bflag, path, calpath, high_contr)

  if tflag:
    orig = 'lower'

  if not tflag:
    orig = 'upper'

  fig, ax = plt.subplots(figsize=(15, 7), frameon=False)

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

  dt2 = (time_h2[-1]-time_h2[0])/len(time_h2)
  time_h2 = time_h2 + dt2

  ax.imshow(img1_center, cmap='gray', extent=[time_h1[0], time_h1[-1], e1[0], e1[-1]], aspect='auto', vmin = vmin1, vmax = vmax1, origin = orig)

  ax.imshow(img2_center, cmap='gray', extent=[time_h2[0], time_h2[-1], e2[0], e2[-1]], aspect='auto', vmin = vmin2, vmax = vmax2, origin = orig)


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

  fig, ax = plt.subplots(figsize=(15, 7), frameon=False)

  ax.imshow(img1_center, cmap='gray', extent=[time_h1[0], time_h1[-1], e1[0], e1[-1]], aspect='auto', vmin = -1, vmax = 1, origin = orig, interpolate='nearest')

  ax.imshow(img2_center, cmap='gray', extent=[time_h2[0], time_h2[-1], e2[0], e2[-1]], aspect='auto', vmin = -1, vmax = 1, origin = orig,interpolate='nearest')

fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.axis('off')
plt.ylim(3, e2[-1])
plt.savefig(savepath+'/'+start+'_jplot.png', bbox_inches = 0, pad_inches=0)

plt.close()

with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([time_h2[0], time_h2[-1], e1[0], e2[-1]], f)
