from time import time as timer

start_t = timer()

import sunpy.map
import sunpy.io
import matplotlib.pyplot as plt
import numpy as np
from functions import scc_sebip
from dwnld_beacon import fitsfil
from dwnld_beacon import datelist_int
from dwnld_beacon import SC
from dwnld_beacon import instrument
from dwnld_beacon import bflag
from astropy.io import fits
from functions import hi_remove_saturation
from functions import hi_delsq
from functions import hi_index_peakpix
from functions import hi_sor2
from functions import hi_desmear
from functions import get_calimg
from functions import get_biasmean
from functions import scc_img_trim
import math
import datetime
from functions import read_sav
import os
import numba

file = open('config.txt', 'r')
path = file.readlines()
path = path[0].splitlines()[0]

savepath = path + '/' + datelist_int[0] + '_' + SC + '_' + bflag + '_red'

if bflag == 'science':
    fits_hi1 = [s for s in fitsfil if "s4h1" in s]
    fits_hi2 = [s for s in fitsfil if "s4h2" in s]

if bflag == 'beacon':
    fits_hi1 = [s for s in fitsfil if "s7h1" in s]
    fits_hi2 = [s for s in fitsfil if "s7h2" in s]

fitslist = [fits_hi1, fits_hi2]

def new_nanmin(arr, axis):
    """New function for finding minimum of stack of images along an axis.
    Works the same as np.nanmin (ignores nans) but returns nan if all-nan slice is ecountered."""
    try:
        return np.nanmin(arr, axis)
    except ValueError:
        return np.nan


f = 0

for fitsfiles in fitslist:
    if len(fitsfiles) > 0:

        ins = instrument[f]
        # correct for on-board sebip modifications to image (division by 2, 4, 16 etc.)
        # calls function scc_sebip from functions.py
        header = [fits.getheader(fitsfiles[i]) for i in range(len(fitsfiles))]

        data_trim = [scc_img_trim(fitsfiles[i]) for i in range(len(fitsfiles))]

        data_sebip = [scc_sebip(data_trim[i], header[i]) for i in range(len(header))]

        print('Creating maps...')

        # maps are created from corrected data
        # header is saved into separate list

        for i in range(len(header)):
            biasmean = get_biasmean(header[i])

            if biasmean != 0:
                header[i]['OFFSETCR'] = biasmean

        for i in range(len(header)):
            data_sebip[i] = data_sebip[i] - biasmean

        fits_map = [sunpy.map.Map(data_sebip[i], header[i]) for i in range(len(header))]

        mapcube = np.empty((header[0]['NAXIS1'], header[0]['NAXIS2'], len(fits_map)))

        for i in range(len(fits_map)):
            mapcube[:, :, i] = fits_map[i].data

        length = np.shape(mapcube)

        # elements where mapcube is 0 must be set to be nan to correctly identify minimum of array

        mapcube = np.where(mapcube == 0, np.nan, mapcube)

        # new function for finding minimum of stack of images along an axis
        # works the same as np.nanmin (ignores nans) but returns nan if all-nan slice is ecountered

        print('Calculating background...')

        # minimum of array along time-axis found

        min_arr_reg = new_nanmin(mapcube, axis=2)

        # array containing minima is subtracted from data

        regnewcube = mapcube - min_arr_reg[:, :, None]

        # nans are converted back to 0

        regnewcube = np.nan_to_num(regnewcube)

        print('Removing saturated pixels...')

        # saturated pixels are removed
        # calls function hi_remove_saturation from functions.py

        ind = []
        divim = []
        desatcube = np.zeros((header[0]['NAXIS1'], header[0]['NAXIS2'], length[2]))

        # rec = np.zeros((header[0]['NAXIS1'], header[0]['NAXIS2'], length[2]))

        for i in range(length[2]):
            desatcube[:, :, i] = hi_remove_saturation(regnewcube[:, :, i], header[i])

        # rec = np.zeros((header[0]['NAXIS1'], header[0]['NAXIS2'], length[2]))

        # print('Removing stars...')

        # stars are removed
        # functions hi_delsq, hi_index_peakpix and hi_sor2 from functions.py are called

        # for i in range(length[2]):
        #    thresh = np.nanmax(desatcube[:, :, i]) * 0.1

        #    divim.append(hi_delsq(desatcube[:, :, i]))
        #    ind.append(hi_index_peakpix(divim[i], thresh))

        #    print('Correcting {} pixels in image {}'.format(len(ind[i]), i))

        #    for j in range(len(ind[i])):
        #        divim[i][ind[i][j][0]][ind[i][j][1]] = 0.0

        #    rec[:, :, i] = hi_sor2(desatcube[:, :, i], divim[i], np.array(ind[i]))

        # after data reduction is done, images are made into map

        print('Desmearing image...')

        des = [hi_desmear(desatcube[:, :, i], header[i]) for i in range(length[2])]
        des = np.array(des)

        print('Calibrating image...')

        cal = [get_calimg(ins, SC, path, header[k]) for k in range(length[2])]
        cal = np.array(cal)

        rec = cal * des
        rec_map = [sunpy.map.Map(rec[k, :, :], header[k]) for k in range(length[2])]

        names1 = []
        names2 = []
        newname = []

        print('Saving .fits files...')

        for i in range(length[2]):
            names1.append(fitsfiles[i].rpartition('/')[2])
            names2.append(names1[i].rpartition('.')[0])
            newname.append(names2[i] + '_red.fts')

            if not os.path.exists(savepath):
                os.mkdir(savepath)
                fits.writeto(savepath + '/' + newname[i], rec_map[i].data, header[i], output_verify='silentfix')

            if os.path.exists(savepath + '/' + newname[i]):
                os.remove(savepath + '/' + newname[i])
                fits.writeto(savepath + '/' + newname[i], rec_map[i].data, header[i], output_verify='silentfix')

            if not os.path.exists(savepath + '/' + newname[i]):
                fits.writeto(savepath + '/' + newname[i], rec_map[i].data, header[i], output_verify='silentfix')

        f = f + 1

    print(f"Elapsed Time: {timer() - start_t}")
