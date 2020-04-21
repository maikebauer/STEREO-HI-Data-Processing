from astropy.io import fits
import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.io import readsav
import datetime
import os
import numba
from sunpy.coordinates.ephemeris import get_earth, get_horizons_coord
from sunpy.coordinates import frames
from astropy.time import Time
import glob
import sunpy.io
import sunpy.map
from skimage.exposure import equalize_adapthist, equalize_hist, adjust_gamma, rescale_intensity, histogram
from scipy import ndimage
import matplotlib.dates as mdates
import math
from astropy import wcs
from time import time as timer
from astropy.coordinates import SkyCoord
from astropy import units as u
import requests
from bs4 import BeautifulSoup
from multiprocessing.pool import ThreadPool
import os
import wget
import pandas as pd
import datetime
from itertools import repeat
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import pickle
#######################################################################################################################################

# listfd makes makes list of urls and corresponding file names to download

def listfd(input_url, extension):
    output_urls = []
    page = requests.get(input_url).text

    soup = BeautifulSoup(page, 'html.parser')
    url_found = [input_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]
    filename = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]

    for i in range(len(filename)):
        output_urls.append((filename[i], url_found[i]))

    return output_urls

#######################################################################################################################################

# mk_dir creteas directory for downloades files

def mk_dir(path, d):

    if not os.path.exists(path):
        try:
            os.mkdir(path)

        except OSError:
            print('Creation of the directory %s failed' % path)
        else:
            print('Directory successfuly created!\n')

    if glob.glob(path + d + '*.fts'):
        flg = False

    if not glob.glob(path + d + '*.fts'):
        flg = True

    return flg

#######################################################################################################################################

# fetch_url downloads the urls specified by listfd

def fetch_url(path, entry):
    filename, uri = entry

    if not os.path.exists(path + filename):
        wget.download(uri, path)

    return path + filename


#start_t = timer()

#######################################################################################################################################

# downloads STEREO beacon images from NASA server

def download_files(line):

    fitsfil = []

    file = open('config.txt', 'r')
    config = file.readlines()
    path = config[0].splitlines()[0]
    ftpsc = config[2].splitlines()[0]
    instrument = config[3].splitlines()[0]
    bflag = config[4].splitlines()[0]
    start = config[6+line].splitlines()[0]

    if ftpsc == 'A':
        sc = 'ahead'

    if ftpsc == 'B':
        sc = 'behind'

    if instrument == 'both':
        instrument = ['hi_1', 'hi_2']

    elif instrument == 'hi_1':
        instrument = ['hi_1']

    elif instrument == 'hi_2':
        instrument = ['hi_2']

    else:
        instrument = ['hi_1', 'hi_2']
        print('Invalid instrument specification, downloading HI-1 and 2 data.')

    if start == 'today':
        start = datetime.datetime.today() - datetime.timedelta(days=7)
        start = start.strftime('%Y%m%d')

    date = datetime.datetime.strptime(start, '%Y%m%d')

    datelist = pd.date_range(date, periods=8).tolist()

    datelist_int = [str(datelist[i].year)+datelist[i].strftime('%m')+datelist[i].strftime('%d') for i in range(len(datelist))]

    print('Fetching files...')

    for ins in instrument:
        for date in datelist_int:

            if bflag == 'beacon':
                url = 'https://stereo-ssc.nascom.nasa.gov/pub/beacon/' + sc + '/secchi/img/' + ins + '/' + str(date)

            else:
                url = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/' + sc[0] + '/img/' + ins + '/' + str(date)

            ext = 'fts'
            file = open('config.txt', 'r')
            path_dir = file.readlines()
            path_dir = path_dir[0].splitlines()[0]
            path = path_dir + str(datelist_int[0]) + '_' + sc + '_' + ins + '_' + bflag + '/'

            flag = mk_dir(path, date)
            if flag:
                urls = listfd(url, ext)
                print(date)
                fitsfil.extend(ThreadPool(8).starmap(fetch_url, zip(repeat(path), urls), chunksize=1))
            if not flag:
                print('Files have already been downloaded')
                for file in glob.glob(path+date+'*.fts'):
                    fitsfil.append(file)


    print('Done!')

    return fitsfil, datelist_int, ftpsc, instrument, bflag
#######################################################################################################################################

def new_nanmin(arr, axis):
    """New function for finding minimum of stack of images along an axis.
    Works the same as np.nanmin (ignores nans) but returns nan if all-nan slice is ecountered."""
    try:
        return np.nanmin(arr, axis)
    except ValueError:
        return np.nan

#######################################################################################################################################

# reads IDL .sav files and returns date in format for interpoaltion (date_sec) and plotting (dates) as well as elongation


def read_sav(filepath):
    """Reads IDL .sav files and returns date in format for interpoaltion (date_sec) and plotting (dates) as well as elongation."""

    sav_file = readsav(filepath)

    track = sav_file['track']

    date_time = track['track_date']
    elon = track['elon']
    # elon = track['track_y']

    date_time_dec = []

    for i in range(len(date_time[0])):
        date_time_dec.append(date_time[0][i].decode('utf-8'))

    dates = [datetime.datetime.strptime(date_time_dec[i], '%d-%b-%Y %H:%M:%S.%f') for i in range(len(date_time[0]))]
    # dates = [datetime.datetime.strptime(date_time_dec[i], '%Y-%m-%dT%H:%M:%S%f') for i in range(len(date_time[0]))]
    date_time_obj = []

    for i in range(len(date_time[0])):
        date_time_obj.append(datetime.datetime.strptime(date_time_dec[i], '%d-%b-%Y %H:%M:%S.%f') - datetime.datetime(1970, 1, 1))
        # date_time_obj.append(datetime.datetime.strptime(date_time_dec[i], '%Y-%m-%dT%H:%M:%S%f') - datetime.datetime(1970, 1, 1))

    date_sec = [date_time_obj[i].total_seconds() for i in range(len(date_time_obj))]

    return date_sec, elon, dates


#######################################################################################################################################


@numba.njit
def hi_delsq(dat):
    """Direct conversion of hi_delsq.pro for IDL.
    Returns the delta^2 field of an image which is later used for computing the starfield."""

    least = 4

    nx = dat.shape[0]
    dhj = dat.copy()
    ctr = dat.copy()

    # forward addressing

    d0 = dat[0:nx - 1, :] - dat[1:nx, :]
    bmap = np.isfinite(d0)
    d0_new = np.where(np.isfinite(d0), d0, 0)

    dhj[0:nx - 1, :] = dhj[0:nx - 1, :] + d0_new
    ctr[0:nx - 1, :] = ctr[0:nx - 1, :] + bmap

    # backward addressing

    d0 = dat[1:nx, :] - dat[0:nx - 1, :]

    bmap = np.isfinite(d0)
    d0_new = np.where(np.isfinite(d0), d0, 0)

    dhj[1:nx, :] = dhj[1:nx, :] + d0_new
    ctr[1:nx, :] = ctr[1:nx, :] + bmap

    # downward addressing

    d0 = dat[:, 0:nx - 1] - dat[:, 1:nx]
    bmap = np.isfinite(d0)
    d0_new = np.where(np.isfinite(d0), d0, 0)

    dhj[:, 0:nx - 1] = dhj[:, 0:nx - 1] + d0_new
    ctr[:, 0:nx - 1] = ctr[:, 0:nx - 1] + bmap

    # upward addressing

    d0 = dat[:, 1:nx] - dat[:, 0:nx - 1]

    bmap = np.isfinite(d0)
    d0_new = np.where(np.isfinite(d0), d0, 0)

    dhj[:, 1:nx] = dhj[:, 1:nx] + d0_new
    ctr[:, 1:nx] = ctr[:, 1:nx] + bmap

    dhj = np.where(ctr < least, np.nan, dhj)

    # return field
    return dhj


#######################################################################################################################################


@numba.njit
def hi_index_peakpix(ddat, thresh):
    """Direct conversion of hi_index_peakpix.pro for IDL.
    Returns indices of pixels and their neighbours whose absolute values are above a given threshold."""

    ny = ddat.shape[1]
    nx = ddat.shape[0]

    ind = []

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):

            if abs(ddat[i, j]) > thresh:

                ind.append(np.array([i, j]))

            else:

                tst1 = abs(ddat[i, j + 1]) > thresh

                tst2 = abs(ddat[i, j - 1]) > thresh

                tst3 = abs(ddat[i + 1, j]) > thresh

                tst4 = abs(ddat[i - 1, j]) > thresh

                if tst1 or tst2 or tst3 or tst4:
                    ind.append(np.array([i, j]))
    return ind


#######################################################################################################################################

# direct conversion of hi_sor2.pro for IDL

@numba.njit
def hi_sor2(dat, ddat, indices):
    """Direct conversion of hi_sor2.pro for IDL. Solves delta^2 = f(x,y) using a sucessive over-relaxation method.
    Reconstructs pixels identified by hi_index_peakpix.
    Takes image data (dat), delta^2 field (ddat) and indices returned by hi_index_peakpix as input."""

    # w = overrelaxation parameter
    # nits = number of iterations

    w = 1.5
    nits = 100
    npts = len(indices)
    recon = dat.copy()

    for n in range(1, nits + 1):
        for k in range(npts):
            dum = 0.25 * (recon[indices[k][0], indices[k][1] - 1] + recon[indices[k][0] - 1, indices[k][1]] +
                          recon[indices[k][0] + 1, indices[k][1]] + recon[indices[k][0], indices[k][1] + 1] -
                          ddat[indices[k][0], indices[k][1]])

            dum2 = w * dum + (1 - w) * recon[indices[k][0], indices[k][1]]
            recon[indices[k][0], indices[k][1]] = dum2

    return recon


#######################################################################################################################################


def hi_remove_saturation(data, header):
    """Direct conversion of hi_remove_saturation.pro for IDL.
    Detects and masks saturated pixels with nan. Takes image data and header as input. Returns fixed image."""

    # threshold value before pixel is considered saturated
    sat_lim = 5000
    # number of pixels in a column before column is considered saturated
    nsaturated = 6

    n_im = header['imgseq'] + 1
    imsum = header['summed']

    dsatval = sat_lim * n_im * (2 ** (imsum - 1)) ** 2

    ind = np.where(data > dsatval)

    # if a pixel has a value greater than the dsatval, begin check to test if column is saturated

    if ind[0].size > 0:

        # all pixels are set to zero, except ones exceeding the saturation limit

        mask = data * 0
        ans = data.copy()
        mask[ind] = 1

        # pixels are summed up column-wise
        # where nsaturated is exceeded, values are replaced by nan

        colmask = np.sum(mask, 0)
        ii = np.where(colmask > nsaturated)

        if ii[0].size > 0:
            ans[:, ii] = np.nan

        else:
            ans = data.copy()

    else:
        ans = data.copy()

    # interpolate_replace_nans takes care of replacing saturated columns (now full of nan)

    # kernel = Gaussian2DKernel(x_stddev=1, x_size=31., y_size=31.)
    # fixed_image = np.nan_to_num(ans)

    # the fixed image with saturated columns replaced by interpolated ones is returend

    return ans


#######################################################################################################################################


def scc_sebip(data, header):
    """Direct conversion of scc_sebip.pro for IDL.
    Determines what has happened in terms of on-board sebip binning and corrects it.
    Takes image data and header as input and returns fixed image."""

    ip_raw = header['IP_00_19']

    while len(ip_raw) < 60:
        ip_raw = ' ' + ip_raw

    ip_bytes = bytearray(ip_raw, encoding='ascii')
    ip_arr = np.array(ip_bytes)
    ip_reform = ip_arr.reshape(-1, 3).transpose()

    ip_temp = []
    ip = []

    for i in range(ip_reform.shape[1]):
        ip_temp.append(ip_reform[:, i].tostring())

    for i in range(len(ip_temp)):
        ip.append(ip_temp[i].decode('ascii'))

    cnt = ip.count('117')

    if cnt == 1:
        print('cnt = 1')
        ind = np.where(ip == '117')
        ip = ip[3 * ind:]
        while len(ip) < 60:
            ip = ip.append('  0')

    cnt1 = ip.count('  1')
    cnt2 = ip.count('  2')
    cntspw = ip.count(' 16')

    if cntspw == 0:
        cntspw = ip.count(' 17')

    cnt50 = ip.count(' 50')
    cnt53 = ip.count(' 53')
    cnt82 = ip.count(' 82')
    cnt83 = ip.count(' 83')
    cnt84 = ip.count(' 84')
    cnt85 = ip.count(' 85')
    cnt86 = ip.count(' 86')
    cnt87 = ip.count(' 87')
    cnt88 = ip.count(' 88')
    cnt118 = ip.count('118')

    if header['DIV2CORR']:
        cnt1 = cnt1 - 1

    if cnt1 < 0:
        cnt1 = 0

    if cnt1 > 0:
        data = data * (2.0 ** cnt1)
        print('Correted for divide by 2 x {}'.format(cnt1))
    if cnt2 > 0:
        data = data * (2.0 ** cnt2)
        print('Correted for square root x {}'.format(cnt2))
    if cntspw > 0:
        data = data * (64.0 ** cntspw)
        print('Correted for HI SPW divide by 64 x {}'.format(cntspw))
    if cnt50 > 0:
        data = data * (4.0 ** cnt50)
        print('Correted for divide by 4 x {}'.format(cnt50))
    if cnt53 > 0 and header['ipsum'] > 0:
        data = data * (4.0 ** cnt53)
        print('Correted for divide by 4 x {}'.format(cnt53))
    if cnt82 > 0:
        data = data * (2.0 ** cnt82)
        print('Correted for divide by 2 x {}'.format(cnt82))
    if cnt83 > 0:
        data = data * (4.0 ** cnt83)
        print('Correted for divide by 4 x {}'.format(cnt83))
    if cnt84 > 0:
        data = data * (8.0 ** cnt84)
        print('Correted for divide by 8 x {}'.format(cnt84))
    if cnt85 > 0:
        data = data * (16.0 ** cnt85)
        print('Correted for divide by 16 x {}'.format(cnt85))
    if cnt86 > 0:
        data = data * (32.0 ** cnt86)
        print('Correted for divide by 32 x {}'.format(cnt86))
    if cnt87 > 0:
        data = data * (64.0 ** cnt87)
        print('Correted for divide by 64 x {}'.format(cnt87))
    if cnt88 > 0:
        data = data * (128.0 ** cnt88)
        print('Correted for divide by 128 x {}'.format(cnt88))
    if cnt118 > 0:
        data = data * (3.0 ** cnt118)
        print('Correted for divide by 3 x {}'.format(cnt118))

    print('------------------------------------------------------')

    return data


#######################################################################################################################################


def rej_out(dat_arr, m):
    """Removes values from an array that deviate from the mean by more than m standard deviations."""
    n_arr = np.where(abs(dat_arr - np.mean(dat_arr)) < m * np.std(dat_arr), dat_arr, np.nanmean(dat_arr))
    return n_arr


#######################################################################################################################################


#######################################################################################################################################


def get_earth_pos(time, ftpsc):
    earth = [get_earth(t) for t in time]
    sta = [get_horizons_coord('STEREO-A', t) for t in time]
    # This is what doesnt work if you ever want to try STB
    pos_earth = [earth[i].transform_to(frames.Helioprojective(observer=sta[i])) for i in range(len(earth))]

    earth_lon = [pos_earth[i].Tx.radian for i in range(len(pos_earth))]
    earth_lat = [pos_earth[i].Ty.radian for i in range(len(pos_earth))]

    elongearth = np.arccos(np.cos(earth_lat) * np.cos(earth_lon)) * 180 / np.pi
    paearth = np.sin(earth_lat) / np.sin(elongearth)

    if ftpsc == 'A':
        paearth = np.arccos(paearth) * 180 / np.pi

    if ftpsc == 'B':
        paearth = (2 * np.pi - np.arccos(paearth)) * 180 / np.pi

    return paearth


#######################################################################################################################################


#######################################################################################################################################


def get_smask(ftpsc, header, path, time, calpath):

    if ftpsc == 'A':
        filename = 'hi2A_mask.fts'
        xy = [1, 51]

    if ftpsc == 'B':
        filename = 'hi2B_mask.fts'
        xy = [129, 79]

    calpath = path + 'calibration/' + filename
    time = datetime.datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')

    smask, hdr = fits.getdata(calpath, header=True)

    fullm = np.zeros((2176, 2176))

    x1 = 2048 - np.shape(fullm[xy[1] - 1:, xy[0] - 1:])[0]
    y1 = 2048 - np.shape(fullm[xy[1] - 1:, xy[0] - 1:])[1]

    fullm[xy[0] - 1:y1, xy[1] - 1:x1] = smask

    tc = datetime.datetime(2015, 5, 19)

    flag = time > tc

    if flag:
        fullm = np.rot90(fullm)
        fullm = np.rot90(fullm)

    mask = rebin(fullm[header['r1row'] - 1:header['r2row'], header['r1col'] - 1:header['r2col']],
                 [header['NAXIS2'], header['NAXIS1']])
    mask = np.where(mask < 1, 0, 1)

    return mask


#######################################################################################################################################


def rebin(a, shape):
    sh = int(shape[0]), int(a.shape[0]) // int(shape[0]), int(shape[1]), int(a.shape[1]) // int(shape[1])
    return a.reshape(sh).mean(-1).mean(1)


#######################################################################################################################################

@numba.njit()
def hi_desmear(data, header_int, header_flt, header_str):
    dstart1, dstart2, dstop1, dstop2, naxis1, naxis2, n_images, post_conj = header_int

    exptime, cleartim, ro_delay, ipsum, line_ro, line_clr = header_flt

    rectify, obsrvtry = header_str

    if dstart1 < int(1) or naxis1 == naxis2:
        image = data

    else:
        image = data[dstart2 - 1:dstop1, dstart1 - 1:dstop1]

    clearest = 0.70
    exp_eff = exptime + float(n_images) * (clearest - cleartim + ro_delay)

    dataweight = float(n_images) * (2. ** ipsum - 1.)

    inverted = 0.0

    if rectify:
        if not obsrvtry:
            inverted = 1
        if obsrvtry:
            if post_conj:
                inverted = 1

    if inverted == 1:

        n = naxis2

        ab = np.zeros((n, n))
        ab[:] = dataweight * line_ro
        bel = np.zeros((n, n))
        bel[:] = dataweight * line_clr

        fixup = np.triu(ab) + np.tril(bel)

        for i in range(0, n):
            fixup[i, i] = exp_eff

    else:

        n = naxis2

        ab = np.zeros((n, n))
        ab[:] = dataweight * line_clr
        bel = np.zeros((n, n))
        bel[:] = dataweight * line_ro

        fixup = np.triu(ab) + np.tril(bel)

        for i in range(0, n):
            fixup[i, i] = exp_eff

    fixup = np.linalg.inv(fixup)
    image = fixup @ image

    if dstart1 < 1 or (naxis1 == naxis2):
        img = image

    else:
        img = image[dstart2 - 1:dstop1, dstart1 - 1:dstop1]

    return img


#######################################################################################################################################


def get_calimg(instr, ftpsc, path, header):
    if instr == 'hi_1':

        if header['summed'] == 1:
            cal_version = '20061129_flatfld_raw_h1' + ftpsc.lower() + '.fts'

        else:
            cal_version = '20100421_flatfld_sum_h1' + ftpsc.lower() + '.fts'
            sumflg = 1

    if instr == 'hi_2':

        if header['summed'] == 1:
            cal_version = '20150701_flatfld_raw_h2' + ftpsc.lower() + '.fts'

        else:
            cal_version = '20150701_flatfld_sum_h2' + ftpsc.lower() + '.fts'
            sumflg = 1

    calpath = path + 'calibration/' + cal_version
    cal_image, cal_hdr = fits.getdata(calpath, header=True)
    try:
        cal_p1col = cal_hdr['P1COL']

    except KeyError:
        cal_p1col = 0

    if cal_p1col < 1:
        if sumflg:

            x1 = 25
            x2 = 1048
            y1 = 0
            y2 = 1023

        else:

            x1 = 50
            x2 = 2047 + 50
            y1 = 0
            y2 = 2047

        cal = cal_image[y1:y2 + 1, x1:x2 + 1]

    else:
        cal = cal_image

    time = cal_hdr['DATE']

    time = datetime.datetime.strptime(time, '%Y-%m-%d')

    if (header['RECTIFY'] == 'T') and (cal_hdr['RECTIFY'] == 'F'):
        cal = secchi_rectify(cal, cal_hdr, ftpsc, instr, flg)

    if sumflg:
        if header['summed'] < 2:
            hdr_sum = 1

        else:
            hdr_sum = 2 ** (header['summed'] - 2)

    else:
        hdr_sum = 2 ** (header['summed'] - 1)

    s = np.shape(cal)
    cal = rebin(cal, [s[1] / hdr_sum, s[0] / hdr_sum])

    tc = datetime.datetime(2015, 5, 19)

    flag_hdr = time > tc

    if flag_hdr:
        cal = np.rot90(cal)
        cal = np.rot90(cal)

    return cal


#######################################################################################################################################


def secchi_rectify(cal, scch, ftpsc, instr, flg):
    stch = scch

    stch['RECTIFY'] = 'T'

    if ftpsc == 'B':
        b = np.rot90(cal)
        b = np.rot90(b)
        stch['r1row'] = 2176 - scch['p2row'] + 1
        stch['r2row'] = 2176 - scch['p1row'] + 1
        stch['r1col'] = 2176 - scch['p2col'] + 1
        stch['r2col'] = 2176 - scch['p1col'] + 1

        stch['crpix1'] = scch['naxis1'] - scch['crpix1'] + 1
        stch['crpix2'] = scch['naxis2'] - scch['crpix2'] + 1
        stch['naxis1'] = scch['naxis1']
        stch['naxis2'] = scch['naxis2']
        stch['rectrota'] = 2
        rotcmt = 'rotate 180 deg CCW'

        stch['dstart1'] = (79 - stch['r1col'] + 1) > 1
        stch['dstop1'] = stch['dstart1'] - 1 + ((stch['r2col'] - stch['r1col'] + 1) < 2048)

        stch['dstart2'] = (129 - stch['r1row'] + 1) > 1
        stch['dstop2'] = stch['dstart2'] - 1 + ((stch['r2row'] - stch['r1row'] + 1) < 2048)

    if (ftpsc == 'A') and flag == 1:
        b = np.rot90(cal)
        b = np.rot90(b)
        stch['r1row'] = 2176 - scch['p2row'] + 1
        stch['r2row'] = 2176 - scch['p1row'] + 1
        stch['r1col'] = 2176 - scch['p2col'] + 1
        stch['r2col'] = 2176 - scch['p1col'] + 1

        stch['crpix1'] = scch['naxis1'] - scch['crpix1'] + 1
        stch['crpix2'] = scch['naxis2'] - scch['crpix2'] + 1
        stch['naxis1'] = scch['naxis1']
        stch['naxis2'] = scch['naxis2']
        stch['rectrota'] = 2
        rotcmt = 'rotate 180 deg CCW'

        stch['dstart1'] = (79 - stch['r1col'] + 1) > 1
        stch['dstop1'] = stch['dstart1'] - 1 + ((stch['r2col'] - stch['r1col'] + 1) < 2048)

        stch['dstart2'] = (129 - stch['r1row'] + 1) > 1
        stch['dstop2'] = stch['dstart2'] - 1 + ((stch['r2row'] - stch['r1row'] + 1) < 2048)

    if (ftpsc == 'A') and flag == 0:
        b = cal
        stch['r1row'] = scch['p1row']
        stch['r2row'] = scch['p2row']
        stch['r1col'] = scch['p1col']
        stch['r2col'] = scch['p2col']

        stch['rectrota'] = 0

    if stch['r1col'] < 1:
        stch['r2col'] = stch['r2col'] + np.abs(stch['r1col']) + 1
        stch['r1col'] = 1

    if stch['r1row'] < 1:
        stch['r2row'] = stch['r2row'] + np.abs(stch['r1row']) + 1
        stch['r1row'] = 1

    xden = 2 ** (scch['ipsum'] + scch['sumcol'] - 2)
    yden = 2 ** (scch['ipsum'] + scch['sumrow'] - 2)

    stch['dstart1'] = math.modf(math.ceil(stch['dstart1'] / xden))[1] > 1
    stch['dstart2'] = math.modf(math.ceil(stch['dstart2'] / yden))[1] > 1
    stch['dstop1'] = math.modf(stch['dstop1'] / xden)[1]
    stch['dstop2'] = math.modf(stch['dstop2'] / yden)[1]

    if stch['naxis1'] > 0 and stch['naxis2'] > 0:
        wcoord = wcs.WCS(stch)

        xycen = wcoord.wcs_pix2world([(stch['naxis1'] - 1.) / 2., (stch['naxis2'] - 1.) / 2.], 0)
        stch['xcen'] = xycen[0]
        stch['ycen'] = xycen[1]

    scch = stch

    return scch


#######################################################################################################################################

def get_biasmean(header):
    bias = header['BIASMEAN']
    ipsum = header['IPSUM']

    if ('103' in header['IP_00_19']) or (' 37' in header['IP_00_19']) or (' 38' in header['IP_00_19']):
        bias = 0

    elif header['OFFSETCR'] > 0:
        bias = 0

    else:

        bias = bias - (bias / header['n_images'])

        if ipsum > 1:
            bias = bias * ((2 ** (ipsum - 1)) ** 2)

    return bias


#######################################################################################################################################

def hi_fill_missing(data, header):
    if header['NMISSING'] == 0:
        data = data

    if header['NMISSING'] > 0:

        if len(header['MISSLIST']) < 1:
            print('Mismatch between nmissing and misslist.')
            data = data

        else:
            fields = scc_get_missing(header)
            data[fields] = np.nan

    return data


#######################################################################################################################################

def scc_get_missing(header):
    base = 34

    lenstr = len(header['MISSLIST'])

    if not lenstr % 2:
        misslisgt = ' ' + header['MISSLIST']
        lenstr = lenstr + 1

    else:
        misslist = header['MISSLIST']

    dex = np.arange(max(int(lenstr / 2), 1)) * 2


#######################################################################################################################################

def scc_img_trim(fit):
    im, header = fits.getdata(fit, header=True)

    x1 = header['DSTART1'] - 1
    x2 = header['DSTOP1'] - 1
    y1 = header['DSTART2'] - 1
    y2 = header['DSTOP2'] - 1

    img = im[y1:y2 + 1, x1:x2 + 1]

    s = np.shape(img)

    if (header['NAXIS1'] != s[0]) or (header['NAXIS2'] != s[1]):
        print('Removing under- and overscan...')

        hdrsum = 2 ** (header['SUMMED'] - 1)

        header['R1COL'] = header['R1COL'] + (x1 * hdrsum)
        header['R2COL'] = header['R1COL'] + (s[0] * hdrsum) - 1
        header['R1ROW'] = header['R1ROW'] + (y1 * hdrsum)
        header['R2ROW'] = header['R1ROW'] + (s[1] * hdrsum) - 1

        header['DSTART1'] = 1
        header['DSTOP1'] = s[0]
        header['DSTART2'] = 1
        header['DSTOP2'] = s[1]

        header['NAXIS1'] = s[0]
        header['NAXIS2'] = s[1]

        header['CRPIX1'] = header['CRPIX1'] - x1
        header['CRPIX1A'] = header['CRPIX1A'] - x1

        header['CRPIX2'] = header['CRPIX2'] - y1
        header['CRPIX2A'] = header['CRPIX2A'] - y1

        wcoord = wcs.WCS(header)
        xycen = wcoord.wcs_pix2world((header['naxis1'] - 1.) / 2., (header['naxis2'] - 1.) / 2., 0)
        header['xcen'] = np.round(xycen[0], 0)
        header['ycen'] = np.round(xycen[1], 0)

    return img


#######################################################################################################################################


@numba.njit()
def create_img(tdiff, maxgap, cadence, data):
    # nandata = np.zeros((np.shape(data[0])[0], np.shape(data[0])[1]), np.float64)
    # nandata[:] = np.nan
    nandata = np.full_like(data[0], np.nan)

    r_dif = []
    nan_ind = []
    nan_dat = []

    for i in range(1, len(data)):

        # produce regular running difference image if time difference is within maxgap * cadence

        if (tdiff[i] <= -maxgap * cadence) & (tdiff[i] > 0.5 * cadence):
            r_dif.append(data[i] - data[i - 1])

        # if not, insert a slice of nans to replace the missing timestep

        else:
            r_dif.append(nandata)
        # must make sure that one column of pixels is always represents equal number of pixels
        # nan slices must be inserted at points where this is not the case (done later in code)
        # this is mostly relevant for beacon data (lots of gaps)

        # E.g.: H1 beacon. Image 1 and 0 have a time difference of 240 minutes. This is within maxgap * cadence.
        # A regular running difference image is produced. Image 2 and 1 have a time difference of 120 minutes.
        # This is also within maxgap * cadence and a regular running difference image is produced.
        # Now, there are two running difference images, each occupying one column. One represents a time difference of 240 min,
        # the other a time difference of 120 min. This leads to errors in the plot.
        # Solution: The time difference is divided by the cadence (240/120 = 2, 120/120 = 1) and an extra column of nans is
        # inserted if result > 1. This ensures that each column always represents a timestep of 120 min (in this example).


        ap_ind = np.round(tdiff[i] / cadence, 0)
        ap_ind = int(ap_ind)

        if ap_ind > 1:
            nan_ind.append(int(i))

            if i - 1 == 0:
                nan_dat.append(int(0))

            else:
                nan_dat.append(int(ap_ind) - 1)

    return nan_dat, nan_ind, r_dif


#######################################################################################################################################


@numba.njit()
def conv_coords(Tx, Ty, ftpsc):
    Tx = Tx * np.pi / 180
    Ty = Ty * np.pi / 180

    hpclat = Ty
    hpclon = Tx

    elongs = np.arccos(np.cos(hpclat) * np.cos(hpclon))

    pas = np.sin(hpclat) / np.sin(elongs)
    elongs = elongs * 180 / np.pi

    if ftpsc == 'A':
        pas = np.arccos(pas) * 180 / np.pi

    if ftpsc == 'B':
        pas = (2 * np.pi - np.arccos(pas)) * 180 / np.pi

    return pas, elongs


#######################################################################################################################################


def bin_elong(noel, res2, tdi, tdata):

    for j in range(noel):

        if len(res2[j][0]) > 1:
            tmp_new = rej_out(tdi[res2[j][0]], 2)
            tmp1 = np.median(tmp_new)
            tdata[j] = tmp1

        elif len(res2[j][0]) == 1:
            tdata[j] = tdi[res2[j][0]]

        else:
            tdata[j] = np.nan

    return tdata


#######################################################################################################################################


def align_time(time_h1, time_h2):
    match1_beg = []
    match2_beg = []

    sdate1 = time_h1[0][0:14]
    sdate2 = time_h2[0][0:14]

    for i in range(len(time_h1)):
        if sdate1 in time_h1[i]:
            match1_beg.append(i)

    for i in range(len(time_h2)):
        if sdate2 in time_h2[i]:
            match2_beg.append(i)

    if match1_beg[0] >= match2_beg[0]:
        hit_beg = match1_beg[0]
        f = 0

    else:
        hit_beg = match2_beg[0]
        f = 1

    match1_end = []
    match2_end = []

    sdate1 = time_h1[-1][0:14]
    sdate2 = time_h2[-1][0:14]

    for i in range(len(time_h1)):
        if sdate1 in time_h1[i]:
            match1_end.append(i)

    for i in range(len(time_h2)):
        if sdate2 in time_h2[i]:
            match2_end.append(i)

    if match1_end[-1] >= match2_end[-1]:
        hit_end = match1_end[-1]
        g = 0

    else:
        hit_end = match2_end[-1]
        g = 1

    return hit_beg, hit_end, f, g

#######################################################################################################################################


def hi_img(start, ftpsc, bflag, path, calpath, high_contr):

    #start_t = timer()

    tc = datetime.datetime(2015, 7, 1)
    # cadence is in minutes

    if (bflag == 'beacon'):
        cadence_h1 = 120.0
        cadence_h2 = 120.0

    if (bflag == 'science'):
        cadence_h1 = 40.0
        cadence_h2 = 120.0

    # define maximum gap between consecutive images
    # if gap > maxgap, no running difference image is produced, timestep is filled with 0 instead

    maxgap = -3.5

    # get path to .fits files from config.txt
    red_path = path + start + '_' + ftpsc + '_' + bflag + '_red/'
    files_h1 = []
    files_h2 = []
   # read in .fits files

    print('Getting files...')

    if bflag == 'science':

        for file in sorted(glob.glob(red_path + '*s4h1' + '*.fts')):
            files_h1.append(file)

        for file in sorted(glob.glob(red_path + '*s4h2' + '*.fts')):
            files_h2.append(file)

    if bflag == 'beacon':

        for file in sorted(glob.glob(red_path + '*s7h1' + '*.fts')):
            files_h1.append(file)

        for file in sorted(glob.glob(red_path + '*s7h2' + '*.fts')):
            files_h2.append(file)

    # get times and headers from .fits files

    header_h1 = [fits.getheader(files_h1[i]) for i in range(len(files_h1))]
    tcomp1 = datetime.datetime.strptime(header_h1[0]['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
    time_h1 = [header_h1[i]['DATE-OBS'] for i in range(len(header_h1))]

    header_h2 = [fits.getheader(files_h2[i]) for i in range(len(files_h2))]
    tcomp2 = datetime.datetime.strptime(header_h2[0]['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
    time_h2 = [header_h2[i]['DATE-OBS'] for i in range(len(header_h2))]

    # align_time is directly adapted from DA, p. 67
    # defines matching beginning and end dates for h1 and h2
    if bflag == 'science':

        print('Aligning images...')

        hit_b, hit_e, f1, f2 = align_time(time_h1, time_h2)

    # following lines directly adapted from DA, p. 68
    # decides which data to get from .fits files based on which start and end times were obtained by align_times

    # f1 notes whether h1 starts later (f1 = 0) or h2 starts later (f1 = 1)
    # f2 notes whether h1 ends later (f2 = 0) or h2 ends later (f2 = 1)

        if (f1 == 0) and (f2 == 0) and (hit_e != -1):

            data_h1_pre = np.array([fits.getdata(files_h1[i]) for i in range(hit_b, hit_e)])
            data_h2_pre = np.array([fits.getdata(files_h2[i]) for i in range(len(files_h2))])
            time_h1_pre = time_h1[hit_b:hit_e]
            time_h2_pre = time_h2

            ran_h1 = [hit_b, hit_e]
            ran_h2 = [0, len(files_h2)]

        elif (f1 == 1) and (f2 == 1):

            data_h1_pre = np.array([fits.getdata(files_h1[i]) for i in range(len(files_h1))])
            data_h2_pre = np.array([fits.getdata(files_h2[i]) for i in range(hit_b, hit_e)])
            time_h1_pre = time_h1
            time_h2_pre = time_h2[hit_b:hit_e]

            ran_h1 = [0, len(files_h1)]
            ran_h2 = [hit_b, hit_e]

        elif (f1 == 0) and (f2 == 1):

            data_h1_pre = np.array([fits.getdata(files_h1[i]) for i in range(hit_b, len(files_h1))])
            data_h2_pre = np.array([fits.getdata(files_h2[i]) for i in range(0, hit_e)])
            time_h1_pre = time_h1[hit_b:]
            time_h2_pre = time_h2[0:hit_e]

            ran_h1 = [hit_b, len(files_h1)]
            ran_h2 = [0, hit_e]

        elif (f1 == 1) and (f2 == 0):

            data_h1_pre = np.array([fits.getdata(files_h1[i]) for i in range(0, hit_e)])
            data_h2_pre = np.array([fits.getdata(files_h2[i]) for i in range(hit_b, len(files_h2))])
            time_h1_pre = time_h1[0:hit_e]
            time_h2_pre = time_h2[hit_b:]

            ran_h1 = [0, hit_e]
            ran_h2 = [hit_b, len(files_h2)]

    if bflag == 'beacon':

        data_h1_pre = np.array([fits.getdata(files_h1[i]) for i in range(len(files_h1))])
        data_h2_pre = np.array([fits.getdata(files_h2[i]) for i in range(len(files_h2))])

        ran_h1 = [0, len(files_h1)]
        ran_h2 = [0, len(files_h2)]

    data_h1 = data_h1_pre
    data_h2 = data_h2_pre

    # missing data is identified from header

    #data_h1 = rej_out(data_h1, 3)
    #data_h2 = rej_out(data_h2, 3)

    missing_h1 = np.array([header_h1[i]['NMISSING'] for i in range(ran_h1[0], ran_h1[1])])
    mis_h1 = np.array(np.where(missing_h1 > 0))
    exp1 = np.array([header_h1[i]['EXPTIME'] for i in range(ran_h1[0], ran_h1[1])])

    missing_h2 = np.array([header_h2[i]['NMISSING'] for i in range(ran_h2[0], ran_h2[1])])
    mis_h2 = np.array(np.where(missing_h2 > 0))
    exp2 = np.array([header_h2[i]['EXPTIME'] for i in range(ran_h2[0], ran_h2[1])])

    # missing data is replaced with nans
    if np.size(mis_h1) > 0:
        for i in mis_h1:
            data_h1[i, :, :] = np.nan

    if np.size(mis_h2) > 0:
        for i in mis_h2:
            data_h2[i, :, :] = np.nan

    #data_h1 = np.nan_to_num(data_h1)
    #data_h2 = np.nan_to_num(data_h2)

    # time difference is taken for producing running difference images

    print('Creating running difference images...')

    time_obj_h1 = [Time(time_h1[i], format='isot', scale='utc') for i in range(len(time_h1))]

    tdiff_h1 = np.array([(time_obj_h1[i] - time_obj_h1[i - 1]).sec / 60 for i in range(len(time_obj_h1))])
    t_h1 = len(time_h1)

    time_obj_h2 = [Time(time_h2[i], format='isot', scale='utc') for i in range(len(time_h2))]
    tdiff_h2 = np.array([(time_obj_h2[i] - time_obj_h2[i - 1]).sec / 60 for i in range(len(time_obj_h2))])
    t_h2 = len(time_h2)

    # create_img produces running difference images
    # create_img also returns indices of running difference array where nan columns must be inserted
    # nan columns must be inserted to ensure that one column of h1 represents 40 min, one column of h2 represents 120 min

    nan_dat_h1, nan_ind_h1, r_dif_h1 = create_img(tdiff_h1, maxgap, cadence_h1, data_h1)


    #r_dif_h1 = np.nan_to_num(r_dif_h1)

    dif_map_h1 = [sunpy.map.Map(r_dif_h1[k], header_h1[k + 1]) for k in range(len(r_dif_h1))]

    nan_dat_h2, nan_ind_h2, r_dif_h2 = create_img(tdiff_h2, maxgap, cadence_h2, data_h2)
    #r_dif_h2 = np.nan_to_num(r_dif_h2)
    dif_map_h2 = [sunpy.map.Map(r_dif_h2[k], header_h2[k + 1]) for k in range(len(r_dif_h2))]


    # save height and width of h1 and h2 data for later

    w_h1 = np.shape(data_h1[0])[0]
    h_h1 = np.min([np.shape(data_h1[i])[1] for i in range(len(data_h1))])

    w_h2 = np.shape(data_h2[0])[0]
    h_h2 = np.min([np.shape(data_h2[i])[1] for i in range(len(data_h2))])

    # define width of line to be cut out from images

    pix = 16

    print('Getting coordinates from map...')

    dif_cut_h1 = np.array([dif_map_h1[i].data[int(w_h1 / 2 - pix):int(w_h1 / 2 + pix), 0:h_h1] for i in range(len(dif_map_h1))])
    dif_cut_h2 = np.array([dif_map_h2[i].data[int(w_h2 / 2 - pix):int(w_h2 / 2 + pix), 0:h_h2] for i in range(len(dif_map_h2))])

    # extract coordinates from map

    all_coord_h1 = [sunpy.map.all_coordinates_from_map(dif_map_h1[i]) for i in range(len(dif_map_h1))]
    all_coord_h2 = [sunpy.map.all_coordinates_from_map(dif_map_h2[i]) for i in range(len(dif_map_h2))]

    # cut line in ecliptic

    coord_cut_h1 = [all_coord_h1[i][int(w_h1 / 2 - pix):int(w_h1 / 2 + pix), 0:h_h1] for i in range(len(all_coord_h1))]
    coord_cut_h2 = [all_coord_h2[i][int(w_h2 / 2 - pix):int(w_h2 / 2 + pix), 0:h_h2] for i in range(len(all_coord_h2))]

    # extract elongation
    # note: used to convert these coordinates using conv_coord function, as Jackie does in her program
    # note: this leads to wrong elongations (off by around 2 °, i.e. h2 starts at 20 ° instead of 18 °), don't know why

    lon_h1 = np.array([coord_cut_h1[i].Tx.to(u.deg).value for i in range(len(coord_cut_h1))])
    lon_h2 = np.array([coord_cut_h2[i].Tx.to(u.deg).value for i in range(len(coord_cut_h2))])

    lat_h1 = np.array([coord_cut_h1[i].Ty.to(u.deg).value for i in range(len(coord_cut_h1))])
    lat_h2 = np.array([coord_cut_h2[i].Ty.to(u.deg).value for i in range(len(coord_cut_h2))])

    _, elongs_h1 = conv_coords(lon_h1, lat_h1, ftpsc)
    _, elongs_h2 = conv_coords(lon_h2, lat_h2, ftpsc)

    # mask bad columns in h2
    # adapted from get_smask.pro for IDL

    mask = np.array([get_smask(ftpsc, header_h2[i], path, time_h2[i], calpath) for i in range(ran_h2[0] + 1, ran_h2[1])])
    r_dif_h2 = np.array(r_dif_h2)
    r_dif_h2[mask == 0] = np.nan

    # convert time to correct format
    # matplotlib uses number of days since 0001-01-01 UTC, plus 1

    time_t_h1 = [datetime.datetime.strptime(day, '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y/%m/%d %H:%M:%S.%f') for day in time_h1]
    x_lims_h1 = mdates.datestr2num(time_t_h1)

    time_t_h2 = [datetime.datetime.strptime(day, '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y/%m/%d %H:%M:%S.%f') for day in time_h2]
    x_lims_h2 = mdates.datestr2num(time_t_h2)

    sh_cut_h1 = np.shape(dif_cut_h1)
    sh_cut_h2 = np.shape(dif_cut_h2)

    print('Adjusting image size...')

    # get maximum and minimum of elongations

    elo_max_h1 = np.nanmax(elongs_h1)
    elo_min_h1 = np.nanmin(elongs_h1)

    elo_max_h2 = np.nanmax(elongs_h2)
    elo_min_h2 = np.nanmin(elongs_h2)

    dif_med_h1_pre = np.zeros((sh_cut_h1[0], sh_cut_h1[2]))
    dif_med_h2 = np.zeros((sh_cut_h2[0], sh_cut_h2[2]))

    # take median of pixels cut out from h1 and h2

    for i in range(sh_cut_h1[0]):
        for j in range(sh_cut_h1[2]):
            dif_med_h1_pre[i, j] = np.nanmedian(dif_cut_h1[i, :, j])

    for i in range(sh_cut_h2[0]):
        for j in range(sh_cut_h2[2]):
            dif_med_h2[i, j] = np.nanmedian(dif_cut_h2[i, :, j])

    elongation_h1 = np.linspace(elo_min_h1, elo_max_h1, h_h1)
    elongation_h2 = np.linspace(elo_min_h2, elo_max_h2, h_h2)

    # insert nan slices to keep correct cadence for beacon images, not necessary for science images since they are binned

    if bflag == 'beacon':

      n_h1 = np.zeros(sh_cut_h1[2])
      n_h1[:] = np.nan

      n_h2 = np.zeros(sh_cut_h2[2])
      n_h2[:] = np.nan

      # in the follwoing lines, the indices defined by create_img are filled with nan values to keep cadence correct

      nan_ind_h1.reverse()
      nan_dat_h1.reverse()
      dif_med_h1_pre = dif_med_h1_pre.tolist()

      nan_ind_h2.reverse()
      nan_dat_h2.reverse()
      dif_med_h2 = dif_med_h2.tolist()

      k = 0

      for i in nan_ind_h1:
          for j in range(nan_dat_h1[k]):
              dif_med_h1_pre.insert(i - 1, n_h1)
          k = k + 1

      k = 0

      for i in nan_ind_h2:
          for j in range(nan_dat_h2[k]):
              dif_med_h2.insert(i - 1, n_h2)
          k = k + 1

    dif_med_h1_pre = np.array(dif_med_h1_pre)
    dif_med_h2 = np.array(dif_med_h2)



    #dif_med_h1_pre = np.nan_to_num(dif_med_h1_pre)
    #dif_med_h2 = np.nan_to_num(dif_med_h2)

    # here, h1 images lying within the time period of one h2 image are summed up
    # adapted from DA, p.72

    if bflag == 'science':
        index1_new = []

        dif_med_h1 = np.zeros((len(dif_med_h2), sh_cut_h1[2]))

        for i in range(len(x_lims_h2) - 1):
            index1_new.append(np.where((x_lims_h1 > x_lims_h2[i]) & (x_lims_h1 < x_lims_h2[i + 1])))

        k = 0

        for i in range(len(index1_new)):
            dif_med_h1[k, :] = dif_med_h1[k, :] + np.sum(dif_med_h1_pre[index1_new[i][0], :], axis=0)

            k = k + 1

        # new time scales are defined
        # I start at index 1, not zero since the time of the first running difference image is x_lims_h_[1]
        # E.g.: 60 images total -> 59 running difference images -> time of first running difference image is x_lims_h_[1]
        # since image is produced by taking image[1]-image[0]

        exp1 = exp1 / 60
        exp2 = exp2 / 60

        # new timescales defined according to DA, p.72

        t1_n = x_lims_h2[1:] - np.mean(exp2) / (60 * 24)
        t2_n = x_lims_h2[1:] + np.mean(exp2) / (60 * 24)

    if bflag == 'beacon':

        dif_med_h1 = dif_med_h1_pre

    # histogram equalization is performed if keyword 'high contrast' is present in config.txt. Image must first be normalized
    # to range [0, 1]

    if (bflag == 'science') and (high_contr == True):

        sh_med_h1 = np.shape(dif_med_h1)
        sh_med_h2 = np.shape(dif_med_h2)

        jmap_h1 = np.zeros((sh_med_h1[1], sh_med_h1[0]))
        jmap_h2 = np.zeros((sh_med_h2[1], sh_med_h2[0]))

        lim_max_h1 = np.nanmax(dif_med_h1)
        lim_min_h1 = np.nanmin(dif_med_h1)

        lim_max_h2 = np.nanmax(dif_med_h2)
        lim_min_h2 = np.nanmin(dif_med_h2)

        if lim_max_h1 > abs(lim_min_h1):
            lim_h1 = lim_max_h1

        else:
            lim_h1 = abs(lim_min_h1)

        if lim_max_h2 > abs(lim_min_h2):
            lim_h2 = lim_max_h2

        else:
            lim_h2 = abs(lim_min_h2)

        zrange_h1 = [-lim_h1, lim_h1]
        zrange_h2 = [-lim_h2, lim_h2]

        for i in range(sh_med_h1[0]):
            jmap_h1[:, i] = (dif_med_h1[i] - zrange_h1[0]) / (zrange_h1[1] - zrange_h1[0])

        for i in range(sh_med_h2[0]):
            jmap_h2[:, i] = (dif_med_h2[i] - zrange_h2[0]) / (zrange_h2[1] - zrange_h2[0])

        j1t = jmap_h1.transpose()
        j2t = jmap_h2.transpose()

        # interpolate images to new timescales
        # done according to DA, p.72

        for i in range(np.shape(j1t)[1]):
            j1t[:, i] = np.interp(t1_n, x_lims_h2[1:], j1t[:, i])

        for i in range(np.shape(j2t)[1]):
            j2t[:, i] = np.interp(t2_n, x_lims_h2[1:], j2t[:, i])

        j1nt = j1t.transpose()
        j2nt = j2t.transpose()

    # no histogram equalization is performed if keyword 'high contrast' is not present, instead just go straight to interpolation

    elif (bflag == 'science') and (high_contr == False):

        # interpolate images to new timescales
        # done according to DA, p.72

        for i in range(np.shape(dif_med_h1)[1]):
            dif_med_h1[:, i] = np.interp(t1_n, x_lims_h2[1:], dif_med_h1[:, i])

        for i in range(np.shape(dif_med_h2)[1]):
            dif_med_h2[:, i] = np.interp(t2_n, x_lims_h2[1:], dif_med_h2[:, i])

        j1nt = dif_med_h1.transpose()
        j2nt = dif_med_h2.transpose()


    # neither interpolation nor histogram equalization are necessary for beacon images

    if bflag == 'beacon':

        j1nt = dif_med_h1.transpose()
        j2nt = dif_med_h2.transpose()


    jmap_h1 = np.nan_to_num(j1nt)
    jmap_h2 = np.nan_to_num(j2nt)

    if bflag == 'science':

        jx1 = np.shape(jmap_h1)[1]
        jy1 = np.shape(jmap_h1)[0]
        jy2 = np.shape(jmap_h2)[0]
        jy = jy1 + jy2

        jmap_combined = np.zeros((jy+2, jx1))

        for i in range(jx1-1):
            jmap_combined[0:jy2, i] = jmap_h2[0:jy2, i]
            jmap_combined[jy2+1:-1, i] = jmap_h1[0:jy1, i]

    # find maximum elongation of h2 and cut h1 off at that elongation
    # perform histogram equalization for better contrast in plot

    el_lim_h2 = np.where(elongation_h2 > 18.)[0][0]
    el2 = [elongation_h2[el_lim_h2], elongation_h2[-1]]

    if tcomp2 < tc:

        el_lim_h2 = np.shape(jmap_h2)[0] - el_lim_h2
        jmap_h2 = jmap_h2[0:el_lim_h2, :]

    if tcomp2 > tc:

        el_lim_h2 = el_lim_h2
        jmap_h2 = jmap_h2[el_lim_h2:-1, :]

    el_lim_h1 = np.where(elongation_h1 < el2[0])[0][-1]
    el1 = [elongation_h1[0], elongation_h1[el_lim_h1]]

    if tcomp1 < tc:

        el_lim_h1 = np.shape(jmap_h1)[0] - el_lim_h1
        jmap_h1 = jmap_h1[el_lim_h1:-1, :]
        tflag = 0

    if tcomp1 > tc:

        el_lim_h1 = el_lim_h1
        jmap_h1 = jmap_h1[0:el_lim_h1, :]
        tflag = 1

    # histogram equalization is performed if appropriate keyword is present

    if (high_contr == True) and (bflag == 'science'):

        jmap_h1 = equalize_hist(jmap_h1)
        jmap_h2 = equalize_hist(jmap_h2)

    # images are returned separately and plotted via imshow

    if bflag == 'science':
        return jmap_h1, jmap_h2, el1, el2, t1_n, t2_n, tflag, jmap_combined

    if bflag == 'beacon':
        return jmap_h1, jmap_h2, el1, el2, x_lims_h1, x_lims_h2, tflag

    #print(f"Elapsed Time: {timer() - start_t}")

#######################################################################################################################################

def data_reduction(line):

    fitsfil, datelist_int, ftpsc, instrument, bflag = download_files(line)

    file = open('config.txt', 'r')
    config = file.readlines()
    path = config[0].splitlines()[0]

    savepath = path + datelist_int[0] + '_' + ftpsc + '_' + bflag + '_red'

    print('Starting data reduction for:', datelist_int[0])

    if bflag == 'science':
        fits_hi1 = [s for s in fitsfil if "s4h1" in s]
        fits_hi2 = [s for s in fitsfil if "s4h2" in s]

    if bflag == 'beacon':
        fits_hi1 = [s for s in fitsfil if "s7h1" in s]
        fits_hi2 = [s for s in fitsfil if "s7h2" in s]

    fitslist = [fits_hi1, fits_hi2]

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

            #desatcube = regnewcube

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

            dateobs = [header[i]['date-obs'] for i in range(length[2])]

            dstart1 = [header[i]['dstart1'] for i in range(length[2])]
            dstart2 = [header[i]['dstart2'] for i in range(length[2])]
            dstop1 = [header[i]['dstop1'] for i in range(length[2])]
            dstop2 = [header[i]['dstop2'] for i in range(length[2])]

            naxis1 = [header[i]['naxis1'] for i in range(length[2])]
            naxis2 = [header[i]['naxis2'] for i in range(length[2])]

            exptime = [header[i]['exptime'] for i in range(length[2])]
            n_images = [header[i]['n_images'] for i in range(length[2])]
            cleartim = [header[i]['cleartim'] for i in range(length[2])]
            ro_delay = [header[i]['ro_delay'] for i in range(length[2])]
            ipsum = [header[i]['ipsum'] for i in range(length[2])]

            rectify = [header[i]['rectify'] for i in range(length[2])]
            obsrvtry = [header[i]['obsrvtry'] for i in range(length[2])]

            for i in range(len(obsrvtry)):

                if obsrvtry[i] == 'STEREO_A':
                    obsrvtry[i] = True

                else:
                    obsrvtry[i] = False

            line_ro = [header[i]['line_ro'] for i in range(length[2])]
            line_clr = [header[i]['line_clr'] for i in range(length[2])]

            time = [datetime.datetime.strptime(dateobs[i], '%Y-%m-%dT%H:%M:%S.%f') for i in range(length[2])]
            tc = datetime.datetime(2015, 7, 1)

            post_conj = [int(time[i] > tc) for i in range(length[2])]

            header_int = np.array([[dstart1[i], dstart2[i], dstop1[i], dstop2[i], naxis1[i], naxis2[i], n_images[i], post_conj[i]]
                                for i in range(length[2])])

            header_flt = np.array([[exptime[i], cleartim[i], ro_delay[i], ipsum[i], line_ro[i], line_clr[i]]
                                for i in range(length[2])])

            header_str = np.array([[rectify[i], obsrvtry[i]] for i in range(length[2])])

            des = [hi_desmear(desatcube[:, :, i], header_int[i], header_flt[i], header_str[i]) for i in range(length[2])]
            des = np.array(des)

            print('Calibrating image...')

            cal = [get_calimg(ins, ftpsc, path, header[k]) for k in range(length[2])]
            cal = np.array(cal)

            rec = cal * des
            rec_map = [sunpy.map.Map(rec[k, :, :], header[k]) for k in range(length[2])]

            #print('Rotating image..')

            #final_map = [rec_map[i].rotate() for i in range(len(rec_map))]
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

                fits.writeto(savepath + '/' + newname[i], rec_map[i].data, header[i], output_verify='silentfix', overwrite=True)


            f = f + 1

    #print(f"Elapsed Time: {timer() - start_t}")

#######################################################################################################################################

def dif_img(line):

  file = open('config.txt', 'r')
  config = file.readlines()
  path = config[0].splitlines()[0]
  calpath = config[1].splitlines()[0]
  ftpsc = config[2].splitlines()[0]
  instrument = config[3].splitlines()[0]
  bflag = config[4].splitlines()[0]
  high_contr = config[5].splitlines()[0]
  start = config[6+line].splitlines()[0]
  register_matplotlib_converters()


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

    fig, ax = plt.subplots(figsize=(10, 5), frameon=False)

    #img_1 = ndimage.gaussian_filter(img1, sigma=(1, 1), order=0)
    #img_2 = ndimage.gaussian_filter(r2, sigma=(1, 1), order=0)

    # images are plotted via imshow, one above the other
    # extent keyword sepcifies beginning and end values of x- and y-axis for h1 and h2 image

    #if high_contr:

    #  vmin1 = np.nanmin(img1)
    #  vmax1 = np.nanmax(img1)
    #  vmin2 = np.nanmin(img2)
    #  vmax2 = np.nanmax(img2)

      #plt.hist(img1_center, bins=100)
      #plt.savefig(savepath+'/'+start+'_hist.png')
    # specific boundaries must be chosen for images on which no histogram equalization was performed
    # since they are not normalized to [0, 1]

    #if not high_contr:

      #vmin1 = -1
      #vmax1 = 1
      #vmin2 = -1
      #vmax2 = 1

      #plt.hist(img1_center, bins=100)

      #plt.savefig(savepath+'/'+start+'_hist.png')

    img1_center = (img1 - np.nanmean(img1)) / np.std(img1)
    img2_center = (img2 - np.nanmean(img2)) / np.std(img2)

    img1_center = rej_out(img1_center, 7)
    img2_center = rej_out(img2_center, 7)

    plt.hist(img1_center, bins=100)
    plt.savefig(savepath+'/'+start+'_hist.png')

    dt2 = (time_h2[-1]-time_h2[0])/len(time_h2)
    time_h2 = time_h2 + dt2

    ax.imshow(img1_center, cmap='gray', extent=[time_h1[0], time_h1[-1], e1[0], e1[-1]], aspect='auto', vmin = np.nanmean(img1_center)-1.5*np.std(img1_center), vmax = np.nanmean(img1_center)+1.5*np.std(img1_center), origin = orig)

    ax.imshow(img2_center, cmap='gray', extent=[time_h2[0], time_h2[-1], e2[0], e2[-1]], aspect='auto', vmin = np.nanmean(img2_center)-1.5*np.std(img2_center), vmax = np.nanmean(img2_center)+1.5*np.std(img2_center), origin = orig)


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

    img1_center = rej_out(img1_center, 7)
    img2_center = rej_out(img2_center, 7)

    #img1_norm = 2*(img1_center - np.nanmin(img1_center)) / (np.nanmax(img1_center) - np.nanmin(img1_center)) - 1
    #img2_norm = 2*(img2_center - np.nanmin(img2_center)) / (np.nanmax(img2_center) - np.nanmin(img2_center)) - 1

    #img1_eqadapt = equalize_adapthist(img1_norm)
    #img2_eqadapt = equalize_adapthist(img2_norm)

    plt.hist(img1_center, bins=100)

    plt.savefig(savepath+'/'+start+'_hist.png')

    print(np.nanmean(img1_center))
    print(np.std(img1_center))

    fig, ax = plt.subplots(figsize=(10, 5), frameon=False)

    ax.imshow(img1_center, cmap='gray', extent=[time_h1[0], time_h1[-1], e1[0], e1[-1]], aspect='auto', vmin = np.nanmean(img1_center)-1.5*np.std(img1_center), vmax = np.nanmean(img1_center)+1.5*np.std(img1_center), origin = orig)

    ax.imshow(img2_center, cmap='gray', extent=[time_h2[0], time_h2[-1], e2[0], e2[-1]], aspect='auto', vmin = np.nanmean(img2_center)-1.5*np.std(img2_center), vmax = np.nanmean(img2_center)+1.5*np.std(img2_center), origin = orig)

  fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.axis('off')
  plt.ylim(3, e2[-1])
  plt.savefig(savepath+'/'+start+'_jplot.png', bbox_inches = 0, pad_inches=0)

  plt.close()

  with open(savepath + '/' + 'objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
      pickle.dump([time_h2[0], time_h2[-1], e1[0], e2[-1]], f)
