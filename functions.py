import numpy as np
import numba
from astropy.time import Time
from astropy.io import fits
import glob
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.colors import ListedColormap
import math
from astropy import wcs
import requests
from bs4 import BeautifulSoup
import os
import wget
import pandas as pd
import datetime
from itertools import repeat
import matplotlib.pyplot as plt
import pickle
import cv2
import sys
from scipy.ndimage import shift
import warnings
from matplotlib import image
import stat
from scipy.interpolate import NearestNDInterpolator
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from requests.adapters import HTTPAdapter, Retry
import psutil
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from functools import partial
import traceback
import logging
from skimage import exposure
import subprocess

warnings.filterwarnings("ignore")

#######################################################################################################################################

def limit_cpu():
    """
    Is called when starting a new multiprocessing pool. Decreases priority of processes to limit total CPU usage. 
    """
    p = psutil.Process(os.getpid())
    # set to lowest priority
    p.nice(19)

#######################################################################################################################################


def listfd(input_url, extension):
    """
    Provides list of urls and corresponding file names to download.

    @param input_url: URL of STEREO-HI image files
    @param extension: File ending of STEREO-HI image files
    @return: List of URLs and corresponding filenames to be downloaded
    """

    disable_warnings(InsecureRequestWarning)

    
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)

    session.mount('http://', adapter)
    session.mount('https://', adapter)

    output_urls = []

    page = session.get(input_url).text

    soup = BeautifulSoup(page, 'html.parser')
    url_found = [input_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]
    filename = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]

    for i in range(len(filename)):
        output_urls.append((filename[i], url_found[i]))

    return output_urls


#######################################################################################################################################

def fetch_url(path, entry):
    """
    Downloads URLs specified by listfd.

    @param path: Path where downloaded files are to be saved
    @param entry: Combination of filename and URL of downloaded file
    """
    filename, uri = entry

    if not os.path.exists(path + '/' + filename):
        wget.download(uri, path)

#######################################################################################################################################

def check_calfiles(path):
    """
    Checks if SSW IDL HI calibration files are present - creates appropriate directory and downloads them if not.

    @param path: Path in which calibration files are located/should be located
    """
    url = "https://soho.nascom.nasa.gov/solarsoft/stereo/secchi/calibration/"


    if not os.path.exists(path + 'calibration/'):
        
        try:
            os.makedirs(path + 'calibration/')
            uri = listfd(url, '.fts')
       
            for entry in uri:
                fetch_url(path + 'calibration', entry)
            
            subprocess.call(['chmod', '-R', '775', path + 'calibration/'])
            return
            
        except KeyboardInterrupt:
            return
       
        except Exception as e:
            logging.error(traceback.format_exc())
            sys.exit()
    else:
        return

#######################################################################################################################################   
def download_files(start, duration, save_path, ftpsc, instrument, bflag, silent):
    """
    Downloads STEREO images from NASA pub directory

    @param start: Beginning date (DDMMYYYY) of files to be downloaded
    @param duration: Timespan for files to be downloaded (in days)
    @param save_path: Path for saving downloaded files
    @param ftpsc: Spacecraft (STEREO-A/STEREO-B) for which to download files
    @param instrument: Instrument (HI-1/HI-2) for which to download files
    @param bflag: Data type (science/beacon) for which to download files
    @param silent: Run in silent mode
    """
    fitsfil = []

    date = datetime.datetime.strptime(start, '%Y%m%d')

    if ftpsc == 'A':
        sc = 'ahead'

    if ftpsc == 'B':
        sc = 'behind'

    if instrument == 'hi1hi2':
        instrument = ['hi_1', 'hi_2']

    elif instrument == 'hi_1':
        instrument = ['hi_1']

    elif instrument == 'hi_2':
        instrument = ['hi_2']

    else:

        print('Invalid instrument specification. Exiting...')
        sys.exit()

    datelist = pd.date_range(date, periods=duration).tolist()
    datelist_int = [str(datelist[i].year) + datelist[i].strftime('%m') + datelist[i].strftime('%d') for i in range(len(datelist))]

    if not silent:
        print('Fetching files...')

    for ins in instrument:
        for date in datelist_int:

            if bflag == 'beacon':

                url = 'https://stereo-ssc.nascom.nasa.gov/pub/beacon/' + sc + '/secchi/img/' + ins + '/' + str(date)

            else:

                url = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/' + sc[0] + '/img/' + ins + '/' + str(date)

            if bflag == 'beacon':
                path_flg = 'beacon'
                path_dir = save_path + 'stereo' + sc[0] + '/' + path_flg + '/secchi/img/' + ins + '/' + str(date)

                if ins == 'hi_1':
                    if sc == 'ahead':
                        ext = 's7h1A.fts'
                    if sc == 'behind':
                        ext = 's7h1B.fts'

                if ins == 'hi_2':
                    if sc == 'ahead':
                        ext = 's7h2A.fts'
                    if sc == 'behind':
                        ext = 's7h2B.fts'

            if bflag == 'science':

                path_flg = 'L0'
                path_dir = save_path + 'stereo' + sc[0] + '/secchi/' + path_flg + '/img/' + ins + '/' + str(date)

                if ins == 'hi_1':
                    if sc == 'ahead':
                        ext = 's4h1A.fts'
                    if sc == 'behind':
                        ext = 's4h1B.fts'

                if ins == 'hi_2':
                    if sc == 'ahead':
                        ext = 's4h2A.fts'
                    if sc == 'behind':
                        ext = 's4h2B.fts'
            
            if not os.path.exists(path_dir):
              os.makedirs(path_dir)
              #flag = True
              
            #else:
            #  if not os.listdir(path_dir):
            #    flag = True
            #  else:
            #    flag = False
              
            num_cpus = cpu_count()

            pool = Pool(int(num_cpus/2), limit_cpu)

            urls = listfd(url, ext)
            inputs = zip(repeat(path_dir), urls)
            
            try:
              results = pool.starmap(fetch_url, inputs, chunksize=5)
              
            except ValueError:
              continue
                    
            if bflag == 'beacon':
                path_flg = 'beacon'
                path_dir = save_path + 'stereo' + sc[0] + '/' + path_flg + '/secchi/img/'
                subprocess.call(['chmod', '-R', '775', path_dir])
                
            if bflag == 'science':
                path_flg = 'L0'
                path_dir = save_path + 'stereo' + sc[0] + '/secchi/' + path_flg + '/img/'
                subprocess.call(['chmod', '-R', '775', path_dir])
                       
            pool.close()
            pool.join()
      
#######################################################################################################################################

def hi_remove_saturation(data, header):
    """Direct conversion of hi_remove_saturation.pro for IDL.
    Detects and masks saturated pixels with nan. Takes image data and header as input. Returns fixed image.
    @param data: Data of .fits file
    @param header: Header of .fits file
    @return: Data with oversaturated columns removed"""

    # threshold value before pixel is considered saturated
    sat_lim = 14000

    # number of pixels in a column before column is considered saturated
    nsaturated = 5

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
            ans[:, ii] = np.nan #np.nanmedian(data)

        else:
            ans = data.copy()

    else:
        ans = data.copy()

    return ans


#######################################################################################################################################

def hi_remove_saturation_rdif(data):
    """Direct conversion of hi_remove_saturation.pro for IDL.
    Detects and masks saturated pixels with nan. Takes image data and header as input. Returns fixed image.
    @param data: Data of .fits file
    @param header: Header of .fits file
    @return: Data with oversaturated columns removed"""

    # number of pixels in a column before column is considered saturated
    nsaturated = 3

    dsatval = np.nanmedian(data) + 2 * np.std(data)

    ind = np.where(np.abs(data) > dsatval)

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
            ans[:, ii] = np.nan  # np.nanmedian(data)

        else:
            ans = data.copy()

    else:
        ans = data.copy()

    return ans

#######################################################################################################################################

def remove_bad_col(data):
    """
    Removes saturated columns from .fits files.

    @param data: Data of .fits file
    @return: Data with saturated columns removed
    """
    nsaturated = 3
    dsatval = 0.95

    ind = np.where(np.abs(data) > dsatval)
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
            ans[:, ii] = np.nanmedian(data)

        else:
            ans = data.copy()

    else:
        ans = data.copy()

    return ans


#######################################################################################################################################

def scc_sebip(data, header, silent):
    """Direct conversion of scc_sebip.pro for IDL.
    Determines what has happened in terms of on-board sebip binning and corrects it.
    Takes image data and header as input and returns fixed image.
    @param data: Data of .fits file
    @param header: Header of .fits file
    @param silent: Run in silent mode
    @return: Data corrected for on-board binning"""

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
        if not silent:
            print('Correted for divide by 2 x {}'.format(cnt1))

    if cnt2 > 0:
        data = data * (2.0 ** cnt2)
        if not silent:
            print('Correted for square root x {}'.format(cnt2))
    if cntspw > 0:
        data = data * (64.0 ** cntspw)
        if not silent:
            print('Correted for HI SPW divide by 64 x {}'.format(cntspw))
    if cnt50 > 0:
        data = data * (4.0 ** cnt50)
        if not silent:
            print('Correted for divide by 4 x {}'.format(cnt50))
    if cnt53 > 0 and header['ipsum'] > 0:
        data = data * (4.0 ** cnt53)
        if not silent:
            print('Correted for divide by 4 x {}'.format(cnt53))
    if cnt82 > 0:
        data = data * (2.0 ** cnt82)
        if not silent:
            print('Correted for divide by 2 x {}'.format(cnt82))
    if cnt83 > 0:
        data = data * (4.0 ** cnt83)
        if not silent:
            print('Correted for divide by 4 x {}'.format(cnt83))
    if cnt84 > 0:
        data = data * (8.0 ** cnt84)
        if not silent:
            print('Correted for divide by 8 x {}'.format(cnt84))
    if cnt85 > 0:
        data = data * (16.0 ** cnt85)
        if not silent:
            print('Correted for divide by 16 x {}'.format(cnt85))
    if cnt86 > 0:
        data = data * (32.0 ** cnt86)
        if not silent:
            print('Correted for divide by 32 x {}'.format(cnt86))
    if cnt87 > 0:
        data = data * (64.0 ** cnt87)
        if not silent:
            print('Correted for divide by 64 x {}'.format(cnt87))
    if cnt88 > 0:
        data = data * (128.0 ** cnt88)
        if not silent:
            print('Correted for divide by 128 x {}'.format(cnt88))
    if cnt118 > 0:
        data = data * (3.0 ** cnt118)
        if not silent:
            print('Correted for divide by 3 x {}'.format(cnt118))

    if not silent:
        print('------------------------------------------------------')

    return data


#######################################################################################################################################

def get_smask(ftpsc, header, timehdr, calpath):
    """
    Conversion of get_smask.pro for IDL. Returns smooth mask array. Checks common block before opening mask file.
    Saves mask file to common block and re-scales the mask array for summing.

    @param ftpsc: Spacecraft (STEREO-A/STEREO-B)
    @param header: Header of .fits file
    @param timehdr: Timestamps of HI images
    @param calpath: Path for calibration files
    @return: Smooth mask array
    """
    if ftpsc == 'A':
        filename = 'hi2A_mask.fts'
        xy = [1, 51]

    if ftpsc == 'B':
        filename = 'hi2B_mask.fts'
        xy = [129, 79]

    calpath = calpath + filename
    timehdr = datetime.datetime.strptime(timehdr, '%Y-%m-%dT%H:%M:%S.%f')

    hdul_smask = fits.open(calpath)

    fullm = np.zeros((2176, 2176))

    x1 = 2048 - np.shape(fullm[xy[1] - 1:, xy[0] - 1:])[0]
    y1 = 2048 - np.shape(fullm[xy[1] - 1:, xy[0] - 1:])[1]

    if ftpsc == 'A':
        zeszc = np.shape(fullm[xy[0] - 1:y1, xy[1] - 1:x1])
        fullm[xy[0] - 1:y1, xy[1] - 1:x1] = hdul_smask[0].data
    if ftpsc == 'B':
        zeszc = np.shape(fullm[xy[0] - 1:, xy[1] - 1:x1])
        fullm[xy[0] - 1:, xy[1] - 1:x1] = hdul_smask[0].data

    tc = datetime.datetime(2015, 5, 19)

    flag = timehdr > tc

    if flag:
        fullm = np.rot90(fullm)
        fullm = np.rot90(fullm)

    mask = rebin(fullm[header['r1row'] - 1:header['r2row'], header['r1col'] - 1:header['r2col']],
                 [header['NAXIS2'], header['NAXIS1']])
    mask = np.where(mask < 1, 0, 1)

    hdul_smask.close()

    return mask


#######################################################################################################################################


def rebin(a, shape_arr):
    """
    Reshapes array to given dimensions.

    @param a: Array to be reshaped
    @param shape_arr: Dimensions for reshaping
    @return: Reshaped array
    """
    sh = int(shape_arr[0]), int(a.shape[0]) // int(shape_arr[0]), int(shape_arr[1]), int(a.shape[1]) // int(shape_arr[1])
    return a.reshape(sh).mean(-1).mean(1)


#######################################################################################################################################

#@numba.njit()
def hi_desmear(data, header_int, header_flt, header_str):
    """
    Conversion of hi_desmear.pro for IDL. Removes smear caused by no shutter. First compute the effective exposure time
    [time the ccd was static, generate matrix with this in diagonal, the clear time below and the read time above;
    invert and multiply by the image.

    @param data: Data of .fits file
    @param header_int: Array containing dstart, dstop, naxis, n_images and a flag for post-conjecture
    derived from the .fits header
    @param header_flt: Array containing exptime, cleartim, ro_delay, ipsum, line_ro, line_clr derived from .fits header
    @param header_str: Array containing rectify, obsrvtry derived from .fits header
    @return: Array corrected for shutterless camera
    """

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
    fixup = np.ascontiguousarray(fixup)

    image = np.ascontiguousarray(image)

    image = fixup @ image

    if dstart1 < 1 or (naxis1 == naxis2):
        img = image

    else:
        img = image[dstart2 - 1:dstop1, dstart1 - 1:dstop1]

    return img


#######################################################################################################################################


def get_calimg(instr, ftpsc, header, calpath):
    """
    Conversion of get_calimg.pro for IDL. Returns calibration correction array. Checks common block before opening
    calibration file. Saves calibration file to common block. Trims calibration array for under/over scan.
    Re-scales calibration array for summing.

    @param instr: STEREO-HI instrument (HI-1/HI-2)
    @param ftpsc: Spacecraft (STEREO-A/STEREO-B)
    @param header: Header of .fits file
    @param calpath: Path to calibration files
    @return: Array to correct for calibration
    """
    if instr == 'hi_1':

        if header['summed'] == 1:
            cal_version = '20061129_flatfld_raw_h1' + ftpsc.lower() + '.fts'
            sumflg = 0
        else:
            cal_version = '20100421_flatfld_sum_h1' + ftpsc.lower() + '.fts'
            sumflg = 1

    if instr == 'hi_2':

        if header['summed'] == 1:
            cal_version = '20150701_flatfld_raw_h2' + ftpsc.lower() + '.fts'
            sumflg = 0
        else:
            cal_version = '20150701_flatfld_sum_h2' + ftpsc.lower() + '.fts'
            sumflg = 1

    calpath = calpath + cal_version

    hdul_cal = fits.open(calpath)

    try:
        cal_p1col = hdul_cal[0].header['P1COL']

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

        cal = hdul_cal[0].data[y1:y2 + 1, x1:x2 + 1]

    else:
        cal = hdul_cal[0].data

    timehdr = hdul_cal[0].header['DATE']

    timehdr = datetime.datetime.strptime(timehdr, '%Y-%m-%d')

    if (header['RECTIFY'] == 'T') and (hdul_cal[0].header['RECTIFY'] == 'F'):
        cal = secchi_rectify(cal, hdul_cal[0].header, ftpsc, sumflg)

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

    flag_hdr = timehdr > tc

    if flag_hdr:
        cal = np.rot90(cal)
        cal = np.rot90(cal)

    hdul_cal.close()

    return cal


#######################################################################################################################################

def get_calfac(header, timehdr):
    """
    Conversion of get_calfac.pro for IDL. Returns calibration factor for a given image.
    If the images was SEB IP summed then the program corrects the calibration factor.

    @param header: Header of .fits file
    @param timehdr: Timestamps of images
    @return: Calibration factor for a given image
    """
    if header['DETECTOR'] == 'HI1':

        if header['OBSRVTRY'] == 'STEREO_A':
            years = (timehdr - datetime.datetime(2009, 1, 1)).total_seconds() / 3600 / 24 / 365.25
            calfac = 3.63e-13
            annualchange = 0.000910

        if header['OBSRVTRY'] == 'STEREO_B':
            years = (timehdr - datetime.datetime(2007, 1, 1)).total_seconds() / 3600 / 24 / 365.25
            calfac = 3.55e-13
            annualchange = 0.001503

        if years < 0:
            years = 0

        calfac = calfac / (1 - annualchange * years)

    if header['DETECTOR'] == 'HI2':

        years = (timehdr - datetime.datetime(2000, 12, 31)).total_seconds() / 3600 / 24 / 365.25

        if header['OBSRVTRY'] == 'STEREO_A':
            calfac = 4.411e-14 + 7.099e-17 * years

        if header['OBSRVTRY'] == 'STEREO_B':
            calfac = 4.293e-14 + 3.014e-17 * years

    header['CALFAC'] = calfac

    if header['IPSUM'] > 1 and calfac != 1.0:
        divfactor = (2 ** (header['IPSUM'] - 1)) ** 2
        sumcount = header['IPSUM'] - 1
        header['IPSUM'] = 1

        calfac = calfac / divfactor

    if (header['POLAR'] == 1001) and (header['SEB_PROG'] != 'DOUBLE'):
        calfac = 2 * calfac

    return calfac


#######################################################################################################################################

def scc_hi_diffuse(header, ipsum):
    """
    Conversion of scc_hi_diffuse.pro for IDL. Compute correction for diffuse sources arrising from changes
    in the solid angle in the optics. In the mapping of the optics the area of sky viewed is not equal off axis.

    @param header: Header of .fits file
    @param ipsum: Allows override of header ipsum value for use in L1 and beyond images
    @return: Correction factor for given image
    """
    summing = 2 ** (ipsum - 1)

    # if header['ravg'] > 0:
    #  mu = header['pv2_1']
    #  cdelt = header['cdelt1']*np.pi/180

    if header['detector'] == 'HI1':

        if header['OBSRVTRY'] == 'STEREO_A':
            mu = 0.102422
            cdelt = 35.96382 / 3600 * np.pi / 180 * summing

        if header['OBSRVTRY'] == 'STEREO_B':
            mu = 0.095092
            cdelt = 35.89977 / 3600 * np.pi / 180 * summing

    if header['detector'] == 'HI2':

        if header['OBSRVTRY'] == 'STEREO_A':
            mu = 0.785486
            cdelt = 130.03175 / 3600 * np.pi / 180 * summing

        if header['OBSRVTRY'] == 'STEREO_B':
            mu = 0.68886
            cdelt = 129.80319 / 3600 * np.pi / 180 * summing

    pixelSize = 0.0135 * summing
    fp = pixelSize / cdelt

    x = np.arange(header['naxis1']) - header['crpix1'] + header['dstart1']
    x = np.array([x for i in range(header['naxis1'])])

    y = np.arange(header['naxis2']) - header['crpix2'] + header['dstart2']
    y = np.transpose(y)
    y = np.array([y for i in range(header['naxis1'])])

    r = np.sqrt(x * x + y * y) * pixelSize

    gamma = fp * (mu + 1.0) / r
    cosalpha1 = (-1.0 * mu + gamma * np.sqrt(1.0 - mu * mu + gamma * gamma)) / (1.0 + gamma * gamma)

    correct = ((mu + 1.0) ** 2 * (mu * cosalpha1 + 1.0)) / ((mu + cosalpha1) ** 3)

    return correct


#######################################################################################################################################

def secchi_rectify(cal, scch, ftpsc, flg):
    """
    Conversion of secchi_rectify.pro for IDL. Function procedure to rectify the CCD image,
    put solar north to the top of the image. Rotates an image so that ecliptic north is up and modifies coordinate
    keywords accordingly.

    @param cal: Flatfield for given image
    @param scch: Header of flatfield
    @param ftpsc: STEREO-HI instrument (HI-1/HI-2)
    @param flg: Flag indicating flatfield to use
    @return: Rectified image
    """
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

    if (ftpsc == 'A') and flg == 1:
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

    if (ftpsc == 'A') and flg == 0:
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
    """
    Conversion of get_biasmean.pro for IDL. Returns mean bias for a give image.

    @param header: Header of .fits file
    @return: Bias to be subtracted from the image
    """
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
    """
    Conversion of fill_missing.pro for IDL. Set missing block values sensibly.

    @param data: Data from .fits file
    @param header:Header of .fits file
    @return: Corrected image
    """
    if header['NMISSING'] == 0:
        data = data

    if header['NMISSING'] > 0:

        if len(header['MISSLIST']) < 1:
            print('Mismatch between nmissing and misslist.')
            data = data

        else:
            fields = scc_get_missing(header)
            data[fields] = np.nanmedian(data)

    header['bunit'] = 'DN/s'

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

    print('This function has not been properly implemented')

    return np.array([])
#######################################################################################################################################

def scc_img_trim(im, header):
    """
    Conversion of scc_img_trim.pro for IDL. Returns rectified images with under/over scan areas removed.
    The program returns the imaging area of the CCD. If the image has not been rectified such that ecliptic north
    is up then the image is rectified.

    @param im: Selected image
    @param header: Header of .fits file
    @return: Rectified image with under-/overscan removed
    """
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

# @numba.njit()
def cadence_corr(tdiff, cadence):
    """
    Accounts for missing images in STEREO-HI data. If the gap between images is larger than the cadence,
    a block of nan is appended to the data. Further correction of missing images is done directly
    during creation of the Jplot.

    @param tdiff: Array of time differences between subsequent images
    @param cadence: Cadence of respective instrument
    @return:Blocks of nan data and indices at which to insert them.
    """
    nan_ind = []
    nan_dat = []

    # time difference is divided by cadence and rounded
    # if the time difference is more than one standard cadence, np.nan must be appended to account for time gaps

    for i in range(len(tdiff)):

        ap_ind = np.round(tdiff[i] / cadence, 0)
        ap_ind = int(ap_ind)

        if ap_ind > 1:
            nan_ind.append(int(i))
            nan_dat.append(int(ap_ind)-1)

    return nan_dat, nan_ind


#######################################################################################################################################
def create_rdif(time_obj, maxgap, cadence, data, hdul, wcoord, bflag, ins):
    """
    Creates running difference images from reduced HI data.
    The preceding image is subtracted from an image if their times are within a certain maximum time gap.
    For science data, processing images into running difference images also removes the starfield.

    @param time_obj: Array of timestamps of images
    @param maxgap: Maximum gap between images
    @param cadence: Cadence of HI instrument
    @param data: Image data
    @param hdul: Header of .fits file
    @param wcoord: World coordinate system extracted from .fits header
    @param bflag: Science or beacon images
    @param ins: HI instrument (HI-1/HI-2)
    @return: Array of running difference images and set of indices where there are time gaps
    """

    # Array of np.nan shaped like image data to replace missing timesteps
    nandata = np.full_like(data[0], np.nan)

    r_dif = []

    data = np.float32(data)

    # Define kernel size fo median filtering
    if bflag == 'science':

        if ins == 'hi_1':
            kernel = 5

        if ins == 'hi_2':
            kernel = 3

    if bflag == 'beacon':
        kernel = 3

    indices = []
    ims = []

    for i in range(1, len(data)):

        # Get center value for preceding and following image
        crval = [hdul[i - 1][0].header['crval1a'], hdul[i - 1][0].header['crval2a']]

        center = [hdul[0][0].header['crpix1'] - 1, hdul[0][0].header['crpix2'] - 1]

        center2 = wcoord[i].all_world2pix(crval[0], crval[1], 0)

        # Determine shift between preceding and following image
        xshift = center2[0] - center[0]
        yshift = center2[1] - center[1]

        shiftarr = [-yshift, xshift]

        # Shift image by calculated shift and interpolate using spline itnerpolation
        ims.append(shift(data[i - 1], shiftarr, mode='wrap'))

        # produce regular running difference images if time difference is within maxgap * cadence

        j_ind = []

        # time difference is taken for every time object, not just the preceding one
        for j in range(len(data) - 1):
            time_i_j = (time_obj[i] - time_obj[i - j]).sec / 60

            # indices of time objects that fit the criteria are appended to the list
            # criteria: time difference smaller than -maxgap * cadence and time difference larger than cadence
            if (np.round(time_i_j) <= -maxgap * cadence) & (np.round(time_i_j) >= cadence):# & (np.mod(np.round(time_i_j), cadence) == 0):

                j_ind.append(i - j)

        # if no adequate preceding image is found, append array of np.nan to the running difference list
        # criteria: time diffference is larger than -maxgap * cadence or time difference is smaller than cadence

        if np.round((time_obj[i] - time_obj[i - 1]).sec / 60) > -maxgap * cadence:
            # indices.append(i)
            r_dif.append(nandata)

        if np.round((time_obj[i] - time_obj[i - 1]).sec / 60) < cadence:
            r_dif.append(nandata)

        # for appropriate time differences, create running differene images and apply median filter
        if len(j_ind) >= 1:
            j = j_ind[0]

            indices.append(i)

            if bflag == 'science':
                ndat = np.float32(data[i] - ims[j])
                ndat = cv2.medianBlur(ndat, kernel)
                r_dif.append(cv2.medianBlur(ndat, kernel))

            if bflag == 'beacon':
                ndat = np.float32(data[i] - ims[j])
                r_dif.append(ndat)

    return r_dif, indices


#######################################################################################################################################

def get_map_xrange(hdul):
    """
    Conversion of get_map_xrange.pro for IDL. Extract min/max X-coordinate of map. Coordinates correspond to the pixel center.

    @param hdul: Header of .fits file
    @return: x-axis extent, x-axis center, x-axis width of pixel in spacecraft coordinates as an array
    """
    nx = [hdul[i][0].header['NAXIS1'] for i in range(len(hdul))]
    xc = [hdul[i][0].header['CRVAL1'] for i in range(len(hdul))]
    dx = [hdul[i][0].header['CDELT1'] for i in range(len(hdul))]

    # xmin = np.nanmin(xc - dx * (nx - 1)/2)
    # xmax = np.nanmax(xc + dx * (nx - 1)/2)

    # x_range = [xmin, xmax]

    return nx, xc, dx


#######################################################################################################################################

def get_map_yrange(hdul):
    """
    Conversion of get_map_yrange.pro for IDL. Extract min/max Y-coordinate of map. Coordinates correspond to the pixel center.

    @param hdul: Header of .fits file
    @return: y-axis extent, y-axis center, y-axis width of pixel in spacecraft coordinates as an array
    """
    ny = [hdul[i][0].header['NAXIS2'] for i in range(len(hdul))]
    yc = [hdul[i][0].header['CRVAL2'] for i in range(len(hdul))]
    dy = [hdul[i][0].header['CDELT2'] for i in range(len(hdul))]

    # xmin = np.nanmin(xc - dx * (nx - 1)/2)
    # xmax = np.nanmax(xc + dx * (nx - 1)/2)

    # x_range = [xmin, xmax]

    return ny, yc, dy

#######################################################################################################################################
def running_difference(start, path, datpath, ftpsc, instrument, bflag, silent, save_img):
    """
    Creates running difference images from reduced STEREO-HI data and saves them to the specified location.

    @param start: First date (DDMMYYYY) for which to create running difference images
    @param path: The path where all reduced images, running difference images and J-Maps are saved
    @param datpath: Path to STEREO-HI calibration files
    @param ftpsc: Spacecraft (A/B)
    @param instrument: STEREO-HI instrument (HI-1/HI-2)
    @param bflag: Science or beacon data
    @param silent: Run in silent mode
    @param save_img: Save running difference images as .pngs
    """
    if not silent:
        print('-------------------')
        print('RUNNING DIFFERENCE')
        print('-------------------')

    # Initialize date as datetime, get day before start date as datetime
    date = datetime.datetime.strptime(start, '%Y%m%d')
    prev_date = date - datetime.timedelta(days=1)
    prev_date = datetime.datetime.strftime(prev_date, '%Y%m%d')

    # Get paths to files one day before start time
    prev_path_h1 = path + 'reduced/data/' + ftpsc + '/' + prev_date + '/' + bflag + '/hi_1/'
    prev_path_h2 = path + 'reduced/data/' + ftpsc + '/' + prev_date + '/' + bflag + '/hi_2/'

    files_h1 = []
    files_h2 = []

    # Append files from day before start to list
    # If no files exist, start running difference images with first file of chosen start date

    if os.path.exists(prev_path_h1):

        if bflag == 'science':

            for file in sorted(glob.glob(prev_path_h1 + '*1bh1' + '*.fts')):
                files_h1.append(file)

        if bflag == 'beacon':

            for file in sorted(glob.glob(prev_path_h1 + '*17h1' + '*.fts')):
                files_h1.append(file)

        try:
            files_h1 = [files_h1[-1]]

        except IndexError:
            files_h1 = []

    if os.path.exists(prev_path_h2):

        if bflag == 'science':

            for file in sorted(glob.glob(prev_path_h2 + '*1bh2' + '*.fts')):
                files_h2.append(file)

        if bflag == 'beacon':

            for file in sorted(glob.glob(prev_path_h2 + '*17h2' + '*.fts')):
                files_h2.append(file)

        try:
            files_h2 = [files_h2[-1]]

        except IndexError:
            files_h2 = []

    calpath = datpath + 'calibration/'

    # tc = datetime.datetime(2015, 7, 1)

    # cadence of instruments in minutes
    if bflag == 'beacon':
        cadence_h1 = 120.0
        cadence_h2 = 120.0

    if bflag == 'science':
        cadence_h1 = 40.0
        cadence_h2 = 120.0

    # define maximum gap between consecutive images
    # if gap > maxgap, no running difference image is produced, timestep is filled with np.nan instead

    maxgap = -3.5

    redpath_h1 = path + 'reduced/data/' + ftpsc + '/' + start + '/' + bflag + '/hi_1/'
    redpath_h2 = path + 'reduced/data/' + ftpsc + '/' + start + '/' + bflag + '/hi_2/'

    # read in .fits files

    if not silent:
        print('Getting files...')

    if bflag == 'science':

        for file in sorted(glob.glob(redpath_h1 + '*.fts')):
            files_h1.append(file)

        for file in sorted(glob.glob(redpath_h2 + '*.fts')):
            files_h2.append(file)

    if bflag == 'beacon':

        for file in sorted(glob.glob(redpath_h1 + '*.fts')):
            files_h1.append(file)

        for file in sorted(glob.glob(redpath_h2 + '*.fts')):
            files_h2.append(file)

    # get times and headers from .fits files

    hdul_h1 = [fits.open(files_h1[i]) for i in range(len(files_h1))]
    tcomp1 = datetime.datetime.strptime(hdul_h1[0][0].header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
    time_h1 = [hdul_h1[i][0].header['DATE-OBS'] for i in range(len(hdul_h1))]
    wcoord_h1 = [wcs.WCS(files_h1[i], key='A') for i in range(len(files_h1))]

    hdul_h2 = [fits.open(files_h2[i]) for i in range(len(files_h2))]
    tcomp2 = datetime.datetime.strptime(hdul_h2[0][0].header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
    time_h2 = [hdul_h2[i][0].header['DATE-OBS'] for i in range(len(hdul_h2))]
    wcoord_h2 = [wcs.WCS(files_h2[i], key='A') for i in range(len(files_h2))]

    naxis_x1, xcenter1, dx1 = get_map_xrange(hdul_h1)
    naxis_x2, xcenter2, dx2 = get_map_xrange(hdul_h2)

    naxis_y1, ycenter1, dy1 = get_map_yrange(hdul_h1)
    naxis_y2, ycenter2, dy2 = get_map_yrange(hdul_h2)

    if not silent: 
        print('Reading data...')

    # times are converted to objects

    time_obj_h1 = [Time(time_h1[i], format='isot', scale='utc') for i in range(len(time_h1))]

    time_obj_h2 = [Time(time_h2[i], format='isot', scale='utc') for i in range(len(time_h2))]

    data_h1 = np.array([hdul_h1[i][0].data for i in range(len(files_h1))])
    data_h2 = np.array([hdul_h2[i][0].data for i in range(len(files_h2))])

    # Create .pngs of reduced images with background

    save_withbg = False

    if save_withbg:

        if not silent:
            print('Saving images with background as jpeg...')

        savepath_h1 = path + 'running_difference/pngs/' + ftpsc + '/' + start + '/' + bflag + '/hi_1/'
        savepath_h2 = path + 'running_difference/pngs/' + ftpsc + '/' + start + '/' + bflag + '/hi_2/'

        if not os.path.exists(savepath_h1):
            os.makedirs(savepath_h1)
            subprocess.call(['chmod', '-R', '775', path + 'running_difference/pngs/'])

        if not os.path.exists(savepath_h2):
            os.makedirs(savepath_h2)
            subprocess.call(['chmod', '-R', '775', path + 'running_difference/pngs/'])

        names_h1 = [files_h1[i].rpartition('/')[2][0:21] for i in range(0, len(files_h1))]
        names_h2 = [files_h2[i].rpartition('/')[2][0:21] for i in range(0, len(files_h2))]

        for i in range(len(data_h1)):
            fig, ax = plt.subplots(frameon=False)
            fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            ax.imshow(data_h1[i], cmap='gray', aspect='auto', origin='lower')

            plt.savefig(savepath_h1 + names_h1[i] + '_withbg.jpeg')
            plt.close()

        for i in range(len(data_h2)):
            fig, ax = plt.subplots(frameon=False)
            fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            ax.imshow(data_h2[i], cmap='gray', aspect='auto', origin='lower')

            plt.savefig(savepath_h2 + names_h2[i] + '_withbg.jpeg')
            plt.close()

    # Subtract coronal background from images

    min_arr_h1 = np.quantile(data_h1, 0.05, axis=0)
    data_h1 = data_h1 - min_arr_h1
    min_arr_h2 = np.nanmin(data_h2, axis=0)
    data_h2 = data_h2 - min_arr_h2

    data_h1 = np.where(np.isnan(data_h1), np.nanmedian(data_h1), data_h1)
    data_h2 = np.where(np.isnan(data_h2), np.nanmedian(data_h2), data_h2)

    # Save reduced images as .pngs without background
    save_nobg = False

    if save_nobg:

        if not silent:
            print('Saving images without background as jpeg...')

        savepath_h1 = path + 'running_difference/pngs/' + ftpsc + '/' + start + '/' + bflag + '/hi_1/'
        savepath_h2 = path + 'running_difference/pngs/' + ftpsc + '/' + start + '/' + bflag + '/hi_2/'

        if not os.path.exists(savepath_h1):
            os.makedirs(savepath_h1)
            subprocess.call(['chmod', '-R', '775', path + 'running_difference/pngs/'])

        if not os.path.exists(savepath_h2):
            os.makedirs(savepath_h2)
            subprocess.call(['chmod', '-R', '775', path + 'running_difference/pngs/'])

        names_h1 = [files_h1[i].rpartition('/')[2][0:21] for i in range(0, len(files_h1))]
        names_h2 = [files_h2[i].rpartition('/')[2][0:21] for i in range(0, len(files_h2))]

        for i in range(len(data_h1)):

            filt_data_h1 = np.float32(data_h1[i])
            fig, ax = plt.subplots(frameon=False)
            fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            ax.imshow(filt_data_h1, cmap='afmhot', vmin=-1e-13, vmax=6e-13, origin='lower')

            plt.savefig(savepath_h1 + names_h1[i] + '_nobg.jpeg')
            plt.close()

        for i in range(len(data_h2)):
            filt_data_h2 = np.float32(data_h2[i])
            fig, ax = plt.subplots(frameon=False)
            fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            ax.imshow(filt_data_h2, cmap='afmhot', vmin=-1e-13, vmax=6e-13, origin='lower')

            plt.savefig(savepath_h2 + names_h2[i] + '_nobg.jpeg')
            plt.close()

    if not silent:
        print('Replacing missing values...')

    # missing data is identified from header

    missing_h1 = np.array([hdul_h1[i][0].header['NMISSING'] for i in range(len(hdul_h1))])
    mis_h1 = np.array(np.where(missing_h1 > 0))
    exp1 = np.array([hdul_h1[i][0].header['EXPTIME'] for i in range(len(hdul_h1))])

    missing_h2 = np.array([hdul_h2[i][0].header['NMISSING'] for i in range(len(hdul_h2))])
    mis_h2 = np.array(np.where(missing_h2 > 0))
    exp2 = np.array([hdul_h2[i][0].header['EXPTIME'] for i in range(len(hdul_h2))])

    # missing data is replaced with nans

    if np.size(mis_h1) > 0:
        for i in mis_h1:
            data_h1[i, :, :] = np.nanmedian(data_h1[i, :, :])

    if np.size(mis_h2) > 0:
        for i in mis_h2:
            data_h2[i, :, :] = np.nanmedian(data_h2[i, :, :])

    # Masked data is replaced with image median

    mask = np.array([get_smask(ftpsc, hdul_h2[i][0].header, time_h2[i], calpath) for i in range(0, len(data_h2))])

    data_h2 = np.array(data_h2)

    data_h2[mask == 0] = np.nanmedian(data_h2)

    top_hat = False
    if top_hat == True:
        # Getting the kernel to be used in Top-Hat
        filterSize =(10, 10)
        kernel_h1 = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        filterSize =(5, 5)
        kernel_h2 = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)

                
        # Applying the Top-Hat operation
        tophat_img_h1 = np.array([cv2.morphologyEx(data_h1[i], cv2.MORPH_TOPHAT,kernel_h1) for i in range(len(data_h1))])
        tophat_img_h2 = np.array([cv2.morphologyEx(data_h2[i], cv2.MORPH_TOPHAT,kernel_h2) for i in range(len(data_h2))])

        data_h1 = data_h1 - tophat_img_h1
        data_h2 = data_h2 - tophat_img_h2

    if not silent:
        print('Creating running difference images...')

    # Creation of running difference images

    r_dif_h1, ind_h1 = create_rdif(time_obj_h1, maxgap, cadence_h1, data_h1, hdul_h1, wcoord_h1, bflag, 'hi_1')
    r_dif_h1 = np.array(r_dif_h1)

    r_dif_h2, ind_h2 = create_rdif(time_obj_h2, maxgap, cadence_h2, data_h2, hdul_h2, wcoord_h2, bflag, 'hi_2')
    r_dif_h2 = np.array(r_dif_h2)

    if bflag == 'science':
        vmin_h1 = -1e-13
        vmax_h1 = 1e-13

        vmin_h2 = -1e-13
        vmax_h2 = 1e-13

    if bflag == 'beacon':
        vmin_h1 = np.nanmedian(r_dif_h1) - np.std(r_dif_h1)
        vmax_h1 = np.nanmedian(r_dif_h1) + np.std(r_dif_h1)

        vmin_h2 = np.nanmedian(r_dif_h2) - np.std(r_dif_h2)
        vmax_h2 = np.nanmedian(r_dif_h2) + np.std(r_dif_h2)

    if save_img:

        val_high_h1 = [np.quantile(r_dif_h1[i], 0.9994) for i in range(len(r_dif_h1))]
        val_low_h1 = [np.quantile(r_dif_h1[i], 0.0005) for i in range(len(r_dif_h1))]

        val_high_h2 = [np.quantile(r_dif_h2[i], 0.99) for i in range(len(r_dif_h2))]
        val_low_h2 = [np.quantile(r_dif_h2[i], 0.01) for i in range(len(r_dif_h2))]

        r_dif_h1_new = [np.where(r_dif_h1[i] > val_high_h1[i], np.nanmedian(r_dif_h1[i]), r_dif_h1[i]) for i in
                        range(len(r_dif_h1))]
        r_dif_h1_new = [np.where(r_dif_h1_new[i] < val_low_h1[i], np.nanmedian(r_dif_h1_new[i]), r_dif_h1_new[i]) for i in
                        range(len(r_dif_h1))]

        r_dif_h2_new = [np.where(r_dif_h2[i] > val_high_h2[i], np.nanmedian(r_dif_h2[i]), r_dif_h2[i]) for i in
                        range(len(r_dif_h2))]
        r_dif_h2_new = [np.where(r_dif_h2_new[i] < val_low_h2[i], np.nanmedian(r_dif_h2_new[i]), r_dif_h2_new[i]) for i in
                        range(len(r_dif_h2))]

        fac = 4e-1

        if not silent:
            print('Saving running difference images as png...')

        names_h1 = [files_h1[i].rpartition('/')[2][0:21] for i in ind_h1]
        names_h2 = [files_h2[i].rpartition('/')[2][0:21] for i in ind_h2]

        savepath_h1 = path + 'running_difference/pngs/' + ftpsc + '/' + start + '/' + bflag + '/hi_1/'
        savepath_h2 = path + 'running_difference/pngs/' + ftpsc + '/' + start + '/' + bflag + '/hi_2/'

        if not os.path.exists(savepath_h1):
            os.makedirs(savepath_h1)
            subprocess.call(['chmod', '-R', '775', path + 'running_difference/'])

        if not os.path.exists(savepath_h2):
            os.makedirs(savepath_h2)
            subprocess.call(['chmod', '-R', '775', path + 'running_difference/'])

        for i in range(len(r_dif_h1_new)):

            fig, ax = plt.subplots(figsize=(1.024, 1.024), dpi=100, frameon=False)
            fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            ax.imshow(r_dif_h1_new[i], cmap='gray', vmin=vmin_h1 * fac, vmax=vmax_h1 * fac, aspect='auto', origin='lower')

            plt.savefig(savepath_h1 + names_h1[i] + '.png', dpi=1000)
            plt.close()

        for i in range(len(r_dif_h2_new)):

            fig, ax = plt.subplots(figsize=(1.024, 1.024), dpi=100, frameon=False)
            fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            ax.imshow(r_dif_h2_new[i], cmap='gray', vmin=vmin_h2 * fac, vmax=vmax_h2 * fac, aspect='auto', origin='lower')

            plt.savefig(savepath_h2 + names_h2[i] + '.png', dpi=1000)
            plt.close()

    if not silent:
        print('Saving image as pickle...')

    savepath_h1 = path + 'running_difference/data/' + ftpsc + '/' + start + '/' + bflag + '/hi_1/'
    savepath_h2 = path + 'running_difference/data/' + ftpsc + '/' + start + '/' + bflag + '/hi_2/'

    names_h1 = [files_h1[i].rpartition('/')[2][0:21] for i in ind_h1]
    names_h2 = [files_h2[i].rpartition('/')[2][0:21] for i in ind_h2]

    if not os.path.exists(savepath_h1):
        os.makedirs(savepath_h1)
        subprocess.call(['chmod', '-R', '775', path + 'running_difference/data/'])

    if not os.path.exists(savepath_h2):
        os.makedirs(savepath_h2)
        subprocess.call(['chmod', '-R', '775', path + 'running_difference/data/'])

    time_h1 = [time_h1[i] for i in ind_h1]
    dx1 = [dx1[i] for i in ind_h1]
    xcenter1 = [xcenter1[i] for i in ind_h1]
    naxis_x1 = [naxis_x1[i] for i in ind_h1]
    r_dif_h1 = [r_dif_h1[i] for i in np.array(ind_h1)-1]

    time_h2 = [time_h2[i] for i in ind_h2]
    dx2 = [dx2[i] for i in ind_h2]
    xcenter2 = [xcenter2[i] for i in ind_h2]
    naxis_x2 = [naxis_x2[i] for i in ind_h2]
    r_dif_h2 = [r_dif_h2[i] for i in np.array(ind_h2)-1]

    dy1 = [dy1[i] for i in ind_h1]
    ycenter1 = [ycenter1[i] for i in ind_h1]
    naxis_y1 = [naxis_y1[i] for i in ind_h1]

    dy2 = [dy2[i] for i in ind_h2]
    ycenter2 = [ycenter2[i] for i in ind_h2]
    naxis_y2 = [naxis_y2[i] for i in ind_h2]

    for i in range(len(r_dif_h1)):
        r_dif_h1_data = {'data': r_dif_h1[i], 'time': time_h1[i], 'dx': dx1[i], 'xcenter': xcenter1[i], 'naxis_x': naxis_x1[i],
                         'dy': dy1[i], 'ycenter': ycenter1[i], 'naxis_y': naxis_y1[i]}
        a_file = open(savepath_h1 + names_h1[i] + '.pkl', 'wb')
        pickle.dump(r_dif_h1_data, a_file)
        a_file.close()

    for i in range(len(r_dif_h2)):
        r_dif_h2_data = {'data': r_dif_h2[i], 'time': time_h2[i], 'dx': dx2[i], 'xcenter': xcenter2[i], 'naxis_x': naxis_x2[i],
                         'dy': dy2[i], 'ycenter': ycenter2[i], 'naxis_y': naxis_y2[i]}
        a_file = open(savepath_h2 + names_h2[i] + '.pkl', 'wb')
        pickle.dump(r_dif_h2_data, a_file)
        a_file.close()

#######################################################################################################################################
def make_jplot(start, duration, path, datpath, ftpsc, instrument, bflag, save_path, path_flg, silent):
    """
    Creates Jplot from running difference images. Method similar to create_jplot_tam.pro written in IDL by Tanja Amerstorfer.
    Middle slice of each running difference is cut out, strips are aligned, time-gaps are filled with nan.

    @param start: Start date of Jplot
    @param duration: Length of Jplot (in days)
    @param path: The path where all reduced images, running difference images and J-Maps are saved
    @param datpath: Path to STEREO-HI calibration files
    @param ftpsc: Spacecraft (A/B)
    @param instrument: STEREO-HI instrument (HI-1/HI-2)
    @param bflag: Science or beacon data
    @param save_path: Path pointing towards downloaded STEREO .fits files
    @param path_flg: Specifies path for downloaded files, depending on wether science or beacon data is used
    @param silent: Run in silent mode
    """
    if not silent:
        print('-------------------')
        print('JPLOT')
        print('-------------------')

    date = datetime.datetime.strptime(start, '%Y%m%d')

    interv = np.arange(duration)

    datelst = [datetime.datetime.strftime(date + datetime.timedelta(days=int(interv[i])), '%Y%m%d') for i in interv]

    savepaths_h1 = [path + 'running_difference/data/' + ftpsc + '/' + datelst[i] + '/' + bflag + '/hi_1/' for i in interv]
    savepaths_h2 = [path + 'running_difference/data/' + ftpsc + '/' + datelst[i] + '/' + bflag + '/hi_2/' for i in interv]

    files_h1 = []
    files_h2 = []

    for savepath in savepaths_h1:

        for file in sorted(glob.glob(savepath + '*.pkl')):
            files_h1.append(file)

    for savepath in savepaths_h2:

        for file in sorted(glob.glob(savepath + '*.pkl')):
            files_h2.append(file)

    if bflag == 'beacon':
        cadence_h1 = 120.0
        cadence_h2 = 120.0

    if bflag == 'science':
        cadence_h1 = 40.0
        cadence_h2 = 120.0

    # define maximum gap between consecutive images
    # if gap > maxgap, no running difference image is produced, timestep is filled with np.nan instead

    maxgap = 3.5

    if not silent:
        print('Getting data...')

    array_h1 = []

    for i in range(len(files_h1)):
        with open(files_h1[i], 'rb') as f:
            array_h1.append(pickle.load(f))

    array_h2 = []

    for i in range(len(files_h2)):
        with open(files_h2[i], 'rb') as f:
            array_h2.append(pickle.load(f))

    array_h1 = np.array(array_h1)
    array_h2 = np.array(array_h2)

    time_h1 = [array_h1[i]['time'] for i in range(len(array_h1))]
    time_obj_h1 = [Time(time_h1[i], format='isot', scale='utc') for i in range(len(time_h1))]
    tdiff_h1 = np.array([(time_obj_h1[i] - time_obj_h1[i - 1]).sec / 60 for i in range(1, len(time_obj_h1))])
    tcomp1 = datetime.datetime.strptime(array_h1[0]['time'], '%Y-%m-%dT%H:%M:%S.%f')

    time_h2 = [array_h2[i]['time'] for i in range(len(array_h2))]
    time_obj_h2 = [Time(time_h2[i], format='isot', scale='utc') for i in range(len(time_h2))]
    tdiff_h2 = np.array([(time_obj_h2[i] - time_obj_h2[i - 1]).sec / 60 for i in range(1, len(time_obj_h2))])

    tcomp2 = datetime.datetime.strptime(array_h2[0]['time'], '%Y-%m-%dT%H:%M:%S.%f')

    tc = datetime.datetime(2015, 7, 1)

    # Determine if STEREO spacecraft are pre- or post-conjunction

    if ftpsc == 'A':
        sc = 'ahead'

    if ftpsc == 'B':
        sc = 'behind'

    fitsfiles = []

    if bflag == 'science':

        for file in sorted(glob.glob(save_path + 'stereo' + sc[0] + '/secchi/' + path_flg + '/img/hi_1/' + str(start) + '/*s4*.fts')):
            fitsfiles.append(file)

    if bflag == 'beacon':

        for file in sorted(glob.glob(save_path + 'stereo' + sc[0] + '/' + path_flg + '/secchi/' + '/img/hi_1/' + str(start) + '/*s7*.fts')):
            fitsfiles.append(file)

    hdul = [fits.open(fitsfiles[i]) for i in range(len(fitsfiles))]

    crval1 = [hdul[i][0].header['crval1'] for i in range(len(fitsfiles))]

    if ftpsc == 'A':    
        post_conj = [int(np.sign(crval1[i])) for i in range(len(crval1))]

    if ftpsc == 'B':    
        post_conj = [int(-1*np.sign(crval1[i])) for i in range(len(crval1))]

    if len(set(post_conj)) == 1:
        post_conj = post_conj[0]

        if post_conj == -1:
            post_conj = False
        if post_conj == 1:
            post_conj = True

    else:
        print('Invalid dates. Exiting...')
        sys.exit()

    # save height and width of h1 and h2 data for later

    r_dif_h1 = np.array([array_h1[i]['data'] for i in range(len(array_h1))])
    w_h1 = np.shape(r_dif_h1[0])[0]
    h_h1 = np.shape(r_dif_h1[0])[1]

    r_dif_h2 = np.array([array_h2[i]['data'] for i in range(len(array_h2))])
    w_h2 = np.shape(r_dif_h2[0])[0]
    h_h2 = np.shape(r_dif_h2[0])[1]

    if not silent:
        print('Making ecliptic cut...')

    # Choose widths for ecliptic cut

    if bflag == 'science':
        pix = 16

    if bflag == 'beacon':
        pix = 8

    # Cut out pre-defined strip from ecliptic

    dif_cut_h1 = np.array([r_dif_h1[i, int(w_h1 / 2 - pix):int(w_h1 / 2 + pix), 0:h_h1] for i in range(len(r_dif_h1))])
    dif_cut_h2 = np.array([r_dif_h2[i, int(w_h2 / 2 - pix):int(w_h2 / 2 + pix), 0:h_h2] for i in range(len(r_dif_h2))])

    # convert time to correct format
    # matplotlib uses number of days since 0001-01-01 UTC, plus 1

    time_t_h1 = [datetime.datetime.strptime(day, '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y/%m/%d %H:%M:%S.%f') for day in time_h1]

    time_file_h1 = str(tcomp1.time())[0:8].replace(':', '')

    x_lims_h1 = mdates.datestr2num(time_t_h1)

    time_t_h2 = [datetime.datetime.strptime(day, '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y/%m/%d %H:%M:%S.%f') for day in time_h2]

    time_file_h2 = str(tcomp2.time())[0:8].replace(':', '')

    x_lims_h2 = mdates.datestr2num(time_t_h2)
    
    time_h1 = x_lims_h1
    time_h2 = x_lims_h2
    
    sh_cut_h1 = np.shape(dif_cut_h1)
    sh_cut_h2 = np.shape(dif_cut_h2)

    if not silent:
        print('Calculating elongation...')

    xcenter_h1 = np.array([array_h1[i]['xcenter'] for i in range(len(array_h1))])
    dx1 = np.nanmedian(np.array([array_h1[i]['dx'] for i in range(len(array_h1))]))
    naxis_h1 = np.nanmedian(np.array([array_h1[i]['naxis_x'] for i in range(len(array_h1))]))

    xcenter_h2 = np.array([array_h2[i]['xcenter'] for i in range(len(array_h2))])
    dx2 = np.nanmedian(np.array([array_h2[i]['dx'] for i in range(len(array_h2))]))
    naxis_h2 = np.nanmedian(np.array([array_h2[i]['naxis_x'] for i in range(len(array_h2))]))

    use_edge = 0

    # Calculate minimum and maximum elongation
    # Use median instead of minimum if files are corrupted (should only happen rarely)

    if np.nanmin(xcenter_h1) < np.nanmedian(xcenter_h1) - 1:
        elo_min_h1 = np.nanmedian(xcenter_h1) - dx1 * (naxis_h1 - 1) / 2 - use_edge * dx1/2
    else:
        elo_min_h1 = np.nanmin(xcenter_h1 - dx1 * (naxis_h1 - 1) / 2 - use_edge * dx1/2)

    elo_max_h1 = np.nanmax(xcenter_h1 + dx1 * (naxis_h1 - 1) / 2 + use_edge * dx1/2)

    if np.nanmin(xcenter_h2) < np.nanmedian(xcenter_h2) - 1:
        elo_min_h2 = np.nanmedian(xcenter_h2) - dx2 * (naxis_h2 - 1) / 2 - use_edge * dx2/2
    else:
        elo_min_h2 = np.nanmin(xcenter_h2 - dx2 * (naxis_h2 - 1) / 2 - use_edge * dx2 / 2)

    elo_max_h2 = np.nanmax(xcenter_h2 + dx2 * (naxis_h2 - 1) / 2 + use_edge * dx2/2)

    dif_med_h1 = np.zeros((sh_cut_h1[0], sh_cut_h1[2]))
    dif_med_h2 = np.zeros((sh_cut_h2[0], sh_cut_h2[2]))

    # take median of pixels cut out from h1 and h2

    for i in range(sh_cut_h1[0]):
        for j in range(sh_cut_h1[2]):
            dif_med_h1[i, j] = np.nanmedian(dif_cut_h1[i, :, j])

    for i in range(sh_cut_h2[0]):
        for j in range(sh_cut_h2[2]):
            dif_med_h2[i, j] = np.nanmedian(dif_cut_h2[i, :, j])


    elongation_h1 = np.zeros(h_h1)
    elongation_h2 = np.zeros(h_h2)


    if post_conj:
        elongation_h1[0] = elo_min_h1

        elongation_h2[0] = elo_min_h2

        for i in range(len(elongation_h1) - 1):
            elongation_h1[i + 1] = elongation_h1[i] + dx1

        for i in range(len(elongation_h2) - 1):
            elongation_h2[i + 1] = elongation_h2[i] + dx2

    if not post_conj:
        elongation_h1[0] = elo_max_h1

        elongation_h2[0] = elo_max_h2

        for i in range(len(elongation_h1) - 1):
            elongation_h1[i + 1] = elongation_h1[i] - dx1

        for i in range(len(elongation_h2) - 1):
            elongation_h2[i + 1] = elongation_h2[i] - dx2

        if ftpsc == 'A':
            elongation_h1 = -elongation_h1
            elongation_h2 = -elongation_h2

        if ftpsc == 'B':
            elongation_h1 = np.flip(elongation_h1)
            elongation_h2 = np.flip(elongation_h2)

    steps_interp = int(np.shape(dif_med_h1)[-1])

    t_interp_beg = max(np.nanmin(time_h1), np.nanmin(time_h2))
    t_interp_end = min(np.nanmax(time_h1), np.nanmax(time_h2))

    dif_med_h2_interp = np.zeros((steps_interp, steps_interp))

    time_new = np.linspace(t_interp_beg, t_interp_end, steps_interp)

    for i in range(steps_interp):
        dif_med_h2_interp[:, i] = np.interp(time_new, time_h2, dif_med_h2[:, i])

    dif_med_h1_interp = np.zeros((steps_interp, steps_interp))

    for i in range(steps_interp):
        dif_med_h1_interp[:, i] = np.interp(time_new, time_h1, dif_med_h1[:, i])

    if not post_conj:

        if ftpsc == 'A':
            tflag = 0

        if ftpsc == 'B':
            tflag = 1

    if post_conj:

        if ftpsc == 'A':
            tflag = 1

        if ftpsc == 'B':
            tflag = 0

    if tflag:
        orig = 'lower'

    if not tflag:
        orig = 'upper'
            
    jmap_h1_interp = np.array(dif_med_h1_interp).transpose()
    jmap_h2_interp = np.array(dif_med_h2_interp).transpose()
            
    # Contrast stretching
    p2, p98 = np.percentile(jmap_h1_interp, (2, 98))
    img_rescale_h1 = exposure.rescale_intensity(jmap_h1_interp, in_range=(p2, p98))

    # Contrast stretching
    p2, p98 = np.percentile(jmap_h2_interp, (2, 98))
    img_rescale_h2 = exposure.rescale_intensity(jmap_h2_interp, in_range=(p2, p98))


    for i in range(1, len(time_h1)):
        
        delta_t1 = (time_h1[i]-time_h1[i-1])*24*60
        
        if delta_t1 > maxgap*cadence_h1:
            
            time_repl_beg = np.array(mdates.num2date(time_new)) > mdates.num2date(time_h1[i-1])
            time_repl_end = np.array(mdates.num2date(time_new)) < mdates.num2date(time_h1[i])
            
            np.transpose(img_rescale_h1)[time_repl_beg & time_repl_end] = np.nan

    for i in range(1, len(time_h2)):
        
        delta_t2 = (time_h2[i]-time_h2[i-1])*24*60
        
        if delta_t2 > maxgap*cadence_h2:
            
            time_repl_beg = np.array(mdates.num2date(time_new)) > mdates.num2date(time_h2[i-1])
            time_repl_end = np.array(mdates.num2date(time_new)) < mdates.num2date(time_h2[i])
            
            np.transpose(img_rescale_h2)[time_repl_beg & time_repl_end] = np.nan

    # Save images separately and together

    vmin_h1 = np.nanmedian(img_rescale_h1) - 2 * np.nanstd(img_rescale_h1)
    vmax_h1 = np.nanmedian(img_rescale_h1) + 2 * np.nanstd(img_rescale_h1)

    vmin_h2 = np.nanmedian(img_rescale_h2) - 2 * np.nanstd(img_rescale_h2)
    vmax_h2 = np.nanmedian(img_rescale_h2) + 2 * np.nanstd(img_rescale_h2)

    savepath_h1 = path + 'jplot/' + ftpsc + '/' + bflag + '/hi_1/' + str(start[0:4]) + '/'
    savepath_h2 = path + 'jplot/' + ftpsc + '/' + bflag + '/hi_2/' + str(start[0:4]) + '/'
    savepath_h1h2 = path + 'jplot/' + ftpsc + '/' + bflag + '/' + instrument + '/' + str(start[0:4]) + '/'

    if not os.path.exists(savepath_h1):
        os.makedirs(savepath_h1)
        subprocess.call(['chmod', '-R', '775', path + 'jplot/'])

    if not os.path.exists(savepath_h2):
        os.makedirs(savepath_h2)
        subprocess.call(['chmod', '-R', '775', path + 'jplot/'])

    if not os.path.exists(savepath_h1h2):
        os.makedirs(savepath_h1h2)
        subprocess.call(['chmod', '-R', '775', path + 'jplot/'])

    t1 = time_new[0]
    t2 = time_new[-1]

    dt = (t2-t1)/(steps_interp)

    dt1 = cadence_h1 / (60 * 24)
    dt2 = cadence_h2 / (60 * 24)


    fig, ax = plt.subplots(frameon=False)
    ax.imshow(img_rescale_h1, cmap='gray', extent=[t1, t2+dt1, elongation_h1[0], elongation_h1[-1]+dx1], vmin=vmin_h1, vmax=vmax_h1, aspect='auto',
                origin=orig, interpolation='none')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.ylim(elongation_h1[0], elongation_h1[-1])
    plt.savefig(savepath_h1 + 'jplot_hi1_' + start + '_' + time_file_h1 + 'UT_' + bflag[0] + '_' + ftpsc + '.png', bbox_inches=0, pad_inches=0)

    plt.close()

    fig, ax = plt.subplots(frameon=False)
    ax.imshow(img_rescale_h2, cmap='gray', extent=[t1, t2+dt2, elongation_h2[0], elongation_h2[-1]+dx2], vmin=vmin_h2, vmax=vmax_h2, aspect='auto',
                origin=orig, interpolation='none')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.ylim(elongation_h2[0], elongation_h2[-1])
    plt.savefig(savepath_h2 + 'jplot_hi2_' + start + '_' + time_file_h2 + 'UT_' + bflag[0] + '_' + ftpsc + '.png', bbox_inches=0, pad_inches=0)

    plt.close()

    fancy_plot = False

    if not fancy_plot:
        fig, ax = plt.subplots(frameon=False)

        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
        bbi = 0
        pi = 0

    if fancy_plot:
        font = {'size': 22}
        plt.rc('font', **font)
        fig, ax = plt.subplots(figsize=(12, 8))

        params = {'xtick.labelsize': 28, 'ytick.labelsize': 40}
        plt.rcParams.update(params)

        ax.set_xlabel('Date [DDMMYYYY]', labelpad=20)
        ax.set_ylabel('Elongation [°]')
        plt.setp(ax.get_xticklabels(), rotation=35, horizontalalignment='right')
        myFmt = mdates.DateFormatter('%Y%m%d')
        ax.xaxis.set_major_formatter(myFmt)
        bbi = 'tight'
        pi = 0.5

    ax.xaxis_date()

    ax.imshow(img_rescale_h2, cmap='gray', aspect='auto', vmin=vmin_h2, vmax=vmax_h2, interpolation='none', origin=orig, extent=[t1, t2+dt2, elongation_h2[0], elongation_h2[-1]+dx2])
    ax.imshow(img_rescale_h1, cmap='gray', aspect='auto', vmin=vmin_h1, vmax=vmax_h1, interpolation='none', origin=orig, extent=[t1, t2+dt1, elongation_h1[0], elongation_h1[-1]+dx1])


    plt.ylim(4, 80)

    if tcomp1 > tcomp2:
        time_file_comb = time_file_h2

    if tcomp1 < tcomp2:
        time_file_comb = time_file_h1

    plt.savefig(savepath_h1h2 + 'jplot_' + instrument + '_' + start + '_' + time_file_comb + 'UT_' + ftpsc + '_' + bflag[0] + '.png',
                bbox_inches=bbi, pad_inches=pi)

    jplot = image.imread(savepath_h1h2 + 'jplot_' + instrument + '_' + start + '_' + time_file_comb + 'UT_' + ftpsc + '_' + bflag[0] + '.png')

    savepath = path + 'jplot/' + ftpsc + '/' + bflag + '/' + instrument + '/' + str(start[0:4]) + '/params/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        subprocess.call(['chmod', '-R', '775', path + 'jplot/'])

    with open(savepath + 'jplot_' + instrument + '_' + start + '_' + time_file_comb + 'UT_' + ftpsc + '_' + bflag[0] + '_params.pkl', 'wb') as f:
        pickle.dump([t1, t2, elongation_h1[0], elongation_h2[-1]], f)

#######################################################################################################################################

def hi_fix_pointing(header, point_path, ftpsc, ins, post_conj, silent_point):
    """
    Conversion of fix_pointing.pro for IDL. To read in the pointing information from the appropriate  pnt_HI??_yyyy-mm-dd_fix_mu_fov.fts file and update the
    supplied HI index with the best fit pointing information and optical parameters calculated by the minimisation
    process of Brown, Bewsher and Eyles (2008).

    @param header: Header of .fits file
    @param point_path: Path of pointing calibration files
    @param ftpsc: STEREO Spacecraft (A/B)
    @param ins: STEREO-HI instrument (HI-1/HI-2)
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param silent_point: Run in silent mode
    """
    extra = 0

    hi_nominal = 1 #I changed this

    if ins == 'hi_1':
        ins = 'HI1'

    elif ins == 'hi_2':
        ins = 'HI2'

    else:
        print('Not a valid instrument, must be hi_1 or hi_2. Exiting...')
        sys.exit()

    try:
        header.rename_keyword('DATE-AVG', 'DATE_AVG')

    except ValueError:
        if not silent_point:
            print('Header information already corrected')

    hdr_date = header['DATE_AVG']
    hdr_date = hdr_date[0:10]

    rtmp = 5.

    point_file = 'pnt_' + ins + ftpsc + '_' + hdr_date + '_' + 'fix_mu_fov.fts'
    fle = point_path + point_file

    if os.path.isfile(fle):

        if not silent_point:
            print(('Reading {}...').format(point_file))

        hdul_point = fits.open(fle)

        for i in range(1, len(hdul_point)):
            extdate = hdul_point[i].header['extname']
            fledate = hdul_point[i].header['filename'][0:16]

            if (header['DATE_AVG'] == extdate) or (header['filename'][0:16] == fledate):
                ec = i
                break

        if (header['DATE_AVG'] == extdate) or (header['filename'][0:16] == fledate):

            stcravg = hdul_point[ec].header['ravg']
            stcnst1 = hdul_point[ec].header['nst1']

            if header['naxis1'] != 0:

                sumdif = np.round(header['cdelt1'] / hdul_point[ec].header['cdelt1'])

            else:
                sumdif = 1

            if stcnst1 < 20:

                if not silent_point:
                    print('Subfield presumed')
                    print('Using calibrated fixed instrument offsets')

                hi_calib_point(header, post_conj, hi_nominal)
                header['ravg'] = -894.

            else:

                if (stcravg < rtmp) & (stcravg > 0.):
                    header['crval1a'] = hdul_point[ec].header['crval1a']
                    header['crval2a'] = hdul_point[ec].header['crval2a']
                    header['pc1_1a'] = hdul_point[ec].header['pc1_1a']
                    header['pc1_2a'] = hdul_point[ec].header['pc1_2a']
                    header['pc2_1a'] = hdul_point[ec].header['pc2_1a']
                    header['pc2_2a'] = hdul_point[ec].header['pc2_2a']
                    header['cdelt1a'] = hdul_point[ec].header['cdelt1a'] * sumdif
                    header['cdelt2a'] = hdul_point[ec].header['cdelt2a'] * sumdif
                    header['pv2_1a'] = hdul_point[ec].header['pv2_1a']
                    header['crval1'] = hdul_point[ec].header['crval1']
                    header['crval2'] = hdul_point[ec].header['crval2']
                    header['pc1_1'] = hdul_point[ec].header['pc1_1']
                    header['pc1_2'] = hdul_point[ec].header['pc1_2']
                    header['pc2_1'] = hdul_point[ec].header['pc2_1']
                    header['pc2_2'] = hdul_point[ec].header['pc2_2']
                    header['cdelt1'] = hdul_point[ec].header['cdelt1'] * sumdif
                    header['cdelt2'] = hdul_point[ec].header['cdelt2'] * sumdif
                    header['pv2_1'] = hdul_point[ec].header['pv2_1']
                    header['xcen'] = hdul_point[ec].header['xcen']
                    header['ycen'] = hdul_point[ec].header['ycen']
                    header['crota'] = hdul_point[ec].header['crota']
                    header['ins_x0'] = hdul_point[ec].header['ins_x0']
                    header['ins_y0'] = hdul_point[ec].header['ins_y0']
                    header['ins_r0'] = hdul_point[ec].header['ins_r0']
                    header['ravg'] = hdul_point[ec].header['ravg']

                else:
                    if not silent_point:
                        print('R_avg does not meet criteria')
                        print('Using calibrated fixed instrument offsets')

                    hi_calib_point(header, post_conj, hi_nominal)
                    header['ravg'] = -883.

        else:
            if not silent_point:
                print(('No pointing calibration file found for file {}').format(point_file))
                print('Using calibrated fixed instrument offsets')

            hi_calib_point(header, post_conj, hi_nominal)
            header['ravg'] = -882.

    if not os.path.isfile(fle):
        if not silent_point:
            print(('No pointing calibration file found for file {}').format(point_file))
            print('Using calibrated fixed instrument offsets')

        hi_calib_point(header, post_conj, hi_nominal)
        header['ravg'] = -881.

        # hdul_point.close()


#######################################################################################################################################

def hi_calib_point(header, post_conj, hi_nominal):
    """
    Conversion of hi_calib_pointing.pro for IDL.

    @param header: Header of .fits file
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    """
    extra = 0

    roll = hi_calib_roll(header, 'gei', extra, post_conj, hi_nominal)

    header['pc1_1a'] = np.cos(roll * np.pi / 180.)
    header['pc1_2a'] = -np.sin(roll * np.pi / 180.)
    header['pc2_1a'] = np.sin(roll * np.pi / 180.)
    header['pc2_2a'] = np.cos(roll * np.pi / 180.)

    roll = hi_calib_roll(header, 'hpc', extra, post_conj, hi_nominal)

    header['crota'] = -roll
    header['pc1_1'] = np.cos(roll * np.pi / 180.)
    header['pc1_2'] = -np.sin(roll * np.pi / 180.)
    header['pc2_1'] = np.sin(roll * np.pi / 180.)
    header['pc2_2'] = np.cos(roll * np.pi / 180.)

    if 'summed' in header:

        naxis1 = 2048 / 2 ** (header['summed'] - 1)
        naxis2 = naxis1

    else:
        naxis1 = header['naxis1']
        naxis2 = header['naxis2']

    if naxis1 <= 0:
        naxis1 = 1024.

    if naxis2 <= 0:
        naxis2 = 1024.

    xv = [0.5 * naxis1, naxis1]
    yv = [0.5 * naxis2, 0.5 * naxis2]

    radec = fov2radec(xv, yv, header, 'gei', hi_nominal, extra)

    header['crval1a'] = radec[0, 0]
    header['crval2a'] = radec[1, 0]

    radec = fov2radec(xv, yv, header, 'hpc', hi_nominal, extra)

    header['crval1'] = -radec[0, 0]
    header['crval2'] = radec[1, 0]

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, extra, hi_nominal)
    header['pv2_1a'] = mu
    header['pv2_1'] = mu

    fp, fp_mm, plate = fparaxial(d, mu, header['naxis1'], header['naxis2'])

    if header['cunit1a'] == 'deg':
        xsize = plate / 3600.

    if header['cunit2a'] == 'deg':
        ysize = plate / 3600.

    if header['cunit1a'] == 'arcsec':
        xsize = plate

    if header['cunit2a'] == 'arcsec':
        ysize = plate

    header['cdelt1a'] = -xsize
    header['cdelt2a'] = ysize

    if header['cunit1'] == 'deg':
        xsize = plate / 3600.

    if header['cunit2'] == 'deg':
        ysize = plate / 3600.

    if header['cunit1'] == 'arcsec':
        xsize = plate

    if header['cunit2'] == 'arcsec':
        ysize = plate

    header['cdelt1'] = xsize
    header['cdelt2'] = ysize

    header['ins_x0'] = -offset_hi
    header['ins_y0'] = pitch_hi
    header['ins_r0'] = -roll_hi

#######################################################################################################################################

def hi_calib_roll(header, system, extra, post_conj, hi_nominal):
    """
    Conversion of hi_calib_roll.pro for IDL. Calculate the total roll angle of the HI image including
    contributions from the pitch and roll of the spacecraft. The total HI roll is a non-straighforward combination
    of the individual rolls of the s/c and HI, along with the pitch of the s/c and the offsets of HI. This
    routine calculates the total roll by taking 2 test points in the HI fov, transforms them to the
    appropriate frame of reference (given by the system keyword) and calculates the angle they make in this frame.

    @param header: Header of .fits file
    @param system: Which coordinate system to work in 'hpc' or 'gei'
    @param extra: This keyword is pointless, but was present in the original IDL code
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @return: Total roll of the spacecraft
    """
    if 'summed' in header:

        naxis1 = 2048 / 2 ** (header['summed'] - 1)
        naxis2 = naxis1

    else:

        naxis1 = header['naxis1']
        naxis2 = header['naxis2']

    if naxis1 <= 0:
        naxis1 = 1024.

    if naxis2 <= 0:
        naxis2 = 1024.

    cpix = np.array([naxis1, naxis2]) / 2. - 0.5
    xv = cpix[0] + [0., 512.]
    yv = np.full_like(cpix, cpix[1])

    xy = fov2pos(xv, yv, header, system, hi_nominal, extra)

    z = xy[2, 1] - xy[2, 0]
    x = -(xy[1, 1] - xy[1, 0])
    y = xy[0, 1] - xy[0, 0]

    tx = xy[0, 0]
    ty = xy[1, 0]
    tz = 0.0

    a = np.sqrt(tx ** 2 + ty ** 2)
    b = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ab = x * tx + y * ty

    val = np.nanmin([np.nanmax([ab / (a * b), -1.0]), 1.0])

    if z >= 0.0:
        oroll = np.arccos(val) * 180. / np.pi

    else:
        oroll = -np.arccos(val) * 180. / np.pi

    if post_conj:
        oroll = oroll - 180.

    return oroll


#######################################################################################################################################

def fov2pos(xv, yv, header, system, hi_nominal, extra):
    """
    Conversion of fov2pos.pro for IDL. To convert HI pixel positions to solar plane of sky
    coordinates in units of AU (actually, not quite, 1 is defined as the distance from the S/C to the Sun) and
    the Sun at (0,0). HI pixel position is converted to general AZP, then to Cartesian coordinates in the HI frame of
    reference. This is then rotated to be in the S/C frame and finally to the end reference frame. The final frame
    is a left-handed Cartesian system with x into the screen (towards the reference point), and z pointing up.

    @param xv: Array of x-pixel positions to be converted
    @param yv: Array of y-pixel positions to be converted
    @param header: Header of .fits file
    @param system: Which coordinate system to work in 'hpc' or 'gei'
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @param extra: This keyword is pointless, but was present in the original IDL code
    @return: An array of (x,y,z,w) quadruplets, where x,y,z have the meaning described above and w is a scale
    factor (see '3D computer graphics' by Alan Watt for further details). The wcolumn can be discounted
    for almost all user applications.
    """
    if system == 'hpc':

        yaw = header['sc_yaw']
        pitch = header['sc_pitch']
        roll = -header['sc_roll']

    else:

        yaw = header['sc_yawa']
        pitch = header['sc_pita']
        roll = header['sc_rolla']

    ccdosx = 0.
    ccdosy = 0.

    naxis = np.array([header['naxis1'], header['naxis2']])

    if naxis.any() == 0:
        naxis[:] = 1024

    pmult = naxis / 2.0 - 0.5

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, extra, hi_nominal)

    ang = (90. - 0.5 * d) * np.pi / 180.
    rng = (1.0 + mu) * np.cos(ang) / (np.sin(ang) + mu)

    vfov = np.array([xv, yv])
    nst = len(vfov[0])

    vv4 = np.zeros((4, nst))

    vv4[0][:] = ((vfov[0] - ccdosx) / pmult[0] - 1.0) * rng
    vv4[1][:] = ((vfov[1] - ccdosy) / pmult[1] - 1.0) * rng
    vv4[2][:] = 1.0
    vv4[3][:] = 1.0

    vv3 = azp2cart(vv4, mu)
    vv3[2, :] = 1.0

    vv2 = hi2sc(vv3, roll_hi, pitch_hi, offset_hi)
    vv = sc2cart(vv2, roll, pitch, yaw)

    return vv


#######################################################################################################################################

def get_hi_params(header, extra, hi_nominal):
    """
    Conversion of get_hi_params.pro for IDL. To detect which HI telescope is being used and return the
    instrument offsets relative to the spacecraft along with the mu parameter and the fov in degrees.
    As 'best pointings' may change as further calibration is done, it was thought more useful if there was one
    central routine to provide this data, rather than having to make the same changes in many different
    codes. Note, if you set one of the output variables to some value, then it will retain that value.

    @param header: Header of .fits file
    @param extra: This keyword is pointless, but was present in the original IDL code
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @return: HI yaw offset, HI pitch offfset, HI roll, HI distortion parameter, HI fov
    """
    if hi_nominal:

        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI1')):
            pitch_hi = 0.0
            offset_hi = -13.98
            roll_hi = 0.0
            mu = 0.16677
            d = 20.2663

        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI2')):
            offset_hi = -53.68
            pitch_hi = 0.
            roll_hi = 0.
            mu = 0.83329
            d = 70.8002

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI1')):
            offset_hi = 13.98
            pitch_hi = 0.
            roll_hi = 0.
            mu = 0.10001
            d = 20.2201

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI2')):
            offset_hi = 53.68
            pitch_hi = 0.
            roll_hi = 0.
            mu = 0.65062
            d = 69.8352

    else:

        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI1')):
            pitch_hi = 0.1159
            offset_hi = -14.0037
            roll_hi = 1.0215
            mu = 0.102422
            d = 20.27528

        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI2')):
            offset_hi = -53.4075
            pitch_hi = 0.0662
            roll_hi = 0.1175
            mu = 0.785486
            d = 70.73507

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI1')):
            offset_hi = 14.10
            pitch_hi = 0.022
            roll_hi = 0.37
            mu = 0.09509
            d = 20.23791

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI2')):
            offset_hi = 53.690
            pitch_hi = 0.213
            roll_hi = -0.052
            mu = 0.68886
            d = 70.20152

    return pitch_hi, offset_hi, roll_hi, mu, d


#######################################################################################################################################

def azp2cart(vec, mu):
    """
    Conversion of azp2cart.pro for IDL. To convert points seen with an AZP projection with
    parameter, mu to a Cartesian frame. Note, this is a low level code, and would usually not be called directly.
    For details, see "Coordinate systems for solar image data", W.T. Thompson, A&A

    @param vec: Array of vector postions to transform
    @param mu: HI distortion parameter
    @return: An array of transformed vector positions
    """
    nstars = np.shape(vec)[1]
    vout = vec.copy()

    for i in range(nstars):

        rth = np.sqrt(vec[0, i] ** 2 + vec[1, i] ** 2)
        rho = rth / (mu + 1.0)
        cc = np.sqrt(1.0 + rho ** 2)
        th = np.arccos(1.0 / cc) + np.arcsin(mu * rho / cc)
        zz = np.cos(th)
        rr = np.sin(th)

        if rth < 1.0e-6:
            vout[0:1, i] = rr * vec[0:1, i]
        else:
            vout[0:1, i] = rr * vec[0:1, i] / rth

        vout[2, i] = zz

    return vout


#######################################################################################################################################

def hi2sc(vec, roll_hi_deg, pitch_hi_deg, offset_hi_deg):
    """
    Conversion of hi2sec.pro for IDL. To transform the given position from the HI frame of
    reference to the spacecraft frame of reference. Note, this is a low level code, and would usually not be
    called directly. For the transformation we use 4x4 transformation
    matrices discussed in e.g. '3D computer graphics' by Alan Watt

    @param vec: Array of vector postions to transform
    @param roll_hi_deg: HI roll angle relative to spacecraft (in degrees)
    @param pitch_hi_deg: HI pitch angle relative to spacecraft (in degrees)
    @param offset_hi_deg: HI yaw angle relative to spacecraft (in degrees)
    @return: An array of transformed vector positions
    """
    npts = len(vec[0, :])

    theta = (90 - pitch_hi_deg) * np.pi / 180.
    phi = offset_hi_deg * np.pi / 180.
    roll = roll_hi_deg * np.pi / 180.

    normz = np.sin(theta) * np.cos(phi)
    normx = np.sin(theta) * np.sin(phi)
    normy = np.cos(theta)

    vdx = 0.
    vdy = 1.
    vdz = 0.

    vd_norm = vdx * normx + vdy * normy + vdz * normz

    vxtmp = vdx - vd_norm * normx
    vytmp = vdy - vd_norm * normy
    vztmp = vdz - vd_norm * normz

    ndiv = np.sqrt(vxtmp ** 2 + vytmp ** 2 + vztmp ** 2)
    vx = vxtmp / ndiv
    vy = vytmp / ndiv
    vz = vztmp / ndiv

    ux = -(normy * vz - normz * vy)
    uy = -(normz * vx - normx * vz)
    uz = -(normx * vy - normy * vx)

    cx = 0.
    cy = 0.
    cz = 0.

    tmat = np.array([[1., 0., 0., -cx], [0., 1., 0., -cy], [0., 0., 1., -cz], [0., 0., 0., 1.]])
    rmat = np.array([[ux, uy, uz, 0.], [vx, vy, vz, 0.], [normx, normy, normz, 0.], [0., 0., 0., 1.]])

    rollmat = np.array(
        [[np.cos(roll), -np.sin(roll), 0., 0.], [np.sin(roll), np.cos(roll), 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    tview = rollmat @ (rmat @ tmat)

    itview = np.linalg.inv(tview)

    vout = np.zeros((4, npts))

    for i in range(npts):
        vout[:, i] = np.transpose(itview @ np.transpose(vec[:, i]))

    return vout


#######################################################################################################################################

def sc2cart(vec, roll_deg, pitch_deg, yaw_deg):
    """
    Conversion of sc2cart.pro for IDL. To convert spacecraft pointing to Cartesian points in a known reference frame.
    Note, this is a low level code, and would usually not be called directly. For the transformation we use 4x4 transformation
    matrices discussed in e.g. '3D computer graphics' by Alan Watt.

    @param vec: An array of vector postions to transform
    @param roll_deg: Spacecraft roll angle (in degrees)
    @param pitch_deg: Spacecraft pitch angle (in degrees)
    @param yaw_deg: Spacecraft yaw angle (in degrees)
    @return: An array of transformed vector positions
    """
    npts = len(vec[0, :])

    theta = (90 - pitch_deg) * np.pi / 180.
    phi = yaw_deg * np.pi / 180.
    roll = roll_deg * np.pi / 180.

    normx = np.sin(theta) * np.cos(phi)
    normy = np.sin(theta) * np.sin(phi)
    normz = np.cos(theta)

    vdx = 0.
    vdy = 0.
    vdz = 1.

    vd_norm = vdx * normx + vdy * normy + vdz * normz

    vxtmp = vdx - vd_norm * normx
    vytmp = vdy - vd_norm * normy
    vztmp = vdz - vd_norm * normz

    ndiv = np.sqrt(vxtmp ** 2 + vytmp ** 2 + vztmp ** 2)
    vx = vxtmp / ndiv
    vy = vytmp / ndiv
    vz = vztmp / ndiv

    ux = normy * vz - normz * vy
    uy = normz * vx - normx * vz
    uz = normx * vy - normy * vx

    cx = 0.
    cy = 0.
    cz = 0.

    tmat = np.array([[1., 0., 0., -cx], [0., 1., 0., -cy], [0., 0., 1., -cz], [0., 0., 0., 1.]])
    rmat = np.array([[ux, uy, uz, 0.], [vx, vy, vz, 0.], [normx, normy, normz, 0.], [0., 0., 0., 1.]])

    rollmat = np.array(
        [[np.cos(roll), -np.sin(roll), 0., 0.], [np.sin(roll), np.cos(roll), 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    tview = rollmat @ (rmat @ tmat)

    itview = np.linalg.inv(tview)

    vout = np.zeros((4, npts))

    for i in range(npts):
        vout[:, i] = np.transpose(itview @ np.transpose(vec[:, i]))

    return vout


#######################################################################################################################################

def fov2radec(xv, yv, header, system, hi_nominal, extra):
    """
    Conversion of fov2radec for IDL. To convert HI pixel positions to RA-Dec pairs.
    HI pixel positions are converted into a general AZP form, which is converted to cartesian coordintes.
    These are then rotated from HI pointing, to S/C pointing then finally aligned with the RA-Dec frame.
    The Cartesian coordinates are finally converted to RA-Dec.

    @param xv: An array of x pixel positions
    @param yv: An array of y pixel positions
    @param header: Header of .fits file
    @param system: Which coordinate system to work in 'hpc' or 'gei'
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @param extra: This keyword is pointless, but was present in the original IDL code
    @return: An array of transformed vector positions
    """
    if system == 'gei':
        yaw = header['sc_yawa']
        pitch = header['sc_pita']
        roll = header['sc_rolla']

    else:
        yaw = header['sc_yaw']
        pitch = header['sc_pitch']
        roll = -header['sc_roll']

    ccdosx = 0.
    ccdosy = 0.
    pmult = header['naxis1'] / 2.0

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, extra, hi_nominal)

    ang = (90. - 0.5 * d) * np.pi / 180.
    rng = (1.0 + mu) * np.cos(ang) / (np.sin(ang) + mu)

    vfov = np.array([xv, yv])

    nst = len(vfov[0])

    vv4 = np.zeros((4, nst))

    vv4[0, :] = ((vfov[0] - ccdosx) / pmult - 1.0) * rng
    vv4[1, :] = ((vfov[1] - ccdosy) / pmult - 1.0) * rng
    vv4[3, :] = 1.0

    vv3 = azp2cart(vv4, mu)
    vv2 = hi2sc(vv3, roll_hi, pitch_hi, offset_hi)

    vv = sc2cart(vv2, roll, pitch, yaw)

    radec = np.zeros((2, nst))

    rd = 180 / np.pi

    for i in range(nst):

        th = np.arccos(vv[2, i])
        radec[1, i] = 90. - th * rd

        cphi = vv[0, i] / np.sin(th)
        sphi = vv[1, i] / np.sin(th)

        th1 = np.arccos(cphi) * rd
        th2 = np.arcsin(sphi) * rd

        if (th2 > 0):
            ra = th1

        else:
            ra = -th1

        radec[0, i] = ra

    return radec


#######################################################################################################################################

def fparaxial(fov, mu, naxis1, naxis2):
    """
    Conversion of fparaxial.pro for IDL. Calculate paraxial platescale of HI.
    This routine calculates the paraxial platescale from the calibrated fov and distortion parameter.

    @param fov: Calibrated field of view
    @param mu: HI distortion parameter
    @param naxis1: NAXIS1 from .fits header
    @param naxis2: NAXIS2 from .fits header
    @return: Paraxial platescale in mm
    """
    theta = (90. - (fov / 2.)) * np.pi / 180.

    tmp1 = (np.sin(theta) + mu) / ((1 + mu) * np.cos(theta))

    fp = (naxis1 / 2.) * tmp1

    widthmm = 0.0135 * 2048.
    fp_mm = widthmm * tmp1 / 2.

    plate = (0.0135 / fp_mm) * (180 / np.pi) * 3600.

    plate = plate * 2048. / naxis1

    return fp, fp_mm, plate


#######################################################################################################################################

def data_reduction(start, path, datpath, ftpsc, instrument, bflag, silent, save_path, path_flg):
    """
    Data reduciton routine calling upon various functions converted from IDL. The default correction procedure involves;
    correcting for shutterless exposure, multiplying by the flatfield, subtracting the bias,
    accounting for seb-ip binning, and desaturating the data. This process implicity performs conversion to DN/s.
    Procedure also adds best calibrated pointing into HI header.

    @param start: Start date of Jplot
    @param path: The path where all reduced images, running difference images and J-Maps are saved
    @param datpath: Path to STEREO-HI calibration files
    @param ftpsc: Spacecraft (A/B)
    @param instrument: STEREO-HI instrument (HI-1/HI-2)
    @param bflag: Science or beacon data
    @param silent: Run in silent mode
    @param save_path: Path pointing towards downloaded STEREO .fits files
    @param path_flg: Specifies path for downloaded files, depending on wether science or beacon data is used
    """
    if not silent:
        print('----------------')
        print('DATA REDUCTION')
        print('----------------')

    date = datetime.datetime.strptime(start, '%Y%m%d')

    if ftpsc == 'A':
        sc = 'ahead'

    if ftpsc == 'B':
        sc = 'behind'

    savepath = path + 'reduced/data/' + ftpsc + '/' + start + '/' + bflag + '/'
    calpath = datpath + 'calibration/'
    pointpath = datpath + 'data' + '/' + 'hi/'

    fits_hi1 = []
    fits_hi2 = []

    if bflag == 'science':

        for file in sorted(glob.glob(save_path + 'stereo' + sc[0] + '/secchi/' + path_flg + '/img/hi_1/' + str(start) + '/*s4*.fts')):
            fits_hi1.append(file)

        for file in sorted(glob.glob(save_path + 'stereo' + sc[0] + '/secchi/' + path_flg + '/img/hi_2/' + str(start) + '/*s4*.fts')):
            fits_hi2.append(file)

    if bflag == 'beacon':

        for file in sorted(glob.glob(save_path + 'stereo' + sc[0] + '/' + path_flg + '/secchi/' + '/img/hi_1/' + str(start) + '/*s7*.fts')):
            fits_hi1.append(file)

        for file in sorted(glob.glob(save_path + 'stereo' + sc[0] + '/' + path_flg + '/secchi/' + '/img/hi_2/' + str(start) + '/*s7*.fts')):
            fits_hi2.append(file)

    fitslist = [fits_hi1, fits_hi2]
    f = 0

    if instrument == 'hi1hi2':
        instrument = ['hi_1', 'hi_2']

    if instrument == 'hi_1':
        instrument = ['hi_1']

    if instrument == 'hi_2':
        instrument = ['hi_2']

    for fitsfiles in fitslist:
        if len(fitsfiles) > 0:

            ins = instrument[f]

            if not silent:
                print('----------------------------------------')
                print('Starting data reduction for', ins, '...')
                print('----------------------------------------')

            # correct for on-board sebip modifications to image (division by 2, 4, 16 etc.)
            # calls function scc_sebip

            hdul = [fits.open(fitsfiles[i]) for i in range(len(fitsfiles))]

            indices = np.arange(len(fitsfiles)).tolist()

            dateobs = [hdul[i][0].header['date-obs'] for i in range(len(fitsfiles))]
            dateavg = [hdul[i][0].header['date-avg'] for i in range(len(fitsfiles))]

            timeobs = [datetime.datetime.strptime(dateobs[i], '%Y-%m-%dT%H:%M:%S.%f') for i in indices]
            timeavg = [datetime.datetime.strptime(dateavg[i], '%Y-%m-%dT%H:%M:%S.%f') for i in indices]

            crval1 = [hdul[i][0].header['crval1'] for i in range(len(fitsfiles))]

            if ftpsc == 'A':    
                post_conj = [int(np.sign(crval1[i])) for i in indices]

            if ftpsc == 'B':    
                post_conj = [int(-1*np.sign(crval1[i])) for i in indices]

            if not silent:
                print('Correcting for binning...')

            bad_ind = []

            if bflag == 'science':

                for i in range(len(fitsfiles)):
                    if np.any(hdul[i][0].data <= 0):
                        bad_ind.append(i)

            data_trim = np.array([scc_img_trim(hdul[i][0].data, hdul[i][0].header) for i in indices])

            data_sebip = [scc_sebip(data_trim[i], hdul[i][0].header, True) for i in indices]

            if not silent:
                print('Getting bias...')

            # maps are created from corrected data
            # header is saved into separate list

            biasmean = [get_biasmean(hdul[i][0].header) for i in indices]
            biasmean = np.array(biasmean)

            for i in indices:

                if biasmean[i] != 0:
                    hdul[i][0].header['OFFSETCR'] = biasmean[i]

            data_sebip = data_sebip - biasmean[:, None, None]

           # perc = [np.percentile(data_sebip[i], 99.956, axis=None) for i in range(len(data_sebip))]
           #
           # data_sebip = np.array([np.where(data_sebip[i] > perc[i], np.nanmedian(data_sebip[i]), data_sebip[i]) for i in
           #              range(len(data_sebip))])

            if not silent:
                print('Removing saturated pixels...')

            # saturated pixels are removed
            # calls function hi_remove_saturation from functions.py

            data_desat = np.array([hi_remove_saturation(data_sebip[i, :, :], hdul[i][0].header) for i in indices])
            # data_desat = data_sebip.copy()

            if not silent:
                print('Desmearing image...')

            dstart1 = [hdul[i][0].header['dstart1'] for i in indices]
            dstart2 = [hdul[i][0].header['dstart2'] for i in indices]
            dstop1 = [hdul[i][0].header['dstop1'] for i in indices]
            dstop2 = [hdul[i][0].header['dstop2'] for i in indices]

            naxis1 = [hdul[i][0].header['naxis1'] for i in indices]
            naxis2 = [hdul[i][0].header['naxis2'] for i in indices]

            exptime = [hdul[i][0].header['exptime'] for i in indices]
            n_images = [hdul[i][0].header['n_images'] for i in indices]
            cleartim = [hdul[i][0].header['cleartim'] for i in indices]
            ro_delay = [hdul[i][0].header['ro_delay'] for i in indices]
            ipsum = [hdul[i][0].header['ipsum'] for i in indices]

            rectify = [hdul[i][0].header['rectify'] for i in indices]
            obsrvtry = [hdul[i][0].header['obsrvtry'] for i in indices]

            for i in range(len(obsrvtry)):

                if obsrvtry[i] == 'STEREO_A':
                    obsrvtry[i] = True

                else:
                    obsrvtry[i] = False

            line_ro = [hdul[i][0].header['line_ro'] for i in indices]
            line_clr = [hdul[i][0].header['line_clr'] for i in indices]

            header_int = np.array(
                [[dstart1[i], dstart2[i], dstop1[i], dstop2[i], naxis1[i], naxis2[i], n_images[i], post_conj[i]] for i in
                 range(len(dstart1))])

            header_flt = np.array(
                [[exptime[i], cleartim[i], ro_delay[i], ipsum[i], line_ro[i], line_clr[i]] for i in range(len(exptime))])

            header_str = np.array([[rectify[i], obsrvtry[i]] for i in range(len(rectify))])

            data_desm = [hi_desmear(data_desat[i, :, :], header_int[i], header_flt[i], header_str[i]) for i in
                         range(len(data_desat))]

            data_desm = np.array(data_desm)

            if not silent:
                print('Calibrating image...')

            ipkeep = [hdul[k][0].header['IPSUM'] for k in indices]

            calimg = [get_calimg(ins, ftpsc, hdul[k][0].header, calpath) for k in indices]
            calimg = np.array(calimg)

            if bflag == 'science':
                calfac = [get_calfac(hdul[k][0].header, timeavg[k]) for k in indices]
                calfac = np.array(calfac)

                diffuse = [scc_hi_diffuse(hdul[k][0].header, ipkeep[k]) for k in indices]
                diffuse = np.array(diffuse)

                data_red = calimg * data_desm * calfac[:, None, None] * diffuse

            if bflag == 'beacon':
                data_red = calimg * data_desm

            if not silent:
                print('Calibrating pointing...')

            for i in indices:
                hi_fix_pointing(hdul[i][0].header, pointpath, ftpsc, ins, post_conj, silent_point=True)

            if not silent:
                print('Saving .fts files...')

            if not os.path.exists(savepath + ins + '/'):
                os.makedirs(savepath + ins + '/')
                subprocess.call(['chmod', '-R', '775', savepath + ins + '/'])

            for i in range(len(fitsfiles)):

                name = fitsfiles[i].rpartition('/')[2]

                if bflag == 'science':
                    newname = name.replace('s4', '1b')

                if bflag == 'beacon':
                    newname = name.replace('s7', '17')

                if i not in bad_ind:
                    fits.writeto(savepath + ins + '/' + newname, data_red[i, :, :], hdul[i][0].header, output_verify='silentfix',
                                 overwrite=True)

                    hdul[i].close()

                if i in bad_ind:
                    hdul[i].close()

            f = f + 1


#######################################################################################################################################

def new_cmap(basemap, up_lim, low_lim):
    """
    Create a new colormap by specifying a pre-existing map and changing its upper and lower limits.

    @param basemap: Pre-existing matplotlib colormap object
    @param up_lim: New upper limit
    @param low_lim: New lower limit
    @return: Modified colormap with new limits
    """
    basemap = cm.get_cmap(basemap, 256)

    newcolors = basemap(np.linspace(0, 1, 256))

    newcolors[0:35, :] = newcolors[0:35, :] - [up_lim, up_lim, up_lim, 0]
    newcolors[35:, :] = newcolors[35:, :] + [low_lim, low_lim, low_lim, 0]
    newcolors = np.where(newcolors < 0, 0, newcolors)
    newcolors = np.where(newcolors > 1, 1, newcolors)

    newcmp = ListedColormap(newcolors)

    return newcmp


#######################################################################################################################################

def clean_hi(data, save_path, date, name, my_cmap):
    """
    Create clean looking .pngs of reduced HI images.

    @param data: HI image data
    @param date: Date for which to produce .pngs
    @param name: Names under which to save .png files
    @param my_cmap: New color map to use on HI data
    """
    newcmp = new_cmap(my_cmap, 0.08, 0)

    med = medfilter(data, 25)
    filt_img = np.where(np.abs(data) > 1.5 * med, med, data)

    img_med = np.nanmedian(filt_img)

    for i in range(len(filt_img)):

        if np.sum(filt_img[:, i]) / len(filt_img) > 50 * img_med:

            cnt = 0

            for j in range(len(filt_img)):
                if filt_img[j, i] > 40 * img_med:
                    cnt = cnt + 1

            if cnt > 400:
                filt_img[:, i] = img_med

    # filt_img = medfilter(filt_img, 5)

    fig, ax = plt.subplots(figsize=(1.024, 1.024), dpi=1000, frameon=False)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    ax.axis('off')

    ax.imshow(filt_img, cmap=newcmp, aspect='auto', origin='lower', vmin=-5e-14, vmax=5e-13)

    savepath = save_path + 'reduced/pngs/' + date + '/'

    if not os.path.exists(savepath):
        os.makedirs(savepath)
        subprocess.call(['chmod', '-R', '775', save_path + 'reduced/pngs/'])

    print('Saving...')
    plt.savefig(savepath + name + '.png')
    plt.close()


#######################################################################################################################################

@numba.njit(parallel=True)
def medfilter(img, kernel):
    """
    Implementation of a median filter.

    @param img: Image data
    @param kernel: Kernel size for median filter
    @return: Image with median filter applied
    """
    imgshape = np.shape(img)
    medimg = np.zeros(imgshape)

    kern = np.round(kernel / 2) - 1

    for i in range(len(img)):
        for j in range(len(img)):

            x_b = int(j - kern)
            x_a = int(j + kern)
            y_b = int(i - kern)
            y_a = int(i + kern)

            if x_b < 0:
                x_b = 0

            if x_a > len(img):
                x_a = len(img)

            if y_b < 0:
                y_b = 0

            if y_a > len(img):
                y_a = len(img)

            medimg[i, j] = np.nanmedian(img[y_b:y_a, x_b:x_a])

    return medimg


#######################################################################################################################################

def reduced_pngs(start, path, bflag, silent):
    """
    Create clean looking .pngs of reduced HI images. Calls clean_hi function.

    @param start: Date for which to produce .pngs
    @param path: Path to Events folder
    @param bflag: Science or beacon data
    @param silent: Run in silent mode
    """
    redpath = path + 'reduced/data/' + start + '/' + bflag + '/hi_1/'

    files = []

    for file in sorted(glob.glob(redpath + '*.fts')):
        files.append(file)

    hdul = [fits.open(files[i]) for i in range(len(files))]

    data = np.array([hdul[i][0].data for i in range(len(files))])

    names = [files[i].split('/')[-1] for i in range(len(files))]
    names = [names[i].split('.')[0] for i in range(len(files))]

    min_arr = np.quantile(data, 0.05, axis=0)

    data_nobg = data - min_arr

    if not silent:
        print(start)
    for i in range(len(data_nobg)):
        clean_hi(data_nobg[i], path, start, names[i], 'afmhot')
